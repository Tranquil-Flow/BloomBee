#!/usr/bin/env python3
"""Tiny HTTP coordinator for join offers, peer heartbeats, and joined plans.

This is bootstrap/roster/planning/operator-proof orchestration infrastructure only. It does not start
BloomBee servers, does not run inference, and every response carries a
no-inference-proof claim boundary.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from mvp_capabilities.cache_generation_proof import build_cache_generation_plan
    from mvp_capabilities.full_generation_proof import build_full_generation_plan
    from mvp_capabilities.join_coordinator import (
        DEFAULT_STATE_DIR,
        create_join_offer,
        load_active_heartbeats,
        record_heartbeat,
    )
    from mvp_capabilities.join_layer_plan import build_join_layer_plan, parse_seed_multiaddr
    from mvp_capabilities.multi_block_proof import build_multi_block_plan
    from mvp_capabilities.multi_request_load_proof import build_multi_request_load_plan
    from mvp_capabilities.proof_orchestrator import HANDOFF_EMBEDDED_SOURCE, build_proof_orchestration_plan
    from mvp_capabilities.route_picker import DEFAULT_PROOF_STATUS, DEFAULT_REGISTRY, expand_quantized_variants, load_proof_status, load_registry, route_report
    from mvp_capabilities.speculative_decode_plan import build_speculative_decode_plan
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.cache_generation_proof import build_cache_generation_plan
    from mvp_capabilities.full_generation_proof import build_full_generation_plan
    from mvp_capabilities.join_coordinator import (
        DEFAULT_STATE_DIR,
        create_join_offer,
        load_active_heartbeats,
        record_heartbeat,
    )
    from mvp_capabilities.join_layer_plan import build_join_layer_plan, parse_seed_multiaddr
    from mvp_capabilities.multi_block_proof import build_multi_block_plan
    from mvp_capabilities.multi_request_load_proof import build_multi_request_load_plan
    from mvp_capabilities.proof_orchestrator import HANDOFF_EMBEDDED_SOURCE, build_proof_orchestration_plan
    from mvp_capabilities.route_picker import DEFAULT_PROOF_STATUS, DEFAULT_REGISTRY, expand_quantized_variants, load_proof_status, load_registry, route_report
    from mvp_capabilities.speculative_decode_plan import build_speculative_decode_plan

HEALTH_CLAIM_BOUNDARY = "coordinator_health_only_no_inference_proof"
ERROR_CLAIM_BOUNDARY = "coordinator_error_no_inference_proof"
PLAN_SOURCE = "coordinator_http_plan_endpoint"
ROUTE_SOURCE = "coordinator_http_route_endpoint"
ROUTE_CLAIM_BOUNDARY = "coordinator_route_only_no_inference_proof"
HANDOFF_SOURCE = "coordinator_http_handoff_endpoint"
HANDOFF_CLAIM_BOUNDARY = "coordinator_handoff_bundle_only_no_server_started"
BOOTSTRAP_SOURCE = "coordinator_http_bootstrap_endpoint"
BOOTSTRAP_CLAIM_BOUNDARY = "coordinator_bootstrap_runbook_only_no_server_started"
SPECULATIVE_SOURCE = "coordinator_http_speculative_endpoint"
SPECULATIVE_CLAIM_BOUNDARY = "coordinator_speculative_plan_only_no_generation_proof"
PROOF_ORCHESTRATION_SOURCE = "coordinator_http_proof_orchestration_endpoint"


def _first(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _requested_model_from_query(query: dict[str, list[str]]) -> str | None:
    """Backward-compatible requested-model alias for route/handoff endpoints."""
    return _first(query, "requested_model") or _first(query, "model")


def _as_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    return None


def _registry_with_quantized_variants(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return base registry plus derived quantized route candidates for planning."""
    return list(registry) + expand_quantized_variants(registry)


def _parse_block_range_value(block_range: str) -> tuple[int, int]:
    start_raw, separator, end_raw = block_range.partition(":")
    if separator != ":":
        raise ValueError(f"invalid block range: {block_range}")
    start = int(start_raw)
    end = int(end_raw)
    if end <= start:
        raise ValueError(f"invalid block range: {block_range}")
    return start, end


def _assignment_block_ranges(plan: dict[str, Any]) -> list[str]:
    placement = plan.get("placement") or {}
    ranges: list[str] = []
    for assignment in placement.get("assignments") or []:
        if not isinstance(assignment, dict):
            continue
        block_range = assignment.get("block_range")
        if isinstance(block_range, str) and block_range:
            ranges.append(block_range)
    return ranges


def _combined_assignment_range(block_ranges: list[str]) -> str:
    parsed = sorted(_parse_block_range_value(item) for item in block_ranges)
    if not parsed:
        raise ValueError("no block ranges available")
    cursor = parsed[0][0]
    for start, end in parsed:
        if start != cursor:
            raise ValueError(f"block ranges must be contiguous: {block_ranges!r}")
        cursor = end
    return f"{parsed[0][0]}:{cursor}"


def _server_maddr_placeholders(count: int) -> list[str]:
    return [f"<PASTE_SERVER_{index}_MULTIADDR>" for index in range(count)]


def _server_placement_strings(plan: dict[str, Any]) -> list[str]:
    placement = plan.get("placement") or {}
    values: list[str] = []
    for assignment in placement.get("assignments") or []:
        if not isinstance(assignment, dict):
            continue
        hostname = assignment.get("hostname")
        block_range = assignment.get("block_range")
        if hostname and block_range:
            values.append(f"{hostname}={block_range}")
    return values


def _assignment_ports(plan: dict[str, Any]) -> list[int]:
    placement = plan.get("placement") or {}
    ports: list[int] = []
    for assignment in placement.get("assignments") or []:
        if not isinstance(assignment, dict):
            continue
        port = assignment.get("port")
        if isinstance(port, int):
            ports.append(port)
    return ports


def _unavailable_runbook(proof_gate: str, reason: str) -> dict[str, Any]:
    return {
        "proof_gate": proof_gate,
        "status": "unavailable",
        "reason": reason,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def _shell_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def build_bootstrap_runbook(
    *,
    coordinator: str,
    token: str,
    ttl_seconds: int = 600,
    now: int | None = None,
    count: int = 180,
    interval_seconds: float = 10.0,
) -> dict[str, Any]:
    """Build a fresh-device join script without starting servers or inference."""
    bounded_count = max(1, int(count))
    bounded_interval = max(0.0, float(interval_seconds))
    offer = create_join_offer(coordinator=coordinator, token=token, ttl_seconds=ttl_seconds, now=now)
    join_url = str(offer["join_url"])
    interval_text = _shell_number(bounded_interval)
    shell_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"# claim_boundary: {BOOTSTRAP_CLAIM_BOUNDARY}",
            "# inference_proven: false",
            "# can_update_proof_status: false",
            "set -euo pipefail",
            "",
            'CAP_PATH="${BLOOMBEE_CAPABILITIES_PATH:-$HOME/.bloombee/capabilities/$(hostname -s).json}"',
            'mkdir -p "$(dirname "$CAP_PATH")"',
            "",
            'python mvp_capabilities/peer_scan.py --out "$CAP_PATH"',
            "python mvp_capabilities/join_client.py "
            f"--join-url {shlex.quote(join_url)} "
            '--capabilities "$CAP_PATH" '
            f"--count {bounded_count} "
            f"--interval-seconds {interval_text}",
            "",
        ]
    )
    return {
        "source": BOOTSTRAP_SOURCE,
        "claim_boundary": BOOTSTRAP_CLAIM_BOUNDARY,
        "offer": offer,
        "token": token,
        "capabilities_path_env": "BLOOMBEE_CAPABILITIES_PATH",
        "default_capabilities_path": "~/.bloombee/capabilities/$(hostname -s).json",
        "heartbeat_loop": {
            "count": bounded_count,
            "interval_seconds": bounded_interval,
            "claim_boundary": "join_client_heartbeat_loop_only_no_inference_proof",
        },
        "shell_script": shell_script,
        "operator_next_steps": [
            "run this script from the BloomBee checkout on the joining laptop",
            "keep the process running until the coordinator /active roster shows the device",
            "do not treat heartbeat success as server start or inference proof",
        ],
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def _build_handoff_proof_runbooks(
    *,
    plan: dict[str, Any],
    model: dict[str, Any],
    registry_models: list[dict[str, Any]],
    request_count: int,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    model_id = str(plan.get("model_id") or model.get("model_id") or "")
    block_ranges = _assignment_block_ranges(plan)
    server_maddrs = _server_maddr_placeholders(len(block_ranges))
    server_placements = _server_placement_strings(plan)
    proof_runbooks: dict[str, Any] = {}

    if len(block_ranges) >= 2:
        try:
            ports = _assignment_ports(plan)
            proof_runbooks["multi_block"] = build_multi_block_plan(
                model_id,
                registry=registry_models,
                block_ranges=block_ranges,
                ports=ports or None,
                server_log_prefix=f".local/{model_id.replace('/', '--')}-multi-block-server",
                client_log=f".local/{model_id.replace('/', '--')}-multi-block-client.log",
            )
        except ValueError as exc:
            proof_runbooks["multi_block"] = _unavailable_runbook("multi_block", str(exc))
    else:
        proof_runbooks["multi_block"] = _unavailable_runbook("multi_block", "multi_block proof requires at least two assigned block ranges")

    if server_maddrs:
        proof_runbooks["full_generation"] = build_full_generation_plan(
            model_id=model_id,
            server_maddrs=server_maddrs,
            server_placements=server_placements,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            evidence_path=f".local/{model_id.replace('/', '--')}-full-generation.json",
        )
        proof_runbooks["cache_generation"] = build_cache_generation_plan(
            model_id=model_id,
            server_maddrs=server_maddrs,
            server_placements=server_placements,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            evidence_path=f".local/{model_id.replace('/', '--')}-cache-generation.json",
        )
    else:
        proof_runbooks["full_generation"] = _unavailable_runbook("full_generation", "no assigned server ranges available")
        proof_runbooks["cache_generation"] = _unavailable_runbook("cache_generation", "no assigned server ranges available")

    hidden_dim = int(model.get("hidden_size") or 0)
    if server_maddrs and block_ranges and hidden_dim > 0:
        try:
            proof_runbooks["multi_request_load"] = build_multi_request_load_plan(
                model_id=model_id,
                block_range=_combined_assignment_range(block_ranges),
                server_maddrs=server_maddrs,
                request_count=request_count,
                hidden_dim=hidden_dim,
                client_log_prefix=f".local/{model_id.replace('/', '--')}-load-client",
            )
        except ValueError as exc:
            proof_runbooks["multi_request_load"] = _unavailable_runbook("multi_request_load", str(exc))
    else:
        proof_runbooks["multi_request_load"] = _unavailable_runbook("multi_request_load", "requires assigned ranges, server multiaddrs, and model hidden_size")

    return proof_runbooks


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _text_bytes(text: str) -> bytes:
    return (text.rstrip("\n") + "\n").encode("utf-8")


def _active_for_query(query: dict[str, list[str]], *, state_dir: str | Path, token: str) -> list[dict[str, Any]]:
    return load_active_heartbeats(
        state_dir,
        token=token,
        now=_as_int(_first(query, "now"), 0) or None,
        max_age_seconds=_as_int(_first(query, "max_age_seconds"), 30),
    )


def _peers_from_heartbeats(active_heartbeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peers: list[dict[str, Any]] = []
    for heartbeat in active_heartbeats:
        capabilities = heartbeat.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        peer = dict(capabilities)
        peer.setdefault("hostname", heartbeat.get("peer_id"))
        peer["joined_peer_id"] = heartbeat.get("peer_id")
        peer["joined_at"] = heartbeat.get("timestamp")
        peer["join_claim_boundary"] = heartbeat.get("claim_boundary")
        peers.append(peer)
    peers.sort(key=lambda peer: str(peer.get("hostname") or peer.get("joined_peer_id") or ""))
    return peers


def _route_from_query(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    active = _active_for_query(query, state_dir=state_dir, token=token)
    peers = _peers_from_heartbeats(active)
    try:
        payload = route_report(
            peers,
            load_registry(registry),
            scenario=_first(query, "scenario"),
            requested_model=_requested_model_from_query(query),
            proof_status=load_proof_status(_first(query, "proof_status", str(DEFAULT_PROOF_STATUS)) or DEFAULT_PROOF_STATUS),
            selector_mode=_first(query, "selector_mode", "planning") or "planning",
        )
    except ValueError as exc:
        return 400, {"error": str(exc), "claim_boundary": ERROR_CLAIM_BOUNDARY}
    payload["claim_boundary"] = ROUTE_CLAIM_BOUNDARY
    payload["source"] = ROUTE_SOURCE
    payload["token"] = token
    payload["active_peer_count"] = len(active)
    payload["active_heartbeats"] = active
    payload["inference_proven"] = False
    payload["can_update_proof_status"] = False
    return 200, payload


def _handle_speculative(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    route_query = {key: list(value) for key, value in query.items()}
    if route_query.get("model") == ["auto"]:
        route_query.pop("model", None)
    route_status, route_payload = _route_from_query(route_query, state_dir=state_dir, registry=registry)
    if route_status != 200:
        return route_status, route_payload
    active = _active_for_query(query, state_dir=state_dir, token=token)
    peers = _peers_from_heartbeats(active)
    plan = build_speculative_decode_plan(
        verifier_route=route_payload,
        peers=peers,
        draft_model_id=_first(query, "draft_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_draft_tokens=_as_int(_first(query, "max_draft_tokens"), 4),
        acceptance_window=_as_int(_first(query, "acceptance_window"), 4),
    )
    return 200, {
        "claim_boundary": SPECULATIVE_CLAIM_BOUNDARY,
        "source": SPECULATIVE_SOURCE,
        "token": token,
        "active_peer_count": len(active),
        "route_decision": route_payload,
        "speculative_plan": plan,
        "inference_proven": False,
        "generation_proven": False,
        "can_update_proof_status": False,
    }


def _handle_plan(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    model_id = _first(query, "model")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    if not model_id:
        return 400, {"error": "missing model", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    route_decision: dict[str, Any] | None = None
    registry_models = load_registry(registry)
    planning_registry = _registry_with_quantized_variants(registry_models)
    if model_id == "auto":
        route_query = {key: list(value) for key, value in query.items() if key != "model"}
        route_status, route_payload = _route_from_query(route_query, state_dir=state_dir, registry=registry)
        if route_status != 200:
            return route_status, route_payload
        route_decision = route_payload
        picked_model_id = (route_payload.get("picked") or {}).get("model_id")
        if not picked_model_id:
            return 409, {"error": "no selectable model for current joined roster", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        model_id = str(picked_model_id)

    model = _find_model(planning_registry, model_id)
    if model is None:
        return 404, {"error": f"model not found: {model_id}", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    seed_multiaddrs: dict[str, str] = {}
    for raw in query.get("seed_multiaddr") or []:
        try:
            host, multiaddr = parse_seed_multiaddr(raw)
        except argparse.ArgumentTypeError as exc:
            return 400, {"error": str(exc), "claim_boundary": ERROR_CLAIM_BOUNDARY}
        seed_multiaddrs[host] = multiaddr

    payload = build_join_layer_plan(
        state_dir=state_dir,
        token=token,
        model=model,
        now=_as_int(_first(query, "now"), 0) or None,
        max_age_seconds=_as_int(_first(query, "max_age_seconds"), 30),
        include_launch_commands=_as_bool(_first(query, "include_launch_commands")),
        include_launch_readiness=_as_bool(_first(query, "include_launch_readiness")),
        device=_first(query, "launch_device", "mps") or "mps",
        dtype=_first(query, "launch_dtype", "float16") or "float16",
        base_port=_as_int(_first(query, "base_port"), 31337),
        dht_prefix=_first(query, "dht_prefix"),
        seed_multiaddrs=seed_multiaddrs,
    )
    payload["source"] = PLAN_SOURCE
    if route_decision is not None:
        payload["route_decision"] = route_decision
    return 200, payload


def _handle_handoff(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    registry_models = load_registry(registry)
    planning_registry = _registry_with_quantized_variants(registry_models)
    offer = create_join_offer(
        coordinator=_first(query, "coordinator", coordinator) or coordinator,
        token=token,
        ttl_seconds=_as_int(_first(query, "ttl_seconds"), 600),
        now=_as_int(_first(query, "now"), 0) or None,
    )
    active = {
        "token": token,
        "active_peers": _active_for_query(query, state_dir=state_dir, token=token),
        "claim_boundary": "heartbeat_roster_only_no_inference_proof",
    }

    plan_query = {key: list(value) for key, value in query.items()}
    plan_query.setdefault("model", ["auto"])
    plan_query.setdefault("include_launch_commands", ["1"])
    plan_query.setdefault("include_launch_readiness", ["1"])
    plan_status, plan = _handle_plan(plan_query, state_dir=state_dir, registry=registry)
    if plan_status != 200:
        return plan_status, plan

    model_id = str(plan.get("model_id") or "")
    model = _find_model(planning_registry, model_id)
    if model is None:
        return 404, {"error": f"model not found: {model_id}", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    proof_runbooks = _build_handoff_proof_runbooks(
        plan=plan,
        model=model,
        registry_models=planning_registry,
        request_count=_as_int(_first(query, "request_count"), 3),
        prompt=_first(query, "prompt", "The moon is") or "The moon is",
        max_new_tokens=_as_int(_first(query, "max_new_tokens"), 4),
    )
    bootstrap_runbook = build_bootstrap_runbook(
        coordinator=_first(query, "coordinator", coordinator) or coordinator,
        token=token,
        ttl_seconds=_as_int(_first(query, "ttl_seconds"), 600),
        now=_as_int(_first(query, "now"), 0) or None,
        count=_as_int(_first(query, "count"), 180),
        interval_seconds=_as_float(_first(query, "interval_seconds"), 10.0),
    )
    speculative_plan = build_speculative_decode_plan(
        verifier_route=plan.get("route_decision") or {"picked": {"model_id": model_id}},
        peers=_peers_from_heartbeats(active.get("active_peers") or []),
        draft_model_id=_first(query, "draft_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_draft_tokens=_as_int(_first(query, "max_draft_tokens"), 4),
        acceptance_window=_as_int(_first(query, "acceptance_window"), 4),
    )
    route_decision = plan.get("route_decision")
    if not isinstance(route_decision, dict):
        route_query = {key: list(value) for key, value in query.items()}
        route_query["model"] = [model_id]
        route_status, route_payload = _route_from_query(route_query, state_dir=state_dir, registry=registry)
        route_decision = route_payload if route_status == 200 else {"status": "unavailable", "error": route_payload.get("error")}

    bundle = {
        "claim_boundary": HANDOFF_CLAIM_BOUNDARY,
        "source": HANDOFF_SOURCE,
        "token": token,
        "offer": offer,
        "active": active,
        "route_decision": route_decision,
        "plan": plan,
        "bootstrap_runbook": bootstrap_runbook,
        "speculative_plan": speculative_plan,
        "proof_runbooks": proof_runbooks,
        "operator_next_steps": [
            "share offer.join_url, join card, or bootstrap_runbook.shell_script with devices",
            "wait for active.active_peers to match physical devices",
            "run plan.placement.assignments launch_command values manually on assigned hosts",
            "replace <SEED_MULTIADDR_FROM_...> / <PASTE_SERVER_..._MULTIADDR> placeholders with captured server multiaddrs",
            "run the proof_runbooks commands and only update proof status after verify mode passes",
        ],
        "inference_proven": False,
        "can_update_proof_status": False,
    }
    bundle["proof_orchestration"] = build_proof_orchestration_plan(bundle, source=HANDOFF_EMBEDDED_SOURCE)
    return 200, bundle


def _handle_proof_orchestration(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    status, handoff = _handle_handoff(query, state_dir=state_dir, coordinator=coordinator, registry=registry)
    if status != 200:
        return status, handoff
    plan = dict(handoff.get("proof_orchestration") or build_proof_orchestration_plan(handoff))
    plan["source"] = PROOF_ORCHESTRATION_SOURCE
    plan["handoff_source"] = handoff.get("source")
    plan["inference_proven"] = False
    plan["can_update_proof_status"] = False
    return 200, plan


def _bootstrap_from_query(query: dict[str, list[str]], *, coordinator: str) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    return 200, build_bootstrap_runbook(
        coordinator=_first(query, "coordinator", coordinator) or coordinator,
        token=token,
        ttl_seconds=_as_int(_first(query, "ttl_seconds"), 600),
        now=_as_int(_first(query, "now"), 0) or None,
        count=_as_int(_first(query, "count"), 180),
        interval_seconds=_as_float(_first(query, "interval_seconds"), 10.0),
    )


def handle_get_text(
    path: str,
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
) -> tuple[int, str, bytes] | None:
    """Return plain-text endpoint responses for shell-friendly join flows."""
    del state_dir, registry  # reserved for parity with handle_get and future text routes
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path != "/bootstrap.sh":
        return None
    status, payload = _bootstrap_from_query(query, coordinator=coordinator)
    if status != 200:
        message = f"error: {payload.get('error')}\nclaim_boundary: {payload.get('claim_boundary')}"
        return status, "text/plain; charset=utf-8", _text_bytes(message)
    return status, "text/x-shellscript; charset=utf-8", _text_bytes(str(payload["shell_script"]))


def handle_get(
    path: str,
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path == "/healthz":
        return 200, {"ok": True, "claim_boundary": HEALTH_CLAIM_BOUNDARY}
    if parsed.path == "/offer":
        return 200, create_join_offer(
            coordinator=_first(query, "coordinator", coordinator) or coordinator,
            token=_first(query, "token"),
            ttl_seconds=_as_int(_first(query, "ttl_seconds"), 600),
            now=_as_int(_first(query, "now"), 0) or None,
        )
    if parsed.path == "/bootstrap":
        return _bootstrap_from_query(query, coordinator=coordinator)
    if parsed.path == "/active":
        token = _first(query, "token")
        if not token:
            return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        return 200, {
            "token": token,
            "active_peers": _active_for_query(query, state_dir=state_dir, token=token),
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }
    if parsed.path == "/route":
        return _route_from_query(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/speculative":
        return _handle_speculative(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/plan":
        return _handle_plan(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/handoff":
        return _handle_handoff(query, state_dir=state_dir, coordinator=coordinator, registry=registry)
    if parsed.path == "/proof-orchestration":
        return _handle_proof_orchestration(query, state_dir=state_dir, coordinator=coordinator, registry=registry)
    return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}


def handle_post(path: str, *, body: bytes, state_dir: str | Path) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    if parsed.path != "/heartbeat":
        return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    token = payload.get("token")
    peer_id = payload.get("peer_id")
    capabilities = payload.get("capabilities")
    if not token or not peer_id or not isinstance(capabilities, dict):
        return 400, {"error": "missing required heartbeat fields", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    return 200, record_heartbeat(
        state_dir,
        token=str(token),
        peer_id=str(peer_id),
        capabilities=capabilities,
        now=payload.get("now"),
    )


class JoinCoordinatorHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass,
        *,
        state_dir: str | Path,
        coordinator: str,
        registry: str | Path = DEFAULT_REGISTRY,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.state_dir = Path(state_dir).expanduser()
        self.coordinator = coordinator
        self.registry = Path(registry).expanduser()


class JoinCoordinatorHandler(BaseHTTPRequestHandler):
    server: JoinCoordinatorHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Keep tests/docs clean and avoid recording peer/token tuples in logs.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "claim_boundary": ERROR_CLAIM_BOUNDARY})

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        text_response = handle_get_text(
            self.path,
            state_dir=self.server.state_dir,
            coordinator=self.server.coordinator,
            registry=self.server.registry,
        )
        if text_response is not None:
            status, content_type, body = text_response
            self._send_text(status, content_type, body)
            return
        status, payload = handle_get(
            self.path,
            state_dir=self.server.state_dir,
            coordinator=self.server.coordinator,
            registry=self.server.registry,
        )
        self._send_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error_json(400, "invalid content length")
            return
        status, payload = handle_post(self.path, body=self.rfile.read(length), state_dir=self.server.state_dir)
        self._send_json(status, payload)


def create_server(
    address: tuple[str, int],
    *,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
) -> JoinCoordinatorHTTPServer:
    return JoinCoordinatorHTTPServer(
        address,
        JoinCoordinatorHandler,
        state_dir=state_dir,
        coordinator=coordinator,
        registry=registry,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--coordinator", default=None, help="Public coordinator URL embedded in join offers")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Model registry for /plan endpoint")
    args = parser.parse_args(argv)

    coordinator = args.coordinator or f"http://{args.host}:{args.port}"
    server = create_server((args.host, args.port), state_dir=args.state_dir, coordinator=coordinator, registry=args.registry)
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                "bootstrap_endpoint": "/bootstrap",
                "bootstrap_script_endpoint": "/bootstrap.sh",
                "handoff_endpoint": "/handoff",
                "proof_orchestration_endpoint": "/proof-orchestration",
                "speculative_endpoint": "/speculative",
                "plan_endpoint": "/plan",
                "registry": str(Path(args.registry).expanduser()),
                "state_dir": str(Path(args.state_dir).expanduser()),
                "claim_boundary": "coordinator_service_only_no_inference_proof",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
