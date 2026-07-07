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
import time
import urllib.request
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
    from mvp_capabilities.layer_planner import attach_launch_commands
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
    from mvp_capabilities.layer_planner import attach_launch_commands
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
MIN_OFFER_TTL_SECONDS = 1
MAX_OFFER_TTL_SECONDS = 3600


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


def _bounded_offer_ttl_seconds(value: str | int | None, default: int = 600) -> int:
    ttl = _as_int(str(value) if value is not None else None, default)
    return max(MIN_OFFER_TTL_SECONDS, min(MAX_OFFER_TTL_SECONDS, ttl))


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
    offer = create_join_offer(coordinator=coordinator, token=token, ttl_seconds=_bounded_offer_ttl_seconds(ttl_seconds), now=now)
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
            f"python mvp_capabilities/join_client.py "
            f"--join-url {shlex.quote(join_url)} "
            '--capabilities "$CAP_PATH" '
            f"--count {bounded_count} "
            f"--interval-seconds {interval_text} "
            "--auto-serve",
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


def _compatible_from_query(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    """Return all models compatible with the current swarm, ranked by params."""
    token = _first(query, "token") or "*"
    active = _active_for_query(query, state_dir=state_dir, token=token)
    peers = _peers_from_heartbeats(active)
    if not peers:
        return 200, {
            "compatible_models": [],
            "best_model": None,
            "peer_count": 0,
            "total_free_gb": 0.0,
            "claim_boundary": "compatible_models_only_no_inference_proof",
            "source": "compatible_endpoint",
        }

    registry_models = load_registry(registry)
    total_free_gb = 0.0
    for p in peers:
        mem = p.get("memory", {})
        gb = _as_float(mem.get("free_gb"), 0) or _as_float(mem.get("available_gb"), 0) or _as_float(mem.get("total_gb"), 0)
        # Also check accelerator VRAM on unified-memory Macs
        acc = p.get("accelerator", {})
        if gb == 0 and acc.get("unified_memory"):
            gb = _as_float(acc.get("vram_free_gb"), 0) or _as_float(mem.get("total_gb"), 0)
        total_free_gb += gb

    compatible: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None
    best_score = 0.0

    for model in registry_models:
        model_id = str(model.get("model_id", ""))
        if not model_id:
            continue
        params_b = _as_float(model.get("params_b"), 0) or _as_float(model.get("active_params_b"), 0)
        required_gb = _as_float(model.get("recommended_min_free_mem_gb"), 0) or _as_float(model.get("min_total_mem_gb"), 0) or (params_b * 2 + 2)
        if model.get("architecture_supported") is False and model.get("blocked_reasons"):
            continue  # only skip models with explicit blocking reasons

        if total_free_gb >= required_gb:
            status = "compatible"
        elif any(
            (_as_float(p.get("memory", {}).get("available_gb"), 0)
             or _as_float(p.get("memory", {}).get("total_gb"), 0)) >= required_gb
            for p in peers
        ):
            status = "single_peer"
        else:
            status = "insufficient"

        num_layers = int(model.get("num_layers", 0))
        score = params_b
        entry = {
            "model_id": model_id,
            "params_b": params_b,
            "active_params_b": _as_float(model.get("active_params_b"), 0) or params_b,
            "num_layers": num_layers,
            "required_gb": required_gb,
            "hidden_size": model.get("hidden_size"),
            "supports_moe": bool(model.get("supports_moe")),
            "status": status,
        }
        if status == "compatible":
            if score > best_score:
                best_score = score
                best_model = entry
            compatible.insert(0, entry)
        else:
            compatible.append(entry)

    return 200, {
        "compatible_models": compatible,
        "best_model": best_model,
        "peer_count": len(peers),
        "total_free_gb": round(total_free_gb, 1),
        "claim_boundary": "compatible_models_only_no_inference_proof",
        "source": "compatible_endpoint",
    }


# ── Deployment state (peer_id → job) ──────────────────────────────────────────

DEPLOYMENT_CLAIM_BOUNDARY = "deployment_plan_only_no_server_started"
JOB_CLAIM_BOUNDARY = "job_assignment_only_no_server_started"
INFER_CLAIM_BOUNDARY = "inference_coordination_only_no_actual_inference"
PIPELINE_CLAIM_BOUNDARY = "pipeline_visualization_only_no_inference_proof"


def _build_pipeline_snapshot(state_dir: str | Path) -> dict[str, Any]:
    """Build a pipeline visualization snapshot from deployment + heartbeats."""
    deployment = _load_deployment(state_dir)
    jobs = deployment.get("jobs", {})
    active = load_active_heartbeats(state_dir, token="*", max_age_seconds=300)

    peers: list[dict[str, Any]] = []
    for hb in active:
        caps = hb.get("capabilities", {})
        peer_id = str(hb.get("peer_id", ""))
        hostname = str(caps.get("hostname", peer_id))
        job = None
        for jh, jd in jobs.items():
            if jh.lower() == hostname.lower() or hostname.lower().startswith(jh.lower()):
                job = jd
                break
        mem_total = caps.get("memory", {}).get("total_gb", 0) or 0
        mem_avail = caps.get("memory", {}).get("available_gb", 0) or 0
        if mem_total > 0:
            mem_used_pct = max(0, min(100, int((1 - mem_avail / mem_total) * 100)))
        else:
            mem_used_pct = -1  # sentinel: unknown

        # Estimate latency: 2.5ms per layer, floor at 0
        if job:
            layers = (job.get("end_layer") or 0) - (job.get("start_layer") or 0)
        else:
            layers = 0
        latency_est = max(0, int(layers * 2.5))

        peers.append({
            "hostname": hostname,
            "peer_id": peer_id,
            "platform": caps.get("platform", "unknown"),
            "role": "draft" if mem_total == 0 else caps.get("role", "compute"),
            "block_range": job.get("block_indices", "") if job else "",
            "start_layer": job.get("start_layer") if job else None,
            "end_layer": job.get("end_layer") if job else None,
            "status": caps.get("status", hb.get("status", "idle")),
            "memory_pct": mem_used_pct,
            "total_gb": mem_total,
            "available_gb": mem_avail,
            "latency_ms_est": latency_est,
            "port": job.get("port") if job else None,
        })

    # Sort by start_layer
    peers.sort(key=lambda p: p.get("start_layer") or 0)

    return {
        "model_id": deployment.get("model_id"),
        "peer_count": len(jobs),
        "serving_count": sum(1 for p in peers if p["status"] == "serving"),
        "peers": peers,
        "deployed_at": deployment.get("created_at"),
        "claim_boundary": PIPELINE_CLAIM_BOUNDARY,
    }


def _handle_inference_feed(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    wfile,
) -> None:
    """GET /inference-feed — SSE stream of pipeline state."""
    import time as _time
    run_id = _first(query, "run_id") or "pipeline"
    interval = max(1, _as_int(_first(query, "interval"), 2))

    def _write_sse(event: str, data: str) -> None:
        msg = f"event: {event}\ndata: {data}\n\n"
        wfile.write(msg.encode("utf-8"))
        wfile.flush()

    snapshot = _build_pipeline_snapshot(state_dir)
    _write_sse("snapshot", json.dumps(snapshot))

    for _ in range(60):  # max 2 min of streaming
        snapshot = _build_pipeline_snapshot(state_dir)
        for peer in snapshot.get("peers", []):
            peer["_ts"] = _time.time()
            _write_sse("peer_update", json.dumps(peer))
        _write_sse("heartbeat", json.dumps({"run_id": run_id, "ts": _time.time()}))
        _time.sleep(interval)


def _handle_infer(
    body: bytes,
    *,
    state_dir: str | Path,
) -> tuple[int, dict[str, Any]]:
    """POST /infer — check deployment readiness and return inference plan."""
    try:
        req = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    model_id = req.get("model_id") or req.get("model")
    prompt = req.get("prompt", "")
    max_tokens = req.get("max_tokens", 128)

    if not model_id:
        return 400, {"error": "missing model_id", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    if not prompt:
        return 400, {"error": "missing prompt", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    deployment = _load_deployment(state_dir)
    jobs = deployment.get("jobs", {})

    if not jobs:
        return 409, {
            "error": "No deployment found. Deploy a model first via POST /deploy.",
            "claim_boundary": INFER_CLAIM_BOUNDARY,
        }

    deployed_model = deployment.get("model_id")
    if deployed_model != model_id:
        return 409, {
            "error": f"Deployed model is '{deployed_model}', requested '{model_id}'. Deploy the requested model first.",
            "claim_boundary": INFER_CLAIM_BOUNDARY,
        }

    # Check which peers are serving (via heartbeat status)
    active = load_active_heartbeats(state_dir, token="*", max_age_seconds=300)
    serving_peers: list[dict[str, Any]] = []
    all_peers: list[dict[str, Any]] = []
    for hb in active:
        peer_id = str(hb.get("peer_id", ""))
        caps = hb.get("capabilities", {})
        hostname = str(caps.get("hostname", peer_id))
        status = caps.get("status", hb.get("status", "unknown"))
        all_peers.append({"peer_id": peer_id, "hostname": hostname, "status": status})
        if status == "serving":
            serving_peers.append({"peer_id": peer_id, "hostname": hostname})

    job_hostnames = set(jobs.keys())
    serving_hostnames = {p["hostname"] for p in serving_peers}
    ready = job_hostnames.issubset(serving_hostnames)

    # Build inference run command
    commands: list[str] = []
    for job_hostname, job in jobs.items():
        cmd = job.get("command", "")
        if cmd:
            commands.append(f"# On {job_hostname}:\n{cmd}")

    inference_cmd = (
        "python scripts/direct_remote_call.py \\\n"
        f"  --model {model_id} \\\n"
        f'  --prompt "{prompt[:100]}"\n'
        "# (requires BloomBee venv with torch + hivemind)"
    )

    return 200, {
        "ready": ready,
        "model_id": model_id,
        "prompt": prompt[:200],
        "max_tokens": max_tokens,
        "peer_count": len(jobs),
        "serving_count": len(serving_peers),
        "peers": all_peers,
        "deployed_jobs": {
            hostname: {
                "block_indices": job.get("block_indices"),
                "status": job.get("status"),
                "command": job.get("command", ""),
            }
            for hostname, job in jobs.items()
        },
        "inference_command": inference_cmd if ready else None,
        "server_commands": commands if not ready else None,
        "next_step": (
            "All peers serving — run the inference_command on the coordinator machine."
            if ready
            else f"Start servers first: {len(job_hostnames - serving_hostnames)} peer(s) not serving. Run the server_commands on each peer."
        ),
        "claim_boundary": INFER_CLAIM_BOUNDARY,
    }


IOS_CLAIM_BOUNDARY = "ios_peer_coordination_only_no_inference_proof"


def _handle_ios_register(
    body: bytes,
    *,
    state_dir: str | Path,
) -> tuple[int, dict[str, Any]]:
    """POST /ios/register — register an iOS peer as a draft contributor."""
    try:
        req = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    device_model = str(req.get("device_model", "iPhone"))
    ios_version = str(req.get("ios_version", "unknown"))
    mlx_model_id = str(req.get("mlx_model_id", "unknown"))
    peer_id = req.get("peer_id") or f"ios-{device_model.lower().replace(' ', '-')}"
    token = str(req.get("token", "ios-gateway"))

    capabilities = {
        "hostname": device_model,
        "platform": "ios",
        "ios_version": ios_version,
        "mlx_model_id": mlx_model_id,
        "device_model": device_model,
        "memory": {"total_gb": 6, "available_gb": 4},
        "gpu": {"available": True, "backend": "ANE", "name": "Apple Neural Engine"},
        "cpu": {"cores": 6, "model": "arm64e"},
        "role": "draft_peer",
        "status": "serving" if req.get("ready", False) else "connected",
    }

    result = record_heartbeat(
        state_dir,
        token=token,
        peer_id=str(peer_id),
        capabilities=capabilities,
    )
    result["claim_boundary"] = IOS_CLAIM_BOUNDARY
    return 200, result


def _handle_ios_draft(
    body: bytes,
    *,
    state_dir: str | Path,
) -> tuple[int, dict[str, Any]]:
    """POST /ios/draft — verify draft tokens from an iOS peer.

    Returns accepted/rejected tokens with spec-decode semantics
    (halt at first rejected token). Currently uses confidence-based
    mock; real forward pass pending coordinator spine + pruner.
    """
    try:
        req = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    prompt = str(req.get("prompt", ""))
    draft_tokens = req.get("draft_tokens", [])
    confidences = req.get("confidences", [])
    peer_id = str(req.get("peer_id", "unknown"))

    if not isinstance(draft_tokens, list) or not isinstance(confidences, list):
        return 400, {"error": "draft_tokens and confidences must be lists",
                     "claim_boundary": ERROR_CLAIM_BOUNDARY}

    # Spec-decode verification: accept consecutive tokens with
    # confidence >= threshold, halt at first rejection.
    threshold = float(req.get("threshold", 0.45))
    accepted: list[int] = []
    rejected: list[int] = []
    for tok, conf in zip(draft_tokens, confidences):
        if conf >= threshold:
            accepted.append(tok)
        else:
            rejected.append(tok)
            break  # halt at first reject per spec-decode semantics
    score = sum(confidences) / len(confidences) if confidences else 0.0

    return 200, {
        "accepted": len(accepted),
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "score": score,
        "peer_id": peer_id,
        "prompt": prompt[:100],
        "verifier": "mock_confidence",
        "claim_boundary": IOS_CLAIM_BOUNDARY,
    }


def _deployment_path(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "deployment.json"


def _load_deployment(state_dir: str | Path) -> dict[str, Any]:
    path = _deployment_path(state_dir)
    if not path.exists():
        return {"model_id": None, "jobs": {}, "created_at": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"model_id": None, "jobs": {}, "created_at": None}


def _save_deployment(state_dir: str | Path, deployment: dict[str, Any]) -> None:
    path = _deployment_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(deployment, indent=2, sort_keys=True), encoding="utf-8")


def _handle_deploy(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    """POST /deploy — generate plan + assign jobs to all connected peers."""
    model_id = _first(query, "model_id") or _first(query, "model")
    if not model_id:
        return 400, {"error": "missing model_id", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    token = _first(query, "token") or "*"

    # _handle_plan expects "model" in query; inject model_id there
    query = dict(query, model=[model_id])

    # Generate plan with launch commands
    plan_status, plan_payload = _handle_plan(query, state_dir=state_dir, registry=registry)
    if plan_status != 200:
        return plan_status, plan_payload

    plan = plan_payload.get("placement", {})
    if not plan.get("supported"):
        return 409, {
            "error": "Plan does not cover all layers",
            "plan_summary": plan.get("reason"),
            "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
        }

    # Attach launch commands to the plan
    device = _first(query, "launch_device", "mps") or "mps"
    dtype = _first(query, "launch_dtype", "float16") or "float16"
    base_port = _as_int(_first(query, "base_port"), 31337)
    plan_with_commands = attach_launch_commands(
        plan, device=device, dtype=dtype, base_port=base_port
    )

    # Build deployment state: map hostname → job
    jobs: dict[str, Any] = {}
    for assignment in plan_with_commands.get("assignments") or []:
        hostname = str(assignment.get("hostname") or "unknown")
        jobs[hostname] = {
            "model_id": model_id,
            "block_indices": assignment.get("block_range", ""),
            "start_layer": assignment.get("start_layer"),
            "end_layer": assignment.get("end_layer"),
            "port": assignment.get("port"),
            "command": assignment.get("launch_command", ""),
            "status": "queued",
        }

    deployment = {
        "model_id": model_id,
        "jobs": jobs,
        "created_at": time.time(),
        "peer_count": len(jobs),
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    _save_deployment(state_dir, deployment)

    return 200, deployment


def _match_job_for_peer(peer_id: str, jobs: dict[str, Any], state_dir: str | Path) -> dict[str, Any] | None:
    """Match a peer_id to a deployment job, using hostname resolution as fallback."""
    # 1) exact match
    if peer_id in jobs:
        return jobs[peer_id]

    # 2) case-insensitive / prefix match
    peer_lower = peer_id.lower()
    for key, val in jobs.items():
        key_lower = key.lower()
        if peer_lower == key_lower or peer_lower.startswith(key_lower) or key_lower.startswith(peer_lower):
            return val

    # 3) resolve peer_id → hostname via heartbeat, then match by hostname
    active = load_active_heartbeats(state_dir, token="*", max_age_seconds=300)
    for hb in active:
        if hb.get("peer_id") == peer_id:
            caps = hb.get("capabilities", {})
            hostname = str(caps.get("hostname") or "")
            if hostname and hostname in jobs:
                return jobs[hostname]
            # also try case-insensitive hostname match
            hostname_lower = hostname.lower()
            for key, val in jobs.items():
                if key.lower() == hostname_lower:
                    return val

    return None


def _handle_job(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
) -> tuple[int, dict[str, Any]]:
    """GET /job?peer_id=X — return this peer's assigned job, or null."""
    peer_id = _first(query, "peer_id")
    if not peer_id:
        return 400, {"error": "missing peer_id", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    deployment = _load_deployment(state_dir)
    jobs = deployment.get("jobs", {})

    # Match by peer_id (which is hostname-based in our bootstrap).
    # 1) exact match on peer_id key
    # 2) case-insensitive / prefix match
    # 3) resolve peer_id → hostname via heartbeat, then match by hostname
    job = _match_job_for_peer(peer_id, jobs, state_dir)

    if job is None:
        return 200, {"peer_id": peer_id, "job": None, "deployed_model": deployment.get("model_id"),
                      "claim_boundary": JOB_CLAIM_BOUNDARY}

    return 200, {
        "peer_id": peer_id,
        "job": job,
        "deployed_model": deployment.get("model_id"),
        "deployed_at": deployment.get("created_at"),
        "claim_boundary": JOB_CLAIM_BOUNDARY,
    }


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
        ttl_seconds=_bounded_offer_ttl_seconds(_first(query, "ttl_seconds")),
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
        ttl_seconds=_bounded_offer_ttl_seconds(_first(query, "ttl_seconds")),
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
        ttl_seconds=_bounded_offer_ttl_seconds(_first(query, "ttl_seconds")),
        now=_as_int(_first(query, "now"), 0) or None,
        count=_as_int(_first(query, "count"), 180),
        interval_seconds=_as_float(_first(query, "interval_seconds"), 10.0),
    )


def _landing_html(coordinator: str) -> str:
    """Return the HTML landing page served at GET /."""
    coord_esc = coordinator.rstrip("/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Join BloomBee Swarm</title>
<script src="https://cdn.jsdelivr.net/npm/qrcodejs@1.0.0/qrcode.min.js"></script>
<style>
:root{{color-scheme:dark;--bg:#07111f;--panel:#0d1b2f;--line:#2a4b73;--text:#e9f3ff;--muted:#95acc8;--ok:#58d68d;--accent:#7dd3fc;--moon:#b9ccff}}
*{{box-sizing:border-box;margin:0}}
body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:radial-gradient(circle at 20% -10%,#233e73 0,transparent 34%),var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 16px}}
.card{{background:linear-gradient(180deg,rgba(18,39,68,.96),rgba(13,27,47,.96));border:1px solid var(--line);border-radius:18px;padding:32px;max-width:580px;width:100%;box-shadow:0 14px 40px rgba(0,0,0,.28);text-align:center}}
h1{{font-size:26px;margin-bottom:6px;letter-spacing:-.03em}}
h2{{color:var(--moon);font-size:14px;font-weight:400;margin-bottom:20px}}
.step{{background:rgba(7,17,31,.6);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;text-align:left}}
.step h3{{color:var(--accent);margin-bottom:6px;font-size:13px}}
.step p{{color:var(--muted);font-size:12px;line-height:1.5}}
pre{{background:#050e1a;color:#d7e5ff;border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto;font-size:12px;line-height:1.6;margin:6px 0;font-family:ui-monospace,Menlo,monospace;text-align:left;white-space:pre-wrap;word-break:break-all}}
.copy-btn{{display:inline-block;background:var(--accent);color:var(--bg);border:none;padding:10px 24px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:700;margin-top:8px;transition:all .15s}}
.copy-btn:hover{{opacity:.85;transform:scale(1.03)}}
.copy-btn.copied{{background:var(--ok)}}
.tabs{{display:flex;gap:6px;margin:12px 0;justify-content:center}}
.tabs button{{background:rgba(185,204,255,.1);border:1px solid var(--line);color:var(--muted);padding:6px 14px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600}}
.tabs button.active{{background:rgba(125,211,252,.18);color:var(--accent);border-color:var(--accent)}}
.tab{{display:none}}
.tab.active{{display:block}}
.note{{color:var(--muted);font-size:11px;margin-top:12px;padding:8px 12px;background:rgba(247,201,72,.06);border:1px solid rgba(247,201,72,.15);border-radius:8px}}
</style>
</head>
<body>
<div class="card">
<h1>🌙 Join the BloomBee Swarm</h1>
<h2 id="coord">Coordinator: {coord_esc}</h2>

<div class="step">
<h3>📱 Scan to open this page on your phone</h3>
<p>Or open <code id="coord-url-text">{coord_esc}/</code> directly. The QR keeps this page so anyone can scan it again to onboard the next person.</p>
<div id="qr" style="display:flex;justify-content:center;margin:10px 0"></div>
</div>

<div class="step">
<h3>📋 One command to join</h3>
<p>Open a terminal and run:</p>
<pre id="cmd">Loading join offer...</pre>
<button class="copy-btn" onclick="copyCmd()">📋 Copy to clipboard</button>
</div>

<div class="tabs">
<button class="active" onclick="showTab('any')">💻 Laptop</button>
<button onclick="showTab('phone')">📱 Android Phone</button>
</div>

<div class="tab active" id="tab-any">
<p style="color:var(--muted);font-size:12px;line-height:1.5;margin-top:10px;">
<strong>Requirements:</strong> Python 3.8+ (pre-installed on macOS/Linux).
The command downloads a self-contained script — no pip, no clone, no setup.
</p>
</div>

<div class="tab" id="tab-phone">
<div class="step">
<h3>1. Install Termux</h3>
<p>Get <strong>Termux</strong> from <a href="https://f-droid.org/packages/com.termux/" style="color:var(--accent);">F-Droid</a> (NOT Play Store). Open it and run:</p>
<pre>pkg update && pkg install python</pre>
</div>
<div class="step">
<h3>2. Run the join command</h3>
<p>Copy and paste the command above into Termux:</p>
<pre id="cmd-phone">Loading...</pre>
</div>
</div>

<div class="note">
⚠️ <strong>What this does:</strong> Scans your hardware, registers with the swarm,
and heartbeats to stay active. No models run yet, no files accessed.
Press Ctrl+C to disconnect.<br><br>
<strong>📱 iOS / iPhone:</strong> iPhones cannot join as inference peers (no background Python, no Termux).
Android devices can join via Termux. iPhones can still view the dashboard.
</div>

<div class="step" style="margin-top:18px">
<h3>🐝 Live swarm <span id="peer-count" style="color:var(--ok);font-weight:700">…</span></h3>
<div id="roster" style="font-size:12px;color:var(--muted);line-height:1.6;text-align:left">Loading…</div>
</div>
</div>

<script>
const C = "{coord_esc}";
const LANDING = C + "/";
async function init() {{
  // Render QR pointing at this landing page (so the next person can scan)
  try {{
    new QRCode(document.getElementById('qr'), {{
      text: LANDING,
      width: 200, height: 200,
      colorDark: '#07111f', colorLight: '#e9f3ff',
      correctLevel: QRCode.CorrectLevel.M,
    }});
  }} catch(e) {{ document.getElementById('qr').textContent = '(QR library unavailable offline)'; }}

  // Build the join command
  try {{
    const r = await fetch(C + "/offer");
    const d = await r.json();
    const cmd = `curl -s ${{C}}/bootstrap.py | python3 - --join-url "${{d.join_url}}" --loop --interval 30`;
    const phoneCmd = `curl -s ${{C}}/bootstrap.py | python3 - --join-url "${{d.join_url}}" --loop --interval 60`;
    document.getElementById('cmd').textContent = cmd;
    document.getElementById('cmd-phone').textContent = phoneCmd;
  }} catch(e) {{
    document.getElementById('cmd').textContent = 'Coordinator unreachable. Check the URL and try again.';
  }}

  // Live roster
  refreshRoster();
  setInterval(refreshRoster, 8000);
}}
async function refreshRoster() {{
  const el = document.getElementById('roster');
  const cnt = document.getElementById('peer-count');
  try {{
    const r = await fetch(C + "/active?max_age_seconds=60");
    const d = await r.json();
    const peers = d.active_peers || [];
    cnt.textContent = `(${{peers.length}} connected)`;
    if (peers.length === 0) {{
      el.innerHTML = '<em>No peers yet — be the first to scan and join!</em>';
      return;
    }}
    el.innerHTML = peers.map(p => {{
      const cap = p.capabilities || {{}};
      const gpu = cap.gpu || {{}};
      const cpu = cap.cpu || {{}};
      const mem = cap.memory || {{}};
      const icon = gpu.available ? '🎮' : '🖥️';
      const gpuname = gpu.name || 'cpu-only';
      return `<div style="padding:6px 0;border-bottom:1px solid rgba(42,75,115,.4)">
        ${{icon}} <strong style="color:var(--text)">${{cap.hostname || p.peer_id}}</strong>
        <span style="color:var(--muted)"> · ${{cap.platform || '?'}} · ${{cpu.cores || '?'}} cores · ${{mem.total_gb || '?'}}GB · ${{gpuname}}</span>
      </div>`;
    }}).join('');
  }} catch(e) {{
    cnt.textContent = '(offline)';
    el.innerHTML = '<em>Cannot reach coordinator.</em>';
  }}
}}
function copyCmd() {{
  const t = document.getElementById('cmd').textContent;
  if(!t.startsWith('curl')) return;
  copyText(t);
  const b = document.querySelector('.copy-btn');
  if(b){{b.textContent='✅ Copied! Paste in terminal.';b.classList.add('copied');setTimeout(()=>{{b.textContent='📋 Copy to clipboard';b.classList.remove('copied')}},2500);}}
}}
function copyText(text) {{
  // navigator.clipboard only works on HTTPS/localhost — fallback for HTTP LAN
  if(navigator.clipboard && window.isSecureContext) {{
    navigator.clipboard.writeText(text).catch(()=>fallbackCopy(text));
  }} else {{
    fallbackCopy(text);
  }}
}}
function fallbackCopy(text) {{
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position='fixed'; ta.style.left='-9999px';
  document.body.appendChild(ta); ta.select();
  try {{ document.execCommand('copy'); }} catch(e) {{}}
  document.body.removeChild(ta);
}}
function showTab(n) {{
  document.querySelectorAll('.tabs button,.tab').forEach(e=>e.classList.remove('active'));
  document.querySelector(`.tabs button:nth-child(${{n==='any'?1:2}})`).classList.add('active');
  document.getElementById('tab-'+n).classList.add('active');
}}
if(/Android|iPhone|iPad/i.test(navigator.userAgent)) showTab('phone');
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def handle_get_text(
    path: str,
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
) -> tuple[int, str, bytes] | None:
    """Return plain-text / HTML endpoint responses."""
    parsed = urlparse(path)
    query = parse_qs(parsed.query)

    # ── GET / — landing page ─────────────────────────────────────────────
    if parsed.path == "/":
        return 200, "text/html; charset=utf-8", _text_bytes(_landing_html(coordinator))

    # ── GET /bootstrap.py — self-contained join script ────────────────────
    if parsed.path == "/bootstrap.py":
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap.py"
        if script_path.exists():
            return 200, "text/x-python; charset=utf-8", script_path.read_bytes()
        # Fallback: use mvp_capabilities path
        alt_path = Path(__file__).resolve().parent.parent / "mvp_capabilities" / "bootstrap.py"
        if alt_path.exists():
            return 200, "text/x-python; charset=utf-8", alt_path.read_bytes()
        return 404, "text/plain; charset=utf-8", _text_bytes("bootstrap.py not found")

    # ── GET /bootstrap.sh — shell script bootstrap ────────────────────────
    if parsed.path == "/bootstrap.sh":
        del state_dir, registry
        status, payload = _bootstrap_from_query(query, coordinator=coordinator)
        if status != 200:
            message = f"error: {payload.get('error')}\nclaim_boundary: {payload.get('claim_boundary')}"
            return status, "text/plain; charset=utf-8", _text_bytes(message)
        return status, "text/x-shellscript; charset=utf-8", _text_bytes(str(payload["shell_script"]))

    return None


def handle_get(
    path: str,
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
    started_at: float | None = None,
) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path == "/healthz":
        body: dict[str, Any] = {
            "ok": True,
            "status": "live",
            "claim_boundary": HEALTH_CLAIM_BOUNDARY,
            "coordinator": coordinator,
            "state_dir": str(Path(state_dir).expanduser()),
            "registry": str(Path(registry).expanduser()),
        }
        if started_at is not None:
            body["uptime_seconds"] = round(time.time() - started_at, 1)
            body["started_at"] = started_at
        return 200, body
    if parsed.path == "/status":
        # Orchestration health: probes iOS gateway, anisette container, IPA build.
        # Dashboard uses this to show green/yellow status dots per step.
        result: dict[str, Any] = {
            "coordinator": {"ok": True, "port": 8787},
            "gateway": {"ok": False, "reason": "not checked yet"},
            "anisette": {"ok": False, "reason": "not checked yet"},
            "ipa": {"ok": False, "reason": "not checked yet"},
        }
        # 4a: iOS gateway at :8432
        gw_host = parsed.hostname or "127.0.0.1"
        try:
            g = urllib.request.urlopen(
                f"http://{gw_host}:8432/healthz", timeout=0.8
            )
            gw_data = json.loads(g.read().decode("utf-8", "replace"))
            if gw_data.get("service") == "bloombee-ios-gateway":
                result["gateway"] = {
                    "ok": True,
                    "port": 8432,
                    "version": gw_data.get("version", "?"),
                }
            else:
                result["gateway"] = {
                    "ok": False,
                    "reason": f"port 8432 answered but service={gw_data.get('service','?')}",
                }
        except Exception:
            result["gateway"] = {
                "ok": False,
                "reason": "iOS gateway not reachable on :8432",
            }
        # 4b: Anisette Docker container
        import subprocess as _sp
        try:
            out = _sp.check_output(
                ["docker", "ps", "--filter", "name=anisette", "--format", "{{.Status}}"],
                timeout=3,
                text=True,
            ).strip()
            if out:
                result["anisette"] = {"ok": True, "port": 6969, "status": out}
            else:
                result["anisette"] = {
                    "ok": False,
                    "reason": "docker container 'anisette' not running",
                }
        except Exception:
            result["anisette"] = {
                "ok": False,
                "reason": "docker not available or anisette container not found",
            }
        # 4c: IPA file
        ipa_path = Path.home() / "Projects/bloombee-ios-gateway/build/BloomBee.ipa"
        if ipa_path.exists():
            size_mb = round(ipa_path.stat().st_size / 1_048_576, 1)
            result["ipa"] = {"ok": True, "path": str(ipa_path), "size_mb": size_mb}
        else:
            result["ipa"] = {
                "ok": False,
                "reason": f"IPA not found at {ipa_path}",
            }
        result["all_ok"] = all(
            result[k]["ok"] for k in ("coordinator", "gateway", "anisette", "ipa")
        )
        return 200, result
    if parsed.path == "/offer":
        return 200, create_join_offer(
            coordinator=_first(query, "coordinator", coordinator) or coordinator,
            token=_first(query, "token"),
            ttl_seconds=_bounded_offer_ttl_seconds(_first(query, "ttl_seconds")),
            now=_as_int(_first(query, "now"), 0) or None,
        )
    if parsed.path == "/bootstrap":
        return _bootstrap_from_query(query, coordinator=coordinator)
    if parsed.path == "/active":
        token = _first(query, "token") or "*"
        if token == "*":
            # Return all active peers regardless of token
            all_peers = []
            now = _as_int(_first(query, "now"), 0) or None
            max_age = _as_int(_first(query, "max_age_seconds"), 30)
            for path in sorted(Path(state_dir).expanduser().glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    # Skip non-heartbeat files (deployment.json, etc.)
                    if "peer_id" not in payload and "timestamp" not in payload:
                        continue
                    age = (now or int(time.time())) - int(payload.get("timestamp", 0))
                    if 0 <= age <= max_age:
                        all_peers.append(payload)
                    elif age > max_age:
                        # Clean expired heartbeat files
                        try:
                            path.unlink()
                        except OSError:
                            pass
                except (OSError, json.JSONDecodeError, ValueError):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
            return 200, {
                "active_peers": all_peers,
                "claim_boundary": "heartbeat_roster_only_no_inference_proof",
            }
        return 200, {
            "token": token[:8] + "...",
            "active_peers": _active_for_query(query, state_dir=state_dir, token=token),
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }
    if parsed.path == "/route":
        return _route_from_query(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/compatible":
        return _compatible_from_query(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/speculative":
        return _handle_speculative(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/plan":
        return _handle_plan(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/handoff":
        return _handle_handoff(query, state_dir=state_dir, coordinator=coordinator, registry=registry)
    if parsed.path == "/proof-orchestration":
        return _handle_proof_orchestration(query, state_dir=state_dir, coordinator=coordinator, registry=registry)
    if parsed.path == "/job":
        return _handle_job(query, state_dir=state_dir)
    if parsed.path == "/pipeline":
        return 200, _build_pipeline_snapshot(state_dir)
    return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}


def handle_post(path: str, *, body: bytes, state_dir: str | Path, registry: str | Path = DEFAULT_REGISTRY) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path == "/heartbeat":
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        token = payload.get("token")
        peer_id = payload.get("peer_id")
        capabilities = payload.get("capabilities")
        if not token or not peer_id or not isinstance(capabilities, dict):
            return 400, {"error": "missing required heartbeat fields", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        heartbeat_now = payload.get("now") if _as_bool(_first(query, "allow_client_now_for_tests")) else None
        return 200, record_heartbeat(
            state_dir,
            token=str(token),
            peer_id=str(peer_id),
            capabilities=capabilities,
            now=heartbeat_now,
        )
    if parsed.path == "/deploy":
        return _handle_deploy(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/infer":
        return _handle_infer(body, state_dir=state_dir)
    if parsed.path == "/ios/register":
        return _handle_ios_register(body, state_dir=state_dir)
    if parsed.path == "/ios/draft":
        return _handle_ios_draft(body, state_dir=state_dir)
    return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}


class JoinCoordinatorHTTPServer(ThreadingHTTPServer):
    # Allow restart while a previous instance is still in TIME_WAIT.
    # Without this, a quick restart after Ctrl+C fails with
    # "OSError: [Errno 48] Address already in use" for ~30-60s.
    allow_reuse_address = True

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
        # Wall-clock start time used by /healthz for uptime reporting.
        self.started_at = time.time()


class JoinCoordinatorHandler(BaseHTTPRequestHandler):
    server: JoinCoordinatorHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Keep tests/docs clean and avoid recording peer/token tuples in logs.
        return

    def _send_cors_headers(self) -> None:
        """Allow the standalone dashboard HTML to poll this coordinator.

        The operator dashboard is intentionally generated as a local static HTML
        file, often opened via ``file://`` or a tiny local static server. Browser
        fetches from that page to ``http://<LAN_IP>:8787`` are cross-origin, so
        the coordinator must emit permissive CORS headers for this room-scale
        onboarding demo.
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "claim_boundary": ERROR_CLAIM_BOUNDARY})

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        # SSE streaming endpoint
        if parsed.path == "/inference-feed":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                _handle_inference_feed(query, state_dir=self.server.state_dir, wfile=self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

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
            started_at=getattr(self.server, "started_at", None),
        )
        self._send_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error_json(400, "invalid content length")
            return
        status, payload = handle_post(self.path, body=self.rfile.read(length), state_dir=self.server.state_dir, registry=self.server.registry)
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


def _probe_existing_coordinator(host: str, port: int, timeout_s: float = 1.0) -> dict | None:
    """If something is already listening on host:port, return its /healthz JSON.

    Returns:
      - None if nothing is listening
      - {"_not_bloombee": True, ...} if a non-BloomBee HTTP server answers /healthz
        with anything other than a 200 + ok=true response
      - dict of the /healthz JSON if it looks like a BloomBee coordinator

    Used by main() to detect "already running" before bind() raises OSError.
    """
    import socket as _socket
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    try:
        with _socket.create_connection((host, port), timeout=timeout_s):
            pass
    except (OSError, _socket.timeout):
        return None

    for scheme in ("http", "https"):
        try:
            req = Request(f"{scheme}://{host}:{port}/healthz", method="GET")
            with urlopen(req, timeout=timeout_s) as r:
                if r.status == 200:
                    body = r.read().decode("utf-8", "replace")
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return {"_not_bloombee": True, "_raw": body[:200]}
                    # A BloomBee coordinator /healthz returns ok=true and includes
                    # the coordinator URL. Other services that happen to have
                    # /healthz returning JSON (Prometheus exporters, etc.) won't
                    # match this signature.
                    if data.get("ok") and (data.get("coordinator") or data.get("status")):
                        return data
                    return {"_not_bloombee": True, "_raw": body[:200]}
        except HTTPError as e:
            # 404 / 401 / 405 → something is listening but it's not a
            # BloomBee coordinator (or it's a BloomBee on a different port
            # space). Either way, don't try to bind.
            return {"_not_bloombee": True, "_http_status": e.code}
        except (URLError, OSError):
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--coordinator", default=None, help="Public coordinator URL embedded in join offers")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Model registry for /plan endpoint")
    args = parser.parse_args(argv)

    coordinator = args.coordinator or f"http://{args.host}:{args.port}"

    # Pre-flight: if a coordinator is already running here, tell the user
    # (don't fail with a cryptic OSError, and don't double-bind).
    existing = _probe_existing_coordinator(args.host, args.port)
    if existing and not existing.get("_not_bloombee"):
        # BloomBee coordinator is live — same or different state dir is fine,
        # but warn if state dirs differ so user doesn't accidentally serve
        # two unrelated heartbeats under the same port.
        existing_state = existing.get("state_dir", "")
        local_state = str(Path(args.state_dir).expanduser())
        same_state = existing_state == local_state
        print(
            f"\n{'='*60}\n"
            f"  🌙  BloomBee Coordinator is ALREADY RUNNING on "
            f"{args.host}:{args.port}\n"
            f"  {'─'*40}\n"
            f"  Dashboard  → {coordinator}/\n"
            f"  Health     → {coordinator}/healthz\n"
            f"  Swarm API  → {coordinator}/active\n"
            f"  Status     → {existing.get('status', '?')} · uptime={existing.get('uptime_seconds', '?')}s\n",
            flush=True,
        )
        if same_state:
            print(
                f"  Same state dir as running instance ({existing_state}).\n"
                f"  No need to start another one — your existing coordinator will\n"
                f"  pick up new peer heartbeats automatically.\n\n"
                f"  If you intended to restart it: Ctrl+C the running instance first,\n"
                f"  then re-run this command.\n"
                f"{'='*60}\n",
                flush=True,
            )
        else:
            print(
                f"  ⚠️  Running instance uses a different state dir ({existing_state!r}).\n"
                f"  Starting a new coordinator here would fail with EADDRINUSE.\n"
                f"  Either: (a) use the running instance, or (b) pick a different --port.\n"
                f"{'='*60}\n",
                flush=True,
            )
        return 0
    if existing and existing.get("_not_bloombee"):
        # Something else is on the port and isn't a BloomBee coordinator.
        # Bail cleanly instead of trying to bind (which would either fail
        # with EADDRINUSE, or worse, silently take over someone else's port).
        print(
            f"\n{'='*60}\n"
            f"  ❌  Port {args.port} is in use by a non-BloomBee service.\n"
            f"  {'─'*40}\n"
            f"  To fix: re-run with a different --port, e.g. --port 8788\n"
            f"  To find what holds it:  lsof -i :{args.port}\n"
            f"{'='*60}\n",
            file=sys.stderr, flush=True,
        )
        return 3

    try:
        server = create_server((args.host, args.port), state_dir=args.state_dir, coordinator=coordinator, registry=args.registry)
    except OSError as e:
        if getattr(e, "errno", None) == 48 or "Address already in use" in str(e):
            # Race: probe said free, but bind() lost the race. Give a clean hint.
            print(
                f"\n{'='*60}\n"
                f"  ❌  Port {args.port} is already in use.\n"
                f"  {'─'*40}\n"
                f"  Likely a BloomBee coordinator is already running here, or\n"
                f"  another service holds the port. To fix:\n\n"
                f"    1. Find what's listening:   lsof -i :{args.port}\n"
                f"    2. Use a different port:   re-run with --port 8788\n"
                f"    3. Verify the existing one: curl {coordinator}/healthz\n"
                f"{'='*60}\n",
                file=sys.stderr, flush=True,
            )
            return 2
        raise
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
    print(
        f"\n{'='*60}\n"
        f"  🌙  BloomBee Coordinator is LIVE\n"
        f"  {'─'*40}\n"
        f"  Dashboard  → {coordinator}/\n"
        f"  Health     → {coordinator}/healthz\n"
        f"  Swarm API  → {coordinator}/active\n"
        f"\n"
        f"  Open the operator dashboard for QR onboarding:\n"
        f"  python3 scripts/operator_dashboard.py \\\n"
        f"    --coordinator \"{coordinator}\" \\\n"
        f"    --out .local/operator-dashboard.html && open .local/operator-dashboard.html\n"
        f"\n"
        f"  Press Ctrl+C to stop.\n"
        f"{'='*60}\n",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
