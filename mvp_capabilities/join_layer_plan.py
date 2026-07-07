#!/usr/bin/env python3
"""Build launch-ready layer plans from active join heartbeats.

This bridges the self-serve join coordinator to the layer planner: active token-
scoped heartbeats become a roster of capabilities, then `layer_planner.py` emits
contiguous block assignments and optional BloomBee server launch commands.

It does not start servers and does not prove inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    from mvp_capabilities.join_coordinator import DEFAULT_STATE_DIR, load_active_heartbeats
    from mvp_capabilities.layer_planner import attach_launch_commands, plan_layer_placement
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/join_layer_plan.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.join_coordinator import DEFAULT_STATE_DIR, load_active_heartbeats
    from mvp_capabilities.layer_planner import attach_launch_commands, plan_layer_placement
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry

CLAIM_BOUNDARY = "joined_roster_layer_plan_only_no_inference_proof"
HEARTBEAT_CLAIM_BOUNDARY = "heartbeat_roster_only_no_inference_proof"
LAUNCH_READINESS_CLAIM_BOUNDARY = "launch_readiness_checklist_only_no_server_started"
MULTIADDR_RESOLUTION_CLAIM_BOUNDARY = "launch_multiaddr_resolution_only_no_server_started"
HTTP_SOURCE = "coordinator_http_active"
LOCAL_SOURCE = "local_state_dir"
_PLACEHOLDER_RE = re.compile(r"<[^>]+>")
_SEED_PLACEHOLDER_RE = re.compile(r"^<SEED_MULTIADDR_FROM_(?P<hostname>.+)>$")


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise SystemExit(f"model not found in registry: {model_id}")


def _capabilities_from_heartbeats(active_heartbeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _sorted_heartbeats(active_heartbeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(active_heartbeats, key=lambda item: str(item.get("peer_id") or ""))


def _default_json_fetcher(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - operator-provided local coordinator URL
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("coordinator /active response must be a JSON object")
    return payload


def fetch_active_heartbeats(
    coordinator_url: str,
    *,
    token: str,
    now: int | None = None,
    max_age_seconds: int = 30,
    fetcher=_default_json_fetcher,
) -> dict[str, Any]:
    query: dict[str, str] = {"token": token, "max_age_seconds": str(max_age_seconds)}
    if now is not None:
        query["now"] = str(now)
    url = f"{coordinator_url.rstrip('/')}/active?{urlencode(query)}"
    payload = fetcher(url)
    if not isinstance(payload, dict):
        raise ValueError("coordinator /active response must be a JSON object")
    active = payload.get("active_peers")
    if not isinstance(active, list):
        raise ValueError("coordinator /active response missing active_peers list")
    return payload


def materialize_launch_readiness(plan: dict[str, Any]) -> dict[str, Any]:
    """Build a no-execution preflight checklist for launch-command runbooks."""
    placement = plan.get("placement") or {}
    assignments = [item for item in placement.get("assignments") or [] if isinstance(item, dict)]
    unresolved: list[str] = []
    server_checks: list[dict[str, Any]] = []
    operator_steps: list[str] = []
    for index, item in enumerate(assignments):
        hostname = str(item.get("hostname") or f"server-{index}")
        command = str(item.get("launch_command") or "")
        placeholders = _PLACEHOLDER_RE.findall(command)
        for placeholder in placeholders:
            if placeholder not in unresolved:
                unresolved.append(placeholder)
        blocked_by: list[str] = []
        if not command:
            blocked_by.append("missing_launch_command")
        if placeholders:
            blocked_by.append("unresolved_multiaddr_placeholder")
        role = str(item.get("role") or ("seed" if index == 0 else "follower"))
        if role == "seed":
            operator_steps.append(f"start seed server {hostname} and capture its announced multiaddr")
        for placeholder in placeholders:
            operator_steps.append(f"replace {placeholder} in {hostname} command before starting it")
        server_checks.append(
            {
                "hostname": hostname,
                "role": role,
                "block_range": item.get("block_range"),
                "port": item.get("port"),
                "launch_command": command,
                "unresolved_placeholders": placeholders,
                "blocked_by": blocked_by,
                "ready": not blocked_by,
            }
        )
    ready_to_start = bool(assignments) and not unresolved and all(check["ready"] for check in server_checks)
    return {
        "claim_boundary": LAUNCH_READINESS_CLAIM_BOUNDARY,
        "multiaddr_resolution_claim_boundary": placement.get("multiaddr_resolution_claim_boundary"),
        "server_count": len(assignments),
        "ready_to_start": ready_to_start,
        "unresolved_placeholders": unresolved,
        "operator_steps": operator_steps,
        "server_checks": server_checks,
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": "resolve placeholders, start servers manually, capture multiaddrs/logs, then run a proof harness",
    }


def parse_seed_multiaddr(raw: str) -> tuple[str, str]:
    """Parse a HOST=MULTIADDR CLI value without validating the libp2p address."""
    hostname, separator, multiaddr = raw.partition("=")
    hostname = hostname.strip()
    multiaddr = multiaddr.strip()
    if not separator or not hostname or not multiaddr:
        raise argparse.ArgumentTypeError("seed multiaddr must be HOST=/ip4/.../p2p/...")
    return hostname, multiaddr


def resolve_launch_multiaddrs(plan: dict[str, Any], seed_multiaddrs: dict[str, str] | None) -> dict[str, Any]:
    """Replace generated seed placeholders with operator-captured multiaddrs.

    This only rewrites launch-command runbooks. It does not start a server, check
    connectivity, or prove inference.
    """
    replacements = {str(host): str(addr) for host, addr in (seed_multiaddrs or {}).items() if str(host) and str(addr)}
    if not replacements:
        return plan

    updated = dict(plan)
    placement = dict(updated.get("placement") or {})
    assignments: list[dict[str, Any]] = []
    resolved_hosts: set[str] = set()
    unresolved: list[str] = []
    for item in placement.get("assignments") or []:
        assignment = dict(item)
        command = str(assignment.get("launch_command") or "")
        for placeholder in _PLACEHOLDER_RE.findall(command):
            match = _SEED_PLACEHOLDER_RE.match(placeholder)
            hostname = match.group("hostname") if match else None
            replacement = replacements.get(hostname or "")
            if replacement:
                command = command.replace(placeholder, replacement)
                resolved_hosts.add(str(hostname))
            elif placeholder not in unresolved:
                unresolved.append(placeholder)
        assignment["launch_command"] = command
        assignments.append(assignment)

    placement["assignments"] = assignments
    placement["multiaddr_resolution_claim_boundary"] = MULTIADDR_RESOLUTION_CLAIM_BOUNDARY
    placement["resolved_multiaddr_hosts"] = sorted(resolved_hosts)
    placement["unresolved_multiaddr_placeholders"] = unresolved
    notes = list(placement.get("launch_command_notes") or [])
    notes.append("Seed multiaddrs were substituted from operator-captured values; no server was started by this planner.")
    placement["launch_command_notes"] = notes
    updated["placement"] = placement
    return updated


def _build_from_active_heartbeats(
    *,
    active: list[dict[str, Any]],
    token: str,
    model: dict[str, Any],
    heartbeat_claim_boundary: str,
    source: str,
    include_launch_commands: bool = False,
    include_launch_readiness: bool = False,
    device: str = "mps",
    dtype: str = "float16",
    base_port: int = 31337,
    dht_prefix: str | None = None,
    seed_multiaddrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    active = _sorted_heartbeats(active)
    peers = _capabilities_from_heartbeats(active)
    placement = plan_layer_placement(peers, model)
    if include_launch_commands:
        placement = attach_launch_commands(
            placement,
            device=device,
            dtype=dtype,
            base_port=base_port,
            dht_prefix=dht_prefix,
        )
    payload = {
        "claim_boundary": CLAIM_BOUNDARY,
        "heartbeat_claim_boundary": heartbeat_claim_boundary,
        "source": source,
        "token": token,
        "model_id": model.get("model_id"),
        "active_peer_count": len(active),
        "planner_peer_count": len(peers),
        "active_heartbeats": active,
        "planner_peers": peers,
        "placement": placement,
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": "start generated server commands manually, capture multiaddrs, run direct client proof, then verify with one_block_proof.py or generation parity harness",
    }
    if include_launch_commands and seed_multiaddrs:
        payload = resolve_launch_multiaddrs(payload, seed_multiaddrs)
    if include_launch_readiness:
        payload["launch_readiness"] = materialize_launch_readiness(payload)
    return payload


def build_join_layer_plan_from_active_payload(
    active_payload: dict[str, Any],
    *,
    model: dict[str, Any],
    include_launch_commands: bool = False,
    include_launch_readiness: bool = False,
    device: str = "mps",
    dtype: str = "float16",
    base_port: int = 31337,
    dht_prefix: str | None = None,
    seed_multiaddrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    active = active_payload.get("active_peers")
    if not isinstance(active, list):
        raise ValueError("active payload missing active_peers list")
    return _build_from_active_heartbeats(
        active=[item for item in active if isinstance(item, dict)],
        token=str(active_payload.get("token") or ""),
        model=model,
        heartbeat_claim_boundary=str(active_payload.get("claim_boundary") or HEARTBEAT_CLAIM_BOUNDARY),
        source=HTTP_SOURCE,
        include_launch_commands=include_launch_commands,
        include_launch_readiness=include_launch_readiness,
        device=device,
        dtype=dtype,
        base_port=base_port,
        dht_prefix=dht_prefix,
        seed_multiaddrs=seed_multiaddrs,
    )


def build_join_layer_plan(
    *,
    state_dir: str | Path,
    token: str,
    model: dict[str, Any],
    now: int | None = None,
    max_age_seconds: int = 30,
    include_launch_commands: bool = False,
    include_launch_readiness: bool = False,
    device: str = "mps",
    dtype: str = "float16",
    base_port: int = 31337,
    dht_prefix: str | None = None,
    seed_multiaddrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    active = load_active_heartbeats(state_dir, token=token, now=now, max_age_seconds=max_age_seconds)
    return _build_from_active_heartbeats(
        active=active,
        token=token,
        model=model,
        heartbeat_claim_boundary=HEARTBEAT_CLAIM_BOUNDARY,
        source=LOCAL_SOURCE,
        include_launch_commands=include_launch_commands,
        include_launch_readiness=include_launch_readiness,
        device=device,
        dtype=dtype,
        base_port=base_port,
        dht_prefix=dht_prefix,
        seed_multiaddrs=seed_multiaddrs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--coordinator-url", default=None, help="Fetch active peers from coordinator /active instead of local state dir")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", required=True, help="Model ID from MODEL_REGISTRY.yaml")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--now", type=int, default=None)
    parser.add_argument("--max-age-seconds", type=int, default=30)
    parser.add_argument("--include-launch-commands", action="store_true")
    parser.add_argument("--include-launch-readiness", action="store_true", help="Embed no-execution readiness checklist for generated launch commands")
    parser.add_argument("--launch-device", default="mps")
    parser.add_argument("--launch-dtype", default="float16")
    parser.add_argument("--base-port", type=int, default=31337)
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument(
        "--seed-multiaddr",
        action="append",
        default=[],
        type=parse_seed_multiaddr,
        metavar="HOST=MULTIADDR",
        help="Resolve <SEED_MULTIADDR_FROM_HOST> placeholders in generated follower launch commands",
    )
    args = parser.parse_args(argv)
    seed_multiaddrs = dict(args.seed_multiaddr)

    registry = load_registry(args.registry)
    model = _find_model(registry, args.model)
    if args.coordinator_url:
        active_payload = fetch_active_heartbeats(
            args.coordinator_url,
            token=args.token,
            now=args.now,
            max_age_seconds=args.max_age_seconds,
        )
        payload = build_join_layer_plan_from_active_payload(
            active_payload,
            model=model,
            include_launch_commands=args.include_launch_commands,
            include_launch_readiness=args.include_launch_readiness,
            device=args.launch_device,
            dtype=args.launch_dtype,
            base_port=args.base_port,
            dht_prefix=args.dht_prefix,
            seed_multiaddrs=seed_multiaddrs,
        )
    else:
        payload = build_join_layer_plan(
            state_dir=args.state_dir,
            token=args.token,
            model=model,
            now=args.now,
            max_age_seconds=args.max_age_seconds,
            include_launch_commands=args.include_launch_commands,
            include_launch_readiness=args.include_launch_readiness,
            device=args.launch_device,
            dtype=args.launch_dtype,
            base_port=args.base_port,
            dht_prefix=args.dht_prefix,
            seed_multiaddrs=seed_multiaddrs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
