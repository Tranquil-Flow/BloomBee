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
import sys
from pathlib import Path
from typing import Any

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


def build_join_layer_plan(
    *,
    state_dir: str | Path,
    token: str,
    model: dict[str, Any],
    now: int | None = None,
    max_age_seconds: int = 30,
    include_launch_commands: bool = False,
    device: str = "mps",
    dtype: str = "float16",
    base_port: int = 31337,
    dht_prefix: str | None = None,
) -> dict[str, Any]:
    active = load_active_heartbeats(state_dir, token=token, now=now, max_age_seconds=max_age_seconds)
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
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "heartbeat_claim_boundary": HEARTBEAT_CLAIM_BOUNDARY,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", required=True, help="Model ID from MODEL_REGISTRY.yaml")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--now", type=int, default=None)
    parser.add_argument("--max-age-seconds", type=int, default=30)
    parser.add_argument("--include-launch-commands", action="store_true")
    parser.add_argument("--launch-device", default="mps")
    parser.add_argument("--launch-dtype", default="float16")
    parser.add_argument("--base-port", type=int, default=31337)
    parser.add_argument("--dht-prefix", default=None)
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    model = _find_model(registry, args.model)
    payload = build_join_layer_plan(
        state_dir=args.state_dir,
        token=args.token,
        model=model,
        now=args.now,
        max_age_seconds=args.max_age_seconds,
        include_launch_commands=args.include_launch_commands,
        device=args.launch_device,
        dtype=args.launch_dtype,
        base_port=args.base_port,
        dht_prefix=args.dht_prefix,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
