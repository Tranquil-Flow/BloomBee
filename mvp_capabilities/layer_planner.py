#!/usr/bin/env python3
"""Plan contiguous BloomBee layer placement from peer capabilities.

This is a planning artifact only. It maps model layers onto peers by estimated
free-memory capacity so the coordinator/dashboard can show whether a live roster
could host every transformer block. It does not start servers and does not prove
inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry, synthetic_m4_laptops
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/layer_planner.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry, synthetic_m4_laptops
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster

CLAIM_BOUNDARY = "placement_plan_only_no_inference_proof"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _peer_free_gb(peer: dict[str, Any]) -> float:
    memory = peer.get("memory") or {}
    accelerator = peer.get("accelerator") or {}
    free = memory.get("free_gb")
    if free is None and accelerator.get("unified_memory"):
        free = accelerator.get("vram_free_gb")
    return _as_float(free)


def _model_required_gb(model: dict[str, Any]) -> float:
    return _as_float(model.get("recommended_min_free_mem_gb") or model.get("min_total_mem_gb"))


def plan_layer_placement(peers: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic contiguous layer assignments for one model.

    Capacity estimate is intentionally simple and conservative for the MVP:
    ``required_free_gb / num_layers`` gives an estimated per-layer memory budget,
    then each peer gets ``floor(free_gb / per_layer_required_gb)`` layers. Peers
    are sorted by hostname so repeated scans produce stable plans.
    """
    num_layers = _as_int(model.get("num_layers") or model.get("num_hidden_layers"))
    required_free_gb = _model_required_gb(model)
    if num_layers <= 0 or required_free_gb <= 0:
        return {
            "model_id": model.get("model_id"),
            "supported": False,
            "reason": "model is missing num_layers or memory requirement",
            "num_layers": num_layers,
            "required_free_gb": required_free_gb,
            "per_layer_required_gb": 0.0,
            "assigned_layers": 0,
            "missing_layers": max(num_layers, 0),
            "assignments": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }

    per_layer_required_gb = required_free_gb / num_layers
    assignments: list[dict[str, Any]] = []
    next_layer = 0
    sorted_peers = sorted(peers, key=lambda peer: str(peer.get("hostname") or "unknown"))

    for peer in sorted_peers:
        if next_layer >= num_layers:
            break
        hostname = str(peer.get("hostname") or "unknown")
        free_gb = round(_peer_free_gb(peer), 4)
        capacity_layers = int(math.floor(free_gb / per_layer_required_gb)) if per_layer_required_gb > 0 else 0
        if capacity_layers <= 0:
            continue
        layer_count = min(capacity_layers, num_layers - next_layer)
        start_layer = next_layer
        end_layer = start_layer + layer_count
        assignments.append(
            {
                "hostname": hostname,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "layer_count": layer_count,
                "free_gb": float(round(free_gb, 2)),
                "capacity_layers": capacity_layers,
            }
        )
        next_layer = end_layer

    assigned_layers = next_layer
    missing_layers = max(num_layers - assigned_layers, 0)
    supported = missing_layers == 0
    if supported:
        reason = f"capacity covers all {num_layers} layers across {len(assignments)} peer(s)"
    else:
        reason = f"capacity covers {assigned_layers}/{num_layers} layers; missing {missing_layers}"

    return {
        "model_id": model.get("model_id"),
        "supported": supported,
        "reason": reason,
        "num_layers": num_layers,
        "required_free_gb": round(required_free_gb, 2),
        "per_layer_required_gb": round(per_layer_required_gb, 4),
        "assigned_layers": assigned_layers,
        "missing_layers": missing_layers,
        "peer_count": len(peers),
        "assignments": assignments,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise SystemExit(f"model not found in registry: {model_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--model", required=True, help="Model ID from MODEL_REGISTRY.yaml")
    parser.add_argument("--cap-dir", action="append", default=None)
    parser.add_argument("--synthetic-m4-laptops", type=int, default=0)
    parser.add_argument("--synthetic-total-gb", type=float, default=24.0)
    parser.add_argument("--synthetic-free-gb", type=float, default=20.0)
    args = parser.parse_args(argv)

    peers = load_roster(args.cap_dir or [DEFAULT_CAP_DIR])
    if args.synthetic_m4_laptops:
        peers.extend(
            synthetic_m4_laptops(
                count=args.synthetic_m4_laptops,
                total_gb=args.synthetic_total_gb,
                free_gb=args.synthetic_free_gb,
            )
        )
    registry = load_registry(args.registry)
    model = _find_model(registry, args.model)
    print(json.dumps(plan_layer_placement(peers, model), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
