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
LAUNCH_COMMANDS_CLAIM_BOUNDARY = "launch_commands_only_no_server_started"


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
    """Estimate peer capacity in GB.

    Reads from multiple historical capability shapes so heartbeats from
    different clients all feed into the planner. Preference order:

      1. ``memory.free_gb`` — explicit free RAM field (Linux ``/proc/meminfo``)
      2. ``memory.available_gb`` — what psutil reports as available
      3. ``memory.total_gb`` — assume the model owns the full system RAM
      4. ``accelerator.vram_free_gb`` — discrete GPU VRAM
      5. ``disk.free_gb`` — last-resort model-on-disk (works only for the
         small models this MVP tests with; planner still flags unsupported
         when even this is below the model requirement)
    """
    memory = peer.get("memory") or {}
    accelerator = peer.get("accelerator") or {}
    disk = peer.get("disk") or {}
    for field in ("free_gb", "available_gb", "total_gb"):
        v = memory.get(field)
        if v is not None:
            return _as_float(v)
    if accelerator.get("vram_free_gb") is not None:
        return _as_float(accelerator.get("vram_free_gb"))
    v = disk.get("free_gb")
    if v is not None:
        return _as_float(v)
    return 0.0


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
            "base_model_id": model.get("base_model_id"),
            "quant_type": model.get("quant_type"),
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
    # First pass: compute each peer's capacity (layers it could hold if it
    # worked alone). Then distribute the model across ALL peers with
    # non-zero capacity, proportional to capacity, so a 16 GB laptop and
    # a 48 GB workstation both get a fair slice of the model rather than
    # the smallest peer greedily eating everything.
    sorted_peers = sorted(peers, key=lambda peer: str(peer.get("hostname") or "unknown"))
    peer_capacity: list[tuple[dict[str, Any], int]] = []
    for peer in sorted_peers:
        free_gb = round(_peer_free_gb(peer), 4)
        capacity_layers = int(math.floor(free_gb / per_layer_required_gb)) if per_layer_required_gb > 0 else 0
        if capacity_layers > 0:
            peer_capacity.append((peer, capacity_layers))

    total_capacity = sum(c for _, c in peer_capacity)
    assignments: list[dict[str, Any]] = []
    next_layer = 0
    if total_capacity > 0:
        # Proportional split: each peer gets cap * (num_layers / total_capacity),
        # capped at its own capacity. Largest-remainder method fills the
        # rounding deficit so the sum exactly equals num_layers when total
        # capacity covers the model.
        raw_shares = [(cap / total_capacity) * num_layers for _, cap in peer_capacity]
        # Cap each share at the peer's actual capacity so an over-provisioned
        # peer doesn't end up with more layers than it can fit.
        capped_shares = [min(int(math.floor(s)), cap) for s, (_, cap) in zip(raw_shares, peer_capacity)]
        remainders = sorted(
            enumerate(raw_shares),
            key=lambda kv: (kv[1] - int(math.floor(kv[1])), -peer_capacity[kv[0]][1]),
            reverse=True,
        )
        deficit = num_layers - sum(capped_shares)
        for idx, _ in remainders[: max(0, deficit)]:
            # Only top up if the peer still has spare capacity.
            if capped_shares[idx] < peer_capacity[idx][1]:
                capped_shares[idx] += 1
            else:
                # Find next peer with spare capacity.
                for jdx, _ in remainders:
                    if capped_shares[jdx] < peer_capacity[jdx][1]:
                        capped_shares[jdx] += 1
                        break

        for (peer, capacity_layers), layer_count in zip(peer_capacity, capped_shares):
            if layer_count <= 0:
                continue
            hostname = str(peer.get("hostname") or "unknown")
            free_gb = round(_peer_free_gb(peer), 4)
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
            if next_layer >= num_layers:
                break

    # If we still didn't fill all layers (defensive — shouldn't happen given
    # total_capacity check), top up greedily from largest peer down.
    if next_layer < num_layers and peer_capacity:
        for peer, capacity_layers in sorted(peer_capacity, key=lambda pc: -pc[1]):
            if next_layer >= num_layers:
                break
            if any(a["hostname"] == peer.get("hostname") for a in assignments):
                continue  # already in the plan; skip
            slot = min(capacity_layers, num_layers - next_layer)
            if slot <= 0:
                continue
            assignments.append(
                {
                    "hostname": str(peer.get("hostname") or "unknown"),
                    "start_layer": next_layer,
                    "end_layer": next_layer + slot,
                    "layer_count": slot,
                    "free_gb": float(round(_peer_free_gb(peer), 2)),
                    "capacity_layers": capacity_layers,
                }
            )
            next_layer += slot

    assigned_layers = next_layer
    missing_layers = max(num_layers - assigned_layers, 0)
    supported = missing_layers == 0
    if supported:
        reason = f"capacity covers all {num_layers} layers across {len(assignments)} peer(s)"
    else:
        reason = f"capacity covers {assigned_layers}/{num_layers} layers; missing {missing_layers}"

    return {
        "model_id": model.get("model_id"),
        "base_model_id": model.get("base_model_id"),
        "quant_type": model.get("quant_type"),
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


def _server_command(
    *,
    model_id: str,
    block_range: str,
    port: int,
    device: str,
    dtype: str,
    dht_prefix: str | None,
    initial_peer_placeholder: str | None,
    quant_type: str | None = None,
) -> str:
    parts = ["PYTHONPATH=.:src"]
    parts.extend(
        [
            # macOS no longer ships a bare `python` binary by default.
            # The QR/bootstrap path is already `python3 ...`, so generated
            # server commands should use the same portable command. Using
            # bare `python` caused fresh macOS peers to fail immediately
            # with exit code 127.
            "python3 -m bloombee.cli.run_server",
            model_id,
            f"--block_indices {block_range}",
            f"--device {device}",
            f"--torch_dtype {dtype}",
            f"--port {port}",
        ]
    )
    if quant_type:
        parts.append(f"--quant_type {str(quant_type).upper()}")
    if dht_prefix:
        parts.append(f"--dht_prefix {dht_prefix}")
    if initial_peer_placeholder:
        parts.append(f"--initial_peers '{initial_peer_placeholder}'")
    else:
        parts.append("--new_swarm")
    return " ".join(parts)


def attach_launch_commands(
    plan: dict[str, Any],
    *,
    device: str = "mps",
    dtype: str = "float16",
    base_port: int = 31337,
    dht_prefix: str | None = None,
) -> dict[str, Any]:
    """Attach copy/paste BloomBee server commands to a placement plan.

    This is still a planning artifact. It does not start servers; it gives the
    coordinator/operator exact commands to launch from each assignment.
    """
    updated = dict(plan)
    assignments: list[dict[str, Any]] = []
    raw_assignments = [dict(item) for item in (plan.get("assignments") or [])]
    seed_hostname: str | None = None
    if raw_assignments:
        # The seed peer only bootstraps the private DHT; it does not need
        # to serve layer 0. Pick the strongest assignment as seed so a
        # small alphabetically-first laptop does not become the critical
        # bootstrap node merely because its hostname sorts first.
        def _seed_rank(item: dict[str, Any]) -> tuple[float, float, int, str]:
            return (
                -float(item.get("capacity_layers") or 0),
                -float(item.get("free_gb") or 0),
                -int(item.get("layer_count") or 0),
                str(item.get("hostname") or ""),
            )

        seed_hostname = str(sorted(raw_assignments, key=_seed_rank)[0].get("hostname") or "peer-1")
    launch_model_id = str(plan.get("base_model_id") or plan.get("model_id"))
    quant_type = plan.get("quant_type")
    for index, assignment in enumerate(raw_assignments):
        item = dict(assignment)
        hostname = str(item.get("hostname") or f"peer-{index + 1}")
        port = base_port + index
        block_range = f"{item['start_layer']}:{item['end_layer']}"
        is_seed = hostname == seed_hostname
        initial_peer = None if is_seed else f"<SEED_MULTIADDR_FROM_{seed_hostname}>"
        item.update(
            {
                "role": "seed" if is_seed else "follower",
                "seed_hostname": seed_hostname,
                "port": port,
                "block_range": block_range,
                "launch_command": _server_command(
                    model_id=launch_model_id,
                    block_range=block_range,
                    port=port,
                    device=device,
                    dtype=dtype,
                    dht_prefix=dht_prefix,
                    initial_peer_placeholder=initial_peer,
                    quant_type=str(quant_type) if quant_type else None,
                ),
            }
        )
        assignments.append(item)
    updated["assignments"] = assignments
    updated["launch_commands_claim_boundary"] = LAUNCH_COMMANDS_CLAIM_BOUNDARY
    updated["launch_command_defaults"] = {
        "device": device,
        "dtype": dtype,
        "base_port": base_port,
        "dht_prefix": dht_prefix,
        "launch_model_id": launch_model_id,
        "route_model_id": plan.get("model_id"),
        "quant_type": quant_type,
        "seed_hostname": seed_hostname,
    }
    updated["launch_command_notes"] = [
        "Commands are a runbook only; no server was started by the planner.",
        "Start the seed command first, copy its printed multiaddr, then replace later <SEED_MULTIADDR_FROM_...> placeholders in --initial_peers.",
        "Follower commands use the current run_server --initial_peers CLI flag; BLOOMBEE_INITIAL_PEERS is not read by run_server.py.",
        "Only promote proof gates after direct client evidence verifies finite outputs/gradients.",
    ]
    return updated


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
    parser.add_argument("--include-launch-commands", action="store_true")
    parser.add_argument("--launch-device", default="mps")
    parser.add_argument("--launch-dtype", default="float16")
    parser.add_argument("--base-port", type=int, default=31337)
    parser.add_argument("--dht-prefix", default=None)
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
    plan = plan_layer_placement(peers, model)
    if args.include_launch_commands:
        plan = attach_launch_commands(
            plan,
            device=args.launch_device,
            dtype=args.launch_dtype,
            base_port=args.base_port,
            dht_prefix=args.dht_prefix,
        )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
