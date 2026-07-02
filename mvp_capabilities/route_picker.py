#!/usr/bin/env python3
"""Pick the strongest feasible model for a BloomBee peer swarm.

The router is conservative: it separates solo fit from aggregate swarm fit,
labels block-parallel candidates honestly, and treats measured benchmark data
as stronger evidence than estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/route_picker.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster


DEFAULT_REGISTRY = Path(__file__).with_name("MODEL_REGISTRY.yaml")
MVP_MODEL_ID = "Qwen/Qwen3-30B-A3B"
STRETCH_MODEL_ID = "Qwen/Qwen3-235B-A22B"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = payload.get("models") or []
    enriched: list[dict[str, Any]] = []
    for rank, model in enumerate(models):
        model = dict(model)
        model.setdefault("quality_rank", _default_quality_rank(model, rank))
        model["mvp_target"] = model.get("model_id") == MVP_MODEL_ID or bool(model.get("mvp_target"))
        model["stretch_target"] = model.get("model_id") == STRETCH_MODEL_ID or bool(model.get("stretch_target"))
        enriched.append(model)
    return enriched


def _default_quality_rank(model: dict[str, Any], fallback_rank: int) -> float:
    # Quality proxy: total params, with a boost for MoE active efficiency.
    params = float(model.get("params_b") or 0.0)
    active = float(model.get("active_params_b") or params or 0.0)
    moe_bonus = 10.0 if model.get("supports_moe") else 0.0
    # Keep deterministic ordering for same-sized models.
    return params + moe_bonus + fallback_rank / 1000.0 + active / 10000.0


def _peer_free_gb(peer: dict[str, Any]) -> float:
    memory = peer.get("memory") or {}
    accelerator = peer.get("accelerator") or {}
    free = memory.get("free_gb")
    if free is None and accelerator.get("unified_memory"):
        free = accelerator.get("vram_free_gb")
    try:
        return float(free or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _model_required_gb(model: dict[str, Any]) -> float:
    try:
        return float(model.get("recommended_min_free_mem_gb") or model.get("min_total_mem_gb") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bench_score(model: dict[str, Any], bench_matrix: dict[str, Any] | None) -> float:
    if not bench_matrix:
        return 0.0
    values: list[float] = []
    model_id = model.get("model_id")
    for peer_result in bench_matrix.values() if isinstance(bench_matrix, dict) else []:
        if not isinstance(peer_result, dict):
            continue
        entry = peer_result.get(model_id) or peer_result.get("models", {}).get(model_id)
        if isinstance(entry, dict):
            try:
                values.append(float(entry.get("decode_tok_per_s") or 0.0))
            except (TypeError, ValueError):
                pass
    return max(values) if values else 0.0


def evaluate_model(
    peers: list[dict[str, Any]],
    model: dict[str, Any],
    bench_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_gb = _model_required_gb(model)
    free_by_peer = [(peer.get("hostname", "unknown"), _peer_free_gb(peer)) for peer in peers]
    solo_hosts = [host for host, free in free_by_peer if free >= required_gb]
    total_free = sum(free for _, free in free_by_peer)
    supported = False
    placement = "unsupported"

    if solo_hosts:
        supported = True
        placement = "solo" if len(solo_hosts) == 1 else "replicated"
        reason = f"{len(solo_hosts)} peer(s) have >= {required_gb:.1f}GB free"
    elif peers and total_free >= required_gb:
        supported = True
        placement = "block_parallel_candidate"
        reason = f"aggregate swarm free memory {total_free:.1f}GB >= {required_gb:.1f}GB required"
    else:
        reason = f"requires {required_gb:.1f}GB free; swarm has {total_free:.1f}GB"

    return {
        "model_id": model.get("model_id"),
        "supported": supported,
        "placement": placement,
        "reason": reason,
        "required_free_gb": required_gb,
        "swarm_free_gb": round(total_free, 2),
        "solo_hosts": solo_hosts,
        "mvp_target": bool(model.get("mvp_target")),
        "stretch_target": bool(model.get("stretch_target")),
        "supports_moe": bool(model.get("supports_moe")),
        "quality_rank": float(model.get("quality_rank") or 0.0),
        "measured_decode_tok_per_s": _bench_score(model, bench_matrix),
    }


def synthetic_m4_laptops(
    *,
    count: int = 10,
    total_gb: float = 24.0,
    free_gb: float = 20.0,
    prefix: str = "m4-laptop",
) -> list[dict[str, Any]]:
    """Create deterministic synthetic peers for pre-showcase planning.

    Synthetic peers let the MVP route picker treat the 10-laptop swarm as a
    first-class target before the physical showcase. They are explicitly marked
    so they cannot be mistaken for evidence from real hardware.
    """
    return [
        {
            "hostname": f"{prefix}-{index:02d}",
            "synthetic": True,
            "memory": {"total_gb": float(total_gb), "free_gb": float(free_gb)},
            "accelerator": {
                "device": "mps",
                "unified_memory": True,
                "vram_total_gb": float(total_gb),
                "vram_free_gb": float(free_gb),
            },
            "network": {"tailscale_ip": None},
        }
        for index in range(1, count + 1)
    ]


def choose_best_route(
    peers: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    scenario: str | None = None,
    requested_model: str | None = None,
    bench_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if requested_model:
        for model in registry:
            if model.get("model_id") == requested_model:
                return evaluate_model(peers, model, bench_matrix)
        return {
            "model_id": requested_model,
            "supported": False,
            "placement": "unsupported",
            "reason": "model not found in registry",
            "mvp_target": requested_model == MVP_MODEL_ID,
            "stretch_target": requested_model == STRETCH_MODEL_ID,
        }

    candidates = [evaluate_model(peers, model, bench_matrix) for model in registry]
    supported = [candidate for candidate in candidates if candidate["supported"]]
    if not supported:
        return {
            "model_id": None,
            "supported": False,
            "placement": "unsupported",
            "reason": "no model fits current swarm",
        }

    if scenario == "mvp-10-laptop":
        for candidate in supported:
            if candidate["model_id"] == MVP_MODEL_ID:
                return candidate

    return max(
        supported,
        key=lambda item: (
            1 if item.get("mvp_target") else 0,
            item.get("quality_rank") or 0.0,
            item.get("measured_decode_tok_per_s") or 0.0,
            -float(item.get("required_free_gb") or 0.0),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--cap-dir", action="append", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model", dest="requested_model", default=None)
    parser.add_argument("--bench-matrix", default=None)
    parser.add_argument("--synthetic-m4-laptops", type=int, default=0, help="Append N synthetic M4 laptop peers")
    parser.add_argument("--synthetic-total-gb", type=float, default=24.0)
    parser.add_argument("--synthetic-free-gb", type=float, default=20.0)
    args = parser.parse_args(argv)

    bench_matrix = None
    if args.bench_matrix:
        bench_matrix = json.loads(Path(args.bench_matrix).expanduser().read_text(encoding="utf-8"))
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
    print(
        json.dumps(
            choose_best_route(
                peers,
                registry,
                scenario=args.scenario,
                requested_model=args.requested_model,
                bench_matrix=bench_matrix,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
