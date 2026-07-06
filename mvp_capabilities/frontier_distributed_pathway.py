#!/usr/bin/env python3
"""Fail-closed pathway planner for frontier models and BloomBee distribution.

Claim boundary: ``frontier_distributed_pathway_plan_no_live_inference_claim``.

This answers a narrow but recurring operator question: can a frontier model that
currently appears as an external GGUF/llama.cpp target be used in the main
BloomBee distributed-inference path? The planner never attempts inference and
never promotes proof/demo status. It separates:

* the **desired** BloomBee block-parallel end state, where peers can pool memory
  by owning transformer block ranges, from
* the **current** external-runtime path, where a GGUF is loaded by one
  llama.cpp/vMLX process and peer RAM is not additive.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "frontier_distributed_pathway_plan_no_live_inference_claim"


@dataclass(frozen=True)
class TargetSpec:
    key: str
    aliases: tuple[str, ...]
    model_id: str
    hf_model_type: str | None
    external_framework: str
    external_single_host_required_free_gb: float
    external_quant_label: str
    bloombee_wrapper_name: str | None
    config_scan_required: bool
    adapter_steps: tuple[str, ...]
    default_next_lane: str


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="minimax_m27_reap",
        aliases=("minimax-m27-reap", "m27", "m2.7-reap", "m27-reap", "minimax_m27_reap"),
        model_id="dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B",
        hf_model_type="minimax_m2",
        external_framework="llama.cpp",
        external_single_host_required_free_gb=40.8,
        external_quant_label="i1-IQ2_XXS",
        bloombee_wrapper_name="minimax_m2",
        config_scan_required=False,
        adapter_steps=(
            "add/verify a BloomBee minimax_m2 block wrapper for the exact REAP architecture",
            "model sparse/MoE router state and MiniMax attention/cache semantics inside a block",
            "prove one-block server forward/backward or forward-only direct RPC on a clean host",
            "prove multi-block split across two peers before any route/demo promotion",
            "then run full/cache generation and load gates for the exact model row",
        ),
        default_next_lane="native_bloombee_adapter_before_route_promotion",
    ),
    TargetSpec(
        key="qwen3_6_35b_a3b",
        aliases=("qwen36a", "qweb36a", "qwen3.6-35b-a3b", "qwen3_6_35b_a3b", "qwen36"),
        model_id="Qwen/Qwen3.6-35B-A3B",
        hf_model_type=None,
        external_framework="llama.cpp/GGUF",
        external_single_host_required_free_gb=80.0,
        external_quant_label="unknown_until_exact_quant_selected",
        bloombee_wrapper_name=None,
        config_scan_required=True,
        adapter_steps=(
            "scan the exact HF config/model_type for Qwen3.6-35B-A3B or the user's exact Qwen36A row",
            "map the config to an existing BloomBee family only if the wrapper/state-cache contract matches",
            "if no wrapper matches, add wrapper tests before any server proof",
            "if using only a GGUF, prove llama.cpp RPC/distributed backend separately before pooling RAM claims",
        ),
        default_next_lane="config_scan_then_native_bloombee_adapter_or_external_rpc",
    ),
)

_ALIAS_TO_TARGET = {alias: target for target in TARGETS for alias in (target.key, *target.aliases)}


def _parse_peer(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"peer spec must be hostname:total_gb:free_gb, got {spec!r}")
    hostname, total_gb, free_gb = parts
    return {"hostname": hostname, "memory": {"total_gb": float(total_gb), "free_gb": float(free_gb)}}


def _normalise_target(target: str) -> TargetSpec:
    key = target.strip().lower().replace(" ", "-")
    if key not in _ALIAS_TO_TARGET:
        known = ", ".join(sorted(_ALIAS_TO_TARGET))
        raise ValueError(f"unknown target {target!r}; known aliases: {known}")
    return _ALIAS_TO_TARGET[key]


def _default_peers() -> list[dict[str, Any]]:
    return [
        {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
        {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 33.5}},
    ]


def _memory_totals(peers: list[dict[str, Any]]) -> dict[str, float]:
    total = 0.0
    free = 0.0
    largest_total = 0.0
    largest_free = 0.0
    for peer in peers:
        memory = peer.get("memory") if isinstance(peer, dict) else None
        if not isinstance(memory, dict):
            continue
        total_gb = float(memory.get("total_gb", 0.0) or 0.0)
        free_gb = float(memory.get("free_gb", 0.0) or 0.0)
        total += total_gb
        free += free_gb
        largest_total = max(largest_total, total_gb)
        largest_free = max(largest_free, free_gb)
    return {
        "combined_total_gb": round(total, 1),
        "combined_free_gb": round(free, 1),
        "largest_single_host_total_gb": round(largest_total, 1),
        "largest_single_host_free_gb": round(largest_free, 1),
    }


def _best_peer(peers: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [peer for peer in peers if isinstance(peer.get("memory"), dict)]
    if not valid:
        return None
    return max(valid, key=lambda peer: float(peer["memory"].get("free_gb", 0.0) or 0.0))


def build_frontier_distributed_pathway_report(
    *,
    target: str,
    peers: list[dict[str, Any]] | None = None,
    native_wrapper_proven: bool = False,
    one_block_proven: bool = False,
) -> dict[str, Any]:
    """Build a claim-bounded main-pathway vs external-runtime report."""
    spec = _normalise_target(target)
    peers = peers or _default_peers()
    totals = _memory_totals(peers)
    best = _best_peer(peers)
    best_free = float(best["memory"].get("free_gb", 0.0) or 0.0) if best else 0.0
    required = spec.external_single_host_required_free_gb
    shortfall = round(max(0.0, required - best_free), 1)

    main_blockers: list[str] = []
    if spec.config_scan_required:
        main_blockers.append("exact_hf_config_not_scanned_in_repo")
    if not native_wrapper_proven:
        wrapper = spec.bloombee_wrapper_name or "unknown_or_unverified"
        main_blockers.append(f"missing_native_bloombee_wrapper:{wrapper}")
    if native_wrapper_proven and not one_block_proven:
        main_blockers.append("one_block_server_proof_missing")

    usable_now = not main_blockers
    if usable_now:
        next_lane = "main_pathway_multiblock_and_generation_proofs"
    else:
        next_lane = spec.default_next_lane

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "target_key": spec.key,
        "model_id": spec.model_id,
        "peers": peers,
        "memory_summary": totals,
        "main_bloombee_pathway": {
            "desired_end_state": True,
            "implementation": "BloomBee block-parallel DHT/RPC with peers owning contiguous transformer block ranges",
            "usable_now": usable_now,
            "can_pool_peer_memory_now": usable_now,
            "can_pool_peer_memory_after_adapter": True,
            "config_scan_required": spec.config_scan_required,
            "hf_model_type": spec.hf_model_type,
            "native_wrapper_proven": native_wrapper_proven,
            "one_block_proven": one_block_proven,
            "blocked_reasons": main_blockers,
            "required_adapter_steps": list(spec.adapter_steps),
        },
        "external_runtime_pathway": {
            "framework": spec.external_framework,
            "quant_or_format": spec.external_quant_label,
            "single_host_required_free_gb": required,
            "can_pool_peer_memory_now": False,
            "why_not_additive": "A GGUF external runtime maps/loads the model inside one process on one host unless a separate distributed backend is proven.",
            "best_peer": (
                {
                    "hostname": best["hostname"],
                    "total_gb": float(best["memory"].get("total_gb", 0.0) or 0.0),
                    "free_gb": best_free,
                }
                if best
                else None
            ),
            "best_peer_shortfall_gb": shortfall,
            "single_host_gate_pass": shortfall == 0.0,
        },
        "recommendation": {
            "next_engineering_lane": next_lane,
            "summary": (
                "Use the main BloomBee pathway only after the exact model has a native wrapper/state-cache contract and at least a one-block proof. "
                "Until then, external GGUF smoke is useful but is a single-host gate."
            ),
        },
        "proof_status_update": {},
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "live_run_attempted": False,
        "do_not_claim": [
            "no live inference was attempted by this planner",
            "no wall-clock speedup proof",
            "no demo/status promotion from planner output alone",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Target alias, e.g. minimax-m27-reap or qwen36a")
    parser.add_argument("--peer", action="append", default=[], help="Peer spec hostname:total_gb:free_gb")
    parser.add_argument("--native-wrapper-proven", action="store_true")
    parser.add_argument("--one-block-proven", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    report = build_frontier_distributed_pathway_report(
        target=args.target,
        peers=[_parse_peer(peer) for peer in args.peer] if args.peer else None,
        native_wrapper_proven=args.native_wrapper_proven,
        one_block_proven=args.one_block_proven,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
