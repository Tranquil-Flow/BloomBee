#!/usr/bin/env python3
"""Claim-bounded quantized Qwen3-30B route-lane planner.

This is a planning/proof-lane helper, not a serving backend. It consumes the
committed Fable quantization foundation and makes three guardrails explicit:

1. Qwen3-30B-A3B@int8 can have different memory math than fp16.
2. Quantized proof rows are keyed separately (``model_id@int8``) and never
   inherit fp16 proof gates.
3. No quantized route is demo-safe until the exact quantized row passes
   full_generation, cache_generation, and multi_request_load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.model_compat_scan import PROOF_KEYS, is_demo_safe, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, evaluate_model, load_registry
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import PROOF_KEYS, is_demo_safe, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, evaluate_model, load_registry

MODEL_ID = "Qwen/Qwen3-30B-A3B"
QUANT_TYPE = "int8"
ROUTE_ID = f"{MODEL_ID}@{QUANT_TYPE}"
CLAIM_BOUNDARY = "quantized_route_lane_planning_only_no_serving_proof"
QUANT_SCHEME = "moe_int8_experts+qint8_attn"
SPIKE_ARTIFACT = "mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json"
DEFAULT_PROOF_STATUS = Path(__file__).with_name("PROOF_STATUS.yaml")
DEFAULT_EVIDENCE = Path(__file__).with_name("distributed_evidence") / "post_mvp" / "quantized-qwen30b-route-lane-20260704.json"
# Best real-dim Qwen3-MoE run from the quantized-block spike: custom int8
# experts + qint8 attention on CPU/MPS both measured 1.996x block weight shrink.
QWEN3_MOE_INT8_COMPRESSION_RATIO = 1.996

OPERATOR_NEXT_STEPS = [
    "base Qwen/Qwen3-30B-A3B@int8 is demo-safe under the current full/cache/load/token-parity gates",
    "keep Qwen/Qwen3-30B-A3B fp16 and @int8 proof rows separate; do not inherit gates across rows",
    "next expensive parity target is Qwen/Qwen3-30B-A3B-Instruct-2507@int8 cache_generation",
]

GUARDRAILS = [
    "do not update Qwen/Qwen3-30B-A3B fp16 proof row",
    "do not inherit fp16 proof gates into Qwen/Qwen3-30B-A3B@int8",
    "do not mark demo_safe until quantized full_generation/cache_generation gates pass and token_parity is exact",
]


def _default_status() -> dict[str, str]:
    return {key: "pending" for key in PROOF_KEYS}


def _status_for(proof_status: dict[str, dict[str, str]] | None, key: str) -> dict[str, str]:
    status = _default_status()
    if proof_status and key in proof_status:
        status.update({str(k): str(v) for k, v in proof_status[key].items()})
    return status


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


def _memory_route(peers: list[dict[str, Any]], *, required_gb: float) -> dict[str, Any]:
    free_by_peer = [(str(peer.get("hostname", "unknown")), _peer_free_gb(peer)) for peer in peers]
    solo_hosts = [host for host, free in free_by_peer if free >= required_gb]
    total_free = sum(free for _, free in free_by_peer)
    if solo_hosts:
        placement = "solo" if len(solo_hosts) == 1 else "replicated"
        memory_fit = True
        reason = f"{len(solo_hosts)} peer(s) have >= {required_gb:.1f}GB free"
    elif peers and total_free >= required_gb:
        placement = "block_parallel_candidate"
        memory_fit = True
        reason = f"aggregate swarm free memory {total_free:.1f}GB >= {required_gb:.1f}GB required"
    else:
        placement = "unsupported"
        memory_fit = False
        reason = f"requires {required_gb:.1f}GB free; swarm has {total_free:.1f}GB"
    return {
        "memory_fit": memory_fit,
        "supported": memory_fit,
        "placement": placement,
        "reason": reason,
        "required_free_gb": round(required_gb, 1),
        "swarm_free_gb": round(total_free, 2),
        "solo_hosts": solo_hosts,
    }


def _next_gate(status: dict[str, str]) -> str | None:
    for gate in PROOF_KEYS:
        if status.get(gate) != "passed":
            return gate
    return None


def _demo_safe(status: dict[str, str]) -> bool:
    # Shared policy: quantized rows also need token_parity: exact.
    return is_demo_safe(status, quant_type=QUANT_TYPE)


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise ValueError(f"model not found in registry: {model_id}")


def build_quantized_qwen30b_lane(
    *,
    peers: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    proof_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    model = _find_model(registry, MODEL_ID)
    fp16_route = evaluate_model(peers, model, proof_status=proof_status, selector_mode="planning")
    fp16_required = float(fp16_route["required_free_gb"])
    quantized_required = round(fp16_required / QWEN3_MOE_INT8_COMPRESSION_RATIO, 1)
    quantized_route = _memory_route(peers, required_gb=quantized_required)
    quantized_status = _status_for(proof_status, ROUTE_ID)
    fp16_status = _status_for(proof_status, MODEL_ID)
    demo_safe_allowed = _demo_safe(quantized_status)
    verification_status = "passed" if quantized_route["memory_fit"] and not fp16_route["memory_fit"] else "needs_review"

    return {
        "source": "quantized_route_lane.py",
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": verification_status,
        "model_id": MODEL_ID,
        "route_id": ROUTE_ID,
        "quant_type": QUANT_TYPE,
        "quant_scheme": QUANT_SCHEME,
        "source_spike_artifact": SPIKE_ARTIFACT,
        "memory_reduction_source": "quantized_block_spike_random_weight_qwen3_moe_layer",
        "compression_ratio_used": QWEN3_MOE_INT8_COMPRESSION_RATIO,
        "fp16_route": {
            "memory_fit": bool(fp16_route["memory_fit"]),
            "supported": bool(fp16_route["supported"]),
            "placement": fp16_route["placement"],
            "reason": fp16_route["reason"],
            "required_free_gb": round(fp16_required, 1),
            "swarm_free_gb": fp16_route["swarm_free_gb"],
            "solo_hosts": fp16_route["solo_hosts"],
        },
        "quantized_route": quantized_route,
        "fp16_proof_status": fp16_status,
        "quantized_proof_key": ROUTE_ID,
        "quantized_proof_status": quantized_status,
        "next_gate": _next_gate(quantized_status),
        "server_proof_status": "not_run" if quantized_status.get("one_block_server") == "pending" else quantized_status.get("one_block_server"),
        "can_inherit_fp16_proof": False,
        "can_update_fp16_proof_row": False,
        "live_server_proven": False,
        "speedup_proven": False,
        "demo_safe_allowed": demo_safe_allowed,
        "guardrails": list(GUARDRAILS),
        "operator_next_steps": list(OPERATOR_NEXT_STEPS),
    }


def _example_peers(name: str) -> list[dict[str, Any]]:
    if name != "m4pro-int8-30b":
        raise ValueError(f"unknown example: {name}")
    return [
        {
            "hostname": "m4pro",
            "memory": {"total_gb": 48.0, "free_gb": 37.0},
            "accelerator": {"device": "mps", "unified_memory": True, "vram_total_gb": 48.0, "vram_free_gb": 37.0},
        }
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["m4pro-int8-30b"], default="m4pro-int8-30b")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--proof-status", default=str(DEFAULT_PROOF_STATUS))
    parser.add_argument("--out", default=str(DEFAULT_EVIDENCE))
    args = parser.parse_args(argv)

    proof = load_proof_status(args.proof_status)
    payload = build_quantized_qwen30b_lane(
        peers=_example_peers(args.example),
        registry=load_registry(args.registry),
        proof_status=proof,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
