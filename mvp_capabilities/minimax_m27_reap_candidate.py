#!/usr/bin/env python3
"""MiniMax-M2.7-REAP-139B-A10B candidate preflight.

Claim boundary: ``minimax_m27_reap_candidate_preflight_no_bloombee_or_live_inference_claim``.

This is an OPERATOR harness — it produces a structured preflight report about
whether ``dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B`` can be smoked on a small
fleet of peers using the **external runtime** (llama.cpp GGUF), NOT whether
BloomBee can serve it natively. It:

- States up-front that this model has NO native BloomBee block wrapper.
- Estimates total memory footprint at multiple GGUF quant levels.
- Picks the smallest quant that fits on the best peer plus margin.
- Emits operator commands that use llama.cpp directly (not BloomBee).
- Marks route/demo eligibility as False (no live BloomBee parity claim).
- Never over-claims a runtime success — that requires actually running it
  and observing tokens, which is the gate this report hands off to.

Why this distinction matters: MiniMax-M3 (full size, 428B / 23B-active) is so
large that even aggressive 2-bit quant lands at >100 GB and won't fit our
combined 64 GB across M4 machines. REAP-139B (a community prune) at
i1-IQ2_XXS estimates ~36.8 GB, which can plausibly fit on a 48 GB M4 Pro
with reasonable free-memory margin. That makes it the right frontier-path
candidate for a smoke, with the explicit caveat that quant at this level is
likely to be partial-quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"
TOTAL_PARAMS_B = 139.0
ACTIVE_PARAMS_B = 10.0  # MoE active per token

# Conservative quant ladder — bytes-per-param approximation then a real-world
# bump for headers, token embeddings, and runtime overhead. Numbers come from
# published llama.cpp / Unsloth conventions for REAP-style MoE merges; treat
# as preflight estimates, NOT measured footprints.
# i1 = importance-majored, IQ = I-quant; XXS = 2.0-ish bpw. We omit IQ1
# variants entirely: their quality is below a useful minimum for any
# evaluation we can do, so recommending them would mislead an operator.
QUANT_TABLE = [
    {"name": "i1-IQ2_XXS", "approx_bytes_per_param": 0.230, "quality_note": "low_quality_but_first_m4pro_plausible_smoke"},
    {"name": "i1-IQ2_XS",  "approx_bytes_per_param": 0.255, "quality_note": "low_quality"},
    {"name": "Q4_K_M",    "approx_bytes_per_param": 0.560, "quality_note": "decent_quality_but_too_large_for_m4_pro_solo"},
]  # fmt: skip

# Real-world overhead multiplier (tokenizer, metadata, KV cache reservation
# for a short context, runtime structs). 1.15 keeps us conservative.
OVERHEAD_MULTIPLIER = 1.15

# Required free-RAM headroom as a fraction. i1-IQ2_XXS is borderline; we want
# the OS to keep ~10% room for file cache + peak bursts in llama.cpp. Lower
# than a 15% margin because once we're fitting a 36 GB model on a 48 GB host,
# the engineering goal IS the smoke-test, not comfortable headroom.
REQUIRED_FREE_HEADROOM = 0.10

# HF model_type reported by this repo. Drives the BloomBee-native
# architecture_support check.
REAP_HF_MODEL_TYPE = "minimax_m2"  # community prune derives from minimax_m2 / REAP


def _approx_size_gb(params_b: float, bytes_per_param: float) -> float:
    raw_gb = params_b * bytes_per_param
    return round(raw_gb * OVERHEAD_MULTIPLIER, 1)


def _best_peer_by_free_memory(peers: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [p for p in peers if isinstance(p, dict) and isinstance(p.get("memory"), dict)]
    if not valid:
        return None
    # pick the one with the most free memory
    return max(valid, key=lambda p: p["memory"].get("free_gb", 0.0))


def _select_quant(peers: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    """Walk quant ladder from smallest up; return the first that fits on the
    best peer. Returns (selected_quant_or_None, blocked_reasons)."""
    best = _best_peer_by_free_memory(peers)
    if best is None:
        return None, ["no_valid_peer_with_memory_metrics_provided"]

    free_gb = best["memory"].get("free_gb", 0.0)
    required_gb = free_gb * (1.0 - REQUIRED_FREE_HEADROOM)

    blocked: list[str] = []
    # Walk from smallest up so we get the best-fit if multiple quants fit.
    for q in sorted(QUANT_TABLE, key=lambda q: q["approx_bytes_per_param"]):
        size_gb = _approx_size_gb(TOTAL_PARAMS_B, q["approx_bytes_per_param"])
        if size_gb <= required_gb:
            return (
                {
                    "name": q["name"],
                    "size_gb": size_gb,
                    "estimated_gb_with_overhead": size_gb,
                    "required_free_gb": round(required_gb, 1),
                    "quality_note": q["quality_note"],
                },
                blocked,
            )
        blocked.append(f"quant_{q['name']}_size_{size_gb}gb_exceeds_required_{round(required_gb, 1)}gb")

    blocked.append("no_peer_has_free_memory_for_i1-IQ2_XXS_plus_margin")
    return None, blocked


def build_minimax_m27_reap_candidate_report(
    *,
    peers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the preflight report.

    Args:
        peers: list of peer dicts, each with structure
               ``{"hostname": str, "memory": {"total_gb": float, "free_gb": float}}``.
               Defaults to a sensible laptop spec if not supplied.
    """
    peers = peers or [
        {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 42.0}},
        {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
    ]

    selected_quant, blocked = _select_quant(peers)
    best = _best_peer_by_free_memory(peers)
    attemptable = selected_quant is not None

    # Operator commands — point at the best peer and llama.cpp directly.
    operator_commands: list[str] = []
    if attemptable and best is not None:
        operator_commands.extend([
            f"# On {best['hostname']} — install llama.cpp release (supports REAP/MoE GGUF as of 2026-Q1):",
            "brew install llama.cpp  # or: git clone https://github.com/ggml-org/llama.cpp && cmake -B build && cmake --build build --config Release",
            "# Download the {selected} GGUF from a community mirror (Hugging Face):",
            "huggingface-cli download --local-dir ./MiniMax-M2.7-REAP-iq2 --include '*IQ2_XXS*' {model}".format(
                selected=selected_quant["name"], model=MODEL_ID,
            ),
            "# Serve via OpenAI-compatible llama-server; do NOT route through BloomBee:",
            "./llama-server -m ./MiniMax-M2.7-REAP-iq2/{safetensors_index}.gguf -ngl 99 --port 8080 --host 127.0.0.1",
            "# Smoke a single curl to confirm token streaming works:",
            "curl -s http://127.0.0.1:8080/v1/chat/completions -d '{\"model\":\"any\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}]}'",
        ])

    return {
        "claim_boundary": "minimax_m27_reap_candidate_preflight_no_bloombee_or_live_inference_claim",
        "model_id": MODEL_ID,
        "params_b": TOTAL_PARAMS_B,
        "active_params_b": ACTIVE_PARAMS_B,
        "architecture_supported": False,
        "native_bloombee_support_proven": False,
        "route_picker_eligible": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "live_run_attempted": False,
        "bloombee_blocked_reasons": [
            f"No BloomBee block wrapper registered for model_type={REAP_HF_MODEL_TYPE}",
            "MiniMax-M2.7-REAP community prune — BloomBee block family covers upstream MiniMax but not pruned/REAPed variants",
        ],
        "gguf_external_runtime": {
            "framework": "llama.cpp",
            "attemptable_on_best_peer": attemptable,
            "best_peer": (
                {
                    "hostname": best["hostname"],
                    "total_gb": best["memory"]["total_gb"],
                    "free_gb": best["memory"]["free_gb"],
                }
                if best is not None
                else None
            ),
            "selected_quant": selected_quant,
            "all_quants_attempted": [
                {
                    "name": q["name"],
                    "size_gb": _approx_size_gb(TOTAL_PARAMS_B, q["approx_bytes_per_param"]),
                    "quality_note": q["quality_note"],
                }
                for q in QUANT_TABLE
            ],
            "overhead_multiplier": OVERHEAD_MULTIPLIER,
            "required_free_headroom": REQUIRED_FREE_HEADROOM,
        },
        "blocked_reasons": blocked,
        "operator_commands": operator_commands,
        "suggested_proof_gate": "minimax_m2_7_reap_external_runtime",
    }


def _parse_peer(spec: str) -> dict[str, Any]:
    """Parse 'hostname:total_gb:free_gb' string format."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"peer spec must be hostname:total_gb:free_gb, got: {spec!r}")
    hostname, total_gb, free_gb = parts[0], float(parts[1]), float(parts[2])
    return {"hostname": hostname, "memory": {"total_gb": total_gb, "free_gb": free_gb}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer spec 'hostname:total_gb:free_gb' (repeatable). Default: synthetic m4pro+local-m4.",
    )
    parser.add_argument("--out", help="Optional path to write JSON report to.")
    args = parser.parse_args(argv)

    if args.peer:
        peers = [_parse_peer(p) for p in args.peer]
    else:
        peers = None

    report = build_minimax_m27_reap_candidate_report(peers=peers)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
