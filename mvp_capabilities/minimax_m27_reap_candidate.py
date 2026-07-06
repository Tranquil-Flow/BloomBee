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
GGUF_REPO_ID = "mradermacher/m51Lab-MiniMax-M2.7-REAP-139B-A10B-i1-GGUF"
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

M3_MODEL_ID = "MiniMaxAI/MiniMax-M3"
M3_GGUF_REPO_ID = "unsloth/MiniMax-M3-GGUF"
M3_REAP_JANG_REPO_ID = "JANGQ-AI/MiniMax-M3-REAP32-Coder"
M3_TOTAL_PARAMS_B = 428.0
M3_ACTIVE_PARAMS_B = 23.0
M3_MINIMUM_KNOWN_GGUF_GB = 128.0
M3_BF16_WEIGHT_GB = 855.0
M3_REAP_JANG_REPORTED_TARGET_GB = 128.0
M3_CONTEXT_TOKENS = 1_048_576
M3_REAP_JANG_FORMAT = "vMLX JANG affine-mixed AWQ + REAP"


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
            "# Download the selected GGUF from the i1/imatrix community mirror (Hugging Face):",
            "huggingface-cli download --local-dir ./MiniMax-M2.7-REAP-i1 --include '*{selected}*' {repo}".format(
                selected=selected_quant["name"], repo=GGUF_REPO_ID,
            ),
            "# Serve via OpenAI-compatible llama-server; do NOT route through BloomBee:",
            "./llama-server -m ./MiniMax-M2.7-REAP-i1/<downloaded-{selected}.gguf> -ngl 99 --ctx-size 4096 --port 8080 --host 127.0.0.1".format(
                selected=selected_quant["name"],
            ),
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


def _parse_runtime(spec: str) -> tuple[str, dict[str, bool]]:
    """Parse 'hostname:key=0:key=1' runtime inventory strings."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"runtime spec must be hostname:key=0:key=1, got: {spec!r}")
    hostname = parts[0]
    flags: dict[str, bool] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ValueError(f"runtime flag must be key=0/1, got: {item!r}")
        key, raw_value = item.split("=", 1)
        value = raw_value.strip().lower()
        if value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(f"runtime flag value must be boolean-like, got: {item!r}")
        flags[key] = value in {"1", "true", "yes"}
    return hostname, flags


def _memory_totals(peers: list[dict[str, Any]]) -> tuple[float, float, float, float]:
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
    return round(total, 1), round(free, 1), round(largest_total, 1), round(largest_free, 1)


def _runtime_present(runtime_inventory: dict[str, dict[str, bool]], key: str) -> bool:
    return any(bool(flags.get(key)) for flags in runtime_inventory.values())


def build_minimax_reap_family_comparison_report(
    *,
    peers: list[dict[str, Any]] | None = None,
    runtime_inventory: dict[str, dict[str, bool]] | None = None,
) -> dict[str, Any]:
    """Compare current MiniMax M2.7 REAP and M3-family local options.

    This report is intentionally a preflight/comparison only. It does not claim
    any live inference, BloomBee native serving, or benchmark result.
    """
    peers = peers or [
        {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 42.0}},
        {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
    ]
    runtime_inventory = runtime_inventory or {}
    m27 = build_minimax_m27_reap_candidate_report(peers=peers)
    combined_total, combined_free, largest_total, largest_free = _memory_totals(peers)

    m3_full_blocked = [
        "No BloomBee block wrapper registered for model_type=minimax_m3",
        "MiniMax-M3 GGUF path is text-only and does not preserve MiniMax Sparse Attention / native multimodality claims",
        "MiniMax-M3 llama.cpp support is preliminary and requires a fresh PR/release-capable build",
    ]
    if M3_MINIMUM_KNOWN_GGUF_GB > largest_total:
        m3_full_blocked.append(
            f"m3_minimum_gguf_{M3_MINIMUM_KNOWN_GGUF_GB}gb_exceeds_largest_single_host_{largest_total}gb"
        )
    if M3_MINIMUM_KNOWN_GGUF_GB > combined_total:
        m3_full_blocked.append(
            f"m3_minimum_gguf_{M3_MINIMUM_KNOWN_GGUF_GB}gb_exceeds_combined_nominal_{combined_total}gb"
        )
    if M3_MINIMUM_KNOWN_GGUF_GB > largest_free:
        m3_full_blocked.append(
            f"m3_minimum_gguf_{M3_MINIMUM_KNOWN_GGUF_GB}gb_exceeds_current_best_free_{largest_free}gb"
        )

    any_vmlx = _runtime_present(runtime_inventory, "vmlx")
    m3_reap_blocked = [
        "JANG REAP variants require vMLX-specific loader/runtime; generic transformers, vLLM, and MLX loaders are not accepted",
        "No BloomBee block wrapper registered for MiniMax-M3 REAP/JANG variants",
    ]
    if not any_vmlx:
        m3_reap_blocked.append("vmlx_not_installed_on_any_peer")
    if M3_REAP_JANG_REPORTED_TARGET_GB > combined_total:
        m3_reap_blocked.append(
            f"m3_reap_jang_reported_target_{M3_REAP_JANG_REPORTED_TARGET_GB}gb_exceeds_combined_nominal_{combined_total}gb"
        )

    best_peer = m27["gguf_external_runtime"].get("best_peer") or {}
    best_peer_name = best_peer.get("hostname")
    best_runtime = runtime_inventory.get(str(best_peer_name), {}) if best_peer_name else {}
    m27_memory_attemptable = bool(m27["gguf_external_runtime"].get("attemptable_on_best_peer"))
    m27_runtime_available = bool(best_runtime.get("llama_cpp")) or not runtime_inventory
    if m27_memory_attemptable and m27_runtime_available:
        preferred = "minimax_m2_7_reap_139b_a10b_external_llamacpp"
    elif m27_memory_attemptable:
        preferred = "minimax_m2_7_reap_memory_fits_but_llamacpp_missing"
    else:
        preferred = "none_current_memory_too_low"

    return {
        "claim_boundary": "minimax_reap_family_comparison_no_live_inference_claim",
        "source": "mvp_capabilities.minimax_m27_reap_candidate",
        "peers": peers,
        "runtime_inventory": runtime_inventory,
        "combined_nominal_memory_gb": combined_total,
        "combined_free_memory_gb": combined_free,
        "largest_single_host_total_gb": largest_total,
        "largest_single_host_free_gb": largest_free,
        "can_pool_m4_and_m4pro_memory_for_external_runtime": False,
        "shared_limitations": [
            "external runtimes need one host memory pool; local M4 + M4 Pro RAM is not additive for llama.cpp/vMLX",
            "BloomBee can distribute only registered block families; MiniMax M2/M3 wrappers are not registered here",
            "any live route/demo promotion still needs real token-generation evidence and verifier output",
        ],
        "decision": {
            "preferred_current_local_target": preferred,
            "keep_m3_as_option": True,
            "m3_is_likely_more_powerful": True,
            "m3_is_easier_to_run_now": False,
            "prefer_m3_if": [
                "a MiniMax-M3 REAP/GGUF/JANG artifact is verified on a single available host with enough memory",
                "vMLX JANG runtime is installed and smoke-tested for JANG REAP variants, or a released llama.cpp build supports the chosen GGUF",
                "a 128GB+ Apple Silicon host or equivalent GPU host is available for M3-family local runtime",
            ],
            "m2_7_reap_can_run_using_both_m4_and_m4pro": False,
            "m2_7_reap_can_run_on_m4pro_alone_if_freed": m27_memory_attemptable,
        },
        "models": {
            "m2_7_reap_139b_a10b": m27,
            "m3_full": {
                "model_id": M3_MODEL_ID,
                "gguf_repo_id": M3_GGUF_REPO_ID,
                "params_b": M3_TOTAL_PARAMS_B,
                "active_params_b": M3_ACTIVE_PARAMS_B,
                "context_tokens": M3_CONTEXT_TOKENS,
                "minimum_known_gguf_gb": M3_MINIMUM_KNOWN_GGUF_GB,
                "bf16_weight_gb": M3_BF16_WEIGHT_GB,
                "likely_more_powerful_than_m2_7_reap": True,
                "easier_to_run_on_current_macs": False,
                "route_picker_eligible": False,
                "native_bloombee_support_proven": False,
                "live_run_attempted": False,
                "blocked_reasons": m3_full_blocked,
            },
            "m3_reap_jang": {
                "repo_id": M3_REAP_JANG_REPO_ID,
                "format": M3_REAP_JANG_FORMAT,
                "reported_target_host_gb": M3_REAP_JANG_REPORTED_TARGET_GB,
                "requires_vmlx": True,
                "generic_runtime_supported": False,
                "vmlx_installed_on_any_peer": any_vmlx,
                "route_picker_eligible": False,
                "native_bloombee_support_proven": False,
                "live_run_attempted": False,
                "blocked_reasons": m3_reap_blocked,
            },
        },
        "source_urls": [
            "https://huggingface.co/mradermacher/m51Lab-MiniMax-M2.7-REAP-139B-A10B-i1-GGUF",
            "https://huggingface.co/unsloth/MiniMax-M3-GGUF",
            "https://huggingface.co/JANGQ-AI/MiniMax-M3-REAP32-Coder",
            "https://github.com/MiniMax-AI/MiniMax-M3",
        ],
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "live_run_attempted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer spec 'hostname:total_gb:free_gb' (repeatable). Default: synthetic m4pro+local-m4.",
    )
    parser.add_argument(
        "--runtime",
        action="append",
        default=[],
        help="Runtime spec 'hostname:llama_cpp=0/1:vmlx=0/1' (repeatable; comparison mode only).",
    )
    parser.add_argument("--compare-family", action="store_true", help="Emit M2.7 REAP vs M3-family comparison report.")
    parser.add_argument("--out", help="Optional path to write JSON report to.")
    args = parser.parse_args(argv)

    if args.peer:
        peers = [_parse_peer(p) for p in args.peer]
    else:
        peers = None

    if args.compare_family:
        runtime_inventory = dict(_parse_runtime(item) for item in args.runtime)
        report = build_minimax_reap_family_comparison_report(
            peers=peers,
            runtime_inventory=runtime_inventory,
        )
    else:
        report = build_minimax_m27_reap_candidate_report(peers=peers)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
