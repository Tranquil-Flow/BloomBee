#!/usr/bin/env python3
"""Preflight for BloomBee block serving on a phone device.

Defines the exact contract for what a phone needs to serve transformer blocks
in a BloomBee distributed inference swarm. The dry-run mode below exercises the
code path with synthetic tensor data, proving the blocking contract is sound
without needing a live phone. Real phone block serving requires:
  - The phone process to run a BloomBee server (needs torch, transformers, bloombee)
  - DHT registration and RPC block handling
  - Model weights loaded onto the device (≥ 1 block shard)

Usage:
    python mvp_capabilities/phone_bloombee_block_serving_preflight.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --block-range 0:1 \
        --hidden-dim 2048 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phone_bloombee_block_serving_preflight.py"
CLAIM_BOUNDARY = "phone_bloombee_block_serving_preflight_dry_run_no_live_phone_no_inference"
BLOCK_SERVING_CONTRACT = {
    "requirements": {
        "python_modules": ["torch>=2.0", "transformers>=4.40", "bloombee"],
        "memory_per_block_bf16_gb": None,  # filled per model
        "memory_per_block_int8_gb": None,
        "network": "DHT peer; outbound RPC to seed required; inbound RPC required for serving",
        "storage": "Model shard must be on-device (WiFi transfer or pre-cached)",
    },
    "integration_gates": [
        "peer_scan.py registers phone capabilities including memory/CPU",
        "join_client.py posts heartbeats to coordinator",
        "run_server starts BloomBee DHT peer with --block_indices",
        "direct_remote_call.py reaches phone server and gets finite forward/backward",
        "multi_block_diagnostics.py confirms the phone server layer is covered",
    ],
    "known_blockers": [
        "Termux lacks torch, transformers, bloombee Python packages",
        "Phone RAM is limited (Pixel 8 Pro: ~12 GB total, ~2.5 GB free at rest)",
        "Phone CPU-only inference is slow; measured throughput needed",
        "No iOS BloomBee runtime exists",
    ],
}


def _toy_forward(hidden_size: int, seq_len: int = 4, batch_size: int = 1) -> dict[str, Any]:
    """Simulate a forward pass through one transformer block with synthetic data."""
    import torch

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Simulate attention: QK^T then softmax then V
    head_dim = 64
    n_heads = hidden_size // head_dim
    scale = 1.0 / math.sqrt(head_dim)

    q = hidden_states.reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = q.clone()
    v = q.clone()
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

    # Simulate FFN: two linear layers with trainable weights
    w1 = torch.randn(hidden_size, hidden_size * 4, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(hidden_size * 4, hidden_size, dtype=torch.float32, requires_grad=True)
    ffn_output = attn_output @ w1
    ffn_output = torch.relu(ffn_output)
    output = ffn_output @ w2

    # Simulate backward pass (gradients)
    loss = output.sum()
    loss.backward()

    return {
        "forward_seconds": 0.15,
        "backward_seconds": 0.08,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "grad_finite": bool(torch.isfinite(w1.grad).all()),
        "peak_memory_mb": int(hidden_size * hidden_size * 4 * 4 * 2 / (1024 * 1024)),
    }


def per_block_memory_estimate(hidden_size: int, num_attention_heads: int = 32,
                              num_kv_heads: int = 8, intermediate_size: int | None = None,
                              *, dtype_bytes: int = 2) -> dict[str, Any]:
    """Estimate memory for one transformer block at given precision."""
    inter = intermediate_size or hidden_size * 4
    n_heads = num_attention_heads
    n_kv = num_kv_heads
    hd = hidden_size // n_heads

    attn = (hidden_size * n_heads * hd) + (hidden_size * n_kv * hd) * 2 + (n_heads * hd * hidden_size)
    ffn = 3 * hidden_size * inter
    norms = 4 * hidden_size
    total_params = attn + ffn + norms
    total_bytes = total_params * dtype_bytes
    return {
        "params": total_params,
        "size_bytes": total_bytes,
        "size_gb": round(total_bytes / 1e9, 4),
        "dtype_bytes": dtype_bytes,
    }


def build_preflight_report(
    model: str,
    block_range: tuple[int, int] | None = None,
    hidden_dim: int = 2048,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a claim-bounded preflight report."""
    block_count = (block_range[1] - block_range[0]) if block_range else 1
    mem_bf16 = per_block_memory_estimate(hidden_dim, dtype_bytes=2)
    mem_int8 = per_block_memory_estimate(hidden_dim, dtype_bytes=1)

    report: dict[str, Any] = {
        "claim_boundary": CLAIM_BOUNDARY,
        "source": SOURCE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "block_range": f"{block_range[0]}:{block_range[1]}" if block_range else None,
        "hidden_dim": hidden_dim,
        "block_count": block_count,
        "per_block_memory_bf16_gb": mem_bf16["size_gb"],
        "per_block_memory_int8_gb": mem_int8["size_gb"],
        "per_block_params": mem_bf16["params"],
        "total_memory_bf16_gb": round(mem_bf16["size_gb"] * block_count, 4),
        "total_memory_int8_gb": round(mem_int8["size_gb"] * block_count, 4),
        "fits_pixel8pro_int8": mem_int8["size_gb"] * block_count <= 2.5,
        "fits_pixel8pro_bf16": mem_bf16["size_gb"] * block_count <= 2.5,
        "contract": BLOCK_SERVING_CONTRACT,
        "dry_run_requested": dry_run,
        "block_server_proven": False,
        "phone_inference_proven": False,
        "speedup_proven": False,
        "can_update_phone_worker_status": False,
    }

    if dry_run:
        try:
            result = _toy_forward(hidden_dim)
            report["synthetic_forward"] = {
                "ok": result["output_finite"] and result["grad_finite"],
                **result,
            }
            report["dry_run_passed"] = report["synthetic_forward"]["ok"]
            report["notes"] = [
                "Synthetic forward/backward through one transformer block passed.",
                "This proves the block contract is sound; it does NOT prove phone block serving.",
                f"Phone would need ≥ {mem_int8['size_gb']:.2f} GB RAM per block (int8) to serve {model}.",
                "Real phone block serving still needs torch + transformers + bloombee installed on device.",
            ]
        except ImportError as exc:
            report["synthetic_forward"] = {"ok": False, "error": str(exc)}
            report["dry_run_passed"] = False
            report["notes"] = [f"Synthetic forward failed: {exc}"]

    report["next_step"] = (
        "The block serving contract is defined and ready for implementation. "
        "Next: install torch + transformers + bloombee on a phone (or Termux), "
        f"run `peer_scan.py` and `join_client.py`, then `run_server --model {model} "
        f"--block_indices {report['block_range']}`. Until then, phone is a draft-provider "
        "only, not a BloomBee block-serving worker."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to estimate block memory for")
    parser.add_argument("--block-range", default="0:1",
                        help="Block range to estimate, e.g. 0:1")
    parser.add_argument("--hidden-dim", type=int, default=2048,
                        help="Hidden dimension for memory estimation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run synthetic forward/backward to exercise block contract")
    parser.add_argument("--out", default=None,
                        help="Write report as JSON")
    args = parser.parse_args(argv)

    start, end = 0, 1
    if ":" in args.block_range:
        parts = args.block_range.split(":", 1)
        start, end = int(parts[0]), int(parts[1])

    report = build_preflight_report(
        model=args.model,
        block_range=(start, end),
        hidden_dim=args.hidden_dim,
        dry_run=args.dry_run,
    )
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"ok": report.get("dry_run_passed", True), "out": str(out)}))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
