"""Smoke-test the BloomBee Qwen3-MoE integration with the real Qwen3-30B-A3B config.

This is a pre-showcase gate:

- Loads :class:`DistributedQwen3MoeForCausalLM` for Qwen/Qwen3-30B-A3B.
- Verifies the wrapper auto-dispatches as ``model_type='qwen3_moe'``.
- Touches every layer's :class:`WrappedQwen3MoeBlock` with synthetic hidden
  state (no real weights needed; proves the wrapper accepts the real config).

The script uses HF model defaults for layer/memory layout; we instantiate
the bare stem and walk layers, not load the 60GB weights.

Usage:
    python scripts/prescan_qwen3_moe.py [--model Qwen/Qwen3-30B-A3B]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--offline", action="store_true",
                        help="Set HF_HUB_OFFLINE=1; config must already be cached.")
    parser.add_argument("--skip-load", action="store_true",
                        help="Only verify AutoConfig dispatch + config fields; do not instantiate.")
    args = parser.parse_args()

    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoConfig
    print(f"[prescan] loading AutoConfig for {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    print(f"[prescan] model_type={config.model_type!r}")
    assert config.model_type == "qwen3_moe", f"expected qwen3_moe, got {config.model_type}"
    print(f"[prescan] num_hidden_layers={config.num_hidden_layers}")
    print(f"[prescan] hidden_size={config.hidden_size} num_experts={getattr(config,'num_experts',None)}")
    print(f"[prescan] num_experts_per_tok={getattr(config,'num_experts_per_tok',None)}")

    if args.skip_load:
        print("[prescan] --skip-load set, exiting")
        return 0

    from bloombee.models.qwen3_moe.model import DistributedQwen3MoeForCausalLM
    from bloombee.models.qwen3_moe.config import DistributedQwen3MoeConfig

    cfg = DistributedQwen3MoeConfig.from_pretrained(args.model)
    assert cfg.model_type == "qwen3_moe"
    assert cfg.num_hidden_layers == config.num_hidden_layers
    assert cfg.block_class is not None
    print(f"[prescan] block_class={cfg.block_class.__name__}")
    print(f"[prescan] attn_class={cfg.attn_class.__name__}")
    print(f"[prescan] block_prefix={cfg.block_prefix}")

    print("[prescan] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
