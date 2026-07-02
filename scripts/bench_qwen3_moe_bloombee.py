"""End-to-end inference bench through BloomBee's Qwen3-MoE wrapper.

Loads the real :class:`DistributedQwen3MoeForCausalLM` from HuggingFace on the
M4 Pro, runs a prefill + a decode step. We deliberately bypass real
``DHT`` / ``RemoteSequential`` peer orchestration — here all 48 layers live
on a single process to keep the bench measurable on the laptop while
exercising the full BloomBee wrapper surface.

The single-process path is what BloomBee uses during smoke tests
(``test_full_model.py``-style); it forces the wrapper, the GQA cache layout,
the LMHead, and the MoE routing at native precision.

Usage::

    python scripts/bench_qwen3_moe_bloombee.py --model Qwen/Qwen3-30B-A3B \
        --device mps --dtype bf16 --max-new-tokens 16 --prefill 32

Exit code 0 on success. ``peak_mem_gb`` reported even if generation fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_dtype(value: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prompt", default="In one sentence, what is distributed inference?")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-host-fallback",
        action="store_true",
        help="Allow running even with insufficient free RAM (will OOM).",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt")

    from bloombee.models.qwen3_moe.model import DistributedQwen3MoeForCausalLM

    print(f"[bench] loading DistributedQwen3MoeForCausalLM from {args.model}", flush=True)
    t0 = time.perf_counter()
    model = DistributedQwen3MoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_parse_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    load_seconds = time.perf_counter() - t0
    print(f"[bench] loaded in {load_seconds:.2f}s", flush=True)

    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t0
    n = out.shape[1] - inputs["input_ids"].shape[1]
    generation = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    peak_mem_gb = torch.mps.current_allocated_memory() / 1024**3 if args.device == "mps" else None

    payload = {
        "model": args.model,
        "wrapper": "bloombee.DistributedQwen3MoeForCausalLM",
        "device": args.device,
        "dtype": args.dtype,
        "prefill_tokens": int(inputs["input_ids"].shape[1]),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": int(n),
        "seconds": round(dt, 3),
        "tokens_per_s_decode": round(n / dt, 3) if dt > 0 else None,
        "peak_mem_gb": peak_mem_gb,
        "load_seconds": round(load_seconds, 3),
        "generation": generation,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
