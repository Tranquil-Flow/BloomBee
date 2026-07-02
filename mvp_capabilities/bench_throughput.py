#!/usr/bin/env python3
"""
bench_throughput.py — real tok/s benchmark for a single HuggingFace
causal LM, intended to populate the "what can peer X actually do?" half
of the BloomBee MVP routing table.

Measures two distinct phases:
  prefill — the single forward pass that consumes the prompt (memory-
            bound, FLOP-bound depending on length).
  decode  — the autoregressive step, one token at a time (memory-bandwidth
            bound; the canonical "tok/s" number in marketing decks).

Outputs a single JSON line on stdout plus a few human-readable status
lines on stderr. No BloomBee import; uses transformers + torch directly.

Defaults pick Qwen2.5-0.5B-Instruct because it downloads fast (~1 GB)
and finishes a 64-token decode in well under a minute on Apple Silicon.

Usage:
    python bench_throughput.py
    python bench_throughput.py --model Qwen/Qwen2.5-3B-Instruct --max-new-tokens 128
    python bench_throughput.py --device cuda --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def pick_dtype(requested: str, device: str) -> str:
    """Resolve the effective dtype string for the chosen device. We
    print what we actually used (not just what was asked for) so the
    output JSON is unambiguous."""
    if requested != "auto":
        return requested
    if device == "mps":
        return "bf16"
    if device == "cuda":
        return "fp16"
    return "fp32"


def resolve_device(requested: str) -> tuple[str, Any]:
    """Returns (device_str, torch.device). Always one of 'mps', 'cuda',
    'cpu'. 'auto' inspects torch + the platform."""
    import torch  # type: ignore

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps", torch.device("mps")
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
        return "cpu", torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("WARN: MPS not available, falling back to CPU", file=sys.stderr)
            return "cpu", torch.device("cpu")
        return "mps", torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU", file=sys.stderr)
            return "cpu", torch.device("cpu")
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


def dtype_from_str(name: str):
    import torch  # type: ignore

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def measure_memory_mb() -> dict[str, float]:
    """Best-effort resident memory probe; not a peak measurement, just a
    snapshot at call time. Used to print a 'where am I now' line."""
    import psutil  # type: ignore

    vm = psutil.virtual_memory()
    return {
        "rss_mb": round(vm.used / (1024**2), 1),
        "available_mb": round(vm.available / (1024**2), 1),
    }


def mem_snapshot(device: str) -> float:
    """Return current device memory usage in GB. MPS/CUDA each have
    their own counter; CPU is reported via psutil RSS delta vs process
    start. Returns 0.0 if nothing measurable is available."""
    try:
        import torch  # type: ignore

        if device == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / (1024**3), 3)
        if device == "mps" and torch.backends.mps.is_available():
            # torch.mps has no public "current allocated" counter pre-2.1;
            # the recommended_max_memory is a static host-side hint, not
            # actual usage. Fall back to psutil for an honest reading.
            import psutil  # type: ignore

            rss = psutil.Process(os.getpid()).memory_info().rss
            return round(rss / (1024**3), 3)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device_str, torch_device = resolve_device(args.device)
    dtype_str = pick_dtype(args.dtype, device_str)
    torch_dtype = dtype_from_str(dtype_str)

    print(f"[bench] device={device_str} dtype={dtype_str} model={args.model}", file=sys.stderr)
    mem_before = measure_memory_mb()
    print(f"[bench] mem_before rss_mb={mem_before['rss_mb']}", file=sys.stderr)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    t_load = time.perf_counter() - t0
    print(f"[bench] load_seconds={t_load:.2f}", file=sys.stderr)

    model = model.to(torch_device)
    model.eval()

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(
        f"[bench] loaded: {n_params_b:.3f}B params, dtype={dtype_str}, device={device_str}",
        file=sys.stderr,
    )
    print(f"[bench] mem_after_load {measure_memory_mb()}", file=sys.stderr)

    # ----- prompt construction -----
    prefill_tokens = args.prefill
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab = max(tokenizer.vocab_size, 1)
    # half the prefill is real-looking input, half is garbage so the warmup
    # really exercises something; the benchmark itself runs over the full
    # prefill sequence either way.
    import_ids = [
        list(range(1, min(vocab, prefill_tokens // 2 + 1))),
        list(range(vocab - 1, vocab - prefill_tokens // 2 - 1, -1)),
    ]
    flat = [tid for sub in import_ids for tid in sub][:prefill_tokens]
    if len(flat) < prefill_tokens:
        flat += [pad_id] * (prefill_tokens - len(flat))
    input_ids = torch.tensor([flat], dtype=torch.long, device=torch_device)
    attention_mask = torch.ones_like(input_ids)

    # ----- warmup (one forward, throwaway) -----
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass
    elif device_str == "cuda":
        torch.cuda.synchronize()

    # ----- prefill phase -----
    prefill_t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask)
    if device_str == "mps":
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass
    elif device_str == "cuda":
        torch.cuda.synchronize()
    prefill_dt = time.perf_counter() - prefill_t0
    prefill_tok_s = prefill_tokens / prefill_dt if prefill_dt > 0 else 0.0

    print(
        f"[bench] prefill {prefill_tokens} tokens in {prefill_dt:.3f}s -> {prefill_tok_s:.1f} tok/s",
        file=sys.stderr,
    )

    # ----- decode phase (autoregressive generate) -----
    mem_after_prefill = mem_snapshot(device_str)
    decode_t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            pad_token_id=pad_id,
        )
    if device_str == "mps":
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass
    elif device_str == "cuda":
        torch.cuda.synchronize()
    decode_dt = time.perf_counter() - decode_t0
    n_new = gen.shape[-1] - input_ids.shape[-1]
    decode_tok_s = n_new / decode_dt if decode_dt > 0 else 0.0
    print(
        f"[bench] decode {n_new} tokens in {decode_dt:.3f}s -> {decode_tok_s:.1f} tok/s",
        file=sys.stderr,
    )

    peak_mem_gb = mem_snapshot(device_str)
    # If the device counter under-reported, fall back to the highest of
    # (after_load, after_prefill, current).
    if peak_mem_gb == 0.0:
        peak_mem_gb = max(mem_after_prefill, measure_memory_mb()["rss_mb"] / 1024)

    context_len = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None
    )

    result = {
        "model": args.model,
        "device": device_str,
        "dtype": dtype_str,
        "prefill_tokens": prefill_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prefill_seconds": round(prefill_dt, 4),
        "prefill_tok_per_s": round(prefill_tok_s, 2),
        "decode_seconds": round(decode_dt, 4),
        "decode_tok_per_s": round(decode_tok_s, 2),
        "peak_mem_gb": round(peak_mem_gb, 3),
        "context_len": context_len,
        "params_b": round(n_params_b, 3),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-model tok/s benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill", type=int, default=128)
    args = parser.parse_args()

    try:
        result = run_bench(args)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # The single JSON line — the only stdout the consumer needs.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())