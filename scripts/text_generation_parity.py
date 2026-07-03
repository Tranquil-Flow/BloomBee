"""Compare local full-model text generation against BloomBee distributed generation.

This is an operator-facing proof harness for the MVP: same prompt, same
checkpoint, greedy decoding, local full Hugging Face model vs BloomBee
client-side embeddings/LM head + remote transformer blocks.

Default mode is ``forward-loop``: at each decode step, recompute the full
prefix through ``model.forward()`` and pick argmax. This deliberately avoids
BloomBee's cached ``rpc_inference`` path, so the parity proof uses the same
RemoteSequential forward route verified by ``scripts/direct_remote_call.py``.
Use ``--mode generate-api`` only when specifically testing the cached
``RemoteGenerationMixin.generate`` path.

Example:
  python scripts/text_generation_parity.py \
    --server-maddr '/ip4/192.168.178.37/tcp/31337/p2p/...' \
    --server-maddr '/ip4/192.168.178.37/tcp/31338/p2p/...' \
    --server-maddr '/ip4/192.168.178.37/tcp/31339/p2p/...' \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt 'The capital of France is' \
    --max-new-tokens 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

# Make the src-layout importable when run from a checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def _decode(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def _ids(ids: torch.Tensor) -> list[int]:
    return [int(x) for x in ids.detach().cpu()[0].tolist()]


def parse_server_placements(
    placement_args: list[str] | None,
    server_maddrs: list[str],
) -> list[dict[str, Any]]:
    """Parse host/layer metadata for dashboard evidence.

    Format is one entry per server, in the same order as --server-maddr:
    ``--server-placement m4pro-seed=0:8``.
    """
    if not placement_args:
        return []
    if len(placement_args) != len(server_maddrs):
        raise ValueError("provide one --server-placement per --server-maddr")
    placements: list[dict[str, Any]] = []
    for raw, maddr in zip(placement_args, server_maddrs):
        if "=" not in raw:
            raise ValueError("server placement must use host=start:end")
        host, layer_range = raw.split("=", 1)
        if not host.strip() or ":" not in layer_range:
            raise ValueError("server placement must use host=start:end")
        start_text, end_text = layer_range.split(":", 1)
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise ValueError("server placement layer range must use integer start:end") from exc
        if start < 0 or end <= start:
            raise ValueError("server placement layer range must use increasing start:end")
        placements.append({"host": host.strip(), "layers": [start, end], "server_maddr": maddr})
    return placements


def _prepare_sandbox_hivemind_runtime() -> None:
    """Apply client-only hivemind fallbacks needed in sandboxed shells.

    `direct_remote_call.py` uses the same guards. They are harmless on normal
    terminals and make this parity harness usable from Hermes local shells where
    sitecustomize disabled MPFuture's global lock and torch_shm_manager cannot
    be executed.
    """
    from hivemind.utils.mpfuture import MPFuture

    MPFuture.reset_backend()
    try:
        torch.empty([4], dtype=torch.uint8).share_memory_()
    except (RuntimeError, OSError, PermissionError):
        from hivemind.utils.mpfuture import SharedBytes
        import os as _os
        import threading as _threading

        _heap_buffer: dict[str, Any] = {"pid": None, "buffer": None, "index": 0}
        _heap_lock = _threading.Lock()

        def _next_heap_byte():
            with _heap_lock:
                if (
                    _heap_buffer["pid"] != _os.getpid()
                    or _heap_buffer["buffer"] is None
                    or _heap_buffer["index"] >= len(_heap_buffer["buffer"])
                ):
                    size = int(_os.environ.get("HIVEMIND_SHM_BUFFER_SIZE", 16))
                    _heap_buffer["pid"] = _os.getpid()
                    _heap_buffer["buffer"] = torch.zeros(size, dtype=torch.uint8)
                    _heap_buffer["index"] = 0
                _heap_buffer["index"] += 1
                return _heap_buffer["buffer"][_heap_buffer["index"] - 1]

        SharedBytes.next = staticmethod(_next_heap_byte)
        print("[parity] (sandbox detected: SharedBytes heap-buffer fallback active)")


def _topk(logits: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=k, dim=-1)
    return [
        {"token_id": int(i), "logit": float(v)}
        for i, v in zip(indices[0].tolist(), values[0].tolist())
    ]


@torch.inference_mode()
def _greedy_forward_loop(
    model,
    input_ids_cpu: torch.Tensor,
    *,
    max_new_tokens: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Greedy decode by recomputing the full prefix through forward()."""
    output_cpu = input_ids_cpu.detach().cpu().clone()
    steps: list[dict[str, Any]] = []
    for step in range(max_new_tokens):
        t0 = time.time()
        logits = model(output_cpu.to(device)).logits[:, -1, :].detach().float().cpu()
        next_id = logits.argmax(dim=-1, keepdim=True).to(torch.long)
        output_cpu = torch.cat([output_cpu, next_id.cpu()], dim=1)
        steps.append(
            {
                "step": step,
                "seconds": time.time() - t0,
                "next_token_id": int(next_id[0, 0]),
                "top5": _topk(logits, 5),
            }
        )
    return output_cpu, steps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--server-maddr", action="append", required=True,
                   help="BloomBee peer multiaddr. Pass once per peer.")
    p.add_argument("--server-placement", action="append", default=None,
                   help="Dashboard metadata in host=start:end form; pass once per --server-maddr in the same order.")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--reference-device", default="mps",
                   help="Device for the full local reference model: mps/cpu/cuda.")
    p.add_argument("--reference-dtype", type=_dtype, default=torch.float16)
    p.add_argument("--distributed-dtype", type=_dtype, default=torch.float16)
    p.add_argument(
        "--mode",
        choices=("forward-loop", "generate-api"),
        default="forward-loop",
        help=("forward-loop recomputes full prefix through model.forward at each step; "
              "generate-api exercises cached RemoteGenerationMixin.generate"),
    )
    p.add_argument("--out", default=None, help="Optional JSON evidence path")
    p.add_argument("--no-server-to-server", action="store_true",
                   help="Disable direct server-to-server rpc_push; client orchestrates every stage")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="Return 0 even when generated token IDs differ")
    args = p.parse_args()
    try:
        server_placements = parse_server_placements(args.server_placement, args.server_maddr)
    except ValueError as exc:
        p.error(str(exc))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _prepare_sandbox_hivemind_runtime()
    torch.manual_seed(0)

    print(f"[parity] model={args.model}")
    print(f"[parity] prompt={args.prompt!r}")
    print(f"[parity] max_new_tokens={args.max_new_tokens}")
    print(f"[parity] mode={args.mode}")
    print(f"[parity] server_to_server={not args.no_server_to_server}")
    print(f"[parity] bootstrap peers ({len(args.server_maddr)}):")
    for addr in args.server_maddr:
        print(f"[parity]   {addr}")
    if server_placements:
        print("[parity] server placements:")
        for placement in server_placements:
            start, end = placement["layers"]
            print(f"[parity]   {placement['host']} layers {start}:{end}")

    print("[parity] loading tokenizer...")
    t0 = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[parity]   ... {time.time() - t0:.1f}s")

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids_cpu = encoded["input_ids"]
    print(f"[parity] input_ids={_ids(input_ids_cpu)}")

    print("[parity] loading full local HF reference model...")
    t0 = time.time()
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.reference_dtype,
        low_cpu_mem_usage=True,
    ).eval()
    if args.reference_device != "cpu":
        ref_model = ref_model.to(args.reference_device)
    print(f"[parity]   ... {time.time() - t0:.1f}s on {args.reference_device} dtype={args.reference_dtype}")

    print("[parity] loading BloomBee distributed model shell...")
    t0 = time.time()
    from bloombee import AutoDistributedModelForCausalLM

    dist_model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.server_maddr,
        torch_dtype=args.distributed_dtype,
        use_server_to_server=not args.no_server_to_server,
    ).eval()
    print(f"[parity]   ... {time.time() - t0:.1f}s dtype={args.distributed_dtype}")

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    evidence: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "server_maddrs": args.server_maddr,
        "server_placements": server_placements,
        "reference_device": args.reference_device,
        "reference_dtype": str(args.reference_dtype),
        "distributed_dtype": str(args.distributed_dtype),
        "mode": args.mode,
        "server_to_server": not args.no_server_to_server,
        "input_ids": _ids(input_ids_cpu),
    }

    print("[parity] local reference forward/decode...")
    t0 = time.time()
    with torch.inference_mode():
        ref_input = input_ids_cpu.to(args.reference_device)
        ref_logits = ref_model(ref_input).logits[:, -1, :].detach().float().cpu()
        if args.mode == "generate-api":
            ref_generated = ref_model.generate(ref_input, **generation_kwargs).detach().cpu()
            ref_steps = []
        else:
            ref_generated, ref_steps = _greedy_forward_loop(
                ref_model,
                input_ids_cpu,
                max_new_tokens=args.max_new_tokens,
                device=args.reference_device,
            )
    ref_seconds = time.time() - t0
    evidence.update(
        reference_seconds=ref_seconds,
        reference_next_token_id=int(ref_logits.argmax(dim=-1)[0]),
        reference_top5=_topk(ref_logits, 5),
        reference_steps=ref_steps,
        reference_ids=_ids(ref_generated),
        reference_text=_decode(tokenizer, ref_generated),
    )
    print(f"[parity]   ... {ref_seconds:.2f}s")
    print(f"[parity] reference_ids={evidence['reference_ids']}")
    print(f"[parity] reference_text={evidence['reference_text']!r}")

    print("[parity] distributed forward/decode...")
    t0 = time.time()
    with torch.inference_mode():
        dist_logits = dist_model(input_ids_cpu).logits[:, -1, :].detach().float().cpu()
        if args.mode == "generate-api":
            dist_generated = dist_model.generate(input_ids_cpu, **generation_kwargs).detach().cpu()
            dist_steps = []
        else:
            dist_generated, dist_steps = _greedy_forward_loop(
                dist_model,
                input_ids_cpu,
                max_new_tokens=args.max_new_tokens,
                device="cpu",
            )
    dist_seconds = time.time() - t0
    evidence.update(
        distributed_seconds=dist_seconds,
        distributed_next_token_id=int(dist_logits.argmax(dim=-1)[0]),
        distributed_top5=_topk(dist_logits, 5),
        distributed_steps=dist_steps,
        distributed_ids=_ids(dist_generated),
        distributed_text=_decode(tokenizer, dist_generated),
    )
    print(f"[parity]   ... {dist_seconds:.2f}s")
    print(f"[parity] distributed_ids={evidence['distributed_ids']}")
    print(f"[parity] distributed_text={evidence['distributed_text']!r}")

    logits_delta = (ref_logits - dist_logits).abs()
    evidence.update(
        next_token_match=evidence["reference_next_token_id"] == evidence["distributed_next_token_id"],
        generated_ids_match=evidence["reference_ids"] == evidence["distributed_ids"],
        generated_text_match=evidence["reference_text"] == evidence["distributed_text"],
        logits_max_abs_diff=float(logits_delta.max().item()),
        logits_mean_abs_diff=float(logits_delta.mean().item()),
    )
    evidence["ok"] = bool(evidence["generated_ids_match"])

    print("[parity] RESULT:", json.dumps(evidence, sort_keys=True))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"[parity] wrote {out}")

    try:
        dist_model.transformer.h.sequence_manager.shutdown()
    except Exception as exc:  # noqa: BLE001
        print(f"[parity] (non-fatal) distributed shutdown: {exc}")

    return 0 if (evidence["ok"] or args.allow_mismatch) else 1


if __name__ == "__main__":
    raise SystemExit(main())
