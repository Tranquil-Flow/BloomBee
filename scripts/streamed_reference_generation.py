#!/usr/bin/env python3
"""Memory-light streamed reference generation for large BloomBee checkpoints.

The normal parity harness loads a full local Hugging Face reference model before
querying the distributed BloomBee route. That is impossible for 30B fp16 on the
48GB m4pro. This script computes the same forward-loop greedy reference trace by
loading only the outer weights plus one transformer block at a time.

Claim boundary: a streamed reference trace is not distributed or quantized proof
by itself. It becomes useful when paired with a separate distributed route trace
and exact token-ID comparison for the same prompt/checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Import bloombee once so model registrations populate AutoDistributedConfig.
import bloombee  # noqa: F401,E402
from bloombee.server.block_utils import get_model_block  # noqa: E402
from bloombee.server.from_pretrained import (  # noqa: E402
    INDEX_FILES,
    _load_state_dict_from_local_file,
    _load_state_dict_from_repo,
    load_pretrained_block,
)
from bloombee.utils.auto_config import AutoDistributedConfig  # noqa: E402
from bloombee.utils.disk_cache import DEFAULT_CACHE_DIR  # noqa: E402
from bloombee.utils.hf_compat import get_file_from_repo  # noqa: E402

CLAIM_BOUNDARY = "streamed_block_reference_generation_only_no_distributed_or_quantized_parity"
KNOWN_QUANT_SUFFIXES = ("@int8", "@nf4")


def dtype_from_name(name: str) -> torch.dtype:
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
    except KeyError as exc:  # pragma: no cover - argparse path
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def checkpoint_model_for_route(model_id: str, checkpoint_model: str | None = None) -> str:
    """Map a route/proof row id such as ``repo/model@int8`` to the HF checkpoint id."""
    if checkpoint_model:
        return checkpoint_model
    for suffix in KNOWN_QUANT_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _effective_cache_dir(model_id: str, cache_dir: str | None) -> str:
    if cache_dir is not None:
        return cache_dir
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return str(model_path)
    return str(DEFAULT_CACHE_DIR)


def _load_prefixed_local_only(model_id: str, prefix: str, *, cache_dir: str) -> dict[str, torch.Tensor]:
    for index_file in INDEX_FILES:
        path = get_file_from_repo(
            model_id,
            index_file,
            use_auth_token=None,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if path is None:
            continue
        if index_file.endswith(".index.json"):
            index = json.loads(Path(path).read_text(encoding="utf-8"))
            filenames = {
                filename
                for param_name, filename in index["weight_map"].items()
                if param_name.startswith(prefix)
            }
            if not filenames:
                raise RuntimeError(f"prefix {prefix!r} not found in local index {path}")
            state: dict[str, torch.Tensor] = {}
            for filename in filenames:
                shard_path = get_file_from_repo(
                    model_id,
                    filename,
                    use_auth_token=None,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                if shard_path is None:
                    raise FileNotFoundError(f"missing local shard {filename} for prefix {prefix!r}")
                shard = _load_state_dict_from_local_file(shard_path, block_prefix=prefix)
                state.update(
                    {
                        param_name[len(prefix) :]: tensor
                        for param_name, tensor in shard.items()
                        if param_name.startswith(prefix)
                    }
                )
            return state
        state = _load_state_dict_from_local_file(path, block_prefix=prefix)
        stripped = {
            param_name[len(prefix) :]: tensor
            for param_name, tensor in state.items()
            if param_name.startswith(prefix)
        }
        if stripped:
            return stripped
    raise FileNotFoundError(f"no local weights found for prefix {prefix!r} in {model_id!r}")


def _load_prefixed(
    model_id: str,
    prefix: str,
    *,
    cache_dir: str | None,
    local_files_only: bool,
) -> dict[str, torch.Tensor]:
    effective_cache_dir = _effective_cache_dir(model_id, cache_dir)
    if local_files_only:
        return _load_prefixed_local_only(model_id, prefix, cache_dir=effective_cache_dir)
    state = _load_state_dict_from_repo(
        model_id,
        prefix,
        token=None,
        cache_dir=effective_cache_dir,
        max_disk_space=None,
    )
    if local_files_only:
        # _load_state_dict_from_repo already prefers local files and only writes
        # when missing. Keep this explicit flag in the public API/evidence even
        # though the current private helper does not expose it directly.
        pass
    return state


def load_outer_weights(
    model_id: str,
    *,
    config: Any,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load only embedding, final norm, and lm_head tensors.

    These weights are small enough to keep resident while transformer blocks are
    streamed one at a time. If the checkpoint ties word embeddings and omits
    ``lm_head.weight``, the embedding weight is reused exactly as HF would.
    """
    embed = _load_prefixed(
        model_id,
        "model.embed_tokens.",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )["weight"].to(device=device, dtype=dtype)
    norm = _load_prefixed(
        model_id,
        "model.norm.",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )["weight"].to(device=device, dtype=dtype)

    try:
        lm_head = _load_prefixed(
            model_id,
            "lm_head.",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )["weight"].to(device=device, dtype=dtype)
    except Exception:
        if not getattr(config, "tie_word_embeddings", False):
            raise
        lm_head = embed

    return {"embed_tokens.weight": embed, "model.norm.weight": norm, "lm_head.weight": lm_head}


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps).to(hidden_states.dtype)
    return hidden_states * weight


def _empty_device_cache(device: str | torch.device) -> None:
    dev = torch.device(device)
    if dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_streamed_block(
    model_id: str,
    layer_idx: int,
    *,
    config: Any,
    dtype: torch.dtype,
    cache_dir: str | None,
    local_files_only: bool,
) -> torch.nn.Module:
    if local_files_only:
        block = get_model_block(config, None, None, None, "/tmp", layer_idx=layer_idx)
        prefix = f"{config.block_prefix}.{layer_idx}."
        state = _load_prefixed(
            model_id,
            prefix,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        block.load_state_dict(state, strict=False)
        return block.to(dtype=dtype)
    return load_pretrained_block(
        model_id,
        layer_idx,
        env=None,
        policy=None,
        weight_home=None,
        path="/tmp",
        config=config,
        torch_dtype=dtype,
        cache_dir=_effective_cache_dir(model_id, cache_dir),
    )


@torch.inference_mode()
def streamed_forward_logits(
    model_id: str,
    input_ids: torch.Tensor,
    *,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    local_files_only: bool = False,
    outer_weights: dict[str, torch.Tensor] | None = None,
    config: Any | None = None,
    record_block_timings: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Compute logits by loading and executing one transformer block at a time."""
    cfg_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cfg_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        cfg_kwargs["local_files_only"] = True
    config = config or AutoDistributedConfig.from_pretrained(model_id, **cfg_kwargs)
    outer_weights = outer_weights or load_outer_weights(
        model_id,
        config=config,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )

    hidden = F.embedding(input_ids.to(device), outer_weights["embed_tokens.weight"])
    block_timings: list[dict[str, Any]] = []
    block_count = int(config.num_hidden_layers)
    for layer_idx in range(block_count):
        t0 = time.time()
        block = _load_streamed_block(
            model_id,
            layer_idx,
            config=config,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).eval()
        block.to(device)
        output = block(hidden, use_cache=False)
        hidden = output[0] if isinstance(output, tuple) else output
        seconds = time.time() - t0
        if record_block_timings:
            block_timings.append({"layer_idx": layer_idx, "seconds": seconds})
        del block, output
        gc.collect()
        _empty_device_cache(device)

    hidden = rms_norm(hidden, outer_weights["model.norm.weight"], float(getattr(config, "rms_norm_eps", 1e-6)))
    logits = hidden[:, -1, :].matmul(outer_weights["lm_head.weight"].t()).detach().float().cpu()
    return logits, block_timings


def _ids(ids: torch.Tensor) -> list[int]:
    return [int(x) for x in ids.detach().cpu()[0].tolist()]


def topk(logits: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=k, dim=-1)
    return [{"token_id": int(i), "logit": float(v)} for i, v in zip(indices[0].tolist(), values[0].tolist())]


@torch.inference_mode()
def streamed_greedy_generate_ids(
    model_id: str,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    local_files_only: bool = False,
    record_block_timings: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Forward-loop greedy decode using streamed block references."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    cfg_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cfg_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        cfg_kwargs["local_files_only"] = True
    config = AutoDistributedConfig.from_pretrained(model_id, **cfg_kwargs)
    outer_weights = load_outer_weights(
        model_id,
        config=config,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )

    output_ids = input_ids.detach().cpu().clone().to(torch.long)
    steps: list[dict[str, Any]] = []
    for step in range(max_new_tokens):
        t0 = time.time()
        logits, block_timings = streamed_forward_logits(
            model_id,
            output_ids,
            cache_dir=cache_dir,
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            outer_weights=outer_weights,
            config=config,
            record_block_timings=record_block_timings,
        )
        next_id = logits.argmax(dim=-1, keepdim=True).to(torch.long)
        output_ids = torch.cat([output_ids, next_id.cpu()], dim=1)
        steps.append(
            {
                "step": step,
                "seconds": time.time() - t0,
                "next_token_id": int(next_id[0, 0]),
                "top5": topk(logits, 5),
                "block_count": int(config.num_hidden_layers),
                "block_timings": block_timings,
            }
        )
    return output_ids, steps


def build_reference_trace(
    *,
    model_id: str,
    checkpoint_model: str | None,
    prompt: str,
    max_new_tokens: int,
    cache_dir: str | None,
    dtype: torch.dtype,
    device: str,
    local_files_only: bool,
    record_block_timings: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint_model_for_route(model_id, checkpoint_model)
    tokenizer_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, use_fast=False, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.long)

    t0 = time.time()
    reference_ids, steps = streamed_greedy_generate_ids(
        checkpoint,
        input_ids,
        max_new_tokens=max_new_tokens,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        record_block_timings=record_block_timings,
    )
    seconds = time.time() - t0
    return {
        "ok": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "model": model_id,
        "checkpoint_model": checkpoint,
        "reference_mode": "streamed-blocks",
        "loads_full_model": False,
        "loads_one_block_at_a_time": True,
        "device": device,
        "dtype": str(dtype),
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "input_ids": _ids(input_ids),
        "reference_ids": _ids(reference_ids),
        "reference_text": tokenizer.decode(reference_ids[0].tolist(), skip_special_tokens=True),
        "reference_steps": steps,
        "reference_seconds": seconds,
        "block_count": steps[0]["block_count"] if steps else None,
        "can_update_full_generation": False,
        "distributed_compared": False,
        "quantized_compared": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Proof/route model id. May include @int8/@nf4 suffix.")
    parser.add_argument("--checkpoint-model", default=None, help="HF checkpoint id/path to load if --model is a route id.")
    parser.add_argument("--prompt", default="The moon is")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", type=dtype_from_name, default=torch.float16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--record-block-timings", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    trace = build_reference_trace(
        model_id=args.model,
        checkpoint_model=args.checkpoint_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        record_block_timings=args.record_block_timings,
    )
    text = json.dumps(trace, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
