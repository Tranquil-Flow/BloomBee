#!/usr/bin/env python3
"""Post-MVP research spike: HF-block weight-only quantization feasibility.

BloomBee's QuantType INT8/NF4 currently only feeds FlexGen group-wise
compression, which lives inside FlexGen's TorchTensor infrastructure and is
never applied to standard HF blocks (qwen3, qwen3_moe, falcon, mixtral). This
spike measures whether optimum-quanto weight-only quantization of a bare HF
decoder layer is a viable serving substrate for those families on Apple
Silicon (MPS) and CPU.

It does NOT start servers, join swarms, or claim distributed proof. Claim
boundary: post_mvp_quantized_block_spike_no_serving_proof.

Measured per (block, backend-dtype, device):
  - parity vs the fp16 reference forward on the same device
    (max abs diff, mean abs diff, cosine similarity over flattened outputs)
  - backward-to-input gradient finiteness (BloomBee load gate requires it)
  - weight bytes before/after quantization
  - rough single-forward wall clock (not a benchmark)

Blocks:
  - TinyLlama layer 0 with REAL weights from the local HF cache
  - Qwen3-30B-A3B qwen3_moe layer with real config dims + seeded random
    weights (weights random, dims real, so memory math is real)

Usage:
  .venv/bin/python scripts/quantized_block_spike.py \
      --output mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-<UTC>.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

CLAIM_BOUNDARY = "post_mvp_quantized_block_spike_no_serving_proof"
SEED = 20260704


def _weight_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _quanto_weight_bytes(module: torch.nn.Module) -> int:
    """quanto freeze() replaces Linear weights with packed QTensor subclasses;
    element_size() on the wrapper reflects the packed dtype, but scales live in
    separate tensors. Walk state_dict for an honest byte count."""
    total = 0
    for value in module.state_dict().values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _forward(block: torch.nn.Module, hidden: torch.Tensor, position_embeddings, attention_mask=None):
    out = block(
        hidden,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().flatten()
    cand = candidate.detach().float().flatten()
    max_abs = float((ref - cand).abs().max())
    mean_abs = float((ref - cand).abs().mean())
    cos = float(torch.nn.functional.cosine_similarity(ref, cand, dim=0))
    ref_scale = float(ref.abs().mean())
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine_similarity": cos,
        "reference_mean_abs": ref_scale,
        "relative_mean_abs_diff": (mean_abs / ref_scale) if ref_scale > 0 else None,
    }


def _run_candidate(
    *,
    label: str,
    block_factory,
    rotary_factory,
    hidden_size: int,
    device: str,
    weights_dtype_name: str,
    seq_len: int = 16,
) -> dict[str, Any]:
    """Build fp16 reference + quantized copy on `device`, compare forwards."""
    from optimum.quanto import freeze, qint4, qint8, quantize

    custom_moe = weights_dtype_name == "moe_int8_experts+qint8_attn"
    qdtype = qint8 if custom_moe else {"qint8": qint8, "qint4": qint4}[weights_dtype_name]
    result: dict[str, Any] = {
        "candidate": label,
        "backend": "bloombee.moe_expert_quant + optimum-quanto" if custom_moe else "optimum-quanto",
        "weights_dtype": weights_dtype_name,
        "device": device,
        "seq_len": seq_len,
    }
    try:
        torch.manual_seed(SEED)
        reference_block = block_factory().to(torch.float16).to(device).eval()
        rotary = rotary_factory().to(device)

        torch.manual_seed(SEED + 1)
        hidden = (torch.randn(1, seq_len, hidden_size, dtype=torch.float16) * 0.5).to(device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = rotary(hidden, position_ids)

        with torch.no_grad():
            reference_out = _forward(reference_block, hidden, position_embeddings)
        fp16_bytes = _weight_bytes(reference_block)

        quantized_block = copy.deepcopy(reference_block)
        if custom_moe:
            from bloombee.utils.moe_expert_quant import quantize_qwen3_moe_block_experts

            result["moe_expert_swap"] = quantize_qwen3_moe_block_experts(quantized_block)
        quantize(quantized_block, weights=qdtype)
        freeze(quantized_block)
        quantized_bytes = _quanto_weight_bytes(quantized_block)

        with torch.no_grad():
            t0 = time.time()
            quantized_out = _forward(quantized_block, hidden, position_embeddings)
            quant_forward_s = time.time() - t0
        with torch.no_grad():
            t0 = time.time()
            _forward(reference_block, hidden, position_embeddings)
            ref_forward_s = time.time() - t0

        grad_input = hidden.clone().detach().requires_grad_(True)
        grad_embeddings = rotary(grad_input, position_ids)
        grad_out = _forward(quantized_block, grad_input, grad_embeddings)
        grad_out.float().sum().backward()
        grad_ok = grad_input.grad is not None and bool(torch.isfinite(grad_input.grad).all())

        result.update(
            {
                "status": "passed",
                "parity_vs_fp16_same_device": _parity(reference_out, quantized_out),
                "outputs_finite": bool(torch.isfinite(quantized_out).all()),
                "backward_to_input_grad_finite": grad_ok,
                "fp16_weight_bytes": fp16_bytes,
                "quantized_weight_bytes": quantized_bytes,
                "compression_ratio": round(fp16_bytes / quantized_bytes, 3) if quantized_bytes else None,
                "rough_forward_seconds": {
                    "fp16": round(ref_forward_s, 4),
                    "quantized": round(quant_forward_s, 4),
                },
            }
        )
        del reference_block, quantized_block, reference_out, quantized_out
    except Exception as exc:  # noqa: BLE001 - spike must record failures, not crash
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        gc.collect()
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return result


def _tinyllama_factories():
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config = AutoConfig.from_pretrained(model_id)
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(model_id, "model.safetensors", local_files_only=True)
    state = load_file(weights_path)
    prefix = "model.layers.0."
    layer_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    del state

    def block_factory():
        block = LlamaDecoderLayer(config, layer_idx=0)
        missing, unexpected = block.load_state_dict(layer_state, strict=False)
        if [m for m in missing if "rotary" not in m] or unexpected:
            raise RuntimeError(f"TinyLlama layer-0 load mismatch: missing={missing} unexpected={unexpected}")
        return block

    def rotary_factory():
        return LlamaRotaryEmbedding(config)

    return block_factory, rotary_factory, config.hidden_size, "real_weights_local_cache"


def _qwen3_moe_factories():
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", local_files_only=True)

    def block_factory():
        torch.manual_seed(SEED)
        block = Qwen3MoeDecoderLayer(config, layer_idx=0)
        # Standalone HF layer construction can leave torch.empty allocations
        # uninitialized (NaN); init explicitly so parity math is meaningful.
        with torch.no_grad():
            for param in block.parameters():
                torch.nn.init.normal_(param, std=0.02)
        return block

    def rotary_factory():
        return Qwen3MoeRotaryEmbedding(config)

    return block_factory, rotary_factory, config.hidden_size, "real_config_random_weights"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Write JSON evidence artifact here")
    parser.add_argument("--skip-moe", action="store_true", help="Skip the ~1.2GB qwen3_moe block (low-memory hosts)")
    parser.add_argument("--devices", default=None, help="Comma-separated device list override, e.g. cpu,mps")
    args = parser.parse_args()

    devices = args.devices.split(",") if args.devices else ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    candidates = []
    tl_block, tl_rotary, tl_hidden, tl_provenance = _tinyllama_factories()
    candidates.append(("tinyllama_layer0", tl_block, tl_rotary, tl_hidden, tl_provenance))
    if not args.skip_moe:
        moe_block, moe_rotary, moe_hidden, moe_provenance = _qwen3_moe_factories()
        candidates.append(("qwen3_30b_a3b_moe_layer", moe_block, moe_rotary, moe_hidden, moe_provenance))

    runs = []
    for label, block_factory, rotary_factory, hidden_size, provenance in candidates:
        modes = ("qint8", "qint4")
        if "moe" in label:
            modes = ("qint8", "moe_int8_experts+qint8_attn", "qint4")
        for device in devices:
            for weights_dtype in modes:
                run = _run_candidate(
                    label=label,
                    block_factory=block_factory,
                    rotary_factory=rotary_factory,
                    hidden_size=hidden_size,
                    device=device,
                    weights_dtype_name=weights_dtype,
                )
                run["weights_provenance"] = provenance
                runs.append(run)
                print(json.dumps(run, indent=2))

    import importlib.metadata as md

    report = {
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "research_question": (
            "Can optimum-quanto weight-only quantization serve as the HF-block "
            "quantization path BloomBee lacks for qwen3/qwen3_moe on MPS/CPU?"
        ),
        "seed": SEED,
        "versions": {
            "torch": torch.__version__,
            "transformers": md.version("transformers"),
            "optimum-quanto": md.version("optimum-quanto"),
        },
        "mps_available": torch.backends.mps.is_available(),
        "runs": runs,
        "do_not_claim": [
            "distributed serving proof",
            "generation parity proof",
            "route/demo promotion",
            "mvp-core status change",
        ],
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[spike] wrote {out_path}")

    failed = [r for r in runs if r["status"] != "passed"]
    print(f"[spike] {len(runs) - len(failed)}/{len(runs)} candidate runs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
