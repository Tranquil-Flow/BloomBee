#!/usr/bin/env python3
"""Scan model config compatibility against BloomBee MVP proof gates.

This tool is intentionally lightweight: it can inspect a local `config.json` or
model directory without downloading weights. It separates three different claims:

- HF config can be read (`prescan=passed`),
- BloomBee has a known block-wrapper family for that `model_type`,
- the model has later proof gates such as one-block or full generation.

It does not import BloomBee runtime modules; the MVP capability tools stay usable
as standalone planning utilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROOF_STATUS = Path(__file__).with_name("PROOF_STATUS.yaml")

SUPPORTED_FAMILIES: dict[str, dict[str, str]] = {
    "bloom": {"bloombee_family": "bloom", "block_prefix": "transformer.h"},
    "falcon": {"bloombee_family": "falcon", "block_prefix": "transformer.h"},
    "gemma4": {"bloombee_family": "gemma4", "block_prefix": "model.layers"},
    "llama": {"bloombee_family": "llama", "block_prefix": "model.layers"},
    "mixtral": {"bloombee_family": "mixtral", "block_prefix": "model.layers"},
    "qwen3": {"bloombee_family": "qwen3", "block_prefix": "model.layers"},
    "qwen3_moe": {"bloombee_family": "qwen3_moe", "block_prefix": "model.layers"},
}

PROOF_KEYS = (
    "prescan",
    "one_block_server",
    "multi_block",
    "full_generation",
    "cache_generation",
    "multi_request_load",
)

# Quantized proof rows are keyed "<model_id>@<quant_type>" (plain id == fp16)
# and NEVER inherit fp16 gates. This is the single home of the demo-safe
# policy; route_picker and proof_ladder import it rather than redefining it.
QUANT_TYPES = ("int8", "nf4")
DEMO_SAFE_GATES = ("full_generation", "cache_generation", "multi_request_load")
TOKEN_PARITY_KEY = "token_parity"


def split_route_id(route_id: str) -> tuple[str, str | None]:
    """Split ``model_id@quant_type`` into ``(model_id, quant_type)``.

    Unknown ``@suffix`` values are NOT treated as quant markers: the id stays
    whole, so a typo'd suffix becomes an unknown (all-pending) row instead of
    silently aliasing the fp16 row.
    """
    base, sep, suffix = str(route_id).rpartition("@")
    if sep and suffix in QUANT_TYPES:
        return base, suffix
    return str(route_id), None


def is_demo_safe(status: dict[str, str], *, quant_type: str | None = None) -> bool:
    """demo_safe requires generation + cached-generation parity + load proof.

    Quantized rows additionally require ``token_parity: exact`` — an exact
    greedy token-ID match against the fp16 reference on the demo prompt set.
    ``diverged`` or absent parity caps the row below demo_safe regardless of
    gate status (no "close enough" promotions).
    """
    if not all(status.get(gate) == "passed" for gate in DEMO_SAFE_GATES):
        return False
    if quant_type is not None and status.get(TOKEN_PARITY_KEY) != "exact":
        return False
    return True


def _read_config(source: str | Path, *, local_files_only: bool = True) -> dict[str, Any]:
    """Read a config dict from config.json, a model dir, or a HF model id."""
    source_path = Path(str(source)).expanduser()
    if source_path.is_dir():
        config_path = source_path / "config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))
    if source_path.is_file():
        return json.loads(source_path.read_text(encoding="utf-8"))

    try:
        from transformers import AutoConfig
    except Exception as exc:  # pragma: no cover - only for missing optional dep
        raise RuntimeError(
            f"{source!s} is not a local config path and transformers is unavailable"
        ) from exc

    cfg = AutoConfig.from_pretrained(str(source), local_files_only=local_files_only)
    return cfg.to_dict()


def load_proof_status(path: str | Path = DEFAULT_PROOF_STATUS) -> dict[str, dict[str, str]]:
    proof_path = Path(path)
    if not proof_path.exists():
        return {}
    payload = yaml.safe_load(proof_path.read_text(encoding="utf-8")) or {}
    models = payload.get("models") or {}
    return {
        str(model_id): {str(key): str(value) for key, value in (status or {}).items()}
        for model_id, status in models.items()
    }


def _default_proof_status() -> dict[str, str]:
    return {key: "pending" for key in PROOF_KEYS}


def _claim_level(
    *,
    architecture_supported: bool,
    proof_status: dict[str, str],
    runtime_blocked_reasons: list[str] | None = None,
    quant_type: str | None = None,
) -> str:
    if not architecture_supported or runtime_blocked_reasons:
        return "blocked"
    if is_demo_safe(proof_status, quant_type=quant_type):
        return "demo_safe"
    return "experimental"


def _nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else {}


def _model_body_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return config section that describes transformer-block dimensions.

    Newer multimodal MoEs such as Qwen3.5-35B-A3B and MiniMax-M3 put the
    language tower under ``text_config`` while the top-level config describes the
    multimodal wrapper. BloomBee still needs block-level facts from the language
    tower even when the wrapper is unsupported.
    """
    text_config = _nested_text_config(config)
    if text_config and not config.get("num_hidden_layers"):
        return text_config
    return config


def _quantization_method(config: dict[str, Any]) -> str | None:
    qcfg = config.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None
    method = qcfg.get("quant_method") or qcfg.get("quant_algo")
    return str(method) if method else "unknown"


def _uses_sparse_attention(config: dict[str, Any]) -> bool:
    sparse_cfg = config.get("sparse_attention_config")
    return bool(isinstance(sparse_cfg, dict) and sparse_cfg.get("use_sparse_attention"))


def scan_model_config(
    source: str | Path,
    *,
    model_id: str | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    config = _read_config(source, local_files_only=local_files_only)
    body_config = _model_body_config(config)
    text_config = _nested_text_config(config)
    hf_model_type = str(config.get("model_type") or "unknown")
    text_model_type = text_config.get("model_type")
    support = SUPPORTED_FAMILIES.get(hf_model_type)
    architecture_supported = support is not None
    quantization_method = _quantization_method(config)
    uses_sparse_attention = _uses_sparse_attention(body_config)

    merged_proof = _default_proof_status()
    merged_proof["prescan"] = "passed"
    if model_id and proof_status and model_id in proof_status:
        merged_proof.update(proof_status[model_id])
    merged_proof["prescan"] = "passed"

    blocked_reasons: list[str] = []
    if not architecture_supported:
        blocked_reasons.append(f"No BloomBee block wrapper registered for model_type={hf_model_type}")
    if uses_sparse_attention:
        blocked_reasons.append(
            "Model text tower uses sparse_attention_config; current BloomBee block wrappers "
            "do not implement this attention state/kernel contract"
        )
    if quantization_method:
        blocked_reasons.append(
            "Quantized HF checkpoint declares quantization_config="
            f"{quantization_method}; current BloomBee HF-block loader instantiates "
            "fp16/bf16 PyTorch blocks and does not build GPTQ/AWQ/FP8/NVFP4/"
            "MXFP quantized Linear kernels"
        )

    result: dict[str, Any] = {
        "model_id": model_id or str(source),
        "hf_model_type": hf_model_type,
        "hf_text_model_type": str(text_model_type) if text_model_type else None,
        "architecture_supported": architecture_supported,
        "bloombee_family": support["bloombee_family"] if support else None,
        "block_prefix": support["block_prefix"] if support else None,
        "num_layers": body_config.get("num_hidden_layers") or body_config.get("n_layer"),
        "hidden_size": body_config.get("hidden_size") or body_config.get("n_embd"),
        "num_attention_heads": body_config.get("num_attention_heads") or body_config.get("n_head"),
        "num_key_value_heads": body_config.get("num_key_value_heads") or body_config.get("n_head_kv"),
        "num_experts": body_config.get("num_experts") or body_config.get("num_local_experts") or body_config.get("n_routed_experts"),
        "experts_per_token": body_config.get("num_experts_per_tok") or body_config.get("num_experts_per_token"),
        "architectures": config.get("architectures") or [],
        "text_architectures": body_config.get("architectures") or [],
        "max_position_embeddings": body_config.get("max_position_embeddings"),
        "uses_sparse_attention": uses_sparse_attention,
        "quantization_method": quantization_method,
        "quantization_supported": False if quantization_method else None,
        "proof_status": merged_proof,
        "claim_level": _claim_level(
            architecture_supported=architecture_supported,
            proof_status=merged_proof,
            runtime_blocked_reasons=blocked_reasons,
            quant_type=split_route_id(model_id)[1] if model_id else None,
        ),
        "blocked_reasons": blocked_reasons,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HF model id, model directory, or config.json path")
    parser.add_argument("--model-id", help="Logical model id to use for proof-status lookup")
    parser.add_argument(
        "--proof-status",
        default=DEFAULT_PROOF_STATUS,
        help="YAML file containing per-model proof status (default: PROOF_STATUS.yaml)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers AutoConfig to download remote config when source is a HF id",
    )
    args = parser.parse_args(argv)

    proof = load_proof_status(args.proof_status)
    result = scan_model_config(
        args.source,
        model_id=args.model_id or args.source,
        proof_status=proof,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
