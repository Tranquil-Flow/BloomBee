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


def _claim_level(*, architecture_supported: bool, proof_status: dict[str, str]) -> str:
    if not architecture_supported:
        return "blocked"
    if proof_status.get("full_generation") == "passed":
        return "demo_safe"
    return "experimental"


def scan_model_config(
    source: str | Path,
    *,
    model_id: str | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    config = _read_config(source, local_files_only=local_files_only)
    hf_model_type = str(config.get("model_type") or "unknown")
    support = SUPPORTED_FAMILIES.get(hf_model_type)
    architecture_supported = support is not None

    merged_proof = _default_proof_status()
    merged_proof["prescan"] = "passed"
    if model_id and proof_status and model_id in proof_status:
        merged_proof.update(proof_status[model_id])
    merged_proof["prescan"] = "passed"

    blocked_reasons: list[str] = []
    if not architecture_supported:
        blocked_reasons.append(f"No BloomBee block wrapper registered for model_type={hf_model_type}")

    result: dict[str, Any] = {
        "model_id": model_id or str(source),
        "hf_model_type": hf_model_type,
        "architecture_supported": architecture_supported,
        "bloombee_family": support["bloombee_family"] if support else None,
        "block_prefix": support["block_prefix"] if support else None,
        "num_layers": config.get("num_hidden_layers") or config.get("n_layer"),
        "hidden_size": config.get("hidden_size") or config.get("n_embd"),
        "num_attention_heads": config.get("num_attention_heads") or config.get("n_head"),
        "num_key_value_heads": config.get("num_key_value_heads") or config.get("n_head_kv"),
        "num_experts": config.get("num_experts") or config.get("n_routed_experts"),
        "experts_per_token": config.get("num_experts_per_tok") or config.get("num_experts_per_token"),
        "architectures": config.get("architectures") or [],
        "proof_status": merged_proof,
        "claim_level": _claim_level(
            architecture_supported=architecture_supported,
            proof_status=merged_proof,
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
