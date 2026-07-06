#!/usr/bin/env python3
"""Map Qwen36A exact text config to BloomBee native state-cache descriptors.

This is still not runtime proof. It verifies the exact Qwen3.6-35B-A3B text
configuration can instantiate the repo's qwen3_5_moe_text config and that the
server backend descriptor seam distinguishes linear-attention raw state from
full-attention KV slabs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # type: ignore[import-not-found]

from bloombee.models.qwen3_5_moe.config import DistributedQwen3_5MoeTextConfig
from bloombee.server.backend import TransformerBackend

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CLAIM_BOUNDARY = "post_mvp_qwen36a_state_cache_mapping_no_server_proof_no_demo_promotion"
CONFIG_SCAN_ARTIFACT = "mvp_capabilities/distributed_evidence/qwen36a/qwen36a-config-scan-20260706.json"


def _read_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_text_config(config_path: str | Path) -> DistributedQwen3_5MoeTextConfig:
    outer = _read_config(config_path)
    text_config = outer.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen36A config snapshot must include text_config")
    cfg = DistributedQwen3_5MoeTextConfig(**text_config)
    # Keep descriptor tests deterministic and independent of backend defaults.
    cfg._attn_implementation = "eager"
    return cfg


def _descriptor_summary(descriptor: object) -> dict[str, Any]:
    shape = getattr(descriptor, "shape", None)
    return {
        "type": type(descriptor).__name__,
        "kind": getattr(descriptor, "kind", None),
        "shape": list(shape) if shape is not None else None,
    }


def _backend_for(cfg: DistributedQwen3_5MoeTextConfig, *, block_index: int) -> TransformerBackend:
    backend = TransformerBackend.__new__(TransformerBackend)
    backend.config = cfg
    backend.block_index = block_index
    backend.dtype = torch.float16
    backend.module = SimpleNamespace(devices=[torch.device("cpu")])
    backend.shard_num_heads = [cfg.num_attention_heads]
    return backend


def _descriptor_contract(
    cfg: DistributedQwen3_5MoeTextConfig,
    *,
    block_index: int,
    batch_size: int = 2,
    max_length: int = 16,
) -> dict[str, Any]:
    backend = _backend_for(cfg, block_index=block_index)
    descriptors = backend.get_inference_cache_descriptors(batch_size=batch_size, max_length=max_length)
    return {
        "block_index": block_index,
        "layer_type": cfg.layer_types[block_index],
        "batch_size": batch_size,
        "max_length": max_length,
        "descriptors": [_descriptor_summary(descriptor) for descriptor in descriptors],
    }


def build_qwen36a_state_cache_mapping_report(
    config: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    cfg = _load_text_config(config)
    layer_types = list(cfg.layer_types)
    linear_index = layer_types.index("linear_attention")
    full_index = layer_types.index("full_attention")

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model_id,
        "config_scan_artifact": CONFIG_SCAN_ARTIFACT,
        "native_bloombee_distributed_path_target": True,
        "text_config_loaded": True,
        "text_model_type": cfg.model_type,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "layer_type_counts": {
            "linear_attention": layer_types.count("linear_attention"),
            "full_attention": layer_types.count("full_attention"),
        },
        "mapping_status": "passed_descriptor_contract_no_live_server",
        "linear_attention_descriptor_contract": _descriptor_contract(cfg, block_index=linear_index),
        "full_attention_descriptor_contract": _descriptor_contract(cfg, block_index=full_index),
        "wrapper_code_written_for_exact_model": False,
        "one_block_server_proven": False,
        "multi_block_proven": False,
        "full_generation_proven": False,
        "cache_generation_proven": False,
        "multi_request_load_proven": False,
        "can_update_route_status": False,
        "can_update_demo_status": False,
        "can_update_mvp_status": False,
        "can_update_proof_status": False,
        "proof_status_update": {},
        "blocked_reasons": [
            "exact Qwen36A one-block server proof has not been run",
            "exact Qwen36A multi-block/full/cache/load gates have not been run",
        ],
        "recommended_next_step": "run_exact_qwen36a_one_block_server_proof_on_suitable_memory",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)

    report = build_qwen36a_state_cache_mapping_report(args.config, model_id=args.model_id)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
