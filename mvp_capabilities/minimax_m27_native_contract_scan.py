#!/usr/bin/env python3
"""MiniMax M2.7 REAP native-contract scan.

This is a claim-boundary scanner for the exact REAP config. It does not create a
BloomBee wrapper, run inference, or update proof status. It records what a native
`minimax_m2` BloomBee block wrapper must support before M2.7 REAP can move from
planning to proof.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"
CLAIM_BOUNDARY = "minimax_m27_reap_native_contract_scan_no_wrapper_or_live_inference"


def _first_architecture(config: dict[str, Any]) -> str | None:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return None


def _count_attention_types(config: dict[str, Any]) -> dict[str, int]:
    raw = config.get("attn_type_list")
    if not isinstance(raw, list):
        return {}
    return {str(key): value for key, value in sorted(Counter(raw).items(), key=lambda item: str(item[0]))}


def _wrapper_package_present() -> bool:
    return (PROJECT_ROOT / "src/bloombee/models/minimax_m2").exists()


def _has_sparse_attention_flag(config: dict[str, Any], attn_counts: dict[str, int]) -> bool:
    sparse_keys = (
        "supports_sparse_attention",
        "sparse_attention",
        "use_sparse_attention",
    )
    if any(bool(config.get(key)) for key in sparse_keys):
        return True
    # The exact REAP config observed so far has attn_type_list all 1 and no
    # sparse/sliding-window flag. Keep this conservative: non-1 attention types
    # require human/model-source review instead of being silently treated as full
    # attention.
    return any(key != "1" for key in attn_counts)


def build_minimax_m27_native_contract_report(
    config: dict[str, Any],
    *,
    model_id: str = MODEL_ID,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    attn_counts = _count_attention_types(config)
    num_layers = int(config.get("num_hidden_layers") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    num_heads = int(config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or 0)
    head_dim = int(config.get("head_dim") or 0)
    num_local_experts = int(config.get("num_local_experts") or 0)
    top_k = int(config.get("num_experts_per_tok") or 0)
    use_mtp = bool(config.get("use_mtp"))
    use_qk_norm = bool(config.get("use_qk_norm"))
    use_routing_bias = bool(config.get("use_routing_bias"))
    sparse_flag = _has_sparse_attention_flag(config, attn_counts)
    wrapper_present = _wrapper_package_present()

    remaining_blockers = []
    if not wrapper_present:
        remaining_blockers.append("bloombee_minimax_m2_wrapper_missing")
    if num_local_experts and top_k:
        remaining_blockers.append("minimax_m2_moe_router_contract_unimplemented")
    if use_mtp:
        remaining_blockers.append("minimax_m2_mtp_contract_unimplemented")
    remaining_blockers.append("one_block_server_proof_missing")

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model_id,
        "native_bloombee_target": True,
        "model_type": config.get("model_type"),
        "architecture": _first_architecture(config),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_local_experts": num_local_experts,
        "num_experts_per_tok": top_k,
        "intermediate_size": config.get("intermediate_size"),
        "shared_intermediate_size": config.get("shared_intermediate_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "attn_type_counts": attn_counts,
        "exact_config_has_sparse_attention_flag": sparse_flag,
        "use_mtp": use_mtp,
        "num_mtp_modules": config.get("num_mtp_modules"),
        "mtp_transformer_layers": config.get("mtp_transformer_layers"),
        "use_qk_norm": use_qk_norm,
        "qk_norm_type": config.get("qk_norm_type"),
        "use_routing_bias": use_routing_bias,
        "scoring_func": config.get("scoring_func"),
        "router_jitter_noise": config.get("router_jitter_noise"),
        "native_wrapper_package_present": wrapper_present,
        "state_cache_contract": {
            "attention_cache_kind": "dynamic_kv_per_layer",
            "kv_layers": num_layers,
            "kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "moe_router_kind": "sigmoid_top_k_with_optional_routing_bias",
            "moe_router_top_k": top_k,
            "local_experts": num_local_experts,
            "mtp_requires_explicit_contract": use_mtp,
            "sparse_attention_requires_explicit_contract": sparse_flag,
        },
        "required_native_components": [
            "MiniMaxM2DecoderLayer block wrapper",
            "MiniMaxM2Attention q/k/v/o projection + QK norm + rotary path",
            "DynamicCache-compatible per-layer KV descriptor select/update",
            "MiniMaxM2SparseMoeBlock sigmoid top-k router with routing bias",
            "MTP module guard/contract so base decoder proof cannot silently include unsupported MTP behavior",
        ],
        "native_bloombee_support_proven": False,
        "live_run_attempted": False,
        "one_block_server_proven": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "remaining_blockers": remaining_blockers,
        "do_not_claim": [
            "no BloomBee minimax_m2 wrapper exists in this repository yet",
            "no live inference was attempted",
            "no one-block server proof",
            "no route/demo promotion",
        ],
        "generated_at_utc": generated_at_utc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = build_minimax_m27_native_contract_report(config, model_id=args.model)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
