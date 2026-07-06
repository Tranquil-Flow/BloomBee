#!/usr/bin/env python3
"""Exact config-scan report for Qwen3.6-35B-A3B / Qwen36A.

This is a claim-boundary tool. It can prove that the public HF config maps to the
same unsupported qwen3_5_moe_text family as AgentWorld-style work, but it cannot
prove runtime serving, route safety, or demo readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvp_capabilities.model_compat_scan import scan_model_config

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_CONFIG_URL = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json"
CLAIM_BOUNDARY = "post_mvp_qwen36a_exact_config_scan_no_runtime_proof_no_demo_promotion"


def _read_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else {}


def _layer_type_counts(config: dict[str, Any]) -> dict[str, int]:
    text_config = _text_config(config)
    layer_types = text_config.get("layer_types") or []
    return dict(Counter(str(item) for item in layer_types))


def _raw_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_qwen36a_report(
    config: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    config_source_url: str = DEFAULT_CONFIG_URL,
) -> dict[str, Any]:
    """Build a fail-closed exact-config report for the Qwen36A native lane."""
    config_path = Path(config)
    config_payload = _read_config(config_path)
    scan = scan_model_config(config_path, model_id=model_id, local_files_only=True)
    layer_counts = _layer_type_counts(config_payload)
    text_config = _text_config(config_payload)

    blocked_reasons = list(scan.get("blocked_reasons") or [])
    if layer_counts.get("linear_attention", 0):
        blocked_reasons.append(
            "qwen3_5_moe_text linear_attention layers require BloomBee backend linear-state cache descriptors/materialization/select/update before one-block proof"
        )
    blocked_reasons.append("exact Qwen36A one-block server proof has not been run")
    blocked_reasons.append("full/cache generation and multi-request load gates have not been run")

    family_match = None
    if scan.get("hf_model_type") == "qwen3_5_moe" and scan.get("hf_text_model_type") == "qwen3_5_moe_text":
        family_match = "qwen3_5_moe_text_linear_attention_family"

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model_id,
        "config_source_url": config_source_url,
        "config_source_kind": "exact_hf_config_json_snapshot",
        "config_sha256": _raw_sha256(config_path),
        "native_bloombee_distributed_path_target": True,
        "exact_config_scan": "passed",
        "config_scan": scan,
        "layer_type_counts": layer_counts,
        "full_attention_interval": text_config.get("full_attention_interval"),
        "linear_attention_state_contract": {
            "requires_conv_state": bool(layer_counts.get("linear_attention")),
            "requires_recurrent_state": bool(layer_counts.get("linear_attention")),
            "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
            "linear_num_key_heads": text_config.get("linear_num_key_heads"),
            "linear_num_value_heads": text_config.get("linear_num_value_heads"),
            "linear_key_head_dim": text_config.get("linear_key_head_dim"),
            "linear_value_head_dim": text_config.get("linear_value_head_dim"),
        },
        "qwen35b_family_match": family_match,
        "wrapper_code_written_for_exact_model": False,
        "one_block_server_proven": False,
        "multi_block_proven": False,
        "full_generation_proven": False,
        "cache_generation_proven": False,
        "multi_request_load_proven": False,
        "can_update_route_status": False,
        "can_update_demo_status": False,
        "can_update_mvp_status": False,
        "blocked_reasons": blocked_reasons,
        "recommended_next_step": "map_qwen36a_to_qwen3_5_moe_text_backend_state_cache_then_one_block_proof",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to exact Qwen36A config.json snapshot")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--config-source-url", default=DEFAULT_CONFIG_URL)
    args = parser.parse_args(argv)

    report = build_qwen36a_report(
        args.config,
        model_id=args.model_id,
        config_source_url=args.config_source_url,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
