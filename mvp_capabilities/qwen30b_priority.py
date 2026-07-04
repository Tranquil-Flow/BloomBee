#!/usr/bin/env python3
"""Post-MVP Qwen3-30B prioritization report.

This module does not run inference. It codifies the post-MVP recommendation that
base Qwen3-30B-A3B should be the substrate/risk reducer, Instruct-2507 should be
the user-facing follow-up after base gates are understood, and Thinking-2507
should stay optional unless the demo explicitly needs reasoning behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, PROOF_KEYS, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/qwen30b_priority.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, PROOF_KEYS, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry

CLAIM_BOUNDARY = "post_mvp_qwen30b_prioritization_only_no_new_inference_proof"
BASE_MODEL_ID = "Qwen/Qwen3-30B-A3B"
INSTRUCT_2507_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
THINKING_2507_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
PRIORITY_ORDER = (BASE_MODEL_ID, INSTRUCT_2507_MODEL_ID, THINKING_2507_MODEL_ID)
BASE_GATES_BEFORE_2507 = ("full_generation", "cache_generation", "multi_request_load")
RECOMMENDATION_SUMMARY = (
    "Use base Qwen3-30B-A3B as the post-MVP substrate, then Instruct-2507 if a "
    "user-facing stronger demo is needed; keep Thinking-2507 optional."
)

_PRIORITY_META: dict[str, dict[str, Any]] = {
    BASE_MODEL_ID: {
        "priority_rank": 1,
        "priority_role": "substrate_risk_reducer",
        "recommended_after_base_gates": [],
        "optional": False,
        "defer_unless": None,
        "why": (
            "Base Qwen3-30B-A3B already has prescan, one-block, and multi-block proof, "
            "so it reduces infrastructure risk before spending proof budget on exact 2507 variants."
        ),
    },
    INSTRUCT_2507_MODEL_ID: {
        "priority_rank": 2,
        "priority_role": "user_facing_followup",
        "recommended_after_base_gates": list(BASE_GATES_BEFORE_2507),
        "optional": False,
        "defer_unless": None,
        "why": (
            "Instruct-2507 is the likely product-facing stronger demo target, but exact-model "
            "proof gates are still pending and should not distract from base 30B substrate proof."
        ),
    },
    THINKING_2507_MODEL_ID: {
        "priority_rank": 3,
        "priority_role": "optional_reasoning_variant",
        "recommended_after_base_gates": list(BASE_GATES_BEFORE_2507),
        "optional": True,
        "defer_unless": "demo_specifically_needs_thinking_or_reasoning_behavior",
        "why": (
            "Thinking-2507 likely shares the same infrastructure class, but it costs another exact-model "
            "proof ladder and should wait unless reasoning-style behavior is a concrete demo requirement."
        ),
    },
}


def _default_status() -> dict[str, str]:
    return {gate: "pending" for gate in PROOF_KEYS}


def _status_for(model_id: str, proof_status: dict[str, dict[str, str]] | None) -> dict[str, str]:
    status = _default_status()
    if proof_status and model_id in proof_status:
        status.update({str(key): str(value) for key, value in proof_status[model_id].items()})
    return status


def _next_gate(status: dict[str, str]) -> str | None:
    return next((gate for gate in PROOF_KEYS if status.get(gate) != "passed"), None)


def _safe_demo_selectable(status: dict[str, str]) -> bool:
    return status.get("full_generation") == "passed"


def _registry_by_id(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(model.get("model_id")): dict(model) for model in registry}


def _shared_architecture(models: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("hidden_size", "num_layers", "num_experts", "num_experts_per_tok")
    shared: dict[str, Any] = {
        "all_qwen3_moe": all(model.get("hf_model_type") == "qwen3_moe" for model in models),
    }
    for key in keys:
        values = {model.get(key) for model in models}
        shared[key] = values.pop() if len(values) == 1 else None
    shared["same_memory_class"] = len({model.get("recommended_min_free_mem_gb") for model in models}) == 1
    return shared


def _model_report(model_id: str, model: dict[str, Any], status: dict[str, str]) -> dict[str, Any]:
    meta = _PRIORITY_META[model_id]
    return {
        "model_id": model_id,
        "priority_rank": meta["priority_rank"],
        "priority_role": meta["priority_role"],
        "optional": meta["optional"],
        "defer_unless": meta["defer_unless"],
        "recommended_after_base_gates": meta["recommended_after_base_gates"],
        "why": meta["why"],
        "mvp_critical": False,
        "hf_model_type": model.get("hf_model_type"),
        "params_b": model.get("params_b"),
        "active_params_b": model.get("active_params_b"),
        "hidden_size": model.get("hidden_size"),
        "num_layers": model.get("num_layers"),
        "num_experts": model.get("num_experts"),
        "num_experts_per_tok": model.get("num_experts_per_tok"),
        "recommended_min_free_mem_gb": model.get("recommended_min_free_mem_gb"),
        "proof_status": status,
        "gates_already_passed": [gate for gate in PROOF_KEYS if status.get(gate) == "passed"],
        "next_gate": _next_gate(status),
        "safe_demo_selectable": _safe_demo_selectable(status),
    }


def build_qwen30b_priority_report(
    *,
    registry: list[dict[str, Any]] | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Return a deterministic post-MVP priority report for the Qwen3-30B family."""

    registry = load_registry(DEFAULT_REGISTRY) if registry is None else registry
    proof_status = load_proof_status(DEFAULT_PROOF_STATUS) if proof_status is None else proof_status
    by_id = _registry_by_id(registry)
    missing = [model_id for model_id in PRIORITY_ORDER if model_id not in by_id]
    if missing:
        raise KeyError(f"missing Qwen30B priority models from registry: {', '.join(missing)}")

    models = [by_id[model_id] for model_id in PRIORITY_ORDER]
    model_reports = [
        _model_report(model_id, by_id[model_id], _status_for(model_id, proof_status))
        for model_id in PRIORITY_ORDER
    ]
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "recommendation_summary": RECOMMENDATION_SUMMARY,
        "mvp_core_dependency": "none_mvp_core_closed_by_qwen3_8b",
        "priority_order": list(PRIORITY_ORDER),
        "shared_architecture": _shared_architecture(models),
        "base_gates_before_2507_focus": list(BASE_GATES_BEFORE_2507),
        "models": model_reports,
        "review_note": (
            "This report is audit/planning metadata only. It does not change PROOF_STATUS.yaml and does not promote "
            "any Qwen3-30B-family model to demo_safe."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--proof-status", default=str(DEFAULT_PROOF_STATUS))
    args = parser.parse_args(argv)

    report = build_qwen30b_priority_report(
        registry=load_registry(args.registry),
        proof_status=load_proof_status(args.proof_status),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
