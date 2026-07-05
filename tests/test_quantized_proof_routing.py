"""Quantized proof rows + quantization-aware routing (handover Tasks 3+4).

Contract under test:

- Proof rows are keyed ``model_id@quant_type`` (plain id == fp16). A missing
  quantized row means all-pending — quantized routes NEVER inherit fp16 gates.
- A quantized row is demo_safe only when full_generation, cache_generation,
  and multi_request_load are passed AND ``token_parity: exact`` is recorded
  (exact greedy token-ID match vs the fp16 reference). ``diverged`` or absent
  parity caps the row below demo_safe no matter what the gates say.
- route_picker can evaluate quantized variants: int8 halves the fp16 memory
  requirement (measured 1.996x, planned as /2.0), nf4 divides by 3.5 and is
  only servable for qwen3_moe; dense nf4 variants are blocked, matching the
  fail-closed loading path.
- fp16 rows are unaffected by the token_parity key.
"""
import json
from pathlib import Path

import pytest

from mvp_capabilities.model_compat_scan import is_demo_safe, split_route_id
from mvp_capabilities.proof_ladder import build_proof_ladder
from mvp_capabilities.route_picker import (
    derive_quantized_variant,
    evaluate_model,
    expand_quantized_variants,
    route_report,
)

M4PRO_LIKE_PEER = {
    "hostname": "m4pro",
    "memory": {"total_gb": 48.0, "free_gb": 37.0},
    "accelerator": {"device": "mps", "unified_memory": True, "vram_total_gb": 48.0, "vram_free_gb": 37.0},
}

QWEN30B = {
    "model_id": "Qwen/Qwen3-30B-A3B",
    "params_b": 30.5,
    "active_params_b": 3.3,
    "supports_moe": True,
    "recommended_min_free_mem_gb": 70,
    "quality_rank": 40.5,
}

TINYLLAMA = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "params_b": 1.1,
    "recommended_min_free_mem_gb": 4,
    "quality_rank": 1.1,
}

ALL_GATES_PASSED = {
    "prescan": "passed",
    "one_block_server": "passed",
    "multi_block": "passed",
    "full_generation": "passed",
    "cache_generation": "passed",
    "multi_request_load": "passed",
}


def test_split_route_id():
    assert split_route_id("Qwen/Qwen3-30B-A3B@int8") == ("Qwen/Qwen3-30B-A3B", "int8")
    assert split_route_id("Qwen/Qwen3-30B-A3B@nf4") == ("Qwen/Qwen3-30B-A3B", "nf4")
    assert split_route_id("Qwen/Qwen3-30B-A3B") == ("Qwen/Qwen3-30B-A3B", None)
    # unknown suffixes are not quant markers; the id stays whole (fail-closed:
    # a typo'd suffix becomes an unknown model, not a silent fp16 row)
    assert split_route_id("org/model@fp8") == ("org/model@fp8", None)


def test_is_demo_safe_quantized_requires_exact_token_parity():
    # fp16 rows: gates alone decide
    assert is_demo_safe(dict(ALL_GATES_PASSED)) is True
    assert is_demo_safe({**ALL_GATES_PASSED, "multi_request_load": "pending"}) is False
    # quantized rows: gates + exact parity
    assert is_demo_safe(dict(ALL_GATES_PASSED), quant_type="int8") is False
    assert is_demo_safe({**ALL_GATES_PASSED, "token_parity": "exact"}, quant_type="int8") is True
    assert is_demo_safe({**ALL_GATES_PASSED, "token_parity": "diverged"}, quant_type="int8") is False
    # fp16 rows ignore the parity key entirely
    assert is_demo_safe({**ALL_GATES_PASSED, "token_parity": "diverged"}) is True


def test_int8_variant_fits_m4pro_where_fp16_does_not():
    fp16 = evaluate_model([M4PRO_LIKE_PEER], dict(QWEN30B))
    assert fp16["memory_fit"] is False

    variant = derive_quantized_variant(QWEN30B, "int8")
    assert variant["model_id"] == "Qwen/Qwen3-30B-A3B@int8"
    int8 = evaluate_model([M4PRO_LIKE_PEER], variant)
    assert int8["memory_fit"] is True
    assert int8["placement"] == "solo"
    assert int8["quant_type"] == "int8"
    assert int8["required_free_gb"] == pytest.approx(35.0)


def test_quantized_variant_does_not_inherit_fp16_proof_row():
    proof = {"Qwen/Qwen3-30B-A3B": dict(ALL_GATES_PASSED)}  # fp16 fully proven
    variant = derive_quantized_variant(QWEN30B, "int8")
    result = evaluate_model([M4PRO_LIKE_PEER], variant, proof_status=proof, selector_mode="safe-demo")
    assert result["proof_status"]["full_generation"] == "pending"
    assert result["claim_level"] == "experimental"
    assert result["selector_allowed"] is False


def test_quantized_variant_demo_safe_needs_gates_plus_parity():
    variant = derive_quantized_variant(QWEN30B, "int8")

    gates_only = {"Qwen/Qwen3-30B-A3B@int8": dict(ALL_GATES_PASSED)}
    result = evaluate_model([M4PRO_LIKE_PEER], variant, proof_status=gates_only, selector_mode="safe-demo")
    assert result["claim_level"] == "experimental"
    assert result["selector_allowed"] is False

    with_parity = {"Qwen/Qwen3-30B-A3B@int8": {**ALL_GATES_PASSED, "token_parity": "exact"}}
    result = evaluate_model([M4PRO_LIKE_PEER], variant, proof_status=with_parity, selector_mode="safe-demo")
    assert result["claim_level"] == "demo_safe"
    assert result["selector_allowed"] is True


def test_nf4_variant_blocked_for_dense_models():
    dense_nf4 = derive_quantized_variant(TINYLLAMA, "nf4")
    result = evaluate_model([M4PRO_LIKE_PEER], dense_nf4)
    assert result["claim_level"] == "blocked"
    assert any("qwen3_moe" in reason for reason in result["blocked_reasons"])

    moe_nf4 = derive_quantized_variant(QWEN30B, "nf4")
    result = evaluate_model([M4PRO_LIKE_PEER], moe_nf4)
    assert result["claim_level"] != "blocked"
    assert result["required_free_gb"] == pytest.approx(20.0)


def test_expand_quantized_variants_skips_blocked_and_orders_below_fp16():
    blocked = {"model_id": "some/unsupported", "architecture_supported": False}
    variants = expand_quantized_variants([QWEN30B, TINYLLAMA, blocked])
    ids = [v["model_id"] for v in variants]
    assert "Qwen/Qwen3-30B-A3B@int8" in ids
    assert "Qwen/Qwen3-30B-A3B@nf4" in ids
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0@int8" in ids
    assert not any(v["model_id"].startswith("some/unsupported") for v in variants)
    by_id = {v["model_id"]: v for v in variants}
    # prefer fp16 at equal fit, int8 over nf4
    assert by_id["Qwen/Qwen3-30B-A3B@int8"]["quality_rank"] < QWEN30B["quality_rank"]
    assert by_id["Qwen/Qwen3-30B-A3B@nf4"]["quality_rank"] < by_id["Qwen/Qwen3-30B-A3B@int8"]["quality_rank"]


def test_route_report_serves_proven_quantized_pin_and_refuses_unproven():
    registry = [dict(QWEN30B), dict(TINYLLAMA)]
    proof = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": dict(ALL_GATES_PASSED),
        "Qwen/Qwen3-30B-A3B@int8": {**ALL_GATES_PASSED, "token_parity": "exact"},
    }
    report = route_report(
        [M4PRO_LIKE_PEER],
        registry,
        requested_model="Qwen/Qwen3-30B-A3B@int8",
        proof_status=proof,
        selector_mode="safe-demo",
    )
    assert report["override_refused"] is False
    assert report["serving"]["model_id"] == "Qwen/Qwen3-30B-A3B@int8"
    assert report["serving"]["quant_type"] == "int8"

    # same pin without the parity fact -> refused, fp16-proven fallback served
    proof_no_parity = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": dict(ALL_GATES_PASSED),
        "Qwen/Qwen3-30B-A3B@int8": dict(ALL_GATES_PASSED),
    }
    report = route_report(
        [M4PRO_LIKE_PEER],
        registry,
        requested_model="Qwen/Qwen3-30B-A3B@int8",
        proof_status=proof_no_parity,
        selector_mode="safe-demo",
    )
    assert report["override_refused"] is True
    assert report["serving"]["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_proof_ladder_quantized_row_reports_parity_gate():
    gates_only = {"Qwen/Qwen3-30B-A3B@int8": dict(ALL_GATES_PASSED)}
    ladder = build_proof_ladder("Qwen/Qwen3-30B-A3B@int8", proof_status=gates_only)
    assert ladder["quant_type"] == "int8"
    assert ladder["base_model_id"] == "Qwen/Qwen3-30B-A3B"
    assert ladder["claim_level"] == "experimental"
    assert ladder["safe_demo_selectable"] is False
    assert ladder["token_parity"] is None

    with_parity = {"Qwen/Qwen3-30B-A3B@int8": {**ALL_GATES_PASSED, "token_parity": "exact"}}
    ladder = build_proof_ladder("Qwen/Qwen3-30B-A3B@int8", proof_status=with_parity)
    assert ladder["claim_level"] == "demo_safe"
    assert ladder["safe_demo_selectable"] is True
    assert ladder["token_parity"] == "exact"


def test_committed_proof_status_has_failclosed_partial_int8_row():
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status

    proof = load_proof_status(DEFAULT_PROOF_STATUS)
    row = proof.get("Qwen/Qwen3-30B-A3B@int8")
    assert row is not None, "quantized proof row must exist explicitly (fail-closed, not implied)"
    assert row["prescan"] == "passed"
    assert row["one_block_server"] == "passed"
    assert row["multi_block"] == "passed"
    assert row["multi_request_load"] == "passed"
    assert row["full_generation"] == "pending"
    assert row["cache_generation"] == "pending"
    assert row["token_parity"] == "not_evaluated_reference_fp16_exceeds_m4pro_memory"
    assert is_demo_safe(row, quant_type="int8") is False


def test_committed_int8_evidence_artifacts_back_partial_proof_row():
    root = Path(__file__).resolve().parents[1] / "mvp_capabilities/distributed_evidence/post_mvp"
    artifacts = {
        "qwen30b-int8-oneblock-20260705T131443Z.json": "qwen30b_int8_one_block_server_direct_rpc_only",
        "qwen30b-int8-multiblock-0-2-20260705T131529Z.json": "qwen30b_int8_multiblock_0_2_server_direct_rpc_only",
        "qwen30b-int8-full-load-0-48-20260705T131803Z.json": "qwen30b_int8_full_48_block_multi_request_load_only_no_token_parity",
    }
    for filename, claim_boundary in artifacts.items():
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        assert payload["model_id"] == "Qwen/Qwen3-30B-A3B"
        assert payload["route_id"] == "Qwen/Qwen3-30B-A3B@int8"
        assert payload["quant_type"] == "int8"
        assert payload["status"] == "passed"
        assert payload["claim_boundary"] == claim_boundary
        assert payload["safe_demo_selectable"] is False
        assert payload["can_update_demo_status"] is False

    load = json.loads((root / "qwen30b-int8-full-load-0-48-20260705T131803Z.json").read_text())
    assert load["quantized_block_count"] == 48
    assert load["verifier"]["status"] == "passed"
    assert load["multi_request_load_proven"] is True
    assert load["full_generation_proven"] is False
    assert load["cache_generation_proven"] is False
    assert load["token_parity"] == "not_evaluated_reference_fp16_exceeds_m4pro_memory"
