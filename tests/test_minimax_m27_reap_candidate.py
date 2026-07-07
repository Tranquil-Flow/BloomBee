import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"


def test_minimax_m27_reap_candidate_marks_bloombee_native_blocked_but_llamacpp_quant_attemptable_on_large_m4():
    from mvp_capabilities.minimax_m27_reap_candidate import build_minimax_m27_reap_candidate_report

    report = build_minimax_m27_reap_candidate_report(
        peers=[{"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 42.0}}]
    )

    assert report["claim_boundary"] == "minimax_m27_reap_candidate_preflight_external_runtime_no_live_inference_claim"
    assert report["model_id"] == MODEL_ID
    assert report["params_b"] == 139.0
    assert report["active_params_b"] == 10.0
    assert report["architecture_supported"] is True
    assert report["native_wrapper_package_present"] is True
    assert report["native_bloombee_support_proven"] is False
    assert report["route_picker_eligible"] is False
    assert report["can_update_proof_status"] is False
    assert report["live_run_attempted"] is False
    assert "real-weight one-block server proof" in " ".join(report["bloombee_blocked_reasons"])
    assert "external runtime is side diagnostics only" in " ".join(report["bloombee_blocked_reasons"])
    assert report["gguf_external_runtime"]["attemptable_on_best_peer"] is True
    assert report["gguf_external_runtime"]["best_peer"]["hostname"] == "m4pro"
    assert report["gguf_external_runtime"]["selected_quant"]["name"] == "i1-IQ2_XXS"
    assert report["gguf_external_runtime"]["selected_quant"]["size_gb"] == 36.8
    assert report["gguf_external_runtime"]["selected_quant"]["quality_note"] == "low_quality_but_first_m4pro_plausible_smoke"
    assert any("llama.cpp" in command for command in report["operator_commands"])


def test_minimax_m27_reap_candidate_fails_closed_when_only_current_macs_are_too_small():
    from mvp_capabilities.minimax_m27_reap_candidate import build_minimax_m27_reap_candidate_report

    report = build_minimax_m27_reap_candidate_report(
        peers=[
            {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
            {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 30.0}},
        ]
    )

    assert report["gguf_external_runtime"]["attemptable_on_best_peer"] is False
    assert report["gguf_external_runtime"]["selected_quant"] is None
    assert "no_peer_has_free_memory_for_i1-IQ2_XXS_plus_margin" in report["blocked_reasons"]
    assert report["can_update_demo_status"] is False


def test_minimax_m27_reap_registry_row_is_visible_but_not_quant_variant_expanded():
    from mvp_capabilities.route_picker import expand_quantized_variants, load_registry

    registry = load_registry(PROJECT_ROOT / "mvp_capabilities" / "MODEL_REGISTRY.yaml")
    by_id = {row["model_id"]: row for row in registry}

    assert MODEL_ID in by_id
    row = by_id[MODEL_ID]
    assert row["architecture_supported"] is True
    assert row["bloombee_family"] == "minimax_m2"
    assert row["block_prefix"] == "model.layers"
    assert row["recommended_min_free_mem_gb"] == 280
    assert row["native_contract_artifact"] == "mvp_capabilities/distributed_evidence/post_mvp/minimax-m27-reap-native-contract-scan-20260706.json"
    assert "minimax_m2" in " ".join(row["blocked_reasons"])

    variant_ids = {row["model_id"] for row in expand_quantized_variants(registry)}
    # Quantized variants now expand for models with architecture support (wrapper exists)
    # even when proof-level blockers exist. Proof blocks don't prevent showing memory reqs.
    assert f"{MODEL_ID}@int8" in variant_ids, "int8 variant should expand for arch-supported models"
    assert f"{MODEL_ID}@nf4" not in variant_ids, "nf4 only for qwen3_moe"


def test_minimax_m27_reap_candidate_cli_writes_json(tmp_path: Path):
    out_path = tmp_path / "minimax-m27-report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/minimax_m27_reap_candidate.py",
            "--peer",
            "m4pro:48:42",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["model_id"] == MODEL_ID
    assert payload["gguf_external_runtime"]["attemptable_on_best_peer"] is True


def test_minimax_family_comparison_keeps_m3_as_powerful_but_not_currently_easier_on_macs():
    from mvp_capabilities.minimax_m27_reap_candidate import build_minimax_reap_family_comparison_report

    report = build_minimax_reap_family_comparison_report(
        peers=[
            {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 4.0}},
            {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 27.3}},
        ],
        runtime_inventory={"local-m4": {"llama_cpp": True, "vmlx": False}, "m4pro": {"llama_cpp": False, "vmlx": False}},
    )

    assert report["claim_boundary"] == "minimax_reap_family_comparison_no_live_inference_claim"
    assert report["combined_nominal_memory_gb"] == 64.0
    assert report["combined_free_memory_gb"] == 31.3
    assert report["can_pool_m4_and_m4pro_memory_for_external_runtime"] is False
    assert report["decision"]["keep_m3_as_option"] is True
    assert report["decision"]["m3_is_likely_more_powerful"] is True
    assert report["decision"]["m3_is_easier_to_run_now"] is False
    assert report["decision"]["preferred_current_local_target"] == "none_current_memory_too_low"
    assert report["models"]["m3_full"]["minimum_known_gguf_gb"] == 128.0
    assert "m3_minimum_gguf_128.0gb_exceeds_combined_nominal_64.0gb" in report["models"]["m3_full"]["blocked_reasons"]
    assert "vmlx_not_installed_on_any_peer" in report["models"]["m3_reap_jang"]["blocked_reasons"]
    assert "external runtimes need one host" in " ".join(report["shared_limitations"])


def test_minimax_family_comparison_prefers_m27_when_one_m4pro_has_enough_free_memory():
    from mvp_capabilities.minimax_m27_reap_candidate import build_minimax_reap_family_comparison_report

    report = build_minimax_reap_family_comparison_report(
        peers=[
            {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
            {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 42.0}},
        ],
        runtime_inventory={"m4pro": {"llama_cpp": True, "vmlx": False}},
    )

    assert report["decision"]["preferred_current_local_target"] == "minimax_m2_7_reap_139b_a10b_external_llamacpp"
    assert report["models"]["m2_7_reap_139b_a10b"]["gguf_external_runtime"]["attemptable_on_best_peer"] is True
    assert report["models"]["m2_7_reap_139b_a10b"]["gguf_external_runtime"]["selected_quant"]["name"] == "i1-IQ2_XXS"
    assert report["can_pool_m4_and_m4pro_memory_for_external_runtime"] is False


def test_minimax_family_comparison_cli_writes_json(tmp_path: Path):
    out_path = tmp_path / "minimax-family-comparison.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/minimax_m27_reap_candidate.py",
            "--compare-family",
            "--peer",
            "local-m4:16:4",
            "--peer",
            "m4pro:48:27.3",
            "--runtime",
            "local-m4:llama_cpp=1:vmlx=0",
            "--runtime",
            "m4pro:llama_cpp=0:vmlx=0",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["decision"]["keep_m3_as_option"] is True
    assert payload["models"]["m2_7_reap_139b_a10b"]["live_run_attempted"] is False
