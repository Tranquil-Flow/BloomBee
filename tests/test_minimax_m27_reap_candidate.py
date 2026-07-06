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

    assert report["claim_boundary"] == "minimax_m27_reap_candidate_preflight_no_bloombee_or_live_inference_claim"
    assert report["model_id"] == MODEL_ID
    assert report["params_b"] == 139.0
    assert report["active_params_b"] == 10.0
    assert report["native_bloombee_support_proven"] is False
    assert report["route_picker_eligible"] is False
    assert report["can_update_proof_status"] is False
    assert report["live_run_attempted"] is False
    assert "No BloomBee block wrapper registered for model_type=minimax_m2" in report["bloombee_blocked_reasons"]
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
    assert row["architecture_supported"] is False
    assert row["recommended_min_free_mem_gb"] == 280
    assert "minimax_m2" in " ".join(row["blocked_reasons"])

    variant_ids = {row["model_id"] for row in expand_quantized_variants(registry)}
    assert f"{MODEL_ID}@int8" not in variant_ids
    assert f"{MODEL_ID}@nf4" not in variant_ids


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
