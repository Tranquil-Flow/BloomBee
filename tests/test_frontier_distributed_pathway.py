import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_minimax_m27_report_separates_desired_main_pathway_from_current_external_runtime():
    from mvp_capabilities.frontier_distributed_pathway import build_frontier_distributed_pathway_report

    report = build_frontier_distributed_pathway_report(
        target="minimax-m27-reap",
        peers=[
            {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
            {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 33.5}},
        ],
    )

    assert report["claim_boundary"] == "frontier_distributed_pathway_plan_no_live_inference_claim"
    assert report["target_key"] == "minimax_m27_reap"
    assert report["model_id"] == "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"
    assert report["main_bloombee_pathway"]["desired_end_state"] is True
    assert report["main_bloombee_pathway"]["usable_now"] is False
    assert report["main_bloombee_pathway"]["can_pool_peer_memory_after_adapter"] is True
    assert "missing_native_bloombee_wrapper:minimax_m2" in report["main_bloombee_pathway"]["blocked_reasons"]
    assert report["external_runtime_pathway"]["framework"] == "llama.cpp"
    assert report["external_runtime_pathway"]["can_pool_peer_memory_now"] is False
    assert report["external_runtime_pathway"]["single_host_required_free_gb"] == 40.8
    assert report["external_runtime_pathway"]["best_peer"]["hostname"] == "m4pro"
    assert report["external_runtime_pathway"]["best_peer_shortfall_gb"] == 7.3
    assert report["recommendation"]["next_engineering_lane"] == "native_bloombee_adapter_before_route_promotion"
    assert report["can_update_demo_status"] is False
    assert report["can_update_proof_status"] is False


def test_minimax_m27_future_adapter_mode_allows_main_pathway_but_still_requires_live_proofs():
    from mvp_capabilities.frontier_distributed_pathway import build_frontier_distributed_pathway_report

    report = build_frontier_distributed_pathway_report(
        target="minimax-m27-reap",
        peers=[
            {"hostname": "local-m4", "memory": {"total_gb": 16.0, "free_gb": 8.0}},
            {"hostname": "m4pro", "memory": {"total_gb": 48.0, "free_gb": 33.5}},
        ],
        native_wrapper_proven=True,
        one_block_proven=True,
    )

    assert report["main_bloombee_pathway"]["usable_now"] is True
    assert report["main_bloombee_pathway"]["blocked_reasons"] == []
    assert report["main_bloombee_pathway"]["can_pool_peer_memory_now"] is True
    assert report["recommendation"]["next_engineering_lane"] == "main_pathway_multiblock_and_generation_proofs"
    assert report["can_update_demo_status"] is False
    assert report["do_not_claim"] == [
        "no live inference was attempted by this planner",
        "no wall-clock speedup proof",
        "no demo/status promotion from planner output alone",
    ]


def test_qwen36a_alias_reports_exact_config_scanned_but_native_wrapper_still_missing():
    from mvp_capabilities.frontier_distributed_pathway import build_frontier_distributed_pathway_report

    report = build_frontier_distributed_pathway_report(target="qwen36a")

    assert report["target_key"] == "qwen3_6_35b_a3b"
    assert report["model_id"] == "Qwen/Qwen3.6-35B-A3B"
    assert report["main_bloombee_pathway"]["usable_now"] is False
    assert report["main_bloombee_pathway"]["config_scan_required"] is False
    assert report["main_bloombee_pathway"]["hf_model_type"] == "qwen3_5_moe"
    assert "exact_hf_config_not_scanned_in_repo" not in report["main_bloombee_pathway"]["blocked_reasons"]
    assert "missing_native_bloombee_wrapper:qwen3_5_moe_text" in report["main_bloombee_pathway"]["blocked_reasons"]
    assert report["external_runtime_pathway"]["can_pool_peer_memory_now"] is False
    assert report["external_runtime_pathway"]["single_host_required_free_gb"] == 80.0
    assert report["recommendation"]["next_engineering_lane"] == "native_bloombee_adapter_before_route_promotion"


def test_frontier_distributed_pathway_cli_writes_json(tmp_path: Path):
    out_path = tmp_path / "pathway.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/frontier_distributed_pathway.py",
            "--target",
            "minimax-m27-reap",
            "--peer",
            "local-m4:16:8",
            "--peer",
            "m4pro:48:33.5",
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
    assert payload["target_key"] == "minimax_m27_reap"
    assert payload["external_runtime_pathway"]["best_peer_shortfall_gb"] == 7.3
