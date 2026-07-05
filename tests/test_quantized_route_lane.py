from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "mvp_capabilities" / "MODEL_REGISTRY.yaml"
EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/quantized-qwen30b-route-lane-20260704.json"


def _m4pro_peer(free_gb: float = 37.0) -> dict:
    return {
        "hostname": "m4pro",
        "memory": {"total_gb": 48.0, "free_gb": free_gb},
        "accelerator": {"device": "mps", "unified_memory": True, "vram_total_gb": 48.0, "vram_free_gb": free_gb},
    }


def test_quantized_qwen30b_lane_fits_m4pro_without_claiming_fp16_fit():
    from mvp_capabilities.quantized_route_lane import build_quantized_qwen30b_lane
    from mvp_capabilities.route_picker import load_registry

    report = build_quantized_qwen30b_lane(peers=[_m4pro_peer()], registry=load_registry(REGISTRY_PATH))

    assert report["claim_boundary"] == "quantized_route_lane_planning_only_no_serving_proof"
    assert report["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert report["route_id"] == "Qwen/Qwen3-30B-A3B@int8"
    assert report["quant_type"] == "int8"
    assert report["quant_scheme"] == "moe_int8_experts+qint8_attn"
    assert report["source_spike_artifact"].endswith("quantized-block-spike-20260704T203500Z.json")
    assert report["fp16_route"]["memory_fit"] is False
    assert report["fp16_route"]["required_free_gb"] == 70.0
    assert report["quantized_route"]["memory_fit"] is True
    assert report["quantized_route"]["required_free_gb"] == 35.1
    assert report["quantized_route"]["placement"] == "solo"
    assert report["quantized_route"]["solo_hosts"] == ["m4pro"]
    assert report["memory_reduction_source"] == "quantized_block_spike_random_weight_qwen3_moe_layer"
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["demo_safe_allowed"] is False


def test_quantized_qwen30b_lane_does_not_inherit_fp16_proof_status():
    from mvp_capabilities.quantized_route_lane import build_quantized_qwen30b_lane
    from mvp_capabilities.route_picker import load_registry

    fp16_all_passed = {
        "Qwen/Qwen3-30B-A3B": {
            "prescan": "passed",
            "one_block_server": "passed",
            "multi_block": "passed",
            "full_generation": "passed",
            "cache_generation": "passed",
            "multi_request_load": "passed",
        }
    }
    report = build_quantized_qwen30b_lane(
        peers=[_m4pro_peer()], registry=load_registry(REGISTRY_PATH), proof_status=fp16_all_passed
    )

    assert report["fp16_proof_status"]["full_generation"] == "passed"
    assert report["quantized_proof_key"] == "Qwen/Qwen3-30B-A3B@int8"
    assert report["quantized_proof_status"] == {
        "prescan": "pending",
        "one_block_server": "pending",
        "multi_block": "pending",
        "full_generation": "pending",
        "cache_generation": "pending",
        "multi_request_load": "pending",
    }
    assert report["can_inherit_fp16_proof"] is False
    assert report["can_update_fp16_proof_row"] is False
    assert report["server_proof_status"] == "not_run"
    assert "do not update Qwen/Qwen3-30B-A3B fp16 proof row" in report["guardrails"]


def test_quantized_qwen30b_lane_reads_quantized_proof_row_when_present():
    from mvp_capabilities.quantized_route_lane import build_quantized_qwen30b_lane
    from mvp_capabilities.route_picker import load_registry

    proof = {
        "Qwen/Qwen3-30B-A3B": {"full_generation": "passed"},
        "Qwen/Qwen3-30B-A3B@int8": {"prescan": "passed", "one_block_server": "passed"},
    }
    report = build_quantized_qwen30b_lane(peers=[_m4pro_peer()], registry=load_registry(REGISTRY_PATH), proof_status=proof)

    assert report["quantized_proof_status"]["prescan"] == "passed"
    assert report["quantized_proof_status"]["one_block_server"] == "passed"
    assert report["quantized_proof_status"]["full_generation"] == "pending"
    assert report["demo_safe_allowed"] is False
    assert report["next_gate"] == "multi_block"


def test_quantized_qwen30b_lane_cli_and_tracked_evidence_are_claim_bounded():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/quantized_route_lane.py", "--example", "m4pro-int8-30b"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    tracked = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["route_id"] == tracked["route_id"] == "Qwen/Qwen3-30B-A3B@int8"
    assert tracked["verification_status"] == "passed"
    assert tracked["claim_boundary"] == "quantized_route_lane_planning_only_no_serving_proof"
    assert tracked["quantized_route"]["memory_fit"] is True
    assert tracked["fp16_route"]["memory_fit"] is False
    assert tracked["quantized_proof_status"]["prescan"] == "passed"
    assert tracked["quantized_proof_status"]["one_block_server"] == "passed"
    assert tracked["quantized_proof_status"]["multi_block"] == "passed"
    assert tracked["quantized_proof_status"]["multi_request_load"] == "passed"
    assert tracked["quantized_proof_status"]["full_generation"] == "passed"
    assert tracked["quantized_proof_status"]["cache_generation"] == "passed"
    assert tracked["quantized_proof_status"]["token_parity"] == "exact"
    assert tracked["next_gate"] is None
    assert tracked["server_proof_status"] == "passed"
    assert tracked["can_inherit_fp16_proof"] is False
    assert tracked["can_update_fp16_proof_row"] is False
    # This artifact remains planning-only; live server/load proof lives in the qwen30b-int8-* artifacts.
    assert tracked["live_server_proven"] is False
    assert tracked["demo_safe_allowed"] is True
    assert tracked["operator_next_steps"] == [
        "base Qwen/Qwen3-30B-A3B@int8 and Qwen/Qwen3-30B-A3B-Instruct-2507@int8 are demo-safe under the current full/cache/load/token-parity gates",
        "keep fp16, @int8, Instruct-2507, and Thinking-2507 proof rows separate; do not inherit gates across rows",
        "optional next target is broader prompt-set parity or Thinking-2507 only if the demo needs reasoning-style behavior",
    ]
