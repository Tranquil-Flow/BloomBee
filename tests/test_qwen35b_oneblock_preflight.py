from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen-AgentWorld-35B-A3B"
QWEN36A_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def _registry():
    return [
        {
            "model_id": MODEL_ID,
            "hidden_size": 2048,
            "recommended_min_free_mem_gb": 80,
        }
    ]


def test_qwen35b_oneblock_preflight_fails_closed_on_small_host_memory():
    from mvp_capabilities.qwen35b_oneblock_preflight import build_qwen35b_oneblock_preflight

    result = build_qwen35b_oneblock_preflight(
        registry=_registry(),
        host_total_mem_gb=16,
        host_free_mem_gb=12,
        host_label="local-m4",
    )

    assert result["claim_boundary"] == "qwen35b_one_block_host_preflight_no_live_inference"
    assert result["model_id"] == MODEL_ID
    assert result["proof_gate"] == "one_block_server"
    assert result["live_run_attempted"] is False
    assert result["one_block_server_proven"] is False
    assert result["can_update_proof_status"] is False
    assert result["ready_to_attempt_live_oneblock"] is False
    assert result["status"] == "blocked-by-host-memory"
    assert "insufficient_host_memory_for_qwen35b_one_block" in result["remaining_blockers"]
    assert result["proof_status_update"] == {}


def test_qwen35b_oneblock_preflight_marks_large_host_as_attempt_ready_not_proven():
    from mvp_capabilities.qwen35b_oneblock_preflight import build_qwen35b_oneblock_preflight

    result = build_qwen35b_oneblock_preflight(
        registry=_registry(),
        host_total_mem_gb=128,
        host_free_mem_gb=96,
        host_label="large-m4",
    )

    assert result["status"] == "ready-to-attempt"
    assert result["ready_to_attempt_live_oneblock"] is True
    assert result["one_block_server_proven"] is False
    assert result["can_update_proof_status"] is False
    assert result["proof_status_update"] == {}
    assert result["one_block_plan"]["claim_boundary"] == "proof_harness_only_no_live_inference"
    assert "one_block_proof.py verify" in result["one_block_plan"]["verify_command"]


def test_qwen36a_oneblock_preflight_uses_exact_model_blockers_and_plan():
    from mvp_capabilities.qwen35b_oneblock_preflight import build_qwen35b_oneblock_preflight

    result = build_qwen35b_oneblock_preflight(
        registry=[
            {
                "model_id": QWEN36A_MODEL_ID,
                "hidden_size": 2048,
                "recommended_min_free_mem_gb": 80,
                "candidate_branch": "qwen36a",
            }
        ],
        model_id=QWEN36A_MODEL_ID,
        host_total_mem_gb=48,
        host_free_mem_gb=33.5,
        host_label="m4pro",
    )

    assert result["claim_boundary"] == "qwen36a_one_block_host_preflight_no_live_inference"
    assert result["model_id"] == QWEN36A_MODEL_ID
    assert result["status"] == "blocked-by-host-memory"
    assert "insufficient_host_memory_for_qwen36a_one_block" in result["remaining_blockers"]
    assert QWEN36A_MODEL_ID in result["one_block_plan"]["server_command"]
    assert result["one_block_server_proven"] is False
    assert result["can_update_proof_status"] is False


def test_qwen35b_oneblock_preflight_cli_writes_json(tmp_path: Path):
    out = tmp_path / "preflight.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/qwen35b_oneblock_preflight.py",
            "--host-total-mem-gb",
            "16",
            "--host-free-mem-gb",
            "12",
            "--host-label",
            "ci-small-host",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["host_label"] == "ci-small-host"
    assert payload["status"] == "blocked-by-host-memory"
    assert json.loads(proc.stdout) == payload
