from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.fable_handoff_check as fable_handoff_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "fable_handoff_check.py"


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_fable_handoff_check_json_is_grunt_free_and_claim_bounded():
    proc = _run_checker("--json")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["project"] == "distributed-inference-mvp"
    assert payload["mvp_status"]["task_summary"] == {
        "complete": 13,
        "partial": 3,
        "pending": 0,
        "blocked": 1,
        "total": 17,
    }
    assert payload["mvp_status"]["post_mvp_task_summary"] == {
        "complete": 4,
        "partial": 3,
        "pending": 0,
        "blocked": 1,
        "total": 8,
    }
    continuous = payload["mvp_status"]["continuous_batching"]
    assert continuous["status"] == "complete"
    assert "live-continuous-batching-loop-unit-20260705.json" in continuous["evidence"]
    assert "strict-live-cbkv-v16-outer-row-local-verified-20260706.json" in continuous["evidence"]
    assert payload["evidence"]["live_loop"]["claim_boundary"] == "live_continuous_decode_loop_unit_no_server_no_speedup"
    assert payload["evidence"]["live_loop"]["live_loop_unit_proven"] is True
    assert payload["evidence"]["live_loop"]["live_server_proven"] is False
    assert payload["evidence"]["live_loop"]["speedup_proven"] is False
    assert payload["evidence"]["live_loop"]["can_update_demo_status"] is False
    assert "remote_download" not in payload
    assert payload["high_value_fable_questions"]
    assert payload["do_not_spend_fable_tokens_on"]


def test_fable_handoff_check_markdown_names_fable_focus_not_grunt():
    proc = _run_checker()

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "# Fable handoff check — PASS" in proc.stdout
    assert "## Fable should focus on" in proc.stdout
    assert "## Fable should not burn tokens on" in proc.stdout
    assert "MVP status" in proc.stdout
    assert "live continuous-batching opt-in tick-row seam" in proc.stdout
    assert "Checking whether Instruct-2507 is still downloading" in proc.stdout


def test_fable_handoff_doc_points_to_checker_commands():
    text = (PROJECT_ROOT / "docs" / "fable-post-mvp-handover.md").read_text(encoding="utf-8")

    assert ".venv/bin/python scripts/fable_handoff_check.py" in text
    assert ".venv/bin/python scripts/fable_handoff_check.py --remote-download" in text
    assert ".venv/bin/python scripts/instruct2507_cache_readiness.py --remote" in text
    assert ".venv/bin/python scripts/instruct2507_full_generation_gate.py --remote-readiness" in text
    assert "scripts/extract_bloombee_multiaddr.py" in text
    assert "server_log_multiaddr_extraction_only_no_connectivity_proof" in text
    assert "cache_download_readiness_only_no_generation_or_load_proof" in text
    assert "instruct2507_full_generation_gate_plan_only_no_live_generation" in text
    assert "ready_to_attempt_demo_safe_ladder" in text
    assert "cache_generation_proof_harness_only_no_live_generation" in text
    assert "multi_request_load_harness_only_no_live_traffic" in text
    assert "Start with the grunt filter" in text



def test_fable_handoff_check_remote_report_includes_demo_safe_ladder_plan(monkeypatch):
    readiness = {
        "ok": True,
        "ready": False,
        "claim_boundary": "cache_download_readiness_only_no_generation_or_load_proof",
        "present_shard_count": 8,
        "expected_shard_count": 16,
        "first_missing_shard": "model-00009-of-00016.safetensors",
        "can_start_expensive_full_generation_gate": False,
        "errors": ["missing 8 expected shard(s)"],
    }
    monkeypatch.setattr(
        fable_handoff_check,
        "_remote_download_state",
        lambda errors: {"STATE": "downloading", "SHARD_COUNT": "8", "CURRENT_FILE": "model-00009-of-00016.safetensors"},
    )
    monkeypatch.setattr(fable_handoff_check, "_remote_cache_readiness", lambda errors: readiness)

    report = fable_handoff_check.build_report(include_remote=True)

    ladder = report["remote_demo_safe_ladder_plan"]
    assert ladder["ready_to_attempt_demo_safe_ladder"] is False
    assert ladder["demo_safe_ladder_gates"] == ["full_generation", "cache_generation", "multi_request_load"]
    assert ladder["claim_boundary"] == "instruct2507_full_generation_gate_plan_only_no_live_generation"
    assert ladder["cache_readiness"]["present_shard_count"] == 8
    assert ladder["cache_generation_plan"]["proof_gate"] == "cache_generation"
    assert ladder["multi_request_load_plan"]["proof_gate"] == "multi_request_load"
    assert ladder["generation_proven"] is False
    assert ladder["cache_generation_proven"] is False
    assert ladder["load_proven"] is False


def test_fable_handoff_check_remote_mode_is_explicit_opt_in():
    help_proc = _run_checker("--help")

    assert help_proc.returncode == 0
    assert "--remote-download" in help_proc.stdout
    assert "Also SSH to m4pro" in help_proc.stdout
