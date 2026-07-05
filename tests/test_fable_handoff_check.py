from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
        "complete": 10,
        "partial": 6,
        "pending": 0,
        "blocked": 1,
        "total": 17,
    }
    assert payload["mvp_status"]["post_mvp_task_summary"] == {
        "complete": 1,
        "partial": 6,
        "pending": 0,
        "blocked": 1,
        "total": 8,
    }
    continuous = payload["mvp_status"]["continuous_batching"]
    assert continuous["status"] == "partial"
    assert "live-continuous-batching-loop-unit-20260705.json" in continuous["evidence"]
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
    assert "live continuous-batching opt-in seam" in proc.stdout
    assert "Rediscovering the Instruct-2507 download failure mode" in proc.stdout


def test_fable_handoff_doc_points_to_checker_commands():
    text = (PROJECT_ROOT / "docs" / "fable-post-mvp-handover.md").read_text(encoding="utf-8")

    assert ".venv/bin/python scripts/fable_handoff_check.py" in text
    assert ".venv/bin/python scripts/fable_handoff_check.py --remote-download" in text
    assert "Start with the grunt filter" in text



def test_fable_handoff_check_remote_mode_is_explicit_opt_in():
    help_proc = _run_checker("--help")

    assert help_proc.returncode == 0
    assert "--remote-download" in help_proc.stdout
    assert "Also SSH to m4pro" in help_proc.stdout
