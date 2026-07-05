import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_remaining_work_checklist_json_is_machine_readable_and_claim_bounded():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/remaining_work_checklist.py", "--json"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "status_derived_remaining_work_no_new_proof"
    assert payload["source"] == "mvp_capabilities.mvp_status.build_status_report"
    assert payload["core_mvp_complete"] is True
    assert payload["mvp_bar"] == "████████████████████ 100%"
    assert payload["remaining_count"] == len(payload["items"])
    assert payload["remaining_count"] == 6
    by_id = {item["id"]: item for item in payload["items"]}
    assert set(by_id) == {
        "qwen35b_candidate",
        "minimax_m3_candidate",
        "speculative_decode",
        "phone_worker",
        "continuous_batching",
        "kv_prefix_reuse",
    }
    assert by_id["qwen35b_candidate"]["status"] == "partial"
    assert "one-block server proof" in by_id["qwen35b_candidate"]["next_step"]
    assert by_id["minimax_m3_candidate"]["status"] == "blocked"
    assert by_id["minimax_m3_candidate"]["blocked"] is True
    assert all(item["done"] is False for item in payload["items"])
    assert payload["by_status"] == {"partial": 5, "blocked": 1}


def test_remaining_work_checklist_markdown_lists_next_steps_without_overclaiming():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/remaining_work_checklist.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert "# Remaining work checklist" in text
    assert "status_derived_remaining_work_no_new_proof" in text
    assert "- [ ] `qwen35b_candidate`" in text
    assert "one-block server proof" in text
    assert "- [ ] `minimax_m3_candidate`" in text
    assert "No new proof is created by this checklist" in text
    assert "speedup proven" not in text.lower()
