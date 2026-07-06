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
    assert payload["remaining_count"] == 4
    by_id = {item["id"]: item for item in payload["items"]}
    assert set(by_id) == {
        "qwen35b_candidate",
        "minimax_m3_candidate",
        "speculative_decode",
        "phone_worker",
    }
    assert by_id["qwen35b_candidate"]["status"] == "partial"
    assert "one-block server proof" in by_id["qwen35b_candidate"]["next_step"]
    assert by_id["minimax_m3_candidate"]["status"] == "blocked"
    assert by_id["minimax_m3_candidate"]["blocked"] is True
    minimax_evidence = by_id["minimax_m3_candidate"]["evidence"]
    assert "minimax-reap-family-comparison-current-20260706.json" in minimax_evidence
    assert "M4+M4Pro memory is not additive" in minimax_evidence
    assert "M3 as likely stronger but not easier" in minimax_evidence
    assert all(item["done"] is False for item in payload["items"])
    assert payload["by_status"] == {"partial": 3, "blocked": 1}
    assert payload["all_remaining_require_human_or_hardware"] is True
    assert payload["by_blocker_category"] == {
        "hardware_memory": 1,
        "hardware_memory_or_real_model_proof": 1,
        "human_operator_devices": 2,
    }
    assert all(item["requires_human_or_hardware"] is True for item in payload["items"])
    assert by_id["qwen35b_candidate"]["blocker_category"] == "hardware_memory"
    assert "one_block_proof_fits_m4pro_48gb" in by_id["qwen35b_candidate"]["blocker_reasons"][0]
    assert "full_distributed_needs" in by_id["qwen35b_candidate"]["blocker_reasons"][1]
    assert "full_distributed_" in by_id["qwen35b_candidate"]["blocker_reasons"][2]
    assert "needs_real_one_block_server_proof" in by_id["qwen35b_candidate"]["blocker_reasons"][3]
    assert by_id["minimax_m3_candidate"]["blocker_category"] == "hardware_memory_or_real_model_proof"
    minimax_reasons = by_id["minimax_m3_candidate"]["blocker_reasons"]
    assert any("one_block_proof_fits_m4pro_48gb" in r for r in minimax_reasons)
    assert any("full_distributed_needs" in r for r in minimax_reasons)
    assert any("requires_real_weight_or_full_mtp_module_proof" in r for r in minimax_reasons)
    assert by_id["speculative_decode"]["blocker_category"] == "human_operator_devices"
    assert "requires_ios_artifact" in by_id["speculative_decode"]["blocker_reasons"]
    assert "requires_three_or_more_ready_phones" in by_id["phone_worker"]["blocker_reasons"]


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
    assert "All remaining items require human/operator hardware or suitable-memory proof gates" in text
    assert "speedup proven" not in text.lower()
