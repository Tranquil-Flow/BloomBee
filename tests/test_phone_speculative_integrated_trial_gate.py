from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"


def _ready_phone(phone_id: str) -> dict[str, object]:
    return {
        "phone_id": phone_id,
        "phone_model": "Pixel 8 Pro",
        "runtime": "termux-llama.cpp",
        "transport_path": "m4pro_usb_adb_pull_termux_json_then_local_forced_batch_verifier",
        "model_sha256": MODEL_SHA,
        "accepted_external_token_count": 8,
        "proposed_external_token_count": 8,
        "context_token_ingestion_proven": True,
        "wallclock_gate_present": True,
        "wallclock_correctness_proven": True,
        "speedup_claimed_by_artifact": False,
        "ready_for_trial": True,
        "blocked_reasons": [],
    }


def _manifest(*, ready: bool = True, phone_count: int = 3) -> dict[str, object]:
    phones = [_ready_phone(f"pixel-{idx}") for idx in range(phone_count)] if ready else []
    return {
        "claim_boundary": "multi_phone_speculative_readiness_manifest_no_speedup_claim",
        "verification_status": "passed" if ready else "failed",
        "trial_ready": ready,
        "min_phone_count": 3,
        "max_phone_count": 4,
        "phone_count": phone_count if ready else 1,
        "ready_phone_count": len(phones),
        "unique_phone_ids": [phone["phone_id"] for phone in phones],
        "model_sha256": MODEL_SHA if ready else None,
        "phones": phones,
        "all_context_token_ingestion_proven": ready,
        "all_wallclock_gates_present": ready,
        "all_wallclock_correctness_proven": ready,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": [] if ready else ["phone_count_below_min:1<3"],
    }


def test_integrated_trial_gate_accepts_passed_readiness_and_faster_integrated_timing():
    from mvp_capabilities.phone_speculative_integrated_trial_gate import (
        build_phone_speculative_integrated_trial_gate,
    )

    report = build_phone_speculative_integrated_trial_gate(
        readiness_manifest=_manifest(),
        verifier_only_elapsed_s=1.8,
        integrated_draft_plus_verifier_elapsed_s=1.2,
        source_artifacts=["multi-phone-readiness.json", "integrated-trial-log.json"],
    )

    assert report["claim_boundary"] == "phone_speculative_integrated_trial_gate_verified_measurement"
    assert report["status"] == "passed"
    assert report["readiness_passed"] is True
    assert report["phone_count"] == 3
    assert report["selected_phone_ids"] == ["pixel-0", "pixel-1", "pixel-2"]
    assert report["measurement_kind"] == "measured_integrated_non_sequential_draft_plus_verifier"
    assert report["verifier_only_elapsed_s"] == 1.8
    assert report["integrated_draft_plus_verifier_elapsed_s"] == 1.2
    assert report["speedup_ratio"] == 1.5
    assert report["speedup_proven"] is True
    assert report["wallclock_speedup_proven"] is True
    assert report["can_update_speculative_speedup_status"] is True
    assert report["proof_status_update"] == {"phone_speculative_integrated_speedup": "passed"}
    assert report["bloombee_block_serving_proven"] is False
    assert report["can_update_phone_worker_status"] is False
    assert report["failed_checks"] == []


def test_integrated_trial_gate_rejects_sequential_or_slow_measurements():
    from mvp_capabilities.phone_speculative_integrated_trial_gate import (
        build_phone_speculative_integrated_trial_gate,
    )

    sequential = build_phone_speculative_integrated_trial_gate(
        readiness_manifest=_manifest(),
        verifier_only_elapsed_s=1.8,
        integrated_draft_plus_verifier_elapsed_s=1.2,
        measurement_kind="sequential_phone_draft_then_verifier",
    )
    assert sequential["status"] == "failed"
    assert sequential["speedup_proven"] is False
    assert "measurement_kind_not_integrated_non_sequential" in sequential["failed_checks"]

    slow = build_phone_speculative_integrated_trial_gate(
        readiness_manifest=_manifest(),
        verifier_only_elapsed_s=1.8,
        integrated_draft_plus_verifier_elapsed_s=2.4,
    )
    assert slow["status"] == "failed"
    assert slow["speedup_proven"] is False
    assert "integrated_draft_plus_verifier_not_faster_than_verifier_only" in slow["failed_checks"]
    assert slow["can_update_speculative_speedup_status"] is False


def test_integrated_trial_gate_fails_closed_until_readiness_manifest_passes():
    from mvp_capabilities.phone_speculative_integrated_trial_gate import (
        build_phone_speculative_integrated_trial_gate,
        build_phone_speculative_integrated_trial_gate_plan,
    )

    report = build_phone_speculative_integrated_trial_gate(
        readiness_manifest=_manifest(ready=False),
        verifier_only_elapsed_s=1.8,
        integrated_draft_plus_verifier_elapsed_s=1.2,
    )

    assert report["status"] == "failed"
    assert report["readiness_passed"] is False
    assert report["selected_phone_ids"] == []
    assert report["speedup_proven"] is False
    assert "readiness_manifest_not_passed" in report["failed_checks"]
    assert "phone_count_below_min:1<3" in report["failed_checks"]

    plan = build_phone_speculative_integrated_trial_gate_plan(
        readiness_manifest=_manifest(ready=False),
        readiness_manifest_path="multi-phone-readiness.json",
        evidence_path=".local/phone/integrated-trial-gate.json",
    )
    assert plan["claim_boundary"] == "phone_speculative_integrated_trial_gate_harness_no_measurement"
    assert plan["plan_status"] == "blocked_by_readiness_manifest"
    assert plan["speedup_proven"] is False
    assert plan["can_update_speculative_speedup_status"] is False
    assert "phone_speculative_integrated_trial_gate.py verify" in plan["verify_command"]


def test_integrated_trial_gate_cli_plan_and_verify_write_json(tmp_path: Path):
    manifest_path = tmp_path / "readiness.json"
    plan_path = tmp_path / "plan.json"
    verify_path = tmp_path / "verify.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    plan_proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_speculative_integrated_trial_gate.py",
            "plan",
            "--readiness-manifest",
            str(manifest_path),
            "--evidence",
            str(verify_path),
            "--out",
            str(plan_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert plan_proc.returncode == 0, plan_proc.stderr
    plan = json.loads(plan_proc.stdout)
    assert plan == json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["plan_status"] == "ready_for_integrated_measurement"
    assert plan["speedup_proven"] is False

    verify_proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_speculative_integrated_trial_gate.py",
            "verify",
            "--readiness-manifest",
            str(manifest_path),
            "--verifier-only-elapsed-s",
            "1.8",
            "--integrated-draft-plus-verifier-elapsed-s",
            "1.2",
            "--out",
            str(verify_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert verify_proc.returncode == 0, verify_proc.stderr
    verified = json.loads(verify_proc.stdout)
    assert verified == json.loads(verify_path.read_text(encoding="utf-8"))
    assert verified["status"] == "passed"
    assert verified["speedup_proven"] is True
