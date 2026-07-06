import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHONE_DIR = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/phone"


def _read(name: str) -> dict:
    return json.loads((PHONE_DIR / name).read_text(encoding="utf-8"))


def test_spec_phone_wrapup_aggregates_current_fail_closed_gates_without_promotion():
    from mvp_capabilities.speculative_phone_worker_wrapup import build_speculative_phone_worker_wrapup_report

    report = build_speculative_phone_worker_wrapup_report(
        cross_platform_readiness=_read("phone-cross-platform-readiness-android-ios-20260706.json"),
        multi_phone_readiness=_read("multi-phone-speculative-readiness-one-phone-20260705T214620Z.json"),
        integrated_trial_plan=_read("phone-speculative-integrated-trial-gate-harness-20260706.json"),
        wallclock_gate=_read("termux-same-gguf-wallclock-gate-20260704T112500Z.json"),
        adb_preflight=_read("phone-adb-multiphone-preflight-20260706.json"),
        source_artifacts=[
            "phone-cross-platform-readiness-android-ios-20260706.json",
            "multi-phone-speculative-readiness-one-phone-20260705T214620Z.json",
            "phone-speculative-integrated-trial-gate-harness-20260706.json",
            "termux-same-gguf-wallclock-gate-20260704T112500Z.json",
            "phone-adb-multiphone-preflight-20260706.json",
        ],
    )

    assert report["claim_boundary"] == "speculative_phone_worker_wrapup_fail_closed_no_speedup_claim"
    assert report["status"] == "blocked_by_missing_ios_and_multiphone_integrated_speedup"
    assert report["android_ready"] is True
    assert report["ios_ready"] is False
    assert report["cross_platform_ready"] is False
    assert report["phone_count"] == 1
    assert report["ready_phone_count"] == 1
    assert report["multi_phone_trial_ready"] is False
    assert report["integrated_trial_plan_ready"] is False
    assert report["wallclock_speedup_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_speculative_speedup_status"] is False
    assert report["can_update_phone_worker_status"] is False
    assert report["bloombee_block_serving_proven"] is False
    assert report["completed_groundwork"] == [
        "android_phone_context_token_ingestion_ready",
        "single_phone_wallclock_correctness_gate_recorded",
        "integrated_trial_gate_harness_exists",
    ]
    assert "missing_required_platform:ios" in report["remaining_blockers"]
    assert "phone_count_below_min:1<3" in report["remaining_blockers"]
    assert "readiness_manifest_not_passed" in report["remaining_blockers"]
    assert "sequential_draft_plus_verifier_not_faster_than_verifier_only" in report["remaining_blockers"]
    assert report["operator_next_steps"] == [
        "collect at least one iOS artifact with the same context-token and wall-clock correctness schema",
        "collect 3-4 distinct ready phone artifacts and rerun multi_phone_speculative_readiness.py",
        "run the integrated non-sequential phone draft-plus-verifier harness instead of sequential draft+verifier timing",
        "keep phone BloomBee block-worker status false until separate block-serving proof passes",
    ]


def test_spec_phone_wrapup_cli_writes_json(tmp_path: Path):
    out_path = tmp_path / "wrapup.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/speculative_phone_worker_wrapup.py",
            "--cross-platform-readiness",
            str(PHONE_DIR / "phone-cross-platform-readiness-android-ios-20260706.json"),
            "--multi-phone-readiness",
            str(PHONE_DIR / "multi-phone-speculative-readiness-one-phone-20260705T214620Z.json"),
            "--integrated-trial-plan",
            str(PHONE_DIR / "phone-speculative-integrated-trial-gate-harness-20260706.json"),
            "--wallclock-gate",
            str(PHONE_DIR / "termux-same-gguf-wallclock-gate-20260704T112500Z.json"),
            "--adb-preflight",
            str(PHONE_DIR / "phone-adb-multiphone-preflight-20260706.json"),
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
    assert payload["status"] == "blocked_by_missing_ios_and_multiphone_integrated_speedup"
    assert payload["source_artifacts"][0].endswith("phone-cross-platform-readiness-android-ios-20260706.json")
