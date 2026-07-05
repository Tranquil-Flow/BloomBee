import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"


def _ready_phone(phone_id: str, *, ready: bool = True) -> dict:
    return {
        "phone_id": phone_id,
        "phone_model": "Pixel 8 Pro",
        "runtime": "termux",
        "transport_path": "m4pro_usb_adb_pull_termux_json_then_local_forced_batch_verifier",
        "model_sha256": MODEL_SHA,
        "termux_context_token_artifact": f".local/phone/{phone_id}-termux-context-token-ids.json",
        "accepted_external_token_count": 8 if ready else 0,
        "proposed_external_token_count": 8,
        "context_token_ingestion_proven": ready,
        "wallclock_gate_present": ready,
        "wallclock_correctness_proven": ready,
        "speedup_claimed_by_artifact": False,
        "ready_for_trial": ready,
        "blocked_reasons": [] if ready else [f"phone:{phone_id}:context_token_ingestion_not_proven"],
    }


def _manifest(*, ready: bool = True) -> dict:
    phones = [_ready_phone("pixel-a", ready=ready), _ready_phone("pixel-b", ready=ready), _ready_phone("pixel-c", ready=ready)]
    return {
        "claim_boundary": "multi_phone_speculative_readiness_manifest_no_speedup_claim",
        "verification_status": "passed" if ready else "failed",
        "trial_ready": ready,
        "phone_count": 3,
        "ready_phone_count": 3 if ready else 0,
        "unique_phone_ids": ["pixel-a", "pixel-b", "pixel-c"],
        "model_sha256": MODEL_SHA,
        "phones": phones,
        "all_context_token_ingestion_proven": ready,
        "all_wallclock_gates_present": ready,
        "all_wallclock_correctness_proven": ready,
        "speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "blocked_reasons": [] if ready else ["phone:pixel-a:context_token_ingestion_not_proven"],
    }


def test_phone_speculative_trial_plan_turns_readiness_manifest_into_operator_commands():
    from mvp_capabilities.phone_speculative_trial_plan import build_phone_speculative_trial_plan

    plan = build_phone_speculative_trial_plan(_manifest(), output_dir=".local/phone/trial-20260706")

    assert plan["claim_boundary"] == "phone_speculative_integrated_trial_plan_no_speedup_claim"
    assert plan["plan_status"] == "ready_for_integrated_trial"
    assert plan["selected_phone_ids"] == ["pixel-a", "pixel-b", "pixel-c"]
    assert plan["phone_count"] == 3
    assert plan["model_sha256"] == MODEL_SHA
    assert plan["speedup_proven"] is False
    assert plan["can_update_speculative_speedup_status"] is False
    assert plan["candidate_timing_kind"] == "operator_measured_integrated_draft_plus_verifier_required"
    assert len(plan["per_phone_commands"]) == 3
    first_commands = "\n".join(plan["per_phone_commands"][0]["commands"])
    assert "phone_llama_cpp_binding_verifier.py" in first_commands
    assert "--phone-context-token-ids" in first_commands
    assert "phone_speculative_wallclock_gate.py" in first_commands
    assert "--measured-draft-plus-verifier-elapsed-s $INTEGRATED_DRAFT_PLUS_VERIFIER_ELAPSED_S" in first_commands
    assert any("compare verifier-only vs integrated draft-plus-verifier" in step for step in plan["operator_sequence"])


def test_phone_speculative_trial_plan_fails_closed_for_unready_manifest():
    from mvp_capabilities.phone_speculative_trial_plan import build_phone_speculative_trial_plan

    plan = build_phone_speculative_trial_plan(_manifest(ready=False), output_dir=".local/phone/trial-20260706")

    assert plan["plan_status"] == "blocked_by_readiness_manifest"
    assert plan["selected_phone_ids"] == []
    assert plan["speedup_proven"] is False
    assert plan["can_update_speculative_speedup_status"] is False
    assert "readiness_manifest_not_passed" in plan["blocked_reasons"]
    assert plan["per_phone_commands"] == []


def test_phone_speculative_trial_plan_cli_reads_manifest_and_writes_json(tmp_path: Path):
    manifest_path = tmp_path / "readiness.json"
    out_path = tmp_path / "trial-plan.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_speculative_trial_plan.py",
            "--readiness-manifest",
            str(manifest_path),
            "--output-dir",
            ".local/phone/trial-20260706",
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
    assert payload["plan_status"] == "ready_for_integrated_trial"
    assert payload["selected_phone_ids"] == ["pixel-a", "pixel-b", "pixel-c"]
