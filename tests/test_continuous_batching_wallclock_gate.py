from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _passed_parity_report() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "claim_boundary": "verified_live_continuous_batching_server_concurrent_arrival_parity",
        "proof_gate": "continuous_batching",
        "status": "passed",
        "request_count": 2,
        "late_arrival_observed": True,
        "batched_tick_count": 1,
        "token_parity_proven": True,
        "logits_fingerprint_parity_proven": True,
        "live_server_late_arrival_parity_proven": True,
        "live_server_proven": True,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
    }


def _valid_wallclock_payload() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "proof_gate": "continuous_batching",
        "claim_boundary": "live_continuous_batching_wallclock_capture_candidate",
        "parity_verification": _passed_parity_report(),
        "telemetry_tags": [
            "continuous_batching",
            "live_server_late_arrival_parity",
            "wallclock_throughput",
        ],
        "baseline": {
            "mode": "serial_baseline",
            "request_count": 2,
            "generated_token_count": 6,
            "total_seconds": 1.5,
        },
        "continuous": {
            "mode": "live_continuous_batching",
            "request_count": 2,
            "generated_token_count": 6,
            "total_seconds": 0.9,
        },
    }


def test_wallclock_gate_accepts_passed_parity_and_positive_speedup():
    from mvp_capabilities.continuous_batching_wallclock_gate import verify_continuous_batching_wallclock_payload

    result = verify_continuous_batching_wallclock_payload(_valid_wallclock_payload(), model_id=MODEL_ID)

    assert result["claim_boundary"] == "verified_live_continuous_batching_wallclock_speedup_after_parity"
    assert result["proof_gate"] == "continuous_batching"
    assert result["status"] == "passed"
    assert result["parity_proven"] is True
    assert result["wallclock_speedup_proven"] is True
    assert result["speedup_proven"] is True
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"continuous_batching_wallclock_speedup": "passed"}
    assert result["can_update_demo_status"] is False
    assert result["failed_checks"] == []
    assert result["evidence_summary"]["baseline_total_seconds"] == 1.5
    assert result["evidence_summary"]["continuous_total_seconds"] == 0.9
    assert result["evidence_summary"]["speedup_ratio"] == round(1.5 / 0.9, 12)
    assert result["evidence_summary"]["throughput_ratio"] == round((6 / 0.9) / (6 / 1.5), 12)


def test_wallclock_gate_rejects_positive_timing_without_live_parity():
    from mvp_capabilities.continuous_batching_wallclock_gate import verify_continuous_batching_wallclock_payload

    payload = _valid_wallclock_payload()
    payload["parity_verification"] = {
        **_passed_parity_report(),
        "status": "failed",
        "live_server_late_arrival_parity_proven": False,
        "failed_checks": ["no batched live-continuous tick observed"],
    }

    result = verify_continuous_batching_wallclock_payload(payload, model_id=MODEL_ID)

    assert result["status"] == "failed"
    assert result["parity_proven"] is False
    assert result["wallclock_speedup_proven"] is False
    assert result["speedup_proven"] is False
    assert result["can_update_proof_status"] is False
    assert result["proof_status_update"] == {}
    assert "live-server parity proof did not pass" in result["failed_checks"]


def test_wallclock_gate_rejects_request_or_token_count_drift():
    from mvp_capabilities.continuous_batching_wallclock_gate import verify_continuous_batching_wallclock_payload

    payload = _valid_wallclock_payload()
    payload["continuous"] = {
        "mode": "live_continuous_batching",
        "request_count": 2,
        "generated_token_count": 5,
        "total_seconds": 0.7,
    }

    result = verify_continuous_batching_wallclock_payload(payload, model_id=MODEL_ID)

    assert result["status"] == "failed"
    assert result["wallclock_speedup_proven"] is False
    assert "generated_token_count differs between baseline and continuous paths" in result["failed_checks"]


def test_wallclock_plan_and_cli_verify_are_claim_bounded(tmp_path: Path):
    from mvp_capabilities.continuous_batching_wallclock_gate import build_continuous_batching_wallclock_plan

    evidence_path = tmp_path / "wallclock.json"
    verify_out = tmp_path / "wallclock.verify.json"
    plan = build_continuous_batching_wallclock_plan(
        model_id=MODEL_ID,
        evidence_path=str(evidence_path),
        parity_report_path=".local/continuous-parity.verify.json",
    )

    assert plan["claim_boundary"] == "live_continuous_batching_wallclock_gate_harness_no_measurement"
    assert plan["proof_gate"] == "continuous_batching"
    assert plan["wallclock_speedup_proven"] is False
    assert plan["can_update_demo_status"] is False
    assert "continuous_batching_live_server_proof.py verify" in plan["operator_commands"][0]
    assert "continuous_batching_wallclock_gate.py verify" in plan["verify_command"]

    evidence_path.write_text(json.dumps(_valid_wallclock_payload()), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/continuous_batching_wallclock_gate.py",
            "verify",
            "--model",
            MODEL_ID,
            "--evidence",
            str(evidence_path),
            "--out",
            str(verify_out),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(verify_out.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["can_update_demo_status"] is False
