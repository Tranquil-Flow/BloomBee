import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _continuous_report() -> dict:
    return {
        "model_id": MODEL_ID,
        "claim_boundary": "verified_live_continuous_batching_server_concurrent_arrival_parity",
        "proof_gate": "continuous_batching",
        "status": "passed",
        "late_arrival_observed": True,
        "batched_tick_count": 1,
        "token_parity_proven": True,
        "logits_fingerprint_parity_proven": True,
        "logits_numeric_parity_proven": False,
        "logits_parity_proven": True,
        "live_server_late_arrival_parity_proven": True,
        "live_server_proven": True,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_demo_status": False,
        "failed_checks": [],
    }


def _kv_report(*, live_reuse: bool = True, status: str = "passed") -> dict:
    return {
        "model_id": MODEL_ID,
        "claim_boundary": "verified_kv_prefix_reuse_same_prefix_varied_suffix_parity_timing",
        "proof_gate": "kv_prefix_reuse",
        "status": status,
        "same_prefix_varied_suffix_proven": True,
        "token_parity_proven": True,
        "logit_parity_proven": True,
        "timing_measured": True,
        "speedup_proven": True,
        "live_kv_cache_reuse_proven": live_reuse,
        "server_observed_kv_cache_reuse": live_reuse,
        "can_update_mvp_status": False,
        "failed_checks": [] if status == "passed" else ["synthetic failure"],
    }


def test_joint_gate_passes_only_when_late_arrival_batching_and_live_kv_reuse_are_both_proven():
    from mvp_capabilities.continuous_kv_joint_readiness import build_continuous_kv_joint_readiness_report

    report = build_continuous_kv_joint_readiness_report(
        continuous_report=_continuous_report(),
        kv_report=_kv_report(live_reuse=True),
    )

    assert report["claim_boundary"] == "continuous_kv_joint_readiness_gate_no_demo_or_wallclock_speedup_claim"
    assert report["model_id"] == MODEL_ID
    assert report["verification_status"] == "passed"
    assert report["continuous_batching_ready"] is True
    assert report["kv_prefix_reuse_ready"] is True
    assert report["joint_batch_kv_ready"] is True
    assert report["can_update_post_mvp_gate_status"] is True
    assert report["can_update_demo_status"] is False
    assert report["wallclock_speedup_proven"] is False
    assert report["proof_status_update"] == {
        "continuous_batching": "passed",
        "kv_prefix_reuse": "passed",
        "continuous_kv_joint_readiness": "passed",
    }


def test_joint_gate_accepts_numeric_logit_parity_when_fingerprints_differ():
    from mvp_capabilities.continuous_kv_joint_readiness import build_continuous_kv_joint_readiness_report

    continuous = _continuous_report()
    continuous["logits_fingerprint_parity_proven"] = False
    continuous["logits_numeric_parity_proven"] = True
    continuous["logits_parity_proven"] = True
    report = build_continuous_kv_joint_readiness_report(
        continuous_report=continuous,
        kv_report=_kv_report(live_reuse=True),
    )

    assert report["verification_status"] == "passed"
    assert report["continuous_batching_ready"] is True
    assert report["continuous_batching"]["logits_parity_proven"] is True


def test_joint_gate_rejects_metadata_only_kv_even_if_kv_verifier_status_passed():
    from mvp_capabilities.continuous_kv_joint_readiness import build_continuous_kv_joint_readiness_report

    report = build_continuous_kv_joint_readiness_report(
        continuous_report=_continuous_report(),
        kv_report=_kv_report(live_reuse=False),
    )

    assert report["verification_status"] == "failed"
    assert report["joint_batch_kv_ready"] is False
    assert report["can_update_post_mvp_gate_status"] is False
    assert "kv_prefix_reuse:live_server_kv_tensor_reuse_not_proven" in report["blocked_reasons"]


def test_joint_gate_rejects_same_arrival_or_failed_continuous_report():
    from mvp_capabilities.continuous_kv_joint_readiness import build_continuous_kv_joint_readiness_report

    continuous = _continuous_report()
    continuous["live_server_late_arrival_parity_proven"] = False
    continuous["late_arrival_observed"] = False
    continuous["status"] = "failed"
    continuous["failed_checks"] = ["no late-arrival request observed"]

    report = build_continuous_kv_joint_readiness_report(
        continuous_report=continuous,
        kv_report=_kv_report(live_reuse=True),
    )

    assert report["verification_status"] == "failed"
    assert report["continuous_batching_ready"] is False
    assert "continuous_batching:late_arrival_parity_not_proven" in report["blocked_reasons"]
    assert "continuous_batching:no late-arrival request observed" in report["blocked_reasons"]


def test_joint_gate_cli_writes_json(tmp_path: Path):
    continuous = tmp_path / "continuous.json"
    kv = tmp_path / "kv.json"
    out = tmp_path / "joint.json"
    continuous.write_text(json.dumps(_continuous_report()), encoding="utf-8")
    kv.write_text(json.dumps(_kv_report(live_reuse=True)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/continuous_kv_joint_readiness.py",
            "--continuous-report",
            str(continuous),
            "--kv-report",
            str(kv),
            "--out",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out.read_text(encoding="utf-8"))
    assert payload["joint_batch_kv_ready"] is True
