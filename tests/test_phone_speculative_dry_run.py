#!/usr/bin/env python3
"""Dry-run integrated trial gate test."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phone_speculative_dry_run_exercises_all_gates_with_existing_manifest():
    """The dry-run harness exercises every code check even without live phones."""
    from mvp_capabilities import phone_speculative_dry_run

    manifest = (
        PROJECT_ROOT
        / "mvp_capabilities"
        / "distributed_evidence"
        / "phone"
        / "multi-phone-speculative-readiness-one-phone-20260705T214620Z.json"
    )
    if not manifest.exists():
        # Allow running in envs where phone evidence may not be checked out
        import pytest
        pytest.skip("phone readiness manifest not available")

    report = phone_speculative_dry_run.build_dry_run_report(
        manifest,
        synthetic_measurement=True,
    )
    assert report["ok"] is True
    assert report["dry_run_complete"] is True
    assert report["speedup_proven"] is False
    assert report["bloombee_block_serving_proven"] is False
    assert report["gates_exercised"] == 4
    assert report["gates_passed"] == 4

    gate_names = {gate["gate"] for gate in report["gates"]}
    assert gate_names == {
        "readiness_manifest_load",
        "integrated_trial_plan",
        "integrated_trial_verify_synthetic",
        "synthetic_measurement_generation",
    }

    # Verify gate exercised code but data conditions not met (expected with 1 phone)
    verify_gate = next(g for g in report["gates"] if g["gate"] == "integrated_trial_verify_synthetic")
    assert verify_gate["code_path_exercised"] is True
    assert verify_gate["data_conditions_met"] is False  # 1 phone, no speedup
