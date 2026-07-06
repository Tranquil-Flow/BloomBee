from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _valid_capture() -> dict:
    return {
        "model_id": MODEL_ID,
        "proof_gate": "continuous_batching",
        "opt_in_flag": "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING",
        "opt_in_enabled": True,
        "server_observed_live_continuous_batches": True,
        "live_server_proven": True,
        "speedup_proven": False,
        "requests": [
            {
                "request_id": "req-a",
                "arrival_tick": 0,
                "baseline": {"generated_token_ids": [10, 11], "logits_sha256": "aa"},
                "continuous": {"generated_token_ids": [10, 11], "logits_sha256": "aa"},
            },
            {
                "request_id": "req-b",
                "arrival_tick": 1,
                "baseline": {"generated_token_ids": [20], "logits_sha256": "bb"},
                "continuous": {"generated_token_ids": [20], "logits_sha256": "bb"},
            },
        ],
        "live_continuous_report": {
            "tick_batches": [
                {"tick": 0, "request_ids": ["req-a"], "output_token_ids": [10]},
                {"tick": 1, "request_ids": ["req-a", "req-b"], "output_token_ids": [11, 20]},
            ]
        },
    }


def test_live_server_continuous_batching_verifier_accepts_late_arrival_token_parity():
    from mvp_capabilities.continuous_batching_live_server_proof import verify_live_server_continuous_batching_payload

    result = verify_live_server_continuous_batching_payload(_valid_capture(), model_id=MODEL_ID)

    assert result["claim_boundary"] == "verified_live_continuous_batching_server_concurrent_arrival_parity"
    assert result["proof_gate"] == "continuous_batching"
    assert result["status"] == "passed"
    assert result["request_count"] == 2
    assert result["batched_tick_count"] == 1
    assert result["late_arrival_observed"] is True
    assert result["token_parity_proven"] is True
    assert result["logits_fingerprint_parity_proven"] is True
    assert result["logits_numeric_parity_proven"] is False
    assert result["logits_parity_proven"] is True
    assert result["live_server_late_arrival_parity_proven"] is True
    assert result["speedup_proven"] is False
    assert result["can_update_demo_status"] is False
    assert result["can_update_proof_status"] is False
    assert result["failed_checks"] == []


def test_live_server_continuous_batching_verifier_accepts_bounded_numeric_logit_drift():
    from mvp_capabilities.continuous_batching_live_server_proof import verify_live_server_continuous_batching_payload

    capture = _valid_capture()
    for row in capture["requests"]:
        row["baseline"]["logits_sha256"] = "baseline-" + row["request_id"]
        row["continuous"]["logits_sha256"] = "continuous-" + row["request_id"]
        row["logits_numeric_comparison"] = {
            "max_abs_diff": 0.001,
            "mean_abs_diff": 0.0001,
            "argmax_token_id_match": True,
            "top1_token_id_match": True,
        }

    result = verify_live_server_continuous_batching_payload(capture, model_id=MODEL_ID)

    assert result["status"] == "passed"
    assert result["logits_fingerprint_parity_proven"] is False
    assert result["logits_numeric_parity_proven"] is True
    assert result["logits_parity_proven"] is True
    assert result["failed_checks"] == []


def test_live_server_continuous_batching_verifier_rejects_unbounded_numeric_logit_drift():
    from mvp_capabilities.continuous_batching_live_server_proof import verify_live_server_continuous_batching_payload

    capture = _valid_capture()
    capture["requests"][0]["baseline"]["logits_sha256"] = "baseline-a"
    capture["requests"][0]["continuous"]["logits_sha256"] = "continuous-a"
    capture["requests"][0]["logits_numeric_comparison"] = {
        "max_abs_diff": 0.25,
        "mean_abs_diff": 0.02,
        "argmax_token_id_match": True,
        "top1_token_id_match": True,
    }

    result = verify_live_server_continuous_batching_payload(capture, model_id=MODEL_ID)

    assert result["status"] == "failed"
    assert "request req-a logits numeric max_abs_diff exceeds tolerance" in result["failed_checks"]
    assert result["logits_parity_proven"] is False


def test_live_server_continuous_batching_verifier_rejects_missing_late_arrival_and_token_mismatch():
    from mvp_capabilities.continuous_batching_live_server_proof import verify_live_server_continuous_batching_payload

    missing_late = _valid_capture()
    for row in missing_late["requests"]:
        row["arrival_tick"] = 0
    missing_late["live_continuous_report"]["tick_batches"] = [
        {"tick": 0, "request_ids": ["req-a", "req-b"], "output_token_ids": [10, 20]}
    ]
    result = verify_live_server_continuous_batching_payload(missing_late, model_id=MODEL_ID)
    assert result["status"] == "failed"
    assert "no late-arrival request observed" in result["failed_checks"]
    assert result["live_server_late_arrival_parity_proven"] is False

    mismatch = _valid_capture()
    mismatch["requests"][1]["continuous"]["generated_token_ids"] = [21]
    result = verify_live_server_continuous_batching_payload(mismatch, model_id=MODEL_ID)
    assert result["status"] == "failed"
    assert "request req-b generated token IDs differ from baseline" in result["failed_checks"]
    assert result["token_parity_proven"] is False


def test_live_server_continuous_batching_verifier_rejects_same_arrival_server_ticks_masquerading_as_late_arrival():
    from mvp_capabilities.continuous_batching_live_server_proof import verify_live_server_continuous_batching_payload

    disguised_same_arrival = _valid_capture()
    # Client-side rows alone can claim req-b arrived at tick 1. A same-arrival
    # server trace for req-b at tick 0 must not satisfy the stricter live/late
    # server gate; otherwise a same-arrival capture can be mislabeled as a
    # late-arrival continuous-batching proof.
    disguised_same_arrival["live_continuous_report"]["tick_batches"] = [
        {"tick": 0, "request_ids": ["req-a", "req-b"], "output_token_ids": [10, 20]},
        {"tick": 1, "request_ids": ["req-a", "req-b"], "output_token_ids": [11, 21]},
    ]

    result = verify_live_server_continuous_batching_payload(disguised_same_arrival, model_id=MODEL_ID)

    assert result["status"] == "failed"
    assert "server observed request req-b before declared arrival tick" in result["failed_checks"]
    assert result["live_server_late_arrival_parity_proven"] is False


    # Full-slot late-arrival batches may include inactive logical slots. The
    # active_mask must prevent an inactive row from being counted as an early
    # server observation.
    full_slot_late_arrival = _valid_capture()
    full_slot_late_arrival["live_continuous_report"]["tick_batches"] = [
        {"tick": 0, "request_ids": ["req-a", "req-b"], "active_mask": [True, False], "output_token_ids": [10, 20]},
        {"tick": 1, "request_ids": ["req-a", "req-b"], "active_mask": [True, True], "output_token_ids": [11, 20]},
    ]
    result = verify_live_server_continuous_batching_payload(full_slot_late_arrival, model_id=MODEL_ID)
    assert result["status"] == "passed"
    assert result["batched_tick_count"] == 1


def test_live_server_continuous_batching_plan_and_cli_verify(tmp_path: Path):
    from mvp_capabilities.continuous_batching_live_server_proof import build_live_server_continuous_batching_plan

    plan = build_live_server_continuous_batching_plan(model_id=MODEL_ID, evidence_path=".local/continuous.json")
    assert plan["claim_boundary"] == "live_continuous_batching_server_proof_harness_only_no_live_traffic"
    assert plan["proof_gate"] == "continuous_batching"
    assert plan["can_update_demo_status"] is False
    assert "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING=1" in plan["operator_commands"][0]
    assert "continuous_batching_live_server_proof.py verify" in plan["verify_command"]

    evidence = tmp_path / "capture.json"
    out = tmp_path / "verify.json"
    evidence.write_text(json.dumps(_valid_capture()), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/continuous_batching_live_server_proof.py",
            "verify",
            "--model",
            MODEL_ID,
            "--evidence",
            str(evidence),
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
