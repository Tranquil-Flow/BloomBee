from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _capture(prompt: str, input_ids: list[int], generated: list[int], *, seconds: float) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "mode": "generate-api",
        "server_maddrs": ["/ip4/127.0.0.1/tcp/31337/p2p/test"],
        "input_ids": input_ids,
        "distributed_ids": [*input_ids, *generated],
        "distributed_text": prompt + " done",
        "distributed_seconds": seconds,
        "distributed_top5": [
            {"token_id": generated[0], "logit": 9.0},
            {"token_id": 2, "logit": 1.0},
        ],
    }


def _server_observed_report() -> dict[str, object]:
    return {
        "source": "bloombee.server.live_continuous_batching_capture",
        "opt_in_flag": "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING",
        "opt_in_enabled": True,
        "server_observed_live_continuous_batches": True,
        "live_server_proven": True,
        "tick_batches": [
            {"tick": 0, "request_ids": ["req-a"], "output_token_ids": [10]},
            {"tick": 1, "request_ids": ["req-a", "req-b"], "output_token_ids": [11, 20]},
        ],
    }


def test_capture_assembler_emits_verifier_accepted_late_arrival_evidence():
    from mvp_capabilities.continuous_batching_live_server_capture import (
        assemble_live_server_continuous_batching_evidence,
    )
    from mvp_capabilities.continuous_batching_live_server_proof import (
        verify_live_server_continuous_batching_payload,
    )

    baseline_by_request = {
        "req-a": _capture("Shared prefix A", [101, 102], [10, 11], seconds=0.50),
        "req-b": _capture("Shared prefix B", [101, 103], [20], seconds=0.30),
    }
    continuous_by_request = {
        "req-a": _capture("Shared prefix A", [101, 102], [10, 11], seconds=0.40),
        "req-b": _capture("Shared prefix B", [101, 103], [20], seconds=0.25),
    }

    payload = assemble_live_server_continuous_batching_evidence(
        model_id=MODEL_ID,
        baseline_by_request=baseline_by_request,
        continuous_by_request=continuous_by_request,
        arrival_ticks={"req-a": 0, "req-b": 1},
        live_report=_server_observed_report(),
        source_artifacts={"req-a": {"baseline": "baseline-a.json", "continuous": "continuous-a.json"}},
    )

    assert payload["claim_boundary"] == "live_continuous_batching_capture_assembler_candidate_no_speedup"
    assert payload["proof_gate"] == "continuous_batching"
    assert payload["opt_in_flag"] == "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
    assert payload["opt_in_enabled"] is True
    assert payload["server_observed_live_continuous_batches"] is True
    assert payload["speedup_proven"] is False
    assert payload["wallclock_speedup_proven"] is False
    assert payload["can_update_demo_status"] is False
    assert payload["requests"][0]["baseline"]["generated_token_ids"] == [10, 11]
    assert payload["requests"][0]["continuous"]["generated_token_ids"] == [10, 11]
    assert payload["requests"][0]["baseline"]["logits_sha256"] == payload["requests"][0]["continuous"]["logits_sha256"]

    verified = verify_live_server_continuous_batching_payload(payload, model_id=MODEL_ID)
    assert verified["status"] == "passed"
    assert verified["live_server_late_arrival_parity_proven"] is True
    assert verified["speedup_proven"] is False
    assert verified["can_update_demo_status"] is False


def test_capture_assembler_fails_closed_without_server_observed_flag():
    from mvp_capabilities.continuous_batching_live_server_capture import (
        assemble_live_server_continuous_batching_evidence,
    )
    from mvp_capabilities.continuous_batching_live_server_proof import (
        verify_live_server_continuous_batching_payload,
    )

    baseline_by_request = {
        "req-a": _capture("A", [101], [10], seconds=0.20),
        "req-b": _capture("B", [102], [20], seconds=0.20),
    }
    continuous_by_request = {
        "req-a": _capture("A", [101], [10], seconds=0.20),
        "req-b": _capture("B", [102], [20], seconds=0.20),
    }
    report = _server_observed_report()
    report.pop("server_observed_live_continuous_batches")
    report.pop("live_server_proven")

    payload = assemble_live_server_continuous_batching_evidence(
        model_id=MODEL_ID,
        baseline_by_request=baseline_by_request,
        continuous_by_request=continuous_by_request,
        arrival_ticks={"req-a": 0, "req-b": 1},
        live_report=report,
    )

    assert payload["server_observed_live_continuous_batches"] is False
    assert payload["live_server_proven"] is False
    verified = verify_live_server_continuous_batching_payload(payload, model_id=MODEL_ID)
    assert verified["status"] == "failed"
    assert "live server did not report continuous batching observation" in verified["failed_checks"]
    assert verified["can_update_demo_status"] is False


def test_capture_plan_and_cli_assemble_write_verifiable_evidence(tmp_path: Path):
    from mvp_capabilities.continuous_batching_live_server_capture import (
        build_live_server_continuous_batching_capture_plan,
    )

    plan = build_live_server_continuous_batching_capture_plan(
        model_id=MODEL_ID,
        evidence_path=".local/live-continuous-capture.json",
    )
    assert plan["claim_boundary"] == "live_continuous_batching_capture_harness_no_live_server_proof"
    assert plan["proof_gate"] == "continuous_batching"
    assert plan["live_server_late_arrival_parity_proven"] is False
    assert plan["speedup_proven"] is False
    assert "continuous_batching_live_server_capture" in plan["assemble_command"]
    assert "continuous_batching_live_server_proof.py verify" in plan["verify_command"]

    baseline_a = tmp_path / "baseline-a.json"
    baseline_b = tmp_path / "baseline-b.json"
    continuous_a = tmp_path / "continuous-a.json"
    continuous_b = tmp_path / "continuous-b.json"
    live_report = tmp_path / "live-report.json"
    out = tmp_path / "assembled.json"
    baseline_a.write_text(json.dumps(_capture("A", [101], [10], seconds=0.20)), encoding="utf-8")
    baseline_b.write_text(json.dumps(_capture("B", [102], [20], seconds=0.30)), encoding="utf-8")
    continuous_a.write_text(json.dumps(_capture("A", [101], [10], seconds=0.10)), encoding="utf-8")
    continuous_b.write_text(json.dumps(_capture("B", [102], [20], seconds=0.20)), encoding="utf-8")
    live_report.write_text(json.dumps(_server_observed_report()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.continuous_batching_live_server_capture",
            "assemble",
            "--model",
            MODEL_ID,
            "--live-report",
            str(live_report),
            "--request",
            f"req-a:0:{baseline_a}:{continuous_a}",
            "--request",
            f"req-b:1:{baseline_b}:{continuous_b}",
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
    assert payload["claim_boundary"] == "live_continuous_batching_capture_assembler_candidate_no_speedup"

    verify = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/continuous_batching_live_server_proof.py",
            "verify",
            "--model",
            MODEL_ID,
            "--evidence",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["status"] == "passed"
