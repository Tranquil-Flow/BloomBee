from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kv_prefix_reuse_session_metadata_fails_closed_without_opt_in(monkeypatch):
    from bloombee.client.inference_session import InferenceSession

    monkeypatch.delenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", raising=False)
    session = InferenceSession(SimpleNamespace(), max_length=8)
    input_ids = [[101, 102, 201], [101, 102, 202]]

    with pytest.raises(RuntimeError, match="BLOOMBEE_ENABLE_KV_PREFIX_REUSE"):
        session.record_kv_prefix_reuse_prefill(input_ids, request_ids=["suffix-a", "suffix-b"])


def test_kv_prefix_reuse_session_metadata_records_same_prefix_varied_suffix(monkeypatch):
    from bloombee.client.inference_session import InferenceSession

    monkeypatch.setenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "1")
    session = InferenceSession(SimpleNamespace(), max_length=8)
    input_ids = [[101, 102, 201], [101, 102, 202, 203]]

    event = session.record_kv_prefix_reuse_prefill(input_ids, request_ids=["suffix-a", "suffix-b"])
    report = session.kv_prefix_reuse_report()

    assert event["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert event["common_prefix_token_ids"] == [101, 102]
    assert event["request_count"] == 2
    assert event["same_prefix_varied_suffix_proven"] is True
    assert event["live_kv_cache_reuse_proven"] is False
    assert event["speedup_proven"] is False
    assert [row["suffix_token_ids"] for row in event["requests"]] == [[201], [202, 203]]

    assert report["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert report["opt_in_enabled"] is True
    assert report["event_count"] == 1
    assert report["events"] == [event]
    assert report["live_kv_cache_reuse_proven"] is False
    assert report["can_update_demo_status"] is False


def _valid_kv_prefix_reuse_evidence() -> dict[str, object]:
    return {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "proof_gate": "kv_prefix_reuse",
        "prefix_reuse_enabled": True,
        "baseline_reuse_enabled": False,
        "telemetry_tags": ["kv_prefix_reuse", "no_reuse_baseline", "same_prefix_varied_suffix"],
        "common_prefix_token_ids": [101, 102, 103, 104],
        "server_observed_kv_cache_reuse": True,
        "live_kv_cache_reuse_proven": True,
        "server_observations": [
            {
                "source": "bloombee.server.handler",
                "claim_boundary": "kv_prefix_reuse_server_cache_read_observed_no_speedup",
                "server_observed_kv_cache_reuse": True,
                "live_kv_cache_reuse_proven": True,
                "prefix_length": 4,
                "cache_handle_count": 2,
            }
        ],
        "requests": [
            {
                "request_id": "suffix-a",
                "prefix_token_ids": [101, 102, 103, 104],
                "suffix_token_ids": [201],
                "baseline": {
                    "generated_token_ids": [3001, 3002],
                    "logits_sha256": "a" * 64,
                    "seconds": 0.42,
                },
                "reuse": {
                    "generated_token_ids": [3001, 3002],
                    "logits_sha256": "a" * 64,
                    "seconds": 0.18,
                    "reused_prefix_token_count": 4,
                },
            },
            {
                "request_id": "suffix-b",
                "prefix_token_ids": [101, 102, 103, 104],
                "suffix_token_ids": [202, 203],
                "baseline": {
                    "generated_token_ids": [4001, 4002, 4003],
                    "logits_sha256": "b" * 64,
                    "seconds": 0.51,
                },
                "reuse": {
                    "generated_token_ids": [4001, 4002, 4003],
                    "logits_sha256": "b" * 64,
                    "seconds": 0.22,
                    "reused_prefix_token_count": 4,
                },
            },
        ],
    }


def test_kv_prefix_reuse_proof_accepts_same_prefix_varied_suffix_parity_and_timing(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_evidence

    evidence_path = tmp_path / "kv-prefix-reuse.json"
    evidence_path.write_text(json.dumps(_valid_kv_prefix_reuse_evidence()), encoding="utf-8")

    result = verify_kv_prefix_reuse_evidence(
        evidence_path=evidence_path,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    assert result["claim_boundary"] == "verified_kv_prefix_reuse_same_prefix_varied_suffix_parity_timing"
    assert result["proof_gate"] == "kv_prefix_reuse"
    assert result["status"] == "passed"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"kv_prefix_reuse": "passed"}
    assert result["can_update_mvp_status"] is False
    assert result["failed_checks"] == []
    assert result["same_prefix_varied_suffix_proven"] is True
    assert result["token_parity_proven"] is True
    assert result["logit_parity_proven"] is True
    assert result["timing_measured"] is True
    assert result["speedup_proven"] is True
    assert result["server_observed_kv_cache_reuse"] is True
    assert result["live_kv_cache_reuse_proven"] is True
    assert result["evidence_summary"]["server_observation_count"] == 1
    assert result["evidence_summary"]["request_count"] == 2
    assert result["evidence_summary"]["prefix_token_count"] == 4
    assert result["evidence_summary"]["reuse_event_count"] == 2
    assert result["evidence_summary"]["baseline_total_seconds"] == 0.93
    assert result["evidence_summary"]["reuse_total_seconds"] == 0.4
    assert result["evidence_summary"]["timing_delta_seconds"] == 0.53
    assert len(result["evidence_summary"]["prefix_sha256"]) == 64
    assert len(result["evidence_summary"]["correctness_sha256"]) == 64


def test_kv_prefix_reuse_proof_rejects_missing_server_observed_cache_reuse(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_evidence

    payload = _valid_kv_prefix_reuse_evidence()
    payload.pop("server_observed_kv_cache_reuse", None)
    payload.pop("live_kv_cache_reuse_proven", None)
    payload.pop("server_observations", None)
    evidence_path = tmp_path / "kv-prefix-reuse-no-server-observation.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_kv_prefix_reuse_evidence(
        evidence_path=evidence_path,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    assert result["status"] == "failed"
    assert result["server_observed_kv_cache_reuse"] is False
    assert result["live_kv_cache_reuse_proven"] is False
    assert "server did not report KV cache tensor reuse" in result["failed_checks"]


def test_kv_prefix_reuse_proof_rejects_token_mismatch_fail_closed(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_evidence

    payload = _valid_kv_prefix_reuse_evidence()
    payload["requests"][1]["reuse"]["generated_token_ids"] = [4001, 9999, 4003]
    evidence_path = tmp_path / "kv-prefix-reuse-mismatch.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_kv_prefix_reuse_evidence(
        evidence_path=evidence_path,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert result["proof_status_update"] == {}
    assert result["token_parity_proven"] is False
    assert result["same_prefix_varied_suffix_proven"] is True
    assert any("request 1 generated token IDs differ from no-reuse baseline" in check for check in result["failed_checks"])


def test_kv_prefix_reuse_proof_rejects_missing_timing_even_with_matching_tokens(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_evidence

    payload = _valid_kv_prefix_reuse_evidence()
    del payload["requests"][0]["reuse"]["seconds"]
    evidence_path = tmp_path / "kv-prefix-reuse-missing-timing.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_kv_prefix_reuse_evidence(
        evidence_path=evidence_path,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    assert result["status"] == "failed"
    assert result["timing_measured"] is False
    assert result["speedup_proven"] is False
    assert any("request 0 reuse seconds missing or not positive" in check for check in result["failed_checks"])


def test_kv_prefix_reuse_proof_rejects_non_varied_suffixes(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_evidence

    payload = _valid_kv_prefix_reuse_evidence()
    payload["requests"][1]["suffix_token_ids"] = [201]
    evidence_path = tmp_path / "kv-prefix-reuse-same-suffix.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_kv_prefix_reuse_evidence(
        evidence_path=evidence_path,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    assert result["status"] == "failed"
    assert result["same_prefix_varied_suffix_proven"] is False
    assert "expected at least 2 distinct suffixes sharing the same prefix" in result["failed_checks"]


def test_kv_prefix_reuse_proof_cli_verifies_evidence_file(tmp_path: Path):
    evidence_path = tmp_path / "kv-prefix-reuse-cli.json"
    evidence_path.write_text(json.dumps(_valid_kv_prefix_reuse_evidence()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.kv_prefix_reuse_proof",
            "verify",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--evidence",
            str(evidence_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["claim_boundary"] == "verified_kv_prefix_reuse_same_prefix_varied_suffix_parity_timing"


def test_kv_prefix_reuse_proof_cli_fails_closed_for_invalid_evidence(tmp_path: Path):
    payload = cast(dict[str, Any], _valid_kv_prefix_reuse_evidence())
    payload["requests"][0]["reuse"]["generated_token_ids"] = [9999]
    evidence_path = tmp_path / "kv-prefix-reuse-invalid-cli.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.kv_prefix_reuse_proof",
            "verify",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--evidence",
            str(evidence_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "failed"
    assert payload["can_update_proof_status"] is False
    assert "request 0 generated token IDs differ from no-reuse baseline" in payload["failed_checks"]


def test_kv_prefix_reuse_live_capture_plan_is_claim_bounded(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_proof import build_kv_prefix_reuse_live_capture_plan

    evidence_path = tmp_path / "kv-prefix-live.json"
    plan = build_kv_prefix_reuse_live_capture_plan(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        evidence_path=str(evidence_path),
        allow_no_speedup=True,
    )

    assert plan["claim_boundary"] == "kv_prefix_reuse_live_capture_plan_no_live_cache_reuse_proof"
    assert plan["proof_gate"] == "kv_prefix_reuse"
    assert plan["opt_in_flag"] == "BLOOMBEE_ENABLE_KV_PREFIX_REUSE"
    assert plan["live_kv_cache_reuse_proven"] is False
    assert plan["speedup_proven"] is False
    assert plan["can_update_proof_status"] is False
    assert plan["can_update_demo_status"] is False
    assert "BLOOMBEE_ENABLE_KV_PREFIX_REUSE=1" in plan["operator_commands"][0]
    assert "scripts/text_generation_parity.py" in "\n".join(plan["operator_commands"])
    assert "--mode generate-api" in "\n".join(plan["operator_commands"])
    assert "--allow-no-speedup" in plan["verify_command"]

    out = tmp_path / "plan.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.kv_prefix_reuse_proof",
            "plan",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--evidence",
            str(evidence_path),
            "--allow-no-speedup",
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
    assert payload["claim_boundary"] == "kv_prefix_reuse_live_capture_plan_no_live_cache_reuse_proof"
