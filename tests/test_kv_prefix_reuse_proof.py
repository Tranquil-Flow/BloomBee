from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_kv_prefix_reuse_evidence() -> dict[str, object]:
    return {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "proof_gate": "kv_prefix_reuse",
        "prefix_reuse_enabled": True,
        "baseline_reuse_enabled": False,
        "telemetry_tags": ["kv_prefix_reuse", "no_reuse_baseline", "same_prefix_varied_suffix"],
        "common_prefix_token_ids": [101, 102, 103, 104],
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
    assert result["evidence_summary"]["request_count"] == 2
    assert result["evidence_summary"]["prefix_token_count"] == 4
    assert result["evidence_summary"]["reuse_event_count"] == 2
    assert result["evidence_summary"]["baseline_total_seconds"] == 0.93
    assert result["evidence_summary"]["reuse_total_seconds"] == 0.4
    assert result["evidence_summary"]["timing_delta_seconds"] == 0.53
    assert len(result["evidence_summary"]["prefix_sha256"]) == 64
    assert len(result["evidence_summary"]["correctness_sha256"]) == 64


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
