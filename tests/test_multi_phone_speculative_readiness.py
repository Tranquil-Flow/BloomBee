import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"


def _phone_artifact(phone_id: str, *, model_sha: str = MODEL_SHA, accepted: bool = True, speedup_claim: bool = False) -> dict:
    return {
        "phone_id": phone_id,
        "phone_model": "Pixel 8 Pro",
        "runtime": "termux",
        "transport_path": "m4pro_usb_adb_pull_termux_json_then_local_forced_batch_verifier",
        "termux_context_token_artifact": f"mvp_capabilities/distributed_evidence/phone/{phone_id}-termux-context-token-ids.json",
        "context_token_verifier": {
            "claim_boundary": "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim",
            "model_id": "ggml-org/tiny-llamas/stories15M.gguf",
            "model_sha256": model_sha,
            "prompt": "Once upon a time",
            "draft_text": "One day, a little girl named Lucy",
            "phone_external_token_ids_ingested": accepted,
            "phone_integrated_verifier_proven": accepted,
            "external_context_token_id_acceptance_proven": accepted,
            "accepted_external_token_count": 8 if accepted else 0,
            "proposed_external_token_count": 8,
            "rejected_external_token_count": 0 if accepted else 8,
            "speedup_proven": speedup_claim,
            "bloombee_block_serving_proven": False,
        },
        "wallclock_gate": {
            "claim_boundary": "phone_speculative_wallclock_gate_fail_closed",
            "verifier_acceptance_proven": accepted,
            "tokenizer_id_match_proven": accepted,
            "phone_draft_elapsed_s": 0.565503,
            "verifier_only_elapsed_s": 1.837976,
            "candidate_draft_plus_verifier_elapsed_s": 2.403479,
            "speedup_ratio": 0.764715,
            "speedup_proven": speedup_claim,
            "wallclock_speedup_proven": speedup_claim,
            "can_update_speculative_speedup_status": speedup_claim,
            "blocked_reason": None if speedup_claim else "sequential_draft_plus_verifier_not_faster_than_verifier_only",
        },
    }


def test_multi_phone_readiness_passes_for_three_distinct_context_token_phones_without_speedup_claim():
    from mvp_capabilities.multi_phone_speculative_readiness import build_multi_phone_readiness_report

    report = build_multi_phone_readiness_report(
        [_phone_artifact("pixel-a"), _phone_artifact("pixel-b"), _phone_artifact("pixel-c")],
        min_phone_count=3,
        max_phone_count=4,
    )

    assert report["claim_boundary"] == "multi_phone_speculative_readiness_manifest_no_speedup_claim"
    assert report["verification_status"] == "passed"
    assert report["trial_ready"] is True
    assert report["phone_count"] == 3
    assert report["ready_phone_count"] == 3
    assert report["unique_phone_ids"] == ["pixel-a", "pixel-b", "pixel-c"]
    assert report["model_sha256"] == MODEL_SHA
    assert report["all_context_token_ingestion_proven"] is True
    assert report["all_wallclock_gates_present"] is True
    assert report["speedup_proven"] is False
    assert report["can_update_speculative_speedup_status"] is False
    assert report["can_update_phone_worker_status"] is False
    assert report["blocked_reasons"] == []
    assert any("run Termux context-token emission on each phone" in step for step in report["tomorrow_runbook"])
    assert any("integrated non-sequential verifier" in step for step in report["operator_next_steps"])


def test_multi_phone_readiness_fails_closed_for_duplicate_missing_or_overclaiming_phone_artifacts():
    from mvp_capabilities.multi_phone_speculative_readiness import build_multi_phone_readiness_report

    duplicate = build_multi_phone_readiness_report(
        [_phone_artifact("pixel-a"), _phone_artifact("pixel-a"), _phone_artifact("pixel-c")],
        min_phone_count=3,
        max_phone_count=4,
    )
    assert duplicate["verification_status"] == "failed"
    assert duplicate["trial_ready"] is False
    assert "duplicate_phone_id:pixel-a" in duplicate["blocked_reasons"]

    incomplete = build_multi_phone_readiness_report(
        [_phone_artifact("pixel-a"), _phone_artifact("pixel-b", accepted=False), _phone_artifact("pixel-c")],
        min_phone_count=3,
        max_phone_count=4,
    )
    assert incomplete["verification_status"] == "failed"
    assert incomplete["trial_ready"] is False
    assert "phone:pixel-b:context_token_ingestion_not_proven" in incomplete["blocked_reasons"]
    assert "phone:pixel-b:wallclock_correctness_not_proven" in incomplete["blocked_reasons"]

    overclaim = build_multi_phone_readiness_report(
        [_phone_artifact("pixel-a"), _phone_artifact("pixel-b", speedup_claim=True), _phone_artifact("pixel-c")],
        min_phone_count=3,
        max_phone_count=4,
    )
    assert overclaim["verification_status"] == "failed"
    assert overclaim["trial_ready"] is False
    assert "phone:pixel-b:unexpected_speedup_claim" in overclaim["blocked_reasons"]
    assert overclaim["speedup_proven"] is False


def test_multi_phone_readiness_cli_reads_phone_artifacts_and_writes_manifest(tmp_path: Path):
    artifact_paths = []
    for phone_id in ("pixel-a", "pixel-b", "pixel-c", "pixel-d"):
        path = tmp_path / f"{phone_id}.json"
        path.write_text(json.dumps(_phone_artifact(phone_id)), encoding="utf-8")
        artifact_paths.append(path)
    out_path = tmp_path / "multi-phone-readiness.json"

    cmd = [
        sys.executable,
        "mvp_capabilities/multi_phone_speculative_readiness.py",
        "--min-phone-count",
        "3",
        "--max-phone-count",
        "4",
        "--out",
        str(out_path),
    ]
    for path in artifact_paths:
        cmd.extend(["--phone-artifact", str(path)])

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=15)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["verification_status"] == "passed"
    assert payload["phone_count"] == 4
    assert payload["trial_ready"] is True
    assert payload["speedup_proven"] is False
    assert "pixel-d" in payload["unique_phone_ids"]
