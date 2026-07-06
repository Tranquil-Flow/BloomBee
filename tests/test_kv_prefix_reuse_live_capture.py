from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
COMMON_PREFIX = [101, 102, 103, 104]


def _capture(prompt: str, input_ids: list[int], generated: list[int], *, seconds: float) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "mode": "generate-api",
        "input_ids": input_ids,
        "distributed_ids": [*input_ids, *generated],
        "distributed_text": prompt + " done",
        "distributed_seconds": seconds,
        "distributed_top5": [
            {"token_id": generated[0], "logit": 9.0},
            {"token_id": 2, "logit": 1.0},
        ],
    }


def _server_report(*, live_reuse: bool) -> dict[str, object]:
    return {
        "source": "bloombee.server.kv_prefix_reuse_capture",
        "claim_boundary": "live_kv_prefix_reuse_server_capture",
        "opt_in_flag": "BLOOMBEE_ENABLE_KV_PREFIX_REUSE",
        "opt_in_enabled": True,
        "server_observed_kv_cache_reuse": live_reuse,
        "live_kv_cache_reuse_proven": live_reuse,
        "requests": {
            "suffix-a": {"reused_prefix_token_count": len(COMMON_PREFIX), "cache_event_id": "cache-a"},
            "suffix-b": {"reused_prefix_token_count": len(COMMON_PREFIX), "cache_event_id": "cache-b"},
        },
    }


def _baseline_rows() -> dict[str, dict[str, object]]:
    return {
        "suffix-a": _capture("Shared prefix A", [*COMMON_PREFIX, 201], [3001, 3002], seconds=0.42),
        "suffix-b": _capture("Shared prefix B", [*COMMON_PREFIX, 202, 203], [4001, 4002], seconds=0.51),
    }


def _reuse_rows() -> dict[str, dict[str, object]]:
    return {
        "suffix-a": _capture("Shared prefix A", [*COMMON_PREFIX, 201], [3001, 3002], seconds=0.18),
        "suffix-b": _capture("Shared prefix B", [*COMMON_PREFIX, 202, 203], [4001, 4002], seconds=0.22),
    }


def test_live_capture_assembler_emits_verifier_accepted_reuse_candidate_without_demo_promotion():
    from mvp_capabilities.kv_prefix_reuse_live_capture import assemble_kv_prefix_reuse_live_capture_evidence
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_payload

    payload = assemble_kv_prefix_reuse_live_capture_evidence(
        model_id=MODEL_ID,
        common_prefix_token_ids=COMMON_PREFIX,
        suffix_token_ids_by_request={"suffix-a": [201], "suffix-b": [202, 203]},
        baseline_by_request=_baseline_rows(),
        reuse_by_request=_reuse_rows(),
        server_report=_server_report(live_reuse=True),
        source_artifacts={"suffix-a": {"baseline": "baseline-a.json", "reuse": "reuse-a.json"}},
    )

    assert payload["claim_boundary"] == "kv_prefix_reuse_live_capture_assembler_candidate"
    assert payload["proof_gate"] == "kv_prefix_reuse"
    assert payload["opt_in_flag"] == "BLOOMBEE_ENABLE_KV_PREFIX_REUSE"
    assert payload["prefix_reuse_enabled"] is True
    assert payload["baseline_reuse_enabled"] is False
    assert payload["live_kv_cache_reuse_proven"] is True
    assert payload["speedup_proven"] is False
    assert payload["can_update_demo_status"] is False
    assert payload["requests"][0]["reuse"]["reused_prefix_token_count"] == len(COMMON_PREFIX)
    assert payload["requests"][0]["baseline"]["generated_token_ids"] == [3001, 3002]
    assert payload["requests"][0]["baseline"]["logits_sha256"] == payload["requests"][0]["reuse"]["logits_sha256"]

    verified = verify_kv_prefix_reuse_payload(payload, model_id=MODEL_ID)
    assert verified["status"] == "passed"
    assert verified["can_update_proof_status"] is True
    assert verified["speedup_proven"] is True
    assert verified["can_update_mvp_status"] is False


def test_live_capture_assembler_fails_closed_for_metadata_only_server_report():
    from mvp_capabilities.kv_prefix_reuse_live_capture import assemble_kv_prefix_reuse_live_capture_evidence
    from mvp_capabilities.kv_prefix_reuse_proof import verify_kv_prefix_reuse_payload

    metadata_only_report = {
        "source": "bloombee.server.handler",
        "claim_boundary": "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse",
        "opt_in_flag": "BLOOMBEE_ENABLE_KV_PREFIX_REUSE",
        "opt_in_enabled": True,
        "server_observed_metadata": True,
        "runtime_prefill_metadata_proven": True,
        "live_kv_cache_reuse_proven": False,
        "requests": {
            "suffix-a": {"reused_prefix_token_count": len(COMMON_PREFIX)},
            "suffix-b": {"reused_prefix_token_count": len(COMMON_PREFIX)},
        },
    }

    payload = assemble_kv_prefix_reuse_live_capture_evidence(
        model_id=MODEL_ID,
        common_prefix_token_ids=COMMON_PREFIX,
        suffix_token_ids_by_request={"suffix-a": [201], "suffix-b": [202, 203]},
        baseline_by_request=_baseline_rows(),
        reuse_by_request=_reuse_rows(),
        server_report=metadata_only_report,
    )

    assert payload["prefix_reuse_enabled"] is False
    assert payload["live_kv_cache_reuse_proven"] is False
    assert payload["server_observed_metadata_only"] is True
    assert payload["can_update_demo_status"] is False

    verified = verify_kv_prefix_reuse_payload(payload, model_id=MODEL_ID)
    assert verified["status"] == "failed"
    assert verified["can_update_proof_status"] is False
    assert "prefix/cache reuse path was not marked enabled" in verified["failed_checks"]


def test_live_capture_plan_and_cli_assemble_write_verifiable_evidence(tmp_path: Path):
    from mvp_capabilities.kv_prefix_reuse_live_capture import build_kv_prefix_reuse_live_capture_assembler_plan

    evidence_path = tmp_path / "kv-prefix-live-capture.json"
    plan = build_kv_prefix_reuse_live_capture_assembler_plan(
        model_id=MODEL_ID,
        evidence_path=str(evidence_path),
    )

    assert plan["claim_boundary"] == "kv_prefix_reuse_live_capture_assembler_harness_no_live_cache_reuse_proof"
    assert plan["proof_gate"] == "kv_prefix_reuse"
    assert plan["live_kv_cache_reuse_proven"] is False
    assert plan["can_update_demo_status"] is False
    assert "kv_prefix_reuse_live_capture" in plan["assemble_command"]
    assert "kv_prefix_reuse_proof" in plan["verify_command"]

    baseline_a = tmp_path / "baseline-a.json"
    baseline_b = tmp_path / "baseline-b.json"
    reuse_a = tmp_path / "reuse-a.json"
    reuse_b = tmp_path / "reuse-b.json"
    server_report = tmp_path / "server-report.json"
    baseline = _baseline_rows()
    reuse = _reuse_rows()
    baseline_a.write_text(json.dumps(baseline["suffix-a"]), encoding="utf-8")
    baseline_b.write_text(json.dumps(baseline["suffix-b"]), encoding="utf-8")
    reuse_a.write_text(json.dumps(reuse["suffix-a"]), encoding="utf-8")
    reuse_b.write_text(json.dumps(reuse["suffix-b"]), encoding="utf-8")
    server_report.write_text(json.dumps(_server_report(live_reuse=True)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.kv_prefix_reuse_live_capture",
            "assemble",
            "--model",
            MODEL_ID,
            "--common-prefix-token-ids",
            ",".join(str(token) for token in COMMON_PREFIX),
            "--server-report",
            str(server_report),
            "--request",
            f"suffix-a:201:{baseline_a}:{reuse_a}",
            "--request",
            f"suffix-b:202,203:{baseline_b}:{reuse_b}",
            "--out",
            str(evidence_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["claim_boundary"] == "kv_prefix_reuse_live_capture_assembler_candidate"

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.kv_prefix_reuse_proof",
            "verify",
            "--model",
            MODEL_ID,
            "--evidence",
            str(evidence_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["status"] == "passed"
