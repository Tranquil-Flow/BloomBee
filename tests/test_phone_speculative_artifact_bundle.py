from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODEL_SHA = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"


def _context() -> dict:
    return {
        "claim_boundary": "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim",
        "model_sha256": MODEL_SHA,
        "phone_external_token_ids_ingested": True,
        "phone_integrated_verifier_proven": True,
        "external_context_token_id_acceptance_proven": True,
        "accepted_external_token_count": 8,
        "proposed_external_token_count": 8,
        "phone_token_json_artifact": "mvp_capabilities/distributed_evidence/phone/termux-context-token-ids-live-adb.json",
        "transport_path": "m4pro_usb_adb_pull_termux_json_then_local_forced_batch_verifier",
        "speedup_proven": False,
    }


def _wallclock() -> dict:
    return {
        "claim_boundary": "phone_speculative_wallclock_gate_fail_closed",
        "verifier_acceptance_proven": True,
        "tokenizer_id_match_proven": True,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
    }


def _termux_tokens() -> dict:
    return {
        "claim_boundary": "phone_context_token_ids_from_termux_llama_tokenize_no_verifier_no_speedup_claim",
        "model_sha256": MODEL_SHA,
        "phone_context_token_ids_emitted": True,
        "phone_context_draft_token_ids": [1, 2, 3],
        "speedup_proven": False,
    }


def test_phone_speculative_artifact_bundle_wraps_split_real_evidence_for_manifest():
    from mvp_capabilities.phone_speculative_artifact_bundle import build_phone_speculative_artifact_bundle
    from mvp_capabilities.multi_phone_speculative_readiness import build_multi_phone_readiness_report

    bundle = build_phone_speculative_artifact_bundle(
        context_verifier=_context(),
        wallclock_gate=_wallclock(),
        termux_context_tokens=_termux_tokens(),
        phone_id="pixel-8-pro-live-adb",
        phone_model="Pixel 8 Pro",
        runtime="termux-llama.cpp",
        termux_context_token_artifact="mvp_capabilities/distributed_evidence/phone/termux-context-token-ids-live-adb.json",
    )

    assert bundle["claim_boundary"] == "phone_speculative_readiness_artifact_bundle_no_speedup_claim"
    assert bundle["phone_id"] == "pixel-8-pro-live-adb"
    assert bundle["phone_model"] == "Pixel 8 Pro"
    assert bundle["runtime"] == "termux-llama.cpp"
    assert bundle["model_sha256"] == MODEL_SHA
    assert bundle["context_token_verifier"] == _context()
    assert bundle["wallclock_gate"] == _wallclock()
    assert bundle["termux_context_tokens"] == _termux_tokens()
    assert bundle["speedup_proven"] is False
    assert bundle["can_update_speculative_speedup_status"] is False
    assert bundle["bundle_ready_for_manifest"] is True
    assert bundle["blocked_reasons"] == []

    manifest = build_multi_phone_readiness_report([bundle], min_phone_count=3, max_phone_count=4)
    assert manifest["ready_phone_count"] == 1
    assert manifest["trial_ready"] is False
    assert "phone_count_below_min:1<3" in manifest["blocked_reasons"]


def test_phone_speculative_artifact_bundle_fails_closed_on_source_speedup_overclaim():
    from mvp_capabilities.phone_speculative_artifact_bundle import build_phone_speculative_artifact_bundle

    context = _context()
    context["speedup_proven"] = True

    bundle = build_phone_speculative_artifact_bundle(
        context_verifier=context,
        wallclock_gate=_wallclock(),
        phone_id="pixel-overclaim",
    )

    assert bundle["bundle_ready_for_manifest"] is False
    assert bundle["speedup_proven"] is False
    assert "source_unexpected_speedup_claim" in bundle["blocked_reasons"]


def test_phone_speculative_artifact_bundle_cli_writes_json(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    context_path = tmp_path / "context.json"
    wallclock_path = tmp_path / "wallclock.json"
    token_path = tmp_path / "tokens.json"
    out_path = tmp_path / "bundle.json"
    context_path.write_text(json.dumps(_context()), encoding="utf-8")
    wallclock_path.write_text(json.dumps(_wallclock()), encoding="utf-8")
    token_path.write_text(json.dumps(_termux_tokens()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_speculative_artifact_bundle.py",
            "--context-verifier",
            str(context_path),
            "--wallclock-gate",
            str(wallclock_path),
            "--termux-context-tokens",
            str(token_path),
            "--termux-context-token-artifact",
            str(token_path),
            "--phone-id",
            "pixel-cli",
            "--phone-model",
            "Pixel 8 Pro",
            "--runtime",
            "termux-llama.cpp",
            "--out",
            str(out_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["phone_id"] == "pixel-cli"
    assert payload["bundle_ready_for_manifest"] is True
