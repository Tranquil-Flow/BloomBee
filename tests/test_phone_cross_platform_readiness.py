import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"


def _phone_artifact(phone_id: str, *, platform: str, runtime: str, accepted: bool = True, speedup_claim: bool = False) -> dict:
    return {
        "phone_id": phone_id,
        "phone_model": "Pixel 8 Pro" if platform == "android" else "iPhone 15 Pro",
        "platform": platform,
        "runtime": runtime,
        "transport_path": "adb_termux_json" if platform == "android" else "ios_shortcuts_local_network_json",
        "model_sha256": MODEL_SHA,
        "context_token_verifier": {
            "claim_boundary": "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim",
            "model_sha256": MODEL_SHA,
            "phone_external_token_ids_ingested": accepted,
            "phone_integrated_verifier_proven": accepted,
            "external_context_token_id_acceptance_proven": accepted,
            "accepted_external_token_count": 8 if accepted else 0,
            "proposed_external_token_count": 8,
            "speedup_proven": speedup_claim,
            "bloombee_block_serving_proven": False,
        },
        "wallclock_gate": {
            "claim_boundary": "phone_speculative_wallclock_gate_fail_closed",
            "verifier_acceptance_proven": accepted,
            "tokenizer_id_match_proven": accepted,
            "speedup_proven": speedup_claim,
            "wallclock_speedup_proven": speedup_claim,
            "can_update_speculative_speedup_status": speedup_claim,
        },
    }


def test_phone_cross_platform_readiness_passes_only_when_android_and_ios_artifacts_are_ready():
    from mvp_capabilities.phone_cross_platform_readiness import build_phone_cross_platform_readiness_report

    report = build_phone_cross_platform_readiness_report(
        [
            _phone_artifact("pixel-a", platform="android", runtime="termux-llama.cpp"),
            _phone_artifact("iphone-a", platform="ios", runtime="ios-shortcuts-llamacpp-bridge"),
        ]
    )

    assert report["claim_boundary"] == "phone_cross_platform_readiness_no_speedup_or_block_worker_claim"
    assert report["verification_status"] == "passed"
    assert report["cross_platform_ready"] is True
    assert report["required_platforms"] == ["android", "ios"]
    assert report["ready_platforms"] == ["android", "ios"]
    assert report["android_ready"] is True
    assert report["ios_ready"] is True
    assert report["speedup_proven"] is False
    assert report["can_update_speculative_speedup_status"] is False
    assert report["can_update_phone_worker_status"] is False
    assert any("Termux" in step for step in report["platform_runbook"]["android"])
    assert any("Shortcuts" in step or "iOS" in step for step in report["platform_runbook"]["ios"])


def test_phone_cross_platform_readiness_fails_closed_without_ios_or_with_overclaim():
    from mvp_capabilities.phone_cross_platform_readiness import build_phone_cross_platform_readiness_report

    missing_ios = build_phone_cross_platform_readiness_report(
        [_phone_artifact("pixel-a", platform="android", runtime="termux-llama.cpp")]
    )
    assert missing_ios["verification_status"] == "failed"
    assert missing_ios["cross_platform_ready"] is False
    assert "missing_required_platform:ios" in missing_ios["blocked_reasons"]

    overclaim = build_phone_cross_platform_readiness_report(
        [
            _phone_artifact("pixel-a", platform="android", runtime="termux-llama.cpp"),
            _phone_artifact("iphone-a", platform="ios", runtime="ios-shortcuts-llamacpp-bridge", speedup_claim=True),
        ]
    )
    assert overclaim["verification_status"] == "failed"
    assert "phone:iphone-a:unexpected_speedup_claim" in overclaim["blocked_reasons"]
    assert overclaim["speedup_proven"] is False


def test_phone_cross_platform_readiness_cli_writes_json(tmp_path: Path):
    android = tmp_path / "android.json"
    ios = tmp_path / "ios.json"
    out = tmp_path / "cross-platform.json"
    android.write_text(json.dumps(_phone_artifact("pixel-a", platform="android", runtime="termux-llama.cpp")), encoding="utf-8")
    ios.write_text(json.dumps(_phone_artifact("iphone-a", platform="ios", runtime="ios-shortcuts-llamacpp-bridge")), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_cross_platform_readiness.py",
            "--phone-artifact",
            str(android),
            "--phone-artifact",
            str(ios),
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
    assert payload["cross_platform_ready"] is True
