import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phone_wallclock_gate_cli_accepts_measured_integrated_timing(tmp_path: Path):
    phone_bridge = tmp_path / "phone-bridge.json"
    same_gguf_verifier = tmp_path / "same-gguf-verifier.json"
    tokenizer_compare = tmp_path / "tokenizer-compare.json"
    out_path = tmp_path / "wallclock-gate.json"

    phone_bridge.write_text(json.dumps({"evidence": {"draft_response": {"elapsed_s": 0.2}}}), encoding="utf-8")
    same_gguf_verifier.write_text(json.dumps({"elapsed_s": 1.0}), encoding="utf-8")
    tokenizer_compare.write_text(
        json.dumps({"verifier_acceptance_proven": True, "tokenizer_id_match_proven": True}),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_speculative_wallclock_gate.py",
            "--phone-bridge",
            str(phone_bridge),
            "--same-gguf-verifier",
            str(same_gguf_verifier),
            "--tokenizer-compare",
            str(tokenizer_compare),
            "--measured-draft-plus-verifier-elapsed-s",
            "0.7",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["speedup_proven"] is True
    assert report["timing_kind"] == "measured_draft_plus_verifier"
    assert report["measured_draft_plus_verifier_elapsed_s"] == 0.7
    assert report["candidate_draft_plus_verifier_elapsed_s"] == 0.7
    assert report["verifier_only_elapsed_s"] == 1.0
    assert report["wallclock_speedup_proven"] is True
    assert report["can_update_speculative_speedup_status"] is True
    assert report["blocked_reason"] is None
