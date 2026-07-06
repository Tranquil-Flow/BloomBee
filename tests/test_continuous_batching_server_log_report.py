from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _observed_line(request_ids: list[str], *, synthetic: bool = False) -> str:
    payload = {
        "claim_boundary": "live_continuous_batching_server_metadata_observed_no_parity_or_speedup",
        "opt_in_flag": "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING",
        "opt_in_enabled": True,
        "server_observed_live_continuous_batches": len(request_ids) > 1,
        "live_server_proven": True,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_demo_status": False,
        "tick_batches": [
            {
                "tick": 7,
                "request_ids": request_ids,
                "positions": [0 for _ in request_ids],
                "input_token_ids": [101 + index for index, _ in enumerate(request_ids)],
            }
        ],
    }
    if synthetic:
        payload["input_note"] = "synthetic harness log for parser only"
    return "INFO bloombee.server.handler [LIVE_CONTINUOUS_BATCHING] " + json.dumps(payload, sort_keys=True)


def test_server_log_report_extracts_server_observed_batched_ticks():
    from mvp_capabilities.continuous_batching_server_log_report import (
        build_live_continuous_batching_server_log_report,
    )

    log_text = "\n".join(
        [
            "ordinary startup line",
            _observed_line(["generate-0"]),
            _observed_line(["generate-0", "generate-1"]),
        ]
    )

    report = build_live_continuous_batching_server_log_report(log_text, source="server.log")

    assert report["claim_boundary"] == "live_continuous_batching_server_log_report_no_parity_or_speedup"
    assert report["opt_in_flag"] == "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
    assert report["opt_in_enabled"] is True
    assert report["event_count"] == 2
    assert report["server_observed_live_continuous_batches"] is True
    assert report["live_server_proven"] is True
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False
    assert report["tick_batches"][1]["request_ids"] == ["generate-0", "generate-1"]


def test_server_log_report_keeps_synthetic_harness_from_live_server_proof():
    from mvp_capabilities.continuous_batching_server_log_report import (
        build_live_continuous_batching_server_log_report,
    )

    report = build_live_continuous_batching_server_log_report(
        _observed_line(["generate-0", "generate-1"], synthetic=True),
        source="synthetic-harness.log",
    )

    assert report["event_count"] == 1
    assert report["synthetic_fixture"] is True
    assert report["server_observed_live_continuous_batches"] is False
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_server_log_report_cli_writes_live_report_json(tmp_path: Path):
    log_path = tmp_path / "server.log"
    out_path = tmp_path / "live-report.json"
    log_path.write_text(_observed_line(["generate-0", "generate-1"]) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp_capabilities.continuous_batching_server_log_report",
            "--log",
            str(log_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_log"] == str(log_path)
    assert payload["server_observed_live_continuous_batches"] is True
