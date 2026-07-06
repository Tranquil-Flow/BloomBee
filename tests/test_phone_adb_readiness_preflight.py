from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_adb_preflight_passes_only_as_hardware_collection_gate_for_three_online_devices():
    from mvp_capabilities.phone_adb_readiness_preflight import build_phone_adb_readiness_preflight

    report = build_phone_adb_readiness_preflight(
        adb_stdout=(
            "List of devices attached\n"
            "pixel-a\tdevice\n"
            "pixel-b\tdevice\n"
            "pixel-c\tdevice\n"
        ),
        adb_stderr="",
        adb_exit_code=0,
        generated_at_utc="2026-07-06T03:07:58Z",
    )

    assert report["claim_boundary"] == "phone_adb_multiphone_preflight_no_speedup_claim"
    assert report["verification_status"] == "passed"
    assert report["adb_command_succeeded"] is True
    assert report["adb_connected_phone_count"] == 3
    assert report["ready_for_multiphone_artifact_collection"] is True
    assert report["ready_for_multiphone_speculative_readiness_manifest"] is False
    assert report["speedup_proven"] is False
    assert report["wallclock_speedup_proven"] is False
    assert report["can_update_speculative_speedup_status"] is False
    assert report["can_update_phone_worker_status"] is False
    assert report["bloombee_block_serving_proven"] is False
    assert report["blocked_reasons"] == []
    assert [device["status"] for device in report["adb_devices"]] == ["device", "device", "device"]
    assert all(len(device["serial_sha256"]) == 64 for device in report["adb_devices"])
    assert "pixel-a" not in json.dumps(report)


def test_adb_preflight_fails_closed_when_daemon_cannot_start():
    from mvp_capabilities.phone_adb_readiness_preflight import build_phone_adb_readiness_preflight

    report = build_phone_adb_readiness_preflight(
        adb_stdout="",
        adb_stderr=(
            "* daemon not running; starting now at tcp:5037\n"
            "ADB server didn't ACK\n"
            "could not install *smartsocket* listener: Operation not permitted\n"
            "* failed to start daemon\n"
            "adb: failed to check server version: cannot connect to daemon\n"
        ),
        adb_exit_code=1,
        generated_at_utc="2026-07-06T03:07:58Z",
    )

    assert report["verification_status"] == "failed"
    assert report["adb_command_succeeded"] is False
    assert report["adb_daemon_available"] is False
    assert report["adb_connected_phone_count"] == 0
    assert report["ready_for_multiphone_artifact_collection"] is False
    assert "adb_command_failed" in report["blocked_reasons"]
    assert "adb_daemon_unavailable" in report["blocked_reasons"]
    assert "phone_count_below_min:0<3" in report["blocked_reasons"]
    assert report["speedup_proven"] is False
    assert report["can_update_phone_worker_status"] is False


def test_adb_preflight_excludes_unauthorized_or_offline_devices_from_ready_count():
    from mvp_capabilities.phone_adb_readiness_preflight import build_phone_adb_readiness_preflight

    report = build_phone_adb_readiness_preflight(
        adb_stdout=(
            "List of devices attached\n"
            "ready-a\tdevice\n"
            "needs-auth\tunauthorized\n"
            "offline-phone\toffline\n"
        ),
        adb_stderr="",
        adb_exit_code=0,
        generated_at_utc="2026-07-06T03:07:58Z",
    )

    assert report["verification_status"] == "failed"
    assert report["adb_connected_phone_count"] == 1
    assert report["adb_non_ready_device_count"] == 2
    assert "adb_non_ready_devices_present" in report["blocked_reasons"]
    assert "phone_count_below_min:1<3" in report["blocked_reasons"]
    assert report["ready_for_multiphone_artifact_collection"] is False


def test_adb_preflight_cli_writes_fail_closed_json_from_fixture_output(tmp_path: Path):
    stdout_path = tmp_path / "adb.stdout"
    stderr_path = tmp_path / "adb.stderr"
    out_path = tmp_path / "phone-adb-preflight.json"
    stdout_path.write_text("List of devices attached\nonly-phone\tdevice\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/phone_adb_readiness_preflight.py",
            "--adb-stdout-file",
            str(stdout_path),
            "--adb-stderr-file",
            str(stderr_path),
            "--adb-exit-code",
            "0",
            "--generated-at-utc",
            "2026-07-06T03:07:58Z",
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
    assert payload["verification_status"] == "failed"
    assert payload["adb_connected_phone_count"] == 1
    assert "phone_count_below_min:1<3" in payload["blocked_reasons"]
    assert payload["speedup_proven"] is False
