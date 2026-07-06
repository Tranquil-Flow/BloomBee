#!/usr/bin/env python3
"""Fail-closed ADB preflight for multi-phone speculative trials.

This helper is a hardware/transport preflight only. It may say that 3-4 ADB
phones are currently online and ready for artifact collection, but it never
proves context-token ingestion, integrated speculative decoding, BloomBee phone
block serving, or wall-clock speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "phone_adb_multiphone_preflight_no_speedup_claim"
SOURCE = "phone_adb_readiness_preflight.py"
DEFAULT_MIN_PHONE_COUNT = 3
DEFAULT_MAX_PHONE_COUNT = 4


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _excerpt(value: str, *, limit: int = 1000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_adb_devices_stdout(adb_stdout: str) -> list[dict[str, Any]]:
    """Parse ``adb devices`` output without exposing raw serial numbers."""

    devices: list[dict[str, Any]] = []
    for raw_line in adb_stdout.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("list of devices attached"):
            continue
        # ADB normally emits "serial<TAB>status". Some versions add product /
        # model details after the status; preserve only status and hashes.
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, status = parts[0], parts[1]
        devices.append(
            {
                "device_ref": f"adb-device-{len(devices) + 1}",
                "serial_sha256": _sha256_text(serial),
                "status": status,
                "ready_for_trial_collection": status == "device",
            }
        )
    return devices


def _adb_daemon_available(adb_exit_code: int, adb_stderr: str) -> bool:
    lowered = adb_stderr.lower()
    daemon_failure_markers = (
        "failed to start daemon",
        "cannot connect to daemon",
        "could not install *smartsocket* listener",
        "adb server didn't ack",
    )
    if any(marker in lowered for marker in daemon_failure_markers):
        return False
    return adb_exit_code == 0


def build_phone_adb_readiness_preflight(
    *,
    adb_stdout: str,
    adb_stderr: str,
    adb_exit_code: int,
    min_phone_count: int = DEFAULT_MIN_PHONE_COUNT,
    max_phone_count: int = DEFAULT_MAX_PHONE_COUNT,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed multi-phone ADB hardware preflight report."""

    if min_phone_count < 1:
        raise ValueError("min_phone_count must be >= 1")
    if max_phone_count < min_phone_count:
        raise ValueError("max_phone_count must be >= min_phone_count")

    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()
    devices = parse_adb_devices_stdout(adb_stdout)
    online_devices = [device for device in devices if device["status"] == "device"]
    non_ready_devices = [device for device in devices if device["status"] != "device"]
    online_count = len(online_devices)
    blocked_reasons: list[str] = []

    adb_command_succeeded = adb_exit_code == 0
    adb_daemon_available = _adb_daemon_available(adb_exit_code, adb_stderr)
    if not adb_command_succeeded:
        blocked_reasons.append("adb_command_failed")
    if not adb_daemon_available:
        blocked_reasons.append("adb_daemon_unavailable")
    if non_ready_devices:
        blocked_reasons.append("adb_non_ready_devices_present")
    if online_count < min_phone_count:
        blocked_reasons.append(f"phone_count_below_min:{online_count}<{min_phone_count}")
    if online_count > max_phone_count:
        blocked_reasons.append(f"phone_count_above_max:{online_count}>{max_phone_count}")

    ready_for_collection = not blocked_reasons
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at_utc": generated_at_utc,
        "verification_status": "passed" if ready_for_collection else "failed",
        "adb_command_succeeded": adb_command_succeeded,
        "adb_exit_code": int(adb_exit_code),
        "adb_daemon_available": adb_daemon_available,
        "adb_stdout_sha256": _sha256_text(adb_stdout),
        "adb_stderr_sha256": _sha256_text(adb_stderr),
        "adb_stderr_excerpt": _excerpt(adb_stderr),
        "min_phone_count": int(min_phone_count),
        "max_phone_count": int(max_phone_count),
        "adb_connected_phone_count": online_count,
        "adb_non_ready_device_count": len(non_ready_devices),
        "adb_devices": devices,
        "ready_for_multiphone_artifact_collection": ready_for_collection,
        "ready_for_multiphone_speculative_readiness_manifest": False,
        "trial_ready": False,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": blocked_reasons,
        "operator_next_steps": [
            f"connect {min_phone_count}-{max_phone_count} authorized Android phones over ADB",
            "run per-phone Termux context-token and wall-clock correctness artifacts",
            "feed those per-phone artifacts into multi_phone_speculative_readiness.py",
            "run phone_speculative_integrated_trial_gate.py only after multi-phone readiness passes",
        ],
        "claim_limitations": [
            "ADB hardware visibility is not phone context-token readiness.",
            "This preflight does not prove integrated speculative speedup or BloomBee phone block serving.",
            "Device serials are hashed; raw ADB output is represented only by SHA-256 digests and a stderr excerpt.",
        ],
    }


def _run_adb_devices(*, adb_path: str = "adb", timeout_s: float = 15.0) -> tuple[str, str, int]:
    resolved = shutil.which(adb_path) if "/" not in adb_path else adb_path
    if not resolved:
        return "", f"adb executable not found: {adb_path}", 127
    try:
        proc = subprocess.run(
            [resolved, "devices"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_message = f"adb devices timed out after {timeout_s}s"
        combined_stderr = "\n".join(part for part in (stderr, timeout_message) if part).strip()
        return stdout, combined_stderr, 124
    return proc.stdout, proc.stderr, int(proc.returncode)


def _read_text_file(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--adb-stdout-file", default=None)
    parser.add_argument("--adb-stderr-file", default=None)
    parser.add_argument("--adb-exit-code", type=int, default=None)
    parser.add_argument("--min-phone-count", type=int, default=DEFAULT_MIN_PHONE_COUNT)
    parser.add_argument("--max-phone-count", type=int, default=DEFAULT_MAX_PHONE_COUNT)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    fixture_mode = args.adb_stdout_file is not None or args.adb_stderr_file is not None or args.adb_exit_code is not None
    if fixture_mode:
        adb_stdout = _read_text_file(args.adb_stdout_file)
        adb_stderr = _read_text_file(args.adb_stderr_file)
        adb_exit_code = int(args.adb_exit_code or 0)
    else:
        adb_stdout, adb_stderr, adb_exit_code = _run_adb_devices(adb_path=args.adb_path, timeout_s=args.timeout_s)

    payload = build_phone_adb_readiness_preflight(
        adb_stdout=adb_stdout,
        adb_stderr=adb_stderr,
        adb_exit_code=adb_exit_code,
        min_phone_count=args.min_phone_count,
        max_phone_count=args.max_phone_count,
        generated_at_utc=args.generated_at_utc,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
