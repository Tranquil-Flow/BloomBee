#!/usr/bin/env python3
"""Read-only context snapshot for the Distributed Inference MVP nightly builder cron.

This script is intentionally side-effect free. The cron job's LLM agent consumes
this JSON before choosing one repo-backed TDD slice to implement.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path("/Users/evinova/Projects/distributed-inference-mvp")
PYTHON = PROJECT / ".venv" / "bin" / "python"


def run(cmd: list[str], *, cwd: Path = PROJECT, timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        return {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-8000:],
        }
    except Exception as exc:  # noqa: BLE001 - context script must not crash the cron run
        return {"cmd": cmd, "error": repr(exc)}


def parse_json_command(cmd: list[str], *, timeout: int = 60) -> dict[str, Any]:
    result = run(cmd, timeout=timeout)
    try:
        result["parsed_json"] = json.loads(result.get("stdout") or "{}")
    except Exception as exc:  # noqa: BLE001
        result["json_error"] = repr(exc)
    return result


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def main() -> int:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "read-only context for distributed-inference-mvp remaining-work autonomous cron",
        "project": str(PROJECT),
        "python": str(PYTHON),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version,
        },
        "repo_exists": PROJECT.exists(),
        "venv_python_exists": PYTHON.exists(),
        "commands": {},
    }

    if PROJECT.exists():
        payload["commands"]["git_status"] = run(["git", "status", "--short", "--branch"], timeout=30)
        payload["commands"]["git_log"] = run(["git", "log", "--oneline", "-8"], timeout=30)
        payload["commands"]["git_remote"] = run(["git", "remote", "-v"], timeout=30)
        payload["commands"]["checklist_json"] = parse_json_command(
            [str(PYTHON), "mvp_capabilities/remaining_work_checklist.py", "--json"], timeout=60
        )
        payload["commands"]["checklist_text"] = run(
            [str(PYTHON), "mvp_capabilities/remaining_work_checklist.py"], timeout=60
        )
        payload["commands"]["recent_evidence"] = run(
            [
                "bash",
                "-lc",
                "find mvp_capabilities/distributed_evidence -maxdepth 4 -type f -name '*.json' 2>/dev/null | sort | tail -40",
            ],
            timeout=30,
        )
        payload["commands"]["disk"] = run(["df", "-h", "."], timeout=30)
    else:
        payload["fatal"] = "project path missing"

    payload["commands"]["date_utc"] = run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], cwd=Path("/"), timeout=10)
    payload["commands"]["host_memory"] = run(
        ["bash", "-lc", "sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/ {print $2*1024}' /proc/meminfo 2>/dev/null || true"],
        cwd=Path("/"),
        timeout=10,
    )
    payload["commands"]["nvidia_smi"] = run(["bash", "-lc", "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"], cwd=Path("/"), timeout=20)
    payload["commands"]["adb_devices"] = run(["bash", "-lc", "command -v adb >/dev/null && adb devices || true"], cwd=Path("/"), timeout=20)
    payload["tool_availability"] = {
        "git": command_exists("git"),
        "adb": command_exists("adb"),
        "ssh": command_exists("ssh"),
        "nvidia-smi": command_exists("nvidia-smi"),
    }
    payload["policy_hints"] = [
        "Use tranquil-flow remote for pushes; origin is read-only/403 for this repo.",
        "Do not claim proof without evidence artifacts and verifier output.",
        "If hardware/phones/NVIDIA/80GB memory are missing, create/update fail-closed blocker evidence or move to a locally runnable slice.",
        "Prefer one small TDD slice per cron tick; commit/push after green verification.",
        "Stop doing code when remaining checklist items are all complete or blocked only by human/hardware access.",
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
