#!/usr/bin/env python3
"""MiniMax-M2.7 REAP GGUF launch-readiness gate.

Claim boundary: ``minimax_m27_reap_launch_readiness_no_inference_proof``.

This checker answers only whether the M4Pro-side external llama.cpp smoke can
start as soon as the GGUF finishes. It does not prove inference, speed, quality,
or BloomBee-native support. A real smoke artifact must be produced separately by
running the emitted command and validating generated output.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

SOURCE = "minimax_m27_reap_launch_gate.py"
CLAIM_BOUNDARY = "minimax_m27_reap_launch_readiness_no_inference_proof"
MODEL_ID = "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"
GGUF_REPO_ID = "mradermacher/m51Lab-MiniMax-M2.7-REAP-139B-A10B-i1-GGUF"
MODEL_FILE = "m51Lab-MiniMax-M2.7-REAP-139B-A10B.i1-IQ2_XXS.gguf"
DEFAULT_MODEL_DIR = Path("/Volumes/Seagate Portable Drive/models/minimax-m27-reap-i1-gguf")
DEFAULT_EXPECTED_MIN_BYTES = 36_000_000_000
DEFAULT_REQUIRED_FREE_GB = 40.8
DEFAULT_PORT = 18087

_PAGE_RE = re.compile(r"page size of\s+(\d+)\s+bytes", re.IGNORECASE)
_PAGE_VALUE_RE = re.compile(r"^Pages\s+([^:]+):\s+(\d+)\.", re.MULTILINE)


def parse_vm_stat_freeable_gb(vm_stat_text: str) -> float | None:
    """Return macOS free+inactive+speculative+purgeable GiB from vm_stat text."""
    page_match = _PAGE_RE.search(vm_stat_text or "")
    if not page_match:
        return None
    page_size = int(page_match.group(1))
    values = {name.strip().lower(): int(value) for name, value in _PAGE_VALUE_RE.findall(vm_stat_text)}
    page_count = sum(
        values.get(key, 0)
        for key in (
            "free",
            "inactive",
            "speculative",
            "purgeable",
        )
    )
    return round(page_count * page_size / (1024 ** 3), 3)


def _path_present(path: str | Path | None) -> tuple[bool, str | None]:
    if not path:
        return False, None
    raw = str(path)
    if "/" in raw:
        p = Path(raw)
        return p.exists(), str(p)
    found = shutil.which(raw)
    return found is not None, found


def _scan_download(model_dir: Path, *, expected_min_bytes: int) -> dict[str, Any]:
    ggufs = sorted(model_dir.rglob("*.gguf")) if model_dir.exists() else []
    exact = [p for p in ggufs if p.name == MODEL_FILE]
    candidates = exact or ggufs
    best = max(candidates, key=lambda p: p.stat().st_size, default=None)
    incomplete_files = sorted(
        [p for p in model_dir.rglob("*.incomplete")] if model_dir.exists() else [],
        key=lambda p: str(p),
    )
    partial_bytes = sum(p.stat().st_size for p in incomplete_files if p.exists())
    size = best.stat().st_size if best and best.exists() else 0
    completed = best is not None and best.name == MODEL_FILE and size >= expected_min_bytes and not incomplete_files
    return {
        "model_dir": str(model_dir),
        "expected_file": MODEL_FILE,
        "expected_min_bytes": expected_min_bytes,
        "completed": completed,
        "gguf_path": str(best) if best else None,
        "gguf_size_bytes": size,
        "gguf_size_gb": round(size / 1_000_000_000, 3) if size else 0.0,
        "incomplete_files": [str(p) for p in incomplete_files],
        "partial_bytes": partial_bytes,
        "partial_gb": round(partial_bytes / 1_000_000_000, 3) if partial_bytes else 0.0,
    }


def build_minimax_m27_reap_launch_report(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    llama_cli_path: str | Path = "llama-cli",
    llama_server_path: str | Path = "llama-server",
    vm_stat_text: str = "",
    expected_min_bytes: int = DEFAULT_EXPECTED_MIN_BYTES,
    required_free_gb: float = DEFAULT_REQUIRED_FREE_GB,
    port: int = DEFAULT_PORT,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    download = _scan_download(model_dir, expected_min_bytes=expected_min_bytes)
    llama_cli_present, llama_cli_resolved = _path_present(llama_cli_path)
    llama_server_present, llama_server_resolved = _path_present(llama_server_path)
    freeable_gb = parse_vm_stat_freeable_gb(vm_stat_text) if vm_stat_text else None
    memory_attemptable = freeable_gb is not None and freeable_gb >= required_free_gb

    blocked: list[str] = []
    if not download["completed"]:
        blocked.append("gguf_file_missing_or_too_small")
    if download["incomplete_files"]:
        blocked.append("download_incomplete_files_present")
    if not llama_cli_present:
        blocked.append("llama_cli_missing")
    if not llama_server_present:
        blocked.append("llama_server_missing")
    if freeable_gb is None:
        blocked.append("vm_stat_unavailable")
    elif not memory_attemptable:
        blocked.append("freeable_memory_below_required")

    launch_ready = not blocked
    model_path = download["gguf_path"] or str(model_dir / MODEL_FILE)
    smoke_command = [
        str(llama_cli_resolved or llama_cli_path),
        "-m",
        model_path,
        "-p",
        "You are MiniMax M2.7 REAP. Reply with the single word READY.",
        "-n",
        "16",
        "--ctx-size",
        "512",
        "--temp",
        "0",
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]
    server_command = [
        str(llama_server_resolved or llama_server_path),
        "-m",
        model_path,
        "--ctx-size",
        "4096",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": MODEL_ID,
        "gguf_repo_id": GGUF_REPO_ID,
        "download": download,
        "runtime": {
            "framework": "llama.cpp",
            "llama_cli_present": llama_cli_present,
            "llama_cli_path": llama_cli_resolved,
            "llama_server_present": llama_server_present,
            "llama_server_path": llama_server_resolved,
        },
        "memory": {
            "freeable_gb": freeable_gb,
            "required_free_gb": required_free_gb,
            "attemptable": memory_attemptable,
            "definition": "macOS vm_stat free+inactive+speculative+purgeable pages",
        },
        "launch_ready": launch_ready,
        "blocked_reasons": blocked,
        "smoke_command": smoke_command,
        "server_command": server_command,
        "external_runtime_smoke_proven": False,
        "native_bloombee_support_proven": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--llama-cli-path", default="llama-cli")
    parser.add_argument("--llama-server-path", default="llama-server")
    parser.add_argument("--vm-stat-file", default=None)
    parser.add_argument("--vm-stat-text", default="")
    parser.add_argument("--expected-min-bytes", type=int, default=DEFAULT_EXPECTED_MIN_BYTES)
    parser.add_argument("--required-free-gb", type=float, default=DEFAULT_REQUIRED_FREE_GB)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    vm_stat_text = args.vm_stat_text
    if args.vm_stat_file:
        vm_stat_text = Path(args.vm_stat_file).read_text(encoding="utf-8")
    payload = build_minimax_m27_reap_launch_report(
        model_dir=args.model_dir,
        llama_cli_path=args.llama_cli_path,
        llama_server_path=args.llama_server_path,
        vm_stat_text=vm_stat_text,
        expected_min_bytes=args.expected_min_bytes,
        required_free_gb=args.required_free_gb,
        port=args.port,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["launch_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
