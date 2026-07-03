#!/usr/bin/env python3
"""Summarize retained proof-prep/download state without claiming inference.

This module is observability glue for long-running proof jobs. It parses retained
status/log/cache facts and emits a small JSON document that dashboards can render.
It must never promote a proof gate; only the dedicated proof verifier may do that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "proof_state_observability_only_no_inference_proof"
_FETCH_RE = re.compile(r"Fetching\s+(?P<total>\d+)\s+files:\s+(?P<percent>\d+)%.*?\|\s*(?P<done>\d+)\s*/\s*(?P=total)\b")
_KEYVAL_RE = re.compile(r"^(?P<key>[A-Z_]+)[= ](?P<value>.+)$")
_SIZE_RE = re.compile(r"^(?P<human>\d+(?:\.\d+)?[KMGTPE]?)(?:i?B)?\s+/.+")


def _read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return ""
    return expanded.read_text(encoding="utf-8", errors="replace")


def _parse_status(text: str) -> tuple[str, int | None]:
    exit_code: int | None = None
    for line in text.splitlines():
        if line.startswith("EXIT_CODE="):
            try:
                exit_code = int(line.split("=", 1)[1].strip())
            except ValueError:
                exit_code = None
            break
    if exit_code == 0:
        return "complete", exit_code
    if exit_code is not None:
        return "failed", exit_code
    return "running_or_missing", None


def _parse_log(text: str) -> dict[str, Any]:
    normalized = text.replace("\r", "\n")
    fields: dict[str, Any] = {}
    fetch_progress: dict[str, int] | None = None
    cache_human: str | None = None
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _KEYVAL_RE.match(line)
        if match:
            key = match.group("key")
            value = match.group("value").strip()
            if key in {"START", "END", "HOST", "MODEL", "SNAPSHOT_PATH"}:
                fields[key.lower()] = value
            elif key == "TOKEN_FILE_PRESENT":
                fields["token_file_present"] = value.lower() == "true"
            elif key == "SECONDS":
                try:
                    fields["seconds"] = float(value)
                except ValueError:
                    fields["seconds"] = value
            elif key == "WEIGHT_FILES":
                try:
                    fields["log_weight_files"] = int(value)
                except ValueError:
                    fields["log_weight_files"] = value
            continue
        fetch = _FETCH_RE.search(line)
        if fetch:
            fetch_progress = {
                "percent": int(fetch.group("percent")),
                "completed_files": int(fetch.group("done")),
                "total_files": int(fetch.group("total")),
            }
            continue
        size = _SIZE_RE.match(line)
        if size:
            cache_human = size.group("human")
    if fetch_progress is not None:
        fields["fetch_progress"] = fetch_progress
    if cache_human is not None:
        fields["cache_human"] = cache_human
    return fields


def _status_from_status_and_log(status_text: str, log_fields: dict[str, Any]) -> tuple[str, int | None]:
    status, exit_code = _parse_status(status_text)
    if status != "running_or_missing":
        return status, exit_code
    if log_fields.get("end"):
        return "complete", exit_code
    if log_fields:
        return "running", exit_code
    return status, exit_code


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            candidate = Path(root) / name
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _inspect_cache(cache_dir: str | Path | None, snapshot_path: str | Path | None) -> dict[str, Any]:
    snapshot = Path(snapshot_path).expanduser() if snapshot_path else None
    cache = Path(cache_dir).expanduser() if cache_dir else None
    if cache is None and snapshot is not None and len(snapshot.parents) >= 2:
        cache = snapshot.parents[1]

    snapshot_weight_files = 0
    snapshot_total_weight_bytes = 0
    snapshot_file_count = None
    snapshot_exists = bool(snapshot and snapshot.exists())
    if snapshot_exists and snapshot is not None:
        snapshot_file_count = 0
        for item in snapshot.iterdir():
            snapshot_file_count += 1
            if item.name.endswith((".safetensors", ".bin")):
                snapshot_weight_files += 1
                try:
                    snapshot_total_weight_bytes += item.resolve().stat().st_size
                except FileNotFoundError:
                    pass

    incomplete_files = 0
    incomplete_bytes = 0
    if cache is not None and cache.exists():
        for item in cache.rglob("*.incomplete"):
            incomplete_files += 1
            try:
                incomplete_bytes += item.stat().st_size
            except FileNotFoundError:
                continue

    snapshot_complete = snapshot_exists and snapshot_weight_files > 0
    return {
        "cache_dir": str(cache) if cache is not None else None,
        "snapshot_path": str(snapshot) if snapshot is not None else None,
        "snapshot_exists": snapshot_exists,
        "snapshot_file_count": snapshot_file_count,
        "snapshot_weight_files": snapshot_weight_files,
        "snapshot_total_weight_bytes": snapshot_total_weight_bytes,
        "snapshot_complete": snapshot_complete,
        "incomplete_files": incomplete_files,
        "incomplete_bytes": incomplete_bytes,
        "stale_incomplete_files": incomplete_files if snapshot_complete else 0,
        "active_incomplete_files": 0 if snapshot_complete else incomplete_files,
        "tree_bytes": _tree_bytes(cache) if cache is not None else None,
    }


def _eta(download_status: str, cache_info: dict[str, Any]) -> tuple[int | None, str]:
    if cache_info.get("snapshot_complete") or download_status == "complete":
        return 0, "snapshot_complete"
    if cache_info.get("active_incomplete_files"):
        return None, "active_download_or_stale_without_complete_snapshot"
    return None, "insufficient_progress_history"


def build_proof_state(
    *,
    model: str,
    gate: str,
    status_file: str | Path | None = None,
    log_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_bytes: int | None = None,
    weight_files: int | None = None,
) -> dict[str, Any]:
    """Build a dashboard-safe proof-prep state report.

    `inference_proven` is always false here. This report is intentionally weaker
    than `one_block_proof.py verify` and cannot update proof status.
    """

    status_text = _read_text(status_file)
    log_text = _read_text(log_file)
    log_fields = _parse_log(log_text)
    download_status, exit_code = _status_from_status_and_log(status_text, log_fields)
    cache_info = _inspect_cache(cache_dir, log_fields.get("snapshot_path"))
    if cache_info.get("snapshot_complete") and download_status in {"running_or_missing", "running"}:
        download_status = "complete"
    effective_weight_files = weight_files if weight_files is not None else log_fields.get("log_weight_files")
    if effective_weight_files is None and cache_info.get("snapshot_weight_files"):
        effective_weight_files = cache_info.get("snapshot_weight_files")
    effective_cache_bytes = cache_bytes if cache_bytes is not None else cache_info.get("tree_bytes")
    eta_seconds, eta_reason = _eta(download_status, cache_info)
    cache: dict[str, Any] = {
        "bytes": effective_cache_bytes,
        "weight_files": effective_weight_files,
        "human": log_fields.get("cache_human"),
        **cache_info,
    }
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "model": model,
        "gate": gate,
        "download_status": download_status,
        "exit_code": exit_code,
        "host": log_fields.get("host"),
        "started_at": log_fields.get("start"),
        "ended_at": log_fields.get("end"),
        "snapshot_path": log_fields.get("snapshot_path"),
        "seconds": log_fields.get("seconds"),
        "token_file_present": log_fields.get("token_file_present"),
        "fetch_progress": log_fields.get("fetch_progress"),
        "cache": cache,
        "eta_seconds": eta_seconds,
        "eta_reason": eta_reason,
        "status_file": str(Path(status_file).expanduser()) if status_file else None,
        "log_file": str(Path(log_file).expanduser()) if log_file else None,
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": "run the dedicated live proof verifier after weights and server/client logs are complete",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", default="one_block_server")
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-bytes", type=int, default=None)
    parser.add_argument("--weight-files", type=int, default=None)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_proof_state(
                model=args.model,
                gate=args.gate,
                status_file=args.status_file,
                log_file=args.log_file,
                cache_dir=args.cache_dir,
                cache_bytes=args.cache_bytes,
                weight_files=args.weight_files,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
