#!/usr/bin/env python3
"""Summarize retained proof-prep/download state without claiming inference.

This module is observability glue for long-running proof jobs. It parses retained
status/log/cache facts and emits a small JSON document that dashboards can render.
It must never promote a proof gate; only the dedicated proof verifier may do that.
"""

from __future__ import annotations

import argparse
import json
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


def build_proof_state(
    *,
    model: str,
    gate: str,
    status_file: str | Path | None = None,
    log_file: str | Path | None = None,
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
    effective_weight_files = weight_files if weight_files is not None else log_fields.get("log_weight_files")
    cache: dict[str, Any] = {
        "bytes": cache_bytes,
        "weight_files": effective_weight_files,
        "human": log_fields.get("cache_human"),
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
