#!/usr/bin/env python3
"""Summarize live direct-client request logs for MVP request telemetry.

This is observability only. It reports requests seen, success/failure counts,
latency summaries, and errors from logs such as ``scripts/direct_remote_call.py``.
It does not prove load handling; proof promotion remains gated by dedicated
verifiers and proof-status updates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "request_telemetry_observability_only_no_load_proof"
_RESULT_PREFIX = "[direct] RESULT:"
_ERROR_RE = re.compile(r"^(?P<kind>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|RuntimeError|TimeoutError):\s*(?P<message>.*)$")
_MODEL_RE = re.compile(r"^\[direct\]\s+model=(?P<model>\S+)\s*$")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _latency_summary(values: list[float], *, unmeasured_count: int = 0) -> dict[str, Any]:
    if not values:
        return {"count": 0, "avg": None, "min": None, "max": None, "p95": None, "unmeasured_count": unmeasured_count}
    return {
        "count": len(values),
        "avg": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "p95": _percentile(values, 0.95),
        "unmeasured_count": unmeasured_count,
    }


def _append_positive_latency(value: Any, latencies: list[float]) -> bool:
    """Append measured latency seconds; return False for missing/zero/unmeasured values."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(seconds) or seconds <= 0.0:
        return False
    latencies.append(seconds)
    return True


def _block_range_label(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 2:
        return f"{value[0]}:{value[1]}"
    if isinstance(value, tuple) and len(value) == 2:
        return f"{value[0]}:{value[1]}"
    if isinstance(value, str):
        return value
    return None


def _read_log(path: str | Path) -> tuple[str, list[str]] | None:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return None
    try:
        return str(expanded), expanded.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def build_request_telemetry(paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Build a request-telemetry report from direct-client logs."""
    scanned_logs: list[str] = []
    models: Counter[str] = Counter()
    block_ranges: Counter[str] = Counter()
    forward_latencies: list[float] = []
    backward_latencies: list[float] = []
    unmeasured_latencies = {"forward": 0, "backward": 0}
    errors: list[dict[str, str]] = []
    result_rows: list[dict[str, Any]] = []
    explicit_failure_logs: set[str] = set()
    current_model_by_log: dict[str, str] = {}

    for raw_path in paths or []:
        loaded = _read_log(raw_path)
        if loaded is None:
            continue
        log_path, lines = loaded
        scanned_logs.append(log_path)
        for line in lines:
            stripped = line.strip()
            model_match = _MODEL_RE.match(stripped)
            if model_match:
                current_model_by_log[log_path] = model_match.group("model")
            if stripped.startswith(_RESULT_PREFIX):
                raw_json = stripped[len(_RESULT_PREFIX) :].strip()
                try:
                    payload = json.loads(raw_json)
                except json.JSONDecodeError as exc:
                    errors.append({"log": log_path, "message": f"malformed result JSON: {exc}"})
                    explicit_failure_logs.add(log_path)
                    continue
                result_rows.append(payload)
                model = payload.get("model") or current_model_by_log.get(log_path)
                if model:
                    models[str(model)] += 1
                block_label = _block_range_label(payload.get("block_range"))
                if block_label:
                    block_ranges[block_label] += 1
                if not _append_positive_latency(payload.get("forward_seconds"), forward_latencies):
                    unmeasured_latencies["forward"] += 1
                if not _append_positive_latency(payload.get("backward_seconds"), backward_latencies):
                    unmeasured_latencies["backward"] += 1
                if payload.get("ok") is not True:
                    explicit_failure_logs.add(log_path)
                    errors.append({"log": log_path, "message": f"result ok={payload.get('ok')}"})
                continue
            error_match = _ERROR_RE.match(stripped)
            if error_match:
                explicit_failure_logs.add(log_path)
                errors.append({"log": log_path, "message": stripped})

    succeeded = sum(1 for row in result_rows if row.get("ok") is True)
    failed = len(explicit_failure_logs)
    total = succeeded + failed
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scanned_logs": scanned_logs,
        "live_requests_seen": total > 0,
        "load_proof_claimed": False,
        "request_counts": {"total": total, "succeeded": succeeded, "failed": failed},
        "models": dict(sorted(models.items())),
        "block_ranges": dict(sorted(block_ranges.items())),
        "latency_seconds": {
            "forward": _latency_summary(forward_latencies, unmeasured_count=unmeasured_latencies["forward"]),
            "backward": _latency_summary(backward_latencies, unmeasured_count=unmeasured_latencies["backward"]),
        },
        "errors": errors[:12],
        "next_step": "Run a dedicated multi-request load proof before promoting multi_request_load.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-log", action="append", default=None, help="Direct-client request log; may be repeated")
    args = parser.parse_args(argv)
    print(json.dumps(build_request_telemetry(args.request_log), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
