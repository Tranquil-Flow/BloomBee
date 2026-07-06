#!/usr/bin/env python3
"""Build a claim-bounded live-continuous server report from BloomBee logs.

The server emits structured ``[LIVE_CONTINUOUS_BATCHING]`` JSON lines when it
receives opt-in live-continuous request metadata. This parser turns those log
lines into the ``live_report`` JSON consumed by
``continuous_batching_live_server_capture.py``. It proves only server-side
observation of the metadata; token/logit parity and speedup remain separate
fail-closed gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LOG_PREFIX = "[LIVE_CONTINUOUS_BATCHING]"
OPT_IN_FLAG = "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
CLAIM_BOUNDARY = "live_continuous_batching_server_log_report_no_parity_or_speedup"
SERVER_EVENT_CLAIM_BOUNDARY = "live_continuous_batching_server_metadata_observed_no_parity_or_speedup"


def _normalize_tick_batch(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    request_ids = raw.get("request_ids")
    positions = raw.get("positions")
    input_token_ids = raw.get("input_token_ids")
    if not isinstance(request_ids, list) or not request_ids:
        return None
    if not isinstance(positions, list) or len(positions) != len(request_ids):
        return None
    if not isinstance(input_token_ids, list) or len(input_token_ids) != len(request_ids):
        return None
    try:
        normalized = {
            "tick": int(raw.get("tick", 0)),
            "request_ids": [str(item) for item in request_ids],
            "positions": [int(item) for item in positions],
            "input_token_ids": [int(item) for item in input_token_ids],
        }
    except Exception:
        return None
    if len(set(normalized["request_ids"])) != len(normalized["request_ids"]):
        return None
    return normalized


def _parse_observation_line(line: str) -> dict[str, Any] | None:
    if LOG_PREFIX not in line:
        return None
    _, json_text = line.split(LOG_PREFIX, 1)
    try:
        payload = json.loads(json_text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("claim_boundary") != SERVER_EVENT_CLAIM_BOUNDARY:
        return None
    if payload.get("opt_in_flag") != OPT_IN_FLAG or payload.get("opt_in_enabled") is not True:
        return None
    tick_batches = [
        batch
        for batch in (_normalize_tick_batch(raw) for raw in payload.get("tick_batches", []))
        if batch is not None
    ]
    if not tick_batches:
        return None
    event = dict(payload)
    event["tick_batches"] = tick_batches
    event["speedup_proven"] = False
    event["wallclock_speedup_proven"] = False
    event["can_update_demo_status"] = False
    event["can_update_proof_status"] = False
    return event


def build_live_continuous_batching_server_log_report(log_text: str, *, source: str | None = None) -> dict[str, Any]:
    """Return a fail-closed server-observation report from log text."""

    events = [event for event in (_parse_observation_line(line) for line in log_text.splitlines()) if event is not None]
    synthetic_fixture = any(
        "synthetic" in str(event.get("input_note", "")).lower()
        or "synthetic" in str(source or "").lower()
        for event in events
    )
    if synthetic_fixture:
        for event in events:
            event["synthetic_fixture"] = True
            event["server_observed_live_continuous_batches"] = False
            event["live_server_proven"] = False
    tick_batches: list[dict[str, Any]] = []
    for event in events:
        tick_batches.extend(dict(batch) for batch in event.get("tick_batches", []))

    batched_tick_seen = any(len(batch.get("request_ids", [])) > 1 for batch in tick_batches)
    server_observed = (not synthetic_fixture) and bool(events) and (
        batched_tick_seen or any(event.get("server_observed_live_continuous_batches") is True for event in events)
    )
    return {
        "source": "mvp_capabilities.continuous_batching_server_log_report",
        "source_log": source,
        "claim_boundary": CLAIM_BOUNDARY,
        "opt_in_flag": OPT_IN_FLAG,
        "opt_in_enabled": bool(events),
        "event_count": len(events),
        "synthetic_fixture": synthetic_fixture,
        "server_observed_live_continuous_batches": server_observed,
        "live_server_proven": server_observed,
        "tick_batches": tick_batches,
        "events": events,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_demo_status": False,
        "can_update_proof_status": False,
        "claim_limitations": [
            "Server log report only: this does not prove token/logit parity.",
            "Late-arrival parity must still pass continuous_batching_live_server_proof.py.",
            "Wall-clock speedup must still pass continuous_batching_wallclock_gate.py.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="BloomBee server log containing [LIVE_CONTINUOUS_BATCHING] JSON lines")
    parser.add_argument("--out", default=None, help="Optional path to write the report JSON")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    payload = build_live_continuous_batching_server_log_report(
        log_path.read_text(encoding="utf-8", errors="replace"),
        source=str(log_path),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
