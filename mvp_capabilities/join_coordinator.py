#!/usr/bin/env python3
"""Join-link and heartbeat primitives for the distributed-inference MVP.

This is the first, deliberately small slice of the QR/link join flow. It creates
shareable join offers and records peer heartbeats, but it does not claim that a
peer has served inference. Heartbeats become inputs for later live-roster and
layer-planner work.
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

DEFAULT_STATE_DIR = Path(".local/join-state")
CLAIM_BOUNDARY = "link_offer_only_no_inference_proof"


def _now_seconds() -> int:
    return int(time.time())


def create_join_offer(
    *,
    coordinator: str,
    token: str | None = None,
    now: int | None = None,
    ttl_seconds: int = 600,
) -> dict[str, Any]:
    created_at = _now_seconds() if now is None else int(now)
    token = token or secrets.token_urlsafe(24)
    query = urlencode({"coordinator": coordinator, "token": token})
    return {
        "coordinator": coordinator,
        "token": token,
        "created_at": created_at,
        "expires_at": created_at + int(ttl_seconds),
        "ttl_seconds": int(ttl_seconds),
        "join_url": f"bloombee://join?{query}",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _heartbeat_path(state_dir: str | Path, peer_id: str) -> Path:
    safe_peer_id = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in peer_id)
    return Path(state_dir).expanduser() / f"{safe_peer_id}.json"


def record_heartbeat(
    state_dir: str | Path,
    *,
    token: str,
    peer_id: str,
    capabilities: dict[str, Any],
    now: int | None = None,
) -> dict[str, Any]:
    timestamp = _now_seconds() if now is None else int(now)
    payload = {
        "peer_id": peer_id,
        "token": token,
        "timestamp": timestamp,
        "capabilities": capabilities,
        "claim_boundary": "heartbeat_only_no_inference_proof",
    }
    path = _heartbeat_path(state_dir, peer_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_active_heartbeats(
    state_dir: str | Path,
    *,
    token: str,
    now: int | None = None,
    max_age_seconds: int = 30,
) -> list[dict[str, Any]]:
    cutoff_now = _now_seconds() if now is None else int(now)
    active: list[dict[str, Any]] = []
    for path in sorted(Path(state_dir).expanduser().glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("token") != token:
            continue
        try:
            age = cutoff_now - int(payload.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if 0 <= age <= int(max_age_seconds):
            active.append(payload)
    active.sort(key=lambda item: str(item.get("peer_id", "")))
    return active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    offer = sub.add_parser("offer", help="Create a shareable join offer JSON payload")
    offer.add_argument("--coordinator", required=True)
    offer.add_argument("--token", default=None)
    offer.add_argument("--ttl-seconds", type=int, default=600)
    offer.add_argument("--now", type=int, default=None)

    heartbeat = sub.add_parser("heartbeat", help="Record a peer heartbeat JSON file")
    heartbeat.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    heartbeat.add_argument("--token", required=True)
    heartbeat.add_argument("--peer-id", required=True)
    heartbeat.add_argument("--capabilities", required=True, help="Path to peer capability JSON")
    heartbeat.add_argument("--now", type=int, default=None)

    active = sub.add_parser("active", help="List active heartbeat peers for a join token")
    active.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    active.add_argument("--token", required=True)
    active.add_argument("--now", type=int, default=None)
    active.add_argument("--max-age-seconds", type=int, default=30)

    args = parser.parse_args(argv)
    if args.command == "offer":
        payload = create_join_offer(
            coordinator=args.coordinator,
            token=args.token,
            now=args.now,
            ttl_seconds=args.ttl_seconds,
        )
    elif args.command == "heartbeat":
        capabilities = json.loads(Path(args.capabilities).expanduser().read_text(encoding="utf-8"))
        payload = record_heartbeat(
            args.state_dir,
            token=args.token,
            peer_id=args.peer_id,
            capabilities=capabilities,
            now=args.now,
        )
    else:
        payload = {
            "token": args.token,
            "active_peers": load_active_heartbeats(
                args.state_dir,
                token=args.token,
                now=args.now,
                max_age_seconds=args.max_age_seconds,
            ),
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
