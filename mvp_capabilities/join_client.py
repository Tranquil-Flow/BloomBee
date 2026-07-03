#!/usr/bin/env python3
"""Physical-device join client for the distributed-inference MVP.

Parses a `bloombee://join?...` offer, loads a peer capability JSON, and posts a
heartbeat to the coordinator. This is bootstrap/roster state only; it does not
start BloomBee servers or claim inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

REQUEST_CLAIM_BOUNDARY = "join_client_request_only_no_inference_proof"
DRY_RUN_CLAIM_BOUNDARY = "join_client_dry_run_only_no_inference_proof"
POST_CLAIM_BOUNDARY = "join_client_post_only_no_inference_proof"


def parse_join_url(join_url: str) -> dict[str, str]:
    parsed = urlparse(join_url)
    query = parse_qs(parsed.query)
    coordinator = (query.get("coordinator") or [None])[0]
    token = (query.get("token") or [None])[0]
    if parsed.scheme != "bloombee" or parsed.netloc != "join" or not coordinator or not token:
        raise ValueError("join URL must include coordinator and token")
    return {"coordinator": coordinator, "token": token}


def _load_capabilities(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def build_heartbeat_payload(
    join: dict[str, str],
    *,
    capabilities_path: str | Path,
    peer_id: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    capabilities = _load_capabilities(capabilities_path)
    resolved_peer_id = peer_id or capabilities.get("hostname") or capabilities.get("peer_id")
    if not resolved_peer_id:
        raise ValueError("peer_id is required when capabilities JSON has no hostname")
    payload: dict[str, Any] = {
        "token": join["token"],
        "peer_id": str(resolved_peer_id),
        "capabilities": capabilities,
        "claim_boundary": REQUEST_CLAIM_BOUNDARY,
    }
    if now is not None:
        payload["now"] = int(now)
    return payload


def _heartbeat_url(coordinator: str) -> str:
    return coordinator.rstrip("/") + "/heartbeat"


def build_heartbeat_request(join: dict[str, str], payload: dict[str, Any]) -> Request:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return Request(
        _heartbeat_url(join["coordinator"]),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def dry_run_report(join_url: str, *, capabilities_path: str | Path, peer_id: str | None = None, now: int | None = None) -> dict[str, Any]:
    join = parse_join_url(join_url)
    payload = build_heartbeat_payload(join, capabilities_path=capabilities_path, peer_id=peer_id, now=now)
    request = build_heartbeat_request(join, payload)
    return {
        "dry_run": True,
        "url": request.full_url,
        "method": request.get_method(),
        "headers": dict(request.headers),
        "body": payload,
        "claim_boundary": DRY_RUN_CLAIM_BOUNDARY,
    }


def post_heartbeat(
    join_url: str,
    *,
    capabilities_path: str | Path,
    peer_id: str | None = None,
    timeout: float = 5.0,
    now: int | None = None,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    join = parse_join_url(join_url)
    payload = build_heartbeat_payload(join, capabilities_path=capabilities_path, peer_id=peer_id, now=now)
    request = build_heartbeat_request(join, payload)
    with urlopen_fn(request, timeout=timeout) as response:
        server_response = json.loads(response.read().decode("utf-8"))
    return {
        "url": request.full_url,
        "peer_id": payload["peer_id"],
        "server_response": server_response,
        "claim_boundary": POST_CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-url", required=True)
    parser.add_argument("--capabilities", required=True, help="Path to peer capability JSON from peer_scan.py")
    parser.add_argument("--peer-id", default=None, help="Override peer id; defaults to capabilities.hostname")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--now", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print request payload without sending it")
    args = parser.parse_args(argv)

    if args.dry_run:
        payload = dry_run_report(args.join_url, capabilities_path=args.capabilities, peer_id=args.peer_id, now=args.now)
    else:
        payload = post_heartbeat(
            args.join_url,
            capabilities_path=args.capabilities,
            peer_id=args.peer_id,
            timeout=args.timeout,
            now=args.now or int(time.time()),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
