#!/usr/bin/env python3
"""Tiny HTTP coordinator for join offers and peer heartbeats.

This is bootstrap/roster infrastructure only. It does not start BloomBee servers,
does not run inference, and every response carries a no-inference-proof claim
boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from mvp_capabilities.join_coordinator import (
        DEFAULT_STATE_DIR,
        create_join_offer,
        load_active_heartbeats,
        record_heartbeat,
    )
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.join_coordinator import (
        DEFAULT_STATE_DIR,
        create_join_offer,
        load_active_heartbeats,
        record_heartbeat,
    )

HEALTH_CLAIM_BOUNDARY = "coordinator_health_only_no_inference_proof"
ERROR_CLAIM_BOUNDARY = "coordinator_error_no_inference_proof"


def _first(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _as_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def handle_get(path: str, *, state_dir: str | Path, coordinator: str) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path == "/healthz":
        return 200, {"ok": True, "claim_boundary": HEALTH_CLAIM_BOUNDARY}
    if parsed.path == "/offer":
        return 200, create_join_offer(
            coordinator=_first(query, "coordinator", coordinator) or coordinator,
            token=_first(query, "token"),
            ttl_seconds=_as_int(_first(query, "ttl_seconds"), 600),
            now=_as_int(_first(query, "now"), 0) or None,
        )
    if parsed.path == "/active":
        token = _first(query, "token")
        if not token:
            return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        return 200, {
            "token": token,
            "active_peers": load_active_heartbeats(
                state_dir,
                token=token,
                now=_as_int(_first(query, "now"), 0) or None,
                max_age_seconds=_as_int(_first(query, "max_age_seconds"), 30),
            ),
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }
    return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}


def handle_post(path: str, *, body: bytes, state_dir: str | Path) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    if parsed.path != "/heartbeat":
        return 404, {"error": "not found", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "invalid json", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    token = payload.get("token")
    peer_id = payload.get("peer_id")
    capabilities = payload.get("capabilities")
    if not token or not peer_id or not isinstance(capabilities, dict):
        return 400, {"error": "missing required heartbeat fields", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    return 200, record_heartbeat(
        state_dir,
        token=str(token),
        peer_id=str(peer_id),
        capabilities=capabilities,
        now=payload.get("now"),
    )


class JoinCoordinatorHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass, *, state_dir: str | Path, coordinator: str):
        super().__init__(server_address, RequestHandlerClass)
        self.state_dir = Path(state_dir).expanduser()
        self.coordinator = coordinator


class JoinCoordinatorHandler(BaseHTTPRequestHandler):
    server: JoinCoordinatorHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Keep tests/docs clean and avoid recording peer/token tuples in logs.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "claim_boundary": ERROR_CLAIM_BOUNDARY})

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        status, payload = handle_get(self.path, state_dir=self.server.state_dir, coordinator=self.server.coordinator)
        self._send_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error_json(400, "invalid content length")
            return
        status, payload = handle_post(self.path, body=self.rfile.read(length), state_dir=self.server.state_dir)
        self._send_json(status, payload)


def create_server(
    address: tuple[str, int],
    *,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    coordinator: str,
) -> JoinCoordinatorHTTPServer:
    return JoinCoordinatorHTTPServer(address, JoinCoordinatorHandler, state_dir=state_dir, coordinator=coordinator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--coordinator", default=None, help="Public coordinator URL embedded in join offers")
    args = parser.parse_args(argv)

    coordinator = args.coordinator or f"http://{args.host}:{args.port}"
    server = create_server((args.host, args.port), state_dir=args.state_dir, coordinator=coordinator)
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                "state_dir": str(Path(args.state_dir).expanduser()),
                "claim_boundary": "coordinator_service_only_no_inference_proof",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
