#!/usr/bin/env python3
"""Tiny HTTP coordinator for join offers, peer heartbeats, and joined plans.

This is bootstrap/roster/planning infrastructure only. It does not start
BloomBee servers, does not run inference, and every response carries a
no-inference-proof claim boundary.
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
    from mvp_capabilities.join_layer_plan import build_join_layer_plan, parse_seed_multiaddr
    from mvp_capabilities.route_picker import DEFAULT_PROOF_STATUS, DEFAULT_REGISTRY, explain_route, load_proof_status, load_registry
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.join_coordinator import (
        DEFAULT_STATE_DIR,
        create_join_offer,
        load_active_heartbeats,
        record_heartbeat,
    )
    from mvp_capabilities.join_layer_plan import build_join_layer_plan, parse_seed_multiaddr
    from mvp_capabilities.route_picker import DEFAULT_PROOF_STATUS, DEFAULT_REGISTRY, explain_route, load_proof_status, load_registry

HEALTH_CLAIM_BOUNDARY = "coordinator_health_only_no_inference_proof"
ERROR_CLAIM_BOUNDARY = "coordinator_error_no_inference_proof"
PLAN_SOURCE = "coordinator_http_plan_endpoint"
ROUTE_SOURCE = "coordinator_http_route_endpoint"
ROUTE_CLAIM_BOUNDARY = "coordinator_route_only_no_inference_proof"


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


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    return None


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _active_for_query(query: dict[str, list[str]], *, state_dir: str | Path, token: str) -> list[dict[str, Any]]:
    return load_active_heartbeats(
        state_dir,
        token=token,
        now=_as_int(_first(query, "now"), 0) or None,
        max_age_seconds=_as_int(_first(query, "max_age_seconds"), 30),
    )


def _peers_from_heartbeats(active_heartbeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peers: list[dict[str, Any]] = []
    for heartbeat in active_heartbeats:
        capabilities = heartbeat.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        peer = dict(capabilities)
        peer.setdefault("hostname", heartbeat.get("peer_id"))
        peer["joined_peer_id"] = heartbeat.get("peer_id")
        peer["joined_at"] = heartbeat.get("timestamp")
        peer["join_claim_boundary"] = heartbeat.get("claim_boundary")
        peers.append(peer)
    peers.sort(key=lambda peer: str(peer.get("hostname") or peer.get("joined_peer_id") or ""))
    return peers


def _route_from_query(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    active = _active_for_query(query, state_dir=state_dir, token=token)
    peers = _peers_from_heartbeats(active)
    try:
        payload = explain_route(
            peers,
            load_registry(registry),
            scenario=_first(query, "scenario"),
            requested_model=_first(query, "model"),
            proof_status=load_proof_status(_first(query, "proof_status", str(DEFAULT_PROOF_STATUS)) or DEFAULT_PROOF_STATUS),
            selector_mode=_first(query, "selector_mode", "planning") or "planning",
        )
    except ValueError as exc:
        return 400, {"error": str(exc), "claim_boundary": ERROR_CLAIM_BOUNDARY}
    payload["claim_boundary"] = ROUTE_CLAIM_BOUNDARY
    payload["source"] = ROUTE_SOURCE
    payload["token"] = token
    payload["active_peer_count"] = len(active)
    payload["active_heartbeats"] = active
    payload["inference_proven"] = False
    payload["can_update_proof_status"] = False
    return 200, payload


def _handle_plan(
    query: dict[str, list[str]],
    *,
    state_dir: str | Path,
    registry: str | Path,
) -> tuple[int, dict[str, Any]]:
    token = _first(query, "token")
    model_id = _first(query, "model")
    if not token:
        return 400, {"error": "missing token", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    if not model_id:
        return 400, {"error": "missing model", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    route_decision: dict[str, Any] | None = None
    registry_models = load_registry(registry)
    if model_id == "auto":
        route_query = {key: list(value) for key, value in query.items() if key != "model"}
        route_status, route_payload = _route_from_query(route_query, state_dir=state_dir, registry=registry)
        if route_status != 200:
            return route_status, route_payload
        route_decision = route_payload
        picked_model_id = (route_payload.get("picked") or {}).get("model_id")
        if not picked_model_id:
            return 409, {"error": "no selectable model for current joined roster", "claim_boundary": ERROR_CLAIM_BOUNDARY}
        model_id = str(picked_model_id)

    model = _find_model(registry_models, model_id)
    if model is None:
        return 404, {"error": f"model not found: {model_id}", "claim_boundary": ERROR_CLAIM_BOUNDARY}

    seed_multiaddrs: dict[str, str] = {}
    for raw in query.get("seed_multiaddr") or []:
        try:
            host, multiaddr = parse_seed_multiaddr(raw)
        except argparse.ArgumentTypeError as exc:
            return 400, {"error": str(exc), "claim_boundary": ERROR_CLAIM_BOUNDARY}
        seed_multiaddrs[host] = multiaddr

    payload = build_join_layer_plan(
        state_dir=state_dir,
        token=token,
        model=model,
        now=_as_int(_first(query, "now"), 0) or None,
        max_age_seconds=_as_int(_first(query, "max_age_seconds"), 30),
        include_launch_commands=_as_bool(_first(query, "include_launch_commands")),
        include_launch_readiness=_as_bool(_first(query, "include_launch_readiness")),
        device=_first(query, "launch_device", "mps") or "mps",
        dtype=_first(query, "launch_dtype", "float16") or "float16",
        base_port=_as_int(_first(query, "base_port"), 31337),
        dht_prefix=_first(query, "dht_prefix"),
        seed_multiaddrs=seed_multiaddrs,
    )
    payload["source"] = PLAN_SOURCE
    if route_decision is not None:
        payload["route_decision"] = route_decision
    return 200, payload


def handle_get(
    path: str,
    *,
    state_dir: str | Path,
    coordinator: str,
    registry: str | Path = DEFAULT_REGISTRY,
) -> tuple[int, dict[str, Any]]:
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
            "active_peers": _active_for_query(query, state_dir=state_dir, token=token),
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }
    if parsed.path == "/route":
        return _route_from_query(query, state_dir=state_dir, registry=registry)
    if parsed.path == "/plan":
        return _handle_plan(query, state_dir=state_dir, registry=registry)
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
    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass,
        *,
        state_dir: str | Path,
        coordinator: str,
        registry: str | Path = DEFAULT_REGISTRY,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.state_dir = Path(state_dir).expanduser()
        self.coordinator = coordinator
        self.registry = Path(registry).expanduser()


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
        status, payload = handle_get(
            self.path,
            state_dir=self.server.state_dir,
            coordinator=self.server.coordinator,
            registry=self.server.registry,
        )
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
    registry: str | Path = DEFAULT_REGISTRY,
) -> JoinCoordinatorHTTPServer:
    return JoinCoordinatorHTTPServer(
        address,
        JoinCoordinatorHandler,
        state_dir=state_dir,
        coordinator=coordinator,
        registry=registry,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--coordinator", default=None, help="Public coordinator URL embedded in join offers")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Model registry for /plan endpoint")
    args = parser.parse_args(argv)

    coordinator = args.coordinator or f"http://{args.host}:{args.port}"
    server = create_server((args.host, args.port), state_dir=args.state_dir, coordinator=coordinator, registry=args.registry)
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                "plan_endpoint": "/plan",
                "registry": str(Path(args.registry).expanduser()),
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
