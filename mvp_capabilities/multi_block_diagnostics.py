#!/usr/bin/env python3
"""Produce operator-friendly multi-block diagnostic reports from proof logs.

This tool reads the same server/client logs as ``multi_block_proof.py verify``
but produces a detailed per-server observability report instead of a binary
pass/fail gate. It is operator diagnostics only — it does not start servers,
does not send traffic, does not promote proof status, and does not claim
inference proof.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range

CLAIM_BOUNDARY = "multi_block_diagnostics_observability_only_no_inference_proof"

_STATUS_ALL_HEALTHY = "all_servers_healthy_client_passed"
_STATUS_UNHEALTHY = "unhealthy_servers_detected"
_STATUS_CLIENT_FAILED = "client_connection_failed"
_STATUS_NO_RPC = "servers_up_no_rpc_evidence"


def _server_has_started(log: str) -> bool:
    return "Started" in log


def _server_has_announced(log: str, start: int, end: int, block_range: str) -> bool:
    for pattern in (
        f"Announced that blocks range({start}, {end}) are joining",
        f"blocks range({start}, {end})",
        f"block_indices {block_range}",
    ):
        if pattern in log:
            return True
    return False


def _server_has_rpc(log: str, block_range: str) -> bool:
    return bool(
        re.search(rf"rpc_(forward|backward)\(blocks={re.escape(block_range)}\b", log)
    )


def _is_server_healthy(log: str, start: int, end: int, block_range: str) -> bool:
    """Server is healthy if it started and announced its block range.

    RPC absence is a client-side or timing issue, not a server health failure.
    """
    return _server_has_started(log) and _server_has_announced(log, start, end, block_range)


def _server_errors(log: str, start: int, end: int, block_range: str) -> list[str]:
    errors: list[str] = []
    if not _server_has_started(log):
        errors.append("server did not reach Started state")
    if not _server_has_announced(log, start, end, block_range):
        errors.append(f"server did not announce block range {block_range}")
    if not _server_has_rpc(log, block_range):
        errors.append(f"server did not record rpc evidence for {block_range}")
    return errors


def _server_diagnostic_summary(log: str, start: int, end: int, block_range: str) -> dict[str, Any]:
    errors = _server_errors(log, start, end, block_range)
    health = "healthy" if _is_server_healthy(log, start, end, block_range) else "unhealthy"
    return {
        "block_range": block_range,
        "start_layer": start,
        "end_layer": end,
        "layer_count": end - start,
        "started": _server_has_started(log),
        "announced_block_range": _server_has_announced(log, start, end, block_range),
        "has_rpc_evidence": _server_has_rpc(log, block_range),
        "errors": errors,
        "health": health,
    }


def _combined_range(block_ranges: list[str]) -> str:
    parsed = sorted(_parse_block_range(item) for item in block_ranges)
    cursor = parsed[0][0]
    for start, end in parsed:
        if start != cursor:
            raise ValueError(f"block ranges must be contiguous, got {block_ranges!r}")
        cursor = end
    return f"{parsed[0][0]}:{cursor}"


def _operator_actions(
    *,
    unhealthy_servers: dict[str, list[str]],
    unresolved_blocks: list[str],
    client_failed: bool,
) -> list[str]:
    actions: list[str] = []
    for hostname, errors in unhealthy_servers.items():
        actions.append(
            f"server {hostname}: {', '.join(errors)}. "
            f"Check server logs for crash/port/block-index details."
        )
    if unresolved_blocks:
        actions.append(
            f"Unresolved block range(s): {', '.join(unresolved_blocks)}. "
            f"Start or re-announce servers for these layers before running proof clients."
        )
    if client_failed:
        actions.append(
            "Client could not reach the DHT swarm or all servers. "
            "Verify BLOOMBEE_INITIAL_PEERS, server multiaddrs, and DHT prefix match."
        )
    if not actions:
        actions.append(
            "All servers healthy; client result ok. "
            "Ready for multi_block_proof.py verify to promote proof status."
        )
    return actions


def build_multi_block_diagnostics(
    *,
    model_id: str,
    block_ranges: list[str],
    server_logs: list[str],
    client_log: str,
) -> dict[str, Any]:
    """Build an operator diagnostic report from multi-block proof logs."""
    if len(block_ranges) < 2:
        raise ValueError("multi_block diagnostics requires at least two block ranges")

    # Pad missing server logs with empty strings.
    padded_logs = list(server_logs) + [""] * (len(block_ranges) - len(server_logs))

    combined = _combined_range(block_ranges)
    parsed_ranges = [_parse_block_range(item) for item in block_ranges]

    servers: list[dict[str, Any]] = []
    unhealthy: dict[str, list[str]] = {}
    for index, (block_range, log) in enumerate(zip(block_ranges, padded_logs, strict=True)):
        start, end = parsed_ranges[index]
        summary = _server_diagnostic_summary(log, start, end, block_range)
        summary["server_index"] = index
        servers.append(summary)
        if summary["health"] == "unhealthy":
            unhealthy[f"server {index} ({block_range})"] = summary["errors"]

    client_result = _extract_direct_result(client_log)
    client_failed = client_result is None or client_result.get("ok") is not True

    covered_layers = sum(
        end - start
        for index, (start, end) in enumerate(parsed_ranges)
        if servers[index]["health"] == "healthy"
    )
    total_layers = sum(end - start for start, end in parsed_ranges)
    missing_layers = total_layers - covered_layers
    # Full coverage requires both healthy-server layer coverage AND verified client result.
    full_coverage = missing_layers == 0 and covered_layers > 0 and not client_failed

    unresolved_blocks = [
        block_ranges[index]
        for index in range(len(block_ranges))
        if servers[index]["health"] == "unhealthy"
    ]

    healthy_servers = sum(1 for srv in servers if srv["health"] == "healthy")
    unhealthy_servers = sum(1 for srv in servers if srv["health"] == "unhealthy")

    if client_failed and healthy_servers < len(servers):
        status = _STATUS_UNHEALTHY
    elif client_failed:
        status = _STATUS_CLIENT_FAILED
    elif healthy_servers == len(servers) and not any(srv["has_rpc_evidence"] for srv in servers):
        status = _STATUS_NO_RPC
    elif unhealthy_servers > 0:
        status = _STATUS_UNHEALTHY
    else:
        status = _STATUS_ALL_HEALTHY

    return {
        "source": "multi_block_diagnostics.py",
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model_id,
        "combined_block_range": combined,
        "block_ranges": block_ranges,
        "server_count": len(servers),
        "servers": servers,
        "client_result": client_result,
        "coverage": {
            "covered_layers": covered_layers,
            "total_layers": total_layers,
            "missing_layers": missing_layers,
            "full_coverage": full_coverage,
        },
        "summary": {
            "healthy_servers": healthy_servers,
            "unhealthy_servers": unhealthy_servers,
            "status": status,
        },
        "operator_actions": _operator_actions(
            unhealthy_servers=unhealthy,
            unresolved_blocks=unresolved_blocks,
            client_failed=client_failed,
        ),
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": (
            "Fix unhealthy servers, re-run proof clients, then use "
            "multi_block_proof.py verify to promote proof status."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--block-range", action="append", dest="block_ranges", required=True)
    parser.add_argument("--server-log", action="append", dest="server_logs", required=True)
    parser.add_argument("--client-log", required=True)
    args = parser.parse_args(argv)

    report = build_multi_block_diagnostics(
        model_id=args.model,
        block_ranges=args.block_ranges,
        server_logs=[Path(item).read_text(encoding="utf-8") for item in args.server_logs],
        client_log=Path(args.client_log).read_text(encoding="utf-8"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
