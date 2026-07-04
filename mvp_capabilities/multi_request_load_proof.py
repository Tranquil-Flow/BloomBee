#!/usr/bin/env python3
"""Plan and verify multi-request BloomBee load proof evidence.

This is the proof harness for the traffic/load gate after request telemetry exists:

- `plan`: emit repeated direct-client commands against already-started servers.
- `verify`: parse captured request logs and decide whether the `multi_request_load`
  proof gate can be marked passed.

Planning mode does not send traffic. Verification mode requires the expected
number of successful direct-client results, no request-log errors, matching model
and block range, and finite forward/backward evidence for every request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range, _shell_join
    from mvp_capabilities.request_telemetry import build_request_telemetry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/multi_request_load_proof.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range, _shell_join
    from mvp_capabilities.request_telemetry import build_request_telemetry

PLAN_CLAIM_BOUNDARY = "multi_request_load_harness_only_no_live_traffic"
VERIFY_CLAIM_BOUNDARY = "verified_multi_request_load_evidence"


def _server_maddr_flags(server_maddrs: Iterable[str]) -> list[str]:
    return [f"--server-maddr '{item}'" for item in server_maddrs]


def _request_log_path(prefix: str, index: int) -> str:
    return f"{prefix}-{index:03d}.log"


def _read_log(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8", errors="replace")


def _block_range_to_list(block_range: str) -> list[int]:
    start, end = _parse_block_range(block_range)
    return [start, end]


def build_multi_request_load_plan(
    *,
    model_id: str,
    block_range: str,
    server_maddrs: list[str],
    request_count: int,
    hidden_dim: int,
    client_log_prefix: str = ".local/load-client",
) -> dict[str, Any]:
    """Emit a no-execution runbook for a repeated direct-client load proof."""
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if not server_maddrs:
        raise ValueError("at least one server multiaddr is required")
    _parse_block_range(block_range)

    client_logs = [_request_log_path(client_log_prefix, index) for index in range(request_count)]
    client_commands = [
        _shell_join(
            [
                "PYTHONPATH=.:src",
                "python scripts/direct_remote_call.py",
                *_server_maddr_flags(server_maddrs),
                f"--model {model_id}",
                f"--hidden-dim {hidden_dim}",
                f"--block-range {block_range}",
                f"2>&1 | tee {log_path}",
            ]
        )
        for log_path in client_logs
    ]
    verify_command = _shell_join(
        [
            "python mvp_capabilities/multi_request_load_proof.py verify",
            f"--model {model_id}",
            f"--block-range {block_range}",
            f"--expected-request-count {request_count}",
            *(f"--request-log {item}" for item in client_logs),
        ]
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": "multi_request_load",
        "block_range": block_range,
        "server_maddrs": server_maddrs,
        "request_count": request_count,
        "hidden_dim": hidden_dim,
        "client_logs": client_logs,
        "client_commands": client_commands,
        "verify_command": verify_command,
        "proof_status_on_success": "multi_request_load: passed",
        "notes": [
            "Planning output is not proof and does not send live traffic.",
            "Run against already-started servers whose block range covers the requested range.",
            "Do not update PROOF_STATUS.yaml until verify mode returns status=passed.",
        ],
    }


def _extract_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        result = _extract_direct_result(_read_log(path))
        if result is not None:
            results.append(result)
    return results


def verify_multi_request_load_evidence(
    *,
    model_id: str,
    block_range: str,
    request_logs: list[str | Path],
    expected_request_count: int,
) -> dict[str, Any]:
    """Verify repeated direct-client logs before promoting multi_request_load."""
    if expected_request_count <= 0:
        raise ValueError("expected_request_count must be positive")
    expected_range = _block_range_to_list(block_range)
    telemetry = build_request_telemetry(request_logs)
    results = _extract_results(request_logs)
    failed: list[str] = []

    succeeded = int((telemetry.get("request_counts") or {}).get("succeeded") or 0)
    failed_count = int((telemetry.get("request_counts") or {}).get("failed") or 0)
    if succeeded < expected_request_count:
        failed.append(f"expected {expected_request_count} successful requests, saw {succeeded}")
    if failed_count:
        noun = "request" if failed_count == 1 else "requests"
        failed.append(f"request telemetry recorded {failed_count} failed {noun}")
    latency = telemetry.get("latency_seconds") or {}
    for direction in ("forward", "backward"):
        summary = latency.get(direction) or {}
        measured = int(summary.get("count") or 0)
        unmeasured = int(summary.get("unmeasured_count") or 0)
        if measured < expected_request_count:
            failed.append(
                f"{direction} latency measured for {measured}/{expected_request_count} requests; "
                f"{unmeasured} unmeasured (0 means unmeasured, not zero)"
            )
    if len(results) < expected_request_count:
        failed.append(f"expected {expected_request_count} direct result rows, saw {len(results)}")

    for index, result in enumerate(results):
        if result.get("model") != model_id:
            failed.append(f"request {index} model mismatch")
        if result.get("block_range") != expected_range:
            failed.append(f"request {index} block range mismatch")
        if result.get("ok") is not True:
            failed.append(f"request {index} ok was not true")
        if result.get("outputs_finite") is not True:
            failed.append(f"request {index} outputs were not finite")
        if result.get("grad_finite") is not True:
            failed.append(f"request {index} gradients were not finite")

    status = "passed" if not failed else "failed"
    return {
        "model_id": model_id,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": "multi_request_load",
        "block_range": block_range,
        "expected_request_count": expected_request_count,
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {"multi_request_load": "passed"} if status == "passed" else {},
        "failed_checks": failed,
        "telemetry": telemetry,
        "request_results": results[:expected_request_count],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit commands for a multi-request load proof run")
    plan.add_argument("--model", required=True)
    plan.add_argument("--block-range", required=True)
    plan.add_argument("--server-maddr", action="append", dest="server_maddrs", required=True)
    plan.add_argument("--request-count", type=int, default=3)
    plan.add_argument("--hidden-dim", type=int, required=True)
    plan.add_argument("--client-log-prefix", default=".local/load-client")

    verify = sub.add_parser("verify", help="Verify captured multi-request load proof logs")
    verify.add_argument("--model", required=True)
    verify.add_argument("--block-range", required=True)
    verify.add_argument("--expected-request-count", type=int, required=True)
    verify.add_argument("--request-log", action="append", dest="request_logs", required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_multi_request_load_plan(
            model_id=args.model,
            block_range=args.block_range,
            server_maddrs=args.server_maddrs,
            request_count=args.request_count,
            hidden_dim=args.hidden_dim,
            client_log_prefix=args.client_log_prefix,
        )
    else:
        payload = verify_multi_request_load_evidence(
            model_id=args.model,
            block_range=args.block_range,
            request_logs=args.request_logs,
            expected_request_count=args.expected_request_count,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
