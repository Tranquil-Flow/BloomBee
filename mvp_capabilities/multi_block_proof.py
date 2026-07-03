#!/usr/bin/env python3
"""Plan and verify multi-block BloomBee proof evidence.

This tool is the next proof gate after one-block serving:

- `plan`: emit exact commands for two or more block servers plus one direct client.
- `verify`: parse captured server/client logs and decide whether the `multi_block`
  proof gate can be marked passed.

Planning mode is only a harness and does not prove inference. Verification mode
requires each server to start, announce its requested block range, record RPC
evidence, and a direct-client result over the combined contiguous range.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range, _shell_join
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/multi_block_proof.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.one_block_proof import _extract_direct_result, _parse_block_range, _shell_join
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry

PLAN_CLAIM_BOUNDARY = "multi_block_proof_harness_only_no_live_inference"
VERIFY_CLAIM_BOUNDARY = "verified_multi_block_server_evidence"


def _find_model(model_id: str, registry: list[dict[str, Any]]) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise ValueError(f"model {model_id!r} not found in registry")


def _combined_range(block_ranges: list[str]) -> str:
    parsed = [_parse_block_range(item) for item in block_ranges]
    if len(parsed) < 2:
        raise ValueError("multi_block proof requires at least two block ranges")
    parsed_sorted = sorted(parsed)
    expected_start = parsed_sorted[0][0]
    cursor = expected_start
    for start, end in parsed_sorted:
        if start != cursor:
            raise ValueError(f"block ranges must be contiguous, got {block_ranges!r}")
        if end <= start:
            raise ValueError(f"block range end must exceed start, got {start}:{end}")
        cursor = end
    return f"{expected_start}:{cursor}"


def build_multi_block_plan(
    model_id: str,
    *,
    registry: list[dict[str, Any]] | None = None,
    block_ranges: list[str] | None = None,
    ports: list[int] | None = None,
    device: str = "mps",
    dtype: str = "float16",
    server_log_prefix: str = ".local/multi-block-server",
    client_log: str = ".local/multi-block-client.log",
) -> dict[str, Any]:
    registry = registry if registry is not None else load_registry(DEFAULT_REGISTRY)
    model = _find_model(model_id, registry)
    hidden_size = int(model.get("hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError(f"model {model_id!r} has no hidden_size in registry")
    block_ranges = block_ranges or ["0:1", "1:2"]
    combined = _combined_range(block_ranges)
    ports = ports or [31337 + index for index in range(len(block_ranges))]
    if len(ports) != len(block_ranges):
        raise ValueError("ports length must match block_ranges length")

    server_commands: list[str] = []
    server_logs: list[str] = []
    for index, (block_range, port) in enumerate(zip(block_ranges, ports, strict=True)):
        log_path = f"{server_log_prefix}-{index}.log"
        server_logs.append(log_path)
        if index == 0:
            prefix_parts = ["PYTHONPATH=.:src"]
            swarm_args = ["--new_swarm"]
        else:
            prefix_parts = ["PYTHONPATH=.:src", "BLOOMBEE_INITIAL_PEERS='<PASTE_SEED_MULTIADDR>'"]
            swarm_args = []
        server_commands.append(
            _shell_join(
                [
                    *prefix_parts,
                    "python -m bloombee.cli.run_server",
                    model_id,
                    *swarm_args,
                    f"--block_indices {block_range}",
                    f"--device {device}",
                    f"--torch_dtype {dtype}",
                    f"--port {port}",
                    f"2>&1 | tee {log_path}",
                ]
            )
        )

    server_maddrs = [f"--server-maddr '<PASTE_SERVER_{index}_MULTIADDR>'" for index in range(len(block_ranges))]
    client_command = _shell_join(
        [
            "PYTHONPATH=.:src",
            "python scripts/direct_remote_call.py",
            *server_maddrs,
            f"--model {model_id}",
            f"--hidden-dim {hidden_size}",
            f"--block-range {combined}",
            f"2>&1 | tee {client_log}",
        ]
    )
    verify_command = _shell_join(
        [
            "python mvp_capabilities/multi_block_proof.py verify",
            f"--model {model_id}",
            *(f"--block-range {item}" for item in block_ranges),
            *(f"--server-log {item}" for item in server_logs),
            f"--client-log {client_log}",
        ]
    )

    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": "multi_block",
        "block_ranges": block_ranges,
        "combined_block_range": combined,
        "hidden_size": hidden_size,
        "device": device,
        "dtype": dtype,
        "server_logs": server_logs,
        "client_log": client_log,
        "server_commands": server_commands,
        "client_command": client_command,
        "verify_command": verify_command,
        "proof_status_on_success": "multi_block: passed",
        "notes": [
            "Planning output is not proof.",
            "Server 0 starts a private swarm; later servers bootstrap to server 0's multiaddr.",
            "The client command needs every server multiaddr so RemoteSequential can discover all blocks.",
            "Do not update PROOF_STATUS.yaml until verify mode returns status=passed.",
        ],
    }


def _server_has_announced(log: str, start: int, end: int, block_range: str) -> bool:
    return any(
        pattern in log
        for pattern in (
            f"Announced that blocks range({start}, {end}) are joining",
            f"blocks range({start}, {end})",
            f"block_indices {block_range}",
        )
    )


def _server_has_rpc(log: str, block_range: str) -> bool:
    return (
        f"rpc_forward(blocks={block_range}" in log
        or f"rpc_backward(blocks={block_range}" in log
        or re.search(rf"rpc_(forward|backward)\(blocks={re.escape(block_range)}\b", log) is not None
    )


def verify_multi_block_evidence(
    *,
    model_id: str,
    block_ranges: list[str],
    server_logs: list[str],
    client_log: str,
) -> dict[str, Any]:
    failed: list[str] = []
    combined = _combined_range(block_ranges)
    combined_start, combined_end = _parse_block_range(combined)
    expected_range = [combined_start, combined_end]

    if len(server_logs) != len(block_ranges):
        failed.append("server log count does not match block range count")

    for index, block_range in enumerate(block_ranges):
        start, end = _parse_block_range(block_range)
        log = server_logs[index] if index < len(server_logs) else ""
        if "Started" not in log:
            failed.append(f"server {index} did not reach Started state")
        if not _server_has_announced(log, start, end, block_range):
            failed.append(f"server {index} did not announce block range {block_range}")
        if not _server_has_rpc(log, block_range):
            failed.append(f"server {index} did not record rpc evidence for {block_range}")

    direct_result = _extract_direct_result(client_log)
    if direct_result is None:
        failed.append("client log did not contain [direct] RESULT JSON")
    else:
        if direct_result.get("model") != model_id:
            failed.append("client result model mismatch")
        if direct_result.get("block_range") != expected_range:
            failed.append("client result combined block range mismatch")
        if direct_result.get("ok") is not True:
            failed.append("client result ok was not true")
        if direct_result.get("outputs_finite") is not True:
            failed.append("client outputs were not finite")
        if direct_result.get("grad_finite") is not True:
            failed.append("client gradients were not finite")

    status = "passed" if not failed else "failed"
    return {
        "model_id": model_id,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": "multi_block",
        "block_ranges": block_ranges,
        "combined_block_range": combined,
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {"multi_block": "passed"} if status == "passed" else {},
        "failed_checks": failed,
        "client_result": direct_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit commands for a multi-block proof run")
    plan.add_argument("--model", required=True)
    plan.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    plan.add_argument("--block-range", action="append", dest="block_ranges", default=None)
    plan.add_argument("--port", action="append", dest="ports", type=int, default=None)
    plan.add_argument("--device", default="mps")
    plan.add_argument("--dtype", default="float16")
    plan.add_argument("--server-log-prefix", default=".local/multi-block-server")
    plan.add_argument("--client-log", default=".local/multi-block-client.log")

    verify = sub.add_parser("verify", help="Verify captured multi-block proof logs")
    verify.add_argument("--model", required=True)
    verify.add_argument("--block-range", action="append", dest="block_ranges", required=True)
    verify.add_argument("--server-log", action="append", dest="server_logs", required=True)
    verify.add_argument("--client-log", required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_multi_block_plan(
            args.model,
            registry=load_registry(args.registry),
            block_ranges=args.block_ranges,
            ports=args.ports,
            device=args.device,
            dtype=args.dtype,
            server_log_prefix=args.server_log_prefix,
            client_log=args.client_log,
        )
    else:
        payload = verify_multi_block_evidence(
            model_id=args.model,
            block_ranges=args.block_ranges,
            server_logs=[Path(item).read_text(encoding="utf-8") for item in args.server_logs],
            client_log=Path(args.client_log).read_text(encoding="utf-8"),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
