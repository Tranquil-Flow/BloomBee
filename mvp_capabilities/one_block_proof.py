#!/usr/bin/env python3
"""Plan and verify one-block BloomBee proof evidence.

This tool has two modes:

- `plan`: emit exact server/client commands for a one-block proof run.
- `verify`: parse captured server/client logs and decide whether the
  `one_block_server` proof gate can be marked passed.

Planning mode is only a harness and does not prove inference. Verification mode
requires both server evidence and direct-client evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/one_block_proof.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry

PLAN_CLAIM_BOUNDARY = "proof_harness_only_no_live_inference"
VERIFY_CLAIM_BOUNDARY = "verified_one_block_server_evidence"


def _find_model(model_id: str, registry: list[dict[str, Any]]) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise ValueError(f"model {model_id!r} not found in registry")


def _parse_block_range(block_range: str) -> tuple[int, int]:
    try:
        start, end = block_range.split(":", 1)
        return int(start), int(end)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"block_range must be START:END, got {block_range!r}") from exc


def _shell_join(parts: list[str]) -> str:
    # Inputs are controlled model ids/flags, not arbitrary shell payloads. Keep
    # this readable for copy/paste in operator docs.
    return " ".join(parts)


def build_one_block_plan(
    model_id: str,
    *,
    registry: list[dict[str, Any]] | None = None,
    block_range: str = "0:1",
    port: int = 31337,
    device: str = "mps",
    dtype: str = "float16",
    server_log: str = ".local/one-block-server.log",
    client_log: str = ".local/one-block-client.log",
) -> dict[str, Any]:
    registry = registry if registry is not None else load_registry(DEFAULT_REGISTRY)
    model = _find_model(model_id, registry)
    hidden_size = int(model.get("hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError(f"model {model_id!r} has no hidden_size in registry")
    start, end = _parse_block_range(block_range)

    server_command = _shell_join(
        [
            "PYTHONPATH=.:src",
            "python -m bloombee.cli.run_server",
            model_id,
            "--new_swarm",
            f"--block_indices {block_range}",
            f"--device {device}",
            f"--torch_dtype {dtype}",
            f"--port {port}",
            f"2>&1 | tee {server_log}",
        ]
    )
    client_command = _shell_join(
        [
            "PYTHONPATH=.:src",
            "python scripts/direct_remote_call.py",
            "--server-maddr '<PASTE_SERVER_MULTIADDR>'",
            f"--model {model_id}",
            f"--hidden-dim {hidden_size}",
            f"--block-range {block_range}",
            f"2>&1 | tee {client_log}",
        ]
    )
    verify_command = _shell_join(
        [
            "python mvp_capabilities/one_block_proof.py verify",
            f"--model {model_id}",
            f"--block-range {block_range}",
            f"--server-log {server_log}",
            f"--client-log {client_log}",
        ]
    )

    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "block_range": block_range,
        "start_block": start,
        "end_block": end,
        "hidden_size": hidden_size,
        "device": device,
        "dtype": dtype,
        "server_log": server_log,
        "client_log": client_log,
        "server_command": server_command,
        "client_command": client_command,
        "verify_command": verify_command,
        "proof_gate": "one_block_server",
        "proof_status_on_success": "one_block_server: passed",
        "notes": [
            "Planning output is not proof.",
            "Do not update PROOF_STATUS.yaml until verify mode returns status=passed.",
            "The client command needs the server multiaddr printed by run_server.",
        ],
    }


def _extract_direct_result(client_log: str) -> dict[str, Any] | None:
    matches = re.findall(r"\[direct\] RESULT:\s*(\{.*\})", client_log)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def verify_one_block_evidence(
    *,
    model_id: str,
    block_range: str,
    server_log: str,
    client_log: str,
) -> dict[str, Any]:
    start, end = _parse_block_range(block_range)
    expected_range = [start, end]
    failed: list[str] = []

    if "Started" not in server_log:
        failed.append("server did not reach Started state")
    announced_patterns = [
        f"Announced that blocks range({start}, {end}) are joining",
        f"blocks range({start}, {end})",
        f"block_indices {block_range}",
    ]
    if not any(pattern in server_log for pattern in announced_patterns):
        failed.append("server did not announce requested block range")

    direct_result = _extract_direct_result(client_log)
    if direct_result is None:
        failed.append("client log did not contain [direct] RESULT JSON")
    else:
        if direct_result.get("model") != model_id:
            failed.append("client result model mismatch")
        if direct_result.get("block_range") != expected_range:
            failed.append("client result block range mismatch")
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
        "proof_gate": "one_block_server",
        "block_range": block_range,
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {"one_block_server": "passed"} if status == "passed" else {},
        "failed_checks": failed,
        "client_result": direct_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit commands for a one-block proof run")
    plan.add_argument("--model", required=True)
    plan.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    plan.add_argument("--block-range", default="0:1")
    plan.add_argument("--port", type=int, default=31337)
    plan.add_argument("--device", default="mps")
    plan.add_argument("--dtype", default="float16")
    plan.add_argument("--server-log", default=".local/one-block-server.log")
    plan.add_argument("--client-log", default=".local/one-block-client.log")

    verify = sub.add_parser("verify", help="Verify captured one-block proof logs")
    verify.add_argument("--model", required=True)
    verify.add_argument("--block-range", default="0:1")
    verify.add_argument("--server-log", required=True)
    verify.add_argument("--client-log", required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_one_block_plan(
            args.model,
            registry=load_registry(args.registry),
            block_range=args.block_range,
            port=args.port,
            device=args.device,
            dtype=args.dtype,
            server_log=args.server_log,
            client_log=args.client_log,
        )
    else:
        payload = verify_one_block_evidence(
            model_id=args.model,
            block_range=args.block_range,
            server_log=Path(args.server_log).read_text(encoding="utf-8"),
            client_log=Path(args.client_log).read_text(encoding="utf-8"),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
