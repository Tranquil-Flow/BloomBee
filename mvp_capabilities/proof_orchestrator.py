#!/usr/bin/env python3
"""Build an operator proof-orchestration plan from a coordinator handoff bundle.

This module intentionally does **not** start BloomBee servers, send client traffic,
or update proof status. It stitches the no-execution handoff artifacts together
into an ordered checklist so operators can see what must be resolved before the
multi-block/full-generation/cache/load proof harnesses are run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "proof_orchestration_plan_only_no_live_inference"
SOURCE = "proof_orchestrator.py"
HANDOFF_EMBEDDED_SOURCE = "coordinator_handoff_embedded_proof_orchestration"
PHASE_ORDER = [
    "start_servers",
    "capture_server_multiaddrs",
    "run_proof_clients",
    "verify_then_promote_manually",
]
PROOF_GATE_ORDER = ["multi_block", "full_generation", "cache_generation", "multi_request_load"]
_PLACEHOLDER_RE = re.compile(r"<[^<>\s]+>")


def _find_placeholders(value: Any) -> list[str]:
    """Return sorted placeholder tokens such as ``<PASTE_SERVER_0_MULTIADDR>``."""
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            found.update(_PLACEHOLDER_RE.findall(item))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _assignments_from_handoff(handoff_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _as_dict(handoff_bundle.get("plan"))
    placement = _as_dict(plan.get("placement"))
    return [item for item in placement.get("assignments") or [] if isinstance(item, dict)]


def _model_id_from_handoff(handoff_bundle: dict[str, Any]) -> str | None:
    plan = _as_dict(handoff_bundle.get("plan"))
    model_id = plan.get("model_id")
    if model_id:
        return str(model_id)
    route = _as_dict(handoff_bundle.get("route_decision"))
    picked = _as_dict(route.get("picked"))
    if picked.get("model_id"):
        return str(picked["model_id"])
    return None


def _launch_step(assignment: dict[str, Any], index: int) -> tuple[dict[str, Any], list[str]]:
    hostname = str(assignment.get("hostname") or assignment.get("peer_id") or f"server-{index}")
    command = str(assignment.get("launch_command") or "")
    role = "seed" if index == 0 else "follower"
    placeholders = _find_placeholders(command)
    forbidden_flags: list[str] = []
    fix_hint = ""
    follower_missing_env = False
    if "--initial_peers" in command:
        forbidden_flags.append(f"launch step {hostname} uses forbidden --initial_peers")
        fix_hint = "Use BLOOMBEE_INITIAL_PEERS=<seed multiaddr> before python -m bloombee.cli.run_server for follower runbooks."
    if role == "follower" and command and "BLOOMBEE_INITIAL_PEERS=" not in command:
        follower_missing_env = True
        if not fix_hint:
            fix_hint = "Follower runbooks must use BLOOMBEE_INITIAL_PEERS=<seed multiaddr>."
    blocked_by: list[str] = []
    if not command:
        blocked_by.append("missing_launch_command")
    if placeholders:
        blocked_by.append("unresolved_placeholders")
    if forbidden_flags:
        blocked_by.append("forbidden_initial_peers_flag")
    if follower_missing_env:
        blocked_by.append("missing_bloombee_initial_peers_env")
    ready = bool(command) and not blocked_by
    return (
        {
            "step_id": f"launch-{index:02d}",
            "phase": "start_servers",
            "role": role,
            "hostname": hostname,
            "block_range": assignment.get("block_range"),
            "port": assignment.get("port"),
            "command": command,
            "placeholders": placeholders,
            "ready": ready,
            "blocked_by": blocked_by,
            "fix_hint": fix_hint,
            "claim_boundary": "launch_step_operator_command_only_no_server_started",
        },
        forbidden_flags,
    )


def _ordered_runbooks(raw_runbooks: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    ordered: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for gate in PROOF_GATE_ORDER:
        runbook = raw_runbooks.get(gate)
        if isinstance(runbook, dict):
            ordered.append((gate, runbook))
            seen.add(gate)
    for gate, runbook in sorted(raw_runbooks.items()):
        if gate not in seen and isinstance(runbook, dict):
            ordered.append((str(gate), runbook))
    return ordered


def _commands_for_runbook(gate: str, runbook: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    if gate == "multi_block":
        for key in ("client_command", "verify_command"):
            if isinstance(runbook.get(key), str):
                commands.append(runbook[key])
        return commands
    if gate in {"full_generation", "cache_generation"}:
        for key in ("parity_command", "verify_command"):
            if isinstance(runbook.get(key), str):
                commands.append(runbook[key])
        return commands
    if gate == "multi_request_load":
        for command in runbook.get("client_commands") or []:
            if isinstance(command, str):
                commands.append(command)
        if isinstance(runbook.get("verify_command"), str):
            commands.append(runbook["verify_command"])
        return commands

    for key, value in sorted(runbook.items()):
        if key.endswith("_command") and isinstance(value, str):
            commands.append(value)
        elif key.endswith("_commands") and isinstance(value, list):
            commands.extend(item for item in value if isinstance(item, str))
    return commands


def _proof_step(gate: str, runbook: dict[str, Any]) -> dict[str, Any]:
    status = str(runbook.get("status") or "planned")
    reason = runbook.get("reason")
    commands = _commands_for_runbook(gate, runbook)
    placeholders = _find_placeholders(commands)
    blocked_by: list[str] = []
    if status == "unavailable":
        blocked_by.append("runbook_unavailable")
    if not commands:
        blocked_by.append("missing_commands")
    if placeholders:
        blocked_by.append("unresolved_placeholders")
    ready = status != "unavailable" and bool(commands) and not placeholders
    return {
        "phase": "run_proof_clients",
        "proof_gate": str(runbook.get("proof_gate") or gate),
        "status": status,
        "reason": reason,
        "claim_boundary": runbook.get("claim_boundary") or runbook.get("status") or "proof_runbook_only_no_live_execution",
        "commands": commands,
        "command_count": len(commands),
        "placeholders": placeholders,
        "ready": ready,
        "blocked_by": blocked_by,
        "proof_status_on_success": runbook.get("proof_status_on_success"),
        "requires_verify_status_passed": True,
    }


def _multiaddr_capture_steps(launch_steps: list[dict[str, Any]], proof_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    placeholders = sorted({item for step in proof_steps for item in step.get("placeholders", []) if "MULTIADDR" in item})
    steps: list[dict[str, Any]] = []
    for index, placeholder in enumerate(placeholders):
        launch = launch_steps[index] if index < len(launch_steps) else {}
        steps.append(
            {
                "step_id": f"capture-multiaddr-{index:02d}",
                "phase": "capture_server_multiaddrs",
                "placeholder": placeholder,
                "hostname": launch.get("hostname"),
                "block_range": launch.get("block_range"),
                "operator_action": "paste the server multiaddr printed by the corresponding run_server process into proof commands",
                "claim_boundary": "server_multiaddr_capture_instruction_only_no_network_proof",
            }
        )
    return steps


def _summary(
    *,
    launch_steps: list[dict[str, Any]],
    proof_steps: list[dict[str, Any]],
    launch_readiness: dict[str, Any],
    forbidden_flags: list[str],
) -> dict[str, Any]:
    unresolved = sorted({item for step in [*launch_steps, *proof_steps] for item in step.get("placeholders", [])})
    available = [step["proof_gate"] for step in proof_steps if step.get("status") != "unavailable"]
    blocked = {
        step["proof_gate"]: step.get("reason") or ", ".join(step.get("blocked_by") or [])
        for step in proof_steps
        if not step.get("ready")
    }
    readiness_value = launch_readiness.get("ready_to_start")
    if isinstance(readiness_value, bool):
        ready_to_start_servers = readiness_value and not forbidden_flags
    else:
        ready_to_start_servers = bool(launch_steps) and all(step.get("ready") for step in launch_steps) and not forbidden_flags
    ready_for_proof_clients = ready_to_start_servers and bool(proof_steps) and all(step.get("ready") for step in proof_steps)
    if forbidden_flags:
        orchestration_status = "blocked_forbidden_launch_flags"
    elif unresolved:
        orchestration_status = "blocked_unresolved_placeholders"
    elif not proof_steps:
        orchestration_status = "blocked_no_proof_runbooks"
    else:
        orchestration_status = "ready_for_manual_operator_execution_no_proof"
    return {
        "orchestration_status": orchestration_status,
        "server_count": len(launch_steps),
        "proof_step_count": len(proof_steps),
        "ready_to_start_servers": ready_to_start_servers,
        "ready_for_proof_clients": ready_for_proof_clients,
        "unresolved_placeholders": unresolved,
        "forbidden_flags": forbidden_flags,
        "available_proof_gates": available,
        "blocked_proof_gates": blocked,
        "launch_readiness_claim_boundary": launch_readiness.get("claim_boundary"),
    }


def build_proof_orchestration_plan(
    handoff_bundle: dict[str, Any],
    *,
    source: str = SOURCE,
) -> dict[str, Any]:
    """Create an ordered, no-execution operator proof plan from a handoff bundle."""
    if not isinstance(handoff_bundle, dict):
        raise ValueError("handoff bundle must be a JSON object")
    plan = _as_dict(handoff_bundle.get("plan"))
    launch_readiness = _as_dict(plan.get("launch_readiness"))
    assignments = _assignments_from_handoff(handoff_bundle)
    launch_steps: list[dict[str, Any]] = []
    forbidden_flags: list[str] = []
    for index, assignment in enumerate(assignments):
        step, step_forbidden = _launch_step(assignment, index)
        launch_steps.append(step)
        forbidden_flags.extend(step_forbidden)

    raw_runbooks = _as_dict(handoff_bundle.get("proof_runbooks"))
    proof_steps = [_proof_step(gate, runbook) for gate, runbook in _ordered_runbooks(raw_runbooks)]
    capture_steps = _multiaddr_capture_steps(launch_steps, proof_steps)
    summary = _summary(
        launch_steps=launch_steps,
        proof_steps=proof_steps,
        launch_readiness=launch_readiness,
        forbidden_flags=forbidden_flags,
    )

    return {
        "source": source,
        "claim_boundary": CLAIM_BOUNDARY,
        "handoff_source": handoff_bundle.get("source"),
        "handoff_claim_boundary": handoff_bundle.get("claim_boundary"),
        "model_id": _model_id_from_handoff(handoff_bundle),
        "phase_order": list(PHASE_ORDER),
        "summary": summary,
        "launch_steps": launch_steps,
        "multiaddr_capture_steps": capture_steps,
        "proof_steps": proof_steps,
        "operator_next_steps": [
            "start launch_steps on the named hosts only after ready_to_start_servers is true",
            "capture each server multiaddr and replace matching <PASTE_SERVER_N_MULTIADDR> placeholders",
            "run proof client commands, then run each verify command and require status=passed",
            "promote PROOF_STATUS.yaml only from verifier output where can_update_proof_status=true",
        ],
        "live_commands_executed": False,
        "proof_status_updates_applied": False,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def load_handoff_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("handoff bundle must be a JSON object")
    return payload


def write_orchestration_artifact(plan: dict[str, Any], out: str | Path) -> Path:
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-bundle", required=True, help="JSON from join_http_server.py /handoff or join_handoff.py")
    parser.add_argument("--out", default=None, help="Optional path to write the orchestration JSON artifact")
    args = parser.parse_args(argv)

    plan = build_proof_orchestration_plan(load_handoff_bundle(args.handoff_bundle))
    if args.out:
        write_orchestration_artifact(plan, args.out)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
