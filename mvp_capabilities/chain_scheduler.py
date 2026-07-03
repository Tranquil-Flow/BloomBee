#!/usr/bin/env python3
"""Plan multi-request chains across a joined BloomBee layer plan.

This scheduler is an operator rehearsal artifact: it turns a joined layer plan into
request waves, per-peer load estimates, and health visibility. It does not start
servers, send live requests, or prove inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "chain_scheduler_plan_only_no_inference_proof"
READY_STATUS = "ready_to_rehearse_no_live_requests"
BLOCKED_STATUS = "blocked_no_supported_layer_plan"


def _assignment_hostname(assignment: dict[str, Any], index: int) -> str:
    return str(assignment.get("hostname") or assignment.get("peer_id") or f"stage-{index}")


def _request_ids(request_count: int) -> list[str]:
    return [f"req-{index:03d}" for index in range(request_count)]


def _waves(request_ids: list[str], *, max_parallel_per_peer: int) -> list[dict[str, Any]]:
    waves: list[dict[str, Any]] = []
    for index in range(0, len(request_ids), max_parallel_per_peer):
        batch = request_ids[index : index + max_parallel_per_peer]
        waves.append(
            {
                "wave_index": len(waves),
                "request_ids": batch,
                "parallel_request_count": len(batch),
            }
        )
    return waves


def _stage_payload(assignment: dict[str, Any], *, stage_index: int) -> dict[str, Any]:
    return {
        "stage_index": stage_index,
        "hostname": _assignment_hostname(assignment, stage_index),
        "block_range": assignment.get("block_range"),
        "assigned_layers": assignment.get("assigned_layers"),
        "port": assignment.get("port"),
        "launch_command_present": bool(assignment.get("launch_command")),
    }


def _blocked_payload(model_id: str | None, *, request_count: int, prompt_tokens: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scheduler_status": BLOCKED_STATUS,
        "model_id": model_id,
        "request_count": request_count,
        "stage_count": 0,
        "wave_count": 0,
        "waves": [],
        "request_chains": [],
        "peer_health": {},
        "token_budget": {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "tokens_per_request": prompt_tokens + max_new_tokens,
            "scheduled_tokens": request_count * (prompt_tokens + max_new_tokens),
        },
        "inference_proven": False,
        "live_requests_sent": False,
        "can_update_proof_status": False,
        "next_step": "produce a supported joined layer plan before scheduling live requests",
    }


def build_chain_schedule(
    joined_layer_plan: dict[str, Any],
    *,
    request_count: int,
    max_parallel_per_peer: int = 2,
    prompt_tokens: int = 0,
    max_new_tokens: int = 0,
) -> dict[str, Any]:
    """Build a no-execution multi-request schedule from layer assignments."""
    if request_count < 0:
        raise ValueError("request_count must be >= 0")
    if max_parallel_per_peer <= 0:
        raise ValueError("max_parallel_per_peer must be > 0")
    if prompt_tokens < 0 or max_new_tokens < 0:
        raise ValueError("token budgets must be >= 0")

    model_id = joined_layer_plan.get("model_id")
    placement = joined_layer_plan.get("placement") or {}
    assignments = [item for item in placement.get("assignments") or [] if isinstance(item, dict)]
    if not placement.get("supported") or not assignments:
        return _blocked_payload(str(model_id) if model_id is not None else None, request_count=request_count, prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens)

    request_ids = _request_ids(request_count)
    waves = _waves(request_ids, max_parallel_per_peer=max_parallel_per_peer)
    stages = [_stage_payload(assignment, stage_index=index) for index, assignment in enumerate(assignments)]
    request_chains = [
        {
            "request_id": request_id,
            "chain_index": index,
            "stages": stages,
            "token_budget": {"prompt_tokens": prompt_tokens, "max_new_tokens": max_new_tokens, "total_tokens": prompt_tokens + max_new_tokens},
        }
        for index, request_id in enumerate(request_ids)
    ]
    wave_count = len(waves)
    tokens_per_request = prompt_tokens + max_new_tokens
    peer_health: dict[str, Any] = {}
    capacity_slots = max(1, wave_count * max_parallel_per_peer)
    for stage in stages:
        hostname = str(stage["hostname"])
        peak = min(max_parallel_per_peer, request_count) if request_count else 0
        peer_health[hostname] = {
            "hostname": hostname,
            "block_range": stage.get("block_range"),
            "assigned_layers": stage.get("assigned_layers"),
            "scheduled_requests": request_count,
            "scheduled_tokens": request_count * tokens_per_request,
            "peak_parallel_requests": peak,
            "max_parallel_per_peer": max_parallel_per_peer,
            "utilization_fraction": round(request_count / capacity_slots, 2) if request_count else 0.0,
            "health_status": "planned_no_live_traffic",
        }

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scheduler_status": READY_STATUS,
        "model_id": model_id,
        "request_count": request_count,
        "stage_count": len(stages),
        "wave_count": wave_count,
        "waves": waves,
        "request_chains": request_chains,
        "peer_health": peer_health,
        "token_budget": {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "tokens_per_request": tokens_per_request,
            "scheduled_tokens": request_count * tokens_per_request,
        },
        "source_claim_boundary": joined_layer_plan.get("claim_boundary"),
        "inference_proven": False,
        "live_requests_sent": False,
        "can_update_proof_status": False,
        "next_step": "start servers, send live requests through the scheduler, capture latency/error telemetry, then promote only with proof logs",
    }


def load_joined_layer_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("joined layer plan must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-layer-plan", required=True, help="JSON from join_layer_plan.py")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-parallel-per-peer", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args(argv)
    plan = load_joined_layer_plan(args.joined_layer_plan)
    schedule = build_chain_schedule(
        plan,
        request_count=args.request_count,
        max_parallel_per_peer=args.max_parallel_per_peer,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(schedule, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
