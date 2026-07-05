#!/usr/bin/env python3
"""Claim-bounded continuous-batching scheduler simulation.

This module is intentionally pure and dependency-free. It proves the core
scheduler semantics needed before wiring live BloomBee decode loops:
late-arriving requests can join later decode ticks, per-request token order is
preserved after deinterleaving, and fairness prevents a long request from
monopolizing a bounded batch. It is not a live server proof and must not be used
as a speedup/demo promotion by itself.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

CLAIM_BOUNDARY = "continuous_batching_scheduler_simulation_no_live_server_proof"
SOURCE = "continuous_batching.py"

OPERATOR_NEXT_STEPS = [
    "wire the scheduler into the live decode request loop behind an opt-in flag",
    "run same-prompt parity against verifier-only decode with concurrent arrivals",
    "measure wall-clock throughput before any demo or speedup promotion",
]


@dataclass(frozen=True)
class DecodeRequest:
    """Small deterministic request model for continuous decode scheduling."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    arrival_tick: int = 0

    def __init__(
        self,
        *,
        request_id: str,
        prompt_token_ids: Sequence[int],
        target_token_ids: Sequence[int],
        arrival_tick: int = 0,
    ) -> None:
        object.__setattr__(self, "request_id", str(request_id))
        object.__setattr__(self, "prompt_token_ids", tuple(int(token) for token in prompt_token_ids))
        object.__setattr__(self, "target_token_ids", tuple(int(token) for token in target_token_ids))
        object.__setattr__(self, "arrival_tick", int(arrival_tick))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.arrival_tick < 0:
            raise ValueError("arrival_tick must be non-negative")


def build_padded_batch(sequences: Sequence[Sequence[int]], *, pad_token_id: int = 0) -> dict[str, list[list[int]] | list[int]]:
    """Build list-backed padded input IDs plus an attention mask.

    Keeping this helper independent from torch makes the scheduler proof cheap to
    run in docs/CI and avoids implying a live model path was exercised.
    """
    normalized = [list(map(int, sequence)) for sequence in sequences]
    if not normalized:
        return {"input_ids": [], "attention_mask": [], "sequence_lengths": []}
    max_len = max(len(sequence) for sequence in normalized)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    lengths: list[int] = []
    for sequence in normalized:
        length = len(sequence)
        lengths.append(length)
        pad_count = max_len - length
        padded.append(sequence + [int(pad_token_id)] * pad_count)
        masks.append([1] * length + [0] * pad_count)
    return {"input_ids": padded, "attention_mask": masks, "sequence_lengths": lengths}


def _validate_requests(requests: Sequence[DecodeRequest]) -> None:
    seen: set[str] = set()
    for request in requests:
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        seen.add(request.request_id)


def _example_requests(name: str) -> list[DecodeRequest]:
    if name != "staggered":
        raise ValueError(f"unknown example: {name}")
    return [
        DecodeRequest(request_id="req-a", prompt_token_ids=(101,), target_token_ids=(10, 11, 12), arrival_tick=0),
        DecodeRequest(request_id="req-b", prompt_token_ids=(201, 202), target_token_ids=(20, 21), arrival_tick=1),
    ]


def simulate_continuous_decode(
    *,
    requests: Sequence[DecodeRequest],
    max_batch_size: int,
) -> dict[str, object]:
    """Simulate continuous decode scheduling for deterministic target tokens.

    Each timeline row is one decode batch. Requests are admitted when their
    arrival tick is reached; active requests are served round-robin so bounded
    batches cannot be monopolized by one long request. The simulator emits the
    previous generated token (or final prompt token for position 0) as each
    request's next input token, mirroring autoregressive decode step inputs.
    """
    max_batch = int(max_batch_size)
    if max_batch < 1:
        raise ValueError("max_batch_size must be positive")
    _validate_requests(requests)

    ordered = sorted(enumerate(requests), key=lambda item: (item[1].arrival_tick, item[0]))
    pending = deque(request for _, request in ordered)
    state = {
        request.request_id: {
            "request": request,
            "position": 0,
            "generated": [],
        }
        for request in requests
    }
    active: deque[str] = deque()
    active_set: set[str] = set()
    completed: list[str] = []
    timeline: list[dict[str, object]] = []
    tick = pending[0].arrival_tick if pending else 0

    while pending or active:
        while pending and pending[0].arrival_tick <= tick:
            request = pending.popleft()
            if len(request.target_token_ids) == 0:
                completed.append(request.request_id)
                continue
            active.append(request.request_id)
            active_set.add(request.request_id)

        if not active:
            if pending:
                tick = max(tick + 1, pending[0].arrival_tick)
                continue
            break

        batch_request_ids: list[str] = []
        for _ in range(max_batch):
            if not active:
                break
            request_id = active.popleft()
            active_set.remove(request_id)
            batch_request_ids.append(request_id)

        positions: list[int] = []
        input_token_ids: list[int] = []
        output_token_ids: list[int] = []
        finished_request_ids: list[str] = []
        requeue: list[str] = []

        for request_id in batch_request_ids:
            request_state = state[request_id]
            request = request_state["request"]
            position = int(request_state["position"])
            generated = request_state["generated"]
            positions.append(position)
            input_token_ids.append(int(request.prompt_token_ids[-1] if position == 0 else generated[-1]))
            next_token = int(request.target_token_ids[position])
            output_token_ids.append(next_token)
            generated.append(next_token)
            request_state["position"] = position + 1
            if position + 1 >= len(request.target_token_ids):
                finished_request_ids.append(request_id)
                completed.append(request_id)
            else:
                requeue.append(request_id)

        for request_id in requeue:
            if request_id not in active_set:
                active.append(request_id)
                active_set.add(request_id)

        timeline.append(
            {
                "tick": tick,
                "request_ids": batch_request_ids,
                "positions": positions,
                "input_token_ids": input_token_ids,
                "output_token_ids": output_token_ids,
                "finished_request_ids": finished_request_ids,
            }
        )
        tick += 1

    outputs_by_request = {
        request.request_id: list(state[request.request_id]["generated"])
        for request in requests
    }
    expected_by_request = {request.request_id: list(request.target_token_ids) for request in requests}
    serial_batches = sum(len(request.target_token_ids) for request in requests)
    total_decode_batches = len(timeline)
    used_slots = sum(len(step["request_ids"]) for step in timeline)
    average_fill = used_slots / (total_decode_batches * max_batch) if total_decode_batches else 0.0
    verification_passed = outputs_by_request == expected_by_request and len(completed) == len(requests)

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": "passed" if verification_passed else "failed",
        "request_count": len(requests),
        "max_batch_size": max_batch,
        "total_decode_batches": total_decode_batches,
        "serial_decode_batches": serial_batches,
        "average_batch_fill": round(average_fill, 6),
        "timeline": timeline,
        "outputs_by_request": outputs_by_request,
        "expected_by_request": expected_by_request,
        "completed_request_ids": completed,
        "live_server_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "operator_next_steps": list(OPERATOR_NEXT_STEPS),
    }


LIVE_LOOP_ADAPTER_CLAIM_BOUNDARY = "continuous_batching_live_loop_adapter_no_server_or_speedup_proof"
LIVE_LOOP_ADAPTER_NEXT_STEPS = [
    "wire tick_batches into src/bloombee/client/inference_session.py behind opt-in flag",
    "run same-prompt parity against verifier-only decode with concurrent arrivals",
    "measure wall-clock throughput before any demo or speedup promotion",
]


def build_live_loop_adapter_plan(
    *,
    requests: Sequence[DecodeRequest],
    max_batch_size: int,
    opt_in_enabled: bool,
    pad_token_id: int = 0,
    opt_in_flag: str = "BLOOMBEE_ENABLE_CONTINUOUS_BATCHING",
) -> dict[str, object]:
    """Build an opt-in adapter plan for wiring scheduler ticks into a live loop.

    This is still a pure report: it converts scheduler timeline rows into padded
    per-tick batch payloads that the BloomBee client decode loop can consume in a
    later integration slice. It deliberately does not call the live server path.
    """
    scheduler = simulate_continuous_decode(requests=requests, max_batch_size=max_batch_size)
    tick_batches: list[dict[str, object]] = []
    if opt_in_enabled:
        for step in scheduler["timeline"]:
            input_sequences = [[token] for token in step["input_token_ids"]]
            tick_batches.append(
                {
                    "tick": step["tick"],
                    "request_ids": step["request_ids"],
                    "positions": step["positions"],
                    "input_batch": build_padded_batch(input_sequences, pad_token_id=pad_token_id),
                    "expected_output_token_ids": step["output_token_ids"],
                    "finished_request_ids": step["finished_request_ids"],
                }
            )

    return {
        "source": SOURCE,
        "claim_boundary": LIVE_LOOP_ADAPTER_CLAIM_BOUNDARY,
        "scheduler_claim_boundary": scheduler["claim_boundary"],
        "scheduler_verification_status": scheduler["verification_status"],
        "opt_in_flag": opt_in_flag,
        "opt_in_enabled": bool(opt_in_enabled),
        "adapter_status": "ready_for_live_loop_wiring" if opt_in_enabled else "disabled",
        "request_count": scheduler["request_count"],
        "max_batch_size": scheduler["max_batch_size"],
        "tick_batches": tick_batches,
        "outputs_by_request": scheduler["outputs_by_request"],
        "live_server_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "operator_next_steps": list(LIVE_LOOP_ADAPTER_NEXT_STEPS),
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["staggered"], default="staggered")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    payload = simulate_continuous_decode(
        requests=_example_requests(args.example),
        max_batch_size=args.max_batch_size,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
