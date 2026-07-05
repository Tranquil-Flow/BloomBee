"""Claim-bounded live continuous batching loop primitives.

This module proves only a local/injected decode-loop seam. It does not touch the
remote server scheduler, does not prove wall-clock speedup, and must not promote
demo status by itself.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

ENV_ENABLE_LIVE_CONTINUOUS_BATCHING = "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
ENV_LIVE_CONTINUOUS_MAX_BATCH_SIZE = "BLOOMBEE_LIVE_CONTINUOUS_MAX_BATCH_SIZE"
CLAIM_BOUNDARY = "live_continuous_decode_loop_unit_no_server_no_speedup"


def is_live_continuous_batching_enabled() -> bool:
    """Return the opt-in flag at call time, not import time."""
    return os.environ.get(ENV_ENABLE_LIVE_CONTINUOUS_BATCHING, "0") == "1"


@dataclass(frozen=True)
class LiveDecodeRequest:
    request_id: str
    input_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    arrival_tick: int = 0

    def __init__(
        self,
        *,
        request_id: str,
        input_token_ids: Sequence[int],
        target_token_ids: Sequence[int],
        arrival_tick: int = 0,
    ) -> None:
        object.__setattr__(self, "request_id", str(request_id))
        object.__setattr__(self, "input_token_ids", tuple(int(token) for token in input_token_ids))
        object.__setattr__(self, "target_token_ids", tuple(int(token) for token in target_token_ids))
        object.__setattr__(self, "arrival_tick", int(arrival_tick))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.input_token_ids:
            raise ValueError("input_token_ids must be non-empty")
        if self.arrival_tick < 0:
            raise ValueError("arrival_tick must be non-negative")


@dataclass(frozen=True)
class LiveDecodeRow:
    request_id: str
    tick: int
    position: int
    input_token_id: int


class LiveContinuousDecodeLoop:
    """Pure opt-in live-loop adapter driven by an injected batch step."""

    def __init__(self, *, max_batch_size: int) -> None:
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

    def run(
        self,
        requests: Sequence[LiveDecodeRequest],
        *,
        step_batch: Callable[[Sequence[LiveDecodeRow]], Mapping[str, int] | Sequence[int]],
    ) -> dict[str, object]:
        self._validate_requests(requests)
        ordered = sorted(enumerate(requests), key=lambda item: (item[1].arrival_tick, item[0]))
        pending = deque(request for _, request in ordered)
        state: dict[str, dict[str, object]] = {
            request.request_id: {"request": request, "position": 0, "generated": []}
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
                if not request.target_token_ids:
                    completed.append(request.request_id)
                    continue
                active.append(request.request_id)
                active_set.add(request.request_id)

            if not active:
                if pending:
                    tick = max(tick + 1, pending[0].arrival_tick)
                    continue
                break

            batch_ids: list[str] = []
            for _ in range(self.max_batch_size):
                if not active:
                    break
                request_id = active.popleft()
                active_set.remove(request_id)
                batch_ids.append(request_id)

            rows: list[LiveDecodeRow] = []
            for request_id in batch_ids:
                request_state = state[request_id]
                request = request_state["request"]
                assert isinstance(request, LiveDecodeRequest)
                position = int(request_state["position"])
                generated = request_state["generated"]
                assert isinstance(generated, list)
                input_token_id = int(request.input_token_ids[-1] if position == 0 else generated[-1])
                rows.append(
                    LiveDecodeRow(
                        request_id=request_id,
                        tick=tick,
                        position=position,
                        input_token_id=input_token_id,
                    )
                )

            outputs = step_batch(rows)
            output_by_request = self._normalize_outputs(rows, outputs)
            finished: list[str] = []
            requeue: list[str] = []
            output_token_ids: list[int] = []

            for row in rows:
                request_state = state[row.request_id]
                request = request_state["request"]
                assert isinstance(request, LiveDecodeRequest)
                generated = request_state["generated"]
                assert isinstance(generated, list)
                next_token = int(output_by_request[row.request_id])
                output_token_ids.append(next_token)
                generated.append(next_token)
                request_state["position"] = row.position + 1
                if row.position + 1 >= len(request.target_token_ids):
                    finished.append(row.request_id)
                    completed.append(row.request_id)
                else:
                    requeue.append(row.request_id)

            for request_id in requeue:
                if request_id not in active_set:
                    active.append(request_id)
                    active_set.add(request_id)

            timeline.append(
                {
                    "tick": tick,
                    "request_ids": [row.request_id for row in rows],
                    "positions": [row.position for row in rows],
                    "input_token_ids": [row.input_token_id for row in rows],
                    "output_token_ids": output_token_ids,
                    "finished_request_ids": finished,
                }
            )
            tick += 1

        outputs_by_request = {
            request.request_id: list(state[request.request_id]["generated"])
            for request in requests
        }
        expected_by_request = {request.request_id: list(request.target_token_ids) for request in requests}
        verification_passed = outputs_by_request == expected_by_request and len(completed) == len(requests)
        return {
            "source": "bloombee.client.live_continuous_batching",
            "claim_boundary": CLAIM_BOUNDARY,
            "verification_status": "passed" if verification_passed else "failed",
            "request_count": len(requests),
            "max_batch_size": self.max_batch_size,
            "timeline": timeline,
            "outputs_by_request": outputs_by_request,
            "expected_by_request": expected_by_request,
            "completed_request_ids": completed,
            "live_loop_unit_proven": verification_passed,
            "live_server_proven": False,
            "speedup_proven": False,
            "can_update_demo_status": False,
        }

    @staticmethod
    def _validate_requests(requests: Sequence[LiveDecodeRequest]) -> None:
        seen: set[str] = set()
        for request in requests:
            if request.request_id in seen:
                raise ValueError(f"duplicate request_id: {request.request_id}")
            seen.add(request.request_id)

    @staticmethod
    def _normalize_outputs(
        rows: Sequence[LiveDecodeRow],
        outputs: Mapping[str, int] | Sequence[int],
    ) -> dict[str, int]:
        if isinstance(outputs, Mapping):
            return {row.request_id: int(outputs[row.request_id]) for row in rows}
        if len(outputs) != len(rows):
            raise ValueError("step_batch output length must match row count")
        return {row.request_id: int(token) for row, token in zip(rows, outputs)}
