from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-scheduler-20260704.json"
LIVE_ADAPTER_EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-live-adapter-20260705.json"


def test_continuous_scheduler_batches_late_arrivals_and_deinterleaves_tokens():
    from mvp_capabilities.continuous_batching import DecodeRequest, simulate_continuous_decode

    report = simulate_continuous_decode(
        requests=[
            DecodeRequest(request_id="req-a", prompt_token_ids=(101,), target_token_ids=(10, 11, 12), arrival_tick=0),
            DecodeRequest(request_id="req-b", prompt_token_ids=(201, 202), target_token_ids=(20, 21), arrival_tick=1),
        ],
        max_batch_size=2,
    )

    assert report["claim_boundary"] == "continuous_batching_scheduler_simulation_no_live_server_proof"
    assert report["total_decode_batches"] == 3
    assert report["serial_decode_batches"] == 5
    assert report["max_batch_size"] == 2
    assert report["average_batch_fill"] == 0.833333
    assert report["outputs_by_request"] == {"req-a": [10, 11, 12], "req-b": [20, 21]}
    assert report["timeline"] == [
        {
            "tick": 0,
            "request_ids": ["req-a"],
            "positions": [0],
            "input_token_ids": [101],
            "output_token_ids": [10],
            "finished_request_ids": [],
        },
        {
            "tick": 1,
            "request_ids": ["req-a", "req-b"],
            "positions": [1, 0],
            "input_token_ids": [10, 202],
            "output_token_ids": [11, 20],
            "finished_request_ids": [],
        },
        {
            "tick": 2,
            "request_ids": ["req-a", "req-b"],
            "positions": [2, 1],
            "input_token_ids": [11, 20],
            "output_token_ids": [12, 21],
            "finished_request_ids": ["req-a", "req-b"],
        },
    ]
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_continuous_scheduler_round_robin_prevents_one_request_from_monopolizing_batch():
    from mvp_capabilities.continuous_batching import DecodeRequest, simulate_continuous_decode

    report = simulate_continuous_decode(
        requests=[
            DecodeRequest(request_id="long", prompt_token_ids=(1,), target_token_ids=(10, 11, 12), arrival_tick=0),
            DecodeRequest(request_id="short", prompt_token_ids=(2,), target_token_ids=(20, 21), arrival_tick=0),
        ],
        max_batch_size=1,
    )

    assert [step["request_ids"] for step in report["timeline"]] == [["long"], ["short"], ["long"], ["short"], ["long"]]
    assert report["outputs_by_request"] == {"long": [10, 11, 12], "short": [20, 21]}
    assert report["completed_request_ids"] == ["short", "long"]


def test_continuous_scheduler_builds_padded_batch_inputs_with_attention_mask():
    from mvp_capabilities.continuous_batching import build_padded_batch

    batch = build_padded_batch([[101, 10], [201, 202, 20], [7]], pad_token_id=-1)

    assert batch == {
        "input_ids": [[101, 10, -1], [201, 202, 20], [7, -1, -1]],
        "attention_mask": [[1, 1, 0], [1, 1, 1], [1, 0, 0]],
        "sequence_lengths": [2, 3, 1],
    }


def test_continuous_live_loop_adapter_requires_opt_in_and_builds_tick_batches():
    from mvp_capabilities.continuous_batching import (
        DecodeRequest,
        build_live_loop_adapter_plan,
    )

    requests = [
        DecodeRequest(request_id="req-a", prompt_token_ids=(101,), target_token_ids=(10, 11), arrival_tick=0),
        DecodeRequest(request_id="req-b", prompt_token_ids=(201, 202), target_token_ids=(20,), arrival_tick=1),
    ]

    disabled = build_live_loop_adapter_plan(requests=requests, max_batch_size=2, opt_in_enabled=False)
    assert disabled["claim_boundary"] == "continuous_batching_live_loop_adapter_no_server_or_speedup_proof"
    assert disabled["adapter_status"] == "disabled"
    assert disabled["opt_in_flag"] == "BLOOMBEE_ENABLE_CONTINUOUS_BATCHING"
    assert disabled["tick_batches"] == []
    assert disabled["live_server_proven"] is False
    assert disabled["speedup_proven"] is False

    enabled = build_live_loop_adapter_plan(requests=requests, max_batch_size=2, opt_in_enabled=True, pad_token_id=-1)
    assert enabled["adapter_status"] == "ready_for_live_loop_wiring"
    assert enabled["tick_batches"] == [
        {
            "tick": 0,
            "request_ids": ["req-a"],
            "positions": [0],
            "input_batch": {"input_ids": [[101]], "attention_mask": [[1]], "sequence_lengths": [1]},
            "expected_output_token_ids": [10],
            "finished_request_ids": [],
        },
        {
            "tick": 1,
            "request_ids": ["req-a", "req-b"],
            "positions": [1, 0],
            "input_batch": {"input_ids": [[10], [202]], "attention_mask": [[1], [1]], "sequence_lengths": [1, 1]},
            "expected_output_token_ids": [11, 20],
            "finished_request_ids": ["req-a", "req-b"],
        },
    ]
    assert enabled["outputs_by_request"] == {"req-a": [10, 11], "req-b": [20]}
    assert enabled["live_server_proven"] is False
    assert enabled["speedup_proven"] is False
    assert enabled["can_update_demo_status"] is False
    assert any(
        "wire tick_batches into src/bloombee/client/inference_session.py" in step
        for step in enabled["operator_next_steps"]
    )



def test_continuous_batching_cli_and_tracked_evidence_are_claim_bounded():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/continuous_batching.py", "--example", "staggered"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    tracked = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["timeline"] == tracked["timeline"]
    assert tracked["verification_status"] == "passed"
    assert tracked["claim_boundary"] == "continuous_batching_scheduler_simulation_no_live_server_proof"
    assert tracked["request_count"] == 2
    assert tracked["total_decode_batches"] == 3
    assert tracked["serial_decode_batches"] == 5
    assert tracked["outputs_by_request"] == {"req-a": [10, 11, 12], "req-b": [20, 21]}
    assert tracked["live_server_proven"] is False
    assert tracked["speedup_proven"] is False
    assert tracked["operator_next_steps"] == [
        "wire the scheduler into the live decode request loop behind an opt-in flag",
        "run same-prompt parity against verifier-only decode with concurrent arrivals",
        "measure wall-clock throughput before any demo or speedup promotion",
    ]


def test_continuous_batching_live_adapter_tracked_evidence_is_claim_bounded():
    tracked = json.loads(LIVE_ADAPTER_EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert tracked["claim_boundary"] == "continuous_batching_live_loop_adapter_no_server_or_speedup_proof"
    assert tracked["scheduler_claim_boundary"] == "continuous_batching_scheduler_simulation_no_live_server_proof"
    assert tracked["adapter_status"] == "ready_for_live_loop_wiring"
    assert tracked["opt_in_flag"] == "BLOOMBEE_ENABLE_CONTINUOUS_BATCHING"
    assert tracked["opt_in_enabled"] is True
    assert tracked["tick_batches"]
    assert tracked["live_server_proven"] is False
    assert tracked["speedup_proven"] is False
    assert tracked["can_update_demo_status"] is False
    assert tracked["operator_next_steps"][0] == "wire tick_batches into src/bloombee/client/inference_session.py behind opt-in flag"
