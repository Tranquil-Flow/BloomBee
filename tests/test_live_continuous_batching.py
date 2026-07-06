from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bloombee.client.remote_generation import RemoteGenerationMixin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_LOOP_EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/live-continuous-batching-loop-unit-20260705.json"
KV_PREFIX_LIVE_GENERATE_EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/kv-prefix-reuse-live-generate-metadata-20260706.json"


class _FallbackGenerate:
    def generate(self, inputs=None, *args, **kwargs):
        self.fallback_calls.append((inputs, args, dict(kwargs)))
        return torch.tensor([[999]], dtype=torch.long)


class _RemoteGenerationStub(RemoteGenerationMixin, _FallbackGenerate):
    def __init__(self):
        self.live_calls = []
        self.fallback_calls = []
        self._active_session = None
        self._supports_cache_class = False
        self.transformer = SimpleNamespace(config=SimpleNamespace(pre_seq_len=0, tuning_mode=None))

    @property
    def active_session(self):
        return self._active_session

    def _fix_generate_kwargs(self, kwargs):
        return None

    def use_session(self, session):
        return contextlib.nullcontext(session)

    def inference_session(self, **kwargs):
        session = SimpleNamespace(position=0, output_ids=None)
        return contextlib.nullcontext(session)

    def _live_continuous_generate_impl(self, inputs, *args, session=None, **kwargs):
        self.live_calls.append((inputs, args, session, dict(kwargs)))
        return torch.tensor([[101, 10]], dtype=torch.long)


class _ScalarTokenEmbedding(torch.nn.Module):
    def forward(self, input_ids):
        return input_ids.to(dtype=torch.float32).unsqueeze(-1)


class _GreedyNextTokenHead(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = dict(mapping)

    def forward(self, hidden_states):
        logits = torch.full((*hidden_states.shape[:2], 256), -1000.0, dtype=torch.float32)
        for batch_idx in range(hidden_states.shape[0]):
            token_id = int(hidden_states[batch_idx, -1, 0].item())
            logits[batch_idx, -1, self.mapping[token_id]] = 1000.0
        return logits


class _FakeLiveRemoteLayers:
    def __init__(self, *, pad_to_batch_size: int | None = None):
        self._active_session = None
        self.hidden_calls = []
        self.hidden_kwargs = []
        self.sessions = []
        self.pending_live_batches_seen = []
        self.pad_to_batch_size = pad_to_batch_size

    @property
    def active_session(self):
        return self._active_session

    @contextlib.contextmanager
    def use_session(self, session):
        previous = self._active_session
        self._active_session = session
        try:
            yield session
        finally:
            self._active_session = previous

    @contextlib.contextmanager
    def inference_session(self, **kwargs):
        from bloombee.client.inference_session import InferenceSession

        session = InferenceSession(SimpleNamespace(), **kwargs)
        self.sessions.append(session)
        with session, self.use_session(session):
            yield session

    def __call__(self, hidden_states, **kwargs):
        assert self._active_session is not None
        pending = getattr(self._active_session, "_pending_live_continuous_tick_batch", None)
        self.pending_live_batches_seen.append(dict(pending) if isinstance(pending, dict) else None)
        self.hidden_kwargs.append(dict(kwargs))
        self.hidden_calls.append(hidden_states.detach().clone())
        if self.pad_to_batch_size is not None and hidden_states.shape[0] < self.pad_to_batch_size:
            pad_rows = self.pad_to_batch_size - hidden_states.shape[0]
            hidden_states = torch.cat([hidden_states, hidden_states[-1:].expand(pad_rows, *hidden_states.shape[1:])], dim=0)
        return hidden_states


class _RemoteGenerationLiveImplStub(RemoteGenerationMixin, _FallbackGenerate):
    def __init__(self):
        self.fallback_calls = []
        self._supports_cache_class = False
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(pre_seq_len=0, tuning_mode=None),
            word_embeddings=_ScalarTokenEmbedding(),
            h=_FakeLiveRemoteLayers(),
            ln_f=torch.nn.Identity(),
        )
        self.lm_head = _GreedyNextTokenHead({101: 10, 10: 11})


def test_live_continuous_decode_loop_batches_late_arrivals_and_deinterleaves_outputs():
    from bloombee.client.live_continuous_batching import LiveContinuousDecodeLoop, LiveDecodeRequest

    requests = [
        LiveDecodeRequest(request_id="req-a", input_token_ids=(101,), target_token_ids=(10, 11, 12), arrival_tick=0),
        LiveDecodeRequest(request_id="req-b", input_token_ids=(201, 202), target_token_ids=(20, 21), arrival_tick=1),
    ]
    calls = []

    def step_batch(rows):
        calls.append(
            [
                {
                    "request_id": row.request_id,
                    "position": row.position,
                    "input_token_id": row.input_token_id,
                    "tick": row.tick,
                }
                for row in rows
            ]
        )
        targets = {request.request_id: request.target_token_ids for request in requests}
        return {row.request_id: targets[row.request_id][row.position] for row in rows}

    report = LiveContinuousDecodeLoop(max_batch_size=2).run(requests, step_batch=step_batch)

    assert calls == [
        [{"request_id": "req-a", "position": 0, "input_token_id": 101, "tick": 0}],
        [
            {"request_id": "req-a", "position": 1, "input_token_id": 10, "tick": 1},
            {"request_id": "req-b", "position": 0, "input_token_id": 202, "tick": 1},
        ],
        [
            {"request_id": "req-a", "position": 2, "input_token_id": 11, "tick": 2},
            {"request_id": "req-b", "position": 1, "input_token_id": 20, "tick": 2},
        ],
    ]
    assert report["claim_boundary"] == "live_continuous_decode_loop_unit_no_server_no_speedup"
    assert report["outputs_by_request"] == {"req-a": [10, 11, 12], "req-b": [20, 21]}
    assert report["live_loop_unit_proven"] is True
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_live_continuous_batching_env_helper_reads_at_call_time(monkeypatch):
    from bloombee.client.live_continuous_batching import is_live_continuous_batching_enabled

    monkeypatch.delenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", raising=False)
    assert is_live_continuous_batching_enabled() is False

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    assert is_live_continuous_batching_enabled() is True

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "0")
    assert is_live_continuous_batching_enabled() is False


def test_remote_generation_delegates_to_live_continuous_scheduler_only_when_opted_in(monkeypatch):
    inputs = torch.tensor([[101]], dtype=torch.long)
    model = _RemoteGenerationStub()

    monkeypatch.delenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", raising=False)
    fallback = model.generate(inputs, max_new_tokens=1)
    assert fallback.tolist() == [[999]]
    assert model.live_calls == []
    assert len(model.fallback_calls) == 1

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    live = model.generate(inputs, max_new_tokens=1)
    assert live.tolist() == [[101, 10]]
    assert len(model.live_calls) == 1
    assert len(model.fallback_calls) == 1


def test_remote_generation_base_live_impl_records_inference_session_tick_rows(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    model = _RemoteGenerationLiveImplStub()
    inputs = torch.tensor([[101]], dtype=torch.long)

    result = model.generate(inputs, max_new_tokens=2)

    assert result.tolist() == [[101, 10, 11]]
    assert model.fallback_calls == []
    assert [call.squeeze(-1).to(dtype=torch.long).tolist() for call in model.transformer.h.hidden_calls] == [
        [[101]],
        [[10]],
    ]
    assert len(model.transformer.h.sessions) == 1
    report = model.transformer.h.sessions[0].live_continuous_batching_report()
    assert report["claim_boundary"] == "live_continuous_inference_session_tick_rows_no_server_parity_or_speedup"
    assert report["opt_in_flag"] == "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
    assert report["request_count"] == 1
    tick_batches = report["tick_batches"]
    assert [len(batch["output_logits_sha256"][0]) for batch in tick_batches] == [64, 64]
    assert [batch["output_logits_summary"] for batch in tick_batches] == [
        [{"top1_token_id": 10, "top1_logit": 1000.0, "top2_logit": -1000.0, "top1_margin": 2000.0}],
        [{"top1_token_id": 11, "top1_logit": 1000.0, "top2_logit": -1000.0, "top1_margin": 2000.0}],
    ]
    assert "output_logits_values" not in tick_batches[0]
    assert [
        {k: v for k, v in batch.items() if k not in {"output_logits_sha256", "output_logits_summary"}}
        for batch in tick_batches
    ] == [
        {
            "tick": 0,
            "request_ids": ["generate-0"],
            "positions": [0],
            "input_token_ids": [101],
            "output_token_ids": [10],
        },
        {
            "tick": 1,
            "request_ids": ["generate-0"],
            "positions": [1],
            "input_token_ids": [10],
            "output_token_ids": [11],
        },
    ]
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_remote_generation_base_live_impl_records_full_logits_when_strict_capture_opted_in(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setenv("BLOOMBEE_LIVE_CONTINUOUS_CAPTURE_LOGITS", "1")
    model = _RemoteGenerationLiveImplStub()
    inputs = torch.tensor([[101]], dtype=torch.long)

    model.generate(inputs, max_new_tokens=1)

    report = model.transformer.h.sessions[0].live_continuous_batching_report()
    logits_values = report["tick_batches"][0]["output_logits_values"]
    assert len(logits_values) == 1
    assert len(logits_values[0]) == 256
    assert logits_values[0][10] == 1000.0
    assert logits_values[0][0] == -1000.0


def test_remote_generation_base_live_impl_batches_same_arrival_rows(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    model = _RemoteGenerationLiveImplStub()
    model.lm_head = _GreedyNextTokenHead({101: 10, 201: 20, 10: 11, 20: 21})
    inputs = torch.tensor([[101], [201]], dtype=torch.long)

    result = model.generate(inputs, max_new_tokens=2)

    assert result.tolist() == [[101, 10, 11], [201, 20, 21]]
    assert model.fallback_calls == []
    assert [call.squeeze(-1).to(dtype=torch.long).tolist() for call in model.transformer.h.hidden_calls] == [
        [[101], [201]],
        [[10], [20]],
    ]
    report = model.transformer.h.sessions[0].live_continuous_batching_report()
    assert [batch["request_ids"] if batch is not None else None for batch in model.transformer.h.pending_live_batches_seen] == [
        ["generate-0", "generate-1"],
        ["generate-0", "generate-1"],
    ]
    assert report["request_count"] == 2
    assert report["total_decode_batches"] == 2
    tick_batches = report["tick_batches"]
    for batch in tick_batches:
        assert [len(value) for value in batch["output_logits_sha256"]] == [64, 64]
    assert tick_batches[0]["output_logits_sha256"][0] != tick_batches[0]["output_logits_sha256"][1]
    assert [{k: v for k, v in batch.items() if k not in {"output_logits_sha256", "output_logits_summary"}} for batch in tick_batches] == [
        {
            "tick": 0,
            "request_ids": ["generate-0", "generate-1"],
            "positions": [0, 0],
            "input_token_ids": [101, 201],
            "output_token_ids": [10, 20],
        },
        {
            "tick": 1,
            "request_ids": ["generate-0", "generate-1"],
            "positions": [1, 1],
            "input_token_ids": [10, 20],
            "output_token_ids": [11, 21],
        },
    ]
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False


def test_remote_generation_base_live_impl_batches_late_arrival_rows(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    model = _RemoteGenerationLiveImplStub()
    model.lm_head = _GreedyNextTokenHead({101: 10, 201: 20, 10: 11, 20: 21, 11: 0})
    inputs = torch.tensor([[101], [201]], dtype=torch.long)

    result = model.generate(inputs, max_new_tokens=2, live_arrival_ticks=[0, 1])

    assert result.tolist() == [[101, 10, 11], [201, 20, 21]]
    assert model.fallback_calls == []
    # Late arrivals preserve full logical tick metadata, but compact inactive
    # rows out of the wire/compute batch. If active rows have different cache
    # positions, compute is split into position-homogeneous slices so one row's
    # past cannot perturb another row's logits.
    assert [call.squeeze(-1).to(dtype=torch.long).tolist() for call in model.transformer.h.hidden_calls] == [
        [[101]],
        [[10]],
        [[201]],
        [[20]],
    ]
    assert len(model.transformer.h.sessions) == 1
    assert [
        {
            "request_ids": batch["request_ids"],
            "active_mask": batch.get("active_mask"),
            "batch_offset": batch.get("batch_offset"),
            "full_batch_size": batch.get("full_batch_size"),
            "micro_batch_size": batch.get("micro_batch_size"),
        }
        for batch in model.transformer.h.pending_live_batches_seen
    ] == [
        {"request_ids": ["generate-0", "generate-1"], "active_mask": [True, False], "batch_offset": 0, "full_batch_size": 2, "micro_batch_size": 1},
        {"request_ids": ["generate-0", "generate-1"], "active_mask": [True, False], "batch_offset": 0, "full_batch_size": 2, "micro_batch_size": 1},
        {"request_ids": ["generate-0", "generate-1"], "active_mask": [False, True], "batch_offset": 1, "full_batch_size": 2, "micro_batch_size": 1},
        {"request_ids": ["generate-0", "generate-1"], "active_mask": [False, True], "batch_offset": 1, "full_batch_size": 2, "micro_batch_size": 1},
    ]
    assert [kwargs["prefill_length"].detach().cpu().tolist() for kwargs in model.transformer.h.hidden_kwargs] == [
        [0],
        [1],
        [0],
        [1],
    ]
    report = model.transformer.h.sessions[0].live_continuous_batching_report()
    assert getattr(model.transformer.h.sessions[0], "_live_continuous_full_batch_size") == 2
    assert report["request_count"] == 2
    assert report["total_decode_batches"] == 3
    tick_batches = report["tick_batches"]
    assert [len(batch["output_logits_sha256"]) for batch in tick_batches] == [2, 2, 2]
    assert tick_batches[1]["output_logits_sha256"][0] != tick_batches[1]["output_logits_sha256"][1]
    assert [{k: v for k, v in batch.items() if k not in {"output_logits_sha256", "output_logits_summary"}} for batch in tick_batches] == [
        {
            "tick": 0,
            "request_ids": ["generate-0", "generate-1"],
            "active_mask": [True, False],
            "positions": [0, 0],
            "input_token_ids": [101, 201],
            "output_token_ids": [10, 201],
        },
        {
            "tick": 1,
            "request_ids": ["generate-0", "generate-1"],
            "active_mask": [True, True],
            "positions": [1, 0],
            "input_token_ids": [10, 201],
            "output_token_ids": [11, 20],
        },
        {
            "tick": 2,
            "request_ids": ["generate-0", "generate-1"],
            "active_mask": [False, True],
            "positions": [2, 1],
            "input_token_ids": [11, 20],
            "output_token_ids": [11, 21],
        },
    ]
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False


def test_remote_generation_late_arrival_trims_full_batch_padded_hidden_states(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setenv("BLOOMBEE_LIVE_CONTINUOUS_MAX_BATCH_SIZE", "2")
    model = _RemoteGenerationLiveImplStub()
    model.transformer.h = _FakeLiveRemoteLayers(pad_to_batch_size=2)
    model.lm_head = _GreedyNextTokenHead({101: 10, 10: 11, 201: 20, 20: 21, 11: 0})
    inputs = torch.tensor([[101], [201]], dtype=torch.long)

    result = model.generate(inputs, max_new_tokens=2, live_arrival_ticks=[0, 1])

    assert result.tolist() == [[101, 10, 11], [201, 20, 21]]
    report = model.transformer.h.sessions[0].live_continuous_batching_report()
    assert [len(batch["request_ids"]) for batch in report["tick_batches"]] == [2, 2, 2]
    assert [batch["active_mask"] for batch in report["tick_batches"]] == [[True, False], [True, True], [False, True]]
    assert [len(batch["output_token_ids"]) for batch in report["tick_batches"]] == [2, 2, 2]
    assert [len(batch["output_logits_sha256"]) for batch in report["tick_batches"]] == [2, 2, 2]


def test_remote_generation_base_live_impl_allows_explicit_session_for_evidence_reports(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    model = _RemoteGenerationLiveImplStub()
    model.lm_head = _GreedyNextTokenHead({101: 10, 10: 11})
    inputs = torch.tensor([[101]], dtype=torch.long)
    with model.transformer.h.inference_session(max_length=4) as session:
        result = model.generate(inputs, max_new_tokens=2, session=session)
        report = session.live_continuous_batching_report()

    assert result.tolist() == [[101, 10, 11]]
    assert report["total_decode_batches"] == 2
    assert report["tick_batches"][0]["output_token_ids"] == [10]
    assert len(report["tick_batches"][0]["output_logits_sha256"][0]) == 64
    assert model.fallback_calls == []


def test_live_generate_records_kv_prefix_metadata_for_same_prefix_batch(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "1")
    model = _RemoteGenerationLiveImplStub()
    model.lm_head = _GreedyNextTokenHead({201: 10, 202: 20})
    inputs = torch.tensor([[101, 102, 201], [101, 102, 202]], dtype=torch.long)

    result = model.generate(inputs, max_new_tokens=1)

    assert result.tolist() == [[101, 102, 201, 10], [101, 102, 202, 20]]
    report = model.transformer.h.sessions[0].kv_prefix_reuse_report()
    assert report["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert report["opt_in_flag"] == "BLOOMBEE_ENABLE_KV_PREFIX_REUSE"
    assert report["event_count"] == 1
    pending_batch = model.transformer.h.pending_live_batches_seen[0]
    assert pending_batch["kv_prefix_reuse"]["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert pending_batch["kv_prefix_reuse"]["event_count"] == 1
    assert report["runtime_prefill_metadata_proven"] is True
    assert report["live_kv_cache_reuse_proven"] is False
    assert report["speedup_proven"] is False
    event = report["events"][0]
    assert event["common_prefix_token_ids"] == [101, 102]
    assert event["same_prefix_varied_suffix_proven"] is True
    assert event["requests"] == [
        {
            "request_id": "generate-0",
            "prefix_token_ids": [101, 102],
            "suffix_token_ids": [201],
            "prefill_token_count": 3,
            "reusable_prefix_token_count": 2,
        },
        {
            "request_id": "generate-1",
            "prefix_token_ids": [101, 102],
            "suffix_token_ids": [202],
            "prefill_token_count": 3,
            "reusable_prefix_token_count": 2,
        },
    ]


def test_live_continuous_tick_rows_are_sent_in_rpc_inference_metadata(monkeypatch):
    import asyncio

    from hivemind.proto import runtime_pb2
    from hivemind.utils.tensor_descr import BatchTensorDescriptor

    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import _ServerInferenceSession
    from bloombee.client.live_continuous_batching import LiveDecodeRow
    from bloombee.utils.hivemind_compat import MSGPackSerializer
    from bloombee.utils.lossless_transport import serialize_torch_tensor
    from bloombee.utils.misc import DUMMY, DUMMY_INT64

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setattr(
        inference_session_mod.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )

    inputs = torch.zeros((2, 1, 4), dtype=torch.float32)
    schema = (BatchTensorDescriptor.from_tensor(inputs, runtime_pb2.CompressionType.NONE),)
    session = _ServerInferenceSession(
        SimpleNamespace(use_server_to_server=False, request_timeout=1.0),
        SimpleNamespace(start=0, end=1, peer_id="peer-1"),
        "block.0",
        {"inference_schema": (schema, {})},
        asyncio.Queue(),
        None,
        max_length=8,
    )
    captured_requests = []

    async def fake_step(request):
        captured_requests.append(request)
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(inputs, runtime_pb2.CompressionType.NONE)]
        )

    session._step = fake_step
    rows = [
        LiveDecodeRow(request_id="generate-0", tick=3, position=0, input_token_id=101),
        LiveDecodeRow(request_id="generate-1", tick=3, position=0, input_token_id=201),
    ]

    staged = session.stage_live_continuous_tick_rows(rows)
    outputs = session.step(
        inputs,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.zeros(inputs.shape[0]),
        step_id="step-live-1",
    )

    assert torch.equal(outputs[0], inputs)
    assert staged["request_ids"] == ["generate-0", "generate-1"]
    metadata = MSGPackSerializer.loads(captured_requests[0].metadata)
    live_metadata = metadata["live_continuous_batching"]
    assert live_metadata["claim_boundary"] == "live_continuous_inference_session_tick_rows_no_server_parity_or_speedup"
    assert live_metadata["opt_in_flag"] == "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
    assert live_metadata["request_count"] == 2
    assert live_metadata["tick_batches"] == [
        {
            "tick": 3,
            "request_ids": ["generate-0", "generate-1"],
            "positions": [0, 0],
            "input_token_ids": [101, 201],
        }
    ]
    assert live_metadata["live_server_proven"] is False
    assert live_metadata["speedup_proven"] is False
    assert live_metadata["can_update_demo_status"] is False

    session.step(
        inputs,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.zeros(inputs.shape[0]),
        step_id="step-live-2",
    )
    stale_metadata = MSGPackSerializer.loads(captured_requests[1].metadata)
    assert "live_continuous_batching" not in stale_metadata


def test_server_session_preserves_live_metadata_after_failed_rpc_attempt(monkeypatch):
    import asyncio

    from hivemind.proto import runtime_pb2
    from hivemind.utils.tensor_descr import BatchTensorDescriptor

    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import _ServerInferenceSession
    from bloombee.utils.hivemind_compat import MSGPackSerializer
    from bloombee.utils.lossless_transport import serialize_torch_tensor
    from bloombee.utils.misc import DUMMY, DUMMY_INT64

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setattr(
        inference_session_mod.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )

    inputs = torch.zeros((1, 1, 4), dtype=torch.float32)
    schema = (BatchTensorDescriptor.from_tensor(inputs, runtime_pb2.CompressionType.NONE),)
    session = _ServerInferenceSession(
        SimpleNamespace(use_server_to_server=False, request_timeout=1.0),
        SimpleNamespace(start=0, end=1, peer_id="peer-1"),
        "block.0",
        {"inference_schema": (schema, {})},
        asyncio.Queue(),
        None,
        max_length=3,
        live_continuous_full_batch_size=2,
    )
    batch = {
        "tick": 1,
        "request_ids": ["generate-0", "generate-1"],
        "positions": [1, 0],
        "input_token_ids": [29889, 350],
        "active_mask": [True, False],
        "batch_offset": 0,
        "full_batch_size": 2,
        "micro_batch_size": 1,
    }
    session.stage_live_continuous_tick_batch(batch)
    captured_metadata = []
    calls = 0

    async def flaky_step(request):
        nonlocal calls
        calls += 1
        captured_metadata.append(MSGPackSerializer.loads(request.metadata))
        if calls == 1:
            raise RuntimeError("transient mps placeholder")
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(inputs, runtime_pb2.CompressionType.NONE)]
        )

    session._step = flaky_step

    with pytest.raises(RuntimeError, match="transient mps placeholder"):
        session.step(
            inputs,
            DUMMY,
            DUMMY_INT64,
            prefill_length=torch.tensor([1], dtype=torch.long),
            step_id="failed",
        )

    assert session._pending_live_continuous_tick_batch == batch

    session.step(
        inputs,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.tensor([1], dtype=torch.long),
        step_id="retry",
    )

    assert captured_metadata[0]["live_continuous_batching"]["tick_batches"][0]["active_mask"] == [True, False]
    assert captured_metadata[1]["live_continuous_batching"]["tick_batches"][0]["active_mask"] == [True, False]
    assert session._pending_live_continuous_tick_batch is None


def test_live_compact_replacement_session_sends_row_local_history(monkeypatch):
    import asyncio

    from hivemind.proto import runtime_pb2
    from hivemind.utils.tensor_descr import BatchTensorDescriptor

    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import _ServerInferenceSession
    from bloombee.utils.hivemind_compat import MSGPackSerializer
    from bloombee.utils.lossless_transport import deserialize_torch_tensor, serialize_torch_tensor
    from bloombee.utils.misc import DUMMY, DUMMY_INT64

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setattr(
        inference_session_mod.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )

    current = torch.full((1, 1, 4), 2.0, dtype=torch.float32)
    schema = (BatchTensorDescriptor.from_tensor(current, runtime_pb2.CompressionType.NONE),)
    session = _ServerInferenceSession(
        SimpleNamespace(use_server_to_server=False, request_timeout=1.0),
        SimpleNamespace(start=0, end=1, peer_id="peer-1"),
        "block.0",
        {"inference_schema": (schema, {})},
        asyncio.Queue(),
        None,
        max_length=3,
        live_continuous_full_batch_size=2,
    )
    session.history = torch.zeros((2, 2, 4), dtype=torch.float32)
    session.history[0, 0, :] = 1.0
    session.stage_live_continuous_tick_batch(
        {
            "tick": 1,
            "request_ids": ["generate-0", "generate-1"],
            "positions": [1, 0],
            "input_token_ids": [29889, 350],
            "active_mask": [True, False],
            "batch_offset": 0,
            "full_batch_size": 2,
            "micro_batch_size": 1,
        }
    )
    captured_inputs = []
    captured_metadata = []

    async def fake_step(request):
        captured_inputs.append(deserialize_torch_tensor(request.tensors[0]))
        captured_metadata.append(MSGPackSerializer.loads(request.metadata))
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(captured_inputs[-1], runtime_pb2.CompressionType.NONE)]
        )

    session._step = fake_step

    session.step(
        current,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.tensor([1], dtype=torch.long),
        step_id="retry-history",
    )

    assert captured_metadata[0]["start_from_position"] == 0
    assert captured_inputs[0].shape == (1, 2, 4)
    assert torch.equal(captured_inputs[0][0, 0, :], torch.ones(4))
    assert torch.equal(captured_inputs[0][0, 1, :], torch.full((4,), 2.0))


def test_live_compact_microbatch_slices_use_row_local_position_for_retries(monkeypatch):
    import asyncio

    from hivemind.proto import runtime_pb2
    from hivemind.utils.tensor_descr import BatchTensorDescriptor

    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import _ServerInferenceSession
    from bloombee.utils.hivemind_compat import MSGPackSerializer
    from bloombee.utils.lossless_transport import serialize_torch_tensor
    from bloombee.utils.misc import DUMMY, DUMMY_INT64

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    monkeypatch.setattr(
        inference_session_mod.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )

    inputs = torch.zeros((1, 1, 4), dtype=torch.float32)
    schema = (BatchTensorDescriptor.from_tensor(inputs, runtime_pb2.CompressionType.NONE),)
    session = _ServerInferenceSession(
        SimpleNamespace(use_server_to_server=False, request_timeout=1.0),
        SimpleNamespace(start=0, end=1, peer_id="peer-1"),
        "block.0",
        {"inference_schema": (schema, {})},
        asyncio.Queue(),
        None,
        max_length=3,
        live_continuous_full_batch_size=2,
    )
    captured_metadata = []

    async def fake_step(request):
        captured_metadata.append(MSGPackSerializer.loads(request.metadata))
        session.stepped = True
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(inputs, runtime_pb2.CompressionType.NONE)]
        )

    session._step = fake_step
    slices = [
        (0, 0, [True, False], "s0"),
        (0, 1, [True, False], "s1"),
        (1, 0, [False, True], "s2"),
        (1, 1, [False, True], "s3"),
    ]

    for batch_offset, row_position, active_mask, step_id in slices:
        session.stage_live_continuous_tick_batch(
            {
                "tick": int(row_position),
                "request_ids": ["generate-0", "generate-1"],
                "positions": [row_position, row_position],
                "input_token_ids": [101, 201],
                "active_mask": active_mask,
                "batch_offset": batch_offset,
                "full_batch_size": 2,
                "micro_batch_size": 1,
            }
        )
        session.step(
            inputs,
            DUMMY,
            DUMMY_INT64,
            prefill_length=torch.tensor([row_position], dtype=torch.long),
            step_id=step_id,
        )

    assert [metadata["start_from_position"] for metadata in captured_metadata] == [0, 1, 0, 1]
    assert session.position == 2
    assert session.history.shape == (2, 2, 4)


def test_server_observes_live_continuous_batching_metadata_only_with_opt_in(monkeypatch):
    from bloombee.server.handler import _extract_live_continuous_batching_metadata

    payload = {
        "claim_boundary": "live_continuous_inference_session_tick_rows_no_server_parity_or_speedup",
        "opt_in_flag": "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING",
        "opt_in_enabled": True,
        "request_count": 2,
        "tick_batches": [
            {
                "tick": 1,
                "request_ids": ["generate-0", "generate-1"],
                "positions": [0, 0],
                "input_token_ids": [101, 201],
                "active_mask": [True, False],
            }
        ],
        "live_server_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
    }

    monkeypatch.delenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", raising=False)
    assert _extract_live_continuous_batching_metadata({"live_continuous_batching": payload}) is None

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    observed = _extract_live_continuous_batching_metadata({"live_continuous_batching": payload})

    assert observed["claim_boundary"] == "live_continuous_batching_server_metadata_observed_no_parity_or_speedup"
    assert observed["opt_in_flag"] == "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
    assert observed["server_observed_live_continuous_batches"] is True
    assert observed["live_server_proven"] is True
    assert observed["speedup_proven"] is False
    assert observed["wallclock_speedup_proven"] is False
    assert observed["can_update_demo_status"] is False
    assert observed["tick_batches"][0]["request_ids"] == ["generate-0", "generate-1"]
    assert observed["tick_batches"][0]["active_mask"] == [True, False]

    assert _extract_live_continuous_batching_metadata({"live_continuous_batching": "not-a-dict"}) is None


def test_inference_session_live_tick_rows_fail_closed_without_opt_in(monkeypatch):
    from bloombee.client.inference_session import InferenceSession
    from bloombee.client.live_continuous_batching import LiveDecodeRow

    monkeypatch.delenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", raising=False)
    session = InferenceSession(SimpleNamespace(), max_length=4)

    with pytest.raises(RuntimeError, match="BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"):
        session.record_live_continuous_tick_rows(
            [LiveDecodeRow(request_id="generate-0", tick=0, position=0, input_token_id=101)],
            output_token_ids=[10],
        )


def test_live_continuous_batching_loop_report_does_not_promote_demo_status():
    import json

    tracked = json.loads(LIVE_LOOP_EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert tracked["claim_boundary"] == "live_continuous_decode_loop_unit_no_server_no_speedup"
    assert tracked["verification_status"] == "passed"
    assert tracked["live_loop_unit_proven"] is True
    assert tracked["live_server_proven"] is False
    assert tracked["speedup_proven"] is False
    assert tracked["can_update_demo_status"] is False
    assert tracked["do_not_claim"] == [
        "live_server_integration",
        "wall_clock_speedup",
        "demo_safe_promotion",
    ]


def test_kv_prefix_live_generate_metadata_evidence_is_claim_bounded():
    import json

    tracked = json.loads(KV_PREFIX_LIVE_GENERATE_EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert tracked["claim_boundary"] == "kv_prefix_reuse_live_generate_metadata_no_cache_reuse_or_speedup"
    assert tracked["verification_status"] == "passed"
    assert tracked["source"] == "tests/test_live_continuous_batching.py::test_live_generate_records_kv_prefix_metadata_for_same_prefix_batch"
    assert tracked["runtime_prefill_metadata_proven"] is True
    assert tracked["live_generate_metadata_attached_to_first_rpc"] is True
    assert tracked["live_kv_cache_reuse_proven"] is False
    assert tracked["speedup_proven"] is False
    assert tracked["can_update_demo_status"] is False
    assert tracked["do_not_claim"] == [
        "server_kv_tensor_reuse",
        "wall_clock_speedup",
        "demo_safe_promotion",
    ]


def test_live_continuous_batching_rejects_risky_generation_modes(monkeypatch):
    inputs = torch.tensor([[101]], dtype=torch.long)
    risky_kwargs = [
        {"do_sample": True},
        {"num_beams": 2},
        {"return_dict_in_generate": True},
        {"attention_mask": torch.tensor([[1]], dtype=torch.long)},
        {"logits_processor": [object()]},
        {"stopping_criteria": [object()]},
        {"generation_config": object()},
    ]

    monkeypatch.setenv("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "1")
    for kwargs in risky_kwargs:
        model = _RemoteGenerationStub()
        result = model.generate(inputs, max_new_tokens=1, **kwargs)
        assert result.tolist() == [[999]]
        assert model.live_calls == []
        assert len(model.fallback_calls) == 1

    model = _RemoteGenerationStub()
    result = model.generate(torch.tensor([[101, 102], [201, 202]], dtype=torch.long), max_new_tokens=1)
    assert result.tolist() == [[999]]
    assert model.live_calls == []

    model = _RemoteGenerationStub()
    explicit_session = SimpleNamespace(position=0, output_ids=None)
    result = model.generate(inputs, max_new_tokens=1, session=explicit_session)
    assert result.tolist() == [[999]]
    assert model.live_calls == []
