from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bloombee.client.remote_generation import RemoteGenerationMixin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_LOOP_EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/live-continuous-batching-loop-unit-20260705.json"


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
    def __init__(self):
        self._active_session = None
        self.hidden_calls = []
        self.sessions = []

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

    def __call__(self, hidden_states):
        assert self._active_session is not None
        self.hidden_calls.append(hidden_states.detach().clone())
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
    assert report["tick_batches"] == [
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
