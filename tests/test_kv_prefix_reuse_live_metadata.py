import asyncio
from types import SimpleNamespace

import pytest
import torch
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.utils.hivemind_compat import MSGPackSerializer
from bloombee.utils.lossless_transport import serialize_torch_tensor
from bloombee.utils.misc import DUMMY, DUMMY_INT64


class _FakeServerSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None


class _FakeSequenceManager:
    def __init__(self):
        self.block_uids = ["block.0"]
        self.config = SimpleNamespace(use_server_to_server=False, request_timeout=1.0)
        self.state = SimpleNamespace(p2p=object())
        self.rpc_info = {}

    def __len__(self):
        return len(self.block_uids)

    def get_request_metadata(self, protocol, *args, **kwargs):
        assert protocol == "rpc_inference"
        return {"existing_metadata": "kept"}


def _kv_prefix_report():
    return {
        "source": "test",
        "claim_boundary": "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse",
        "opt_in_flag": "BLOOMBEE_ENABLE_KV_PREFIX_REUSE",
        "opt_in_enabled": True,
        "event_count": 1,
        "events": [
            {
                "common_prefix_token_ids": [101, 102],
                "request_count": 2,
                "same_prefix_varied_suffix_proven": True,
                "runtime_prefill_metadata_proven": True,
                "live_kv_cache_reuse_proven": False,
                "speedup_proven": False,
                "can_update_demo_status": False,
            }
        ],
        "runtime_prefill_metadata_proven": True,
        "live_kv_cache_reuse_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
    }


def test_kv_prefix_report_is_attached_to_first_server_session_metadata(monkeypatch):
    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import InferenceSession

    monkeypatch.setenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "1")
    captured_create_kwargs = {}

    async def fake_create(*args, **kwargs):
        captured_create_kwargs.update(kwargs)
        return _FakeServerSession()

    monkeypatch.setattr(
        inference_session_mod.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )
    monkeypatch.setattr(inference_session_mod._ServerInferenceSession, "create", fake_create)

    session = InferenceSession(_FakeSequenceManager(), max_length=8)
    session.record_kv_prefix_reuse_prefill([[101, 102, 201], [101, 102, 202]], request_ids=["a", "b"])

    span = SimpleNamespace(start=0, end=1, peer_id="peer-1")
    created = session._enter_server_sessions([span])

    assert created and isinstance(created[0], _FakeServerSession)
    assert captured_create_kwargs["existing_metadata"] == "kept"
    report = captured_create_kwargs["kv_prefix_reuse"]
    assert report["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert report["runtime_prefill_metadata_proven"] is True
    assert report["live_kv_cache_reuse_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_kv_prefix_metadata_is_sent_in_first_rpc_inference_request(monkeypatch):
    from bloombee.client import inference_session as inference_session_mod
    from bloombee.client.inference_session import _ServerInferenceSession

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
        kv_prefix_reuse=_kv_prefix_report(),
    )
    captured_request = {}

    async def fake_step(request):
        captured_request["request"] = request
        session.stepped = True
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(inputs, runtime_pb2.CompressionType.NONE)]
        )

    session._step = fake_step

    outputs = session.step(
        inputs,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.zeros(inputs.shape[0]),
        step_id="step-1",
    )

    assert torch.equal(outputs[0], inputs)
    metadata = MSGPackSerializer.loads(captured_request["request"].metadata)
    assert metadata["session_id"] == session.session_id
    assert metadata["step_id"] == "step-1"
    assert metadata["max_length"] == 8
    assert metadata["kv_prefix_reuse"]["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert metadata["kv_prefix_reuse"]["live_kv_cache_reuse_proven"] is False
    assert metadata["kv_prefix_reuse"]["speedup_proven"] is False


def test_server_observes_kv_prefix_reuse_metadata_only_with_opt_in(monkeypatch):
    from bloombee.server.handler import _extract_kv_prefix_reuse_metadata

    payload = _kv_prefix_report()
    metadata = {"kv_prefix_reuse": payload}

    monkeypatch.delenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", raising=False)
    assert _extract_kv_prefix_reuse_metadata(metadata) is None

    monkeypatch.setenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "1")
    observed = _extract_kv_prefix_reuse_metadata(metadata)

    assert observed["claim_boundary"] == "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
    assert observed["runtime_prefill_metadata_proven"] is True
    assert observed["server_observed_metadata"] is True
    assert observed["live_kv_cache_reuse_proven"] is False
    assert observed["speedup_proven"] is False
    assert observed["can_update_demo_status"] is False

    assert _extract_kv_prefix_reuse_metadata({"kv_prefix_reuse": "not-a-dict"}) is None
