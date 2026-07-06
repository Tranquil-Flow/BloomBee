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


def test_server_response_metadata_reports_kv_cache_reuse_only_after_real_handle_handoff(monkeypatch):
    from bloombee.server.handler import _build_rpc_inference_response_metadata, _extract_kv_prefix_reuse_metadata

    monkeypatch.setenv("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "1")
    metadata = {"kv_prefix_reuse_server_observed": _extract_kv_prefix_reuse_metadata({"kv_prefix_reuse": _kv_prefix_report()})}
    requested_uids = ("block.0",)
    cache_handles = ((11, 12),)

    cold = _build_rpc_inference_response_metadata(
        metadata,
        {"step_id": "cold", "_prefix_length": 0},
        requested_uids=requested_uids,
        cache_handles=cache_handles,
    )
    metadata_only_warm = _build_rpc_inference_response_metadata(
        metadata,
        {"step_id": "warm", "_prefix_length": 2},
        requested_uids=requested_uids,
        cache_handles=cache_handles,
    )

    assert "kv_prefix_reuse_server_observed" not in cold
    assert "kv_prefix_reuse_server_observed" not in metadata_only_warm

    metadata["kv_prefix_reuse_server_handoff"] = {
        "server_handle_handoff_observed": True,
        "cache_read_source_handle_id": 11,
        "cache_write_destination_handle_id": 12,
        "server_recovered_prefix_token_count": 2,
        "prefix_length": 2,
        "kv_prefix_byte_checksum_sha256": "a" * 64,
    }
    warm = _build_rpc_inference_response_metadata(
        metadata,
        {"step_id": "warm", "_prefix_length": 2},
        requested_uids=requested_uids,
        cache_handles=cache_handles,
    )

    observed = warm["kv_prefix_reuse_server_observed"]
    assert observed["claim_boundary"] == "kv_prefix_reuse_server_cache_read_observed_no_speedup"
    assert observed["server_observed_kv_cache_reuse"] is True
    assert observed["live_kv_cache_reuse_proven"] is True
    assert observed["server_handle_handoff_observed"] is True
    assert observed["cache_read_source_handle_id"] == 11
    assert observed["cache_write_destination_handle_id"] == 12
    assert observed["prefix_length"] == 2
    assert observed["cache_handle_count"] == 2
    assert observed["requested_block_count"] == 1
    assert observed["speedup_proven"] is False


def test_client_records_server_response_kv_cache_reuse(monkeypatch):
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

    response_metadata = {
        "kv_prefix_reuse_server_observed": {
            "claim_boundary": "kv_prefix_reuse_server_cache_read_observed_no_speedup",
            "server_observed_kv_cache_reuse": True,
            "live_kv_cache_reuse_proven": True,
            "prefix_length": 2,
            "cache_handle_count": 2,
        }
    }

    async def fake_step(_request):
        session.stepped = True
        return runtime_pb2.ExpertResponse(
            tensors=[serialize_torch_tensor(inputs, runtime_pb2.CompressionType.NONE)],
            metadata=MSGPackSerializer.dumps(response_metadata),
        )

    session._step = fake_step
    session.step(
        inputs,
        DUMMY,
        DUMMY_INT64,
        prefill_length=torch.zeros(inputs.shape[0]),
        step_id="step-1",
    )

    events = session.consume_server_response_metadata_events()
    assert events == [response_metadata]
