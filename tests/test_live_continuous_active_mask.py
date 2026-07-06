from types import SimpleNamespace

import torch


class _RecordingCacheManager:
    def __init__(self):
        self.update_calls = []

    def update_cache(self, new_kvs, start_position, **kwargs):
        self.update_calls.append((new_kvs, start_position, kwargs))


class _DummyBackend:
    from bloombee.server.backend import TransformerBackend


def _kv_for_batch(batch_size: int, heads: int = 2, dim: int = 3, tokens: int = 1):
    bh = batch_size * heads
    key = torch.arange(bh * dim * tokens, dtype=torch.float32).reshape(bh, dim, tokens)
    value = (1000 + torch.arange(bh * tokens * dim, dtype=torch.float32)).reshape(bh, tokens, dim)
    return key, value


def test_live_continuous_active_mask_is_preserved_and_microbatch_sliced():
    from bloombee.server.microbatch import build_inference_metadata, slice_microbatch_inputs

    active_mask = torch.tensor([True, False, True, False])
    info = build_inference_metadata(
        "uid",
        (),
        0,
        None,
        active_mask=active_mask,
        batch_offset=0,
        full_batch_size=4,
        micro_batch_size=4,
    )

    assert info.active_mask.tolist() == [True, False, True, False]

    mb = slice_microbatch_inputs(
        torch.zeros(4, 1, 8),
        torch.arange(4),
        None,
        None,
        None,
        None,
        None,
        active_mask,
        mb_start=1,
        mb_end=3,
        full_batch_size=4,
    )

    assert mb.active_mask.tolist() == [False, True]


def test_server_extracts_active_mask_slice_for_compact_live_continuous_microbatch():
    from bloombee.server.block_functions import _live_active_mask_from_step_metadata

    metadata = {
        "batch_offset": 1,
        "micro_batch_size": 1,
        "full_batch_size": 2,
        "live_continuous_batching": {
            "tick_batches": [
                {
                    "tick": 2,
                    "request_ids": ["generate-0", "generate-1"],
                    "positions": [2, 1],
                    "active_mask": [False, True],
                }
            ]
        },
    }

    mask = _live_active_mask_from_step_metadata(metadata, batch_size=1)

    assert mask is not None
    assert mask.tolist() == [True]


def test_live_continuous_mixed_positions_build_row_specific_decode_mask_and_position_ids():
    from bloombee.server.backend import (
        _live_decode_attention_mask_for_row_positions,
        _live_decode_position_ids_for_row_positions,
    )

    mask = _live_decode_attention_mask_for_row_positions(
        cache_len=2,
        row_positions=[2, 1],
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [[[True, True, True]], [[True, False, True]]]

    position_ids = _live_decode_position_ids_for_row_positions(
        row_positions=[2, 1],
        chunk_length=1,
        offset=0,
        device=torch.device("cpu"),
    )
    assert position_ids.tolist() == [[2], [1]]


def test_backend_cache_update_uses_row_prefill_positions_for_mixed_live_rows():
    from bloombee.server.backend import TransformerBackend

    backend = object.__new__(TransformerBackend)
    backend.cache_manager = _RecordingCacheManager()
    backend._is_spec_decoding = False
    backend._is_linear_attention_block = lambda: False

    info = SimpleNamespace(
        uid="uid",
        batch_offset=0,
        full_batch_size=2,
        micro_batch_size=2,
        active_mask=torch.tensor([True, True]),
        prefill_length=torch.tensor([1, 0]),
    )
    key, value = _kv_for_batch(batch_size=2, heads=2)

    backend._finalize_cache_update(
        (key, value),
        cache_len=1,
        inference_info=info,
        kv_cache_position_ids=None,
        cache_tensors=None,
    )

    calls = backend.cache_manager.update_calls
    assert len(calls) == 2
    (first_k, first_v), first_pos, first_kwargs = calls[0]
    (second_k, second_v), second_pos, second_kwargs = calls[1]

    assert first_pos == 1
    assert second_pos == 0
    assert first_kwargs == {"batch_offset": 0, "full_batch_size": 2, "micro_batch_size": 1}
    assert second_kwargs == {"batch_offset": 1, "full_batch_size": 2, "micro_batch_size": 1}
    assert torch.equal(first_k, key[0:2])
    assert torch.equal(first_v, value[0:2])
    assert torch.equal(second_k, key[2:4])
    assert torch.equal(second_v, value[2:4])


def test_backend_cache_update_writes_only_active_live_continuous_rows():
    from bloombee.server.backend import TransformerBackend

    backend = object.__new__(TransformerBackend)
    backend.cache_manager = _RecordingCacheManager()
    backend._is_spec_decoding = False
    backend._is_linear_attention_block = lambda: False

    info = SimpleNamespace(
        uid="uid",
        batch_offset=0,
        full_batch_size=3,
        micro_batch_size=3,
        active_mask=torch.tensor([True, False, True]),
    )
    key, value = _kv_for_batch(batch_size=3, heads=2)

    backend._finalize_cache_update(
        (key, value),
        cache_len=5,
        inference_info=info,
        kv_cache_position_ids=None,
        cache_tensors=None,
    )

    calls = backend.cache_manager.update_calls
    assert len(calls) == 2
    (first_k, first_v), first_pos, first_kwargs = calls[0]
    (second_k, second_v), second_pos, second_kwargs = calls[1]

    assert first_pos == second_pos == 5
    assert first_kwargs == {"batch_offset": 0, "full_batch_size": 3, "micro_batch_size": 1}
    assert second_kwargs == {"batch_offset": 2, "full_batch_size": 3, "micro_batch_size": 1}
    assert torch.equal(first_k, key[0:2])
    assert torch.equal(first_v, value[0:2])
    assert torch.equal(second_k, key[4:6])
    assert torch.equal(second_v, value[4:6])
