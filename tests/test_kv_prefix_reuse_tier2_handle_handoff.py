from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("hivemind")

import torch

from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.server.prefix_index import KVPrefixIndex, prefix_tokens_sha256


class _TT:
    def __init__(self, tensor: torch.Tensor):
        self.data = tensor
        self.shape = tensor.shape


def _manager_with_slabs(*, current_size_tokens: int = 1600):
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.cache = SimpleNamespace(_allocated_tensors={}, current_size_tokens=current_size_tokens)
    return manager


def _kv_pair(s_total: int = 8, bh: int = 4, d: int = 3, *, fill: float = 0.0):
    return (
        _TT(torch.full((s_total, bh, d), fill, dtype=torch.float32)),
        _TT(torch.full((s_total, bh, d), fill + 100.0, dtype=torch.float32)),
    )


def test_kv_prefix_index_lookup_round_trip_and_lru_eviction():
    index = KVPrefixIndex(max_size_tokens=5)
    first = index.register([101, 102, 103], handle=11)

    assert first.prefix_length == 3
    assert first.token_sha256 == prefix_tokens_sha256([101, 102, 103])
    assert index.lookup([101, 102, 103, 201]) == first
    assert index.lookup([99, 102, 103]) is None

    second = index.register([201, 202, 203], handle=12)

    # 3 + 3 > cap=5, so the least-recently-used prefix is evicted.
    assert index.current_size_tokens == 3
    assert len(index) == 1
    assert index.lookup([101, 102, 103, 201]) is None
    assert index.lookup([201, 202, 203, 204]) == second


def test_kv_cache_manager_copy_prefix_from_handle_copies_kv_bytes():
    manager = _manager_with_slabs()
    src_kv = _kv_pair()
    dst_kv = _kv_pair(fill=-1.0)
    manager.cache._allocated_tensors[101] = src_kv
    manager.cache._allocated_tensors[202] = dst_kv

    src_k, src_v = src_kv
    dst_k, dst_v = dst_kv
    src_k.data.copy_(torch.arange(src_k.data.numel(), dtype=torch.float32).reshape_as(src_k.data))
    src_v.data.copy_(src_k.data + 10_000.0)

    report = manager.copy_prefix_from_handle(
        source_handle=101,
        destination_handle=202,
        prefix_length=5,
        bh_slice=(1, 3),
    )

    torch.testing.assert_close(dst_k.data[:5, 1:3], src_k.data[:5, 1:3])
    torch.testing.assert_close(dst_v.data[:5, 1:3], src_v.data[:5, 1:3])
    torch.testing.assert_close(dst_k.data[:5, :1], torch.full_like(dst_k.data[:5, :1], -1.0))
    torch.testing.assert_close(dst_v.data[:5, :1], torch.full_like(dst_v.data[:5, :1], 99.0))
    assert report["server_handle_handoff_observed"] is True
    assert report["cache_read_source_handle_id"] == 101
    assert report["cache_write_destination_handle_id"] == 202
    assert report["server_recovered_prefix_token_count"] == 5
    assert len(report["kv_prefix_byte_checksum_sha256"]) == 64


def test_kv_cache_manager_copy_prefix_does_not_double_count_cache_tokens():
    manager = _manager_with_slabs(current_size_tokens=1600)
    manager.cache._allocated_tensors[101] = _kv_pair()
    manager.cache._allocated_tensors[202] = _kv_pair(fill=-1.0)

    before = manager.cache.current_size_tokens
    manager.copy_prefix_from_handle(
        source_handle=101,
        destination_handle=202,
        prefix_length=4,
    )

    assert manager.cache.current_size_tokens == before


def test_kv_cache_manager_copy_prefix_rejects_same_handle():
    manager = _manager_with_slabs()
    manager.cache._allocated_tensors[101] = _kv_pair()

    with pytest.raises(ValueError, match="distinct"):
        manager.copy_prefix_from_handle(source_handle=101, destination_handle=101, prefix_length=1)
