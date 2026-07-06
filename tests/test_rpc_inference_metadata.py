from bloombee.server.block_functions import _fullbatch_metadata_kwargs


_KV_OBSERVED = {
    "common_prefix_token_count": 3,
    "request_count": 2,
}

_KV_REPORT_OBSERVED = {
    "event_count": 1,
    "events": [
        {
            "common_prefix_token_ids": [101, 102],
            "request_count": 2,
            "requests": [{"request_id": "a"}, {"request_id": "b"}],
        }
    ],
}


def test_fullbatch_metadata_defaults_to_live_batch_size_when_client_metadata_absent():
    assert _fullbatch_metadata_kwargs({}, batch_size=1) == {
        "batch_offset": 0,
        "full_batch_size": 1,
        "micro_batch_size": 1,
    }


def test_fullbatch_metadata_preserves_client_batch_sizes():
    assert _fullbatch_metadata_kwargs(
        {"batch_offset": 2, "full_batch_size": 8, "micro_batch_size": 3},
        batch_size=8,
    ) == {
        "batch_offset": 2,
        "full_batch_size": 8,
        "micro_batch_size": 3,
    }


def test_fullbatch_metadata_accepts_cross_stage_aliases():
    assert _fullbatch_metadata_kwargs(
        {"offset": 4, "size": 2, "full_batch_size": 9},
        batch_size=9,
    ) == {
        "batch_offset": 4,
        "full_batch_size": 9,
        "micro_batch_size": 2,
    }


def test_fullbatch_metadata_propagates_kv_prefix_reuse_runtime_fields():
    assert _fullbatch_metadata_kwargs(
        {"kv_prefix_reuse_server_observed": _KV_OBSERVED},
        batch_size=2,
    ) == {
        "batch_offset": 0,
        "full_batch_size": 2,
        "micro_batch_size": 2,
        "kv_prefix_reuse_enabled": True,
        "kv_prefix_reuse_common_prefix_token_count": 3,
        "kv_prefix_reuse_request_count": 2,
    }


def test_fullbatch_metadata_propagates_kv_prefix_reuse_report_event_fields():
    assert _fullbatch_metadata_kwargs(
        {"kv_prefix_reuse_server_observed": _KV_REPORT_OBSERVED},
        batch_size=2,
    ) == {
        "batch_offset": 0,
        "full_batch_size": 2,
        "micro_batch_size": 2,
        "kv_prefix_reuse_enabled": True,
        "kv_prefix_reuse_common_prefix_token_count": 2,
        "kv_prefix_reuse_request_count": 2,
    }
