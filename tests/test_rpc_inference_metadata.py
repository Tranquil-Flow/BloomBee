from bloombee.server.block_functions import _fullbatch_metadata_kwargs


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
