import torch

from bloombee.client.inference_session import (
    _server_session_tokens_to_advance,
    _trim_recovered_history_for_existing_downstream,
)
from bloombee.models.llama.model import _remote_seen_tokens_after_forward


def test_remote_seen_tokens_uses_session_position_after_recovery_slice():
    hidden_states = torch.zeros(1, 1, 8)

    assert _remote_seen_tokens_after_forward(hidden_states, remote_position=7) == 7


def test_remote_seen_tokens_falls_back_to_hidden_sequence_without_active_session():
    hidden_states = torch.zeros(1, 6, 8)

    assert _remote_seen_tokens_after_forward(hidden_states, remote_position=0) == 6


def test_replacement_server_session_advances_by_full_history_sent():
    sent_inputs = torch.zeros(1, 8, 16)

    assert _server_session_tokens_to_advance(sent_inputs, current_step_tokens=1, is_spec_dec=False) == 8


def test_speculative_server_session_keeps_logical_current_token_count():
    sent_inputs = torch.zeros(1, 8, 16)

    assert _server_session_tokens_to_advance(sent_inputs, current_step_tokens=2, is_spec_dec=True) == 2


def test_existing_downstream_stage_gets_only_current_token_after_upstream_recovery():
    full_history = torch.arange(7 * 3, dtype=torch.float32).view(1, 7, 3)

    trimmed = _trim_recovered_history_for_existing_downstream(
        full_history,
        current_step_tokens=1,
        downstream_position=6,
        is_spec_dec=False,
    )

    assert trimmed.shape == (1, 1, 3)
    assert torch.equal(trimmed, full_history[:, -1:, :])


def test_replacement_downstream_stage_keeps_full_history_for_cache_rebuild():
    full_history = torch.arange(7 * 3, dtype=torch.float32).view(1, 7, 3)

    rebuilt = _trim_recovered_history_for_existing_downstream(
        full_history,
        current_step_tokens=1,
        downstream_position=0,
        is_spec_dec=False,
    )

    assert torch.equal(rebuilt, full_history)


def test_speculative_downstream_stage_keeps_full_history_shape():
    full_history = torch.arange(7 * 3, dtype=torch.float32).view(1, 7, 3)

    spec = _trim_recovered_history_for_existing_downstream(
        full_history,
        current_step_tokens=1,
        downstream_position=6,
        is_spec_dec=True,
    )

    assert torch.equal(spec, full_history)
