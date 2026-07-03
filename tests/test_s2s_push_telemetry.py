from bloombee.server.handler import _format_s2s_push_event, _s2s_push_error_code


def test_s2s_push_error_code_classifies_rpc_timeout():
    assert _s2s_push_error_code(TimeoutError("rpc_push timed out")) == "rpc_push_timeout"


def test_s2s_push_event_is_structured_for_log_scraping():
    event = _format_s2s_push_event(
        event_type="push_failed",
        action="direct_fallback",
        reason="rpc_push_timeout",
        step_id=7,
        from_blocks="0:8",
        to_blocks="8:15",
        to_peer="peer123",
        session_id="session456",
        tensor_bytes=1024,
        metadata_bytes=128,
        elapsed_ms=12.5,
    )

    assert event.startswith("[S2S_PUSH_EVENT]")
    assert "type=push_failed" in event
    assert "action=direct_fallback" in event
    assert "reason=rpc_push_timeout" in event
    assert "step_id=7" in event
    assert "from_blocks=0:8" in event
    assert "to_blocks=8:15" in event
    assert "to_peer=peer123" in event
    assert "session_id=session456" in event
    assert "tensor_bytes=1024" in event
    assert "metadata_bytes=128" in event
    assert "elapsed_ms=12.50" in event
