import importlib


def _reload_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE", raising=False)
    else:
        monkeypatch.setenv("BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE", value)
    import bloombee.client.config as config

    return importlib.reload(config)


def test_push_only_downstream_decode_defaults_to_direct_fallback(monkeypatch):
    config = _reload_config(monkeypatch, None)

    assert config.DEFAULT_PUSH_ONLY_DOWNSTREAM_DECODE is False
    assert config.ClientConfig().push_only_downstream_decode is False


def test_push_only_downstream_decode_can_be_opted_in(monkeypatch):
    config = _reload_config(monkeypatch, "1")

    assert config.DEFAULT_PUSH_ONLY_DOWNSTREAM_DECODE is True
    assert config.ClientConfig().push_only_downstream_decode is True


def test_push_only_downstream_decode_explicit_false_values(monkeypatch):
    for value in ("0", "false", "no", "off"):
        config = _reload_config(monkeypatch, value)
        assert config.DEFAULT_PUSH_ONLY_DOWNSTREAM_DECODE is False
        assert config.ClientConfig().push_only_downstream_decode is False
