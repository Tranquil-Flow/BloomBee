from __future__ import annotations

import sys

from bloombee.cli import run_server
from bloombee.utils.convert_block import QuantType


class _FakeServer:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs

    def run(self):
        return None

    def shutdown(self):
        return None


def _invoke(monkeypatch, *args: str):
    _FakeServer.captured = None
    monkeypatch.setattr(run_server, "Server", _FakeServer)
    monkeypatch.setattr(run_server, "validate_version", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_server.py", "test/model", "--new_swarm", "--throughput", "dry_run", *args],
    )
    run_server.main()
    return _FakeServer.captured


def test_run_server_cli_defaults_quant_type_to_none(monkeypatch):
    captured = _invoke(monkeypatch)

    assert captured["quant_type"] is QuantType.NONE


def test_run_server_cli_accepts_uppercase_quant_type(monkeypatch):
    captured = _invoke(monkeypatch, "--quant_type", "INT8")

    assert captured["quant_type"] is QuantType.INT8


def test_run_server_cli_accepts_lowercase_quant_type(monkeypatch):
    captured = _invoke(monkeypatch, "--quant_type", "nf4")

    assert captured["quant_type"] is QuantType.NF4
