from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "text_generation_parity.py"


def _load_parity_module():
    spec = importlib.util.spec_from_file_location("text_generation_parity_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_server_placements_records_host_layers_and_matching_maddr():
    parity = _load_parity_module()
    maddrs = [
        "/ip4/192.168.178.37/tcp/31337/p2p/seed",
        "/ip4/192.168.178.37/tcp/31338/p2p/mid",
        "/ip4/192.168.178.37/tcp/31339/p2p/tail",
    ]

    placements = parity.parse_server_placements(
        ["m4pro-seed=0:8", "m4pro-mid=8:15", "m4pro-tail=15:22"],
        maddrs,
    )

    assert placements == [
        {"host": "m4pro-seed", "layers": [0, 8], "server_maddr": maddrs[0]},
        {"host": "m4pro-mid", "layers": [8, 15], "server_maddr": maddrs[1]},
        {"host": "m4pro-tail", "layers": [15, 22], "server_maddr": maddrs[2]},
    ]


def test_parse_server_placements_requires_one_entry_per_server():
    parity = _load_parity_module()

    try:
        parity.parse_server_placements(["m4pro=0:22"], ["a", "b"])
    except ValueError as exc:
        assert "one --server-placement per --server-maddr" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ValueError")


def test_parse_server_placements_rejects_bad_layer_range():
    parity = _load_parity_module()

    try:
        parity.parse_server_placements(["m4pro=8:8"], ["a"])
    except ValueError as exc:
        assert "start:end" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ValueError")
