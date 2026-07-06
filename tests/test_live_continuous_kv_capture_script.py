from __future__ import annotations

import importlib.util
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "live_continuous_kv_capture.py"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("live_continuous_kv_capture", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_command_bootstraps_server_into_private_dht_peer():
    module = _load_script_module()
    initial_peer = "/ip4/127.0.0.1/tcp/31337/p2p/12D3KooWprivate"

    command = module.build_server_command(
        model_id=MODEL_ID,
        initial_peer=initial_peer,
        num_blocks=4,
        port=31338,
        batch_size=2,
        device="mps",
    )

    assert command[:3] == [module.sys.executable, "-m", "bloombee.cli.run_server"]
    assert command[3] == MODEL_ID
    assert "--initial_peers" in command
    assert command[command.index("--initial_peers") + 1] == initial_peer
    assert "--new_swarm" not in command
    assert "--num_blocks" in command
    assert command[command.index("--num_blocks") + 1] == "4"


def test_blocked_evidence_keeps_continuous_and_kv_claims_false():
    module = _load_script_module()

    payload = module.build_blocked_live_capture_evidence(
        reason="socket_bind_denied",
        detail="Operation not permitted while binding 127.0.0.1:31337",
        model_id=MODEL_ID,
    )

    assert payload["claim_boundary"] == "live_continuous_kv_capture_blocked_no_live_server_proof"
    assert payload["model"] == MODEL_ID
    assert payload["blocked"] is True
    assert payload["blocker"] == "socket_bind_denied"
    assert payload["continuous_batching"]["live_server_late_arrival_parity_proven"] is False
    assert payload["continuous_batching"]["speedup_proven"] is False
    assert payload["kv_prefix_reuse"]["live_kv_cache_reuse_proven"] is False
    assert payload["kv_prefix_reuse"]["speedup_proven"] is False
    assert payload["can_update_demo_status"] is False
    assert payload["can_update_proof_status"] is False


def test_wait_for_server_blocks_fails_fast_when_server_process_exits(tmp_path, monkeypatch):
    module = _load_script_module()

    class ExitedProcess:
        def poll(self):
            return 17

    monkeypatch.setattr(module, "SERVER_LOG", tmp_path / "server.log")
    started = time.monotonic()

    ready = module.wait_for_server_blocks(
        timeout=10,
        server_proc=ExitedProcess(),
        poll_interval=0.001,
    )

    assert ready is False
    assert time.monotonic() - started < 0.5


def test_reset_capture_workspace_removes_stale_peer_and_marker_files(tmp_path, monkeypatch):
    module = _load_script_module()
    evidence_dir = tmp_path / "evidence"
    server_log = evidence_dir / "server.log"
    peer_file = evidence_dir / "dht_peer.txt"
    marker_file = evidence_dir / "dht_ready.marker"
    evidence_dir.mkdir()
    server_log.write_text("old server log", encoding="utf-8")
    peer_file.write_text("12D3KooWstale", encoding="utf-8")
    marker_file.write_text("ready", encoding="utf-8")
    monkeypatch.setattr(module, "EVIDENCE_DIR", evidence_dir)
    monkeypatch.setattr(module, "SERVER_LOG", server_log)

    module.reset_capture_workspace()

    assert evidence_dir.exists()
    assert not server_log.exists()
    assert not peer_file.exists()
    assert not marker_file.exists()
