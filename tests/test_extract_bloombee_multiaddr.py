from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "extract_bloombee_multiaddr.py"

SERVER_LOG = """
[INFO] This server is accessible directly
[INFO] Running a server on [
  /ip4/100.84.252.4/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn,
  /ip4/127.0.0.1/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn,
  /ip6/::1/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn]
[INFO] Inference throughput: 328.3 tokens/sec per block
"""


def test_extract_bloombee_multiaddr_prefers_non_loopback_ip4_and_keeps_claim_boundary():
    mod = import_module("scripts.extract_bloombee_multiaddr")

    report = mod.build_multiaddr_report(SERVER_LOG, source="fixture.log")

    assert report["ok"] is True
    assert report["claim_boundary"] == "server_log_multiaddr_extraction_only_no_connectivity_proof"
    assert report["preferred_multiaddr"] == "/ip4/100.84.252.4/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn"
    assert report["multiaddr_count"] == 3
    assert "/ip4/127.0.0.1/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn" in report["loopback_multiaddrs"]
    assert report["connectivity_proven"] is False
    assert report["server_liveness_proven"] is False


def test_extract_bloombee_multiaddr_cli_reads_log_file_as_json(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text(SERVER_LOG, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", str(log_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["preferred_multiaddr"].startswith("/ip4/100.84.252.4/tcp/31337/p2p/")
    assert payload["source"] == str(log_path)


def test_extract_bloombee_multiaddr_blocks_when_no_multiaddr_present(tmp_path: Path):
    mod = import_module("scripts.extract_bloombee_multiaddr")

    report = mod.build_multiaddr_report("[INFO] booting but no address yet", source="empty.log")

    assert report["ok"] is False
    assert report["preferred_multiaddr"] is None
    assert report["blocked_reason"] == "no /ip4 or /ip6 tcp/p2p multiaddr found in log"

    log_path = tmp_path / "empty.log"
    log_path.write_text("[INFO] booting but no address yet", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", str(log_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
