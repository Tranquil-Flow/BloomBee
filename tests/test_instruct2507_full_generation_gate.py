from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.instruct2507_full_generation_gate import (
    CLAIM_BOUNDARY,
    MODEL_ID,
    PLACEHOLDER_MADDR,
    build_gate_plan,
    build_server_launch_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "instruct2507_full_generation_gate.py"


def _readiness(*, ready: bool) -> dict:
    return {
        "ok": True,
        "ready": ready,
        "claim_boundary": "cache_download_readiness_only_no_generation_or_load_proof",
        "model_id": MODEL_ID,
        "present_shard_count": 16 if ready else 5,
        "expected_shard_count": 16,
        "first_missing_shard": None if ready else "model-00006-of-00016.safetensors",
        "can_start_expensive_full_generation_gate": ready,
        "errors": [] if ready else ["missing 11 expected shard(s)"],
    }


def test_gate_plan_blocks_until_cache_ready_and_server_maddr_captured():
    plan = build_gate_plan(readiness=_readiness(ready=False))

    assert plan["ready_to_attempt_full_generation"] is False
    assert plan["claim_boundary"] == CLAIM_BOUNDARY
    assert plan["cache_readiness"]["present_shard_count"] == 5
    assert any("cache readiness is BLOCKED" in reason for reason in plan["blocked_reasons"])
    assert any("server multiaddr is not captured" in reason for reason in plan["blocked_reasons"])
    assert PLACEHOLDER_MADDR in plan["full_generation_plan"]["parity_command"]
    assert plan["generation_proven"] is False
    assert plan["cache_generation_proven"] is False
    assert plan["load_proven"] is False
    assert plan["can_update_proof_status"] is False


def test_gate_plan_ready_with_server_maddr_emits_full_generation_commands():
    maddr = "/ip4/192.168.178.37/tcp/31347/p2p/12D3KooWMoon"
    plan = build_gate_plan(readiness=_readiness(ready=True), server_maddrs=[maddr])

    assert plan["ready_to_attempt_full_generation"] is True
    assert plan["blocked_reasons"] == []
    proof = plan["full_generation_plan"]
    assert proof["model_id"] == MODEL_ID
    assert proof["proof_gate"] == "full_generation"
    assert proof["claim_boundary"] == "full_generation_proof_harness_only_no_live_generation"
    assert "scripts/text_generation_parity.py" in proof["parity_command"]
    assert "--mode forward-loop" in proof["parity_command"]
    assert "--max-new-tokens 1" in proof["parity_command"]
    assert "m4pro-full=0:48" in proof["parity_command"]
    assert maddr in proof["parity_command"]
    assert "full_generation_proof.py verify" in proof["verify_command"]
    assert "--require-server-placements" in proof["verify_command"]


def test_server_launch_command_uses_real_run_server_flags_and_external_cache():
    command = build_server_launch_command()

    assert "HF_HUB_DISABLE_XET=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "python -m bloombee.cli.run_server" in command
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in command
    assert "--block_indices 0:48" in command
    assert "--new_swarm" in command
    assert "--cache_dir '/Volumes/Seagate Portable Drive/huggingface/hub'" in command
    assert "--skip_reachability_check" in command
    assert "BLOOMBEE_INITIAL_PEERS" not in command


def test_instruct2507_full_generation_gate_cli_blocks_from_fixture(tmp_path: Path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(_readiness(ready=False)), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--readiness-json", str(readiness_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ready_to_attempt_full_generation"] is False
    assert payload["claim_boundary"] == CLAIM_BOUNDARY
    assert payload["cache_readiness"]["first_missing_shard"] == "model-00006-of-00016.safetensors"


def test_instruct2507_full_generation_gate_markdown_names_negative_flags(tmp_path: Path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(_readiness(ready=True)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--readiness-json",
            str(readiness_path),
            "--server-maddr",
            "/ip4/192.168.178.37/tcp/31347/p2p/12D3KooWMoon",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "# Instruct-2507 full-generation gate plan — READY TO ATTEMPT" in proc.stdout
    assert "instruct2507_full_generation_gate_plan_only_no_live_generation" in proc.stdout
    assert "generation_proven: `False`" in proc.stdout
    assert "cache_generation_proven: `False`" in proc.stdout
    assert "load_proven: `False`" in proc.stdout
