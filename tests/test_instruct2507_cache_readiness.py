from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.instruct2507_cache_readiness import CLAIM_BOUNDARY, _run_remote, validate_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "instruct2507_cache_readiness.py"


def _write_fixture(snapshot: Path, *, include_second_shard: bool = True) -> Path:
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3_moe"}', encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    index = {
        "metadata": {"total_size": 240},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a" * 128)
    if include_second_shard:
        blob = snapshot.parent / "blob-second"
        blob.write_bytes(b"b" * 128)
        (snapshot / "model-00002-of-00002.safetensors").symlink_to(blob)
    return snapshot


def test_validate_cache_ready_fixture_with_symlinked_shard(tmp_path: Path):
    snapshot = _write_fixture(tmp_path / "snapshot")
    status = tmp_path / "download.status"
    status.write_text("STATE=downloaded\nSHARD_COUNT=2\n", encoding="utf-8")

    report = validate_cache(
        snapshot_dir=snapshot,
        status_path=status,
        stage_root=tmp_path,
        expected_count=2,
        min_shard_bytes=1,
    )

    assert report["ready"] is True
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["present_shard_count"] == 2
    assert report["missing_shard_count"] == 0
    assert report["can_start_expensive_full_generation_gate"] is True
    assert report["generation_proven"] is False
    assert report["cache_generation_proven"] is False
    assert report["load_proven"] is False


def test_validate_cache_blocks_missing_shard_and_reports_stage_part(tmp_path: Path):
    snapshot = _write_fixture(tmp_path / "snapshot", include_second_shard=False)
    status = tmp_path / "download.status"
    status.write_text(
        "STATE=downloading\nSHARD_COUNT=1\nCURRENT_FILE=model-00002-of-00002.safetensors\nCURRENT_PHASE=curl\n",
        encoding="utf-8",
    )
    (tmp_path / "instruct2507-stage-model-00002-of-00002.safetensors.part").write_bytes(b"partial")

    report = validate_cache(
        snapshot_dir=snapshot,
        status_path=status,
        stage_root=tmp_path,
        expected_count=2,
        min_shard_bytes=1,
    )

    assert report["ready"] is False
    assert report["present_shard_count"] == 1
    assert report["missing_shards"] == ["model-00002-of-00002.safetensors"]
    assert report["stage_part"]["bytes"] == len(b"partial")
    assert report["can_start_expensive_full_generation_gate"] is False
    assert "missing 1 expected shard(s)" in report["errors"]


def test_instruct2507_cache_readiness_cli_json_and_markdown(tmp_path: Path):
    snapshot = _write_fixture(tmp_path / "snapshot")
    status = tmp_path / "download.status"
    status.write_text("STATE=downloaded\nSHARD_COUNT=2\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json",
            "--snapshot-dir",
            str(snapshot),
            "--status-path",
            str(status),
            "--stage-root",
            str(tmp_path),
            "--expected-shard-count",
            "2",
            "--min-shard-bytes",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ready"] is True
    assert payload["claim_boundary"] == CLAIM_BOUNDARY

    md = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot-dir",
            str(snapshot),
            "--status-path",
            str(status),
            "--stage-root",
            str(tmp_path),
            "--expected-shard-count",
            "2",
            "--min-shard-bytes",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert md.returncode == 0
    assert "# Instruct-2507 cache readiness — READY" in md.stdout
    assert "cache_download_readiness_only_no_generation_or_load_proof" in md.stdout


def test_remote_validator_quotes_snapshot_paths_with_spaces(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True, "ready": False, "claim_boundary": CLAIM_BOUNDARY}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(
        snapshot_dir="/Volumes/Seagate Portable Drive/huggingface/hub/snapshots/moon",
        status_path="~/Projects/distributed-inference-mvp/.local/status/instruct2507-full-download.status",
        stage_root="/Volumes/Exchange",
        expected_shard_count=16,
        min_shard_bytes=1,
        remote_host="m4pro",
        remote_timeout=45,
    )

    report = _run_remote(args)

    assert report["ok"] is True
    remote_command = captured["command"][-1]
    assert "'/Volumes/Seagate Portable Drive/huggingface/hub/snapshots/moon'" in remote_command
    assert captured["kwargs"]["input"].startswith("#!/usr/bin/env python3")



def test_instruct2507_cache_readiness_remote_mode_is_explicit_opt_in():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0
    assert "--remote" in proc.stdout
    assert "Run this same validator on m4pro over SSH" in proc.stdout
