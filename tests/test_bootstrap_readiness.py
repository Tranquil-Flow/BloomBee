"""Bootstrap readiness detection + weight-cache preflight.

Regression tests for the masking bug where scripts/bootstrap.py reported a
peer as `serving` the instant the server printed "Running a server on ..."
— which happens *before* model weights are loaded. When the weights were not
in the HF cache, the server hung forever downloading, never announced its
blocks, yet the coordinator dashboard showed it green. See
inference-blocker-missing-weights memory / HANDOVER.md.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

bootstrap = import_module("scripts.bootstrap")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── is_server_ready_line: only the real readiness marker counts ──────────────

def test_running_a_server_on_is_not_ready():
    """"Running a server on ..." prints before weights load — NOT ready."""
    line = ("Jul 08 07:38:36.884 [INFO] Running a server on "
            "['/ip4/127.0.0.1/tcp/31370/p2p/12D3KooWCq6dhdgXTKby162P266D1s7yAZCeSCd355N8dA6Taf4J']")
    assert bootstrap.is_server_ready_line(line) is False


def test_announced_joining_is_not_ready():
    """"Announced that blocks ... are joining" fires before weights load."""
    line = "Jul 08 07:38:36.994 [INFO] Announced that blocks [0, 1, 2] are joining"
    assert bootstrap.is_server_ready_line(line) is False


def test_loading_weights_is_not_ready():
    line = "Jul 08 07:36:36.549 [INFO] Loading HF weights for model.layers.0. from Qwen/Qwen3-8B"
    assert bootstrap.is_server_ready_line(line) is False


def test_started_marker_is_ready():
    """hivemind Runtime logs "Started" right after ready.set() — the real signal."""
    line = "Jul 08 07:38:40.504 [INFO] Started"
    assert bootstrap.is_server_ready_line(line) is True


def test_bare_started_line_is_ready():
    assert bootstrap.is_server_ready_line("Started") is True


def test_lowercase_computation_started_is_not_ready():
    """"Inference computation started - step 5" must not be mistaken for readiness."""
    line = "[DEBUG]  Inference computation started - step 5"
    assert bootstrap.is_server_ready_line(line) is False


# ── model_weights_cached: preflight so we fail fast instead of hanging ────────

def _make_snapshot(tmp_path, model_id):
    d = tmp_path / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc123"
    d.mkdir(parents=True)
    return d


def test_weights_missing_when_only_config(tmp_path):
    d = _make_snapshot(tmp_path, "Qwen/Qwen3-8B")
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert bootstrap.model_weights_cached("Qwen/Qwen3-8B", cache_dir=tmp_path) is False


def test_weights_missing_when_model_dir_absent(tmp_path):
    assert bootstrap.model_weights_cached("Qwen/Qwen3-8B", cache_dir=tmp_path) is False


def test_weights_index_json_alone_is_not_enough(tmp_path):
    """A safetensors *index* is metadata, not weights."""
    d = _make_snapshot(tmp_path, "Qwen/Qwen3-8B")
    (d / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert bootstrap.model_weights_cached("Qwen/Qwen3-8B", cache_dir=tmp_path) is False


def test_weights_present_when_safetensors(tmp_path):
    d = _make_snapshot(tmp_path, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    (d / "model.safetensors").write_bytes(b"\x00" * 4096)
    assert bootstrap.model_weights_cached("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir=tmp_path) is True


def test_weights_present_when_pytorch_bin(tmp_path):
    d = _make_snapshot(tmp_path, "gpt2")
    (d / "pytorch_model.bin").write_bytes(b"\x00" * 4096)
    assert bootstrap.model_weights_cached("gpt2", cache_dir=tmp_path) is True


def test_empty_weight_file_does_not_count(tmp_path):
    """A zero-byte placeholder shard is an incomplete download, not present."""
    d = _make_snapshot(tmp_path, "Qwen/Qwen3-8B")
    (d / "model-00001-of-00004.safetensors").write_bytes(b"")
    assert bootstrap.model_weights_cached("Qwen/Qwen3-8B", cache_dir=tmp_path) is False


# ── execute_job_command wiring: fail fast, never launch a doomed server ──────

def test_execute_job_command_fails_fast_when_weights_missing(tmp_path, monkeypatch):
    """With no weights in the (empty) HF cache, the launcher must return an
    error WITHOUT starting the server subprocess (which would hang forever)."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # empty cache → no weights

    result = bootstrap.execute_job_command(
        "python3 -m bloombee.cli.run_server Qwen/Qwen3-8B --block_indices 0:9 --port 31337",
        cwd=str(PROJECT_ROOT),
        model_id="Qwen/Qwen3-8B",
    )

    assert result["weights_missing"] is True
    assert result["exit_code"] == 2
    assert "huggingface-cli download Qwen/Qwen3-8B" in result["stderr_tail"]
