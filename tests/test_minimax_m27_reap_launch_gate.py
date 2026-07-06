import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = "m51Lab-MiniMax-M2.7-REAP-139B-A10B.i1-IQ2_XXS.gguf"


def _vm_stat(*, free: int, inactive: int, speculative: int, purgeable: int, page_size: int = 16384) -> str:
    return f"""Mach Virtual Memory Statistics: (page size of {page_size} bytes)
Pages free: {free}.
Pages active: 1.
Pages inactive: {inactive}.
Pages speculative: {speculative}.
Pages wired down: 1.
Pages purgeable: {purgeable}.
"""


def test_m27_launch_gate_fails_closed_for_partial_download(tmp_path: Path):
    from mvp_capabilities.minimax_m27_reap_launch_gate import build_minimax_m27_reap_launch_report

    model_dir = tmp_path / "models"
    partial = model_dir / ".cache" / "huggingface" / "download" / f"{MODEL_FILE}.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    report = build_minimax_m27_reap_launch_report(
        model_dir=model_dir,
        llama_cli_path="/opt/homebrew/bin/llama-cli",
        llama_server_path="/opt/homebrew/bin/llama-server",
        vm_stat_text=_vm_stat(free=3_000_000, inactive=3_000_000, speculative=0, purgeable=0),
        expected_min_bytes=100,
        required_free_gb=1.0,
    )

    assert report["claim_boundary"] == "minimax_m27_reap_launch_readiness_no_inference_proof"
    assert report["download"]["completed"] is False
    assert report["download"]["partial_bytes"] == len(b"partial")
    assert report["launch_ready"] is False
    assert report["external_runtime_smoke_proven"] is False
    assert "gguf_file_missing_or_too_small" in report["blocked_reasons"]
    assert "download_incomplete_files_present" in report["blocked_reasons"]


def test_m27_launch_gate_ready_when_file_runtime_and_memory_are_present(tmp_path: Path):
    from mvp_capabilities.minimax_m27_reap_launch_gate import build_minimax_m27_reap_launch_report

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / MODEL_FILE
    model_path.write_bytes(b"x" * 128)

    report = build_minimax_m27_reap_launch_report(
        model_dir=model_dir,
        llama_cli_path="/opt/homebrew/bin/llama-cli",
        llama_server_path="/opt/homebrew/bin/llama-server",
        vm_stat_text=_vm_stat(free=3_000_000, inactive=3_000_000, speculative=0, purgeable=0),
        expected_min_bytes=100,
        required_free_gb=1.0,
    )

    assert report["download"]["completed"] is True
    assert report["download"]["gguf_path"].endswith(MODEL_FILE)
    assert report["runtime"]["llama_cli_present"] is True
    assert report["runtime"]["llama_server_present"] is True
    assert report["memory"]["attemptable"] is True
    assert report["launch_ready"] is True
    assert report["external_runtime_smoke_proven"] is False
    assert report["can_update_demo_status"] is False
    assert "llama-cli" in report["smoke_command"][0]
    assert str(model_path) in " ".join(report["smoke_command"])


def test_m27_launch_gate_fails_closed_when_memory_is_below_gate(tmp_path: Path):
    from mvp_capabilities.minimax_m27_reap_launch_gate import build_minimax_m27_reap_launch_report

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / MODEL_FILE).write_bytes(b"x" * 128)

    report = build_minimax_m27_reap_launch_report(
        model_dir=model_dir,
        llama_cli_path="/opt/homebrew/bin/llama-cli",
        llama_server_path="/opt/homebrew/bin/llama-server",
        vm_stat_text=_vm_stat(free=1, inactive=1, speculative=0, purgeable=0),
        expected_min_bytes=100,
        required_free_gb=40.8,
    )

    assert report["download"]["completed"] is True
    assert report["memory"]["attemptable"] is False
    assert report["launch_ready"] is False
    assert "freeable_memory_below_required" in report["blocked_reasons"]


def test_m27_launch_gate_cli_writes_json(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / MODEL_FILE).write_bytes(b"x" * 128)
    out_path = tmp_path / "m27-launch.json"
    vm_path = tmp_path / "vm_stat.txt"
    vm_path.write_text(_vm_stat(free=3_000_000, inactive=3_000_000, speculative=0, purgeable=0), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/minimax_m27_reap_launch_gate.py",
            "--model-dir",
            str(model_dir),
            "--llama-cli-path",
            "/opt/homebrew/bin/llama-cli",
            "--llama-server-path",
            "/opt/homebrew/bin/llama-server",
            "--vm-stat-file",
            str(vm_path),
            "--expected-min-bytes",
            "100",
            "--required-free-gb",
            "1.0",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["launch_ready"] is True
