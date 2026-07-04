#!/usr/bin/env python3
"""Build a guarded Termux tiny-GGUF draft-runtime plan from probe evidence.

This is a planning artifact only. It never installs packages, downloads models, or
runs inference. The goal is to make the next side-effecting phone step explicit:
which gates pass, which blockers remain, and which commands would be run if the
operator chooses to install a tiny GGUF/llama.cpp draft runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE = "termux_gguf_runtime_plan.py"
CLAIM_BOUNDARY = "termux_gguf_runtime_plan_only_no_install_no_inference_proof"

MIN_STORAGE_FREE_GB = 2.0
MIN_MEMORY_AVAILABLE_GB = 1.0


def _unwrap_probe(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept either a raw probe payload or verifier report with `evidence`."""
    evidence = payload.get("evidence")
    if isinstance(evidence, dict) and evidence.get("claim_boundary") == "termux_tiny_model_probe_only_no_inference_proof":
        return evidence
    return payload


def _command_present(commands: dict[str, Any], name: str) -> bool:
    value = commands.get(name)
    return isinstance(value, str) and bool(value)


def build_termux_gguf_runtime_plan(probe_payload: dict[str, Any]) -> dict[str, Any]:
    probe = _unwrap_probe(probe_payload)
    runtime = probe.get("phone_runtime") or probe_payload.get("runtime_summary") or {}
    memory = probe.get("memory") or probe_payload.get("memory") or {}
    storage = probe.get("storage") or probe_payload.get("storage") or {}
    modules = probe.get("python_modules") or probe_payload.get("python_modules") or {}
    commands = probe.get("commands") or probe_payload.get("commands") or {}
    feasibility = probe.get("feasibility") or probe_payload.get("feasibility") or {}

    storage_free_gb = None
    home_storage = storage.get("home") if isinstance(storage, dict) else None
    if isinstance(home_storage, dict) and isinstance(home_storage.get("free_gb"), (int, float)):
        storage_free_gb = float(home_storage["free_gb"])
    mem_available_gb = memory.get("mem_available_gb") if isinstance(memory, dict) else None
    if isinstance(mem_available_gb, (int, float)):
        mem_available_gb = float(mem_available_gb)
    else:
        mem_available_gb = None

    gate_checks = {
        "termux_detected": bool(isinstance(runtime, dict) and runtime.get("is_termux") is True),
        "python_available": _command_present(commands, "python") or _command_present(commands, "python3"),
        "pip_available": _command_present(commands, "pip") or _command_present(commands, "pip3"),
        "pkg_available": _command_present(commands, "pkg"),
        "build_tools_present": all(_command_present(commands, name) for name in ("clang", "cmake", "make", "git")),
        "storage_free_gb_min_2": storage_free_gb is not None and storage_free_gb >= MIN_STORAGE_FREE_GB,
        "memory_available_gb_min_1": mem_available_gb is not None and mem_available_gb >= MIN_MEMORY_AVAILABLE_GB,
        "llama_cpp_not_already_installed": modules.get("llama_cpp") is not True,
        "bloombee_block_serving_not_ready": feasibility.get("bloombee_block_serving_ready") is not True,
    }
    ready_for_guarded_install_attempt = all(
        gate_checks[key]
        for key in (
            "termux_detected",
            "python_available",
            "pip_available",
            "pkg_available",
            "build_tools_present",
            "storage_free_gb_min_2",
            "memory_available_gb_min_1",
        )
    )

    blockers = []
    if not gate_checks["termux_detected"]:
        blockers.append("not_verified_termux")
    if not gate_checks["build_tools_present"]:
        blockers.append("missing_clang_cmake_make_or_git")
    if not gate_checks["pip_available"]:
        blockers.append("pip_missing")
    if storage_free_gb is None or storage_free_gb < MIN_STORAGE_FREE_GB:
        blockers.append("insufficient_storage_for_tiny_gguf_runtime")
    if mem_available_gb is None or mem_available_gb < MIN_MEMORY_AVAILABLE_GB:
        blockers.append("insufficient_available_memory_for_tiny_runtime_smoke")

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "ready_for_guarded_install_attempt": ready_for_guarded_install_attempt,
        "gate_checks": gate_checks,
        "remaining_blockers": blockers,
        "observed_runtime": runtime,
        "observed_memory": memory,
        "observed_storage": storage,
        "observed_missing_modules": [
            name
            for name in ("torch", "transformers", "tokenizers", "llama_cpp", "bloombee")
            if modules.get(name) is False
        ],
        "recommended_path": "tiny_gguf_llama_cpp_draft_runtime" if ready_for_guarded_install_attempt else "control_plane_or_static_draft_until_runtime_ready",
        "why_not_bloombee_block_serving": [
            "BloomBee block serving on the phone still requires torch + transformers + bloombee runtime support",
            "Termux probe showed those Python modules are missing",
            "Static contract and GGUF draft-provider paths do not prove transformer-block serving",
        ],
        "guarded_commands_not_executed": [
            "pkg update",
            "pkg install -y python clang cmake make git",
            "python -m pip install --upgrade pip setuptools wheel",
            "CMAKE_ARGS='-DLLAMA_NATIVE=OFF' python -m pip install --no-cache-dir llama-cpp-python",
            "python - <<'PY'\nfrom llama_cpp import Llama\nprint('llama_cpp_import_ok')\nPY",
            "# Only after operator supplies a tiny GGUF file: run a one-token local draft smoke",
        ],
        "operator_warnings": [
            "These commands modify the phone and may take time/battery/storage; do not run without explicit approval",
            "Do not claim speedup until a real tiny model proposes tokens and an authoritative verifier accepts them",
            "Do not count phone as a BloomBee block worker without separate BloomBee block-serving proof",
        ],
        "install_executed": False,
        "download_executed": False,
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="Termux tiny-model probe evidence or verifier report JSON")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.probe).expanduser().read_text(encoding="utf-8"))
    print(json.dumps(build_termux_gguf_runtime_plan(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
