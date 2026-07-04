#!/usr/bin/env python3
"""Render and verify a Termux tiny-model/BloomBee feasibility probe.

The probe is intentionally read-only: it checks runtime facts, installed Python
modules, basic build tools, memory/storage, and explicit blockers. It does not
install packages, download models, run inference, prove speedup, or count the
phone as a BloomBee worker.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import stat
from pathlib import Path
from typing import Any

SOURCE = "termux_tiny_model_probe.py"
SCRIPT_SOURCE = "termux_tiny_model_probe.sh"
CLAIM_BOUNDARY = "termux_tiny_model_probe_only_no_inference_proof"
ERROR_CLAIM_BOUNDARY = "termux_tiny_model_probe_error_no_inference_proof"
VERIFY_CLAIM_BOUNDARY = "termux_tiny_model_probe_verifier_only_no_inference_proof"

REQUIRED_FLAGS_FALSE = (
    "generation_proven",
    "speedup_proven",
    "inference_proven",
    "can_update_proof_status",
)


def render_termux_tiny_model_probe_script() -> str:
    """Return a pasteable Termux shell script for the feasibility probe."""
    template = '''#!/usr/bin/env sh
# BloomBee Termux tiny-model/BloomBee feasibility probe.
# claim_boundary: {CLAIM_BOUNDARY}
# Read-only: no installs, downloads, inference, speedup, or block serving.
set -u
PY_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PY_BIN" ]; then
  printf '%s\n' '{"ok":false,"source":"{SCRIPT_SOURCE}","claim_boundary":"{ERROR_CLAIM_BOUNDARY}","error":"python_missing_install_with_pkg_install_python","generation_proven":false,"speedup_proven":false,"inference_proven":false}'
  exit 2
fi
"$PY_BIN" - <<'PY'
import importlib.util
import json
import os
import platform
import shutil
import subprocess

CLAIM_BOUNDARY = "{CLAIM_BOUNDARY}"

def getprop(key):
    try:
        return subprocess.check_output(["getprop", key], text=True, stderr=subprocess.DEVNULL, timeout=1).strip() or None
    except Exception:
        return None

def which(cmd):
    path = shutil.which(cmd)
    return path or None

def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

def read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for raw in handle:
                parts = raw.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out[parts[0].rstrip(":")] = int(parts[1])
    except Exception:
        pass
    return out

def disk_payload(path):
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(usage.total / 1_000_000_000, 3),
            "free_gb": round(usage.free / 1_000_000_000, 3),
        }
    except Exception as exc:
        return {"path": path, "error": str(exc)}

prefix = os.environ.get("PREFIX")
home = os.environ.get("HOME")
is_termux = bool(prefix and prefix.startswith("/data/data/com.termux/"))
modules = {
    name: module_available(name)
    for name in [
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "safetensors",
        "huggingface_hub",
        "llama_cpp",
        "onnxruntime",
        "ml_dtypes",
        "bloombee",
    ]
}
commands = {name: which(name) for name in ["python", "python3", "pip", "pip3", "git", "clang", "cmake", "make", "pkg", "termux-info", "proot-distro"]}
meminfo = read_meminfo()
mem_available_gb = round(meminfo.get("MemAvailable", 0) / 1_000_000, 3) if meminfo else None
mem_total_gb = round(meminfo.get("MemTotal", 0) / 1_000_000, 3) if meminfo else None
transformers_ready = all(modules.get(name) for name in ["numpy", "torch", "transformers", "tokenizers", "safetensors"])
llama_cpp_import_ready = bool(modules.get("llama_cpp"))
llama_cpp_build_tools_present = bool(commands.get("clang") and commands.get("cmake") and commands.get("make") and (commands.get("pip") or commands.get("pip3")))
bloombee_ready = bool(modules.get("bloombee") and modules.get("torch") and modules.get("transformers"))
blockers = []
if not modules.get("torch"):
    blockers.append("python_torch_missing_for_transformers_or_bloombee")
if not modules.get("transformers"):
    blockers.append("transformers_missing_for_python_tiny_model")
if not modules.get("tokenizers"):
    blockers.append("tokenizers_missing_for_hf_tokenization")
if not modules.get("llama_cpp"):
    blockers.append("llama_cpp_python_missing_for_gguf_tiny_model")
if not llama_cpp_build_tools_present:
    blockers.append("llama_cpp_build_tools_incomplete")
if not modules.get("bloombee"):
    blockers.append("bloombee_python_package_missing_for_block_serving")
if mem_available_gb is not None and mem_available_gb < 1.0:
    blockers.append("available_memory_below_1gb")

payload = {
    "ok": True,
    "source": "{SCRIPT_SOURCE}",
    "claim_boundary": CLAIM_BOUNDARY,
    "phone_runtime": {
        "is_mobile": True,
        "kind": "android",
        "runtime": "termux" if is_termux else "unknown_non_termux",
        "is_termux": is_termux,
        "prefix": prefix,
        "home": home,
        "android_model": getprop("ro.product.model"),
        "android_manufacturer": getprop("ro.product.manufacturer"),
        "android_sdk": getprop("ro.build.version.sdk"),
        "soc": getprop("ro.soc.model") or getprop("ro.hardware"),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    },
    "memory": {
        "mem_total_gb": mem_total_gb,
        "mem_available_gb": mem_available_gb,
        "swap_total_gb": round(meminfo.get("SwapTotal", 0) / 1_000_000, 3) if meminfo else None,
        "swap_free_gb": round(meminfo.get("SwapFree", 0) / 1_000_000, 3) if meminfo else None,
    },
    "storage": {
        "home": disk_payload(home or "."),
        "prefix": disk_payload(prefix or "."),
    },
    "python_modules": modules,
    "commands": commands,
    "feasibility": {
        "static_draft_contract_ready": True,
        "python_transformers_tiny_model_ready": transformers_ready,
        "llama_cpp_tiny_model_import_ready": llama_cpp_import_ready,
        "llama_cpp_tiny_model_build_possible_unproven": llama_cpp_build_tools_present,
        "bloombee_block_serving_ready": bloombee_ready,
        "known_blockers": blockers,
        "recommended_next_step": "install/enable a tiny GGUF draft path if llama_cpp is available, otherwise keep phone as control-plane/static-draft until a tiny runtime is installed",
    },
    "generation_proven": False,
    "speedup_proven": False,
    "inference_proven": False,
    "can_update_proof_status": False,
}
print(json.dumps(payload, sort_keys=True))
PY
'''
    return (
        template.replace("{CLAIM_BOUNDARY}", CLAIM_BOUNDARY)
        .replace("{SCRIPT_SOURCE}", SCRIPT_SOURCE)
        .replace("{ERROR_CLAIM_BOUNDARY}", ERROR_CLAIM_BOUNDARY)
    )


def write_termux_tiny_model_probe_script(path: str | Path) -> dict[str, object]:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_termux_tiny_model_probe_script(), encoding="utf-8")
    out.chmod(out.stat().st_mode | stat.S_IXUSR)
    return {
        "source": SOURCE,
        "claim_boundary": "termux_tiny_model_probe_script_written_no_phone_run",
        "out": str(out),
        "local_command": f"sh {shlex.quote(str(out))}",
        "inference_proven": False,
        "generation_proven": False,
        "speedup_proven": False,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if "\n" in text:
        candidates = [line for line in text.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
        if len(candidates) == 1:
            text = candidates[0]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Termux tiny-model probe evidence must be a JSON object")
    return payload


def _finite_optional_nonnegative(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0)


def verify_termux_tiny_model_probe(path: str | Path, *, require_termux: bool = True) -> dict[str, object]:
    try:
        payload = _load_json(path)
    except Exception as exc:
        return {
            "source": SOURCE,
            "claim_boundary": VERIFY_CLAIM_BOUNDARY,
            "verification_status": "failed",
            "ok": False,
            "error": f"could not parse evidence: {type(exc).__name__}: {exc}",
            "inference_proven": False,
            "generation_proven": False,
            "speedup_proven": False,
            "can_update_proof_status": False,
        }

    errors: list[str] = []
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("unexpected claim_boundary")
    if payload.get("ok") is not True:
        errors.append("payload ok is not true")
    runtime = payload.get("phone_runtime") or {}
    if require_termux and not (isinstance(runtime, dict) and runtime.get("is_termux") is True):
        errors.append("evidence was not captured from Termux")
    modules = payload.get("python_modules") or {}
    commands = payload.get("commands") or {}
    feasibility = payload.get("feasibility") or {}
    memory = payload.get("memory") or {}
    if not isinstance(modules, dict) or "torch" not in modules or "transformers" not in modules:
        errors.append("python_modules must include torch and transformers keys")
    if not isinstance(commands, dict) or "pkg" not in commands:
        errors.append("commands must include pkg key")
    if feasibility.get("static_draft_contract_ready") is not True:
        errors.append("static_draft_contract_ready must be true")
    if not isinstance(feasibility.get("known_blockers"), list):
        errors.append("known_blockers must be a list")
    for key in ("mem_total_gb", "mem_available_gb", "swap_total_gb", "swap_free_gb"):
        if not _finite_optional_nonnegative(memory.get(key)):
            errors.append(f"memory.{key} must be non-negative or null")
    for flag in REQUIRED_FLAGS_FALSE:
        if payload.get(flag) is not False:
            errors.append(f"{flag} must be false")

    passed = not errors
    return {
        "source": SOURCE,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "verification_status": "passed" if passed else "failed",
        "ok": passed,
        "errors": errors,
        "evidence": payload,
        "termux_detected": bool(isinstance(runtime, dict) and runtime.get("is_termux") is True),
        "runtime_summary": runtime,
        "memory": memory,
        "python_modules": modules,
        "commands": commands,
        "feasibility": feasibility,
        "inference_proven": False,
        "generation_proven": False,
        "speedup_proven": False,
        "can_update_proof_status": False,
        "next_step": feasibility.get("recommended_next_step") or "install a real tiny-model runtime before claiming phone inference",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Write or print the Termux feasibility probe shell script")
    render.add_argument("--out", default=None, help="Optional path to write; omit to print script")
    render.add_argument("--json", action="store_true", help="When --out is used, print JSON metadata")

    verify = sub.add_parser("verify", help="Verify captured Termux feasibility JSON")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--allow-non-termux", action="store_true", help="Allow local/non-Termux evidence; tests only")

    args = parser.parse_args(argv)
    if args.command == "render":
        if args.out:
            payload = write_termux_tiny_model_probe_script(args.out)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(payload["local_command"])
        else:
            print(render_termux_tiny_model_probe_script(), end="")
        return 0
    if args.command == "verify":
        report = verify_termux_tiny_model_probe(args.evidence, require_termux=not args.allow_non_termux)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
