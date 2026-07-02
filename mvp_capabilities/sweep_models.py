#!/usr/bin/env python3
"""Plan or execute per-peer throughput sweeps for the distributed-inference MVP.

Tests use dry-run mode so we can validate routing/selection without expensive
model downloads. Real execution shells out to ``bench_throughput.py`` and stores
JSONL-compatible results for the route picker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path(__file__).with_name("MODEL_REGISTRY.yaml")
DEFAULT_OUT_DIR = Path.home() / ".bloombee" / "benchmarks"


def _free_gb(peer: dict[str, Any]) -> float:
    memory = peer.get("memory") or {}
    try:
        return float(memory.get("free_gb") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _required_gb(model: dict[str, Any]) -> float:
    try:
        return float(model.get("recommended_min_free_mem_gb") or model.get("min_total_mem_gb") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_peer(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    return (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("models") or []


def build_bench_command(model_id: str, *, device: str = "auto", dtype: str = "auto", max_new_tokens: int = 32, prefill: int = 64) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("bench_throughput.py")),
        "--model",
        model_id,
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--prefill",
        str(prefill),
    ]


def build_sweep_plan(
    peer_path: str | Path,
    registry: list[dict[str, Any]],
    *,
    max_models: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    peer = load_peer(peer_path)
    free_gb = _free_gb(peer)
    feasible = [model for model in registry if _required_gb(model) <= free_gb]
    feasible.sort(key=lambda model: (_required_gb(model), model.get("model_id") or ""))
    if max_models is not None:
        feasible = feasible[:max_models]
    return {
        "peer": peer,
        "free_gb": free_gb,
        "models": [
            {
                "model_id": model["model_id"],
                "required_free_gb": _required_gb(model),
                "command": build_bench_command(model["model_id"], device=device, dtype=dtype),
            }
            for model in feasible
        ],
    }


def parse_bench_output(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def execute_plan(plan: dict[str, Any], *, offline: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {"peer": plan["peer"], "models": {}}
    env = os.environ.copy()
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    for item in plan["models"]:
        proc = subprocess.run(item["command"], text=True, capture_output=True, env=env, timeout=1800)
        parsed = parse_bench_output(proc.stdout + "\n" + proc.stderr)
        results["models"][item["model_id"]] = {
            "exit_code": proc.returncode,
            "result": parsed,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", required=True, help="Peer JSON from peer_scan.py")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands")
    parser.add_argument("--execute", action="store_true", help="Run bench commands")
    parser.add_argument("--offline", action="store_true", help="Set HF offline env for cached models")
    args = parser.parse_args(argv)

    plan = build_sweep_plan(args.peer, load_registry(args.registry), max_models=args.max_models, device=args.device, dtype=args.dtype)
    if args.execute and not args.dry_run:
        output = execute_plan(plan, offline=args.offline)
    else:
        output = {"dry_run": True, **plan}

    text = json.dumps(output, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
