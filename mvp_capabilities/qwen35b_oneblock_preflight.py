#!/usr/bin/env python3
"""Fail-closed host preflight for the Qwen35B one-block proof gate.

This tool does not run inference and can never update proof status by itself. It
only answers whether the current or provided host memory is sufficient to attempt
``one_block_proof.py`` for Qwen/Qwen-AgentWorld-35B-A3B.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.one_block_proof import build_one_block_plan
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/qwen35b_oneblock_preflight.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.one_block_proof import build_one_block_plan
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, load_registry

MODEL_ID = "Qwen/Qwen-AgentWorld-35B-A3B"
CLAIM_BOUNDARY = "qwen35b_one_block_host_preflight_no_live_inference"


def _find_model(model_id: str, registry: list[dict[str, Any]]) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise ValueError(f"model {model_id!r} not found in registry")


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def detect_total_mem_gb() -> float | None:
    """Best-effort total memory detection without third-party dependencies."""

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip()
            return _bytes_to_gb(int(out))
        except Exception:  # noqa: BLE001 - fail closed below
            return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return _bytes_to_gb(int(page_size) * int(pages))
    except Exception:  # noqa: BLE001 - platform may not expose sysconf keys
        return None


def detect_free_mem_gb() -> float | None:
    """Best-effort available memory detection; returns None when unavailable."""

    try:
        import psutil  # type: ignore[import-not-found]

        return _bytes_to_gb(int(psutil.virtual_memory().available))
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


def build_qwen35b_oneblock_preflight(
    *,
    registry: list[dict[str, Any]] | None = None,
    model_id: str = MODEL_ID,
    host_total_mem_gb: float | None = None,
    host_free_mem_gb: float | None = None,
    host_label: str | None = None,
    block_range: str = "0:1",
    device: str = "mps",
    dtype: str = "float16",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    registry = registry if registry is not None else load_registry(DEFAULT_REGISTRY)
    model = _find_model(model_id, registry)
    candidate_branch = str(model.get("candidate_branch") or "qwen35b").replace("-", "_")
    claim_boundary = (
        CLAIM_BOUNDARY
        if candidate_branch == "qwen35b"
        else f"{candidate_branch}_one_block_host_preflight_no_live_inference"
    )
    required_free_gb = float(model.get("recommended_min_free_mem_gb") or 0.0)
    total_mem_gb = detect_total_mem_gb() if host_total_mem_gb is None else float(host_total_mem_gb)
    free_mem_gb = detect_free_mem_gb() if host_free_mem_gb is None else float(host_free_mem_gb)
    host_label = host_label or platform.node() or "unknown-host"
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()

    remaining_blockers: list[str] = []
    if required_free_gb <= 0:
        remaining_blockers.append("registry_missing_recommended_min_free_mem_gb")
    if total_mem_gb is None:
        remaining_blockers.append("host_total_memory_unknown")
    elif required_free_gb > 0 and total_mem_gb < required_free_gb:
        remaining_blockers.append(f"insufficient_host_memory_for_{candidate_branch}_one_block")
    elif free_mem_gb is None:
        remaining_blockers.append("host_free_memory_unknown")
    elif required_free_gb > 0 and free_mem_gb < required_free_gb:
        remaining_blockers.append(f"insufficient_free_memory_for_{candidate_branch}_one_block")

    ready = not remaining_blockers
    status = "ready-to-attempt" if ready else "blocked-by-host-memory"
    plan = build_one_block_plan(
        model_id,
        registry=registry,
        block_range=block_range,
        device=device,
        dtype=dtype,
        server_log=f".local/qwen35b-oneblock-{host_label}-server.log",
        client_log=f".local/qwen35b-oneblock-{host_label}-client.log",
    )

    return {
        "model_id": model_id,
        "claim_boundary": claim_boundary,
        "generated_at_utc": generated_at_utc,
        "host_label": host_label,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "proof_gate": "one_block_server",
        "status": status,
        "registry_recommended_min_free_mem_gb": required_free_gb,
        "observed_total_mem_gb": total_mem_gb,
        "observed_free_mem_gb": free_mem_gb,
        "ready_to_attempt_live_oneblock": ready,
        "live_run_attempted": False,
        "one_block_server_proven": False,
        "can_update_proof_status": False,
        "proof_status_update": {},
        "remaining_blockers": remaining_blockers,
        "one_block_plan": plan,
        "next_step": (
            "run the one_block_plan server/client/verify commands on this host"
            if ready
            else "rerun this preflight on a host with enough free memory before attempting live one-block proof"
        ),
        "do_not_claim": [
            "no live inference was attempted",
            "no one-block server proof",
            "no route/demo promotion",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--host-total-mem-gb", type=float, default=None)
    parser.add_argument("--host-free-mem-gb", type=float, default=None)
    parser.add_argument("--host-label", default=None)
    parser.add_argument("--block-range", default="0:1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_qwen35b_oneblock_preflight(
        registry=load_registry(args.registry),
        model_id=args.model,
        host_total_mem_gb=args.host_total_mem_gb,
        host_free_mem_gb=args.host_free_mem_gb,
        host_label=args.host_label,
        block_range=args.block_range,
        device=args.device,
        dtype=args.dtype,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
