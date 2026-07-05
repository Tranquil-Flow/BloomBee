#!/usr/bin/env python3
"""Plan and verify full distributed generation proof evidence.

This harness wraps ``scripts/text_generation_parity.py`` evidence. Planning mode
emits an operator runbook; verification mode inspects a captured JSON artifact
and only allows the ``full_generation`` gate to pass when distributed generation
matches reference generation for the expected model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.one_block_proof import _shell_join
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/full_generation_proof.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.one_block_proof import _shell_join

PLAN_CLAIM_BOUNDARY = "full_generation_proof_harness_only_no_live_generation"
VERIFY_CLAIM_BOUNDARY = "verified_full_generation_evidence"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _count_new_tokens(payload: dict[str, Any]) -> int:
    max_new = payload.get("max_new_tokens")
    if isinstance(max_new, int):
        return max_new
    distributed = payload.get("distributed_ids")
    inputs = payload.get("input_ids")
    if isinstance(distributed, list) and isinstance(inputs, list) and len(distributed) >= len(inputs):
        return len(distributed) - len(inputs)
    return 0


def _valid_placement(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    layers = item.get("layers")
    return (
        isinstance(layers, list)
        and len(layers) == 2
        and all(isinstance(value, int) for value in layers)
        and layers[1] > layers[0]
        and bool(item.get("host"))
    )


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    placements = payload.get("server_placements") or payload.get("layer_placements") or []
    return {
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "model": payload.get("model"),
        "prompt": payload.get("prompt"),
        "max_new_tokens": _count_new_tokens(payload),
        "generated_ids_match": payload.get("generated_ids_match"),
        "generated_text_match": payload.get("generated_text_match"),
        "next_token_match": payload.get("next_token_match"),
        "distributed_seconds": payload.get("distributed_seconds"),
        "reference_seconds": payload.get("reference_seconds"),
        "server_count": len(payload.get("server_maddrs") or []),
        "server_placement_count": len(placements) if isinstance(placements, list) else 0,
    }


def build_full_generation_plan(
    *,
    model_id: str,
    server_maddrs: list[str],
    server_placements: list[str] | None = None,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 6,
    mode: str = "generate-api",
    evidence_path: str = ".local/full-generation-evidence.json",
    reference_device: str = "mps",
    reference_dtype: str = "float16",
    reference_mode: str = "full-model",
    checkpoint_model: str | None = None,
    reference_cache_dir: str | None = None,
    reference_local_files_only: bool = False,
    distributed_dtype: str = "float16",
) -> dict[str, Any]:
    if not server_maddrs:
        raise ValueError("at least one server multiaddr is required")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    server_placements = server_placements or []
    parts = [
        "PYTHONPATH=.:src",
        "python scripts/text_generation_parity.py",
        f"--model {model_id}",
        f"--prompt {_quote(prompt)}",
        f"--max-new-tokens {max_new_tokens}",
        f"--mode {mode}",
        f"--reference-device {reference_device}",
        f"--reference-dtype {reference_dtype}",
        f"--reference-mode {reference_mode}",
        f"--distributed-dtype {distributed_dtype}",
        f"--out {evidence_path}",
    ]
    if checkpoint_model:
        parts.append(f"--checkpoint-model {_quote(checkpoint_model)}")
    if reference_cache_dir:
        parts.append(f"--reference-cache-dir {_quote(reference_cache_dir)}")
    if reference_local_files_only:
        parts.append("--reference-local-files-only")
    parts.extend(f"--server-maddr {_quote(item)}" for item in server_maddrs)
    parts.extend(f"--server-placement {_quote(item)}" for item in server_placements)
    parity_command = _shell_join(parts)
    verify_command = _shell_join(
        [
            "python mvp_capabilities/full_generation_proof.py verify",
            f"--model {model_id}",
            f"--evidence {evidence_path}",
            f"--min-new-tokens {max_new_tokens}",
            "--require-server-placements",
        ]
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": "full_generation",
        "server_maddrs": server_maddrs,
        "server_placements": server_placements,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "mode": mode,
        "reference_mode": reference_mode,
        "checkpoint_model": checkpoint_model,
        "reference_cache_dir": reference_cache_dir,
        "reference_local_files_only": reference_local_files_only,
        "evidence_path": evidence_path,
        "parity_command": parity_command,
        "verify_command": verify_command,
        "proof_status_on_success": "full_generation: passed",
        "notes": [
            "Planning output is not proof and does not run generation.",
            "Verify mode requires exact generated token IDs and text to match reference output.",
            "Do not update PROOF_STATUS.yaml until verify mode returns status=passed.",
        ],
    }


def verify_full_generation_evidence(
    *,
    evidence_path: str | Path,
    model_id: str,
    min_new_tokens: int = 1,
    require_server_placements: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(evidence_path).expanduser().read_text(encoding="utf-8"))
    failed: list[str] = []
    if payload.get("model") != model_id:
        failed.append("evidence model mismatch")
    if payload.get("ok") is not True:
        failed.append("parity evidence ok was not true")
    if payload.get("generated_ids_match") is not True:
        failed.append("generated token IDs did not match reference")
    if payload.get("generated_text_match") is not True:
        failed.append("generated text did not match reference")
    if payload.get("next_token_match") is not True:
        failed.append("next-token check did not match reference")
    if _count_new_tokens(payload) < min_new_tokens:
        failed.append(f"expected at least {min_new_tokens} generated token(s)")
    distributed_ids = payload.get("distributed_ids")
    reference_ids = payload.get("reference_ids")
    if not isinstance(distributed_ids, list) or not distributed_ids:
        failed.append("distributed token IDs are missing")
    if not isinstance(reference_ids, list) or not reference_ids:
        failed.append("reference token IDs are missing")
    if isinstance(distributed_ids, list) and isinstance(reference_ids, list) and distributed_ids != reference_ids:
        failed.append("distributed token IDs differ from reference IDs")
    if payload.get("distributed_text") is None or payload.get("reference_text") is None:
        failed.append("distributed/reference text is missing")
    if payload.get("server_maddrs") and not isinstance(payload.get("server_maddrs"), list):
        failed.append("server_maddrs must be a list")

    placements = payload.get("server_placements") or payload.get("layer_placements") or []
    if require_server_placements and not placements:
        failed.append("server placements are required for dashboard/proof attribution")
    if placements and not all(_valid_placement(item) for item in placements):
        failed.append("server placements must include host and increasing integer layer ranges")

    status = "passed" if not failed else "failed"
    return {
        "model_id": model_id,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": "full_generation",
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {"full_generation": "passed"} if status == "passed" else {},
        "failed_checks": failed,
        "evidence_path": str(evidence_path),
        "evidence_summary": _summary(payload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit commands for a full-generation proof run")
    plan.add_argument("--model", required=True)
    plan.add_argument("--server-maddr", action="append", dest="server_maddrs", required=True)
    plan.add_argument("--server-placement", action="append", dest="server_placements", default=None)
    plan.add_argument("--prompt", default="The capital of France is")
    plan.add_argument("--max-new-tokens", type=int, default=6)
    plan.add_argument("--mode", choices=("forward-loop", "generate-api"), default="generate-api")
    plan.add_argument("--evidence", default=".local/full-generation-evidence.json")
    plan.add_argument("--reference-device", default="mps")
    plan.add_argument("--reference-dtype", default="float16")
    plan.add_argument("--reference-mode", choices=("full-model", "streamed-blocks"), default="full-model")
    plan.add_argument("--checkpoint-model", default=None)
    plan.add_argument("--reference-cache-dir", default=None)
    plan.add_argument("--reference-local-files-only", action="store_true")
    plan.add_argument("--distributed-dtype", default="float16")

    verify = sub.add_parser("verify", help="Verify captured full-generation parity evidence")
    verify.add_argument("--model", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--min-new-tokens", type=int, default=1)
    verify.add_argument("--require-server-placements", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_full_generation_plan(
            model_id=args.model,
            server_maddrs=args.server_maddrs,
            server_placements=args.server_placements,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            mode=args.mode,
            evidence_path=args.evidence,
            reference_device=args.reference_device,
            reference_dtype=args.reference_dtype,
            reference_mode=args.reference_mode,
            checkpoint_model=args.checkpoint_model,
            reference_cache_dir=args.reference_cache_dir,
            reference_local_files_only=args.reference_local_files_only,
            distributed_dtype=args.distributed_dtype,
        )
    else:
        payload = verify_full_generation_evidence(
            evidence_path=args.evidence,
            model_id=args.model,
            min_new_tokens=args.min_new_tokens,
            require_server_placements=args.require_server_placements,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
