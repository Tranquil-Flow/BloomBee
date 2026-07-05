#!/usr/bin/env python3
"""Plan and verify cached distributed generation proof evidence.

`full_generation_proof.py` verifies generated-output parity in general. This
harness narrows the gate to BloomBee's cached generation path by requiring
`text_generation_parity.py --mode generate-api` evidence before allowing the
`cache_generation` proof gate to pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.full_generation_proof import (
        build_full_generation_plan,
        verify_full_generation_evidence,
    )
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/cache_generation_proof.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.full_generation_proof import (
        build_full_generation_plan,
        verify_full_generation_evidence,
    )

PLAN_CLAIM_BOUNDARY = "cache_generation_proof_harness_only_no_live_generation"
VERIFY_CLAIM_BOUNDARY = "verified_cache_generation_evidence"


def build_cache_generation_plan(
    *,
    model_id: str,
    server_maddrs: list[str],
    server_placements: list[str] | None = None,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 6,
    evidence_path: str = ".local/cache-generation-evidence.json",
    reference_device: str = "mps",
    reference_dtype: str = "float16",
    distributed_dtype: str = "float16",
    reference_mode: str = "full-model",
    checkpoint_model: str | None = None,
    reference_cache_dir: str | None = None,
    reference_local_files_only: bool = False,
) -> dict[str, Any]:
    plan = build_full_generation_plan(
        model_id=model_id,
        server_maddrs=server_maddrs,
        server_placements=server_placements,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        mode="generate-api",
        evidence_path=evidence_path,
        reference_device=reference_device,
        reference_dtype=reference_dtype,
        distributed_dtype=distributed_dtype,
        reference_mode=reference_mode,
        checkpoint_model=checkpoint_model,
        reference_cache_dir=reference_cache_dir,
        reference_local_files_only=reference_local_files_only,
    )
    plan.update(
        {
            "claim_boundary": PLAN_CLAIM_BOUNDARY,
            "proof_gate": "cache_generation",
            "proof_status_on_success": "cache_generation: passed",
            "verify_command": plan["verify_command"].replace(
                "mvp_capabilities/full_generation_proof.py verify",
                "mvp_capabilities/cache_generation_proof.py verify",
            ),
            "notes": [
                "Planning output is not proof and does not run cached generation.",
                "Verify mode requires text_generation_parity.py evidence captured with --mode generate-api.",
                "Do not update PROOF_STATUS.yaml until verify mode returns status=passed.",
            ],
        }
    )
    return plan


def _summary_mode(payload: dict[str, Any]) -> str | None:
    mode = payload.get("mode")
    return mode if isinstance(mode, str) else None


def verify_cache_generation_evidence(
    *,
    evidence_path: str | Path,
    model_id: str,
    min_new_tokens: int = 1,
    require_server_placements: bool = True,
) -> dict[str, Any]:
    path = Path(evidence_path).expanduser()
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    result = verify_full_generation_evidence(
        evidence_path=path,
        model_id=model_id,
        min_new_tokens=min_new_tokens,
        require_server_placements=require_server_placements,
    )
    failed = list(result.get("failed_checks") or [])
    if _summary_mode(payload) != "generate-api":
        failed.append("cache_generation requires mode=generate-api evidence")
    if payload.get("distributed_steps") not in (None, []):
        failed.append("cache_generation evidence should use cached generate API, not forward-loop distributed steps")
    if payload.get("reference_steps") not in (None, []):
        if payload.get("reference_mode") != "streamed-blocks":
            failed.append("cache_generation full-model reference evidence should use cached generate API, not forward-loop reference steps")
        elif payload.get("reference_generation_path") != "streamed-forward-loop-correctness-fallback":
            failed.append("streamed cache_generation reference steps must declare streamed-forward-loop-correctness-fallback")

    status = "passed" if not failed else "failed"
    return {
        "model_id": model_id,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": "cache_generation",
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {"cache_generation": "passed"} if status == "passed" else {},
        "failed_checks": failed,
        "evidence_path": str(path),
        "evidence_summary": result.get("evidence_summary") or {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit commands for a cache-generation proof run")
    plan.add_argument("--model", required=True)
    plan.add_argument("--server-maddr", action="append", dest="server_maddrs", required=True)
    plan.add_argument("--server-placement", action="append", dest="server_placements", default=None)
    plan.add_argument("--prompt", default="The capital of France is")
    plan.add_argument("--max-new-tokens", type=int, default=6)
    plan.add_argument("--evidence", default=".local/cache-generation-evidence.json")
    plan.add_argument("--reference-device", default="mps")
    plan.add_argument("--reference-dtype", default="float16")
    plan.add_argument("--reference-mode", choices=("full-model", "streamed-blocks"), default="full-model")
    plan.add_argument("--checkpoint-model", default=None)
    plan.add_argument("--reference-cache-dir", default=None)
    plan.add_argument("--reference-local-files-only", action="store_true")
    plan.add_argument("--distributed-dtype", default="float16")

    verify = sub.add_parser("verify", help="Verify captured cache-generation parity evidence")
    verify.add_argument("--model", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--min-new-tokens", type=int, default=1)
    verify.add_argument("--require-server-placements", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_cache_generation_plan(
            model_id=args.model,
            server_maddrs=args.server_maddrs,
            server_placements=args.server_placements,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            evidence_path=args.evidence,
            reference_device=args.reference_device,
            reference_dtype=args.reference_dtype,
            distributed_dtype=args.distributed_dtype,
            reference_mode=args.reference_mode,
            checkpoint_model=args.checkpoint_model,
            reference_cache_dir=args.reference_cache_dir,
            reference_local_files_only=args.reference_local_files_only,
        )
    else:
        payload = verify_cache_generation_evidence(
            evidence_path=args.evidence,
            model_id=args.model,
            min_new_tokens=args.min_new_tokens,
            require_server_placements=args.require_server_placements,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
