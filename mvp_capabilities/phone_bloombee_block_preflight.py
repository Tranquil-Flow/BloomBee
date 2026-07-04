#!/usr/bin/env python3
"""Fail-closed phone BloomBee block-serving preflight.

This consumes phone runtime/probe evidence and decides whether the phone is ready
to run BloomBee's Python/Hivemind block server. It deliberately treats Termux
llama.cpp GGUF generation as separate draft evidence, not block serving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "phone_bloombee_block_serving_preflight_no_block_execution_no_speedup_claim"
REQUIRED_MODULES = ("torch", "transformers", "bloombee")
OPTIONAL_BUT_EXPECTED_MODULES = ("hivemind",)


def _module_missing_reason(module: str) -> str:
    if module == "bloombee":
        return "bloombee_python_package_missing_for_block_serving"
    return f"{module}_missing_for_bloombee_block_serving"


def build_phone_bloombee_block_serving_preflight(probe: dict[str, Any]) -> dict[str, Any]:
    modules = dict(probe.get("python_modules") or {})
    feasibility = dict(probe.get("feasibility") or {})
    known_blockers = [str(item) for item in feasibility.get("known_blockers") or []]

    blocking: list[str] = []
    for module in REQUIRED_MODULES:
        if modules.get(module) is not True:
            blocking.append(_module_missing_reason(module))
    for blocker in known_blockers:
        if blocker not in blocking:
            blocking.append(blocker)

    ready = not blocking
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "source_probe_claim_boundary": probe.get("claim_boundary"),
        "runtime_summary": probe.get("runtime_summary") or probe.get("phone_runtime") or {},
        "memory": probe.get("memory") or {},
        "python_modules": modules,
        "required_modules": list(REQUIRED_MODULES),
        "optional_expected_modules": list(OPTIONAL_BUT_EXPECTED_MODULES),
        "blocking_reasons": blocking,
        "bloombee_block_serving_ready": ready,
        "bloombee_block_serving_proven": False,
        "phone_block_worker_proven": False,
        "inference_proven": False,
        "speedup_proven": False,
        "can_update_proof_status": False,
        "proof_status_update": {},
        "gguf_draft_path_is_not_block_serving": True,
        "recommended_next_step": (
            "install/prove the BloomBee Python stack on phone or use a proot/Linux environment, "
            "then run one_block_proof.py verify with real server/client logs"
            if not ready
            else "run one_block_proof.py plan/verify with real phone server and direct-client logs"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    report = build_phone_bloombee_block_serving_preflight(probe)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ready": report["bloombee_block_serving_ready"],
                "proven": report["bloombee_block_serving_proven"],
                "blocking_reasons": report["blocking_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
