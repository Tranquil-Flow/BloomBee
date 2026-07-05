#!/usr/bin/env python3
"""Audit the BloomBee MVP proof ladder for prepared models.

This module intentionally does not run inference. It turns the proof-status
registry into an ordered, human/auditor-readable report so operators can see the
next gate before a model may be promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.model_compat_scan import (
        DEFAULT_PROOF_STATUS,
        DEMO_SAFE_GATES,
        PROOF_KEYS,
        TOKEN_PARITY_KEY,
        is_demo_safe,
        load_proof_status,
        split_route_id,
    )
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/proof_ladder.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import (
        DEFAULT_PROOF_STATUS,
        DEMO_SAFE_GATES,
        PROOF_KEYS,
        TOKEN_PARITY_KEY,
        is_demo_safe,
        load_proof_status,
        split_route_id,
    )

CLAIM_BOUNDARY = "proof_ladder_audit_only_no_inference_proof"
FALLBACK_LADDER = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B@int8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
)

GATE_DESCRIPTIONS = {
    "prescan": "Config can be read and mapped to a known or blocked architecture family.",
    "one_block_server": "One real BloomBee block server can load weights and return finite RPC output.",
    "multi_block": "Multiple block servers compose over direct RPC with finite hidden states.",
    "full_generation": "Distributed text generation succeeds end-to-end with the target checkpoint.",
    "cache_generation": "Cached generation path matches the direct correctness fallback.",
    "multi_request_load": "Multiple requests route through healthy chains with visible latency/throughput.",
}


def _status_is_blocked(status: object) -> bool:
    return str(status).lower().startswith("blocked")


def _default_status() -> dict[str, str]:
    return {gate: "pending" for gate in PROOF_KEYS}


# DEMO_SAFE_GATES is imported from model_compat_scan — the single policy home
# shared with route_picker. Quantized rows (model_id@int8 / @nf4) additionally
# require token_parity: exact; is_demo_safe encodes both rules.


def _claim_level(status: dict[str, str], *, quant_type: str | None = None) -> str:
    if any(_status_is_blocked(value) for value in status.values()):
        return "blocked"
    if is_demo_safe(status, quant_type=quant_type):
        return "demo_safe"
    return "experimental"


def build_proof_ladder(
    model_id: str,
    *,
    proof_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Return ordered proof-gate state for one model.

    The report is audit-only. Even when gates are `passed`, this function does
    not itself prove inference; it only reflects the registry supplied to it.
    """
    merged = _default_status()
    if proof_status and model_id in proof_status:
        merged.update({str(key): str(value) for key, value in proof_status[model_id].items()})

    base_model_id, quant_type = split_route_id(model_id)
    gates = [
        {
            "name": gate,
            "status": merged[gate],
            "description": GATE_DESCRIPTIONS.get(gate, "Proof gate."),
            "passed": merged[gate] == "passed",
            "blocked": _status_is_blocked(merged[gate]),
        }
        for gate in PROOF_KEYS
    ]
    next_gate = next((gate["name"] for gate in gates if gate["status"] != "passed"), None)
    claim_level = _claim_level(merged, quant_type=quant_type)
    token_parity = merged.get(TOKEN_PARITY_KEY) if quant_type else None
    if quant_type and next_gate is None and token_parity != "exact":
        # all gates passed but the parity fact is missing/diverged: that IS
        # the next thing standing between this row and demo_safe
        next_gate = TOKEN_PARITY_KEY

    return {
        "model_id": model_id,
        "base_model_id": base_model_id,
        "quant_type": quant_type,
        "token_parity": token_parity,
        "claim_boundary": CLAIM_BOUNDARY,
        "claim_level": claim_level,
        "safe_demo_selectable": claim_level == "demo_safe",
        "next_gate": next_gate,
        "proof_status": merged,
        "gates": gates,
    }


def build_ladder_report(
    model_ids: list[str] | tuple[str, ...],
    *,
    proof_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "models": [build_proof_ladder(model_id, proof_status=proof_status) for model_id in model_ids],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-status", default=str(DEFAULT_PROOF_STATUS))
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--fallback-ladder",
        action="store_true",
        help="Report the prepared fallback ladder: TinyLlama, Qwen3-8B, Qwen3-14B, Qwen3-30B family.",
    )
    args = parser.parse_args(argv)

    proof = load_proof_status(args.proof_status)
    model_ids = list(args.model)
    if args.fallback_ladder:
        model_ids.extend(model_id for model_id in FALLBACK_LADDER if model_id not in model_ids)
    if not model_ids:
        parser.error("provide --model MODEL_ID or --fallback-ladder")

    if len(model_ids) == 1 and not args.fallback_ladder:
        payload = build_proof_ladder(model_ids[0], proof_status=proof)
    else:
        payload = build_ladder_report(model_ids, proof_status=proof)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
