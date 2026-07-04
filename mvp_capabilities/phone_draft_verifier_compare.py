#!/usr/bin/env python3
"""Compare phone GGUF draft text against an authoritative verifier text.

The comparison is exact UTF-8 byte-prefix acceptance. That makes this dependency
free and audit-safe, but it is intentionally *not* a model-tokenizer proof unless
a later artifact supplies matching tokenizer IDs. It also does not prove speedup:
it only records proposed/accepted/rejected byte counts and keeps the verifier
text authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.draft_provider import evaluate_draft_against_verifier
except ModuleNotFoundError:  # Allow `python mvp_capabilities/phone_draft_verifier_compare.py ...`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.draft_provider import evaluate_draft_against_verifier

CLAIM_BOUNDARY = "phone_gguf_draft_verifier_text_prefix_comparison_no_tokenizer_no_speedup_claim"
SOURCE = "phone_draft_verifier_compare.py"


def _bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def compare_phone_draft_to_verifier(
    bridge_payload: dict[str, Any],
    *,
    verifier_text: str,
    verifier_source: str,
    verifier_kind: str = "operator_supplied_text",
) -> dict[str, Any]:
    """Compare phone bridge generated text against authoritative verifier text."""
    evidence = bridge_payload.get("evidence") if "evidence" in bridge_payload else bridge_payload
    response = evidence.get("draft_response") or {}
    draft_text = str(response.get("generated_text") or "")
    draft_bytes = _bytes(draft_text)
    verifier_bytes = _bytes(verifier_text)
    verdict = evaluate_draft_against_verifier(draft_bytes, verifier_bytes)
    acceptance_rate = verdict["acceptance_rate"]
    accepted_text = bytes(verdict["accepted_tokens"]).decode("utf-8", errors="replace")
    rejected_text = bytes(verdict["rejected_tokens"]).decode("utf-8", errors="replace")
    committed_text = bytes(verdict["committed_tokens"]).decode("utf-8", errors="replace")
    live_model = verifier_kind == "live_model_generation" or verifier_kind.endswith("_live_model_generation")
    comparison_valid = bool(draft_text) and bool(verifier_text) and evidence.get("generation_proven") is True
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": "passed" if comparison_valid else "failed",
        "phone_bridge_source": bridge_payload.get("source") or evidence.get("source"),
        "phone_runtime": evidence.get("phone_runtime"),
        "model": evidence.get("model"),
        "draft": {
            "prompt": (evidence.get("draft_request") or {}).get("prompt"),
            "generated_text": draft_text,
            "byte_tokens": draft_bytes,
            "byte_token_count": len(draft_bytes),
            "elapsed_s": response.get("elapsed_s"),
        },
        "verifier": {
            "source": verifier_source,
            "kind": verifier_kind,
            "authoritative": True,
            "generated_text": verifier_text,
            "byte_tokens": verifier_bytes,
            "byte_token_count": len(verifier_bytes),
            "live_model_generation_proven": live_model,
        },
        "verdict": {
            **verdict,
            "accepted_text": accepted_text,
            "rejected_text": rejected_text,
            "committed_text": committed_text,
            "token_unit": "utf8_byte",
        },
        "dashboard_counters": {
            "proposed": verdict["proposed_count"],
            "accepted": verdict["accepted_count"],
            "rejected": verdict["rejected_count"],
            "acceptance_rate": acceptance_rate,
        },
        "comparison_proven": comparison_valid,
        "verifier_acceptance_proven": comparison_valid and verdict["accepted_count"] > 0,
        "full_draft_accepted": comparison_valid and verdict["accepted_count"] == verdict["proposed_count"] and verdict["proposed_count"] > 0,
        "tokenizer_match_proven": False,
        "generation_proven": evidence.get("generation_proven") is True,
        "speedup_proven": False,
        "bloombee_block_serving_proven": False,
        "can_update_speculative_speedup_status": False,
        "can_update_bloombee_block_worker_status": False,
        "operator_next_steps": [
            "run the authoritative verifier with matching tokenizer IDs if available",
            "measure verifier-only wall clock vs phone-draft-plus-verifier wall clock before speedup claims",
            "keep phones out of BloomBee block-worker accounting until block serving is separately proven",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-evidence", required=True)
    parser.add_argument("--verifier-text", default=None)
    parser.add_argument("--verifier-text-file", default=None)
    parser.add_argument("--verifier-source", required=True)
    parser.add_argument("--verifier-kind", default="operator_supplied_text")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.verifier_text_file:
        verifier_text = Path(args.verifier_text_file).read_text(encoding="utf-8")
    elif args.verifier_text is not None:
        verifier_text = args.verifier_text
    else:
        raise SystemExit("one of --verifier-text or --verifier-text-file is required")
    bridge = json.loads(Path(args.bridge_evidence).read_text(encoding="utf-8"))
    payload = compare_phone_draft_to_verifier(
        bridge,
        verifier_text=verifier_text,
        verifier_source=args.verifier_source,
        verifier_kind=args.verifier_kind,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
