#!/usr/bin/env python3
"""Plan speculative decoding roles without running generation.

This module is intentionally a planning/contract artifact. It describes how a
small draft provider could propose tokens while a BloomBee verifier remains the
authoritative source of accepted output. It does not start servers, send traffic,
or prove generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "speculative_decode_plan_only_no_generation_proof"
SOURCE = "speculative_decode_plan.py"
DEFAULT_DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _picked_route(verifier_route: dict[str, Any]) -> dict[str, Any]:
    picked = verifier_route.get("picked") if isinstance(verifier_route, dict) else None
    if isinstance(picked, dict):
        return picked
    return verifier_route if isinstance(verifier_route, dict) else {}


def _is_phone(peer: dict[str, Any]) -> bool:
    mobile = peer.get("mobile") or {}
    if isinstance(mobile, dict) and mobile.get("is_mobile") is True:
        return True
    runtime = str((mobile or {}).get("runtime") or peer.get("runtime") or "").lower()
    device = str((peer.get("accelerator") or {}).get("device") or "").lower()
    return runtime in {"termux", "android", "ios", "mobile"} or device in {"android", "ios"}


def _draft_candidate(peer: dict[str, Any]) -> dict[str, Any]:
    mobile = peer.get("mobile") or {}
    memory = peer.get("memory") or {}
    return {
        "hostname": peer.get("hostname") or peer.get("joined_peer_id") or "unknown",
        "runtime": mobile.get("runtime") or peer.get("runtime"),
        "kind": mobile.get("kind"),
        "free_gb": memory.get("free_gb"),
        "role": "async_draft_provider_only",
        "can_serve_transformer_blocks": False,
    }


def build_speculative_decode_plan(
    *,
    verifier_route: dict[str, Any],
    peers: list[dict[str, Any]] | None = None,
    draft_model_id: str = DEFAULT_DRAFT_MODEL,
    max_draft_tokens: int = 4,
    acceptance_window: int = 4,
) -> dict[str, Any]:
    """Return a dashboard/coordinator-safe speculative decode plan.

    The verifier is always authoritative. Draft providers may propose candidate
    tokens, including phones, but accepted output must match verifier logits. This
    keeps the plan useful for future speedups without overclaiming correctness or
    phone block-serving ability.
    """
    picked = _picked_route(verifier_route)
    bounded_max_draft = max(1, int(max_draft_tokens))
    bounded_window = max(1, int(acceptance_window))
    peers = list(peers or [])
    phone_candidates = [_draft_candidate(peer) for peer in peers if _is_phone(peer)]
    non_phone_candidates = [_draft_candidate(peer) for peer in peers if not _is_phone(peer)][:3]

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verifier": {
            "model_id": picked.get("model_id"),
            "authoritative": True,
            "claim_level": picked.get("claim_level"),
            "selector_mode": picked.get("selector_mode"),
            "proof_status": picked.get("proof_status") or {},
            "route_placement": picked.get("placement"),
        },
        "draft": {
            "mode": "async_draft_provider",
            "model_id": draft_model_id,
            "max_draft_tokens": bounded_max_draft,
            "acceptance_window": bounded_window,
            "phone_candidates": phone_candidates,
            "fallback_candidates": non_phone_candidates,
            "draft_quality_proven": False,
        },
        "phone_policy": {
            "phones_as_block_workers": False,
            "phones_as_draft_providers_only": True,
            "reason": "phones need separate throughput and correctness proof before any stronger role",
        },
        "correctness_contract": {
            "accepted_tokens_require_verifier_match": True,
            "verifier_logits_are_source_of_truth": True,
            "fallback_on_mismatch": "discard draft tokens and continue verifier-only decode",
            "updates_proof_status": False,
        },
        "execution_plan": [
            {
                "stage": "draft_propose",
                "actor": "draft provider",
                "claim_boundary": CLAIM_BOUNDARY,
                "description": "draft provider proposes up to max_draft_tokens candidate token ids",
            },
            {
                "stage": "verifier_validate",
                "actor": "BloomBee verifier route",
                "description": "verifier checks candidates and remains authoritative for accepted output",
            },
            {
                "stage": "accept_or_fallback",
                "actor": "coordinator/client",
                "description": "accept matching prefix only; otherwise drop to verifier-only decode",
            },
        ],
        "operator_next_steps": [
            "measure draft provider latency and token match rate before enabling speculative decode",
            "keep verifier-only generation as correctness fallback",
            "do not count phones as transformer block workers without a separate block-serving proof",
        ],
        "inference_proven": False,
        "generation_proven": False,
        "can_update_proof_status": False,
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-json", required=True, help="Route/explain JSON containing picked verifier model")
    parser.add_argument("--peers-json", default=None, help="JSON list of peer capability records")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--acceptance-window", type=int, default=4)
    args = parser.parse_args(argv)

    peers = _load_json(args.peers_json) if args.peers_json else []
    if not isinstance(peers, list):
        peers = peers.get("peers") or peers.get("active_peers") or []
    payload = build_speculative_decode_plan(
        verifier_route=_load_json(args.route_json),
        peers=peers,
        draft_model_id=args.draft_model,
        max_draft_tokens=args.max_draft_tokens,
        acceptance_window=args.acceptance_window,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
