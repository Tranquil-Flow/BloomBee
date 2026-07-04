#!/usr/bin/env python3
"""Deterministic draft-provider contract for speculative decoding MVP work.

This module is deliberately small and dependency-free. It lets us test the
phone-as-draft interface on a laptop before any Android/Termux transport exists:
a provider proposes token ids, the verifier remains authoritative, and the report
counts proposed/accepted/rejected tokens without claiming generation speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

CLAIM_BOUNDARY = "draft_provider_contract_only_no_generation_proof"
SOURCE = "draft_provider.py"


class DraftProvider(Protocol):
    """Minimal interface a local model, phone bridge, or fake provider must satisfy."""

    provider_id: str
    provider_kind: str

    def propose(self, prompt_tokens: Sequence[int], max_draft_tokens: int) -> "DraftProposal":
        """Return up to ``max_draft_tokens`` proposed token ids."""


@dataclass(frozen=True)
class DraftProposal:
    provider_id: str
    provider_kind: str
    prompt_token_count: int
    max_draft_tokens: int
    draft_tokens: tuple[int, ...]
    elapsed_ms: float

    def to_payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "prompt_token_count": self.prompt_token_count,
            "max_draft_tokens": self.max_draft_tokens,
            "draft_tokens": list(self.draft_tokens),
            "draft_token_count": len(self.draft_tokens),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


@dataclass(frozen=True)
class StaticDraftProvider:
    """Test/smoke provider that returns a fixed token list."""

    draft_tokens: tuple[int, ...]
    provider_id: str = "static-draft-provider"
    provider_kind: str = "deterministic_fake"

    def propose(self, prompt_tokens: Sequence[int], max_draft_tokens: int) -> DraftProposal:
        start = time.perf_counter()
        bounded = max(0, int(max_draft_tokens))
        tokens = self.draft_tokens[:bounded]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return DraftProposal(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            prompt_token_count=len(prompt_tokens),
            max_draft_tokens=bounded,
            draft_tokens=tokens,
            elapsed_ms=elapsed_ms,
        )


@dataclass(frozen=True)
class DeterministicHashDraftProvider:
    """Dependency-free fake provider for RED/GREEN tests and transport smoke.

    It hashes the prompt plus a counter into token ids. This is not a model; it is
    a deterministic stand-in with the same interface a local tiny model or phone
    bridge must implement later.
    """

    provider_id: str = "hash-draft-provider"
    provider_kind: str = "deterministic_hash_fake"
    vocab_size: int = 32_000
    seed: str = "bloombee-draft-provider"

    def propose(self, prompt_tokens: Sequence[int], max_draft_tokens: int) -> DraftProposal:
        start = time.perf_counter()
        bounded = max(0, int(max_draft_tokens))
        vocab = max(1, int(self.vocab_size))
        prompt = ",".join(str(int(token)) for token in prompt_tokens)
        draft_tokens: list[int] = []
        for index in range(bounded):
            material = f"{self.seed}|{prompt}|{index}".encode("utf-8")
            draft_tokens.append(int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % vocab)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return DraftProposal(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            prompt_token_count=len(prompt_tokens),
            max_draft_tokens=bounded,
            draft_tokens=tuple(draft_tokens),
            elapsed_ms=elapsed_ms,
        )


def evaluate_draft_against_verifier(
    draft_tokens: Sequence[int],
    verifier_tokens: Sequence[int],
) -> dict[str, object]:
    """Compare a draft proposal with authoritative verifier token ids.

    Acceptance is the longest matching prefix. The first mismatch forces fallback
    to the verifier token at that position. If the entire draft is accepted and
    the verifier supplied one more token, that token is reported as a bonus token;
    no generation proof or speedup claim is made.
    """
    draft = [int(token) for token in draft_tokens]
    verifier = [int(token) for token in verifier_tokens]
    accepted: list[int] = []
    for index, token in enumerate(draft):
        if index >= len(verifier) or verifier[index] != token:
            break
        accepted.append(token)

    accepted_count = len(accepted)
    rejected = draft[accepted_count:]
    fallback_token = verifier[accepted_count] if rejected and accepted_count < len(verifier) else None
    bonus_token = verifier[accepted_count] if not rejected and accepted_count < len(verifier) else None
    acceptance_rate = accepted_count / len(draft) if draft else 0.0
    committed = list(accepted)
    if fallback_token is not None:
        committed.append(fallback_token)
    elif bonus_token is not None:
        committed.append(bonus_token)

    return {
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "proposed_count": len(draft),
        "acceptance_rate": round(acceptance_rate, 6),
        "verifier_fallback_token": fallback_token,
        "verifier_bonus_token": bonus_token,
        "committed_tokens": committed,
        "verifier_authoritative": True,
        "accepted_tokens_require_verifier_match": True,
        "fallback_on_mismatch": "discard rejected draft suffix and continue with verifier token",
    }


def build_draft_provider_report(
    *,
    provider: DraftProvider,
    prompt_tokens: Sequence[int],
    verifier_tokens: Sequence[int],
    max_draft_tokens: int,
) -> dict[str, object]:
    proposal = provider.propose(prompt_tokens, max_draft_tokens)
    verdict = evaluate_draft_against_verifier(proposal.draft_tokens, verifier_tokens)
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "provider": {
            "provider_id": proposal.provider_id,
            "provider_kind": proposal.provider_kind,
            "phone_compatible_interface": True,
            "can_serve_transformer_blocks": False,
        },
        "prompt_tokens": [int(token) for token in prompt_tokens],
        "verifier_tokens": [int(token) for token in verifier_tokens],
        "proposal": proposal.to_payload(),
        "verdict": verdict,
        "dashboard_counters": {
            "proposed": verdict["proposed_count"],
            "accepted": verdict["accepted_count"],
            "rejected": verdict["rejected_count"],
            "acceptance_rate": verdict["acceptance_rate"],
        },
        "operator_next_steps": [
            "run this same provider contract over a phone HTTP/stdio bridge",
            "measure draft latency and exact-token acceptance rate against a live verifier",
            "keep verifier-only decode as fallback until wall-clock speedup is proven",
        ],
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def _parse_int_list(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", default="", help="Comma-separated prompt token ids")
    parser.add_argument("--verifier-tokens", required=True, help="Comma-separated authoritative verifier token ids")
    parser.add_argument("--draft-tokens", default=None, help="Comma-separated fixed draft token ids; omit for hash fake")
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--provider-id", default=None)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    args = parser.parse_args(argv)

    prompt_tokens = _parse_int_list(args.prompt_tokens)
    verifier_tokens = _parse_int_list(args.verifier_tokens)
    if args.draft_tokens is not None:
        provider = StaticDraftProvider(
            draft_tokens=_parse_int_list(args.draft_tokens),
            provider_id=args.provider_id or "static-draft-provider",
        )
    else:
        provider = DeterministicHashDraftProvider(
            provider_id=args.provider_id or "hash-draft-provider",
            vocab_size=args.vocab_size,
        )
    payload = build_draft_provider_report(
        provider=provider,
        prompt_tokens=prompt_tokens,
        verifier_tokens=verifier_tokens,
        max_draft_tokens=args.max_draft_tokens,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
