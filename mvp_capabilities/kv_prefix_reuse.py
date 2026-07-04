#!/usr/bin/env python3
"""Claim-bounded KV prefix-reuse planner simulation.

This module proves the pure planning invariant for prefill prefix reuse: a
request may reuse only a true prefix already represented by cache metadata, then
prefill only the suffix, and the reused prefix + suffix must reconstruct the
original token sequence exactly. It does not touch live KV tensors or BloomBee
server sessions; those remain separate proof gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

CLAIM_BOUNDARY = "kv_prefix_reuse_planner_simulation_no_live_cache_proof"
SOURCE = "kv_prefix_reuse.py"

OPERATOR_NEXT_STEPS = [
    "wire prefix lookup into real prefill/session cache metadata behind an opt-in flag",
    "prove hidden-state/token parity for reused-prefix and full-prefill paths",
    "measure memory and wall-clock impact before any demo or speedup promotion",
]


@dataclass(frozen=True)
class PrefixRequest:
    request_id: str
    token_ids: tuple[int, ...]

    def __init__(self, *, request_id: str, token_ids: Sequence[int]) -> None:
        object.__setattr__(self, "request_id", str(request_id))
        object.__setattr__(self, "token_ids", tuple(int(token) for token in token_ids))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.token_ids:
            raise ValueError("token_ids must be non-empty")


@dataclass(frozen=True)
class PrefixCacheEntry:
    cache_id: str
    token_ids: tuple[int, ...]

    def __init__(self, *, cache_id: str, token_ids: Sequence[int]) -> None:
        object.__setattr__(self, "cache_id", str(cache_id))
        object.__setattr__(self, "token_ids", tuple(int(token) for token in token_ids))
        if not self.cache_id:
            raise ValueError("cache_id must be non-empty")
        if not self.token_ids:
            raise ValueError("token_ids must be non-empty")


def _prefix_hash(token_ids: Sequence[int]) -> str:
    material = ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def _find_best_prefix(
    request_tokens: Sequence[int],
    entries: Sequence[PrefixCacheEntry],
    *,
    min_reuse_tokens: int,
) -> tuple[PrefixCacheEntry | None, int]:
    best_entry: PrefixCacheEntry | None = None
    best_len = 0
    for entry in entries:
        candidate_len = _common_prefix_len(request_tokens, entry.token_ids)
        if candidate_len > best_len:
            best_entry = entry
            best_len = candidate_len
    if best_len < int(min_reuse_tokens):
        return None, 0
    return best_entry, best_len


def _example_requests(name: str) -> list[PrefixRequest]:
    if name != "shared-doc-prefix":
        raise ValueError(f"unknown example: {name}")
    return [
        PrefixRequest(request_id="doc-a", token_ids=(101, 102, 103, 104)),
        PrefixRequest(request_id="doc-b", token_ids=(101, 102, 103, 205)),
        PrefixRequest(request_id="other", token_ids=(7, 8)),
    ]


def _validate_unique_request_ids(requests: Sequence[PrefixRequest]) -> None:
    seen: set[str] = set()
    for request in requests:
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        seen.add(request.request_id)


def plan_kv_prefix_reuse(
    *,
    requests: Sequence[PrefixRequest],
    initial_cache_entries: Sequence[PrefixCacheEntry] = (),
    min_reuse_tokens: int = 1,
) -> dict[str, object]:
    """Plan prefix reuse and prove reconstructed token sequences match.

    Requests are processed in order. After a request is planned, its full token
    sequence becomes a cache metadata entry so later requests may reuse any true
    prefix of it. Reuse is deliberately prefix-only; matching tokens later in a
    cache entry are ignored because KV state is positional.
    """
    if int(min_reuse_tokens) < 1:
        raise ValueError("min_reuse_tokens must be positive")
    _validate_unique_request_ids(requests)

    cache_entries = list(initial_cache_entries)
    plan: list[dict[str, object]] = []
    total_original = 0
    total_prefill = 0
    saved_prefill = 0

    for request in requests:
        total_original += len(request.token_ids)
        matched, prefix_len = _find_best_prefix(
            request.token_ids,
            cache_entries,
            min_reuse_tokens=int(min_reuse_tokens),
        )
        reused = list(request.token_ids[:prefix_len])
        suffix = list(request.token_ids[prefix_len:])
        reconstructed = reused + suffix
        total_prefill += len(suffix)
        saved_prefill += prefix_len
        plan.append(
            {
                "request_id": request.request_id,
                "matched_cache_id": matched.cache_id if matched is not None else None,
                "reused_token_count": prefix_len,
                "reused_token_ids": reused,
                "prefill_token_ids": suffix,
                "reconstructed_token_ids": reconstructed,
                "reconstruction_matches": reconstructed == list(request.token_ids),
            }
        )
        cache_entries.append(PrefixCacheEntry(cache_id=f"{request.request_id}:full", token_ids=request.token_ids))

    all_match = all(item["reconstruction_matches"] for item in plan)
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": "passed" if all_match else "failed",
        "request_count": len(requests),
        "min_reuse_tokens": int(min_reuse_tokens),
        "total_original_prefill_tokens": total_original,
        "total_planned_prefill_tokens": total_prefill,
        "saved_prefill_tokens": saved_prefill,
        "reuse_event_count": sum(1 for item in plan if item["reused_token_count"]),
        "all_reconstructions_match": all_match,
        "cache_entry_count_after_plan": len(cache_entries),
        "cache_token_hashes": {
            entry.cache_id: _prefix_hash(entry.token_ids)
            for entry in cache_entries
        },
        "plan": plan,
        "live_kv_cache_reuse_proven": False,
        "live_server_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "operator_next_steps": list(OPERATOR_NEXT_STEPS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["shared-doc-prefix"], default="shared-doc-prefix")
    parser.add_argument("--min-reuse-tokens", type=int, default=2)
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    payload = plan_kv_prefix_reuse(
        requests=_example_requests(args.example),
        min_reuse_tokens=args.min_reuse_tokens,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
