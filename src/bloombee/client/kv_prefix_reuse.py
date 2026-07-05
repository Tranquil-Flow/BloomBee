"""Claim-bounded KV-prefix reuse metadata helpers.

These helpers only detect and record same-prefix/varied-suffix prefill metadata.
They do not enable server-side KV cache reuse by themselves and must not be used
as a speedup/demo proof without separate live cache evidence.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

ENV_ENABLE_KV_PREFIX_REUSE = "BLOOMBEE_ENABLE_KV_PREFIX_REUSE"
CLAIM_BOUNDARY = "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"


def is_kv_prefix_reuse_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_KV_PREFIX_REUSE, "0") == "1"


def _row_tokens(row: Any) -> list[int]:
    if hasattr(row, "detach"):
        row = row.detach().cpu().tolist()
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise TypeError("input rows must be sequences of token IDs")
    tokens: list[int] = []
    for token in row:
        if hasattr(token, "item"):
            token = token.item()
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("input rows must contain integer token IDs")
        tokens.append(int(token))
    if not tokens:
        raise ValueError("input rows must be non-empty")
    return tokens


def normalize_token_rows(input_ids: Any) -> list[list[int]]:
    if hasattr(input_ids, "detach"):
        if getattr(input_ids, "ndim", None) != 2:
            raise ValueError("input_ids tensor must be rank-2 [batch, seq]")
        input_ids = input_ids.detach().cpu().tolist()
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise TypeError("input_ids must be a batch sequence")
    rows = [_row_tokens(row) for row in input_ids]
    if len(rows) < 2:
        raise ValueError("KV prefix reuse metadata requires at least two requests")
    return rows


def common_prefix(rows: Sequence[Sequence[int]]) -> list[int]:
    if not rows:
        return []
    prefix: list[int] = []
    min_len = min(len(row) for row in rows)
    for idx in range(min_len):
        token = int(rows[0][idx])
        if all(int(row[idx]) == token for row in rows[1:]):
            prefix.append(token)
        else:
            break
    return prefix


def build_prefill_metadata_event(
    input_ids: Any,
    *,
    request_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = normalize_token_rows(input_ids)
    prefix = common_prefix(rows)
    if not prefix:
        raise ValueError("requests do not share a non-empty prefix")
    if request_ids is None:
        request_ids = [f"request-{idx}" for idx in range(len(rows))]
    if len(request_ids) != len(rows):
        raise ValueError("request_ids length must match input batch size")

    requests: list[dict[str, Any]] = []
    suffixes: set[tuple[int, ...]] = set()
    for request_id, row in zip(request_ids, rows):
        suffix = list(row[len(prefix):])
        suffixes.add(tuple(suffix))
        requests.append(
            {
                "request_id": str(request_id),
                "prefix_token_ids": list(prefix),
                "suffix_token_ids": suffix,
                "prefill_token_count": len(row),
                "reusable_prefix_token_count": len(prefix),
            }
        )

    return {
        "source": "bloombee.client.kv_prefix_reuse",
        "claim_boundary": CLAIM_BOUNDARY,
        "opt_in_flag": ENV_ENABLE_KV_PREFIX_REUSE,
        "request_count": len(rows),
        "common_prefix_token_ids": list(prefix),
        "common_prefix_token_count": len(prefix),
        "requests": requests,
        "same_prefix_varied_suffix_proven": len(suffixes) >= len(rows),
        "runtime_prefill_metadata_proven": True,
        "live_kv_cache_reuse_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
    }
