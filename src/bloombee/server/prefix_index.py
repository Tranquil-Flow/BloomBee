"""Server-side prefix index for KV prefix reuse.

This module is intentionally small and deterministic: it indexes completed KV
prefixes by token IDs so a later same-prefix request can locate a source cache
handle before a handle-to-handle tensor copy. It is not a proof by itself; proof
still requires server response metadata showing an actual copy/checksum.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

from bloombee.data_structures import Handle


def _normalize_tokens(tokens: Iterable[int]) -> tuple[int, ...]:
    out: list[int] = []
    for token in tokens:
        if not isinstance(token, int) or isinstance(token, bool):
            raise ValueError("token prefixes must contain integer token IDs")
        out.append(int(token))
    if not out:
        raise ValueError("token prefix must be non-empty")
    return tuple(out)


def prefix_tokens_sha256(tokens: Iterable[int]) -> str:
    normalized = _normalize_tokens(tokens)
    material = json.dumps(list(normalized), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class KVPrefixIndexEntry:
    token_prefix: tuple[int, ...]
    handle: Handle
    prefix_length: int
    token_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KVPrefixIndex:
    """LRU index from token prefix hashes to source cache handles."""

    def __init__(self, *, max_size_tokens: int | None = None):
        self.max_size_tokens = max_size_tokens
        self._entries: OrderedDict[str, KVPrefixIndexEntry] = OrderedDict()
        self.current_size_tokens = 0

    def register(
        self,
        token_prefix: Iterable[int],
        *,
        handle: Handle,
        prefix_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KVPrefixIndexEntry:
        tokens = _normalize_tokens(token_prefix)
        length = len(tokens) if prefix_length is None else int(prefix_length)
        if length <= 0 or length > len(tokens):
            raise ValueError("prefix_length must be within the token_prefix length")
        indexed_tokens = tokens[:length]
        key = prefix_tokens_sha256(indexed_tokens)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_size_tokens -= previous.prefix_length
        entry = KVPrefixIndexEntry(
            token_prefix=indexed_tokens,
            handle=handle,
            prefix_length=length,
            token_sha256=key,
            metadata=dict(metadata or {}),
        )
        self._entries[key] = entry
        self.current_size_tokens += length
        self._evict_if_needed()
        return entry

    def lookup(self, tokens: Iterable[int], *, min_prefix_length: int = 1) -> KVPrefixIndexEntry | None:
        query = _normalize_tokens(tokens)
        min_len = max(1, int(min_prefix_length))
        for length in range(len(query), min_len - 1, -1):
            key = prefix_tokens_sha256(query[:length])
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.token_prefix != query[:length]:
                continue
            self._entries.move_to_end(key)
            return entry
        return None

    def remove(self, token_prefix: Iterable[int]) -> KVPrefixIndexEntry | None:
        key = prefix_tokens_sha256(token_prefix)
        entry = self._entries.pop(key, None)
        if entry is not None:
            self.current_size_tokens -= entry.prefix_length
        return entry

    def _evict_if_needed(self) -> None:
        if self.max_size_tokens is None:
            return
        while self.current_size_tokens > self.max_size_tokens and self._entries:
            _, removed = self._entries.popitem(last=False)
            self.current_size_tokens -= removed.prefix_length

    def __len__(self) -> int:
        return len(self._entries)
