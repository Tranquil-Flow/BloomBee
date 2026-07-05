#!/usr/bin/env python3
"""Fail-closed scans for raw join URLs/tokens in committed evidence artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("raw_join_url_value", re.compile(r"bloombee://join", re.IGNORECASE)),
    ("raw_token_query", re.compile(r"[?&]token=", re.IGNORECASE)),
    ("raw_token_key", re.compile(r'"token"\s*:', re.IGNORECASE)),
    ("raw_join_url_key", re.compile(r'"(?:raw_)?join_url"\s*:', re.IGNORECASE)),
    ("raw_authorization_key", re.compile(r'"authorization"\s*:', re.IGNORECASE)),
    ("raw_hf_token_key", re.compile(r'"(?:hf_token|HF_TOKEN)"\s*:', re.IGNORECASE)),
    ("raw_access_token_key", re.compile(r'"access_token"\s*:', re.IGNORECASE)),
    ("raw_api_key", re.compile(r'"api_key"\s*:', re.IGNORECASE)),
    ("raw_password_key", re.compile(r'"password"\s*:', re.IGNORECASE)),
    ("raw_hf_token_value", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
)


def _json_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".json" else []
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def scan_evidence_tree(root: str | Path) -> list[dict[str, Any]]:
    """Return raw-secret findings for JSON evidence files under ``root``.

    Hash fields such as ``token_sha256``/``join_url_sha256`` and the boolean
    marker ``raw_join_url_recorded_in_scratch_only`` do not match these exact
    raw-key/value patterns. Any finding here is intentionally treated as a
    commit-blocking issue.
    """
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    for path in _json_files(root_path):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern_id, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "pattern_id": pattern_id,
                            "excerpt": line.strip()[:240],
                        }
                    )
    return findings


__all__ = ["FORBIDDEN_PATTERNS", "scan_evidence_tree"]
