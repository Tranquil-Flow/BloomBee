#!/usr/bin/env python3
"""Extract BloomBee server multiaddrs from retained server logs.

This is a convenience/runbook helper only. Finding a multiaddr in a log does not
prove the server is still alive or reachable; it only removes manual copy/paste
work after a server launch emits its advertised addresses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "server_log_multiaddr_extraction_only_no_connectivity_proof"
MULTIADDR_RE = re.compile(r"/ip[46]/[^\s,\]\)]+/tcp/\d+/p2p/[A-Za-z0-9]+")


def extract_multiaddrs(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in MULTIADDR_RE.finditer(text):
        value = match.group(0).rstrip(",]")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _addr_host(multiaddr: str) -> str:
    parts = multiaddr.split("/")
    # split('/ip4/1.2.3.4/tcp/...') => ['', 'ip4', '1.2.3.4', ...]
    return parts[2] if len(parts) > 2 else ""


def _is_loopback_or_unspecified(multiaddr: str) -> bool:
    host = _addr_host(multiaddr)
    return (
        host == "::1"
        or host == "0.0.0.0"
        or host.startswith("127.")
        or host.lower() == "localhost"
    )


def _preferred_multiaddr(multiaddrs: list[str]) -> str | None:
    for item in multiaddrs:
        if item.startswith("/ip4/") and not _is_loopback_or_unspecified(item):
            return item
    for item in multiaddrs:
        if not _is_loopback_or_unspecified(item):
            return item
    return multiaddrs[0] if multiaddrs else None


def build_multiaddr_report(text: str, *, source: str = "stdin") -> dict[str, Any]:
    multiaddrs = extract_multiaddrs(text)
    loopback = [item for item in multiaddrs if _is_loopback_or_unspecified(item)]
    preferred = _preferred_multiaddr(multiaddrs)
    ok = preferred is not None
    return {
        "ok": ok,
        "claim_boundary": CLAIM_BOUNDARY,
        "source": source,
        "multiaddrs": multiaddrs,
        "multiaddr_count": len(multiaddrs),
        "preferred_multiaddr": preferred,
        "loopback_multiaddrs": loopback,
        "blocked_reason": None if ok else "no /ip4 or /ip6 tcp/p2p multiaddr found in log",
        "connectivity_proven": False,
        "server_liveness_proven": False,
        "notes": [
            "Log extraction is not a live reachability check.",
            "Prefer non-loopback /ip4 addresses for clients outside the server process.",
            "Pass preferred_multiaddr to scripts/instruct2507_full_generation_gate.py --server-maddr after cache readiness is READY.",
        ],
    }


def _read_input(path_arg: str | None) -> tuple[str, str]:
    if not path_arg or path_arg == "-":
        return sys.stdin.read(), "stdin"
    path = Path(path_arg).expanduser()
    return path.read_text(encoding="utf-8", errors="replace"), str(path)


def render_markdown(report: dict[str, Any]) -> str:
    verdict = "FOUND" if report["ok"] else "BLOCKED"
    lines = [
        f"# BloomBee server multiaddr extraction — {verdict}",
        "",
        f"Claim boundary: `{report['claim_boundary']}`",
        f"Source: `{report['source']}`",
        f"Preferred multiaddr: `{report['preferred_multiaddr']}`",
        f"Multiaddrs found: `{report['multiaddr_count']}`",
        "",
        "## Negative proof flags",
        "",
        f"- connectivity_proven: `{report['connectivity_proven']}`",
        f"- server_liveness_proven: `{report['server_liveness_proven']}`",
    ]
    if report["multiaddrs"]:
        lines.extend(["", "## Candidates", ""])
        lines.extend(f"- `{item}`" for item in report["multiaddrs"])
    if report["blocked_reason"]:
        lines.extend(["", "## Blocked reason", "", f"- {report['blocked_reason']}"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", help="Server log path, or '-' / omitted for stdin")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args(argv)

    text, source = _read_input(args.log_path)
    report = build_multiaddr_report(text, source=source)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
