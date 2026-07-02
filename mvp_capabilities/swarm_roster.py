#!/usr/bin/env python3
"""Aggregate BloomBee peer capability JSON files into a swarm roster.

Standalone by design: no BloomBee imports. Inputs are JSON files emitted by
``peer_scan.py``. Output is either compact JSON for routers or a simple table
for humans during MVP demos.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAP_DIR = Path.home() / ".bloombee" / "capabilities"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_peer(peer: dict[str, Any], source: str) -> dict[str, Any]:
    memory = peer.get("memory") or {}
    accelerator = peer.get("accelerator") or {}
    network = peer.get("network") or {}
    hostname = peer.get("hostname") or peer.get("host") or Path(source).stem
    return {
        **peer,
        "hostname": str(hostname),
        "source": source,
        "memory": {
            "total_gb": _as_float(memory.get("total_gb")),
            "free_gb": _as_float(memory.get("free_gb")),
        },
        "accelerator": {
            **accelerator,
            "device": accelerator.get("device") or "cpu",
            "unified_memory": bool(accelerator.get("unified_memory", False)),
        },
        "network": {
            **network,
            "tailscale_ip": network.get("tailscale_ip"),
        },
    }


def _iter_json_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.exists():
            files.append(path)
    return files


def load_roster(paths: Iterable[str | Path] | None = None) -> list[dict[str, Any]]:
    """Load and normalize all peer JSON files from paths/directories.

    Multiple files for the same hostname can appear when operators capture both
    stdout and ``--out`` from ``peer_scan.py`` into the same directory. Treat a
    hostname as the peer identity and let the later path in deterministic sort
    order win, preventing accidental double-counting of one device.
    """
    paths = list(paths or [DEFAULT_CAP_DIR])
    peers_by_host: dict[str, dict[str, Any]] = {}
    for file_path in _iter_json_files(paths):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        peer = _normalise_peer(payload, str(file_path))
        peers_by_host[peer["hostname"]] = peer
    return [peers_by_host[hostname] for hostname in sorted(peers_by_host)]


def summarize_roster(peers: list[dict[str, Any]]) -> dict[str, Any]:
    accelerators = Counter((peer.get("accelerator") or {}).get("device") or "cpu" for peer in peers)
    return {
        "peer_count": len(peers),
        "total_memory_gb": round(sum(_as_float((peer.get("memory") or {}).get("total_gb")) for peer in peers), 2),
        "free_memory_gb": round(sum(_as_float((peer.get("memory") or {}).get("free_gb")) for peer in peers), 2),
        "accelerators": dict(sorted(accelerators.items())),
    }


def roster_document(peers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"summary": summarize_roster(peers), "peers": peers}


def render_table(peers: list[dict[str, Any]]) -> str:
    if not peers:
        return "No peers found. Run peer_scan.py on at least one node."
    lines = ["HOSTNAME              DEVICE  TOTAL_GB  FREE_GB  TAILSCALE"]
    for peer in peers:
        memory = peer.get("memory") or {}
        accelerator = peer.get("accelerator") or {}
        network = peer.get("network") or {}
        lines.append(
            f"{peer['hostname']:<21} {accelerator.get('device', 'cpu'):<6} "
            f"{_as_float(memory.get('total_gb')):>8.1f} {_as_float(memory.get('free_gb')):>7.1f} "
            f"{network.get('tailscale_ip') or '-'}"
        )
    summary = summarize_roster(peers)
    lines.append(
        f"TOTAL peers={summary['peer_count']} total_gb={summary['total_memory_gb']} "
        f"free_gb={summary['free_memory_gb']} accelerators={summary['accelerators']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Capability JSON files or directories")
    parser.add_argument("--cap-dir", action="append", help="Capability directory; may be repeated")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--min-free-gb", type=float, default=None, help="Filter peers below free-memory threshold")
    parser.add_argument("--device", default=None, help="Filter by accelerator device, e.g. mps/cuda/cpu")
    args = parser.parse_args(argv)

    paths: list[str] = []
    if args.cap_dir:
        paths.extend(args.cap_dir)
    paths.extend(args.paths)
    peers = load_roster(paths or [DEFAULT_CAP_DIR])
    if args.min_free_gb is not None:
        peers = [peer for peer in peers if _as_float((peer.get("memory") or {}).get("free_gb")) >= args.min_free_gb]
    if args.device:
        peers = [peer for peer in peers if (peer.get("accelerator") or {}).get("device") == args.device]

    if args.json:
        print(json.dumps(roster_document(peers), indent=2, sort_keys=True))
    else:
        print(render_table(peers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
