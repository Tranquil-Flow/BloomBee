#!/usr/bin/env python3
"""Build a consistent per-peer benchmark matrix from raw JSONL bench output.

The route picker consumes a matrix keyed by peer hostname. Real
``bench_throughput.py`` JSONL outputs are single records without host labels.
This helper merges JSONL files (each row tagged with its source host) into
the structure the router expects, so measured numbers actually influence the
route decision instead of being silently ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_bench_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def group_records_by_host(records: Iterable[dict[str, Any]], *, default_host: str) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        host = record.get("host") or record.get("hostname") or default_host
        grouped.setdefault(host, {})[record["model"]] = {
            "decode_tok_per_s": record.get("decode_tok_per_s", 0.0),
            "prefill_tok_per_s": record.get("prefill_tok_per_s", 0.0),
            "params_b": record.get("params_b", 0.0),
            "device": record.get("device"),
            "dtype": record.get("dtype"),
        }
    return grouped


def build_matrix(
    jsonl_paths: Iterable[str | Path],
    *,
    peer_hosts: Iterable[str] | None = None,
    default_host: str = "this-host",
) -> dict[str, dict[str, Any]]:
    records = parse_bench_jsonl(jsonl_paths)
    by_host = group_records_by_host(records, default_host=default_host)
    matrix: dict[str, dict[str, Any]] = {}
    for host, models in by_host.items():
        matrix[host] = {"summary": {"hostname": host}, "models": models}
    if peer_hosts:
        for host in peer_hosts:
            matrix.setdefault(host, {"summary": {"hostname": host}, "models": {}})
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", help="bench_throughput.py JSONL files")
    parser.add_argument("--default-host", default="this-host")
    parser.add_argument("--peer-hosts", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    matrix = build_matrix(args.jsonl, peer_hosts=args.peer_hosts, default_host=args.default_host)
    text = json.dumps(matrix, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
