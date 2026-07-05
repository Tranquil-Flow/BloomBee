#!/usr/bin/env python3
"""Validate Qwen3-30B-A3B-Instruct-2507 cache readiness without proving generation.

This is a grunt-filter for reviewers: it checks whether the external Seagate HF
snapshot has the required sidecars and all expected safetensor shards. It does
not load the model, start BloomBee servers, prove generation, prove cached
generation, or prove load.

Run locally against an arbitrary fixture path, or use --remote to execute this
same validator on m4pro over SSH without requiring the remote repo to be up to
date.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CLAIM_BOUNDARY = "cache_download_readiness_only_no_generation_or_load_proof"
DEFAULT_SNAPSHOT_DIR = (
    "/Volumes/Seagate Portable Drive/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
    "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
DEFAULT_STATUS_PATH = (
    "~/Projects/distributed-inference-mvp/.local/status/"
    "instruct2507-full-download.status"
)
DEFAULT_STAGE_ROOT = "/Volumes/Exchange"
REQUIRED_SIDECARS = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_SIDECARS = (
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)
DEFAULT_EXPECTED_SHARD_COUNT = 16
MIN_SHARD_BYTES = 100 * 1024 * 1024


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser()


def _stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "bytes": st.st_size,
        "mtime": int(st.st_mtime),
        "is_symlink": path.is_symlink(),
    }


def _read_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key] = value
    return result


def _expected_shards(snapshot_dir: Path, errors: list[str], expected_count: int) -> list[str]:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        errors.append("missing model.safetensors.index.json; using numeric shard fallback")
        return [f"model-{idx:05d}-of-{expected_count:05d}.safetensors" for idx in range(1, expected_count + 1)]
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid model.safetensors.index.json: {exc}")
        return [f"model-{idx:05d}-of-{expected_count:05d}.safetensors" for idx in range(1, expected_count + 1)]
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        errors.append("model.safetensors.index.json missing weight_map; using numeric shard fallback")
        return [f"model-{idx:05d}-of-{expected_count:05d}.safetensors" for idx in range(1, expected_count + 1)]
    names = sorted({str(name) for name in weight_map.values() if str(name).endswith(".safetensors")})
    if not names:
        errors.append("model.safetensors.index.json weight_map has no safetensor shards; using numeric shard fallback")
        return [f"model-{idx:05d}-of-{expected_count:05d}.safetensors" for idx in range(1, expected_count + 1)]
    return names


def validate_cache(
    *,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    stage_root: str | Path = DEFAULT_STAGE_ROOT,
    expected_count: int = DEFAULT_EXPECTED_SHARD_COUNT,
    min_shard_bytes: int = MIN_SHARD_BYTES,
) -> dict[str, Any]:
    snapshot = _expand(snapshot_dir)
    status_file = _expand(status_path)
    stage = _expand(stage_root)
    errors: list[str] = []

    sidecars = {name: _stat(snapshot / name) for name in REQUIRED_SIDECARS}
    optional_sidecars = {name: _stat(snapshot / name) for name in OPTIONAL_SIDECARS}
    missing_sidecars = [name for name, item in sidecars.items() if not item["exists"]]
    if missing_sidecars:
        errors.append(f"missing required sidecars: {', '.join(missing_sidecars)}")

    expected_shards = _expected_shards(snapshot, errors, expected_count)
    shard_stats = {name: _stat(snapshot / name) for name in expected_shards}
    present_shards = [name for name, item in shard_stats.items() if item["exists"]]
    missing_shards = [name for name, item in shard_stats.items() if not item["exists"]]
    tiny_shards = [
        name
        for name, item in shard_stats.items()
        if item["exists"] and int(item.get("bytes") or 0) < min_shard_bytes
    ]
    if missing_shards:
        errors.append(f"missing {len(missing_shards)} expected shard(s)")
    if tiny_shards:
        errors.append(f"{len(tiny_shards)} shard(s) smaller than {min_shard_bytes} bytes")

    status = _read_status(status_file)
    current_file = status.get("CURRENT_FILE")
    stage_part = None
    if current_file:
        stage_part = _stat(stage / f"instruct2507-stage-{current_file}.part")

    total_bytes = sum(int(item.get("bytes") or 0) for item in shard_stats.values() if item["exists"])
    ready = not errors and len(present_shards) == len(expected_shards)

    return {
        "ok": True,
        "ready": ready,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": MODEL_ID,
        "snapshot_dir": str(snapshot),
        "status_path": str(status_file),
        "stage_root": str(stage),
        "expected_shard_count": len(expected_shards),
        "present_shard_count": len(present_shards),
        "missing_shard_count": len(missing_shards),
        "total_shard_bytes": total_bytes,
        "required_sidecars_present": not missing_sidecars,
        "missing_required_sidecars": missing_sidecars,
        "optional_sidecars": optional_sidecars,
        "missing_shards": missing_shards,
        "tiny_shards": tiny_shards,
        "first_missing_shard": missing_shards[0] if missing_shards else None,
        "download_status": status,
        "stage_part": stage_part,
        "can_start_expensive_full_generation_gate": ready,
        "generation_proven": False,
        "cache_generation_proven": False,
        "load_proven": False,
        "errors": errors,
    }


def _run_remote(args: argparse.Namespace) -> dict[str, Any]:
    script_text = Path(__file__).read_text(encoding="utf-8")
    remote_args = [
        "--json",
        "--snapshot-dir",
        args.snapshot_dir,
        "--status-path",
        args.status_path,
        "--stage-root",
        args.stage_root,
        "--expected-shard-count",
        str(args.expected_shard_count),
        "--min-shard-bytes",
        str(args.min_shard_bytes),
    ]
    remote_command = " ".join(shlex.quote(part) for part in ["python3", "-", *remote_args])
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            args.remote_host,
            remote_command,
        ],
        input=script_text,
        capture_output=True,
        text=True,
        timeout=args.remote_timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "ready": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "remote_host": args.remote_host,
            "remote_exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "errors": [proc.stderr.strip() or proc.stdout.strip() or "remote cache readiness check failed"],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "ready": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "remote_host": args.remote_host,
            "remote_exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "errors": [f"remote cache readiness emitted invalid JSON: {exc}"],
        }
    payload["remote_host"] = args.remote_host
    payload["remote_exit_code"] = proc.returncode
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    verdict = "READY" if report.get("ready") else "BLOCKED"
    status = report.get("download_status") or {}
    stage_part = report.get("stage_part") or {}
    lines = [
        f"# Instruct-2507 cache readiness — {verdict}",
        "",
        f"Model: `{report.get('model_id', MODEL_ID)}`",
        f"Claim boundary: `{report.get('claim_boundary', CLAIM_BOUNDARY)}`",
        f"Snapshot: `{report.get('snapshot_dir')}`",
        "",
        "## Shards",
        "",
        f"- Present: `{report.get('present_shard_count')}/{report.get('expected_shard_count')}`",
        f"- Missing: `{report.get('missing_shard_count')}`",
        f"- Total shard bytes present: `{report.get('total_shard_bytes')}`",
        f"- First missing shard: `{report.get('first_missing_shard')}`",
        "",
        "## Download process",
        "",
        f"- STATE: `{status.get('STATE')}`",
        f"- SHARD_COUNT: `{status.get('SHARD_COUNT')}`",
        f"- CURRENT_FILE: `{status.get('CURRENT_FILE')}`",
        f"- CURRENT_PHASE: `{status.get('CURRENT_PHASE')}`",
        f"- Stage part bytes: `{stage_part.get('bytes')}`",
        "",
        "## Proof flags",
        "",
        f"- can_start_expensive_full_generation_gate: `{report.get('can_start_expensive_full_generation_gate')}`",
        f"- generation_proven: `{report.get('generation_proven')}`",
        f"- cache_generation_proven: `{report.get('cache_generation_proven')}`",
        f"- load_proven: `{report.get('load_proven')}`",
    ]
    errors = report.get("errors") or []
    if errors:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--remote", action="store_true", help="Run this same validator on m4pro over SSH")
    parser.add_argument("--remote-host", default="m4pro", help="SSH host for --remote")
    parser.add_argument("--remote-timeout", type=int, default=45, help="Seconds to wait for remote SSH validator")
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--status-path", default=DEFAULT_STATUS_PATH)
    parser.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--expected-shard-count", type=int, default=DEFAULT_EXPECTED_SHARD_COUNT)
    parser.add_argument("--min-shard-bytes", type=int, default=MIN_SHARD_BYTES)
    args = parser.parse_args(argv)

    if args.remote:
        report = _run_remote(args)
    else:
        report = validate_cache(
            snapshot_dir=args.snapshot_dir,
            status_path=args.status_path,
            stage_root=args.stage_root,
            expected_count=args.expected_shard_count,
            min_shard_bytes=args.min_shard_bytes,
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
