#!/usr/bin/env python3
"""Claim-bounded Instruct-2507 full-generation gate planner.

This script removes post-download grunt work without running the expensive gate.
It combines cache-readiness output with exact-model defaults, real BloomBee
server CLI syntax, and the existing ``mvp_capabilities.full_generation_proof``
plan/verify harness.

It never starts a server, never runs generation, and never updates
PROOF_STATUS.yaml. It only says whether the expensive full-generation gate is
*ready to attempt* and emits the commands needed once a server multiaddr exists.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvp_capabilities.full_generation_proof import build_full_generation_plan
from scripts.instruct2507_cache_readiness import MODEL_ID, validate_cache

CLAIM_BOUNDARY = "instruct2507_full_generation_gate_plan_only_no_live_generation"
DEFAULT_CACHE_DIR = "/Volumes/Seagate Portable Drive/huggingface/hub"
DEFAULT_EVIDENCE = (
    "mvp_capabilities/distributed_evidence/post_mvp/"
    "instruct2507-full-generation-forward-loop.json"
)
DEFAULT_PROMPT = "The moon is"
DEFAULT_SERVER_PLACEMENT = "m4pro-full=0:48"
DEFAULT_BLOCK_INDICES = "0:48"
DEFAULT_PORT = 31347
DEFAULT_MODE = "forward-loop"
DEFAULT_MAX_NEW_TOKENS = 1
PLACEHOLDER_MADDR = "<PASTE_M4PRO_FULL_SERVER_MULTIADDR>"


def _quote(value: str | int) -> str:
    return shlex.quote(str(value))


def _command(parts: list[str | int]) -> str:
    return " ".join(_quote(part) for part in parts)


def build_server_launch_command(
    *,
    model_id: str = MODEL_ID,
    cache_dir: str = DEFAULT_CACHE_DIR,
    block_indices: str = DEFAULT_BLOCK_INDICES,
    port: int = DEFAULT_PORT,
    public_name: str = "instruct2507-full-generation-gate",
    torch_dtype: str = "float16",
    device: str = "mps",
    throughput: str = "1",
) -> str:
    env = (
        "PYTHONPATH=.:src "
        f"HF_HOME={_quote(str(Path(cache_dir).parent))} "
        f"HF_HUB_CACHE={_quote(cache_dir)} "
        f"HF_XET_CACHE={_quote(str(Path(cache_dir).parent / 'xet'))} "
        "HF_HUB_DISABLE_XET=1 "
        "TRANSFORMERS_OFFLINE=1 "
        "HF_HUB_OFFLINE=1"
    )
    args: list[str | int] = [
        "python",
        "-m",
        "bloombee.cli.run_server",
        model_id,
        "--block_indices",
        block_indices,
        "--new_swarm",
        "--cache_dir",
        cache_dir,
        "--torch_dtype",
        torch_dtype,
        "--device",
        device,
        "--throughput",
        throughput,
        "--port",
        port,
        "--public_name",
        public_name,
        "--skip_reachability_check",
    ]
    return env + " " + _command(args)


def _read_readiness(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return validate_cache()
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _remote_readiness(timeout: int = 90) -> dict[str, Any]:
    script = PROJECT_ROOT / "scripts" / "instruct2507_cache_readiness.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--remote", "--json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "ready": False,
            "claim_boundary": "cache_readiness_remote_call_failed",
            "errors": [proc.stderr.strip() or proc.stdout.strip() or "remote readiness failed"],
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "ready": False,
            "claim_boundary": "cache_readiness_remote_invalid_json",
            "errors": [f"invalid remote readiness JSON: {exc}", proc.stdout.strip()],
        }


def build_gate_plan(
    *,
    readiness: dict[str, Any],
    server_maddrs: list[str] | None = None,
    model_id: str = MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    mode: str = DEFAULT_MODE,
    evidence_path: str = DEFAULT_EVIDENCE,
    server_placement: str = DEFAULT_SERVER_PLACEMENT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    server_maddrs = server_maddrs or []
    cache_ready = readiness.get("ready") is True
    blocked_reasons: list[str] = []
    if not cache_ready:
        first_missing = readiness.get("first_missing_shard")
        shard_state = f"{readiness.get('present_shard_count')}/{readiness.get('expected_shard_count')}"
        blocked_reasons.append(f"cache readiness is BLOCKED ({shard_state} shards; first missing: {first_missing})")
    if not server_maddrs:
        blocked_reasons.append("server multiaddr is not captured yet; launch server, then rerun with --server-maddr")

    placeholder_maddrs = server_maddrs or [PLACEHOLDER_MADDR]
    proof_plan = build_full_generation_plan(
        model_id=model_id,
        server_maddrs=placeholder_maddrs,
        server_placements=[server_placement],
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        mode=mode,
        evidence_path=evidence_path,
        reference_device="mps",
        reference_dtype="float16",
        distributed_dtype="float16",
    )
    server_launch_command = build_server_launch_command(model_id=model_id, cache_dir=cache_dir)

    return {
        "ok": True,
        "ready_to_attempt_full_generation": not blocked_reasons,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model_id,
        "proof_gate": "full_generation",
        "cache_readiness": {
            "ready": readiness.get("ready"),
            "claim_boundary": readiness.get("claim_boundary"),
            "present_shard_count": readiness.get("present_shard_count"),
            "expected_shard_count": readiness.get("expected_shard_count"),
            "first_missing_shard": readiness.get("first_missing_shard"),
            "can_start_expensive_full_generation_gate": readiness.get("can_start_expensive_full_generation_gate"),
            "errors": readiness.get("errors") or [],
        },
        "blocked_reasons": blocked_reasons,
        "server_launch_command": server_launch_command,
        "capture_multiaddr_instruction": (
            "After the server advertises its /ip4/.../tcp/.../p2p/... multiaddr, rerun this planner with "
            "--server-maddr '<that-multiaddr>' to remove the placeholder."
        ),
        "full_generation_plan": proof_plan,
        "post_success_instructions": [
            "Run the emitted parity_command only after cache readiness is READY and the server multiaddr is real.",
            "Run the emitted verify_command against the captured evidence JSON.",
            "Only update PROOF_STATUS.yaml full_generation after verify returns status=passed.",
            "This planner is not generation proof and does not prove cache_generation or multi_request_load.",
        ],
        "generation_proven": False,
        "cache_generation_proven": False,
        "load_proven": False,
        "can_update_proof_status": False,
    }


def render_markdown(plan: dict[str, Any]) -> str:
    verdict = "READY TO ATTEMPT" if plan["ready_to_attempt_full_generation"] else "BLOCKED"
    readiness = plan["cache_readiness"]
    proof = plan["full_generation_plan"]
    lines = [
        f"# Instruct-2507 full-generation gate plan — {verdict}",
        "",
        f"Model: `{plan['model_id']}`",
        f"Claim boundary: `{plan['claim_boundary']}`",
        "",
        "## Cache readiness",
        "",
        f"- READY: `{readiness.get('ready')}`",
        f"- Shards: `{readiness.get('present_shard_count')}/{readiness.get('expected_shard_count')}`",
        f"- First missing shard: `{readiness.get('first_missing_shard')}`",
        f"- Can start expensive gate: `{readiness.get('can_start_expensive_full_generation_gate')}`",
        "",
        "## Server launch command",
        "",
        "```bash",
        plan["server_launch_command"],
        "```",
        "",
        "## Full-generation parity command",
        "",
        "```bash",
        proof["parity_command"],
        "```",
        "",
        "## Verify command",
        "",
        "```bash",
        proof["verify_command"],
        "```",
    ]
    if plan["blocked_reasons"]:
        lines.extend(["", "## Blocked reasons", ""])
        lines.extend(f"- {reason}" for reason in plan["blocked_reasons"])
    lines.extend(["", "## Negative proof flags", ""])
    for key in ("generation_proven", "cache_generation_proven", "load_proven", "can_update_proof_status"):
        lines.append(f"- {key}: `{plan[key]}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    parser.add_argument("--readiness-json", help="Path to a cache-readiness JSON fixture/report")
    parser.add_argument("--remote-readiness", action="store_true", help="Fetch current m4pro cache readiness first")
    parser.add_argument("--server-maddr", action="append", dest="server_maddrs", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--mode", choices=("forward-loop", "generate-api"), default=DEFAULT_MODE)
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    args = parser.parse_args(argv)

    if args.remote_readiness:
        readiness = _remote_readiness()
    else:
        readiness = _read_readiness(args.readiness_json)
    plan = build_gate_plan(
        readiness=readiness,
        server_maddrs=args.server_maddrs,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        evidence_path=args.evidence,
    )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(render_markdown(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
