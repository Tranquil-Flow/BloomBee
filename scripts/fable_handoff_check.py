#!/usr/bin/env python3
"""Grunt-free Fable handoff checker for distributed-inference-mvp.

This script is intentionally local-first: it validates committed handoff docs,
claim-bounded evidence artifacts, and machine-readable MVP status without
requiring a live swarm or remote model cache. Pass --remote-download when the
reviewer wants the current m4pro Instruct-2507 download pulse too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DOC = PROJECT_ROOT / "docs" / "fable-post-mvp-handover.md"
LIVE_LOOP_EVIDENCE = (
    PROJECT_ROOT
    / "mvp_capabilities"
    / "distributed_evidence"
    / "post_mvp"
    / "live-continuous-batching-loop-unit-20260705.json"
)
LIVE_ADAPTER_EVIDENCE = (
    PROJECT_ROOT
    / "mvp_capabilities"
    / "distributed_evidence"
    / "post_mvp"
    / "continuous-batching-live-adapter-20260705.json"
)
INSTRUCT_MULTI_BLOCK_EVIDENCE = (
    PROJECT_ROOT
    / "mvp_capabilities"
    / "distributed_evidence"
    / "post_mvp"
    / "instruct2507-seagate-multiblock-proof-20260705T064511Z.json"
)
PHYSICAL_SHOWCASE_EVIDENCE = (
    PROJECT_ROOT
    / "mvp_capabilities"
    / "distributed_evidence"
    / "physical_showcase"
    / "qwen3-8b-final-physical-showcase-20260704T155722Z.json"
)

REQUIRED_HANDOFF_PHRASES = (
    "MVP-core claim boundary",
    "mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-live-adapter-20260705.json",
    "mvp_capabilities/distributed_evidence/post_mvp/live-continuous-batching-loop-unit-20260705.json",
    "live_continuous_decode_loop_unit_no_server_no_speedup",
    "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING",
    "staged-root `curl | dd`",
    "**Active background operation:** none known",
    "Instruct-2507 download and both INT8 streamed-reference parity gates are complete",
    "scripts/instruct2507_cache_readiness.py --remote",
    "scripts/instruct2507_full_generation_gate.py --remote-readiness",
    "scripts/extract_bloombee_multiaddr.py",
    "server_log_multiaddr_extraction_only_no_connectivity_proof",
    "ready_to_attempt_demo_safe_ladder",
    "cache_generation_proof_harness_only_no_live_generation",
    "multi_request_load_harness_only_no_live_traffic",
    "cache_download_readiness_only_no_generation_or_load_proof",
    "instruct2507_full_generation_gate_plan_only_no_live_generation",
    "448 passed, 23 skipped, 4 warnings",
    "Do **not** claim from this artifact",
)

STALE_HANDOFF_PHRASES = (
    "full model download is running on Seagate for the future full-generation gate",
    "Unfiltered default suite after handoff cleanup and continuous adapter: `423 passed",
    "Focused docs/continuous-batching/status guard suite: `8 passed",
    "Qwen/Qwen3-30B-A3B-Instruct-2507 cache: absent",
    "Qwen/Qwen3-30B-A3B-Instruct-2507:\n  lower gates passed: none\n  next gate: prescan",
)

HIGH_VALUE_FABLE_QUESTIONS = (
    "Is the live continuous-batching opt-in tick-row seam conservative enough before concurrent live-server parity work?",
    "Should the next expensive model proof target broader 30B INT8 prompt-set parity, Thinking-2507, NF4, or the x10-m4pro 235B route?",
    "Are deterministic scaled hidden-state load probes acceptable, or should the next gate use token-derived hidden states?",
    "Can route/demo selectors accidentally inherit proof between fp16, INT8, NF4, base, Instruct, and Thinking exact rows?",
    "Which stable skipped tests deserve CI jobs or smaller deterministic replacements?",
)

DO_NOT_SPEND_FABLE_TOKENS_ON = (
    "Reconstructing MVP-core status; mvp_status.py reports 100% and this checker validates the summary.",
    "Checking whether Instruct-2507 is still downloading; the handoff states no active background operation and completed INT8 parity gates.",
    "Checking whether the live batching unit artifact claims speedup; this checker enforces negative flags.",
    "Manually hunting for the key evidence paths; this checker verifies them directly.",
    "Counting Instruct-2507 shards by hand; completed evidence and cache-readiness helpers already encode this.",
    "Rebuilding the already-passed base/Instruct INT8 full/cache/load/token-parity ladder from scratch.",
)


def _run(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"missing required artifact: {path.relative_to(PROJECT_ROOT)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def _check_handoff_doc(errors: list[str]) -> dict[str, Any]:
    if not HANDOFF_DOC.exists():
        errors.append("missing docs/fable-post-mvp-handover.md")
        return {"path": str(HANDOFF_DOC.relative_to(PROJECT_ROOT)), "exists": False}

    text = HANDOFF_DOC.read_text(encoding="utf-8")
    missing = [phrase for phrase in REQUIRED_HANDOFF_PHRASES if phrase not in text]
    stale = [phrase for phrase in STALE_HANDOFF_PHRASES if phrase in text]
    for phrase in missing:
        errors.append(f"handoff missing required phrase: {phrase}")
    for phrase in stale:
        errors.append(f"handoff contains stale phrase: {phrase}")
    return {
        "path": str(HANDOFF_DOC.relative_to(PROJECT_ROOT)),
        "exists": True,
        "missing_required_phrases": missing,
        "stale_phrases_present": stale,
    }


def _check_mvp_status(errors: list[str]) -> dict[str, Any]:
    from mvp_capabilities.mvp_status import build_status_report

    report = build_status_report()
    planned = {item["id"]: item for item in report.get("planned_tasks", [])}
    continuous = planned.get("continuous_batching")

    if report.get("overall_percent") != 100:
        errors.append(f"unexpected MVP percent: {report.get('overall_percent')}")
    expected_task_summary = {"complete": 13, "partial": 3, "pending": 0, "blocked": 1, "total": 17}
    if report.get("task_summary") != expected_task_summary:
        errors.append(f"unexpected task_summary: {report.get('task_summary')}")
    expected_post_summary = {"complete": 4, "partial": 3, "pending": 0, "blocked": 1, "total": 8}
    if report.get("post_mvp_task_summary") != expected_post_summary:
        errors.append(f"unexpected post_mvp_task_summary: {report.get('post_mvp_task_summary')}")
    if not continuous:
        errors.append("missing continuous_batching planned task")
    else:
        evidence = continuous.get("evidence", "")
        if continuous.get("status") != "complete":
            errors.append(f"continuous_batching must be complete, got {continuous.get('status')}")
        if "live-continuous-batching-loop-unit-20260705.json" not in evidence:
            errors.append("continuous_batching evidence missing live-loop unit artifact")
        if "strict-live-cbkv-v16-outer-row-local-verified-20260706.json" not in evidence:
            errors.append("continuous_batching evidence missing strict live parity artifact")
        if "continuous-kv-joint-readiness-current-20260706.json" not in evidence:
            errors.append("continuous_batching evidence missing joint readiness artifact")

    return {
        "overall_bar": report.get("overall_bar"),
        "next_gate": report.get("next_gate"),
        "task_summary": report.get("task_summary"),
        "post_mvp_task_summary": report.get("post_mvp_task_summary"),
        "continuous_batching": continuous,
    }


def _check_evidence(errors: list[str]) -> dict[str, Any]:
    live_loop = _load_json(LIVE_LOOP_EVIDENCE, errors)
    live_adapter = _load_json(LIVE_ADAPTER_EVIDENCE, errors)
    instruct_multi_block = _load_json(INSTRUCT_MULTI_BLOCK_EVIDENCE, errors)
    physical = _load_json(PHYSICAL_SHOWCASE_EVIDENCE, errors)

    if live_loop:
        if live_loop.get("claim_boundary") != "live_continuous_decode_loop_unit_no_server_no_speedup":
            errors.append("live-loop evidence has unexpected claim_boundary")
        for flag in ("live_server_proven", "speedup_proven", "can_update_demo_status"):
            if live_loop.get(flag) is not False:
                errors.append(f"live-loop evidence must keep {flag}=false")
        if live_loop.get("live_loop_unit_proven") is not True:
            errors.append("live-loop evidence must prove only the injected unit seam")

    if live_adapter:
        for flag in ("live_server_proven", "speedup_proven", "can_update_demo_status"):
            if live_adapter.get(flag) is not False:
                errors.append(f"live-adapter evidence must keep {flag}=false")

    if instruct_multi_block:
        text = json.dumps(instruct_multi_block, sort_keys=True)
        if "Qwen/Qwen3-30B-A3B-Instruct-2507" not in text:
            errors.append("Instruct-2507 multi-block evidence does not mention exact model ID")

    if physical:
        text = json.dumps(physical, sort_keys=True)
        forbidden_values = ("bloombee://join", "token=")
        leaked = [needle for needle in forbidden_values if needle in text]
        if leaked:
            errors.append(f"physical showcase evidence contains raw join/token values: {leaked}")

    return {
        "live_loop": _artifact_summary(LIVE_LOOP_EVIDENCE, live_loop),
        "live_adapter": _artifact_summary(LIVE_ADAPTER_EVIDENCE, live_adapter),
        "instruct2507_multiblock": _artifact_summary(INSTRUCT_MULTI_BLOCK_EVIDENCE, instruct_multi_block),
        "physical_showcase": _artifact_summary(PHYSICAL_SHOWCASE_EVIDENCE, physical),
    }


def _artifact_summary(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path.relative_to(PROJECT_ROOT)), "exists": path.exists()}
    if payload is not None:
        for key in (
            "claim_boundary",
            "proof_gate",
            "model_id",
            "live_loop_unit_proven",
            "live_server_proven",
            "speedup_proven",
            "can_update_demo_status",
        ):
            if key in payload:
                summary[key] = payload[key]
    return summary


def _git_state() -> dict[str, Any]:
    head = _run(["/usr/bin/git", "rev-parse", "HEAD"])
    branch = _run(["/usr/bin/git", "branch", "--show-current"])
    status = _run(["/usr/bin/git", "status", "--short", "--untracked-files=all"])
    remote = _run(["/usr/bin/git", "ls-remote", "tranquil-flow", "refs/heads/main"], timeout=60)
    remote_sha = remote["stdout"].split()[0] if remote["exit_code"] == 0 and remote["stdout"] else None
    return {
        "head": head["stdout"],
        "branch": branch["stdout"],
        "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
        "tranquil_flow_main": remote_sha,
        "head_matches_tranquil_flow_main": bool(head["stdout"] and remote_sha == head["stdout"]),
    }


def _remote_download_state(errors: list[str]) -> dict[str, Any]:
    remote_script = r'''
cd ~/Projects/distributed-inference-mvp || exit 2
STATUS=.local/status/instruct2507-full-download.status
if [ -f "$STATUS" ]; then
  cat "$STATUS"
  current=$(awk -F= '/^CURRENT_FILE=/{print $2}' "$STATUS" || true)
  if [ -n "$current" ]; then
    part="/Volumes/Exchange/instruct2507-stage-${current}.part"
    final="/Volumes/Seagate Portable Drive/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/${current}"
    if [ -e "$part" ]; then stat -f "STAGE_PART_BYTES=%z" "$part"; fi
    if [ -e "$final" ]; then stat -f "FINAL_BYTES=%z" "$final"; fi
  fi
else
  echo STATE=missing_status
fi
echo TMUX_SESSIONS_BEGIN
PATH=/opt/homebrew/bin:$HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH tmux list-sessions 2>/dev/null || true
echo TMUX_SESSIONS_END
'''
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "m4pro", "bash", "-lc", remote_script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        errors.append(f"remote download pulse failed: {result.stderr.strip() or result.stdout.strip()}")
    parsed: dict[str, Any] = {"exit_code": result.returncode, "raw": result.stdout.strip()}
    for line in result.stdout.splitlines():
        if "=" in line and not line.startswith("TMUX_"):
            key, value = line.split("=", 1)
            parsed[key] = value
    return parsed


def _remote_cache_readiness(errors: list[str]) -> dict[str, Any]:
    script = PROJECT_ROOT / "scripts" / "instruct2507_cache_readiness.py"
    if not script.exists():
        errors.append("missing scripts/instruct2507_cache_readiness.py")
        return {"ok": False, "ready": False, "errors": ["missing script"]}
    result = subprocess.run(
        [sys.executable, str(script), "--remote", "--json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=75,
        check=False,
    )
    if result.returncode != 0:
        errors.append(f"remote cache readiness failed: {result.stderr.strip() or result.stdout.strip()}")
        return {
            "ok": False,
            "ready": False,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"remote cache readiness emitted invalid JSON: {exc}")
        return {"ok": False, "ready": False, "stdout": result.stdout.strip(), "errors": [str(exc)]}
    if not payload.get("ok", False):
        errors.append("remote cache readiness script returned ok=false")
    return payload



def _remote_demo_safe_ladder_plan(readiness: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    try:
        from scripts.instruct2507_full_generation_gate import build_gate_plan
    except Exception as exc:  # pragma: no cover - defensive handoff checker path
        errors.append(f"failed to import Instruct-2507 ladder planner: {exc}")
        return {"ok": False, "ready_to_attempt_demo_safe_ladder": False, "errors": [str(exc)]}
    try:
        return build_gate_plan(readiness=readiness)
    except Exception as exc:  # pragma: no cover - defensive handoff checker path
        errors.append(f"failed to build Instruct-2507 ladder plan: {exc}")
        return {"ok": False, "ready_to_attempt_demo_safe_ladder": False, "errors": [str(exc)]}



def build_report(*, include_remote: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    report: dict[str, Any] = {
        "ok": True,
        "project": "distributed-inference-mvp",
        "purpose": "Let Fable skip grunt reconstruction and spend tokens on hard review decisions.",
        "high_value_fable_questions": list(HIGH_VALUE_FABLE_QUESTIONS),
        "do_not_spend_fable_tokens_on": list(DO_NOT_SPEND_FABLE_TOKENS_ON),
    }
    report["git"] = _git_state()
    report["handoff_doc"] = _check_handoff_doc(errors)
    report["mvp_status"] = _check_mvp_status(errors)
    report["evidence"] = _check_evidence(errors)
    if include_remote:
        report["remote_download"] = _remote_download_state(errors)
        report["remote_cache_readiness"] = _remote_cache_readiness(errors)
        report["remote_demo_safe_ladder_plan"] = _remote_demo_safe_ladder_plan(
            report["remote_cache_readiness"], errors
        )
    report["errors"] = errors
    report["ok"] = not errors
    return report


def render_markdown(report: dict[str, Any]) -> str:
    ok = "PASS" if report["ok"] else "FAIL"
    git = report["git"]
    status = report["mvp_status"]
    lines = [
        f"# Fable handoff check — {ok}",
        "",
        f"Project: `{report['project']}`",
        f"HEAD: `{git.get('head')}`",
        f"Branch: `{git.get('branch')}`",
        f"Remote match: `{git.get('head_matches_tranquil_flow_main')}`",
        f"Dirty files: `{len(git.get('dirty_files', []))}`",
        "",
        "## MVP status",
        "",
        f"- Bar: `{status.get('overall_bar')}`",
        f"- Next gate: `{status.get('next_gate')}`",
        f"- Task summary: `{status.get('task_summary')}`",
        f"- Post-MVP summary: `{status.get('post_mvp_task_summary')}`",
        "",
        "## Fable should focus on",
        "",
    ]
    for question in report["high_value_fable_questions"]:
        lines.append(f"- {question}")
    lines.extend(["", "## Fable should not burn tokens on", ""])
    for item in report["do_not_spend_fable_tokens_on"]:
        lines.append(f"- {item}")

    remote = report.get("remote_download")
    if remote:
        lines.extend(
            [
                "",
                "## Remote download pulse",
                "",
                f"- STATE: `{remote.get('STATE')}`",
                f"- SHARD_COUNT: `{remote.get('SHARD_COUNT')}`",
                f"- CURRENT_FILE: `{remote.get('CURRENT_FILE')}`",
                f"- STAGE_PART_BYTES: `{remote.get('STAGE_PART_BYTES')}`",
                f"- FINAL_BYTES: `{remote.get('FINAL_BYTES')}`",
            ]
        )

    readiness = report.get("remote_cache_readiness")
    if readiness:
        stage_part = readiness.get("stage_part") or {}
        download_status = readiness.get("download_status") or {}
        lines.extend(
            [
                "",
                "## Remote cache readiness",
                "",
                f"- READY: `{readiness.get('ready')}`",
                f"- Claim boundary: `{readiness.get('claim_boundary')}`",
                f"- Shards: `{readiness.get('present_shard_count')}/{readiness.get('expected_shard_count')}`",
                f"- First missing shard: `{readiness.get('first_missing_shard')}`",
                f"- Current file: `{download_status.get('CURRENT_FILE')}`",
                f"- Current phase: `{download_status.get('CURRENT_PHASE')}`",
                f"- Stage part bytes: `{stage_part.get('bytes')}`",
                f"- Can start expensive full-generation gate: `{readiness.get('can_start_expensive_full_generation_gate')}`",
            ]
        )

    ladder = report.get("remote_demo_safe_ladder_plan")
    if ladder:
        lines.extend(
            [
                "",
                "## Remote demo-safe ladder plan",
                "",
                f"- READY_TO_ATTEMPT_DEMO_SAFE_LADDER: `{ladder.get('ready_to_attempt_demo_safe_ladder')}`",
                f"- Claim boundary: `{ladder.get('claim_boundary')}`",
                f"- Gates: `{ladder.get('demo_safe_ladder_gates')}`",
                f"- Full-generation command emitted: `{bool((ladder.get('full_generation_plan') or {}).get('parity_command'))}`",
                f"- Cache-generation command emitted: `{bool((ladder.get('cache_generation_plan') or {}).get('parity_command'))}`",
                f"- Load client command count: `{len((ladder.get('multi_request_load_plan') or {}).get('client_commands') or [])}`",
                f"- Blocked reasons: `{ladder.get('blocked_reasons')}`",
            ]
        )

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown")
    parser.add_argument("--remote-download", action="store_true", help="Also SSH to m4pro for the active download pulse")
    args = parser.parse_args(argv)

    report = build_report(include_remote=args.remote_download)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
