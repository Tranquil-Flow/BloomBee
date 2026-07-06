#!/usr/bin/env python3
"""Plan and verify live-server continuous batching parity evidence.

This verifier is intentionally narrower than a speedup proof. It requires a
captured live-server artifact with opt-in continuous batching enabled, at least
one late-arriving request, at least one batched decode tick, and exact generated
-token/logit fingerprint parity against baseline rows. It never promotes demo or
speedup status by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PLAN_CLAIM_BOUNDARY = "live_continuous_batching_server_proof_harness_only_no_live_traffic"
VERIFY_CLAIM_BOUNDARY = "verified_live_continuous_batching_server_concurrent_arrival_parity"
PROOF_GATE = "continuous_batching"
OPT_IN_FLAG = "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"


def _token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        tokens.append(int(item))
    return tokens


def _first_token_list(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[int] | None:
    for key in keys:
        tokens = _token_list(mapping.get(key))
        if tokens is not None:
            return tokens
    return None


def _first_mapping(mapping: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any] | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _fingerprint(mapping: Mapping[str, Any]) -> str | None:
    for key in ("logits_sha256", "logits_hash", "logits_checksum", "logits_fingerprint"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _arrival_tick(row: Mapping[str, Any]) -> int | None:
    for key in ("arrival_tick", "arrival_step", "arrival_index"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    for key in ("arrival_offset_ms", "arrival_offset_s"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 1 if float(value) > 0 else 0
    return None


def build_live_server_continuous_batching_plan(
    *,
    model_id: str,
    evidence_path: str = ".local/live-continuous-batching-evidence.json",
    block_range: str = "0:1",
    server_log: str = ".local/live-continuous-batching-server.log",
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "evidence_path": evidence_path,
        "block_range": block_range,
        "operator_commands": [
            f"{OPT_IN_FLAG}=1 PYTHONPATH=.:src python -m bloombee.cli.run_server {model_id} --new_swarm --block_indices {block_range} 2>&1 | tee {server_log}",
            "capture two or more live client requests with staggered arrival ticks into the evidence JSON; do not edit generated token IDs by hand",
            f"python mvp_capabilities/continuous_batching_live_server_proof.py verify --model {model_id} --evidence {evidence_path}",
        ],
        "verify_command": f"python mvp_capabilities/continuous_batching_live_server_proof.py verify --model {model_id} --evidence {evidence_path}",
        "live_server_late_arrival_parity_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "notes": [
            "Planning output is not proof.",
            "The evidence must come from a live server with the opt-in flag enabled.",
            "Speedup requires a later wall-clock throughput artifact after parity passes.",
        ],
    }


def verify_live_server_continuous_batching_payload(
    payload: Mapping[str, Any],
    *,
    model_id: str | None = None,
    min_requests: int = 2,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    failed: list[str] = []
    evidence_model = payload.get("model_id") or payload.get("model")
    if model_id is not None and evidence_model != model_id:
        failed.append("evidence model mismatch")
    if evidence_model is None:
        failed.append("evidence model is missing")

    proof_gate = payload.get("proof_gate")
    if proof_gate not in (None, PROOF_GATE):
        failed.append("evidence proof_gate is not continuous_batching")
    if payload.get("opt_in_flag") != OPT_IN_FLAG:
        failed.append(f"opt_in_flag must be {OPT_IN_FLAG}")
    if payload.get("opt_in_enabled") is not True:
        failed.append("live continuous batching opt-in was not enabled")
    if payload.get("server_observed_live_continuous_batches") is not True and payload.get("live_server_proven") is not True:
        failed.append("live server did not report continuous batching observation")
    if payload.get("speedup_proven") is True or payload.get("wallclock_speedup_proven") is True:
        failed.append("evidence unexpectedly claims speedup")

    rows = payload.get("requests") or payload.get("request_results")
    if not isinstance(rows, list):
        failed.append("requests must be a list")
        rows = []
    if len(rows) < min_requests:
        failed.append(f"expected at least {min_requests} requests")

    seen_ids: set[str] = set()
    arrival_ticks: list[int] = []
    arrival_tick_by_request: dict[str, int] = {}
    row_summaries: list[dict[str, Any]] = []
    token_mismatch = False
    logits_missing_or_mismatch = False
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            failed.append(f"request {index} must be an object")
            continue
        row: Mapping[str, Any] = raw
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = f"request-{index}"
        if request_id in seen_ids:
            failed.append(f"duplicate request_id: {request_id}")
        seen_ids.add(request_id)

        tick = _arrival_tick(row)
        if tick is None:
            failed.append(f"request {request_id} arrival tick/offset missing")
        else:
            arrival_ticks.append(tick)
            arrival_tick_by_request[request_id] = tick

        baseline = _first_mapping(row, ("baseline", "serial_baseline", "verifier_only")) or {}
        continuous = _first_mapping(row, ("continuous", "live_continuous", "batched")) or {}
        if not baseline:
            failed.append(f"request {request_id} baseline object is missing")
        if not continuous:
            failed.append(f"request {request_id} continuous object is missing")
        baseline_tokens = _first_token_list(baseline, ("generated_token_ids", "output_token_ids", "token_ids", "tokens"))
        continuous_tokens = _first_token_list(continuous, ("generated_token_ids", "output_token_ids", "token_ids", "tokens"))
        if baseline_tokens is None or not baseline_tokens:
            failed.append(f"request {request_id} baseline generated token IDs missing or empty")
        if continuous_tokens is None or not continuous_tokens:
            failed.append(f"request {request_id} continuous generated token IDs missing or empty")
        if baseline_tokens is not None and continuous_tokens is not None and baseline_tokens != continuous_tokens:
            failed.append(f"request {request_id} generated token IDs differ from baseline")
            token_mismatch = True

        baseline_fp = _fingerprint(baseline)
        continuous_fp = _fingerprint(continuous)
        if baseline_fp is None or continuous_fp is None:
            failed.append(f"request {request_id} logits fingerprint missing")
            logits_missing_or_mismatch = True
        elif baseline_fp != continuous_fp:
            failed.append(f"request {request_id} logits fingerprint differs from baseline")
            logits_missing_or_mismatch = True

        row_summaries.append(
            {
                "request_id": request_id,
                "arrival_tick": tick,
                "generated_token_count": len(baseline_tokens or []),
                "tokens_match": baseline_tokens is not None and baseline_tokens == continuous_tokens,
                "logits_fingerprint_match": baseline_fp is not None and baseline_fp == continuous_fp,
            }
        )

    late_arrival_observed = bool(arrival_ticks and max(arrival_ticks) > min(arrival_ticks))
    if not late_arrival_observed:
        failed.append("no late-arrival request observed")

    report = payload.get("live_continuous_report") or payload.get("session_report") or {}
    tick_batches = report.get("tick_batches") if isinstance(report, Mapping) else None
    if not isinstance(tick_batches, list):
        failed.append("live_continuous_report.tick_batches must be a list")
        tick_batches = []
    batched_tick_count = 0
    server_first_seen_tick_by_request: dict[str, int] = {}
    for index, tick in enumerate(tick_batches):
        if isinstance(tick, Mapping) and isinstance(tick.get("request_ids"), list):
            tick_value = tick.get("tick")
            if not isinstance(tick_value, int) or isinstance(tick_value, bool):
                failed.append(f"live_continuous_report tick batch {index} tick missing")
                continue
            request_ids = tick["request_ids"]
            active_mask_raw = tick.get("active_mask")
            if isinstance(active_mask_raw, list) and len(active_mask_raw) == len(request_ids):
                active_mask = [bool(value) for value in active_mask_raw]
            else:
                active_mask = [True] * len(request_ids)
            active_request_ids = [
                request_id
                for request_id, is_active in zip(request_ids, active_mask)
                if is_active
            ]
            if len(active_request_ids) > 1:
                batched_tick_count += 1
            for raw_request_id in active_request_ids:
                if not isinstance(raw_request_id, str) or not raw_request_id:
                    failed.append(f"live_continuous_report tick batch {index} has invalid request_id")
                    continue
                if raw_request_id not in seen_ids:
                    failed.append(f"server tick batch references unknown request_id: {raw_request_id}")
                    continue
                server_first_seen_tick_by_request.setdefault(raw_request_id, int(tick_value))
    if batched_tick_count <= 0:
        failed.append("no batched live-continuous tick observed")

    for request_id in sorted(seen_ids):
        first_seen_tick = server_first_seen_tick_by_request.get(request_id)
        if first_seen_tick is None:
            failed.append(f"server tick batches never mention request {request_id}")
            continue
        declared_arrival_tick = arrival_tick_by_request.get(request_id)
        if declared_arrival_tick is not None and first_seen_tick < declared_arrival_tick:
            failed.append(f"server observed request {request_id} before declared arrival tick")

    status = "passed" if not failed else "failed"
    token_parity = bool(rows) and not token_mismatch and not any(
        "generated token IDs" in check or "token mismatch" in check
        for check in failed
    )
    logits_parity = bool(rows) and not logits_missing_or_mismatch and not any(
        "logits" in check for check in failed
    )
    return {
        "model_id": evidence_model,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "status": status,
        "evidence_path": str(evidence_path) if evidence_path is not None else None,
        "request_count": len(rows),
        "late_arrival_observed": late_arrival_observed,
        "batched_tick_count": batched_tick_count,
        "token_parity_proven": token_parity,
        "logits_fingerprint_parity_proven": logits_parity,
        "live_server_late_arrival_parity_proven": status == "passed",
        "live_server_proven": status == "passed",
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_demo_status": False,
        "can_update_proof_status": False,
        "proof_status_update": {},
        "failed_checks": failed,
        "requests": row_summaries,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a live-server proof capture plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--evidence", default=".local/live-continuous-batching-evidence.json")
    plan.add_argument("--block-range", default="0:1")
    plan.add_argument("--server-log", default=".local/live-continuous-batching-server.log")

    verify = sub.add_parser("verify", help="Verify captured live-server evidence")
    verify.add_argument("--model", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_live_server_continuous_batching_plan(
            model_id=args.model,
            evidence_path=args.evidence,
            block_range=args.block_range,
            server_log=args.server_log,
        )
    else:
        payload = verify_live_server_continuous_batching_payload(
            _read_json(args.evidence),
            model_id=args.model,
            evidence_path=args.evidence,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload.get("status") in (None, "passed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
