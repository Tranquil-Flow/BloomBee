#!/usr/bin/env python3
"""Fail-closed wrap-up report for speculative decode + phone-worker lanes.

This is a consolidation artifact, not new runtime proof. It consumes the latest
cross-platform, multi-phone, integrated-trial, wall-clock, and ADB readiness
gates and records exactly what is finished versus what remains blocked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "speculative_phone_worker_wrapup_fail_closed_no_speedup_claim"
SOURCE = "speculative_phone_worker_wrapup.py"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _dedupe(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _blocked_from(payload: dict[str, Any], *keys: str) -> list[str]:
    values: list[Any] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    return _dedupe(values)


def _status_for(*, ios_ready: bool, multi_phone_ready: bool, integrated_ready: bool, speedup: bool) -> str:
    if speedup and ios_ready and multi_phone_ready and integrated_ready:
        return "passed"
    if not ios_ready or not multi_phone_ready or not integrated_ready:
        return "blocked_by_missing_ios_and_multiphone_integrated_speedup"
    return "blocked_by_wallclock_speedup"


def build_speculative_phone_worker_wrapup_report(
    *,
    cross_platform_readiness: dict[str, Any],
    multi_phone_readiness: dict[str, Any],
    integrated_trial_plan: dict[str, Any],
    wallclock_gate: dict[str, Any],
    adb_preflight: dict[str, Any] | None = None,
    source_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    """Aggregate current speculative/phone gates into one conservative report."""
    adb_preflight = adb_preflight or {}

    android_ready = cross_platform_readiness.get("android_ready") is True
    ios_ready = cross_platform_readiness.get("ios_ready") is True
    cross_platform_ready = cross_platform_readiness.get("cross_platform_ready") is True
    phone_count = int(multi_phone_readiness.get("phone_count") or cross_platform_readiness.get("phone_count") or 0)
    ready_phone_count = int(
        multi_phone_readiness.get("ready_phone_count")
        or cross_platform_readiness.get("ready_phone_count")
        or 0
    )
    multi_phone_ready = multi_phone_readiness.get("trial_ready") is True
    integrated_ready = integrated_trial_plan.get("plan_status") == "ready_for_integrated_measurement"
    wallclock_speedup = wallclock_gate.get("wallclock_speedup_proven") is True or wallclock_gate.get("speedup_proven") is True

    completed: list[str] = []
    if android_ready and ready_phone_count >= 1:
        completed.append("android_phone_context_token_ingestion_ready")
    if wallclock_gate.get("verifier_acceptance_proven") is True and wallclock_gate.get("tokenizer_id_match_proven") is True:
        completed.append("single_phone_wallclock_correctness_gate_recorded")
    if integrated_trial_plan.get("claim_boundary") == "phone_speculative_integrated_trial_gate_harness_no_measurement":
        completed.append("integrated_trial_gate_harness_exists")

    blockers = _dedupe(
        [
            *_blocked_from(cross_platform_readiness, "blocked_reasons"),
            *_blocked_from(multi_phone_readiness, "blocked_reasons"),
            *_blocked_from(integrated_trial_plan, "blocked_reasons", "failed_checks"),
            *_blocked_from(wallclock_gate, "blocked_reasons", "failed_checks", "blocked_reason"),
            *_blocked_from(adb_preflight, "blocked_reasons", "failed_checks", "blocked_reason"),
        ]
    )

    speedup = bool(wallclock_speedup and multi_phone_ready and integrated_ready and cross_platform_ready)
    status = _status_for(
        ios_ready=ios_ready,
        multi_phone_ready=multi_phone_ready,
        integrated_ready=integrated_ready,
        speedup=speedup,
    )

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": status,
        "source_artifacts": list(source_artifacts),
        "android_ready": android_ready,
        "ios_ready": ios_ready,
        "cross_platform_ready": cross_platform_ready,
        "phone_count": phone_count,
        "ready_phone_count": ready_phone_count,
        "multi_phone_trial_ready": multi_phone_ready,
        "integrated_trial_plan_ready": integrated_ready,
        "wallclock_speedup_proven": False,
        "speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "can_update_demo_status": False,
        "completed_groundwork": completed,
        "remaining_blockers": blockers,
        "operator_next_steps": [
            "collect at least one iOS artifact with the same context-token and wall-clock correctness schema",
            "collect 3-4 distinct ready phone artifacts and rerun multi_phone_speculative_readiness.py",
            "run the integrated non-sequential phone draft-plus-verifier harness instead of sequential draft+verifier timing",
            "keep phone BloomBee block-worker status false until separate block-serving proof passes",
        ],
        "do_not_claim": [
            "no speculative speedup proof",
            "no cross-platform Android+iOS readiness",
            "no 3-4 phone trial readiness",
            "no BloomBee phone block-serving proof",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-platform-readiness", required=True)
    parser.add_argument("--multi-phone-readiness", required=True)
    parser.add_argument("--integrated-trial-plan", required=True)
    parser.add_argument("--wallclock-gate", required=True)
    parser.add_argument("--adb-preflight")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    paths = [
        args.cross_platform_readiness,
        args.multi_phone_readiness,
        args.integrated_trial_plan,
        args.wallclock_gate,
    ]
    if args.adb_preflight:
        paths.append(args.adb_preflight)
    payload = build_speculative_phone_worker_wrapup_report(
        cross_platform_readiness=_read_json(args.cross_platform_readiness),
        multi_phone_readiness=_read_json(args.multi_phone_readiness),
        integrated_trial_plan=_read_json(args.integrated_trial_plan),
        wallclock_gate=_read_json(args.wallclock_gate),
        adb_preflight=_read_json(args.adb_preflight) if args.adb_preflight else None,
        source_artifacts=paths,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
