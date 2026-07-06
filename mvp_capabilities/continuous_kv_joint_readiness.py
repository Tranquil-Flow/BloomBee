#!/usr/bin/env python3
"""Joint readiness gate for continuous batching plus KV prefix reuse.

The two features can each have partial evidence, but a useful post-MVP gate needs
both at once:
- live-server late-arrival continuous-batching parity, and
- explicit server-observed KV tensor/cache reuse, not metadata-only prefill rows.

This verifier consumes verifier reports, not raw model outputs. It deliberately
keeps demo and wall-clock speedup promotion false; it only answers whether both
post-MVP feature gates are simultaneously ready to be treated as passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

CLAIM_BOUNDARY = "continuous_kv_joint_readiness_gate_no_demo_or_wallclock_speedup_claim"
SOURCE = "continuous_kv_joint_readiness.py"
CONTINUOUS_GATE = "continuous_batching"
KV_GATE = "kv_prefix_reuse"
JOINT_GATE = "continuous_kv_joint_readiness"


def _failed_checks(report: Mapping[str, Any]) -> list[str]:
    raw = report.get("failed_checks") or report.get("blocked_reasons") or []
    if not isinstance(raw, list):
        return [str(raw)]
    return [str(item) for item in raw]


def _model_id(*reports: Mapping[str, Any]) -> str | None:
    values = []
    for report in reports:
        value = report.get("model_id") or report.get("model")
        if isinstance(value, str) and value:
            values.append(value)
    return values[0] if values else None


def _continuous_ready(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    blocked: list[str] = []
    if report.get("proof_gate") != CONTINUOUS_GATE:
        blocked.append("continuous_batching:proof_gate_mismatch")
    if report.get("status") != "passed":
        blocked.append("continuous_batching:verifier_status_not_passed")
    if report.get("live_server_late_arrival_parity_proven") is not True:
        blocked.append("continuous_batching:late_arrival_parity_not_proven")
    if report.get("late_arrival_observed") is not True:
        blocked.append("continuous_batching:late_arrival_not_observed")
    if not isinstance(report.get("batched_tick_count"), int) or report.get("batched_tick_count", 0) <= 0:
        blocked.append("continuous_batching:no_batched_tick_observed")
    if report.get("token_parity_proven") is not True:
        blocked.append("continuous_batching:token_parity_not_proven")
    if report.get("logits_fingerprint_parity_proven") is not True:
        blocked.append("continuous_batching:logits_parity_not_proven")
    blocked.extend(f"continuous_batching:{check}" for check in _failed_checks(report))
    return not blocked, blocked


def _kv_ready(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    blocked: list[str] = []
    if report.get("proof_gate") != KV_GATE:
        blocked.append("kv_prefix_reuse:proof_gate_mismatch")
    if report.get("status") != "passed":
        blocked.append("kv_prefix_reuse:verifier_status_not_passed")
    if report.get("same_prefix_varied_suffix_proven") is not True:
        blocked.append("kv_prefix_reuse:same_prefix_varied_suffix_not_proven")
    if report.get("token_parity_proven") is not True:
        blocked.append("kv_prefix_reuse:token_parity_not_proven")
    if report.get("logit_parity_proven") is not True:
        blocked.append("kv_prefix_reuse:logit_parity_not_proven")
    if report.get("timing_measured") is not True:
        blocked.append("kv_prefix_reuse:timing_not_measured")
    live_reuse = report.get("live_kv_cache_reuse_proven") is True and report.get("server_observed_kv_cache_reuse") is True
    if not live_reuse:
        blocked.append("kv_prefix_reuse:live_server_kv_tensor_reuse_not_proven")
    blocked.extend(f"kv_prefix_reuse:{check}" for check in _failed_checks(report))
    return not blocked, blocked


def build_continuous_kv_joint_readiness_report(
    *,
    continuous_report: Mapping[str, Any],
    kv_report: Mapping[str, Any],
) -> dict[str, Any]:
    continuous_ok, continuous_blocked = _continuous_ready(continuous_report)
    kv_ok, kv_blocked = _kv_ready(kv_report)
    blocked = continuous_blocked + kv_blocked
    model = _model_id(continuous_report, kv_report)
    other_model = _model_id(kv_report)
    if model and other_model and model != other_model:
        blocked.append("model_id_mismatch")

    passed = not blocked
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_id": model,
        "verification_status": "passed" if passed else "failed",
        "continuous_batching_ready": continuous_ok,
        "kv_prefix_reuse_ready": kv_ok,
        "joint_batch_kv_ready": passed,
        "blocked_reasons": blocked,
        "continuous_batching": {
            "claim_boundary": continuous_report.get("claim_boundary"),
            "status": continuous_report.get("status"),
            "late_arrival_observed": continuous_report.get("late_arrival_observed"),
            "batched_tick_count": continuous_report.get("batched_tick_count"),
            "live_server_late_arrival_parity_proven": continuous_report.get("live_server_late_arrival_parity_proven"),
        },
        "kv_prefix_reuse": {
            "claim_boundary": kv_report.get("claim_boundary"),
            "status": kv_report.get("status"),
            "same_prefix_varied_suffix_proven": kv_report.get("same_prefix_varied_suffix_proven"),
            "timing_measured": kv_report.get("timing_measured"),
            "speedup_proven": kv_report.get("speedup_proven") is True,
            "live_kv_cache_reuse_proven": kv_report.get("live_kv_cache_reuse_proven") is True,
            "server_observed_kv_cache_reuse": kv_report.get("server_observed_kv_cache_reuse") is True,
        },
        "wallclock_speedup_proven": False,
        "speedup_proven": False,
        "can_update_post_mvp_gate_status": passed,
        "proof_status_update": {
            CONTINUOUS_GATE: "passed",
            KV_GATE: "passed",
            JOINT_GATE: "passed",
        } if passed else {},
        "can_update_demo_status": False,
        "can_update_mvp_status": False,
        "claim_limitations": [
            "Joint verifier only; it does not itself execute live traffic.",
            "KV prefix reuse must be server-observed tensor/cache reuse; metadata-only reports fail closed.",
            "Wall-clock speedup and demo-safety promotion remain separate gates.",
        ],
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-report", required=True)
    parser.add_argument("--kv-report", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_continuous_kv_joint_readiness_report(
        continuous_report=_read_json(args.continuous_report),
        kv_report=_read_json(args.kv_report),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["verification_status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
