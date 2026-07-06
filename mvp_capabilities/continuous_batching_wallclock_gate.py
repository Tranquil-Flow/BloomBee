#!/usr/bin/env python3
"""Verify continuous-batching wall-clock speedup evidence.

This gate is intentionally downstream of
``continuous_batching_live_server_proof.py``. A timing artifact may only pass if
it carries a passed live-server late-arrival parity verification; positive timing
alone is not proof. Passing this verifier still does not promote demo status by
itself because the artifact must be reviewed as the specific live run it
represents.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

VERIFY_CLAIM_BOUNDARY = "verified_live_continuous_batching_wallclock_speedup_after_parity"
PLAN_CLAIM_BOUNDARY = "live_continuous_batching_wallclock_gate_harness_no_measurement"
CAPTURE_CLAIM_BOUNDARY = "live_continuous_batching_wallclock_capture_candidate"
PARITY_CLAIM_BOUNDARY = "verified_live_continuous_batching_server_concurrent_arrival_parity"
PROOF_GATE = "continuous_batching"
SPEEDUP_PROOF_KEY = "continuous_batching_wallclock_speedup"
REQUIRED_TELEMETRY_TAGS = (
    "continuous_batching",
    "live_server_late_arrival_parity",
    "wallclock_throughput",
)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 12)


def _positive_seconds(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seconds = float(value)
            if math.isfinite(seconds) and seconds > 0:
                return seconds
    return None


def _positive_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _parity_proven(parity: Mapping[str, Any], failed: list[str]) -> bool:
    proven = True
    if parity.get("claim_boundary") != PARITY_CLAIM_BOUNDARY:
        failed.append("parity verification claim_boundary is not the live-server parity verifier")
        proven = False
    if parity.get("proof_gate") != PROOF_GATE:
        failed.append("parity verification proof_gate is not continuous_batching")
        proven = False
    if parity.get("status") != "passed":
        failed.append("live-server parity proof did not pass")
        proven = False
    required_true = {
        "live_server_late_arrival_parity_proven": "live-server late-arrival parity was not proven",
        "live_server_proven": "live-server execution was not proven",
        "late_arrival_observed": "late-arrival request was not observed",
        "token_parity_proven": "token parity was not proven",
        "logits_fingerprint_parity_proven": "logits fingerprint parity was not proven",
    }
    for key, message in required_true.items():
        if parity.get(key) is not True:
            failed.append(message)
            proven = False
    batched_tick_count = parity.get("batched_tick_count")
    if not isinstance(batched_tick_count, int) or isinstance(batched_tick_count, bool) or batched_tick_count <= 0:
        failed.append("batched live-continuous tick was not observed")
        proven = False
    return proven


def build_continuous_batching_wallclock_plan(
    *,
    model_id: str,
    evidence_path: str = ".local/continuous-batching-wallclock-evidence.json",
    parity_report_path: str = ".local/continuous-parity.verify.json",
) -> dict[str, Any]:
    """Return an operator recipe for the wall-clock gate without claiming proof."""

    verify_command = (
        "python mvp_capabilities/continuous_batching_wallclock_gate.py verify "
        f"--model {model_id} --evidence {evidence_path}"
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "evidence_path": evidence_path,
        "parity_report_path": parity_report_path,
        "required_telemetry_tags": list(REQUIRED_TELEMETRY_TAGS),
        "operator_commands": [
            "python mvp_capabilities/continuous_batching_live_server_proof.py verify "
            f"--model {model_id} --evidence .local/live-continuous-batching-capture.json "
            f"--out {parity_report_path}",
            "capture serial-baseline total_seconds/request_count/generated_token_count for the exact same live requests",
            "capture opt-in live-continuous total_seconds/request_count/generated_token_count without hand-editing outputs",
            f"assemble {evidence_path} with claim_boundary={CAPTURE_CLAIM_BOUNDARY}, the parity_verification JSON, baseline, continuous, and telemetry_tags",
            verify_command,
        ],
        "verify_command": verify_command,
        "parity_required": True,
        "wallclock_speedup_proven": False,
        "speedup_proven": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "notes": [
            "Planning output is not measurement proof.",
            "Positive timing is rejected unless the live-server parity verifier passed first.",
            "Demo promotion remains false; review the exact live-run artifact before changing user-facing status.",
        ],
    }


def verify_continuous_batching_wallclock_payload(
    payload: Mapping[str, Any],
    *,
    model_id: str | None = None,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify wall-clock speedup evidence after a live parity proof."""

    failed: list[str] = []
    evidence_model = payload.get("model_id") or payload.get("model")
    if model_id is not None and evidence_model != model_id:
        failed.append("evidence model mismatch")
    if evidence_model is None:
        failed.append("evidence model is missing")

    if payload.get("proof_gate") != PROOF_GATE:
        failed.append("evidence proof_gate is not continuous_batching")
    claim_boundary = payload.get("claim_boundary")
    if claim_boundary not in (None, CAPTURE_CLAIM_BOUNDARY):
        failed.append("evidence claim_boundary is not a continuous batching wall-clock capture candidate")

    telemetry_tags = payload.get("telemetry_tags")
    if not isinstance(telemetry_tags, list) or not all(isinstance(item, str) for item in telemetry_tags):
        failed.append("telemetry_tags must list continuous batching wall-clock proof tags")
        telemetry_tag_set: set[str] = set()
    else:
        telemetry_tag_set = set(telemetry_tags)
    missing_tags = [tag for tag in REQUIRED_TELEMETRY_TAGS if tag not in telemetry_tag_set]
    if missing_tags:
        failed.append("missing telemetry tags: " + ", ".join(missing_tags))

    parity = _mapping(payload.get("parity_verification"))
    if parity is None:
        failed.append("parity_verification must be an object from the live-server parity verifier")
        parity = {}
    parity_ok = _parity_proven(parity, failed)

    baseline = _mapping(payload.get("baseline") or payload.get("serial_baseline"))
    continuous = _mapping(payload.get("continuous") or payload.get("live_continuous"))
    if baseline is None:
        failed.append("baseline timing object is missing")
        baseline = {}
    if continuous is None:
        failed.append("continuous timing object is missing")
        continuous = {}

    baseline_seconds = _positive_seconds(baseline, "total_seconds", "wall_seconds", "elapsed_seconds", "seconds")
    continuous_seconds = _positive_seconds(continuous, "total_seconds", "wall_seconds", "elapsed_seconds", "seconds")
    if baseline_seconds is None:
        failed.append("baseline total_seconds missing or not positive")
    if continuous_seconds is None:
        failed.append("continuous total_seconds missing or not positive")

    baseline_requests = _positive_int(baseline, "request_count", "requests")
    continuous_requests = _positive_int(continuous, "request_count", "requests")
    parity_requests = _positive_int(parity, "request_count", "requests")
    if baseline_requests is None:
        failed.append("baseline request_count missing or not positive")
    if continuous_requests is None:
        failed.append("continuous request_count missing or not positive")
    if baseline_requests is not None and continuous_requests is not None and baseline_requests != continuous_requests:
        failed.append("request_count differs between baseline and continuous paths")
    if parity_requests is not None and baseline_requests is not None and parity_requests != baseline_requests:
        failed.append("baseline request_count differs from parity verification request_count")

    baseline_tokens = _positive_int(baseline, "generated_token_count", "token_count", "output_token_count")
    continuous_tokens = _positive_int(continuous, "generated_token_count", "token_count", "output_token_count")
    if baseline_tokens is None:
        failed.append("baseline generated_token_count missing or not positive")
    if continuous_tokens is None:
        failed.append("continuous generated_token_count missing or not positive")
    if baseline_tokens is not None and continuous_tokens is not None and baseline_tokens != continuous_tokens:
        failed.append("generated_token_count differs between baseline and continuous paths")

    baseline_throughput: float | None = None
    continuous_throughput: float | None = None
    if baseline_seconds is not None and baseline_tokens is not None:
        baseline_throughput = baseline_tokens / baseline_seconds
    if continuous_seconds is not None and continuous_tokens is not None:
        continuous_throughput = continuous_tokens / continuous_seconds

    timing_speedup = False
    timing_delta: float | None = None
    speedup_ratio: float | None = None
    throughput_ratio: float | None = None
    if baseline_seconds is not None and continuous_seconds is not None:
        timing_delta = baseline_seconds - continuous_seconds
        timing_speedup = timing_delta > 0
        speedup_ratio = baseline_seconds / continuous_seconds
        if not timing_speedup:
            failed.append("continuous path was not faster than serial baseline")
    if baseline_throughput is not None and continuous_throughput is not None and baseline_throughput > 0:
        throughput_ratio = continuous_throughput / baseline_throughput

    status = "passed" if not failed else "failed"
    speedup_proven = status == "passed" and parity_ok and timing_speedup
    summary = {
        "baseline_total_seconds": _round(baseline_seconds),
        "continuous_total_seconds": _round(continuous_seconds),
        "timing_delta_seconds": _round(timing_delta),
        "baseline_generated_token_count": baseline_tokens,
        "continuous_generated_token_count": continuous_tokens,
        "baseline_request_count": baseline_requests,
        "continuous_request_count": continuous_requests,
        "baseline_tokens_per_second": _round(baseline_throughput),
        "continuous_tokens_per_second": _round(continuous_throughput),
        "speedup_ratio": _round(speedup_ratio),
        "throughput_ratio": _round(throughput_ratio),
        "required_telemetry_tags": list(REQUIRED_TELEMETRY_TAGS),
        "observed_telemetry_tags": sorted(telemetry_tag_set),
    }
    output_model = evidence_model if isinstance(evidence_model, str) else model_id
    return {
        "model_id": output_model,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "status": status,
        "evidence_path": str(evidence_path) if evidence_path is not None else None,
        "failed_checks": failed,
        "parity_proven": parity_ok and status == "passed",
        "timing_speedup_measured": timing_speedup and baseline_seconds is not None and continuous_seconds is not None,
        "wallclock_speedup_proven": speedup_proven,
        "speedup_proven": speedup_proven,
        "can_update_proof_status": speedup_proven,
        "proof_status_update": {SPEEDUP_PROOF_KEY: "passed"} if speedup_proven else {},
        "can_update_mvp_status": False,
        "can_update_demo_status": False,
        "evidence_summary": summary,
        "claim_limitations": [
            "Verifier only; requires a real captured live-server parity report and matching wall-clock timing evidence.",
            "Passing evidence proves only the supplied live-run artifact, not broad production coverage.",
            "Demo status remains unchanged by this verifier alone.",
        ],
    }


def verify_continuous_batching_wallclock_evidence(
    *,
    evidence_path: str | Path,
    model_id: str | None = None,
) -> dict[str, Any]:
    path = Path(evidence_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return verify_continuous_batching_wallclock_payload(
            {},
            model_id=model_id,
            evidence_path=path,
        ) | {"failed_checks": ["evidence root must be a JSON object"]}
    return verify_continuous_batching_wallclock_payload(payload, model_id=model_id, evidence_path=path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a claim-bounded wall-clock proof recipe")
    plan.add_argument("--model", required=True)
    plan.add_argument("--evidence", default=".local/continuous-batching-wallclock-evidence.json")
    plan.add_argument("--parity-report", default=".local/continuous-parity.verify.json")
    plan.add_argument("--out", default=None)

    verify = sub.add_parser("verify", help="Verify captured wall-clock speedup evidence")
    verify.add_argument("--model", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_continuous_batching_wallclock_plan(
            model_id=args.model,
            evidence_path=args.evidence,
            parity_report_path=args.parity_report,
        )
        exit_code = 0
    else:
        payload = verify_continuous_batching_wallclock_evidence(
            evidence_path=args.evidence,
            model_id=args.model,
        )
        exit_code = 0 if payload.get("status") == "passed" else 1

    text = json.dumps(payload, indent=2, sort_keys=True)
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
