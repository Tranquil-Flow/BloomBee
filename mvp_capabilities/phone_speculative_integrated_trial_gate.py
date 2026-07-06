#!/usr/bin/env python3
"""Fail-closed gate for a multi-phone integrated speculative trial.

This verifier is the claim boundary after ``multi_phone_speculative_readiness``:
it accepts only a passed 3-4 phone readiness manifest plus a measured
non-sequential draft-plus-verifier wall-clock run. It is a verifier/gate, not a
phone transport implementation and not BloomBee phone block-serving proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

SOURCE = "phone_speculative_integrated_trial_gate.py"
VERIFY_CLAIM_BOUNDARY = "phone_speculative_integrated_trial_gate_verified_measurement"
PLAN_CLAIM_BOUNDARY = "phone_speculative_integrated_trial_gate_harness_no_measurement"
READINESS_CLAIM_BOUNDARY = "multi_phone_speculative_readiness_manifest_no_speedup_claim"
DEFAULT_MEASUREMENT_KIND = "measured_integrated_non_sequential_draft_plus_verifier"
SPEEDUP_PROOF_KEY = "phone_speculative_integrated_speedup"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _round_s(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _positive_seconds(value: float | int | None, *, field: str, failed: list[str]) -> float | None:
    if isinstance(value, bool) or value is None:
        failed.append(f"{field}_missing_or_not_positive")
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        failed.append(f"{field}_missing_or_not_positive")
        return None
    return seconds


def _ready_phones(readiness_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    phones = readiness_manifest.get("phones")
    if not isinstance(phones, list):
        return []
    return [phone for phone in phones if isinstance(phone, Mapping) and phone.get("ready_for_trial") is True]


def _readiness_failed_checks(readiness_manifest: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    if readiness_manifest.get("claim_boundary") != READINESS_CLAIM_BOUNDARY:
        failed.append("readiness_manifest_claim_boundary_mismatch")
    if readiness_manifest.get("verification_status") != "passed" or readiness_manifest.get("trial_ready") is not True:
        failed.append("readiness_manifest_not_passed")
    if readiness_manifest.get("speedup_proven") is not False:
        failed.append("readiness_manifest_already_claims_speedup")
    if readiness_manifest.get("can_update_speculative_speedup_status") is not False:
        failed.append("readiness_manifest_already_allows_speedup_status_update")
    if readiness_manifest.get("bloombee_block_serving_proven") is True:
        failed.append("readiness_manifest_unexpected_bloombee_block_worker_claim")

    min_phone_count = int(readiness_manifest.get("min_phone_count") or 3)
    max_phone_count = int(readiness_manifest.get("max_phone_count") or 4)
    phone_count = int(readiness_manifest.get("phone_count") or 0)
    ready_count = len(_ready_phones(readiness_manifest))
    if phone_count < min_phone_count:
        failed.append(f"phone_count_below_min:{phone_count}<{min_phone_count}")
    if phone_count > max_phone_count:
        failed.append(f"phone_count_above_max:{phone_count}>{max_phone_count}")
    if ready_count < min_phone_count:
        failed.append(f"ready_phone_count_below_min:{ready_count}<{min_phone_count}")
    if readiness_manifest.get("all_context_token_ingestion_proven") is not True:
        failed.append("all_context_token_ingestion_not_proven")
    if readiness_manifest.get("all_wallclock_correctness_proven") is not True:
        failed.append("all_wallclock_correctness_not_proven")

    for reason in readiness_manifest.get("blocked_reasons") or []:
        text = str(reason)
        if text not in failed:
            failed.append(text)
    return failed


def build_phone_speculative_integrated_trial_gate(
    *,
    readiness_manifest: Mapping[str, Any],
    verifier_only_elapsed_s: float | int | None,
    integrated_draft_plus_verifier_elapsed_s: float | int | None,
    measurement_kind: str = DEFAULT_MEASUREMENT_KIND,
    source_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify a measured integrated phone draft-plus-verifier timing artifact."""

    failed = _readiness_failed_checks(readiness_manifest)
    readiness_passed = not failed
    verifier_seconds = _positive_seconds(verifier_only_elapsed_s, field="verifier_only_elapsed_s", failed=failed)
    integrated_seconds = _positive_seconds(
        integrated_draft_plus_verifier_elapsed_s,
        field="integrated_draft_plus_verifier_elapsed_s",
        failed=failed,
    )

    if measurement_kind != DEFAULT_MEASUREMENT_KIND:
        failed.append("measurement_kind_not_integrated_non_sequential")

    timing_speedup = False
    speedup_ratio: float | None = None
    timing_delta: float | None = None
    if verifier_seconds is not None and integrated_seconds is not None:
        timing_delta = verifier_seconds - integrated_seconds
        timing_speedup = integrated_seconds < verifier_seconds
        speedup_ratio = verifier_seconds / integrated_seconds
        if not timing_speedup:
            failed.append("integrated_draft_plus_verifier_not_faster_than_verifier_only")

    status = "passed" if not failed else "failed"
    speedup_proven = status == "passed" and readiness_passed and timing_speedup
    selected_phone_ids = [str(phone["phone_id"]) for phone in _ready_phones(readiness_manifest)]
    return {
        "source": SOURCE,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "status": status,
        "readiness_claim_boundary": readiness_manifest.get("claim_boundary"),
        "readiness_passed": readiness_passed,
        "model_sha256": readiness_manifest.get("model_sha256"),
        "phone_count": int(readiness_manifest.get("phone_count") or 0),
        "ready_phone_count": len(selected_phone_ids),
        "selected_phone_ids": selected_phone_ids if speedup_proven or readiness_passed else [],
        "measurement_kind": measurement_kind,
        "verifier_only_elapsed_s": _round_s(verifier_seconds),
        "integrated_draft_plus_verifier_elapsed_s": _round_s(integrated_seconds),
        "timing_delta_s": _round_s(timing_delta),
        "speedup_ratio": _round_s(speedup_ratio),
        "timing_speedup_measured": timing_speedup,
        "wallclock_speedup_proven": speedup_proven,
        "speedup_proven": speedup_proven,
        "can_update_speculative_speedup_status": speedup_proven,
        "proof_status_update": {SPEEDUP_PROOF_KEY: "passed"} if speedup_proven else {},
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "source_artifacts": list(source_artifacts),
        "failed_checks": failed,
        "claim_limitations": [
            "Verifier only; it requires real 3-4 phone readiness evidence plus measured integrated timing.",
            "Sequential phone-draft-then-verifier timings are rejected as speculative-speedup proof.",
            "Passing this gate does not prove BloomBee phone block serving.",
        ],
        "operator_next_steps": [
            "run the actual non-sequential draft-plus-verifier harness with live phone token inputs",
            "preserve verifier-only and integrated timing logs as source artifacts",
            "keep phone block-worker status false until separate BloomBee block-serving evidence passes",
        ],
    }


def build_phone_speculative_integrated_trial_gate_plan(
    *,
    readiness_manifest: Mapping[str, Any],
    readiness_manifest_path: str,
    evidence_path: str = ".local/phone/integrated-trial-gate.json",
) -> dict[str, Any]:
    """Return a claim-bounded operator recipe for the integrated timing gate."""

    failed = _readiness_failed_checks(readiness_manifest)
    plan_ready = not failed
    verify_command = (
        "python mvp_capabilities/phone_speculative_integrated_trial_gate.py verify "
        f"--readiness-manifest {readiness_manifest_path} "
        "--verifier-only-elapsed-s $VERIFIER_ONLY_ELAPSED_S "
        "--integrated-draft-plus-verifier-elapsed-s $INTEGRATED_DRAFT_PLUS_VERIFIER_ELAPSED_S "
        f"--out {evidence_path}"
    )
    return {
        "source": SOURCE,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "plan_status": "ready_for_integrated_measurement" if plan_ready else "blocked_by_readiness_manifest",
        "readiness_claim_boundary": readiness_manifest.get("claim_boundary"),
        "readiness_manifest_path": readiness_manifest_path,
        "evidence_path": evidence_path,
        "selected_phone_ids": [str(phone["phone_id"]) for phone in _ready_phones(readiness_manifest)] if plan_ready else [],
        "measurement_kind": DEFAULT_MEASUREMENT_KIND,
        "verify_command": verify_command,
        "operator_commands": [
            "run the verifier-only baseline and save the raw timing log",
            "run the integrated non-sequential phone draft-plus-verifier path and save the raw timing log",
            verify_command,
        ],
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": failed,
        "notes": [
            "Planning output is not proof.",
            "The verify command exits non-zero unless readiness is passed and integrated timing beats verifier-only timing.",
            "BloomBee phone block-worker status remains a separate proof gate.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a claim-bounded integrated trial gate plan")
    plan.add_argument("--readiness-manifest", required=True)
    plan.add_argument("--evidence", default=".local/phone/integrated-trial-gate.json")
    plan.add_argument("--out", default=None)

    verify = sub.add_parser("verify", help="Verify integrated phone speculative timing evidence")
    verify.add_argument("--readiness-manifest", required=True)
    verify.add_argument("--verifier-only-elapsed-s", type=float, required=True)
    verify.add_argument("--integrated-draft-plus-verifier-elapsed-s", type=float, required=True)
    verify.add_argument("--measurement-kind", default=DEFAULT_MEASUREMENT_KIND)
    verify.add_argument("--source-artifact", action="append", default=[])
    verify.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    readiness_path = Path(args.readiness_manifest)
    readiness_manifest = _read_json(readiness_path)
    if args.command == "plan":
        payload = build_phone_speculative_integrated_trial_gate_plan(
            readiness_manifest=readiness_manifest,
            readiness_manifest_path=str(readiness_path),
            evidence_path=args.evidence,
        )
        exit_code = 0
    else:
        payload = build_phone_speculative_integrated_trial_gate(
            readiness_manifest=readiness_manifest,
            verifier_only_elapsed_s=args.verifier_only_elapsed_s,
            integrated_draft_plus_verifier_elapsed_s=args.integrated_draft_plus_verifier_elapsed_s,
            measurement_kind=args.measurement_kind,
            source_artifacts=[str(readiness_path), *args.source_artifact],
        )
        exit_code = 0 if payload["status"] == "passed" else 1

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
