#!/usr/bin/env python3
"""Verify KV prefix-reuse proof evidence.

This verifier is deliberately fail-closed. It does not enable BloomBee cache
reuse and it does not claim a live integration by itself; it only inspects a
captured evidence artifact for the proof contract Task 5 requires:

* at least two requests share the exact same prefix token IDs;
* those requests have varied suffix token IDs;
* the reuse path reports that it reused the shared prefix;
* generated token IDs and logits fingerprints match the no-reuse baseline; and
* per-request timing exists with a positive aggregate reuse speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

VERIFY_CLAIM_BOUNDARY = "verified_kv_prefix_reuse_same_prefix_varied_suffix_parity_timing"
PLAN_CLAIM_BOUNDARY = "kv_prefix_reuse_live_capture_plan_no_live_cache_reuse_proof"
PROOF_GATE = "kv_prefix_reuse"
REQUIRED_TELEMETRY_TAGS = ("kv_prefix_reuse", "no_reuse_baseline", "same_prefix_varied_suffix")
OPT_IN_FLAG = "BLOOMBEE_ENABLE_KV_PREFIX_REUSE"


def build_kv_prefix_reuse_live_capture_plan(
    *,
    model_id: str,
    evidence_path: str = ".local/kv-prefix-reuse-live-evidence.json",
    allow_no_speedup: bool = False,
    min_requests: int = 2,
) -> dict[str, Any]:
    """Return the next live capture contract without claiming reuse proof."""

    speedup_flag = " --allow-no-speedup" if allow_no_speedup else ""
    verify_command = (
        "python -m mvp_capabilities.kv_prefix_reuse_proof verify "
        f"--model {model_id} --evidence {evidence_path} --min-requests {min_requests}{speedup_flag}"
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "evidence_path": evidence_path,
        "opt_in_flag": OPT_IN_FLAG,
        "min_requests": min_requests,
        "operator_commands": [
            f"{OPT_IN_FLAG}=1 PYTHONPATH=.:src python scripts/text_generation_parity.py --server-maddr '<PASTE_SERVER_MULTIADDR>' --model {model_id} --prompt '<COMMON_PREFIX><SUFFIX_A>' --max-new-tokens 4 --mode generate-api --out .local/kv-prefix-baseline-suffix-a.json",
            f"{OPT_IN_FLAG}=1 PYTHONPATH=.:src python scripts/text_generation_parity.py --server-maddr '<PASTE_SERVER_MULTIADDR>' --model {model_id} --prompt '<COMMON_PREFIX><SUFFIX_B>' --max-new-tokens 4 --mode generate-api --out .local/kv-prefix-reuse-suffix-b.json",
            f"assemble {evidence_path} with common_prefix_token_ids, varied suffix_token_ids, no-reuse baseline rows, reuse rows, generated_token_ids, logits fingerprints, reused_prefix_token_count, and positive timings",
            verify_command,
        ],
        "verify_command": verify_command,
        "live_kv_cache_reuse_proven": False,
        "speedup_proven": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "notes": [
            "Planning output is not proof.",
            "The existing metadata path proves only first-rpc prefill metadata, not KV tensor reuse.",
            "Use --allow-no-speedup only for parity/timing capture; do not promote demo status without actual measured speedup and memory impact.",
        ],
    }


def _stable_sha256(value: Any) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


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


def _seconds(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seconds = float(value)
            if math.isfinite(seconds):
                return seconds
    return None


def _positive_seconds(mapping: Mapping[str, Any], *keys: str) -> float | None:
    seconds = _seconds(mapping, *keys)
    if seconds is None or seconds <= 0:
        return None
    return seconds


def _fingerprint(mapping: Mapping[str, Any]) -> str | None:
    for key in ("logits_sha256", "logits_hash", "logits_checksum", "logits_fingerprint"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    if "logits" in mapping:
        return _stable_sha256(mapping["logits"])
    return None


def _correctness_fingerprint(mapping: Mapping[str, Any]) -> str | None:
    for key in ("correctness_sha256", "correctness_hash", "output_sha256"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _reused_prefix_count(row: Mapping[str, Any], reuse: Mapping[str, Any]) -> int | None:
    for mapping in (reuse, row):
        value = mapping.get("reused_prefix_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 12)


def verify_kv_prefix_reuse_payload(
    payload: Mapping[str, Any],
    *,
    model_id: str | None = None,
    min_requests: int = 2,
    require_speedup: bool = True,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a parsed KV prefix-reuse evidence payload.

    The verifier derives correctness from raw evidence fields rather than from
    optimistic booleans. Missing fields are failures, not unknowns.
    """
    if min_requests <= 0:
        raise ValueError("min_requests must be positive")

    failed: list[str] = []
    requested_model = model_id
    evidence_model = payload.get("model_id") or payload.get("model")
    if requested_model is not None and evidence_model != requested_model:
        failed.append("evidence model mismatch")
    if evidence_model is None:
        failed.append("evidence model is missing")

    proof_gate = payload.get("proof_gate")
    if proof_gate not in (None, PROOF_GATE):
        failed.append("evidence proof_gate is not kv_prefix_reuse")

    if payload.get("prefix_reuse_enabled") is not True and payload.get("cache_reuse_enabled") is not True:
        failed.append("prefix/cache reuse path was not marked enabled")
    if payload.get("baseline_reuse_enabled") is not False and payload.get("no_reuse_baseline") is not True:
        failed.append("no-reuse baseline was not marked as the baseline path")

    telemetry_tags = payload.get("telemetry_tags")
    if not isinstance(telemetry_tags, list) or not all(isinstance(item, str) for item in telemetry_tags):
        failed.append("telemetry_tags must list kv_prefix_reuse proof tags")
        telemetry_tags_set: set[str] = set()
    else:
        telemetry_tags_set = set(telemetry_tags)
    missing_tags = [tag for tag in REQUIRED_TELEMETRY_TAGS if tag not in telemetry_tags_set]
    if missing_tags:
        failed.append("missing telemetry tags: " + ", ".join(missing_tags))

    request_rows = payload.get("requests") or payload.get("request_results")
    if not isinstance(request_rows, list):
        failed.append("requests must be a list")
        request_rows = []
    if len(request_rows) < min_requests:
        failed.append(f"expected at least {min_requests} same-prefix requests")

    common_prefix = _token_list(payload.get("common_prefix_token_ids"))
    if payload.get("common_prefix_token_ids") is not None and common_prefix is None:
        failed.append("common_prefix_token_ids must be a list of integer token IDs")

    prefixes: list[tuple[int, ...]] = []
    suffixes: list[tuple[int, ...]] = []
    row_summaries: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    reuse_event_count = 0
    baseline_total = 0.0
    reuse_total = 0.0
    timing_row_count = 0
    token_mismatch_seen = False
    logit_mismatch_seen = False
    timing_failure_seen = False

    for index, raw_row in enumerate(request_rows):
        if not isinstance(raw_row, Mapping):
            failed.append(f"request {index} must be an object")
            continue
        row: Mapping[str, Any] = raw_row
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = f"request-{index}"
        if request_id in seen_request_ids:
            failed.append(f"duplicate request_id: {request_id}")
        seen_request_ids.add(request_id)

        prefix = _first_token_list(row, ("prefix_token_ids", "prefix_ids", "common_prefix_token_ids"))
        if prefix is None and common_prefix is not None:
            prefix = list(common_prefix)
        suffix = _first_token_list(row, ("suffix_token_ids", "suffix_ids"))
        if prefix is None or not prefix:
            failed.append(f"request {index} prefix token IDs missing or empty")
            prefix = []
        if suffix is None or not suffix:
            failed.append(f"request {index} suffix token IDs missing or empty")
            suffix = []
        if prefix:
            prefixes.append(tuple(prefix))
            if common_prefix is not None and prefix != common_prefix:
                failed.append(f"request {index} prefix differs from common_prefix_token_ids")
        if suffix:
            suffixes.append(tuple(suffix))

        baseline = _first_mapping(row, ("baseline", "no_reuse", "baseline_no_reuse"))
        reuse = _first_mapping(row, ("reuse", "cache_reuse", "with_reuse"))
        if baseline is None:
            failed.append(f"request {index} no-reuse baseline object is missing")
            baseline = {}
        if reuse is None:
            failed.append(f"request {index} reuse object is missing")
            reuse = {}

        baseline_tokens = _first_token_list(
            baseline,
            ("generated_token_ids", "output_token_ids", "token_ids", "tokens"),
        )
        reuse_tokens = _first_token_list(
            reuse,
            ("generated_token_ids", "output_token_ids", "token_ids", "tokens"),
        )
        if baseline_tokens is None or not baseline_tokens:
            failed.append(f"request {index} baseline generated token IDs missing or empty")
        if reuse_tokens is None or not reuse_tokens:
            failed.append(f"request {index} reuse generated token IDs missing or empty")
        if baseline_tokens is not None and reuse_tokens is not None and baseline_tokens != reuse_tokens:
            failed.append(f"request {index} generated token IDs differ from no-reuse baseline")
            token_mismatch_seen = True
        if row.get("tokens_match") is False or row.get("token_ids_match") is False:
            failed.append(f"request {index} evidence explicitly reported token mismatch")
            token_mismatch_seen = True

        baseline_logits = _fingerprint(baseline)
        reuse_logits = _fingerprint(reuse)
        if baseline_logits is None:
            failed.append(f"request {index} baseline logits fingerprint missing")
        if reuse_logits is None:
            failed.append(f"request {index} reuse logits fingerprint missing")
        if baseline_logits is not None and reuse_logits is not None and baseline_logits != reuse_logits:
            failed.append(f"request {index} logits fingerprint differs from no-reuse baseline")
            logit_mismatch_seen = True
        if row.get("logits_match") is False:
            failed.append(f"request {index} evidence explicitly reported logits mismatch")
            logit_mismatch_seen = True

        baseline_correctness = _correctness_fingerprint(baseline)
        reuse_correctness = _correctness_fingerprint(reuse)
        if (
            baseline_correctness is not None
            and reuse_correctness is not None
            and baseline_correctness != reuse_correctness
        ):
            failed.append(f"request {index} correctness hash differs from no-reuse baseline")
            token_mismatch_seen = True

        prefix_count = _reused_prefix_count(row, reuse)
        if prefix_count is None or prefix_count <= 0:
            failed.append(f"request {index} reused_prefix_token_count missing or not positive")
        elif prefix and prefix_count < len(prefix):
            failed.append(f"request {index} reused fewer prefix tokens than the shared prefix length")
        else:
            reuse_event_count += 1

        baseline_seconds = _positive_seconds(baseline, "seconds", "elapsed_seconds", "wall_seconds")
        if baseline_seconds is None:
            baseline_seconds = _positive_seconds(row, "baseline_seconds", "no_reuse_seconds")
        reuse_seconds = _positive_seconds(reuse, "seconds", "elapsed_seconds", "wall_seconds")
        if reuse_seconds is None:
            reuse_seconds = _positive_seconds(row, "reuse_seconds", "cache_reuse_seconds")
        if baseline_seconds is None:
            failed.append(f"request {index} baseline seconds missing or not positive")
            timing_failure_seen = True
        if reuse_seconds is None:
            failed.append(f"request {index} reuse seconds missing or not positive")
            timing_failure_seen = True
        if baseline_seconds is not None and reuse_seconds is not None:
            baseline_total += baseline_seconds
            reuse_total += reuse_seconds
            timing_row_count += 1
            declared_delta = _seconds(row, "timing_delta_seconds", "delta_seconds", "saved_seconds")
            actual_delta = baseline_seconds - reuse_seconds
            if declared_delta is not None and abs(declared_delta - actual_delta) > 1e-6:
                failed.append(f"request {index} timing_delta_seconds does not match baseline-reuse timing")
                timing_failure_seen = True

        row_summaries.append(
            {
                "request_id": request_id,
                "prefix_sha256": _stable_sha256(prefix),
                "suffix_sha256": _stable_sha256(suffix),
                "baseline_tokens_sha256": _stable_sha256(baseline_tokens),
                "reuse_tokens_sha256": _stable_sha256(reuse_tokens),
                "baseline_logits": baseline_logits,
                "reuse_logits": reuse_logits,
                "baseline_seconds": _round_seconds(baseline_seconds),
                "reuse_seconds": _round_seconds(reuse_seconds),
                "reused_prefix_token_count": prefix_count,
            }
        )

    distinct_prefixes = set(prefixes)
    distinct_suffixes = set(suffixes)
    if len(distinct_prefixes) != 1:
        failed.append("requests did not share exactly one identical prefix")
    if len(distinct_suffixes) < min_requests:
        failed.append(f"expected at least {min_requests} distinct suffixes sharing the same prefix")
    if request_rows and reuse_event_count < len(request_rows):
        failed.append("not every request reported a reused prefix event")

    timing_measured = timing_row_count == len(request_rows) and not timing_failure_seen and bool(request_rows)
    total_delta = baseline_total - reuse_total if timing_measured else None
    speedup_proven = bool(timing_measured and total_delta is not None and total_delta > 0)
    if require_speedup and timing_measured and not speedup_proven:
        failed.append("reuse path was not faster than the no-reuse baseline")

    same_prefix_varied_suffix_proven = (
        len(request_rows) >= min_requests and len(distinct_prefixes) == 1 and len(distinct_suffixes) >= min_requests
    )
    token_parity_proven = bool(request_rows) and not token_mismatch_seen and not any(
        "generated token IDs" in check or "token mismatch" in check or "correctness hash differs" in check
        for check in failed
    )
    logit_parity_proven = bool(request_rows) and not logit_mismatch_seen and not any(
        "logits" in check for check in failed
    )

    prefix_for_summary: tuple[int, ...] | None = next(iter(distinct_prefixes)) if len(distinct_prefixes) == 1 else None
    summary = {
        "request_count": len(request_rows),
        "min_requests": min_requests,
        "prefix_token_count": len(prefix_for_summary) if prefix_for_summary is not None else 0,
        "prefix_sha256": _stable_sha256(list(prefix_for_summary)) if prefix_for_summary is not None else None,
        "distinct_suffix_count": len(distinct_suffixes),
        "suffix_sha256s": [_stable_sha256(list(suffix)) for suffix in sorted(distinct_suffixes)],
        "reuse_event_count": reuse_event_count,
        "baseline_total_seconds": _round_seconds(baseline_total) if timing_row_count else None,
        "reuse_total_seconds": _round_seconds(reuse_total) if timing_row_count else None,
        "timing_delta_seconds": _round_seconds(total_delta) if total_delta is not None else None,
        "speedup_ratio": _round_seconds(baseline_total / reuse_total) if timing_measured and reuse_total > 0 else None,
        "required_telemetry_tags": list(REQUIRED_TELEMETRY_TAGS),
        "observed_telemetry_tags": sorted(telemetry_tags_set),
        "request_summaries": row_summaries,
        "correctness_sha256": _stable_sha256(row_summaries),
    }

    status = "passed" if not failed else "failed"
    output_model = evidence_model if isinstance(evidence_model, str) else requested_model
    return {
        "model_id": output_model,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "status": status,
        "can_update_proof_status": status == "passed",
        "proof_status_update": {PROOF_GATE: "passed"} if status == "passed" else {},
        "can_update_mvp_status": False,
        "failed_checks": failed,
        "same_prefix_varied_suffix_proven": same_prefix_varied_suffix_proven,
        "token_parity_proven": token_parity_proven,
        "logit_parity_proven": logit_parity_proven,
        "timing_measured": timing_measured,
        "speedup_proven": speedup_proven,
        "evidence_path": str(evidence_path) if evidence_path is not None else None,
        "evidence_summary": summary,
        "claim_limitations": [
            "Verifier only; does not wire BloomBee runtime prefix-cache reuse.",
            "Passing evidence proves the supplied same-prefix/varied-suffix artifact, not broad production coverage.",
            "MVP-core status remains unchanged.",
        ],
    }


def verify_kv_prefix_reuse_evidence(
    *,
    evidence_path: str | Path,
    model_id: str | None = None,
    min_requests: int = 2,
    require_speedup: bool = True,
) -> dict[str, Any]:
    """Load and verify a KV prefix-reuse evidence JSON file."""
    path = Path(evidence_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return verify_kv_prefix_reuse_payload(
            {},
            model_id=model_id,
            min_requests=min_requests,
            require_speedup=require_speedup,
            evidence_path=path,
        ) | {"failed_checks": ["evidence root must be a JSON object"]}
    return verify_kv_prefix_reuse_payload(
        payload,
        model_id=model_id,
        min_requests=min_requests,
        require_speedup=require_speedup,
        evidence_path=path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a claim-bounded live capture plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--evidence", default=".local/kv-prefix-reuse-live-evidence.json")
    plan.add_argument("--min-requests", type=int, default=2)
    plan.add_argument(
        "--allow-no-speedup",
        action="store_true",
        help="Plan for parity/timing capture without requiring positive aggregate speedup.",
    )
    plan.add_argument("--out", default=None)

    verify = sub.add_parser("verify", help="Verify captured KV prefix-reuse parity/timing evidence")
    verify.add_argument("--model", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--min-requests", type=int, default=2)
    verify.add_argument(
        "--allow-no-speedup",
        action="store_true",
        help="Check parity and timing presence without requiring positive aggregate speedup.",
    )

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_kv_prefix_reuse_live_capture_plan(
            model_id=args.model,
            evidence_path=args.evidence,
            min_requests=args.min_requests,
            allow_no_speedup=args.allow_no_speedup,
        )
    else:
        payload = verify_kv_prefix_reuse_evidence(
            evidence_path=args.evidence,
            model_id=args.model,
            min_requests=args.min_requests,
            require_speedup=not args.allow_no_speedup,
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
