#!/usr/bin/env python3
"""Aggregate 3-4 phone speculative-decoding readiness artifacts.

This is a tomorrow-test manifest checker, not a speedup proof. It validates that
multiple phones have the prerequisite draft-token evidence needed to attempt an
integrated speculative decoding wall-clock test, while keeping all route/status
promotion flags false until a real draft-plus-verifier path is measured.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "multi_phone_speculative_readiness_manifest_no_speedup_claim"
SOURCE = "multi_phone_speculative_readiness.py"
DEFAULT_MIN_PHONE_COUNT = 3
DEFAULT_MAX_PHONE_COUNT = 4


def _as_bool(value: Any) -> bool:
    return bool(value) is True


def _as_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _phone_id(artifact: dict[str, Any], index: int) -> str:
    for key in ("phone_id", "device_id", "peer_id"):
        value = artifact.get(key)
        if value:
            return str(value)
    runtime = artifact.get("phone_runtime") or artifact.get("runtime") or {}
    if isinstance(runtime, dict):
        for key in ("phone_id", "device_id", "serial", "model"):
            value = runtime.get(key)
            if value:
                return str(value)
    return f"phone-{index + 1}"


def _context_report(artifact: dict[str, Any]) -> dict[str, Any]:
    for key in ("context_token_verifier", "token_verifier", "phone_context_token_verifier"):
        value = artifact.get(key)
        if isinstance(value, dict):
            return value
    if artifact.get("claim_boundary") == "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim":
        return artifact
    return {}


def _wallclock_report(artifact: dict[str, Any]) -> dict[str, Any]:
    for key in ("wallclock_gate", "speedup_gate", "phone_speculative_wallclock_gate"):
        value = artifact.get(key)
        if isinstance(value, dict):
            return value
    if artifact.get("claim_boundary") == "phone_speculative_wallclock_gate_fail_closed":
        return artifact
    return {}


def _model_sha(context: dict[str, Any], artifact: dict[str, Any]) -> str | None:
    value = context.get("model_sha256") or artifact.get("model_sha256")
    if value:
        return str(value)
    model = context.get("model") or artifact.get("model")
    if isinstance(model, dict):
        value = model.get("sha256") or model.get("model_sha256")
        if value:
            return str(value)
    return None


def _runtime_dict(artifact: dict[str, Any]) -> dict[str, Any]:
    runtime = artifact.get("phone_runtime")
    return runtime if isinstance(runtime, dict) else {}


def _phone_model(artifact: dict[str, Any]) -> Any:
    runtime = _runtime_dict(artifact)
    return artifact.get("phone_model") or runtime.get("android_model") or runtime.get("model")


def _runtime_name(artifact: dict[str, Any]) -> Any:
    runtime = _runtime_dict(artifact)
    return artifact.get("runtime") or runtime.get("runtime")


def _phone_summary(artifact: dict[str, Any], index: int) -> tuple[dict[str, Any], list[str]]:
    phone_id = _phone_id(artifact, index)
    context = _context_report(artifact)
    wallclock = _wallclock_report(artifact)
    blocked: list[str] = []

    external_ingested = _as_bool(context.get("phone_external_token_ids_ingested"))
    integrated = _as_bool(context.get("phone_integrated_verifier_proven"))
    acceptance = _as_bool(context.get("external_context_token_id_acceptance_proven"))
    accepted_count = _as_positive_int(
        context.get("accepted_external_token_count")
        or context.get("accepted_context_token_count")
        or context.get("accepted_generated_token_count")
    )
    proposed_count = _as_positive_int(
        context.get("proposed_external_token_count")
        or len(context.get("phone_context_draft_token_ids") or [])
        or len(context.get("accepted_context_token_ids") or [])
    )
    context_ok = bool(external_ingested and integrated and acceptance and accepted_count > 0)
    if not context_ok:
        blocked.append(f"phone:{phone_id}:context_token_ingestion_not_proven")

    wallclock_acceptance = _as_bool(wallclock.get("verifier_acceptance_proven"))
    tokenizer_match = _as_bool(wallclock.get("tokenizer_id_match_proven"))
    wallclock_ok = bool(wallclock and wallclock_acceptance and tokenizer_match)
    if not wallclock_ok:
        blocked.append(f"phone:{phone_id}:wallclock_correctness_not_proven")

    context_speedup = _as_bool(context.get("speedup_proven")) or _as_bool(
        context.get("can_update_speculative_speedup_status")
    )
    wallclock_speedup = _as_bool(wallclock.get("speedup_proven")) or _as_bool(
        wallclock.get("wallclock_speedup_proven")
    ) or _as_bool(wallclock.get("can_update_speculative_speedup_status"))
    if context_speedup or wallclock_speedup:
        blocked.append(f"phone:{phone_id}:unexpected_speedup_claim")

    model_sha = _model_sha(context, artifact)
    if not model_sha:
        blocked.append(f"phone:{phone_id}:model_sha256_missing")

    summary = {
        "phone_id": phone_id,
        "phone_model": _phone_model(artifact),
        "runtime": _runtime_name(artifact),
        "transport_path": artifact.get("transport_path") or context.get("transport_path"),
        "model_sha256": model_sha,
        "termux_context_token_artifact": artifact.get("termux_context_token_artifact") or context.get("phone_token_json_artifact"),
        "accepted_external_token_count": accepted_count,
        "proposed_external_token_count": proposed_count,
        "context_token_ingestion_proven": context_ok,
        "wallclock_gate_present": bool(wallclock),
        "wallclock_correctness_proven": wallclock_ok,
        "speedup_claimed_by_artifact": bool(context_speedup or wallclock_speedup),
        "ready_for_trial": not blocked,
        "blocked_reasons": blocked,
    }
    return summary, blocked


def _tomorrow_runbook(min_phone_count: int, max_phone_count: int) -> list[str]:
    return [
        f"collect {min_phone_count}-{max_phone_count} distinct physical phones with Termux + llama.cpp available",
        "run Termux context-token emission on each phone: tokenize rendered_prompt and rendered_prompt+draft, then slice prompt tokens off the full token list",
        "pull or copy each phone's termux-context-token JSON and run phone_llama_cpp_binding_verifier.py against the exact same GGUF hash",
        "run phone_speculative_wallclock_gate.py for each phone; speedup_proven must remain false unless an integrated draft-plus-verifier path is actually faster",
        "wrap each phone's context verifier + wallclock gate into one per-phone artifact and run multi_phone_speculative_readiness.py with --phone-artifact for every device",
        "only after this manifest passes, run the integrated non-sequential verifier experiment and compare verifier-only vs phone-draft-plus-verifier wall clock",
    ]


def build_multi_phone_readiness_report(
    phone_artifacts: Iterable[dict[str, Any]],
    *,
    min_phone_count: int = DEFAULT_MIN_PHONE_COUNT,
    max_phone_count: int = DEFAULT_MAX_PHONE_COUNT,
) -> dict[str, Any]:
    artifacts = list(phone_artifacts)
    phone_summaries: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []

    if min_phone_count < 1:
        raise ValueError("min_phone_count must be >= 1")
    if max_phone_count < min_phone_count:
        raise ValueError("max_phone_count must be >= min_phone_count")

    for index, artifact in enumerate(artifacts):
        summary, blocked = _phone_summary(artifact, index)
        phone_summaries.append(summary)
        blocked_reasons.extend(blocked)

    phone_ids = [summary["phone_id"] for summary in phone_summaries]
    for phone_id, count in Counter(phone_ids).items():
        if count > 1:
            blocked_reasons.append(f"duplicate_phone_id:{phone_id}")

    phone_count = len(phone_summaries)
    if phone_count < min_phone_count:
        blocked_reasons.append(f"phone_count_below_min:{phone_count}<{min_phone_count}")
    if phone_count > max_phone_count:
        blocked_reasons.append(f"phone_count_above_max:{phone_count}>{max_phone_count}")

    model_hashes = sorted({summary["model_sha256"] for summary in phone_summaries if summary.get("model_sha256")})
    if len(model_hashes) > 1:
        blocked_reasons.append("model_sha256_mismatch")

    ready_phone_count = sum(1 for summary in phone_summaries if summary["ready_for_trial"])
    all_context = phone_count > 0 and all(summary["context_token_ingestion_proven"] for summary in phone_summaries)
    all_wallclock = phone_count > 0 and all(summary["wallclock_gate_present"] for summary in phone_summaries)
    all_wallclock_correct = phone_count > 0 and all(summary["wallclock_correctness_proven"] for summary in phone_summaries)
    no_speedup_claims = not any(summary["speedup_claimed_by_artifact"] for summary in phone_summaries)
    trial_ready = not blocked_reasons and min_phone_count <= phone_count <= max_phone_count

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": "passed" if trial_ready else "failed",
        "trial_ready": trial_ready,
        "min_phone_count": min_phone_count,
        "max_phone_count": max_phone_count,
        "phone_count": phone_count,
        "ready_phone_count": ready_phone_count,
        "unique_phone_ids": sorted(set(phone_ids)),
        "model_sha256": model_hashes[0] if len(model_hashes) == 1 else None,
        "model_sha256_values": model_hashes,
        "phones": phone_summaries,
        "all_context_token_ingestion_proven": all_context,
        "all_wallclock_gates_present": all_wallclock,
        "all_wallclock_correctness_proven": all_wallclock_correct,
        "no_input_speedup_claims": no_speedup_claims,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": blocked_reasons,
        "tomorrow_runbook": _tomorrow_runbook(min_phone_count, max_phone_count),
        "operator_next_steps": [
            "run this manifest after each physical phone produces context-token and wall-clock gate artifacts",
            "build the integrated non-sequential verifier path that consumes phone token IDs without rerunning verifier-only decode",
            "measure verifier-only and phone-draft-plus-verifier wall clock in the same harness before speedup claims",
            "keep phone BloomBee block-worker status false until separate block-serving evidence passes",
        ],
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-artifact", action="append", default=[], help="Per-phone JSON artifact containing context_token_verifier and wallclock_gate sections; repeat for each phone")
    parser.add_argument("--min-phone-count", type=int, default=DEFAULT_MIN_PHONE_COUNT)
    parser.add_argument("--max-phone-count", type=int, default=DEFAULT_MAX_PHONE_COUNT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_multi_phone_readiness_report(
        [_read_json(path) for path in args.phone_artifact],
        min_phone_count=args.min_phone_count,
        max_phone_count=args.max_phone_count,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["verification_status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
