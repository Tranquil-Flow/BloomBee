#!/usr/bin/env python3
"""Cross-platform Android+iOS phone readiness for speculative decoding.

This is a readiness/plan checker, not a speedup or BloomBee block-worker proof.
It requires at least one ready Android artifact and one ready iOS artifact before
marking cross-platform readiness true, while all promotion flags remain false.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "phone_cross_platform_readiness_no_speedup_or_block_worker_claim"
SOURCE = "phone_cross_platform_readiness.py"
DEFAULT_REQUIRED_PLATFORMS = ("android", "ios")


def _as_bool(value: Any) -> bool:
    return bool(value) is True


def _as_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _runtime_dict(artifact: dict[str, Any]) -> dict[str, Any]:
    runtime = artifact.get("phone_runtime")
    return runtime if isinstance(runtime, dict) else {}


def _phone_id(artifact: dict[str, Any], index: int) -> str:
    runtime = _runtime_dict(artifact)
    for source in (artifact, runtime):
        for key in ("phone_id", "device_id", "serial", "peer_id"):
            value = source.get(key)
            if value:
                return str(value)
    return f"phone-{index + 1}"


def _platform(artifact: dict[str, Any]) -> str:
    runtime = _runtime_dict(artifact)
    value = artifact.get("platform") or artifact.get("os") or runtime.get("platform") or runtime.get("os")
    if value:
        text = str(value).strip().lower()
        if text in {"iphone", "ipad", "ios", "ipados"}:
            return "ios"
        if text in {"android", "termux"}:
            return "android"
        return text
    runtime_name = str(artifact.get("runtime") or runtime.get("runtime") or "").lower()
    if "termux" in runtime_name or "adb" in runtime_name:
        return "android"
    if "ios" in runtime_name or "iphone" in runtime_name or "shortcuts" in runtime_name or "ish" in runtime_name:
        return "ios"
    return "unknown"


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
        nested = model.get("sha256") or model.get("model_sha256")
        if nested:
            return str(nested)
    return None


def _phone_summary(artifact: dict[str, Any], index: int) -> tuple[dict[str, Any], list[str]]:
    phone_id = _phone_id(artifact, index)
    platform = _platform(artifact)
    context = _context_report(artifact)
    wallclock = _wallclock_report(artifact)
    blocked: list[str] = []

    accepted = _as_positive_int(context.get("accepted_external_token_count") or context.get("accepted_context_token_count"))
    context_ok = bool(
        _as_bool(context.get("phone_external_token_ids_ingested"))
        and _as_bool(context.get("phone_integrated_verifier_proven"))
        and _as_bool(context.get("external_context_token_id_acceptance_proven"))
        and accepted > 0
    )
    if not context_ok:
        blocked.append(f"phone:{phone_id}:context_token_ingestion_not_proven")

    wallclock_ok = bool(
        wallclock
        and _as_bool(wallclock.get("verifier_acceptance_proven"))
        and _as_bool(wallclock.get("tokenizer_id_match_proven"))
    )
    if not wallclock_ok:
        blocked.append(f"phone:{phone_id}:wallclock_correctness_not_proven")

    speedup_claim = (
        _as_bool(context.get("speedup_proven"))
        or _as_bool(context.get("can_update_speculative_speedup_status"))
        or _as_bool(wallclock.get("speedup_proven"))
        or _as_bool(wallclock.get("wallclock_speedup_proven"))
        or _as_bool(wallclock.get("can_update_speculative_speedup_status"))
    )
    if speedup_claim:
        blocked.append(f"phone:{phone_id}:unexpected_speedup_claim")
    if _as_bool(context.get("bloombee_block_serving_proven")) or _as_bool(artifact.get("bloombee_block_serving_proven")):
        blocked.append(f"phone:{phone_id}:unexpected_bloombee_block_worker_claim")

    model_sha = _model_sha(context, artifact)
    if not model_sha:
        blocked.append(f"phone:{phone_id}:model_sha256_missing")

    runtime = _runtime_dict(artifact)
    return {
        "phone_id": phone_id,
        "platform": platform,
        "phone_model": artifact.get("phone_model") or runtime.get("model"),
        "runtime": artifact.get("runtime") or runtime.get("runtime"),
        "transport_path": artifact.get("transport_path") or context.get("transport_path"),
        "model_sha256": model_sha,
        "accepted_external_token_count": accepted,
        "context_token_ingestion_proven": context_ok,
        "wallclock_correctness_proven": wallclock_ok,
        "speedup_claimed_by_artifact": speedup_claim,
        "ready_for_cross_platform_trial": not blocked,
        "blocked_reasons": blocked,
    }, blocked


def _platform_runbook() -> dict[str, list[str]]:
    return {
        "android": [
            "Use Termux + llama.cpp on Android; prefer ADB/SSH bridge so the operator does not type long commands.",
            "Emit rendered-prompt and rendered-prompt+draft context token IDs, then verify with phone_llama_cpp_binding_verifier.py.",
            "Run phone_speculative_wallclock_gate.py and keep speedup flags false until integrated non-sequential timing passes.",
        ],
        "ios": [
            "Use an iOS Shortcuts/local-network JSON bridge, iSH/a-Shell, or a small Swift/TestFlight wrapper to emit the same context-token artifact schema.",
            "Pull the iOS artifact to the verifier host and run the same phone_llama_cpp_binding_verifier.py path against the exact GGUF hash.",
            "Do not claim iOS phone-worker readiness until the iOS artifact has context-token ingestion and wall-clock correctness gates.",
        ],
    }


def build_phone_cross_platform_readiness_report(
    phone_artifacts: Iterable[dict[str, Any]],
    *,
    required_platforms: Iterable[str] = DEFAULT_REQUIRED_PLATFORMS,
) -> dict[str, Any]:
    required = sorted({str(platform).lower() for platform in required_platforms})
    phone_rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, artifact in enumerate(phone_artifacts):
        summary, phone_blocked = _phone_summary(artifact, index)
        phone_rows.append(summary)
        blocked.extend(phone_blocked)
        if summary["ready_for_cross_platform_trial"]:
            by_platform[summary["platform"]].append(summary)

    ready_platforms = sorted(platform for platform in required if by_platform.get(platform))
    for platform in required:
        if platform not in ready_platforms:
            blocked.append(f"missing_required_platform:{platform}")

    hashes = sorted({row["model_sha256"] for row in phone_rows if row.get("model_sha256")})
    if len(hashes) > 1:
        blocked.append("model_sha256_mismatch")

    passed = not blocked
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_status": "passed" if passed else "failed",
        "required_platforms": required,
        "ready_platforms": ready_platforms,
        "platforms_present": sorted({row["platform"] for row in phone_rows}),
        "android_ready": "android" in ready_platforms,
        "ios_ready": "ios" in ready_platforms,
        "cross_platform_ready": passed,
        "phone_count": len(phone_rows),
        "ready_phone_count": sum(1 for row in phone_rows if row["ready_for_cross_platform_trial"]),
        "model_sha256": hashes[0] if len(hashes) == 1 else None,
        "model_sha256_values": hashes,
        "phones": phone_rows,
        "blocked_reasons": blocked,
        "platform_runbook": _platform_runbook(),
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "bloombee_block_serving_proven": False,
        "can_update_speculative_speedup_status": False,
        "can_update_phone_worker_status": False,
        "can_update_demo_status": False,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-artifact", action="append", required=True)
    parser.add_argument("--required-platform", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_phone_cross_platform_readiness_report(
        [_read_json(path) for path in args.phone_artifact],
        required_platforms=args.required_platform or DEFAULT_REQUIRED_PLATFORMS,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
