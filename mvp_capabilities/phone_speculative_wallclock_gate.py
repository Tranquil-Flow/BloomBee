#!/usr/bin/env python3
"""Fail-closed wall-clock gate for phone draft-provider speedup claims.

This module does not implement speculative decoding. It consumes already-measured
phone-draft and verifier-only timings, then refuses to promote speedup unless a
measured draft-plus-verifier path is both correct and faster than verifier-only.
Sequential phone draft + verifier-only is useful evidence, but it is expected to
be slower and must keep speedup false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "phone_speculative_wallclock_gate_fail_closed"
SOURCE = "phone_speculative_wallclock_gate.py"


def _round_s(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def build_phone_speculative_wallclock_gate(
    *,
    phone_draft_elapsed_s: float,
    verifier_only_elapsed_s: float,
    verifier_acceptance_proven: bool,
    tokenizer_id_match_proven: bool,
    measured_draft_plus_verifier_elapsed_s: float | None = None,
    source_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a conservative speedup-gate report.

    If no integrated draft-plus-verifier timing exists, the only measured combined
    path is sequential: phone draft first, then verifier-only. That is never a
    speculative-speedup proof unless it is faster than verifier-only and the
    correctness gates are also true.
    """
    phone_s = float(phone_draft_elapsed_s)
    verifier_s = float(verifier_only_elapsed_s)
    sequential_s = phone_s + verifier_s
    if measured_draft_plus_verifier_elapsed_s is None:
        candidate_s = sequential_s
        timing_kind = "sequential_phone_draft_plus_verifier_only"
    else:
        candidate_s = float(measured_draft_plus_verifier_elapsed_s)
        timing_kind = "measured_draft_plus_verifier"

    correctness_ok = bool(verifier_acceptance_proven and tokenizer_id_match_proven)
    faster = candidate_s < verifier_s
    speedup_proven = correctness_ok and faster
    if not verifier_acceptance_proven:
        blocked_reason = "verifier_acceptance_not_proven"
    elif not tokenizer_id_match_proven:
        blocked_reason = "tokenizer_id_match_not_proven"
    elif not faster and measured_draft_plus_verifier_elapsed_s is None:
        blocked_reason = "sequential_draft_plus_verifier_not_faster_than_verifier_only"
    elif not faster:
        blocked_reason = "measured_draft_plus_verifier_not_faster_than_verifier_only"
    else:
        blocked_reason = None

    speedup_ratio = verifier_s / candidate_s if candidate_s > 0 else None
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_artifacts": list(source_artifacts),
        "phone_draft_elapsed_s": _round_s(phone_s),
        "verifier_only_elapsed_s": _round_s(verifier_s),
        "sequential_draft_plus_verifier_elapsed_s": _round_s(sequential_s),
        "measured_draft_plus_verifier_elapsed_s": _round_s(measured_draft_plus_verifier_elapsed_s),
        "candidate_draft_plus_verifier_elapsed_s": _round_s(candidate_s),
        "timing_kind": timing_kind,
        "verifier_acceptance_proven": bool(verifier_acceptance_proven),
        "tokenizer_id_match_proven": bool(tokenizer_id_match_proven),
        "wallclock_speedup_proven": bool(speedup_proven),
        "speedup_proven": bool(speedup_proven),
        "speedup_ratio": _round_s(speedup_ratio),
        "blocked_reason": blocked_reason,
        "can_update_speculative_speedup_status": bool(speedup_proven),
        "bloombee_block_serving_proven": False,
        "can_update_bloombee_block_worker_status": False,
        "operator_next_steps": [
            "build a real integrated verifier path that validates draft token IDs without rerunning full verifier-only decode",
            "measure verifier-only and draft-plus-verifier wall clock in the same harness",
            "keep BloomBee block-worker status false until phone block serving is separately proven",
        ],
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_from_artifacts(
    *,
    phone_bridge: dict[str, Any],
    same_gguf_verifier: dict[str, Any],
    tokenizer_compare: dict[str, Any],
    source_artifacts: Iterable[str],
    measured_draft_plus_verifier_elapsed_s: float | None = None,
) -> dict[str, Any]:
    evidence = phone_bridge.get("evidence") if "evidence" in phone_bridge else phone_bridge
    phone_elapsed = float((evidence.get("draft_response") or {}).get("elapsed_s"))
    verifier_elapsed = float(same_gguf_verifier.get("elapsed_s"))
    verifier_acceptance = bool(tokenizer_compare.get("verifier_acceptance_proven"))
    tokenizer_match = bool(tokenizer_compare.get("tokenizer_id_match_proven"))
    return build_phone_speculative_wallclock_gate(
        phone_draft_elapsed_s=phone_elapsed,
        verifier_only_elapsed_s=verifier_elapsed,
        verifier_acceptance_proven=verifier_acceptance,
        tokenizer_id_match_proven=tokenizer_match,
        measured_draft_plus_verifier_elapsed_s=measured_draft_plus_verifier_elapsed_s,
        source_artifacts=source_artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-bridge", required=True)
    parser.add_argument("--same-gguf-verifier", required=True)
    parser.add_argument("--tokenizer-compare", required=True)
    parser.add_argument(
        "--measured-draft-plus-verifier-elapsed-s",
        type=float,
        default=None,
        help="Measured integrated draft-plus-verifier wall-clock elapsed seconds. Omit to keep the fail-closed sequential timing path.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    paths = [args.phone_bridge, args.same_gguf_verifier, args.tokenizer_compare]
    payload = build_from_artifacts(
        phone_bridge=_read_json(args.phone_bridge),
        same_gguf_verifier=_read_json(args.same_gguf_verifier),
        tokenizer_compare=_read_json(args.tokenizer_compare),
        source_artifacts=paths,
        measured_draft_plus_verifier_elapsed_s=args.measured_draft_plus_verifier_elapsed_s,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
