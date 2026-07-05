#!/usr/bin/env python3
"""Build an operator plan for the 3-4 phone speculative-decoding trial.

This consumes a multi-phone readiness manifest and emits the next runnable command
shape. It is a plan/command generator only: no speedup or phone worker promotion
is allowed until an operator measures an integrated draft-plus-verifier path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "phone_speculative_integrated_trial_plan_no_speedup_claim"
SOURCE = "phone_speculative_trial_plan.py"


def _ready_phones(readiness_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        phone
        for phone in readiness_manifest.get("phones", [])
        if phone.get("ready_for_trial") is True
    ]


def _phone_output_base(output_dir: str, phone_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in phone_id)
    return f"{output_dir.rstrip('/')}/{safe}"


def _per_phone_commands(phone: dict[str, Any], output_dir: str) -> dict[str, Any]:
    phone_id = str(phone["phone_id"])
    base = _phone_output_base(output_dir, phone_id)
    token_artifact = phone.get("termux_context_token_artifact") or f"{base}-termux-context-token-ids.json"
    # The concrete token IDs are intentionally left as a jq/substitution placeholder
    # because the phone artifact schema stores them in operator-captured JSON, not in
    # the aggregate readiness summary.
    token_placeholder = f"$(python -c 'import json;print(json.load(open(\"{token_artifact}\"))[\"phone_context_draft_token_ids\"])')"
    return {
        "phone_id": phone_id,
        "phone_model": phone.get("phone_model"),
        "runtime": phone.get("runtime"),
        "termux_context_token_artifact": token_artifact,
        "commands": [
            "# Re-run or refresh this phone's context-token verifier from the phone-emitted token JSON.",
            "python mvp_capabilities/phone_llama_cpp_binding_verifier.py "
            "--model $GGUF_MODEL_PATH "
            "--prompt $PROMPT "
            "--draft-text $DRAFT_TEXT "
            f"--phone-context-token-ids '{token_placeholder}' "
            f"--out {base}-context-token-verifier.json",
            "# Record the per-phone wall-clock gate; this must stay speedup_proven=false unless integrated timing is truly faster.",
            "python mvp_capabilities/phone_speculative_wallclock_gate.py "
            f"--phone-bridge {base}-draft-bridge.json "
            f"--same-gguf-verifier {base}-context-token-verifier.json "
            f"--tokenizer-compare {base}-tokenizer-compare.json "
            "--measured-draft-plus-verifier-elapsed-s $INTEGRATED_DRAFT_PLUS_VERIFIER_ELAPSED_S "
            f"--out {base}-wallclock-gate.json",
        ],
    }


def build_phone_speculative_trial_plan(
    readiness_manifest: dict[str, Any],
    *,
    output_dir: str = ".local/phone/trial",
) -> dict[str, Any]:
    readiness_passed = bool(
        readiness_manifest.get("verification_status") == "passed"
        and readiness_manifest.get("trial_ready") is True
        and readiness_manifest.get("speedup_proven") is False
        and readiness_manifest.get("can_update_speculative_speedup_status") is False
    )
    phones = _ready_phones(readiness_manifest) if readiness_passed else []
    blocked_reasons: list[str] = []
    if not readiness_passed:
        blocked_reasons.append("readiness_manifest_not_passed")
        blocked_reasons.extend(str(reason) for reason in readiness_manifest.get("blocked_reasons", []))
    if readiness_passed and len(phones) < 3:
        blocked_reasons.append(f"ready_phone_count_below_trial_min:{len(phones)}<3")
        phones = []

    selected_ids = [str(phone["phone_id"]) for phone in phones]
    plan_ready = not blocked_reasons and bool(phones)
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "plan_status": "ready_for_integrated_trial" if plan_ready else "blocked_by_readiness_manifest",
        "readiness_claim_boundary": readiness_manifest.get("claim_boundary"),
        "phone_count": len(phones),
        "selected_phone_ids": selected_ids,
        "model_sha256": readiness_manifest.get("model_sha256"),
        "output_dir": output_dir,
        "candidate_timing_kind": "operator_measured_integrated_draft_plus_verifier_required",
        "per_phone_commands": [_per_phone_commands(phone, output_dir) for phone in phones],
        "operator_sequence": [
            "refresh each phone's Termux context-token artifact immediately before the trial",
            "run per-phone context-token verifier and wall-clock gate commands",
            "run all phone draft providers concurrently for the same prompt batch or separate queued prompts",
            "compare verifier-only vs integrated draft-plus-verifier wall clock in the same harness",
            "promote speedup only if integrated timing is faster and exact-token acceptance remains proven",
        ],
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": blocked_reasons,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-manifest", required=True)
    parser.add_argument("--output-dir", default=".local/phone/trial")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_phone_speculative_trial_plan(
        _read_json(args.readiness_manifest),
        output_dir=args.output_dir,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["plan_status"] == "ready_for_integrated_trial" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
