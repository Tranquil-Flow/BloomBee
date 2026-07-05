#!/usr/bin/env python3
"""Bundle split phone speculative artifacts into one readiness input.

The multi-phone readiness manifest intentionally consumes one JSON document per
physical phone. Earlier proof steps produce separate context-token verifier,
Termux token-emission, and wall-clock gate artifacts. This helper wraps those
split artifacts without creating a speedup or block-serving proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "phone_speculative_readiness_artifact_bundle_no_speedup_claim"
SOURCE = "phone_speculative_artifact_bundle.py"


def _as_bool(value: Any) -> bool:
    return bool(value) is True


def _model_sha(context_verifier: dict[str, Any], termux_context_tokens: dict[str, Any] | None) -> str | None:
    for source in (context_verifier, termux_context_tokens or {}):
        value = source.get("model_sha256")
        if value:
            return str(value)
        model = source.get("model")
        if isinstance(model, dict):
            value = model.get("sha256") or model.get("model_sha256")
            if value:
                return str(value)
    return None


def _has_speedup_claim(*sources: dict[str, Any] | None) -> bool:
    speedup_keys = (
        "speedup_proven",
        "wallclock_speedup_proven",
        "can_update_speculative_speedup_status",
        "can_update_phone_worker_status",
        "can_update_bloombee_block_worker_status",
    )
    return any(_as_bool(source.get(key)) for source in sources if source for key in speedup_keys)


def build_phone_speculative_artifact_bundle(
    *,
    context_verifier: dict[str, Any],
    wallclock_gate: dict[str, Any],
    termux_context_tokens: dict[str, Any] | None = None,
    phone_id: str,
    phone_model: str | None = None,
    runtime: str | None = None,
    termux_context_token_artifact: str | None = None,
) -> dict[str, Any]:
    blocked_reasons: list[str] = []
    if not context_verifier:
        blocked_reasons.append("context_verifier_missing")
    if not wallclock_gate:
        blocked_reasons.append("wallclock_gate_missing")
    if _has_speedup_claim(context_verifier, wallclock_gate, termux_context_tokens):
        blocked_reasons.append("source_unexpected_speedup_claim")

    model_sha = _model_sha(context_verifier, termux_context_tokens)
    if not model_sha:
        blocked_reasons.append("model_sha256_missing")

    token_artifact = termux_context_token_artifact or context_verifier.get("phone_token_json_artifact")
    transport_path = context_verifier.get("transport_path")
    if not transport_path and termux_context_tokens:
        transport_path = termux_context_tokens.get("transport_path")

    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "phone_id": phone_id,
        "phone_model": phone_model,
        "runtime": runtime,
        "transport_path": transport_path,
        "model_sha256": model_sha,
        "termux_context_token_artifact": token_artifact,
        "context_token_verifier": context_verifier,
        "wallclock_gate": wallclock_gate,
        "termux_context_tokens": termux_context_tokens,
        "bundle_ready_for_manifest": not blocked_reasons,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_speculative_speedup_status": False,
        "bloombee_block_serving_proven": False,
        "can_update_phone_worker_status": False,
        "blocked_reasons": blocked_reasons,
        "do_not_claim": [
            "no new phone runtime proof",
            "no integrated speculative speedup proof",
            "no BloomBee phone block-serving proof",
        ],
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-verifier", required=True)
    parser.add_argument("--wallclock-gate", required=True)
    parser.add_argument("--termux-context-tokens", default=None)
    parser.add_argument("--termux-context-token-artifact", default=None)
    parser.add_argument("--phone-id", required=True)
    parser.add_argument("--phone-model", default=None)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_phone_speculative_artifact_bundle(
        context_verifier=_read_json(args.context_verifier),
        wallclock_gate=_read_json(args.wallclock_gate),
        termux_context_tokens=_read_json(args.termux_context_tokens) if args.termux_context_tokens else None,
        phone_id=args.phone_id,
        phone_model=args.phone_model,
        runtime=args.runtime,
        termux_context_token_artifact=args.termux_context_token_artifact,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
