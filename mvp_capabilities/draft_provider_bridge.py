#!/usr/bin/env python3
"""Stdio JSONL bridge for the draft-provider contract.

This is the transport groundwork for phone/Termux experiments. It accepts JSON
requests on stdin, invokes the same deterministic DraftProvider contract used by
``draft_provider.py``, and emits one JSON response per input line. It does not run
live generation, prove speedup, or count a phone as an inference worker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

try:
    from mvp_capabilities.draft_provider import (
        CLAIM_BOUNDARY as DRAFT_PROVIDER_CLAIM_BOUNDARY,
        DeterministicHashDraftProvider,
        StaticDraftProvider,
        build_draft_provider_report,
    )
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.draft_provider import (
        CLAIM_BOUNDARY as DRAFT_PROVIDER_CLAIM_BOUNDARY,
        DeterministicHashDraftProvider,
        StaticDraftProvider,
        build_draft_provider_report,
    )

SOURCE = "draft_provider_bridge.py"
CLAIM_BOUNDARY = "draft_provider_stdio_bridge_only_no_generation_proof"
ERROR_CLAIM_BOUNDARY = "draft_provider_stdio_bridge_error_no_generation_proof"


def _int_list(value: Any, *, field: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list of token ids")
    try:
        return tuple(int(token) for token in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integer token ids") from exc


def handle_draft_bridge_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle one JSON request for a remote/stdio draft-provider bridge."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "request must be a JSON object", "claim_boundary": ERROR_CLAIM_BOUNDARY}
    request_id = payload.get("request_id")
    try:
        prompt_tokens = _int_list(payload.get("prompt_tokens"), field="prompt_tokens")
        verifier_tokens = _int_list(payload.get("verifier_tokens"), field="verifier_tokens")
        max_draft_tokens = int(payload.get("max_draft_tokens", 4))
        if max_draft_tokens < 0:
            raise ValueError("max_draft_tokens must be non-negative")
        static_tokens_raw = payload.get("draft_tokens")
        if static_tokens_raw is not None:
            provider = StaticDraftProvider(
                _int_list(static_tokens_raw, field="draft_tokens"),
                provider_id=str(payload.get("provider_id") or "stdio-static-draft-provider"),
            )
        else:
            provider = DeterministicHashDraftProvider(
                provider_id=str(payload.get("provider_id") or "stdio-hash-draft-provider"),
                vocab_size=int(payload.get("vocab_size", 32_000)),
                seed=str(payload.get("seed") or "bloombee-draft-provider"),
            )
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "request_id": request_id,
            "source": SOURCE,
            "claim_boundary": ERROR_CLAIM_BOUNDARY,
            "error": str(exc),
            "generation_proven": False,
            "speedup_proven": False,
            "inference_proven": False,
        }

    report = build_draft_provider_report(
        provider=provider,
        prompt_tokens=prompt_tokens,
        verifier_tokens=verifier_tokens,
        max_draft_tokens=max_draft_tokens,
    )
    return {
        "ok": True,
        "request_id": request_id,
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "draft_provider_claim_boundary": DRAFT_PROVIDER_CLAIM_BOUNDARY,
        "report": report,
        "dashboard_counters": report.get("dashboard_counters"),
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def serve_stdio(input_stream: TextIO, output_stream: TextIO) -> int:
    """Serve JSONL requests from ``input_stream`` to ``output_stream``."""
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "ok": False,
                "source": SOURCE,
                "claim_boundary": ERROR_CLAIM_BOUNDARY,
                "error": f"invalid json: {exc.msg}",
                "generation_proven": False,
                "speedup_proven": False,
                "inference_proven": False,
            }
        else:
            response = handle_draft_bridge_request(payload)
        output_stream.write(json.dumps(response, sort_keys=True) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="Handle one request JSON file")
    once.add_argument("--request-json", required=True)

    sub.add_parser("serve-stdio", help="Read JSONL requests from stdin and write JSONL responses")
    args = parser.parse_args(argv)

    if args.command == "once":
        payload = json.loads(Path(args.request_json).expanduser().read_text(encoding="utf-8"))
        print(json.dumps(handle_draft_bridge_request(payload), indent=2, sort_keys=True))
        return 0
    if args.command == "serve-stdio":
        return serve_stdio(sys.stdin, sys.stdout)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
