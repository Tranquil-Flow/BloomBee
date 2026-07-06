#!/usr/bin/env python3
"""Assemble live-server continuous batching capture evidence.

This module is an operator harness, not proof by itself. It turns real capture
artifacts (for example ``scripts/text_generation_parity.py`` JSON outputs plus a
server-observed live-continuous report) into the evidence shape consumed by
``continuous_batching_live_server_proof.py``. The assembler deliberately carries
through server-observed flags instead of inferring them from client-only rows,
so missing live-server evidence remains fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mvp_capabilities.continuous_batching_live_server_proof import OPT_IN_FLAG, PROOF_GATE

PLAN_CLAIM_BOUNDARY = "live_continuous_batching_capture_harness_no_live_server_proof"
ASSEMBLER_CLAIM_BOUNDARY = "live_continuous_batching_capture_assembler_candidate_no_speedup"
DEFAULT_EVIDENCE_PATH = ".local/live-continuous-batching-capture.json"


def _stable_sha256(value: Any) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _token_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of integer token IDs")
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{field} must be a list of integer token IDs")
        tokens.append(int(item))
    return tokens


def _maybe_token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        out.append(int(item))
    return out


def _capture_model(capture: Mapping[str, Any]) -> str | None:
    value = capture.get("model_id") or capture.get("model")
    return value if isinstance(value, str) else None


def _generated_token_ids(capture: Mapping[str, Any]) -> list[int]:
    for key in ("generated_token_ids", "output_token_ids", "distributed_generated_token_ids"):
        tokens = _maybe_token_list(capture.get(key))
        if tokens is not None:
            return tokens

    input_ids = _token_list(capture.get("input_ids"), field="input_ids")
    output_ids = _token_list(capture.get("distributed_ids"), field="distributed_ids")
    if len(output_ids) < len(input_ids) or output_ids[: len(input_ids)] != input_ids:
        raise ValueError("distributed_ids must start with input_ids when generated_token_ids is absent")
    generated = output_ids[len(input_ids) :]
    if not generated:
        raise ValueError("generated token IDs must be non-empty")
    return generated


def _logits_sha256(capture: Mapping[str, Any]) -> str:
    for key in ("logits_sha256", "logits_hash", "distributed_logits_sha256"):
        value = capture.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("distributed_top5", "top5", "logits"):
        value = capture.get(key)
        if value is not None:
            return _stable_sha256(value)
    return _stable_sha256(_generated_token_ids(capture))


def _seconds(capture: Mapping[str, Any]) -> float | None:
    for key in ("distributed_seconds", "seconds", "elapsed_seconds", "wall_seconds"):
        value = capture.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
            return float(value)
    return None


def _live_tick_batches(live_report: Mapping[str, Any]) -> list[Any]:
    nested = live_report.get("live_continuous_report") or live_report.get("session_report")
    if isinstance(nested, Mapping) and isinstance(nested.get("tick_batches"), list):
        return list(nested["tick_batches"])
    tick_batches = live_report.get("tick_batches")
    return list(tick_batches) if isinstance(tick_batches, list) else []


def _source_entry(source_artifacts: Mapping[str, Any] | None, request_id: str) -> dict[str, Any]:
    if not source_artifacts:
        return {}
    value = source_artifacts.get(request_id)
    return dict(value) if isinstance(value, Mapping) else {}


def _capture_row(capture: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generated_token_ids": _generated_token_ids(capture),
        "logits_sha256": _logits_sha256(capture),
    }
    seconds = _seconds(capture)
    if seconds is not None:
        row["seconds"] = seconds
    text = capture.get("distributed_text") or capture.get("text")
    if isinstance(text, str):
        row["text_sha256"] = _stable_sha256(text)
    return row


def assemble_live_server_continuous_batching_evidence(
    *,
    model_id: str,
    baseline_by_request: Mapping[str, Mapping[str, Any]],
    continuous_by_request: Mapping[str, Mapping[str, Any]],
    arrival_ticks: Mapping[str, int],
    live_report: Mapping[str, Any],
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build verifier-ready continuous batching evidence from capture rows.

    The returned payload may still fail the strict verifier if the supplied live
    report lacks explicit server-observed evidence. This is intentional: client
    tick rows alone do not prove server-side continuous batching.
    """

    baseline_ids = set(baseline_by_request)
    continuous_ids = set(continuous_by_request)
    if baseline_ids != continuous_ids:
        missing_continuous = sorted(baseline_ids - continuous_ids)
        missing_baseline = sorted(continuous_ids - baseline_ids)
        raise ValueError(
            "baseline/continuous request IDs differ"
            f"; missing continuous={missing_continuous}; missing baseline={missing_baseline}"
        )

    rows: list[dict[str, Any]] = []
    for request_id in sorted(baseline_ids, key=lambda item: (int(arrival_ticks.get(item, 0)), item)):
        if request_id not in arrival_ticks:
            raise ValueError(f"arrival tick missing for {request_id}")
        baseline_capture = baseline_by_request[request_id]
        continuous_capture = continuous_by_request[request_id]
        for label, capture in (("baseline", baseline_capture), ("continuous", continuous_capture)):
            capture_model = _capture_model(capture)
            if capture_model is not None and capture_model != model_id:
                raise ValueError(f"{label} capture for {request_id} has model {capture_model!r}, expected {model_id!r}")
        row: dict[str, Any] = {
            "request_id": request_id,
            "arrival_tick": int(arrival_ticks[request_id]),
            "baseline": _capture_row(baseline_capture),
            "continuous": _capture_row(continuous_capture),
        }
        source = _source_entry(source_artifacts, request_id)
        if source:
            row["source_artifacts"] = source
        prompt = continuous_capture.get("prompt") or baseline_capture.get("prompt")
        if isinstance(prompt, str):
            row["prompt_sha256"] = _stable_sha256(prompt)
        rows.append(row)

    tick_batches = _live_tick_batches(live_report)
    server_observed = live_report.get("server_observed_live_continuous_batches") is True
    live_server_proven = live_report.get("live_server_proven") is True
    opt_in_enabled = live_report.get("opt_in_enabled") is True
    return {
        "model_id": model_id,
        "claim_boundary": ASSEMBLER_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "source": "mvp_capabilities.continuous_batching_live_server_capture",
        "opt_in_flag": OPT_IN_FLAG,
        "opt_in_enabled": opt_in_enabled,
        "server_observed_live_continuous_batches": server_observed,
        "live_server_proven": live_server_proven,
        "requests": rows,
        "live_continuous_report": {
            "tick_batches": tick_batches,
            "source": live_report.get("source"),
        },
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "can_update_demo_status": False,
        "can_update_proof_status": False,
        "claim_limitations": [
            "Assembler only; proof still depends on strict verifier output.",
            "Server-observed flags are carried from the live report and are never inferred from client-only rows.",
            "This artifact does not prove wall-clock speedup or demo readiness.",
        ],
    }


def build_live_server_continuous_batching_capture_plan(
    *,
    model_id: str,
    evidence_path: str = DEFAULT_EVIDENCE_PATH,
) -> dict[str, Any]:
    log_report_command = (
        "python -m mvp_capabilities.continuous_batching_server_log_report "
        "--log .local/live-continuous-batching-server.log "
        "--out .local/live-continuous-server-report.json"
    )
    assemble_command = (
        "python -m mvp_capabilities.continuous_batching_live_server_capture assemble "
        f"--model {model_id} --live-report .local/live-continuous-server-report.json "
        "--request req-a:0:.local/baseline-req-a.json:.local/continuous-req-a.json "
        "--request req-b:1:.local/baseline-req-b.json:.local/continuous-req-b.json "
        f"--out {evidence_path}"
    )
    verify_command = (
        "python mvp_capabilities/continuous_batching_live_server_proof.py verify "
        f"--model {model_id} --evidence {evidence_path}"
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "evidence_path": evidence_path,
        "operator_commands": [
            f"Start the server with {OPT_IN_FLAG}=1 and capture at least two staggered live requests.",
            "Capture no-continuous baseline and opt-in continuous JSON rows without hand-editing token IDs.",
            log_report_command,
            assemble_command,
            verify_command,
        ],
        "log_report_command": log_report_command,
        "assemble_command": assemble_command,
        "verify_command": verify_command,
        "live_server_late_arrival_parity_proven": False,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "notes": [
            "Planning output is not proof.",
            "The assembler refuses to infer server-observed status from client-only tick rows.",
            "Promotion still requires continuous_batching_live_server_proof.py to pass on real live-server evidence and a later wall-clock throughput proof.",
        ],
    }


def _parse_request_spec(raw: str) -> tuple[str, int, Path, Path]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("request spec must be request_id:arrival_tick:baseline_json:continuous_json")
    request_id, tick_text, baseline, continuous = parts
    if not request_id:
        raise argparse.ArgumentTypeError("request_id must be non-empty")
    try:
        tick = int(tick_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arrival_tick must be an integer") from exc
    return request_id, tick, Path(baseline), Path(continuous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a claim-bounded capture harness plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--evidence", default=DEFAULT_EVIDENCE_PATH)
    plan.add_argument("--out", default=None)

    assemble = sub.add_parser("assemble", help="Assemble capture JSON rows into verifier evidence")
    assemble.add_argument("--model", required=True)
    assemble.add_argument("--live-report", required=True)
    assemble.add_argument(
        "--request",
        action="append",
        required=True,
        help="request_id:arrival_tick:baseline_json:continuous_json; pass once per request",
    )
    assemble.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_live_server_continuous_batching_capture_plan(
            model_id=args.model,
            evidence_path=args.evidence,
        )
    else:
        baseline_by_request: dict[str, dict[str, Any]] = {}
        continuous_by_request: dict[str, dict[str, Any]] = {}
        arrival_ticks: dict[str, int] = {}
        source_artifacts: dict[str, dict[str, str]] = {}
        for raw in args.request:
            request_id, tick, baseline_path, continuous_path = _parse_request_spec(raw)
            if request_id in baseline_by_request:
                raise SystemExit(f"duplicate request_id: {request_id}")
            baseline_by_request[request_id] = _read_json(baseline_path)
            continuous_by_request[request_id] = _read_json(continuous_path)
            arrival_ticks[request_id] = tick
            source_artifacts[request_id] = {
                "baseline": str(baseline_path),
                "continuous": str(continuous_path),
            }
        payload = assemble_live_server_continuous_batching_evidence(
            model_id=args.model,
            baseline_by_request=baseline_by_request,
            continuous_by_request=continuous_by_request,
            arrival_ticks=arrival_ticks,
            live_report=_read_json(args.live_report),
            source_artifacts=source_artifacts,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
