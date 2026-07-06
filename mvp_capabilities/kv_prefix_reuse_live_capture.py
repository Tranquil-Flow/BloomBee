#!/usr/bin/env python3
"""Assemble live KV-prefix reuse capture evidence.

This is a claim-bounded operator harness. It converts no-reuse baseline rows,
reuse-path rows, explicit suffix metadata, and a server-side report into the
shape consumed by :mod:`mvp_capabilities.kv_prefix_reuse_proof`. The assembler
must not infer live KV tensor reuse from client-only rows or from the existing
metadata-only path: without an explicit server-observed live-cache-reuse report,
the generated artifact remains fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mvp_capabilities.kv_prefix_reuse_proof import OPT_IN_FLAG, PROOF_GATE

PLAN_CLAIM_BOUNDARY = "kv_prefix_reuse_live_capture_assembler_harness_no_live_cache_reuse_proof"
ASSEMBLER_CLAIM_BOUNDARY = "kv_prefix_reuse_live_capture_assembler_candidate"
METADATA_ONLY_CLAIM_BOUNDARY = "kv_prefix_reuse_prefill_metadata_no_live_cache_reuse"
DEFAULT_EVIDENCE_PATH = ".local/kv-prefix-reuse-live-capture.json"


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
    if not tokens:
        raise ValueError(f"{field} must be non-empty")
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


def _parse_token_csv(raw: str, *, field: str) -> list[int]:
    text = raw.strip()
    if not text:
        raise argparse.ArgumentTypeError(f"{field} must contain at least one token ID")
    tokens: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            raise argparse.ArgumentTypeError(f"{field} contains an empty token ID")
        try:
            tokens.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{field} token IDs must be integers") from exc
    return tokens


def _capture_model(capture: Mapping[str, Any]) -> str | None:
    value = capture.get("model_id") or capture.get("model")
    return value if isinstance(value, str) else None


def _generated_token_ids(capture: Mapping[str, Any]) -> list[int]:
    for key in ("generated_token_ids", "output_token_ids", "distributed_generated_token_ids"):
        tokens = _maybe_token_list(capture.get(key))
        if tokens is not None and tokens:
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
    for key in ("logits_sha256", "logits_hash", "distributed_logits_sha256", "logits_fingerprint"):
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
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seconds = float(value)
            if seconds > 0:
                return seconds
    return None


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


def _metadata_only_report(server_report: Mapping[str, Any]) -> bool:
    return (
        server_report.get("claim_boundary") == METADATA_ONLY_CLAIM_BOUNDARY
        or server_report.get("server_observed_metadata") is True
    ) and server_report.get("live_kv_cache_reuse_proven") is not True


def _server_handle_handoff_observed(server_report: Mapping[str, Any]) -> bool:
    if server_report.get("server_handle_handoff_observed") is not True:
        return False
    source_handle = server_report.get("cache_read_source_handle_id")
    destination_handle = server_report.get("cache_write_destination_handle_id")
    checksum = server_report.get("kv_prefix_byte_checksum_sha256")
    return (
        source_handle is not None
        and destination_handle is not None
        and str(source_handle) != str(destination_handle)
        and isinstance(checksum, str)
        and len(checksum) == 64
    )


def _live_kv_reuse_proven(server_report: Mapping[str, Any]) -> bool:
    if _metadata_only_report(server_report):
        return False
    return (
        server_report.get("opt_in_enabled") is True
        and server_report.get("server_observed_kv_cache_reuse") is True
        and server_report.get("live_kv_cache_reuse_proven") is True
        and _server_handle_handoff_observed(server_report)
    )


def _server_request_entries(server_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = server_report.get("requests") or server_report.get("request_reports")
    if isinstance(raw, Mapping):
        return {
            str(request_id): value
            for request_id, value in raw.items()
            if isinstance(value, Mapping)
        }
    if isinstance(raw, list):
        out: dict[str, Mapping[str, Any]] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            request_id = item.get("request_id")
            if isinstance(request_id, str) and request_id:
                out[request_id] = item
        return out
    return {}


def _reused_prefix_count(
    *,
    request_id: str,
    server_requests: Mapping[str, Mapping[str, Any]],
    live_reuse_proven: bool,
) -> int:
    if not live_reuse_proven:
        return 0
    entry = server_requests.get(request_id, {})
    value = entry.get("reused_prefix_token_count") or entry.get("reused_token_count")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return 0


def _source_entry(source_artifacts: Mapping[str, Any] | None, request_id: str) -> dict[str, Any]:
    if not source_artifacts:
        return {}
    value = source_artifacts.get(request_id)
    return dict(value) if isinstance(value, Mapping) else {}


def assemble_kv_prefix_reuse_live_capture_evidence(
    *,
    model_id: str,
    common_prefix_token_ids: list[int],
    suffix_token_ids_by_request: Mapping[str, list[int]],
    baseline_by_request: Mapping[str, Mapping[str, Any]],
    reuse_by_request: Mapping[str, Mapping[str, Any]],
    server_report: Mapping[str, Any],
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build verifier-ready KV prefix-reuse evidence from live capture rows.

    A verifier-passing artifact is possible only when ``server_report`` explicitly
    states that server-side KV cache reuse was observed. Metadata-only reports
    are preserved in the output but force ``prefix_reuse_enabled=False`` so the
    downstream verifier fails closed.
    """

    common_prefix = _token_list(list(common_prefix_token_ids), field="common_prefix_token_ids")
    baseline_ids = set(baseline_by_request)
    reuse_ids = set(reuse_by_request)
    suffix_ids = set(suffix_token_ids_by_request)
    if baseline_ids != reuse_ids or baseline_ids != suffix_ids:
        raise ValueError(
            "baseline/reuse/suffix request IDs differ"
            f"; missing reuse={sorted(baseline_ids - reuse_ids)}"
            f"; missing baseline={sorted(reuse_ids - baseline_ids)}"
            f"; missing suffix={sorted((baseline_ids | reuse_ids) - suffix_ids)}"
        )

    metadata_only = _metadata_only_report(server_report)
    live_reuse_proven = _live_kv_reuse_proven(server_report)
    server_requests = _server_request_entries(server_report)

    server_observations: list[dict[str, Any]] = []
    if live_reuse_proven:
        server_observations.append(
            {
                "source": server_report.get("source") or "bloombee.server.kv_prefix_reuse_capture",
                "claim_boundary": server_report.get("claim_boundary") or "live_kv_prefix_reuse_server_capture",
                "server_observed_kv_cache_reuse": True,
                "live_kv_cache_reuse_proven": True,
                "server_handle_handoff_observed": True,
                "cache_read_source_handle_id": server_report.get("cache_read_source_handle_id"),
                "cache_write_destination_handle_id": server_report.get("cache_write_destination_handle_id"),
                "client_claimed_prefix_token_count": server_report.get(
                    "client_claimed_prefix_token_count", len(common_prefix)
                ),
                "server_recovered_prefix_token_count": server_report.get(
                    "server_recovered_prefix_token_count", len(common_prefix)
                ),
                "kv_prefix_byte_checksum_sha256": server_report.get("kv_prefix_byte_checksum_sha256"),
                "prefix_length": server_report.get("prefix_length", len(common_prefix)),
                "cache_handle_count": server_report.get("cache_handle_count", 2),
            }
        )

    rows: list[dict[str, Any]] = []
    for request_id in sorted(baseline_ids):
        baseline_capture = baseline_by_request[request_id]
        reuse_capture = reuse_by_request[request_id]
        for label, capture in (("baseline", baseline_capture), ("reuse", reuse_capture)):
            capture_model = _capture_model(capture)
            if capture_model is not None and capture_model != model_id:
                raise ValueError(f"{label} capture for {request_id} has model {capture_model!r}, expected {model_id!r}")
        suffix = _token_list(suffix_token_ids_by_request[request_id], field=f"suffix_token_ids[{request_id}]")
        baseline_row = _capture_row(baseline_capture)
        reuse_row = _capture_row(reuse_capture)
        reuse_row["reused_prefix_token_count"] = _reused_prefix_count(
            request_id=request_id,
            server_requests=server_requests,
            live_reuse_proven=live_reuse_proven,
        )
        row: dict[str, Any] = {
            "request_id": request_id,
            "prefix_token_ids": list(common_prefix),
            "suffix_token_ids": suffix,
            "baseline": baseline_row,
            "reuse": reuse_row,
        }
        source = _source_entry(source_artifacts, request_id)
        if source:
            row["source_artifacts"] = source
        prompt = reuse_capture.get("prompt") or baseline_capture.get("prompt")
        if isinstance(prompt, str):
            row["prompt_sha256"] = _stable_sha256(prompt)
        rows.append(row)

    return {
        "model_id": model_id,
        "claim_boundary": ASSEMBLER_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "source": "mvp_capabilities.kv_prefix_reuse_live_capture",
        "opt_in_flag": OPT_IN_FLAG,
        "opt_in_enabled": server_report.get("opt_in_enabled") is True,
        "prefix_reuse_enabled": live_reuse_proven,
        "cache_reuse_enabled": live_reuse_proven,
        "baseline_reuse_enabled": False,
        "no_reuse_baseline": True,
        "telemetry_tags": [
            "kv_prefix_reuse",
            "no_reuse_baseline",
            "same_prefix_varied_suffix",
            "live_server_capture",
        ],
        "common_prefix_token_ids": list(common_prefix),
        "requests": rows,
        "server_report_source": server_report.get("source"),
        "server_report_claim_boundary": server_report.get("claim_boundary"),
        "server_observations": server_observations,
        "server_observed_metadata_only": metadata_only,
        "server_observed_kv_cache_reuse": server_report.get("server_observed_kv_cache_reuse") is True,
        "live_kv_cache_reuse_proven": live_reuse_proven,
        "speedup_proven": False,
        "can_update_demo_status": False,
        "claim_limitations": [
            "Assembler only; proof depends on strict kv_prefix_reuse_proof.py verification output.",
            "Metadata-only prefill reports do not prove live KV tensor reuse and are forced fail-closed.",
            "This artifact does not update demo status; review real live-run evidence before promotion.",
        ],
    }


def build_kv_prefix_reuse_live_capture_assembler_plan(
    *,
    model_id: str,
    evidence_path: str = DEFAULT_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Return the live capture assembler recipe without claiming proof."""

    assemble_command = (
        "python -m mvp_capabilities.kv_prefix_reuse_live_capture assemble "
        f"--model {model_id} --common-prefix-token-ids '<COMMON_PREFIX_IDS>' "
        "--server-report .local/kv-prefix-reuse-server-report.json "
        "--request suffix-a:<SUFFIX_A_IDS>:.local/kv-prefix-baseline-suffix-a.json:.local/kv-prefix-reuse-suffix-a.json "
        "--request suffix-b:<SUFFIX_B_IDS>:.local/kv-prefix-baseline-suffix-b.json:.local/kv-prefix-reuse-suffix-b.json "
        f"--out {evidence_path}"
    )
    verify_command = (
        "python -m mvp_capabilities.kv_prefix_reuse_proof verify "
        f"--model {model_id} --evidence {evidence_path}"
    )
    return {
        "model_id": model_id,
        "claim_boundary": PLAN_CLAIM_BOUNDARY,
        "proof_gate": PROOF_GATE,
        "evidence_path": evidence_path,
        "operator_commands": [
            f"Start the target server with {OPT_IN_FLAG}=1 and a KV-cache reuse implementation enabled.",
            "Capture no-reuse baseline JSON rows and reuse-path JSON rows for at least two same-prefix varied-suffix requests.",
            "Capture a server report that explicitly states server_observed_kv_cache_reuse=true and live_kv_cache_reuse_proven=true.",
            assemble_command,
            verify_command,
        ],
        "assemble_command": assemble_command,
        "verify_command": verify_command,
        "live_kv_cache_reuse_proven": False,
        "speedup_proven": False,
        "can_update_proof_status": False,
        "can_update_demo_status": False,
        "notes": [
            "Planning output is not proof.",
            "The assembler refuses to infer KV tensor reuse from client-only rows or metadata-only reports.",
            "Promotion still requires kv_prefix_reuse_proof.py to pass on real live-server cache-reuse evidence and measured memory/wall-clock impact.",
        ],
    }


def _parse_request_spec(raw: str) -> tuple[str, list[int], Path, Path]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("request spec must be request_id:suffix_csv:baseline_json:reuse_json")
    request_id, suffix_csv, baseline, reuse = parts
    if not request_id:
        raise argparse.ArgumentTypeError("request_id must be non-empty")
    suffix = _parse_token_csv(suffix_csv, field="suffix_csv")
    return request_id, suffix, Path(baseline), Path(reuse)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Emit a claim-bounded live capture assembler plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--evidence", default=DEFAULT_EVIDENCE_PATH)
    plan.add_argument("--out", default=None)

    assemble = sub.add_parser("assemble", help="Assemble live capture rows into KV-prefix reuse verifier evidence")
    assemble.add_argument("--model", required=True)
    assemble.add_argument("--common-prefix-token-ids", required=True)
    assemble.add_argument("--server-report", required=True)
    assemble.add_argument(
        "--request",
        action="append",
        required=True,
        help="request_id:suffix_csv:baseline_json:reuse_json; pass once per request",
    )
    assemble.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = build_kv_prefix_reuse_live_capture_assembler_plan(
            model_id=args.model,
            evidence_path=args.evidence,
        )
    else:
        baseline_by_request: dict[str, dict[str, Any]] = {}
        reuse_by_request: dict[str, dict[str, Any]] = {}
        suffix_by_request: dict[str, list[int]] = {}
        source_artifacts: dict[str, dict[str, str]] = {}
        for raw in args.request:
            request_id, suffix, baseline_path, reuse_path = _parse_request_spec(raw)
            if request_id in baseline_by_request:
                raise SystemExit(f"duplicate request_id: {request_id}")
            baseline_by_request[request_id] = _read_json(baseline_path)
            reuse_by_request[request_id] = _read_json(reuse_path)
            suffix_by_request[request_id] = suffix
            source_artifacts[request_id] = {
                "baseline": str(baseline_path),
                "reuse": str(reuse_path),
            }
        payload = assemble_kv_prefix_reuse_live_capture_evidence(
            model_id=args.model,
            common_prefix_token_ids=_parse_token_csv(args.common_prefix_token_ids, field="common_prefix_token_ids"),
            suffix_token_ids_by_request=suffix_by_request,
            baseline_by_request=baseline_by_request,
            reuse_by_request=reuse_by_request,
            server_report=_read_json(args.server_report),
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
