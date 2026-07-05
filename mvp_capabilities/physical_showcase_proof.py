#!/usr/bin/env python3
"""Verify operator-captured physical/self-serve showcase evidence.

This verifier does not start servers, scan cameras, or run inference. It is a
fail-closed claim boundary for the final MVP gate: a real fresh device must join
through a physical QR/link path, emit repeated successful heartbeats, appear in
operator-visible dashboard evidence, and the selected model proof must be tied
back to the joined devices that served the showcase.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status
    from mvp_capabilities.multi_request_load_proof import qwen_load_metadata_report
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status
    from mvp_capabilities.multi_request_load_proof import qwen_load_metadata_report

INPUT_CLAIM_BOUNDARY = "physical_showcase_operator_evidence"
PASSED_CLAIM_BOUNDARY = "verified_physical_showcase_evidence"
FAILED_CLAIM_BOUNDARY = "physical_showcase_evidence_failed_closed"
CROSS_ARTIFACT_CLAIM_BOUNDARY = "physical_showcase_evidence_verifier_no_remote_execution"
HEARTBEAT_LOOP_CLAIM_BOUNDARY = "join_client_heartbeat_loop_only_no_inference_proof"
REQUIRED_MODEL_GATES = (
    "prescan",
    "one_block_server",
    "multi_block",
    "full_generation",
    "cache_generation",
    "multi_request_load",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _heartbeat_ok(row: Any) -> bool:
    payload = _as_mapping(row)
    server_response = _as_mapping(payload.get("server_response"))
    return server_response.get("ok") is True


def _proof_status_for_model(proof_status: Mapping[str, Mapping[str, str]], model_id: str) -> dict[str, str]:
    raw = proof_status.get(model_id) or {}
    return {gate: str(raw.get(gate, "pending")) for gate in REQUIRED_MODEL_GATES}


def _token_sha256_values(value: Any) -> list[str]:
    """Collect token hash values from nested evidence without exposing raw tokens."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "token_sha256" and isinstance(item, str) and item:
                found.append(item)
            else:
                found.extend(_token_sha256_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_token_sha256_values(item))
    return found


def _token_consistency_report(*values: Any) -> dict[str, Any]:
    unique = sorted(set().union(*(_token_sha256_values(value) for value in values)))
    return {
        "token_sha256_values": unique,
        "token_sha256_consistent": len(unique) <= 1,
    }


def _verify_operator_evidence(
    evidence: Mapping[str, Any],
    *,
    proof_status: Mapping[str, Mapping[str, str]] | None = None,
    min_heartbeat_results: int = 3,
) -> dict[str, Any]:
    """Return fail-closed verification for one operator-captured showcase evidence blob."""
    failed_checks: list[str] = []
    proof_status = proof_status or {}

    model_id = str(evidence.get("model_id") or "")
    if not model_id:
        failed_checks.append("missing_model_id")

    if evidence.get("claim_boundary") != INPUT_CLAIM_BOUNDARY:
        failed_checks.append("unexpected_claim_boundary")

    selected_model_status = _proof_status_for_model(proof_status, model_id) if model_id else {}
    for gate, status in selected_model_status.items():
        if status != "passed":
            failed_checks.append(f"selected_model_{gate}_not_passed")

    fresh_join = _as_mapping(evidence.get("fresh_join"))
    if fresh_join.get("physical_scanner_interop_proven") is not True:
        failed_checks.append("physical_scanner_interop_unproven")

    heartbeat_loop = _as_mapping(fresh_join.get("heartbeat_loop"))
    if heartbeat_loop.get("claim_boundary") != HEARTBEAT_LOOP_CLAIM_BOUNDARY:
        failed_checks.append("heartbeat_loop_claim_boundary_missing")
    if heartbeat_loop.get("inference_proven") is True:
        failed_checks.append("heartbeat_loop_overclaims_inference")

    heartbeat_results = _as_list(heartbeat_loop.get("results"))
    ok_heartbeats = [row for row in heartbeat_results if _heartbeat_ok(row)]
    if len(ok_heartbeats) < min_heartbeat_results:
        failed_checks.append("insufficient_successful_heartbeats")

    fresh_devices = _as_list(fresh_join.get("fresh_devices"))
    physical_devices = [
        _as_mapping(device)
        for device in fresh_devices
        if _as_mapping(device).get("transport") == "physical_qr_scan"
    ]
    if not physical_devices:
        failed_checks.append("fresh_device_not_joined_via_physical_qr")

    device_ids = {str(device.get("peer_id")) for device in physical_devices if device.get("peer_id")}
    heartbeat_peer_ids = {str(_as_mapping(row).get("peer_id")) for row in ok_heartbeats if _as_mapping(row).get("peer_id")}
    if device_ids and not (device_ids & heartbeat_peer_ids):
        failed_checks.append("physical_device_missing_successful_heartbeat")

    dashboard = _as_mapping(evidence.get("dashboard"))
    observed_peer_ids = {str(peer) for peer in _as_list(dashboard.get("observed_peer_ids"))}
    if dashboard.get("rendered") is not True:
        failed_checks.append("dashboard_not_rendered")
    if device_ids and not (device_ids <= observed_peer_ids):
        failed_checks.append("dashboard_missing_fresh_device")

    token_report = _token_consistency_report(evidence)
    if token_report["token_sha256_consistent"] is not True:
        failed_checks.append("token_sha256_mismatch")

    passed = not failed_checks
    return {
        "claim_boundary": PASSED_CLAIM_BOUNDARY if passed else FAILED_CLAIM_BOUNDARY,
        "status": "passed" if passed else "failed",
        "proof_gate": "physical_showcase",
        "model_id": model_id,
        "verifier_params": {"min_heartbeat_results": min_heartbeat_results},
        "token_sha256_values": token_report["token_sha256_values"],
        "token_sha256_consistent": token_report["token_sha256_consistent"],
        "selected_model_proof_status": selected_model_status,
        "fresh_device_count": len(physical_devices),
        "heartbeat_result_count": len(heartbeat_results),
        "successful_heartbeat_count": len(ok_heartbeats),
        "physical_scanner_interop_proven": fresh_join.get("physical_scanner_interop_proven") is True,
        "physical_showcase_proven": passed,
        "phone_worker_proven": False,
        "qwen3_30b_generation_proven": False,
        "can_update_mvp_status": passed,
        "mvp_status_update": {"physical_showcase": "passed"} if passed else {},
        "failed_checks": failed_checks,
        "next_step": None if passed else "Capture real physical QR scan, repeated heartbeat loop, dashboard observation, and selected-model proof status before promoting physical_showcase.",
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_layer_span(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    start = _parse_int(row.get("start_layer"))
    end = _parse_int(row.get("end_layer"))
    if start is not None and end is not None:
        return start, end
    layers = row.get("layers")
    if isinstance(layers, list) and len(layers) == 2:
        return _parse_int(layers[0]), _parse_int(layers[1])
    block_range = row.get("block_range")
    if isinstance(block_range, str) and ":" in block_range:
        left, right = block_range.split(":", 1)
        return _parse_int(left), _parse_int(right)
    return None, None


def _normalized_plan_assignments(joined_layer_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    placement = _as_mapping(joined_layer_plan.get("placement"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(placement.get("assignments")):
        row = _as_mapping(item)
        start, end = _parse_layer_span(row)
        host = row.get("hostname") or row.get("host") or row.get("peer_id")
        rows.append({"host": str(host) if host else None, "layers": [start, end], "block_range": f"{start}:{end}" if start is not None and end is not None else None})
    return rows


def _normalized_generation_placements(generation_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    parity = _as_mapping(generation_evidence.get("parity"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(parity.get("server_placements")):
        row = _as_mapping(item)
        start, end = _parse_layer_span(row)
        host = row.get("host") or row.get("hostname") or row.get("peer_id")
        rows.append({"host": str(host) if host else None, "layers": [start, end], "server_maddr": row.get("server_maddr")})
    return rows


def _placement_key(row: Mapping[str, Any]) -> tuple[str | None, int | None, int | None]:
    layers = _as_list(row.get("layers"))
    start = _parse_int(layers[0]) if len(layers) > 0 else None
    end = _parse_int(layers[1]) if len(layers) > 1 else None
    host = row.get("host")
    return str(host) if host else None, start, end


def _fresh_active_peers(active_roster: Mapping[str, Any], *, now: int | float | None, max_age_seconds: int | float) -> list[dict[str, Any]]:
    fresh: list[dict[str, Any]] = []
    for item in _as_list(active_roster.get("active_peers")):
        row = dict(_as_mapping(item))
        timestamp = row.get("timestamp")
        age_ok = True
        if now is not None and isinstance(timestamp, (int, float)):
            age_ok = (float(now) - float(timestamp)) <= float(max_age_seconds)
        elif now is not None:
            age_ok = False
        if not age_ok:
            continue
        caps = _as_mapping(row.get("capabilities"))
        row["hostname"] = row.get("hostname") or caps.get("hostname") or row.get("peer_id")
        fresh.append(row)
    return fresh


def _generation_report(generation_evidence: Mapping[str, Any], joined_layer_plan: Mapping[str, Any]) -> dict[str, Any]:
    parity = _as_mapping(generation_evidence.get("parity"))
    verify = _as_mapping(generation_evidence.get("verify"))
    generation_status = "passed" if (
        generation_evidence.get("cache_generation_proven") is True
        and parity.get("ok") is True
        and parity.get("generated_ids_match") is True
        and parity.get("generated_text_match") is True
        and parity.get("next_token_match") is True
        and (verify.get("status") in (None, "passed"))
    ) else "failed"
    plan_assignments = _normalized_plan_assignments(joined_layer_plan)
    generation_placements = _normalized_generation_placements(generation_evidence)
    plan_keys = {_placement_key(row) for row in plan_assignments}
    generation_keys = {_placement_key(row) for row in generation_placements}
    placements_match = bool(plan_keys) and plan_keys == generation_keys
    return {
        "proof_gate": generation_evidence.get("proof_gate") or verify.get("proof_gate") or "cache_generation",
        "status": generation_status,
        "model_id": generation_evidence.get("model_id") or parity.get("model"),
        "server_placements_match_joined_plan": placements_match,
        "joined_plan_assignments": plan_assignments,
        "server_placements": generation_placements,
    }


def _load_report(load_evidence: Mapping[str, Any] | None, *, model_id: str | None, num_layers: int | None, require_load: bool) -> dict[str, Any]:
    if load_evidence is None:
        return {
            "proof_gate": "multi_request_load",
            "status": "not_required" if not require_load else "missing",
            "required": require_load,
            "request_count": 0,
        }
    request_results = [_as_mapping(row) for row in _as_list(load_evidence.get("request_results"))]
    successful = [
        row
        for row in request_results
        if row.get("ok") is True and row.get("outputs_finite") is True and row.get("grad_finite") is True
    ]
    block_range = load_evidence.get("block_range")
    start, end = (None, None)
    if isinstance(block_range, str):
        start, end = _parse_layer_span({"block_range": block_range})
    covers_model = True
    if require_load and num_layers is not None:
        covers_model = (start, end) == (0, num_layers)
    model_matches = True
    if model_id and load_evidence.get("model_id"):
        model_matches = load_evidence.get("model_id") == model_id
    load_model_id = str(load_evidence.get("model_id") or model_id or "")
    metadata_policy, metadata_failures = qwen_load_metadata_report(load_model_id, request_results)
    status = "passed" if (
        load_evidence.get("status") == "passed"
        and successful
        and len(successful) == len(request_results)
        and covers_model
        and model_matches
        and not metadata_failures
    ) else "failed"
    return {
        "proof_gate": load_evidence.get("proof_gate") or "multi_request_load",
        "status": status,
        "model_id": load_evidence.get("model_id"),
        "block_range": block_range,
        "request_count": len(request_results),
        "successful_request_count": len(successful),
        "covers_model": covers_model,
        "model_matches": model_matches,
        "metadata_policy": metadata_policy,
        "failed_checks": metadata_failures,
        "required": require_load,
    }


def _verify_cross_artifacts(
    *,
    active_roster_path: str | Path,
    joined_layer_plan_path: str | Path,
    generation_evidence_path: str | Path,
    load_evidence_path: str | Path | None = None,
    operator_evidence_path: str | Path | None = None,
    proof_status: Mapping[str, Mapping[str, str]] | None = None,
    now: int | float | None = None,
    max_heartbeat_age_seconds: int | float = 120,
    min_joined_peers: int = 1,
    min_heartbeat_results: int = 3,
    require_load: bool = True,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    proof_status = proof_status or {}
    active_roster = load_json(active_roster_path)
    joined_layer_plan = load_json(joined_layer_plan_path)
    generation_evidence = load_json(generation_evidence_path)
    load_evidence = load_json(load_evidence_path) if load_evidence_path else None
    operator_evidence = load_json(operator_evidence_path) if operator_evidence_path else None

    fresh_peers = _fresh_active_peers(active_roster, now=now, max_age_seconds=max_heartbeat_age_seconds)
    fresh_peer_ids = sorted(str(peer.get("peer_id")) for peer in fresh_peers if peer.get("peer_id"))
    fresh_hostnames = {str(peer.get("hostname")) for peer in fresh_peers if peer.get("hostname")}
    if active_roster.get("claim_boundary") != "heartbeat_roster_only_no_inference_proof":
        failed_checks.append("active roster claim boundary missing")
    if len(fresh_peers) < min_joined_peers:
        failed_checks.append("insufficient fresh joined peers")

    placement = _as_mapping(joined_layer_plan.get("placement"))
    selected_model = joined_layer_plan.get("model_id") or generation_evidence.get("model_id") or _as_mapping(generation_evidence.get("parity")).get("model")
    num_layers = _parse_int(placement.get("num_layers"))
    assigned_layers = _parse_int(placement.get("assigned_layers"))
    plan_assignments = _normalized_plan_assignments(joined_layer_plan)
    if placement.get("supported") is not True:
        failed_checks.append("joined layer plan is not supported")
    if num_layers is None or assigned_layers != num_layers:
        failed_checks.append("joined layer plan does not cover every model layer")
    if fresh_hostnames and any(row.get("host") not in fresh_hostnames for row in plan_assignments):
        failed_checks.append("joined layer plan uses hosts that are not fresh active peers")

    selected_model_status = _proof_status_for_model(proof_status, str(selected_model)) if selected_model else {}
    for gate, status in selected_model_status.items():
        if status != "passed":
            failed_checks.append(f"selected model {gate} not passed")

    generation = _generation_report(generation_evidence, joined_layer_plan)
    if generation.get("model_id") and selected_model and generation.get("model_id") != selected_model:
        failed_checks.append("generation model does not match joined layer plan")
    if generation["status"] != "passed":
        failed_checks.append("generation evidence did not pass cache_generation gate")
    if generation["server_placements_match_joined_plan"] is not True:
        failed_checks.append("generation server placements do not match joined layer plan")

    load = _load_report(load_evidence, model_id=str(selected_model) if selected_model else None, num_layers=num_layers, require_load=require_load)
    if require_load and load["status"] != "passed":
        failed_checks.append("load evidence did not pass multi_request_load gate")

    operator_report: dict[str, Any]
    if operator_evidence is None:
        operator_report = {"status": "missing", "physical_scanner_interop_proven": False}
        failed_checks.append("operator physical QR evidence missing")
    else:
        if operator_evidence.get("model_id") and selected_model and operator_evidence.get("model_id") != selected_model:
            failed_checks.append("operator evidence model does not match joined layer plan")
        operator_report = _verify_operator_evidence(
            operator_evidence,
            proof_status=proof_status,
            min_heartbeat_results=min_heartbeat_results,
        )
        if operator_report["status"] != "passed":
            failed_checks.extend(f"operator evidence failed: {item}" for item in operator_report.get("failed_checks", []))

    token_report = _token_consistency_report(active_roster, operator_evidence or {})
    if token_report["token_sha256_consistent"] is not True:
        failed_checks.append("token_sha256_mismatch")

    passed = not failed_checks
    return {
        "claim_boundary": CROSS_ARTIFACT_CLAIM_BOUNDARY,
        "source": "physical_showcase_proof.py",
        "proof_gate": "physical_showcase",
        "status": "passed" if passed else "failed",
        "physical_showcase_proven": passed,
        "inference_proven": passed,
        "selected_model": selected_model,
        "verifier_params": {
            "now": now,
            "max_heartbeat_age_seconds": max_heartbeat_age_seconds,
            "min_joined_peers": min_joined_peers,
            "min_heartbeat_results": min_heartbeat_results,
            "require_load": require_load,
        },
        "token_sha256_values": token_report["token_sha256_values"],
        "token_sha256_consistent": token_report["token_sha256_consistent"],
        "selected_model_proof_status": selected_model_status,
        "required_joined_peer_count": min_joined_peers,
        "fresh_joined_peer_count": len(fresh_peers),
        "fresh_peer_ids": fresh_peer_ids,
        "join": {
            "self_serve_join_proven": len(fresh_peers) >= min_joined_peers,
            "claim_boundary": active_roster.get("claim_boundary"),
        },
        "layer_plan": {
            "source": joined_layer_plan.get("source"),
            "supported": placement.get("supported") is True,
            "assignments": plan_assignments,
            "assigned_layers": assigned_layers,
            "num_layers": num_layers,
        },
        "generation": generation,
        "load": load,
        "operator_evidence": {
            "status": operator_report.get("status"),
            "physical_scanner_interop_proven": operator_report.get("physical_scanner_interop_proven") is True,
            "failed_checks": operator_report.get("failed_checks", []),
        },
        "failed_checks": failed_checks,
        "can_update_mvp_status": passed,
        "can_update_proof_status": False,
        "mvp_status_update": {"physical_showcase": "passed"} if passed else {},
    }


def verify_physical_showcase_evidence(
    evidence: Mapping[str, Any] | None = None,
    *,
    proof_status: Mapping[str, Mapping[str, str]] | None = None,
    min_heartbeat_results: int = 3,
    active_roster_path: str | Path | None = None,
    joined_layer_plan_path: str | Path | None = None,
    generation_evidence_path: str | Path | None = None,
    load_evidence_path: str | Path | None = None,
    operator_evidence_path: str | Path | None = None,
    now: int | float | None = None,
    max_heartbeat_age_seconds: int | float = 120,
    min_joined_peers: int = 1,
    require_load: bool = True,
) -> dict[str, Any]:
    """Verify either a single operator evidence blob or a cross-artifact showcase bundle."""
    if active_roster_path or joined_layer_plan_path or generation_evidence_path:
        if not (active_roster_path and joined_layer_plan_path and generation_evidence_path):
            raise ValueError("active_roster_path, joined_layer_plan_path, and generation_evidence_path are required together")
        return _verify_cross_artifacts(
            active_roster_path=active_roster_path,
            joined_layer_plan_path=joined_layer_plan_path,
            generation_evidence_path=generation_evidence_path,
            load_evidence_path=load_evidence_path,
            operator_evidence_path=operator_evidence_path,
            proof_status=proof_status,
            now=now,
            max_heartbeat_age_seconds=max_heartbeat_age_seconds,
            min_joined_peers=min_joined_peers,
            min_heartbeat_results=min_heartbeat_results,
            require_load=require_load,
        )
    if evidence is None:
        raise ValueError("evidence is required when cross-artifact paths are not supplied")
    return _verify_operator_evidence(
        evidence,
        proof_status=proof_status,
        min_heartbeat_results=min_heartbeat_results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["verify"], default="verify")
    parser.add_argument("--evidence", help="Operator-captured physical-showcase evidence JSON")
    parser.add_argument("--active-roster", help="JSON from /active or join_http_server active handler")
    parser.add_argument("--joined-layer-plan", help="JSON from join_layer_plan.py or /plan")
    parser.add_argument("--generation-evidence", help="Cache/full generation evidence JSON with server_placements")
    parser.add_argument("--load-evidence", help="Optional multi_request_load evidence JSON")
    parser.add_argument("--operator-evidence", help="Optional physical QR/dashboard operator evidence JSON")
    parser.add_argument("--proof-status", default=DEFAULT_PROOF_STATUS, help="Proof status YAML (default: mvp_capabilities/PROOF_STATUS.yaml)")
    parser.add_argument("--min-heartbeat-results", type=int, default=3)
    parser.add_argument("--min-joined-peers", type=int, default=1)
    parser.add_argument("--max-heartbeat-age-seconds", type=float, default=120.0)
    parser.add_argument("--now", type=float, default=None)
    parser.add_argument("--no-require-load", action="store_true", help="Allow physical-showcase cross-checks to omit multi-request load evidence")
    args = parser.parse_args(argv)

    proof_status = load_proof_status(args.proof_status)
    if args.active_roster or args.joined_layer_plan or args.generation_evidence:
        report = verify_physical_showcase_evidence(
            proof_status=proof_status,
            min_heartbeat_results=args.min_heartbeat_results,
            active_roster_path=args.active_roster,
            joined_layer_plan_path=args.joined_layer_plan,
            generation_evidence_path=args.generation_evidence,
            load_evidence_path=args.load_evidence,
            operator_evidence_path=args.operator_evidence or args.evidence,
            now=args.now,
            max_heartbeat_age_seconds=args.max_heartbeat_age_seconds,
            min_joined_peers=args.min_joined_peers,
            require_load=not args.no_require_load,
        )
    else:
        if not args.evidence:
            parser.error("--evidence is required unless --active-roster/--joined-layer-plan/--generation-evidence are supplied")
        report = verify_physical_showcase_evidence(
            load_json(args.evidence),
            proof_status=proof_status,
            min_heartbeat_results=args.min_heartbeat_results,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
