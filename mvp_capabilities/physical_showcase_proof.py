#!/usr/bin/env python3
"""Verify operator-captured physical/self-serve showcase evidence.

This verifier does not start servers, scan cameras, or run inference. It is a
fail-closed claim boundary for the final MVP gate: a real fresh device must join
through a physical QR/link path, emit repeated successful heartbeats, appear in
operator-visible dashboard evidence, and the selected model must already have
proof-backed generation/load gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status

INPUT_CLAIM_BOUNDARY = "physical_showcase_operator_evidence"
PASSED_CLAIM_BOUNDARY = "verified_physical_showcase_evidence"
FAILED_CLAIM_BOUNDARY = "physical_showcase_evidence_failed_closed"
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


def verify_physical_showcase_evidence(
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

    passed = not failed_checks
    return {
        "claim_boundary": PASSED_CLAIM_BOUNDARY if passed else FAILED_CLAIM_BOUNDARY,
        "status": "passed" if passed else "failed",
        "proof_gate": "physical_showcase",
        "model_id": model_id,
        "selected_model_proof_status": selected_model_status,
        "fresh_device_count": len(physical_devices),
        "heartbeat_result_count": len(heartbeat_results),
        "successful_heartbeat_count": len(ok_heartbeats),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Operator-captured physical-showcase evidence JSON")
    parser.add_argument("--proof-status", default=DEFAULT_PROOF_STATUS, help="Proof status YAML (default: mvp_capabilities/PROOF_STATUS.yaml)")
    parser.add_argument("--min-heartbeat-results", type=int, default=3)
    args = parser.parse_args(argv)

    report = verify_physical_showcase_evidence(
        load_json(args.evidence),
        proof_status=load_proof_status(args.proof_status),
        min_heartbeat_results=args.min_heartbeat_results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
