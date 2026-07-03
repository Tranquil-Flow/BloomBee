#!/usr/bin/env python3
"""Simulate BloomBee MVP routing and layer placement under peer failures.

The simulator is deliberately claim-bounded: it never starts BloomBee servers and
never proves inference. It lets operators rehearse variable-device demo rosters,
failed peers, and synthetic load before the live swarm exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mvp_capabilities.layer_planner import plan_layer_placement
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, choose_best_route, load_registry, synthetic_m4_laptops
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, summarize_roster
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/swarm_simulator.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.layer_planner import plan_layer_placement
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, load_proof_status
    from mvp_capabilities.route_picker import DEFAULT_REGISTRY, choose_best_route, load_registry, synthetic_m4_laptops
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, summarize_roster

CLAIM_BOUNDARY = "simulation_only_no_inference_proof"


def _find_model(registry: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in registry:
        if model.get("model_id") == model_id:
            return model
    raise ValueError(f"model not found in registry: {model_id}")


def _build_peers(
    *,
    cap_dirs: list[str | Path] | None,
    synthetic_m4_laptops: int,
    synthetic_total_gb: float,
    synthetic_free_gb: float,
) -> list[dict[str, Any]]:
    # Pure synthetic scenarios must stay pure even on machines that already have
    # ~/.bloombee/capabilities populated. Load default live capabilities only
    # when no synthetic peers and no explicit cap-dir were provided.
    if cap_dirs:
        peers = load_roster(cap_dirs)
    elif synthetic_m4_laptops:
        peers = []
    else:
        peers = load_roster([DEFAULT_CAP_DIR])
    if synthetic_m4_laptops:
        peers.extend(
            synthetic_m4_laptops_fn(
                count=synthetic_m4_laptops,
                total_gb=synthetic_total_gb,
                free_gb=synthetic_free_gb,
            )
        )
    return peers


# Indirection keeps tests able to monkeypatch if needed while preserving the
# public imported name from route_picker.
synthetic_m4_laptops_fn = synthetic_m4_laptops


def _filter_failed_hosts(peers: list[dict[str, Any]], failed_hosts: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    failed_set = set(failed_hosts)
    seen_hosts = {str(peer.get("hostname") or "unknown") for peer in peers}
    active = [peer for peer in peers if str(peer.get("hostname") or "unknown") not in failed_set]
    missing = [host for host in failed_hosts if host not in seen_hosts]
    return active, missing


def simulate_swarm(
    *,
    scenario: str | None = None,
    model_id: str,
    cap_dirs: list[str | Path] | None = None,
    synthetic_m4_laptops: int = 0,
    synthetic_total_gb: float = 24.0,
    synthetic_free_gb: float = 20.0,
    failed_hosts: list[str] | None = None,
    request_count: int = 1,
    registry_path: str | Path = DEFAULT_REGISTRY,
    proof_status_path: str | Path = DEFAULT_PROOF_STATUS,
    selector_mode: str = "planning",
) -> dict[str, Any]:
    failed_hosts = list(failed_hosts or [])
    registry = load_registry(registry_path)
    proof_status = load_proof_status(proof_status_path)
    model = _find_model(registry, model_id)
    peers = _build_peers(
        cap_dirs=cap_dirs,
        synthetic_m4_laptops=synthetic_m4_laptops,
        synthetic_total_gb=synthetic_total_gb,
        synthetic_free_gb=synthetic_free_gb,
    )
    active_peers, missing_failed_hosts = _filter_failed_hosts(peers, failed_hosts)

    route = choose_best_route(
        active_peers,
        registry,
        scenario=scenario,
        requested_model=model_id,
        proof_status=proof_status,
        selector_mode=selector_mode,
    )
    layer_plan = plan_layer_placement(active_peers, model)

    return {
        "scenario": scenario,
        "model_id": model_id,
        "selector_mode": selector_mode,
        "request_count": int(request_count),
        "input_peer_count": len(peers),
        "active_peer_count": len(active_peers),
        "failed_hosts": failed_hosts,
        "missing_failed_hosts": missing_failed_hosts,
        "active_roster_summary": summarize_roster(active_peers),
        "route": route,
        "layer_plan": layer_plan,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--proof-status", default=str(DEFAULT_PROOF_STATUS))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--selector-mode", default="planning", choices=("planning", "showcase-attempt", "safe-demo"))
    parser.add_argument("--cap-dir", action="append", default=None)
    parser.add_argument("--synthetic-m4-laptops", type=int, default=0)
    parser.add_argument("--synthetic-total-gb", type=float, default=24.0)
    parser.add_argument("--synthetic-free-gb", type=float, default=20.0)
    parser.add_argument("--fail-host", action="append", default=[])
    parser.add_argument("--request-count", type=int, default=1)
    args = parser.parse_args(argv)

    payload = simulate_swarm(
        scenario=args.scenario,
        model_id=args.model,
        cap_dirs=args.cap_dir,
        synthetic_m4_laptops=args.synthetic_m4_laptops,
        synthetic_total_gb=args.synthetic_total_gb,
        synthetic_free_gb=args.synthetic_free_gb,
        failed_hosts=args.fail_host,
        request_count=args.request_count,
        registry_path=args.registry,
        proof_status_path=args.proof_status,
        selector_mode=args.selector_mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
