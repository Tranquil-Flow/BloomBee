#!/usr/bin/env python3
"""Report weighted Distributed Inference MVP progress.

This is a planning/status tool, not demo proof. It encodes the current plan into
weighted milestones so operators can see how much has been built and what gate is
next without reading every doc.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

CLAIM_BOUNDARY = "weighted_plan_status_not_demo_proof"
NEXT_GATE = "Qwen3-8B multi-block or full-generation proof"


@dataclass(frozen=True)
class Milestone:
    id: str
    label: str
    weight: int
    completion: float
    status: str
    evidence: str
    next_step: str | None = None


MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        id="model_foundation",
        label="Model catalog, compatibility scanner, proof-status registry, proof ladder audit",
        weight=12,
        completion=1.00,
        status="complete",
        evidence="MODEL_REGISTRY.yaml, model_compat_scan.py, PROOF_STATUS.yaml, proof_ladder.py",
    ),
    Milestone(
        id="dynamic_selector",
        label="Prepared model ladder and proof-aware selector modes",
        weight=10,
        completion=1.00,
        status="complete",
        evidence="route_picker.py supports planning/showcase-attempt/safe-demo; Qwen3 2507 variants registered",
    ),
    Milestone(
        id="dashboard_visibility",
        label="Dashboard/operator visibility with claim boundaries",
        weight=8,
        completion=1.00,
        status="complete",
        evidence="demo_dashboard.py renders real peers, route cards, evidence, telemetry, layer placement, mvp_status.py progress/next gate, proof_state.py live prep feed, and coordinator /handoff proof-runbook bundles",
    ),
    Milestone(
        id="join_flow",
        label="Self-serve join flow, QR/link, heartbeat, live coordinator service",
        weight=10,
        completion=0.93,
        status="partial",
        evidence="join_coordinator.py creates link offers/heartbeats; join_http_server.py exposes health/offer/heartbeat/active/route/plan/bootstrap/handoff endpoints with auto model selection, token-scoped bootstrap scripts, and operator proof-runbook bundles; join_handoff.py fetches/redacts dashboard-ready handoff artifacts; join_client.py posts one-shot or bounded repeated peer heartbeats; join_card.py renders SVG cards plus exact URL JSON/TXT sidecars; join_qr_preflight.py reports scanner-proof dependency blockers fail-closed; generated follower launch runbooks use the verified BLOOMBEE_INITIAL_PEERS join path",
        next_step="install QR encoder+decoder deps, prove exact QR decode, then run repeated-heartbeat fresh-device showcase through the HTTP coordinator",
    ),
    Milestone(
        id="layer_planning",
        label="Layer planner and launch-ready worker assignment",
        weight=10,
        completion=1.00,
        status="complete",
        evidence="layer_planner.py emits deterministic ranges/runbooks; join_layer_plan.py converts local or HTTP /active coordinator heartbeats into layer plans, resolves operator-captured seed multiaddrs, and emits no-execution readiness checklists",
    ),
    Milestone(
        id="simulation_harness",
        label="Variable-device simulation harness",
        weight=5,
        completion=1.00,
        status="complete",
        evidence="swarm_simulator.py rehearses live/synthetic rosters and failed hosts",
    ),
    Milestone(
        id="tinyllama_runtime_proof",
        label="TinyLlama distributed fallback proof ladder",
        weight=8,
        completion=1.00,
        status="complete",
        evidence="TinyLlama forward/backward, text parity, cached generate parity, multi-peer evidence",
    ),
    Milestone(
        id="qwen3_dense_fallbacks",
        label="Qwen3 dense fallback proof ladder: 8B then 14B",
        weight=8,
        completion=0.70,
        status="partial",
        evidence="Qwen3-8B prescan and one-block server proof passed on M4 Pro; Qwen3-14B config-only prescan passed; Qwen3-8B multi-block, full-generation, cache-generation, and load proof harnesses now exist but live gates remain pending",
        next_step="Run Qwen3-8B multi-block, full-generation, or cache-generation parity proof, then Qwen3-14B one-block proof if memory allows",
    ),
    Milestone(
        id="qwen3_30b_proof_ladder",
        label="Qwen3-30B-A3B / Instruct-2507 multi-block and full-generation proof ladder",
        weight=15,
        completion=0.15,
        status="partial",
        evidence="Qwen3-MoE wrapper exists; one live M4 Pro block shard proof passed for Qwen3-30B-A3B",
        next_step="multi-block Qwen3-30B direct RPC proof",
    ),
    Milestone(
        id="chain_scheduler",
        label="Multi-request chain scheduler and load proof",
        weight=8,
        completion=0.50,
        status="partial",
        evidence="chain_scheduler.py turns joined layer plans into multi-request waves, per-peer load estimates, and no-live-traffic health reports; request_telemetry.py summarizes direct-client success/failure and latency logs for dashboards; multi_request_load_proof.py verifies repeated direct-client logs before proof promotion",
        next_step="send multi-request live traffic through started servers and pass multi_request_load_proof.py verify",
    ),
    Milestone(
        id="physical_showcase",
        label="Physical/self-serve live showcase with fresh joined devices",
        weight=6,
        completion=0.00,
        status="pending",
        evidence="not yet run",
        next_step="run fresh QR/link joined laptop swarm and prove selected generation",
    ),
)


def render_bar(percent: int, *, width: int = 20) -> str:
    filled = round(width * max(0, min(percent, 100)) / 100)
    return f"{'█' * filled}{'░' * (width - filled)} {percent}%"


def _milestone_payload(item: Milestone) -> dict[str, Any]:
    weighted_points = item.weight * item.completion
    return {
        "id": item.id,
        "label": item.label,
        "weight": item.weight,
        "completion": item.completion,
        "percent": round(item.completion * 100),
        "weighted_points": round(weighted_points, 2),
        "status": item.status,
        "evidence": item.evidence,
        "next_step": item.next_step,
    }


def build_status_report() -> dict[str, Any]:
    total_weight = sum(item.weight for item in MILESTONES)
    earned = sum(item.weight * item.completion for item in MILESTONES)
    overall_percent = round(earned / total_weight * 100) if total_weight else 0
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "overall_percent": overall_percent,
        "remaining_percent": 100 - overall_percent,
        "overall_bar": render_bar(overall_percent),
        "earned_weighted_points": round(earned, 2),
        "total_weight": total_weight,
        "next_gate": NEXT_GATE,
        "interpretation": (
            "Weighted engineering-build progress from the current MVP plan. "
            "It is not a public-demo proof percentage and does not promote unproven models."
        ),
        "milestones": [_milestone_payload(item) for item in MILESTONES],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Distributed Inference MVP status",
        "",
        f"**Built from plan:** `{report['overall_bar']}`",
        f"**Remaining:** `{report['remaining_percent']}%`",
        f"**Next gate:** {report['next_gate']}",
        f"**Claim boundary:** `{report['claim_boundary']}`",
        "",
        report["interpretation"],
        "",
        "| Milestone | Weight | Status | Built | Evidence / next step |",
        "|---|---:|---|---:|---|",
    ]
    for item in report["milestones"]:
        evidence = item["evidence"]
        if item.get("next_step"):
            evidence = f"{evidence}<br>Next: {item['next_step']}"
        lines.append(
            f"| {item['label']} | {item['weight']} | {item['status']} | "
            f"{item['percent']}% | {evidence} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown")
    args = parser.parse_args(argv)

    report = build_status_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
