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


@dataclass(frozen=True)
class PlanTask:
    id: str
    label: str
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
        evidence="route_picker.py supports planning/showcase-attempt/safe-demo, infers registry HF model_type, and blocks unsupported wrappers from showcase/safe-demo; Qwen3 2507 variants registered",
    ),
    Milestone(
        id="dashboard_visibility",
        label="Dashboard/operator visibility with claim boundaries",
        weight=8,
        completion=1.00,
        status="complete",
        evidence="demo_dashboard.py renders real peers, route cards, evidence, telemetry, layer placement, mvp_status.py progress/next gate, proof_state.py live prep feed, speculative decode plans, multi-block diagnostics, and coordinator /handoff bootstrap/speculative/proof-runbook bundles",
    ),
    Milestone(
        id="join_flow",
        label="Self-serve join flow, QR/link, heartbeat, live coordinator service",
        weight=10,
        completion=0.98,
        status="partial",
        evidence="join_coordinator.py creates link offers/heartbeats; join_http_server.py exposes health/offer/heartbeat/active/route/plan/speculative/bootstrap/bootstrap.sh/handoff/proof-orchestration endpoints with auto model selection, token-scoped JSON and plain-shell bootstrap scripts, verifier-authoritative speculative plans, embedded proof orchestration checklists, and operator proof-runbook bundles; join_handoff.py fetches/redacts dashboard-ready handoff artifacts; join_client.py posts one-shot or bounded repeated peer heartbeats; join_card.py renders SVG cards plus exact URL JSON/TXT sidecars; join_qr_preflight.py reports scanner-proof dependency blockers fail-closed; join_qr_proof.py generated a true QR PNG and decoded it back to the exact redacted join URL locally with qrcode+PIL/cv2; generated follower launch runbooks use current run_server --initial_peers join path",
        next_step="scan the generated QR artifact with physical devices, then run repeated-heartbeat fresh-device showcase through the HTTP coordinator",
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
        label="Multi-request chain scheduler, speculative decode scaffold, and load proof",
        weight=8,
        completion=0.65,
        status="partial",
        evidence="chain_scheduler.py turns joined layer plans into multi-request waves, per-peer load estimates, and no-live-traffic health reports; proof_orchestrator.py orders handoff launch/proof runbooks and blocks unresolved placeholders or legacy peer flags before operator execution; speculative_decode_plan.py defines verifier-authoritative draft-provider plans and phone-as-draft-only policy; draft_provider.py adds a deterministic DraftProvider.propose contract with verifier-prefix accepted/rejected counters for dashboard smoke reports; draft_provider_bridge.py exposes the same contract over stdio JSONL for Termux/ADB/SSH bridge experiments; termux_draft_smoke.py and termux_draft_latency.py render/verify self-contained Termux phone evidence; Pixel 8 Pro Termux smoke evidence passed at mvp_capabilities/distributed_evidence/phone/termux-draft-smoke-20260704T095557Z.json, 50-iteration static-contract latency passed at mvp_capabilities/distributed_evidence/phone/termux-draft-latency-20260704T100644Z.json, tiny-model feasibility blockers are tracked at mvp_capabilities/distributed_evidence/phone/termux-tiny-model-probe-20260704T101232Z.json, a no-install GGUF runtime plan is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-plan-20260704T101232Z.json, and real Pixel 8 Pro Termux llama.cpp/stories15M GGUF generation is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-generation-20260704T104506Z.json; request_telemetry.py summarizes direct-client success/failure and latency logs for dashboards, treating zero latency as unmeasured; multi_request_load_proof.py verifies repeated direct-client logs and now blocks unmeasured latency before proof promotion",
        next_step="send multi-request live traffic through started servers and pass multi_request_load_proof.py verify; wrap the phone GGUF model as a draft-provider bridge and compare accepted tokens against an authoritative verifier before claiming speculative speedup",
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


PLANNED_TASKS: tuple[PlanTask, ...] = (
    PlanTask(
        id="join_link_foundation",
        label="Join-link offer, heartbeat roster, bootstrap script, and handoff bundle",
        status="complete",
        evidence="join_coordinator.py, join_http_server.py, join_client.py, join_handoff.py, and token-scoped heartbeat state exist and are test-covered",
    ),
    PlanTask(
        id="fresh_laptop_join",
        label="Fresh laptop can join through link/QR without bespoke setup",
        status="partial",
        evidence="copy/paste join URL, bootstrap.sh, bounded heartbeat client, SVG join card sidecars, QR dependency preflight, and local true-QR exact decode proof exist; physical camera scanner interop and fresh-device heartbeat loop remain unproven",
        next_step="scan the generated QR artifact with physical devices, then run a fresh physical-device heartbeat loop",
    ),
    PlanTask(
        id="dashboard_real_devices",
        label="Dashboard shows real connected devices and live claim boundaries",
        status="complete",
        evidence="demo_dashboard.py renders real capability/join artifacts, MVP status, proof prep, route decisions, layer plans, handoff bundles, and telemetry panels",
    ),
    PlanTask(
        id="layer_assignment",
        label="Coordinator assigns concrete start:end layer ranges from joined peers",
        status="complete",
        evidence="layer_planner.py and join_layer_plan.py emit deterministic contiguous assignments plus no-server-start launch-readiness checks",
    ),
    PlanTask(
        id="server_launch_runbooks",
        label="Operator-ready BloomBee server launch runbooks",
        status="complete",
        evidence="seed commands use --new_swarm and follower commands use current run_server --initial_peers placeholders with explicit readiness blockers",
    ),
    PlanTask(
        id="tinyllama_distributed_generation",
        label="TinyLlama distributed fallback generation proof",
        status="complete",
        evidence="TinyLlama has two-server/two-laptop/three-peer forward-backward, forward-loop text parity, cached .generate() parity, and S2S opportunistic fallback evidence",
    ),
    PlanTask(
        id="qwen3_8b_proof",
        label="Qwen3-8B multi-block or full-generation proof",
        status="partial",
        evidence="Qwen3-8B prescan and one-block server proof passed; multi-block/full-generation/cache-generation/load harnesses exist but live gates remain pending",
        next_step="run Qwen3-8B multi-block or full-generation proof on a clean-memory M4 Pro session and promote only after verifier logs pass",
    ),
    PlanTask(
        id="qwen3_30b_core_proof",
        label="Qwen3-30B-A3B core laptop-swarm proof ladder",
        status="partial",
        evidence="qwen3_moe wrapper exists and one live M4 Pro Qwen3-30B-A3B block shard passed; full distributed generation remains pending",
        next_step="run multi-block Qwen3-30B direct RPC proof, then full-generation parity when enough clean memory/devices are available",
    ),
    PlanTask(
        id="qwen3_30b_2507_shelf",
        label="Prepared Qwen3-30B-A3B Instruct/Thinking 2507 shelf",
        status="partial",
        evidence="2507 variants are registered with config metadata and pending proof gates",
        next_step="run prescan and one-block live proof for each selected 2507 checkpoint before making either primary",
    ),
    PlanTask(
        id="qwen35b_candidate",
        label="Qwen35B candidate branch",
        status="blocked",
        evidence="Qwen/Qwen-AgentWorld-35B-A3B is memory-fit for synthetic 10-laptop planning but HF model_type=qwen3_5_moe / qwen3_5_moe_text lacks a BloomBee wrapper",
        next_step="add and prove native qwen3_5_moe wrapper before any showcase/safe-demo selection",
    ),
    PlanTask(
        id="minimax_m3_candidate",
        label="MiniMax M3 high-compute candidate",
        status="blocked",
        evidence="MiniMaxAI/MiniMax-M3 needs ~900GB runtime memory and lacks minimax_m3_vl + sparse-attention BloomBee support; quantized variants are not native BloomBee-compatible",
        next_step="defer to post-core LayerExecutor/quantized-backend path or implement native sparse-attention wrapper/kernels",
    ),
    PlanTask(
        id="multi_request_load",
        label="Multiple requests routed through healthy chains with visible utilisation",
        status="partial",
        evidence="chain_scheduler.py, request_telemetry.py, and multi_request_load_proof.py exist; no successful live multi-request traffic proof yet",
        next_step="send repeated direct-client traffic through started servers and verify multi_request_load proof logs with nonzero measured latency",
    ),
    PlanTask(
        id="speculative_decode",
        label="Speculative/draft-provider speedup plan",
        status="partial",
        evidence="speculative_decode_plan.py defines verifier-authoritative draft-provider roles and phone-as-draft-only policy; draft_provider.py provides a deterministic provider interface and accepted/rejected exact-token counters for dashboard smoke reports; draft_provider_bridge.py exposes stdio JSONL transport for phone/Termux bridge tests; termux_draft_smoke.py verified a real Pixel 8 Pro Termux draft-contract smoke with proposed=3 accepted=2 rejected=1; termux_draft_latency.py verified a 50-iteration Pixel 8 Pro static-contract loop with proposed=150 accepted=100 rejected=50 and latency p95=0.001669ms; termux_tiny_model_probe.py showed no torch/transformers/tokenizers/llama_cpp/bloombee installed; Termux llama.cpp CLI is now available and generated text from ggml-org/tiny-llamas/stories15M.gguf, but no verifier-accepted speculative decode or speedup proof exists yet",
        next_step="wrap the phone GGUF generation path behind the draft-provider bridge and compare against verifier-only baseline before speedup claims",
    ),
    PlanTask(
        id="phone_worker",
        label="Phone as useful inference or draft worker",
        status="partial",
        evidence="mobile capability fields exist in peer_scan.py; draft_provider.py defines the phone-compatible draft-provider contract; draft_provider_bridge.py provides stdio JSONL bridge groundwork; m4pro ADB pushed/typed short commands into Termux and verified real Pixel 8 Pro JSON evidence (Android SDK 36, Tensor G3, aarch64) for one-shot contract smoke plus 50-iteration static-contract latency p95=0.001669ms; feasibility probe showed 11.851GB total RAM, 2.557GB available, 28.425GB free storage, build tools present, and missing torch/transformers/tokenizers/llama_cpp/bloombee Python modules; after approval, Termux llama.cpp CLI plus ggml-org/tiny-llamas/stories15M.gguf generated `One day, a little girl named Lucy` in 0.347524s with SHA256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a; this is real tiny GGUF generation but still not BloomBee block serving or speculative speedup",
        next_step="wrap phone GGUF CLI as a draft provider and run verifier-accepted token comparison; separately prove BloomBee block serving before counting phone as a block worker",
    ),
    PlanTask(
        id="physical_showcase",
        label="Physical/self-serve N-laptop showcase",
        status="pending",
        evidence="not yet run",
        next_step="run fresh QR/link joined laptop swarm, launch selected model servers, and prove selected generation through the dashboard",
    ),
    PlanTask(
        id="continuous_batching",
        label="True continuous batching",
        status="pending",
        evidence="not yet implemented",
        next_step="post-MVP after correctness/showcase gates; keep separate from proof of basic distributed generation",
    ),
    PlanTask(
        id="kv_prefix_reuse",
        label="Real prefill KV prefix reuse",
        status="pending",
        evidence="not yet implemented",
        next_step="post-MVP optimization after cached-generation correctness remains stable",
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


def _task_payload(item: PlanTask) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label,
        "status": item.status,
        "done": item.status == "complete",
        "evidence": item.evidence,
        "next_step": item.next_step,
    }


def build_status_report() -> dict[str, Any]:
    total_weight = sum(item.weight for item in MILESTONES)
    earned = sum(item.weight * item.completion for item in MILESTONES)
    overall_percent = round(earned / total_weight * 100) if total_weight else 0
    planned_tasks = [_task_payload(item) for item in PLANNED_TASKS]
    task_summary = {
        status: sum(1 for item in planned_tasks if item["status"] == status)
        for status in ("complete", "partial", "pending", "blocked")
    }
    task_summary["total"] = len(planned_tasks)
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
        "planned_tasks": planned_tasks,
        "task_summary": task_summary,
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
    lines.extend(
        [
            "",
            "## Planned tasks",
            "",
            "| Task | Status | Done? | Evidence / next step |",
            "|---|---|---:|---|",
        ]
    )
    for item in report["planned_tasks"]:
        evidence = item["evidence"]
        if item.get("next_step"):
            evidence = f"{evidence}<br>Next: {item['next_step']}"
        done = "yes" if item["done"] else "no"
        lines.append(f"| {item['label']} | {item['status']} | {done} | {evidence} |")
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
