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
NEXT_GATE = "Qwen3-8B multi-block proof from clean m4pro archive"


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
        evidence="Qwen3-8B prescan and one-block server proof passed on M4 Pro; Qwen3-14B config-only prescan passed; Qwen3-8B multi-block, full-generation, cache-generation, and load proof harnesses now exist; clean-tree m4pro preflight at mvp_capabilities/distributed_evidence/qwen3-8b-clean-tree-preflight-20260704T122930Z.json confirms cache present, 48GB host memory, clean archive path, and Python 3.11 venv requirement, but live gates remain pending",
        next_step="Run Qwen3-8B multi-block from the clean archive on M4 Pro using the project Python 3.11 venv, then Qwen3-14B one-block proof if memory allows",
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
        completion=0.70,
        status="partial",
        evidence="chain_scheduler.py turns joined layer plans into multi-request waves, per-peer load estimates, and no-live-traffic health reports; proof_orchestrator.py orders handoff launch/proof runbooks and blocks unresolved placeholders or legacy peer flags before operator execution; speculative_decode_plan.py defines verifier-authoritative draft-provider plans and phone-as-draft-only policy; draft_provider.py adds a deterministic DraftProvider.propose contract with verifier-prefix accepted/rejected counters for dashboard smoke reports; draft_provider_bridge.py exposes the same contract over stdio JSONL for Termux/ADB/SSH bridge experiments; termux_draft_smoke.py and termux_draft_latency.py render/verify self-contained Termux phone evidence; Pixel 8 Pro Termux smoke evidence passed at mvp_capabilities/distributed_evidence/phone/termux-draft-smoke-20260704T095557Z.json, 50-iteration static-contract latency passed at mvp_capabilities/distributed_evidence/phone/termux-draft-latency-20260704T100644Z.json, tiny-model feasibility blockers are tracked at mvp_capabilities/distributed_evidence/phone/termux-tiny-model-probe-20260704T101232Z.json, a no-install GGUF runtime plan is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-plan-20260704T101232Z.json, real Pixel 8 Pro Termux llama.cpp/stories15M GGUF generation is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-generation-20260704T104506Z.json, phone GGUF draft-bridge smoke is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-bridge-20260704T105400Z.json, phone_draft_verifier_compare.py tracks UTF-8 verifier-prefix evidence including a live Qwen/Qwen2.5-0.5B-Instruct mismatch with accepted=0/33 at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-qwen05-20260704T110000Z.json plus an independent local same-GGUF verifier copied from the phone with accepted=33/33 at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-same-gguf-20260704T111215Z.json, termux-local tokenizer-ID comparison accepted 8/8 same-GGUF draft token IDs at mvp_capabilities/distributed_evidence/phone/termux-local-tokenizer-id-compare-20260704T111800Z.json, phone_speculative_wallclock_gate.py records the sequential phone-draft+verifier path as slower (2.403479s vs 1.837976s verifier-only) at mvp_capabilities/distributed_evidence/phone/termux-same-gguf-wallclock-gate-20260704T112500Z.json, local llama.cpp speculative harness accepted 8/8 draft tokens with same GGUF at mvp_capabilities/distributed_evidence/phone/local-same-gguf-llama-speculative-harness-20260704T113600Z.json but did not involve the phone, phone-integrated verifier preflight at mvp_capabilities/distributed_evidence/phone/phone-integrated-verifier-preflight-20260704T114000Z.json records the external-token-ID gap, and phone_llama_cpp_binding_verifier.py accepted the phone draft text bytes under the exact llama.cpp chat template at mvp_capabilities/distributed_evidence/phone/phone-llama-cpp-binding-verifier-20260704T120000Z.json, then ingested Termux-emitted context draft token IDs from mvp_capabilities/distributed_evidence/phone/termux-context-token-ids-20260704T121646Z.json and accepted 8/8 external phone context tokens at mvp_capabilities/distributed_evidence/phone/phone-context-token-id-verifier-20260704T121646Z.json; request_telemetry.py summarizes direct-client success/failure and latency logs for dashboards, treating zero latency as unmeasured; multi_request_load_proof.py verifies repeated direct-client logs and now blocks unmeasured latency before proof promotion",
        next_step="send multi-request live traffic through started servers and pass multi_request_load_proof.py verify; bridge live phone token transport into the verifier path, then measure phone-backed wall clock before any speculative speedup claim",
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
        evidence="Qwen3-8B prescan and one-block server proof passed; multi-block/full-generation/cache-generation/load harnesses exist; clean-tree m4pro preflight confirms Qwen3-8B cache, 48GB memory, current git archive path, and Python 3.11 venv requirement; live gates remain pending",
        next_step="run Qwen3-8B multi-block proof from the clean m4pro archive with project Python 3.11 venv and promote only after verifier logs pass",
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
        evidence="speculative_decode_plan.py defines verifier-authoritative draft-provider roles and phone-as-draft-only policy; draft_provider.py provides a deterministic provider interface and accepted/rejected exact-token counters for dashboard smoke reports; draft_provider_bridge.py exposes stdio JSONL transport for phone/Termux bridge tests; termux_draft_smoke.py verified a real Pixel 8 Pro Termux draft-contract smoke with proposed=3 accepted=2 rejected=1; termux_draft_latency.py verified a 50-iteration Pixel 8 Pro static-contract loop with proposed=150 accepted=100 rejected=50 and latency p95=0.001669ms; termux_tiny_model_probe.py showed no torch/transformers/tokenizers/llama_cpp/bloombee installed; Termux llama.cpp CLI generated text from ggml-org/tiny-llamas/stories15M.gguf; termux_gguf_draft_bridge.py verified a phone GGUF draft-provider-candidate JSON bridge; phone_draft_verifier_compare.py proves exact byte-prefix acceptance math; live Qwen/Qwen2.5-0.5B-Instruct verifier comparison rejected the phone draft with accepted=0/33, independent local same-GGUF verifier comparison accepted 33/33 bytes from the exact phone-copied GGUF, same-GGUF tokenizer-ID comparison accepted 8/8 draft token IDs, wall-clock gate shows sequential phone-draft+verifier is slower (2.403479s vs 1.837976s verifier-only), local llama.cpp speculative harness accepted 8/8 draft tokens with same GGUF but without phone involvement, preflight showed the raw llama.cpp CLI cannot ingest phone-provided external draft token IDs, llama-cpp-python binding verifier accepted the phone draft text bytes with context token IDs [6716, 2462, 29892, 263, 2217, 7826, 4257, 28846], and the binding verifier now ingests Termux-emitted context token IDs and accepts 8/8 external phone tokens; no phone-backed speculative speedup proof exists yet",
        next_step="bridge live phone token transport into the verifier path, then measure phone-backed wall clock before speedup claims",
    ),
    PlanTask(
        id="phone_worker",
        label="Phone as useful inference or draft worker",
        status="partial",
        evidence="mobile capability fields exist in peer_scan.py; draft_provider.py defines the phone-compatible draft-provider contract; draft_provider_bridge.py provides stdio JSONL bridge groundwork; m4pro ADB pushed/typed short commands into Termux and verified real Pixel 8 Pro JSON evidence (Android SDK 36, Tensor G3, aarch64) for one-shot contract smoke plus 50-iteration static-contract latency p95=0.001669ms; feasibility probe showed 11.851GB total RAM, 2.557GB available, 28.425GB free storage, build tools present, and missing torch/transformers/tokenizers/llama_cpp/bloombee Python modules; after approval, Termux llama.cpp CLI plus ggml-org/tiny-llamas/stories15M.gguf generated `One day, a little girl named Lucy` in 0.347524s with SHA256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a; termux_gguf_draft_bridge.py wrapped that phone generation as a draft-provider-candidate JSON bridge in 0.565503s; positive-control verifier comparison accepted 33/33 UTF-8 bytes; live Qwen/Qwen2.5-0.5B-Instruct verifier generated `In the vast and mysterious universe of the` and accepted 0/33; independent local llama-cli using the exact phone-copied GGUF generated the same text and accepted 33/33 bytes, same-GGUF tokenizer-ID comparison accepted 8/8 draft token IDs, wall-clock gate rejected speedup because sequential draft+verifier is slower than verifier-only, local same-GGUF speculative harness accepted 8/8 tokens without involving the phone, preflight showed the raw CLI cannot directly consume phone-provided draft token IDs, llama-cpp-python binding verifier accepted the phone draft text bytes under exact CLI chat template, Termux emitted context token IDs that the binding verifier ingested and accepted 8/8, and phone_bloombee_block_preflight.py confirms GGUF draft evidence is not BloomBee block serving while Termux lacks torch/transformers/bloombee; still not BloomBee block serving or speculative speedup",
        next_step="bridge live token transport plus wall-clock measurement, or separately prove BloomBee block serving before counting phone as a block worker",
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
