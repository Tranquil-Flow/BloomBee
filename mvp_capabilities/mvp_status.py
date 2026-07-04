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
NEXT_GATE = "MVP core complete; post-MVP improvements next"
MVP_SCOPE = "mvp_core"
MVP_COMPLETION_DEFINITION = (
    "MVP reaches 100% when a fresh/self-serve joined swarm can run a selected "
    "demo-safe distributed model with proof-backed generation, visible utilisation, "
    "and dashboard/operator evidence. Larger-model ladders and optimisations are "
    "tracked as post-MVP/stretch work, not part of the 100% denominator."
)


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
        weight=10,
        completion=1.00,
        status="complete",
        evidence="MODEL_REGISTRY.yaml, model_compat_scan.py, PROOF_STATUS.yaml, proof_ladder.py",
    ),
    Milestone(
        id="dynamic_selector",
        label="Prepared model ladder and proof-aware selector modes",
        weight=8,
        completion=1.00,
        status="complete",
        evidence="route_picker.py supports planning/showcase-attempt/safe-demo, infers registry HF model_type, blocks unsupported wrappers from showcase/safe-demo, and promotes Qwen3-8B only after proof gates pass; Qwen3 2507 variants remain stretch candidates",
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
        completion=1.00,
        status="complete",
        evidence="join_coordinator.py creates link offers/heartbeats with successful ok:true response records; join_http_server.py exposes health/offer/heartbeat/active/route/plan/speculative/bootstrap/bootstrap.sh/handoff/proof-orchestration endpoints; join_client.py posts bounded repeated peer heartbeats; join_card.py renders SVG cards plus exact URL JSON/TXT sidecars; join_qr_preflight.py reports scanner-proof dependency blockers fail-closed; join_qr_proof.py generated a true QR PNG and decoded it back to the exact redacted join URL locally; the final same-session physical showcase captured a real Pixel camera/browser QR scan plus Pixel Termux heartbeat loop and m4pro capacity heartbeat in mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json",
    ),
    Milestone(
        id="layer_planning",
        label="Layer planner and launch-ready worker assignment",
        weight=8,
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
        label="Qwen3-8B demo-safe fallback proof ladder",
        weight=17,
        completion=1.00,
        status="complete",
        evidence="Qwen3-8B prescan, one-block server proof, clean-tree m4pro preflight, minimal two-server multi-block direct RPC proof, full-generation forward-loop parity, cache-generation generate-api parity, and live full-range multi-request load proof passed; full-generation evidence is tracked at mvp_capabilities/distributed_evidence/QWEN3_8B_FULL_GENERATION_FORWARD_LOOP_2026-07-04.json, cache-generation evidence at mvp_capabilities/distributed_evidence/QWEN3_8B_CACHE_GENERATION_2026-07-04.json, and load evidence at mvp_capabilities/distributed_evidence/QWEN3_8B_MULTI_REQUEST_LOAD_2026-07-04.json",
    ),
    Milestone(
        id="chain_scheduler",
        label="Multi-request chain scheduler, load proof, and draft-provider scaffold",
        weight=12,
        completion=1.00,
        status="complete",
        evidence="chain_scheduler.py turns joined layer plans into multi-request waves, per-peer load estimates, and no-live-traffic health reports; live Qwen3-8B full-range load proof passed with 3/3 direct-client requests and measured forward/backward latencies at mvp_capabilities/distributed_evidence/QWEN3_8B_MULTI_REQUEST_LOAD_2026-07-04.json; proof_orchestrator.py orders handoff launch/proof runbooks and blocks unresolved placeholders or legacy peer flags before operator execution; speculative_decode_plan.py defines verifier-authoritative draft-provider plans and phone-as-draft-only policy; draft_provider.py adds a deterministic DraftProvider.propose contract with verifier-prefix accepted/rejected counters for dashboard smoke reports; draft_provider_bridge.py exposes the same contract over stdio JSONL for Termux/ADB/SSH bridge experiments; termux_draft_smoke.py and termux_draft_latency.py render/verify self-contained Termux phone evidence; Pixel 8 Pro Termux smoke evidence passed at mvp_capabilities/distributed_evidence/phone/termux-draft-smoke-20260704T095557Z.json, 50-iteration static-contract latency passed at mvp_capabilities/distributed_evidence/phone/termux-draft-latency-20260704T100644Z.json, tiny-model feasibility blockers are tracked at mvp_capabilities/distributed_evidence/phone/termux-tiny-model-probe-20260704T101232Z.json, a no-install GGUF runtime plan is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-plan-20260704T101232Z.json, real Pixel 8 Pro Termux llama.cpp/stories15M GGUF generation is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-generation-20260704T104506Z.json, phone GGUF draft-bridge smoke is tracked at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-bridge-20260704T105400Z.json, phone_draft_verifier_compare.py tracks UTF-8 verifier-prefix evidence including a live Qwen/Qwen2.5-0.5B-Instruct mismatch with accepted=0/33 at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-qwen05-20260704T110000Z.json plus an independent local same-GGUF verifier copied from the phone with accepted=33/33 at mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-same-gguf-20260704T111215Z.json, termux-local tokenizer-ID comparison accepted 8/8 same-GGUF draft token IDs at mvp_capabilities/distributed_evidence/phone/termux-local-tokenizer-id-compare-20260704T111800Z.json, phone_speculative_wallclock_gate.py records the sequential phone-draft+verifier path as slower (2.403479s vs 1.837976s verifier-only) at mvp_capabilities/distributed_evidence/phone/termux-same-gguf-wallclock-gate-20260704T112500Z.json, local llama.cpp speculative harness accepted 8/8 draft tokens with same GGUF at mvp_capabilities/distributed_evidence/phone/local-same-gguf-llama-speculative-harness-20260704T113600Z.json but did not involve the phone, phone-integrated verifier preflight at mvp_capabilities/distributed_evidence/phone/phone-integrated-verifier-preflight-20260704T114000Z.json records the external-token-ID gap, and phone_llama_cpp_binding_verifier.py accepted the phone draft text bytes under the exact llama.cpp chat template at mvp_capabilities/distributed_evidence/phone/phone-llama-cpp-binding-verifier-20260704T120000Z.json, then ingested Termux-emitted context draft token IDs from mvp_capabilities/distributed_evidence/phone/termux-context-token-ids-20260704T121646Z.json and accepted 8/8 external phone context tokens using forced-batch logits_all argmax checks at mvp_capabilities/distributed_evidence/phone/phone-context-token-id-verifier-20260704T121646Z.json; request_telemetry.py summarizes direct-client success/failure and latency logs for dashboards, treating zero latency as unmeasured; multi_request_load_proof.py verifies repeated direct-client logs and now blocks unmeasured latency before proof promotion",
        next_step=None,
    ),
    Milestone(
        id="physical_showcase",
        label="Physical/self-serve live showcase with fresh joined devices",
        weight=14,
        completion=1.00,
        status="complete",
        evidence="physical_showcase_proof.py strict cross-artifact verifier passed in the same session for Qwen/Qwen3-8B: real Pixel 8 Pro camera/browser QR scan, Pixel Termux join_client.py 3-heartbeat loop with server_response.ok=true, fresh m4pro-full capacity heartbeat, joined layer plan assigning 0:36 to m4pro-full, cache-generation exact ID/text parity with server_placements=m4pro-full:0:36, and deterministic scaled 3/3 multi-request load proof with finite forward/backward. Redacted commit artifact: mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json",
    ),
)


POST_MVP_MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        id="qwen3_30b_proof_ladder",
        label="Qwen3-30B-A3B base-first proof ladder with Instruct-2507 follow-up",
        weight=15,
        completion=0.45,
        status="stretch",
        evidence="Qwen3-MoE wrapper exists; one live M4 Pro block shard proof passed for Qwen3-30B-A3B, and a clean-archive two-server multi-block 0:2 direct RPC proof passed at mvp_capabilities/distributed_evidence/qwen30b/qwen3-30b-a3b-multiblock-20260704T144934Z.json (finite forward/backward, failed_checks=[]). Instruct-2507 now has a Seagate-backed prescan + one-block live RPC proof at mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-oneblock-proof-20260704T222230Z.json. qwen30b_priority.py codifies the post-MVP order: base Qwen3-30B-A3B first, Instruct-2507 follow-up, Thinking-2507 optional. This remains post-MVP/stretch and does not move the MVP-core 100% denominator after Qwen3-8B became demo-safe.",
        next_step="full-generation parity for base Qwen3-30B-A3B, then cache-generation and multi-request load; for Instruct-2507, multi-block/full-generation gates remain before any route/demo promotion",
    ),
    Milestone(
        id="layerexecutor_quantized_backend_spike",
        label="LayerExecutor / quantized-backend feasibility spike",
        weight=5,
        completion=1.00,
        status="research_complete",
        evidence="No runnable backend proof. Config-only scans and research artifact block MiniMaxAI/MiniMax-M3, zai-org/GLM-5.2, deepseek-ai/DeepSeek-V4-Flash, and moonshotai/Kimi-K2-Instruct from native BloomBee route/demo use today; evidence is tracked at mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json and docs/layerexecutor-quantized-backend-spike.md.",
        next_step="keep base Qwen3-30B full/cache/load ahead of frontier backend work; if continued, pick one target and start with external-runtime smoke only",
    ),
    Milestone(
        id="quantization_routing_handoff",
        label="Quantization + route override handoff",
        weight=5,
        completion=0.45,
        status="route_lane_committed",
        evidence="Fable's quantization/routing foundation is committed and now has a claim-bounded Qwen3-30B-A3B@int8 route lane artifact: docs/quantization-routing-handover.md, mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json, mvp_capabilities/distributed_evidence/post_mvp/quantized-qwen30b-route-lane-20260704.json, src/bloombee/utils/moe_expert_quant.py, route_picker/proof_ladder demo_safe-gate alignment, route_report CLI behavior, and default-suite quantization tests. This is still post-MVP only: no quantized serving proof, no fp16 proof-row promotion, no route/demo promotion, and no MVP-core denominator change.",
        next_step="wire server-side INT8 quantized loading behind the existing quant_type flag, run Qwen3-30B-A3B@int8 one-block proof, then full/cache/load gates before any demo-safe promotion",
    ),
)


POST_MVP_TASK_IDS = frozenset(
    {
        "qwen3_30b_core_proof",
        "qwen3_30b_2507_shelf",
        "qwen35b_candidate",
        "minimax_m3_candidate",
        "speculative_decode",
        "phone_worker",
        "continuous_batching",
        "kv_prefix_reuse",
    }
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
        status="complete",
        evidence="copy/paste join URL, bootstrap.sh, bounded heartbeat client, SVG join card sidecars, QR dependency preflight, and local true-QR exact decode proof exist; final same-session showcase captured a real Pixel 8 Pro camera/browser QR scan, matching scan URL hash, and 3 successful Termux join_client.py heartbeats with server_response.ok=true at mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json",
        next_step=None,
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
        label="Qwen3-8B full-generation/cache-generation proof",
        status="complete",
        evidence="Qwen3-8B prescan, one-block server proof, clean-tree m4pro preflight, minimal two-server multi-block direct RPC proof, full-generation forward-loop parity, and cache-generation generate-api parity passed; load gate remains separate under multi_request_load",
        next_step=None,
    ),
    PlanTask(
        id="qwen3_30b_core_proof",
        label="Qwen3-30B-A3B core laptop-swarm proof ladder",
        status="partial",
        evidence="qwen3_moe wrapper exists; one live M4 Pro Qwen3-30B-A3B block shard passed; clean-archive two-server multi-block 0:2 direct RPC proof passed with finite forward/backward; full distributed generation remains pending",
        next_step="run full-generation parity for Qwen3-30B-A3B, then cache-generation and multi-request load when enough clean memory/devices are available",
    ),
    PlanTask(
        id="qwen3_30b_2507_shelf",
        label="Prepared Qwen3-30B-A3B Instruct-2507 shelf; Thinking-2507 optional",
        status="complete",
        evidence="2507 variants are registered with config metadata; qwen30b_priority.py codifies Instruct-2507 as the user-facing follow-up after base 30B gates, with Thinking-2507 deferred unless the demo needs reasoning behavior. The earlier Seagate NTFS blocker is tracked at mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-readonly-blocker-20260704.json; after APFS+exFAT setup, the Seagate-backed Instruct-2507 download, prescan, and one-block live RPC proof passed at mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-oneblock-proof-20260704T222230Z.json. Multi-block/full-generation/cache/load remain separate promotion gates.",
        next_step=None,
    ),
    PlanTask(
        id="qwen35b_candidate",
        label="Qwen35B candidate branch",
        status="partial",
        evidence="Qwen/Qwen-AgentWorld-35B-A3B is memory-fit for synthetic 10-laptop planning. Text-tower qwen3_5_moe_text wrapper/package now has import/config and full_attention block contract tests green at mvp_capabilities/distributed_evidence/qwen35b/qwen-agentworld-35b-text-wrapper-gate-20260704.json; linear_attention cache remains blocked fail-closed, so there is still no one-block server proof and no demo/route promotion.",
        next_step="implement/test Qwen3.5 linear-attention cache mapping before one-block server proof; keep showcase/safe-demo blocked until linear-attention, one-block, full/cache/load gates pass",
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
        status="complete",
        evidence="Qwen3-8B full-range live load proof passed with 3/3 successful direct-client requests over block range 0:36, finite outputs/gradients, and measured forward/backward latency; evidence tracked at mvp_capabilities/distributed_evidence/QWEN3_8B_MULTI_REQUEST_LOAD_2026-07-04.json",
        next_step=None,
    ),
    PlanTask(
        id="speculative_decode",
        label="Speculative/draft-provider speedup plan",
        status="partial",
        evidence="speculative_decode_plan.py defines verifier-authoritative draft-provider roles and phone-as-draft-only policy; draft_provider.py provides a deterministic provider interface and accepted/rejected exact-token counters for dashboard smoke reports; draft_provider_bridge.py exposes stdio JSONL transport for phone/Termux bridge tests; termux_draft_smoke.py verified a real Pixel 8 Pro Termux draft-contract smoke with proposed=3 accepted=2 rejected=1; termux_draft_latency.py verified a 50-iteration Pixel 8 Pro static-contract loop with proposed=150 accepted=100 rejected=50 and latency p95=0.001669ms; termux_tiny_model_probe.py showed no torch/transformers/tokenizers/llama_cpp/bloombee installed; Termux llama.cpp CLI generated text from ggml-org/tiny-llamas/stories15M.gguf; termux_gguf_draft_bridge.py verified a phone GGUF draft-provider-candidate JSON bridge; phone_draft_verifier_compare.py proves exact byte-prefix acceptance math; live Qwen/Qwen2.5-0.5B-Instruct verifier comparison rejected the phone draft with accepted=0/33, independent local same-GGUF verifier comparison accepted 33/33 bytes from the exact phone-copied GGUF, same-GGUF tokenizer-ID comparison accepted 8/8 draft token IDs, wall-clock gate shows sequential phone-draft+verifier is slower (2.403479s vs 1.837976s verifier-only), local llama.cpp speculative harness accepted 8/8 draft tokens with same GGUF but without phone involvement, preflight showed the raw llama.cpp CLI cannot ingest phone-provided external draft token IDs, llama-cpp-python binding verifier accepted the phone draft text bytes with context token IDs [6716, 2462, 29892, 263, 2217, 7826, 4257, 28846], and the binding verifier now ingests Termux-emitted context token IDs with forced-batch logits checks and accepts 8/8 external phone tokens; fresh live Pixel/m4pro ADB rerun artifacts termux-context-token-ids-live-adb-20260704T210323Z.json and phone-context-token-id-verifier-live-adb-20260704T210323Z.json confirm the same 8/8 external-token acceptance through push/type/pull transport; no phone-backed speculative speedup proof exists yet",
        next_step="build an integrated non-sequential verifier path that consumes phone token IDs without rerunning verifier-only decode, then prove faster wall clock before speedup claims",
    ),
    PlanTask(
        id="phone_worker",
        label="Phone as useful inference or draft worker",
        status="partial",
        evidence="mobile capability fields exist in peer_scan.py; draft_provider.py defines the phone-compatible draft-provider contract; draft_provider_bridge.py provides stdio JSONL bridge groundwork; m4pro ADB pushed/typed short commands into Termux and verified real Pixel 8 Pro JSON evidence (Android SDK 36, Tensor G3, aarch64) for one-shot contract smoke plus 50-iteration static-contract latency p95=0.001669ms; feasibility probe showed 11.851GB total RAM, 2.557GB available, 28.425GB free storage, build tools present, and missing torch/transformers/tokenizers/llama_cpp/bloombee Python modules; after approval, Termux llama.cpp CLI plus ggml-org/tiny-llamas/stories15M.gguf generated `One day, a little girl named Lucy` in 0.347524s with SHA256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a; termux_gguf_draft_bridge.py wrapped that phone generation as a draft-provider-candidate JSON bridge in 0.565503s; positive-control verifier comparison accepted 33/33 UTF-8 bytes; live Qwen/Qwen2.5-0.5B-Instruct verifier generated `In the vast and mysterious universe of the` and accepted 0/33; independent local llama-cli using the exact phone-copied GGUF generated the same text and accepted 33/33 bytes, same-GGUF tokenizer-ID comparison accepted 8/8 draft token IDs, wall-clock gate rejected speedup because sequential draft+verifier is slower than verifier-only, local same-GGUF speculative harness accepted 8/8 tokens without involving the phone, preflight showed the raw CLI cannot directly consume phone-provided draft token IDs, llama-cpp-python binding verifier accepted the phone draft text bytes under exact CLI chat template, Termux emitted context token IDs at termux-context-token-ids-20260704T121646Z.json, phone-context-token-id-verifier-20260704T121646Z.json ingested them with forced-batch logits checks and accepted 8/8, fresh live Pixel/m4pro ADB rerun artifacts termux-context-token-ids-live-adb-20260704T210323Z.json and phone-context-token-id-verifier-live-adb-20260704T210323Z.json confirm the same 8/8 external-token acceptance through push/type/pull transport, termux-same-gguf-wallclock-gate-20260704T112500Z.json measured 2.403479s sequential phone-draft+verifier versus 1.837976s verifier-only and rejected speedup, and phone_bloombee_block_preflight.py confirms GGUF draft evidence is not BloomBee block serving while Termux lacks torch/transformers/bloombee; still not BloomBee block serving or speculative speedup",
        next_step="build an integrated non-sequential verifier path for phone token IDs and prove faster wall clock, or separately prove BloomBee block serving before counting phone as a block worker",
    ),
    PlanTask(
        id="physical_showcase",
        label="Physical/self-serve N-laptop showcase",
        status="complete",
        evidence="Strict physical_showcase_proof.py verifier passed in a same-session final run: Pixel physical QR scan and heartbeat loop, m4pro-full capacity heartbeat, Qwen3-8B joined layer plan 0:36, live server placement alignment, cache-generation parity, and 3/3 deterministic scaled multi-request load proof. Redacted artifact: mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json",
        next_step=None,
    ),
    PlanTask(
        id="continuous_batching",
        label="True continuous batching",
        status="partial",
        evidence="continuous_batching.py adds a pure round-robin decode scheduler simulation with late-arrival admission, padded batch inputs, per-request deinterleaving, and claim-bounded evidence at mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-scheduler-20260704.json; no live server integration, parity proof, wall-clock speedup, or demo promotion yet",
        next_step="wire the scheduler into the live decode request loop behind an opt-in flag, prove parity with concurrent arrivals, then measure wall-clock throughput before any demo or speedup promotion",
    ),
    PlanTask(
        id="kv_prefix_reuse",
        label="Real prefill KV prefix reuse",
        status="partial",
        evidence="kv_prefix_reuse.py adds a pure prefix-only reuse planner that proves reused prefix + planned suffix reconstructs every request, rejects non-prefix overlap, and tracks claim-bounded evidence at mvp_capabilities/distributed_evidence/post_mvp/kv-prefix-reuse-planner-20260704.json; no live KV cache tensor reuse, server integration, parity proof, or speedup claim yet",
        next_step="wire prefix lookup into real prefill/session cache metadata behind an opt-in flag, prove hidden-state/token parity, then measure memory and wall-clock impact before any demo promotion",
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


def _task_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        status: sum(1 for item in items if item["status"] == status)
        for status in ("complete", "partial", "pending", "blocked")
    }
    summary["total"] = len(items)
    return summary


def build_status_report() -> dict[str, Any]:
    total_weight = sum(item.weight for item in MILESTONES)
    earned = sum(item.weight * item.completion for item in MILESTONES)
    overall_percent = round(earned / total_weight * 100) if total_weight else 0
    planned_tasks = [_task_payload(item) for item in PLANNED_TASKS]
    post_mvp_tasks = [item for item in planned_tasks if item["id"] in POST_MVP_TASK_IDS]
    core_tasks = [item for item in planned_tasks if item["id"] not in POST_MVP_TASK_IDS]
    task_summary = _task_summary(planned_tasks)
    core_task_summary = _task_summary(core_tasks)
    post_mvp_task_summary = _task_summary(post_mvp_tasks)
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scope": MVP_SCOPE,
        "mvp_completion_definition": MVP_COMPLETION_DEFINITION,
        "overall_percent": overall_percent,
        "remaining_percent": 100 - overall_percent,
        "overall_bar": render_bar(overall_percent),
        "earned_weighted_points": round(earned, 2),
        "total_weight": total_weight,
        "next_gate": NEXT_GATE,
        "interpretation": (
            "Weighted MVP-core engineering progress. The denominator ends at a "
            "working proof-backed MVP; larger models and performance refinements "
            "are tracked separately as post-MVP/stretch work."
        ),
        "milestones": [_milestone_payload(item) for item in MILESTONES],
        "post_mvp_milestones": [_milestone_payload(item) for item in POST_MVP_MILESTONES],
        "planned_tasks": planned_tasks,
        "core_tasks": core_tasks,
        "post_mvp_tasks": post_mvp_tasks,
        "task_summary": task_summary,
        "task_summary_scope": "all_tasks_including_post_mvp_backlog",
        "core_task_summary": core_task_summary,
        "post_mvp_task_summary": post_mvp_task_summary,
        "core_tasks_complete": core_task_summary == {"complete": core_task_summary["total"], "partial": 0, "pending": 0, "blocked": 0, "total": core_task_summary["total"]},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Distributed Inference MVP status",
        "",
        f"**Built from plan:** `{report['overall_bar']}`",
        f"**Remaining:** `{report['remaining_percent']}%`",
        f"**Next gate:** {report['next_gate']}",
        f"**Claim boundary:** `{report['claim_boundary']}`",
        f"**MVP scope:** `{report['scope']}`",
        "",
        report["interpretation"],
        "",
        report["mvp_completion_definition"],
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
    post_mvp = report.get("post_mvp_milestones") or []
    if post_mvp:
        lines.extend(
            [
                "",
                "## Post-MVP / stretch milestones",
                "",
                "These remain visible, but they do not drag the MVP 100% denominator.",
                "",
                "| Milestone | Weight | Status | Built | Evidence / next step |",
                "|---|---:|---|---:|---|",
            ]
        )
        for item in post_mvp:
            evidence = item["evidence"]
            if item.get("next_step"):
                evidence = f"{evidence}<br>Next: {item['next_step']}"
            lines.append(
                f"| {item['label']} | {item['weight']} | {item['status']} | "
                f"{item['percent']}% | {evidence} |"
            )
    def _summary_text(summary: dict[str, Any]) -> str:
        return ", ".join(
            f"{summary.get(key, 0)} {key}" for key in ("complete", "partial", "pending", "blocked")
        )

    def _append_task_table(title: str, summary_label: str, summary: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"{summary_label}: {_summary_text(summary)}",
                "",
                "| Task | Status | Done? | Evidence / next step |",
                "|---|---|---:|---|",
            ]
        )
        for item in tasks:
            evidence = item["evidence"]
            if item.get("next_step"):
                evidence = f"{evidence}<br>Next: {item['next_step']}"
            done = "yes" if item["done"] else "no"
            lines.append(f"| {item['label']} | {item['status']} | {done} | {evidence} |")

    lines.extend(
        [
            "",
            "## Planned tasks",
            "",
            f"All-task summary: {_summary_text(report.get('task_summary') or {})}",
            "",
            "The all-task summary includes post-MVP backlog and should not be read as an MVP-core blocker.",
        ]
    )
    _append_task_table("MVP-core tasks", "MVP-core task summary", report.get("core_task_summary") or {}, report.get("core_tasks") or [])
    _append_task_table(
        "Post-MVP backlog tasks",
        "Post-MVP backlog task summary",
        report.get("post_mvp_task_summary") or {},
        report.get("post_mvp_tasks") or [],
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
