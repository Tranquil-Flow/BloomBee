# Distributed Inference Post-MVP Scope Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after the MVP-core physical showcase reaches 100%.

**Goal:** Define the first post-MVP workstreams after the Qwen3-8B physical showcase gate: stronger models, higher throughput, reusable KV, phone draft workers, and larger-model wrapper expansion.

**Architecture:** Keep MVP-core status frozen around the final physical showcase gate. Post-MVP work lives in explicit stretch milestones, each with proof artifacts under `mvp_capabilities/distributed_evidence/` and tests that prevent optimistic routing before proof gates pass.

**Tech Stack:** Python 3.11, pytest, BloomBee `src/bloombee`, `mvp_capabilities/*_proof.py`, m4pro MPS runtime, Pixel/Termux bridge evidence where useful.

---

## Non-goals before this plan starts

- Do not count any item here toward the MVP-core 100% denominator.
- Do not call Qwen3-30B-A3B or 2507 variants `demo_safe` until full-generation, cache-generation, and multi-request-load proof gates pass.
- Do not claim phone-backed speedup until a wall-clock verifier accepts the phone draft tokens and shows net improvement over the baseline.
- Do not claim MiniMax/GLM/DeepSeek native BloomBee support until wrappers or LayerExecutor backend proofs exist.

## MVP-tail handoff boundary

MVP-core is complete. The strict physical showcase passed in the final same-session artifact:

```text
mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json
```

That artifact closes the old cross-artifact gate: Pixel physical QR scan, Pixel Termux heartbeats, `m4pro-full` capacity heartbeat, joined Qwen3-8B layer plan, matching cache-generation placements, deterministic scaled 3/3 load proof, and `physical_showcase_proof.py` pass.

Post-MVP work below must not mutate the MVP-core 100% denominator. It may improve stronger-model capability, throughput, cache reuse, phone draft work, or route clarity only when each new claim has its own proof artifact and fail-closed tests.

## Qwen3-30B family priority decision

The Qwen3-30B-family ordering is now encoded in a test-backed audit helper:

```bash
.venv/bin/python -m mvp_capabilities.qwen30b_priority
```

Use this order unless Fable finds a concrete reason to change it:

1. `Qwen/Qwen3-30B-A3B` — substrate/risk reducer; already has prescan + one-block + multi-block proof, next gate `full_generation`.
2. `Qwen/Qwen3-30B-A3B-Instruct-2507` — user-facing follow-up with exact-model prescan + one-block + multi-block now passed; next gate `full_generation`.
3. `Qwen/Qwen3-30B-A3B-Thinking-2507` — optional reasoning variant; do not spend proof budget unless the demo specifically needs thinking/reasoning behavior.

Do not make both base 30B and Instruct-2507 required for the same post-MVP milestone. Exact 2507 lower gates are now proven for Instruct only; full-generation/cache/load remain required before route/demo promotion.

---

### Task 1: Promote Qwen3-30B-A3B from multi-block to full-generation proof

**Objective:** Extend the proven two-server `0:2` Qwen3-30B-A3B slice to a full-generation parity artifact.

**Files:**
- Modify: `mvp_capabilities/PROOF_STATUS.yaml`
- Add evidence: `mvp_capabilities/distributed_evidence/qwen30b/qwen3-30b-a3b-full-generation-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Start from `mvp_capabilities/distributed_evidence/qwen30b/qwen3-30b-a3b-multiblock-20260704T144934Z.json` and reuse its cache/run metadata.
2. Launch enough Qwen3-30B-A3B ranges on m4pro or a real joined peer set to cover the requested generation range.
3. Run `mvp_capabilities/full_generation_proof.py` with explicit `server_placements`.
4. Verify generated IDs/text/next-token parity.
5. Add a test asserting `full_generation` remains the next gate until artifact status is passed.
6. Commit with `test(mvp): prove qwen30b full generation`.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
```

---

### Task 2: Add Qwen3-30B cache-generation and multi-request-load gates

**Objective:** Prove the Qwen3-30B path survives cached `.generate()` and repeated direct-client requests.

**Files:**
- Add evidence: `mvp_capabilities/distributed_evidence/qwen30b/qwen3-30b-a3b-cache-generation-<timestamp>.json`
- Add evidence: `mvp_capabilities/distributed_evidence/qwen30b/qwen3-30b-a3b-multi-request-load-<timestamp>.json`
- Modify: `mvp_capabilities/PROOF_STATUS.yaml`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Reuse the full-generation server placements from Task 1.
2. Run `cache_generation_proof.py verify` with `--require-server-placements`.
3. Run `multi_request_load_proof.py verify` with at least 3 requests and finite forward/backward checks.
4. Promote Qwen3-30B-A3B only to `safe-demo` candidate after all gates pass.
5. Add tests preventing route picker from selecting Qwen3-30B before all three new gates pass.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py tests/test_demo_dashboard.py -q
```

---

### Task 3: Advance Qwen3-30B-A3B Instruct-2507 after lower-gate proof

**Objective:** Use the completed Seagate-backed Instruct-2507 lower gates as the starting point for full-generation/cache/load proof, while keeping Thinking-2507 deferred unless the demo specifically needs reasoning behavior.

**Files:**
- Modify: `mvp_capabilities/PROOF_STATUS.yaml`
- Existing evidence: `mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-oneblock-proof-20260704T222230Z.json`
- Existing evidence: `mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-multiblock-proof-20260705T064511Z.json`
- Future evidence: `mvp_capabilities/distributed_evidence/post_mvp/instruct2507-seagate-full-generation-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Run `python -m mvp_capabilities.qwen30b_priority` and confirm the priority order is still base 30B → Instruct-2507 → optional Thinking-2507.
2. Treat Instruct-2507 prescan, one-block, and multi-block as passed from committed artifacts; do not repeat them unless the cache/model revision changes.
3. Treat the full 16-shard Instruct-2507 cache as READY and `Instruct-2507@int8` multi-request load as passed from committed evidence.
4. Use the streamed-block fp16 reference harness for the first full-generation attempt on m4pro (`scripts/text_generation_parity.py --reference-mode streamed-blocks --checkpoint-model <base-model> --reference-cache-dir /Volumes/Seagate\ Portable\ Drive/huggingface/hub --reference-local-files-only`). Keep this as a harness until the live int8 distributed trace exactly matches generated IDs/text.
5. Keep Instruct-2507 blocked from `safe-demo` until cache generation and exact token parity pass; full-generation parity is now passed for the int8 route row.
6. Only repeat the lower-gate ladder for Thinking-2507 if the demo spec requires reasoning-style behavior.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
```

---

### Task 4: Continuous batching proof harness

**Objective:** Move from deterministic scheduler proof to live request-loop integration without hiding correctness failures. The scheduler simulation, replayable adapter plan, and injected live-loop unit seam are now committed; remaining work is real `inference_session.py` wiring, parity, and throughput.

**Files:**
- Existing: `mvp_capabilities/continuous_batching.py`
- Existing: `src/bloombee/client/live_continuous_batching.py`
- Existing evidence: `mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-scheduler-20260704.json`
- Existing evidence: `mvp_capabilities/distributed_evidence/post_mvp/continuous-batching-live-adapter-20260705.json`
- Existing evidence: `mvp_capabilities/distributed_evidence/post_mvp/live-continuous-batching-loop-unit-20260705.json`
- Future modify: `src/bloombee/client/inference_session.py`
- Test: `tests/test_live_continuous_batching.py`

**Steps:**
1. Treat the pure scheduler, adapter-plan, and injected-step live-loop unit as passed; do not call them live-server proof.
2. Wire `LiveContinuousDecodeLoop` tick rows into `inference_session.py` behind `BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING`.
3. Prove same-prompt parity with concurrent arrivals before measuring throughput.
4. Add request telemetry for per-request latency, overlap window, success/failure counts, and tokens/sec.
5. Run on TinyLlama or Qwen3-8B first; only then scale to Qwen3-30B.
6. Store future live proof artifacts under `mvp_capabilities/distributed_evidence/load/`.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_live_continuous_batching.py tests/test_continuous_batching.py -q
```

---

### Task 5: KV prefix reuse / cache reuse proof

**Objective:** Prove repeated prompts can reuse prefix KV/cache safely and measurably.

**Files:**
- Create: `mvp_capabilities/kv_prefix_reuse_proof.py`
- Modify: `README.environment-switches.md`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Define the proof contract: same prefix, varied suffixes, equal logits/tokens against no-reuse baseline.
2. Add RED tests for accepted and rejected cache-reuse evidence.
3. Capture timing deltas and correctness hashes.
4. Fail closed if any token/logit mismatch appears.
5. Document required environment flags and telemetry tags.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py::test_kv_prefix_reuse_proof_* -q
```

---

### Task 6: Phone draft-provider wall-clock gate

**Objective:** Upgrade phone draft evidence from static/standalone correctness to end-to-end accepted-token wall-clock comparison.

**Files:**
- Modify: `mvp_capabilities/phone_draft_verifier_compare.py`
- Modify: `mvp_capabilities/termux_gguf_draft_bridge.py`
- Add evidence: `mvp_capabilities/distributed_evidence/phone/phone-draft-wallclock-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Keep exact-token acceptance as the correctness gate.
2. Add baseline verifier-only wall-clock measurement.
3. Add phone draft + verifier wall-clock measurement.
4. Report accepted/rejected tokens and net speedup/slowdown.
5. Only mark `phone_speedup_proven` true when correctness passes and wall-clock improves.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
```

---

### Task 7: Android/Termux capability fidelity

**Objective:** Make phone and Android peer scans report usable memory/storage facts without overclaiming BloomBee serving support.

**Files:**
- Modify: `mvp_capabilities/peer_scan.py`
- Add evidence: `mvp_capabilities/distributed_evidence/phone/android-peer-scan-memory-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Add RED tests for parsing `/proc/meminfo` on Android.
2. Report `memory.total_gb` and `memory.free_gb` when available.
3. Preserve `mobile.is_mobile`, SoC, ABI, and runtime fields.
4. Keep `torch`/BloomBee serving blockers explicit when modules are missing.
5. Re-run Pixel peer scan and verify planner behavior remains honest.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py::test_peer_scan_* -q
```

---

### Task 8: qwen3_5_moe wrapper feasibility for AgentWorld-35B-A3B

**Status:** Config-only wrapper scout complete. Wrapper remains blocked; no runtime proof or demo promotion.

**Objective:** Decide whether `Qwen/Qwen-AgentWorld-35B-A3B` can become a native BloomBee stretch target.

**Completed artifact:**
- Evidence: `mvp_capabilities/distributed_evidence/qwen35b/qwen-agentworld-35b-wrapper-scout-20260704.json`
- Test: `tests/test_mvp_capabilities.py::test_qwen_agentworld_wrapper_scout_blocks_copying_qwen3_moe_wrapper`

**Claim boundary:** `post_mvp_wrapper_scout_no_runtime_proof_no_demo_promotion`

**Decision:** Do not copy the existing `qwen3_moe` wrapper. AgentWorld config uses `qwen3_5_moe` / `qwen3_5_moe_text`, alternating `linear_attention` and `full_attention` layers, mRoPE parameters, and linear-attention head fields; import/config-dispatch TDD must come before any wrapper code or one-block proof.

**Files:**
- Create or modify: `src/bloombee/models/qwen3_5_moe/`
- Modify: `mvp_capabilities/model_compat_scan.py`
- Add evidence: `mvp_capabilities/distributed_evidence/qwen35b/qwen-agentworld-35b-wrapper-scout-<timestamp>.json`
- Test: model wrapper tests under existing test layout

**Steps:**
1. Inspect HF config and module names for the qwen3_5_moe text tower.
2. Compare against existing `src/bloombee/models/qwen3_moe/` wrapper assumptions.
3. Write a compatibility scout artifact before coding a wrapper.
4. If feasible, TDD wrapper import/config dispatch first.
5. Run one-block proof only after wrapper tests pass.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
```

---

### Task 9: LayerExecutor / quantized-backend feasibility spike

**Status:** Complete as a research spike only. No runnable backend proof. No route/demo promotion.

**Objective:** Create a bounded feasibility report for models that are too large or structurally unsupported for native BloomBee today.

**Completed artifacts:**
- Doc: `docs/layerexecutor-quantized-backend-spike.md`
- Evidence: `mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json`
- Test: `tests/test_mvp_capabilities.py::test_layerexecutor_quantized_backend_spike_artifact_is_conservative`

**Claim boundary:** `post_mvp_research_spike_no_runnable_backend_proof`

**Files:**
- Create: `docs/layerexecutor-quantized-backend-spike.md`
- Add evidence: `mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-<timestamp>.json`

**Steps:**
1. Enumerate target families: MiniMax M3, GLM-5.2, DeepSeek-V4-Flash, Kimi K2.x.
2. Record blocker type: wrapper missing, sparse attention, memory, quantized runtime, expert paging.
3. Define one minimal proof for each family.
4. Keep this as a spike until a runnable backend proof exists.

**Verification command:**

```bash
.venv/bin/python -m json.tool mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-<timestamp>.json
```

---

### Task 10: Live chain-scheduler proof

**Objective:** Promote `chain_scheduler.py` from no-live-traffic rehearsal to a telemetry-backed live scheduler proof.

**Files:**
- Modify: `mvp_capabilities/chain_scheduler.py`
- Modify: `mvp_capabilities/proof_orchestrator.py`
- Modify: `mvp_capabilities/request_telemetry.py`
- Modify: `mvp_capabilities/multi_request_load_proof.py`
- Modify: `mvp_capabilities/demo_dashboard.py`
- Add evidence: `mvp_capabilities/distributed_evidence/scheduler/chain-scheduler-live-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`, `tests/test_demo_dashboard.py`

**Steps:**
1. Keep existing `chain_scheduler_plan_only_no_inference_proof` output as rehearsal-only.
2. Add RED tests for a new live proof artifact with `live_requests_sent: true`, `inference_proven: true`, request results, telemetry, joined-plan assignments, server placements, and fail-closed `failed_checks`.
3. Add a `run` or `verify-live` command that consumes a joined layer plan, server multiaddrs, selected model, and a schedule JSON.
4. Reuse `request_telemetry.py` and `multi_request_load_proof.py` checks so every request has finite output/grad and measured nonzero latency.
5. Fail if joined-plan assignments and server placements differ.
6. Render live scheduler status in the dashboard separately from rehearsal status.
7. Commit only after live-proof tests and dashboard tests pass.

**Candidate live command shape:**

```bash
.venv/bin/python -m mvp_capabilities.chain_scheduler run \
  --joined-layer-plan .local/joined-layer-plan.json \
  --schedule .local/qwen30b-chain-schedule.json \
  --server-maddr '<SERVER_0_MULTIADDR>' \
  --server-maddr '<SERVER_1_MULTIADDR>' \
  --model Qwen/Qwen3-30B-A3B \
  --out mvp_capabilities/distributed_evidence/scheduler/chain-scheduler-live-<timestamp>.json
```

**Successful artifact shape:**

```json
{
  "claim_boundary": "verified_chain_scheduler_live_request_evidence",
  "live_requests_sent": true,
  "inference_proven": true,
  "scheduler_status": "passed",
  "server_placements_match_joined_plan": true,
  "request_results": [],
  "telemetry": {},
  "failed_checks": []
}
```

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py tests/test_demo_dashboard.py -q
```

---

### Task 11: Post-MVP dashboard and status separation

**Objective:** Keep MVP-core and post-MVP progress visible without mixing denominators.

**Files:**
- Modify: `mvp_capabilities/mvp_status.py`
- Modify: `mvp_capabilities/demo_dashboard.py`
- Test: `tests/test_mvp_capabilities.py`, `tests/test_demo_dashboard.py`

**Steps:**
1. Add a post-MVP progress table with per-workstream status and next gate.
2. Keep MVP-core `overall_percent` unchanged by post-MVP progress.
3. Add dashboard rendering for post-MVP workstreams.
4. Add tests proving Qwen3-30B progress does not move the MVP-core bar.
5. Commit after tests pass.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py tests/test_demo_dashboard.py -q
```

---

## Suggested parallelization

- **Lane A:** Qwen3-30B full/cache/load proofs on m4pro.
- **Lane B:** Android/phone capability and draft-provider wall-clock gates.
- **Lane C:** Live chain scheduler + continuous batching + KV prefix reuse proof harnesses.
- **Lane D:** Wrapper/backend feasibility for qwen3_5_moe and quantized frontier models.

Each lane must return evidence paths, exact commands, test results, and claim boundaries. No lane may mutate MVP-core status without a test proving the denominator remains separate.
