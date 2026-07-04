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
2. `Qwen/Qwen3-30B-A3B-Instruct-2507` — user-facing follow-up after base 30B full/cache/load behavior is understood; next gate `prescan`.
3. `Qwen/Qwen3-30B-A3B-Thinking-2507` — optional reasoning variant; do not spend proof budget unless the demo specifically needs thinking/reasoning behavior.

Do not make both base 30B and 2507 required for the same post-MVP milestone. Exact 2507 model IDs still need prescan + one-block before route selection.

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

### Task 3: Scope Qwen3-30B-A3B Instruct-2507 first; keep Thinking-2507 optional

**Objective:** Turn the user-facing Instruct-2507 checkpoint from a shelf entry into a measured candidate, while keeping Thinking-2507 deferred unless the demo specifically needs reasoning behavior.

**Files:**
- Modify: `mvp_capabilities/MODEL_REGISTRY.yaml`
- Modify: `mvp_capabilities/PROOF_STATUS.yaml`
- Add evidence: `mvp_capabilities/distributed_evidence/qwen30b-2507/<model>-prescan-<timestamp>.json`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Run `python -m mvp_capabilities.qwen30b_priority` and confirm the priority order is still base 30B → Instruct-2507 → optional Thinking-2507.
2. Run prescan against `Qwen/Qwen3-30B-A3B-Instruct-2507` first.
3. Record exact `model_type`, layer count, hidden size, expert count, and top-k routing.
4. Run one-block live server proof before any multi-block attempt.
5. Keep Instruct-2507 blocked from `safe-demo` until full/cache/load gates pass.
6. Add tests that Instruct-2507 route selection reports pending proof blockers.
7. Only repeat the same prescan/one-block ladder for Thinking-2507 if the demo spec requires reasoning-style behavior.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
```

---

### Task 4: Continuous batching proof harness

**Objective:** Build a controlled benchmark that proves throughput gains from concurrent request scheduling without hiding correctness failures.

**Files:**
- Create: `mvp_capabilities/continuous_batching_proof.py`
- Modify: `mvp_capabilities/request_telemetry.py`
- Test: `tests/test_mvp_capabilities.py`

**Steps:**
1. Write RED tests for a synthetic request-log bundle with overlapping request windows.
2. Parse per-request latency, overlap window, success/failure counts, and tokens/sec.
3. Add a verifier that fails if any request has non-finite output/grad or missing timing.
4. Run on TinyLlama or Qwen3-8B first; only then scale to Qwen3-30B.
5. Store proof artifacts under `mvp_capabilities/distributed_evidence/load/`.

**Verification command:**

```bash
.venv/bin/python -m pytest tests/test_mvp_capabilities.py::test_continuous_batching_proof_* -q
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

**Objective:** Decide whether `Qwen/Qwen-AgentWorld-35B-A3B` can become a native BloomBee stretch target.

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

**Objective:** Create a bounded feasibility report for models that are too large or structurally unsupported for native BloomBee today.

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
