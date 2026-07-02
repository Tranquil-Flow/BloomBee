# Distributed Inference MVP Implementation Plan

> **For Hermes:** execute this plan as verified TDD slices. Commit after each verified task. Direct SSH to the M4 Pro is `ssh m4pro`; do not rely on the local Tailscale API from the sandbox.

**Goal:** Build a hardware-aware BloomBee distributed-inference MVP that can scan peers, predict and measure model fit/speed, route to the strongest feasible model, and be ready for a 10+ M4 laptop showcase as part of MVP scope.

**Architecture:** A standalone `mvp_capabilities/` toolchain gathers peer capability JSON, benchmarks real model throughput, aggregates a swarm roster, selects the strongest model/placement from `MODEL_REGISTRY.yaml`, and renders a demo-ready status. BloomBee server bring-up remains the execution layer; the MVP router prepares verified plans and inputs for it. The 10-laptop swarm is in MVP scope, with final physical showcase gated until all local and two-device verification passes.

**Tech Stack:** Python 3.11, pytest, PyYAML, psutil, torch/transformers for optional benches, SSH to `m4pro` (`evinova-self`, 48GB), local `evinova` M4 16GB.

---

## Correct host map

- Local current session: `evinova` on `Evis-MacBook-Pro`, M4 16GB.
- M4 Pro: `evinova-self` on `m4pro`, 48GB, reachable via `ssh m4pro`.
- Astra/laptop peer: `astra-macbook`, Tailscale IP `100.117.33.124`, SSH may be unavailable.
- Do not infer availability from `tailscale status` inside this sandbox; direct SSH is the verified path.

## MVP scope

### In scope for MVP

1. **Peer capability scan**
   - Real per-node JSON: CPU, RAM/free RAM, MPS/CUDA/CPU, disk, Python/torch availability, mesh/address metadata when accessible.
   - Must run locally and remotely via SSH.

2. **Swarm roster aggregation**
   - Collect `~/.bloombee/capabilities/*.json` and optionally direct peer JSON files.
   - Render machine-readable JSON and a human-readable table.

3. **Model fit and quality registry**
   - Registry includes TinyLlama, Qwen2.5 dense tiers, Qwen3 dense tiers, Qwen3 MoE tiers.
   - Annotate 10-laptop MVP target: Qwen3-30B-A3B.
   - Annotate stretch target: Qwen3-235B-A22B for many M4 Pros / larger swarm.

4. **Measured throughput matrix**
   - `bench_throughput.py` already works after `sitecustomize.py` fix.
   - Add `sweep_models.py` so each peer can bench all feasible models and save JSON results.
   - Router must prefer measured speed when available; use roofline estimates only as fallback.

5. **Swarm-aware route picker**
   - Given peers + registry + optional bench matrix, choose strongest feasible model.
   - Support current 2-device setup and planned 10-laptop swarm.
   - Output placement plan: solo, replicated, block-parallel candidate, or unsupported.

6. **Documentation page named `distributed-inference-mvp`**
   - Do not call it `bloombee-mvp`.
   - Include exact host map, current verified state, local/M4 Pro commands, 10-laptop showcase recipe, and blockers.

7. **10-laptop swarm readiness as MVP requirement**
   - Physical showcase is later, but code/docs/tests must treat 10 laptops as a first-class MVP scenario.
   - Include synthetic 10-peer tests with 16GB/24GB/36GB laptop tiers.
   - Target model: Qwen3-30B-A3B first; Qwen3-235B-A22B documented as stretch.

8. **Verification gates**
   - Local M4 16GB: run unit tests + peer scan + TinyLlama bench already proven.
   - M4 Pro 48GB: run peer scan via `ssh m4pro`; run at least registry/route tests remotely; run Qwen2.5-7B bench/serve only when model cache/download is available and not disruptive.
   - 10-laptop: synthetic route test now; physical showcase later.

### Explicit non-goals for this build slice

- No claim that 10 physical laptops have already run.
- No claim that Qwen3-30B-A3B MoE is served until BloomBee MoE block handler is implemented and verified.
- No destructive changes to existing BloomBee core files already modified in this worktree without diff review.
- No pushing upstream without explicit approval.

---

## Task 1: Stabilize plan and host identity

**Objective:** Record correct MVP scope and host map, update task tracking, avoid repeating evinova/evinova-self confusion.

**Files:**
- Create: `.hermes/plans/2026-07-02_162926-distributed-inference-mvp.md`

**Verification:**
- `ssh m4pro 'whoami; hostname; sysctl -n hw.memsize'` shows `evinova-self`, `m4pro`, `51539607552`.
- Local `whoami; hostname; sysctl -n hw.memsize` shows `evinova`, `Evis-MacBook-Pro`, `17179869184`.

**Commit:** after plan + first code slice, not plan-only if repo already dirty.

---

## Task 2: Add failing tests for swarm roster aggregation

**Objective:** Define expected behavior before adding `swarm_roster.py`.

**Files:**
- Create/modify: `tests/test_mvp_capabilities.py`
- Create: `mvp_capabilities/swarm_roster.py`

**Tests:**
- Load two peer JSON files from a temp dir.
- Normalize missing optional fields.
- Sort peers by hostname.
- Produce aggregate totals: peer count, total/free memory, accelerator counts.

**Command:**
- `pytest tests/test_mvp_capabilities.py::test_load_roster_from_capability_dir -q`
- Expected RED: import/module missing.

---

## Task 3: Implement `swarm_roster.py`

**Objective:** Build a standalone roster loader/CLI.

**Files:**
- Create: `mvp_capabilities/swarm_roster.py`

**Implementation notes:**
- No BloomBee imports.
- Functions: `load_capability_files(paths)`, `summarize_roster(peers)`, `main()`.
- CLI options: `--cap-dir`, `--json`, `--min-free-gb`, `--device`.
- Output JSON by default when `--json`; compact table otherwise.

**Verification:**
- Focused test passes.
- `python mvp_capabilities/swarm_roster.py --cap-dir ~/.bloombee/capabilities --json` returns valid JSON or empty roster.

---

## Task 4: Add failing tests for route picker

**Objective:** Encode model-selection rules, including 10-laptop MVP scenario.

**Files:**
- Modify: `tests/test_mvp_capabilities.py`
- Create: `mvp_capabilities/route_picker.py`

**Tests:**
1. Single 16GB M4 chooses Qwen2.5-3B or smaller, not Qwen2.5-7B if free memory is too low.
2. Single 48GB M4 Pro can choose Qwen2.5-7B/Qwen3-14B class based on registry fit.
3. Synthetic 10×24GB laptop swarm recommends Qwen3-30B-A3B as MVP target.
4. Qwen3-235B-A22B remains stretch/unsupported unless total swarm memory clears threshold.
5. Measured bench data overrides roofline estimate.

**Command:**
- `pytest tests/test_mvp_capabilities.py::test_route_picker_recommends_qwen3_30b_for_10_laptops -q`
- Expected RED: import/module missing.

---

## Task 5: Implement `route_picker.py`

**Objective:** Choose strongest feasible model and placement plan.

**Files:**
- Create: `mvp_capabilities/route_picker.py`
- Possibly modify: `mvp_capabilities/MODEL_REGISTRY.yaml` to add `quality_rank`, `mvp_target`, `stretch_target`, `requires_moe_handler`, `placement` metadata.

**Implementation notes:**
- Parse YAML via `yaml.safe_load`.
- Compute per-peer fit: `recommended_min_free_mem_gb <= peer_memory`.
- Compute swarm fit: `sum_free_gb >= model_min_mem * safety_factor` and per-layer sharding feasible.
- Ranking: prefer highest `quality_rank`, then measured tok/s, then lower memory.
- Placement labels:
  - `solo` — one peer fits.
  - `replicated` — multiple peers can each run it.
  - `block_parallel_candidate` — model fits aggregate swarm memory but not one peer.
  - `unsupported` — no safe placement.

**Verification:**
- All route picker tests pass.
- CLI example works with synthetic peer JSON.

---

## Task 6: Add failing tests for bench sweep output

**Objective:** Define bench-matrix JSON format without running expensive model loads in tests.

**Files:**
- Modify: `tests/test_mvp_capabilities.py`
- Create: `mvp_capabilities/sweep_models.py`

**Tests:**
- Given registry + peer with 8GB free, select only tiny models.
- Existing bench JSON lines are parsed and merged into `bench_matrix.json`.
- Dry-run mode prints planned commands without loading models.

**Command:**
- `pytest tests/test_mvp_capabilities.py::test_sweep_models_dry_run_selects_feasible_models -q`
- Expected RED.

---

## Task 7: Implement `sweep_models.py`

**Objective:** Generate real throughput matrix per peer.

**Files:**
- Create: `mvp_capabilities/sweep_models.py`

**Implementation notes:**
- Use `bench_throughput.py` as subprocess in non-test mode.
- Default to `--dry-run` unless `--execute` is passed to avoid surprise downloads/OOM.
- Output: `~/.bloombee/benchmarks/<hostname>.json` or `--out`.
- Respect `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` when requested.

**Verification:**
- Dry-run tests pass.
- Optional local execute only for TinyLlama/Qwen2.5-0.5B if cached.

---

## Task 8: Add docs page `docs/distributed-inference-mvp.md`

**Objective:** Make the MVP understandable and demo-ready.

**Files:**
- Create: `docs/distributed-inference-mvp.md`
- Modify: `mvp_capabilities/README.md`

**Content:**
- Correct host map.
- MVP deliverables and status.
- Current verified local bench numbers.
- M4 Pro verification commands.
- 10-laptop plan and model targets.
- Showcase checklist.
- Known blockers and no-overclaiming section.

**Verification:**
- Static grep tests or docs assertions in `tests/test_mvp_capabilities.py` ensure required names/claims exist.

---

## Task 9: Remote verification on M4 Pro

**Objective:** Prove the tools work on `evinova-self` 48GB.

**Commands:**
```bash
ssh m4pro 'cd ~/Projects/distributed-inference-mvp && git status --short && source .venv/bin/activate && python mvp_capabilities/peer_scan.py'
ssh m4pro 'cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate && pytest tests/test_mvp_capabilities.py -q'
```

**If remote repo lacks latest local files:**
- Use tar/rsync over SSH for bounded sync, excluding `.git`, caches, `.venv` unless needed.
- Verify remote files before tests.

**Verification:**
- Remote peer scan reports ~48GB RAM and MPS.
- Remote tests pass.

---

## Task 10: Optional Qwen2.5-7B / BloomBee server gate

**Objective:** Attempt real server bring-up only after router/tools are green and M4 Pro access is stable.

**Commands:**
```bash
ssh m4pro 'cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m bloombee.server.server --model_name_or_path Qwen/Qwen2.5-7B-Instruct --num_blocks 28 --throughput --device mps --port 31330'
```

**Acceptance:**
- Server reaches `Started` or equivalent.
- Peer ID captured.
- Client inference returns real text and tok/s.

**If blocked:**
- Record exact blocker in docs, do not mark server gate complete.

---

## Definition of done

- Plan exists under `.hermes/plans/`.
- Tests cover roster, route picker, sweep dry-run, docs naming, and 10-laptop target.
- Local tests pass on M4 16GB.
- Local peer scan + route picker run with real peer JSON.
- Remote M4 Pro peer scan/tests attempted via direct SSH and results recorded.
- Docs page is named `distributed-inference-mvp` and includes 10-laptop MVP scope.
- No claim that the physical 10-laptop showcase completed.
- Git status reviewed; intended files committed if project workflow allows.
