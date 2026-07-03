# mvp_capabilities — BloomBee MVP capability & routing tooling

Small, independent tools that together answer:
**"Given a swarm of BloomBee peers, who can serve which model, and at
what speed?"**

The final MVP plan is in
[`docs/distributed-inference-mvp-final-plan.md`](../docs/distributed-inference-mvp-final-plan.md).
The target behavior is dynamic best-model selection: when users join the live
demo by link/QR, the coordinator should choose the strongest **proven** model the
connected devices can actually run, while showing stronger-but-unproven models as
experimental/blocked instead of silently overclaiming them. Speculative decoding
may speed up the selected verifier model, but output is only exact-equivalent to
that verifier; GLM-5.2 / DeepSeek-V4-class output requires those models to run as
verifiers through a later high-compute backend path.

They are deliberately decoupled from BloomBee itself — no `import
bloombee` anywhere. The JSON they produce can be hand-fed into a
scheduler, printed in a CLI dashboard, or pushed into the DHT that
BloomBee's runtime already maintains.

---

## The MVP layers

| Layer | File | What it does | Output |
|------:|------|--------------|--------|
| 1. Hardware | `peer_scan.py` | Probes the local node: hostname, Tailscale IP, CPU model & counts, RAM, MPS/CUDA VRAM, ping latency to peers, free disk on `~/.cache/huggingface`. | JSON to stdout AND `~/.bloombee/capabilities/<hostname>.json` |
| 2. Catalog | `MODEL_REGISTRY.yaml` | Static footprint + arch metadata for ~20 candidate models (TinyLlama through Qwen3-235B-A22B, dense and MoE). | YAML, loaded by the scheduler |
| 3. Compatibility | `model_compat_scan.py` + `PROOF_STATUS.yaml` | Reads `config.json`, maps HF `model_type` to BloomBee support, merges proof gates, and emits honest claim level. | JSON compatibility report |
| 4. Proof ladder | `proof_ladder.py` | Audits ordered proof gates for prepared models and shows the next gate before any promotion. This is audit state only, not inference proof. | JSON proof-ladder report |
| 5. One-block proof harness | `one_block_proof.py` | Emits exact one-block server/client commands and verifies captured logs before a proof gate can be promoted. Planning mode is not proof. | JSON plan / verification report |
| 6. MVP status | `mvp_status.py` | Emits the weighted plan-completion percentage, progress bar, and next gate. This is status accounting only, not demo proof. | Markdown or JSON status report |
| 7. Benchmark | `bench_throughput.py` | Loads a model with transformers, runs prefill + autoregressive decode, prints `prefill_tok_per_s` and `decode_tok_per_s` plus peak memory. | Single JSON line on stdout |
| 8. Roster | `swarm_roster.py` | Aggregates one or more capability JSON directories, de-duplicates hosts, and prints a swarm summary. | JSON or table |
| 9. Join | `join_coordinator.py` + `join_http_server.py` + `join_client.py` + `join_card.py` | Creates shareable join-link offers, records token-scoped peer heartbeats, exposes HTTP health/offer/heartbeat/active endpoints, lets physical devices post peer-scan heartbeats, and renders SVG join cards. This is join/roster state only, not inference proof; SVG visual-grid scanner interop is explicitly unproven. | JSON offer / active heartbeat roster / SVG join card |
| 10. Route picker | `route_picker.py` | Chooses the strongest feasible model for the current roster or synthetic 10-laptop MVP scenario. Selector modes now separate planning from proof-gated demo choices. | JSON route decision |
| 11. Layer planner | `layer_planner.py` | Converts a selected model + roster into deterministic contiguous layer ranges by estimated free-memory capacity. This is placement planning only, not inference proof. | JSON layer-placement plan |
| 12. Joined layer plans | `join_layer_plan.py` | Converts active token-scoped coordinator heartbeats into `layer_planner.py` placements and optional launch-command runbooks. This is coordinator-to-planner handoff only, not inference proof. | JSON joined-roster layer plan |
| 13. Simulator | `swarm_simulator.py` | Rehearses synthetic/live rosters with failed hosts, selected model, route, and layer plan. Simulation only, not inference proof. | JSON scenario report |
| 14. Sweep planner | `sweep_models.py` | Builds or executes a benchmark sweep for all models that fit a peer. | Dry-run commands or measured JSON |

Layer 1 says *what the hardware is*. Layer 2 says *what models exist and how big they are*. Layer 3 says *whether a model is BloomBee-runnable and how proven it is*. Layer 4 says *which proof gate comes next*. Layer 5 prepares and verifies one-block proof evidence. Layer 6 says *how much of the plan is built*. Layer 7 says *what each model actually achieves on this hardware*.

A naive router is `peer_free_gb >= model.min_total_mem_gb`. A better router is `peer_decode_tok_s >= request.min_decode_tok_s`, using the benchmark numbers instead of the catalog alone.

---

## Quickstart

```bash
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate

# 1. Probe this machine. Writes ~/.bloombee/capabilities/<host>.json.
python mvp_capabilities/peer_scan.py

#    With peer ping list:
python mvp_capabilities/peer_scan.py --peers m4-pro,m4-laptop,node3.tail.ts.net

# 2. Inspect the model catalog (no code needed; it's data).
cat mvp_capabilities/MODEL_REGISTRY.yaml

# 3. Scan a local model config or cached HF config for BloomBee compatibility.
python mvp_capabilities/model_compat_scan.py \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/<snapshot> \
  --model-id Qwen/Qwen3-30B-A3B

# 4. Audit prepared proof ladders and next gates.
python mvp_capabilities/proof_ladder.py --fallback-ladder

# 5. Generate the Qwen3-8B one-block proof runbook.
python mvp_capabilities/one_block_proof.py plan --model Qwen/Qwen3-8B

#    After live server/client logs exist, verify them before updating proof status.
python mvp_capabilities/one_block_proof.py verify \
  --model Qwen/Qwen3-8B \
  --server-log .local/one-block-server.log \
  --client-log .local/one-block-client.log

# 6. Show weighted MVP build status.
python mvp_capabilities/mvp_status.py

#    Machine-readable form for dashboards/automation:
python mvp_capabilities/mvp_status.py --json

# 7. Benchmark the default small model on this machine (MPS, bf16).
python mvp_capabilities/bench_throughput.py

#    A bigger target — same shape of output:
python mvp_capabilities/bench_throughput.py --model Qwen/Qwen2.5-3B-Instruct --max-new-tokens 128

#    Force a specific device/dtype (handy for the M4 laptop when you're
#    remote-debugging on a CUDA box):
python mvp_capabilities/bench_throughput.py --device cuda --dtype fp16 --model Qwen/Qwen2.5-7B-Instruct

# 8. Aggregate real peer scans.
python mvp_capabilities/swarm_roster.py --cap-dir ~/.bloombee/capabilities --json

# 9. Create a join-link offer. This is roster/bootstrap state only.
python mvp_capabilities/join_coordinator.py offer \
  --coordinator http://m4pro.local:8787 \
  --ttl-seconds 600

#    Or run the stdlib HTTP coordinator on a normal host.
python mvp_capabilities/join_http_server.py \
  --host 0.0.0.0 \
  --port 8787 \
  --coordinator http://m4pro.local:8787

#    On a joining device, scan capabilities then post a heartbeat.
python mvp_capabilities/peer_scan.py --out ~/.bloombee/capabilities/$(hostname -s).json
python mvp_capabilities/join_client.py \
  --join-url 'bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token' \
  --capabilities ~/.bloombee/capabilities/$(hostname -s).json

#    Render a dependency-free visual join card for operator handoff.
#    This embeds the exact URL but does not yet claim QR scanner compatibility.
python mvp_capabilities/join_card.py \
  --join-url 'bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token' \
  --out .local/join-card.svg

# 10. Pick the strongest feasible route for real devices.
python mvp_capabilities/route_picker.py --cap-dir ~/.bloombee/capabilities

#    Safe demo mode only auto-selects models with full_generation proof.
python mvp_capabilities/route_picker.py \
  --cap-dir ~/.bloombee/capabilities \
  --selector-mode safe-demo

#    Showcase-attempt permits experimental proven-wrapper models, but still
#    blocks missing-wrapper frontier candidates.
python mvp_capabilities/route_picker.py \
  --cap-dir ~/.bloombee/capabilities \
  --selector-mode showcase-attempt \
  --explain

# 11. Plan the 10-laptop MVP showcase route before physical showcase day.
python mvp_capabilities/route_picker.py \
  --cap-dir ~/.bloombee/capabilities \
  --scenario mvp-10-laptop \
  --synthetic-m4-laptops 10 \
  --synthetic-total-gb 24 \
  --synthetic-free-gb 20

# 12. Plan deterministic layer placement for the selected model.
python mvp_capabilities/layer_planner.py \
  --cap-dir ~/.bloombee/capabilities \
  --model Qwen/Qwen3-30B-A3B

#    Or plan the synthetic 10-laptop showcase split.
python mvp_capabilities/layer_planner.py \
  --model Qwen/Qwen3-30B-A3B \
  --synthetic-m4-laptops 10 \
  --synthetic-total-gb 24 \
  --synthetic-free-gb 20

# 13. Simulate a 10-laptop failure scenario before showcase day.
python mvp_capabilities/swarm_simulator.py \
  --scenario mvp-10-laptop \
  --model Qwen/Qwen3-30B-A3B \
  --synthetic-m4-laptops 10 \
  --fail-host m4-laptop-01 \
  --request-count 2

# 14. Generate the local real-demo dashboard (real connected peers only).
python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --out .local/demo-dashboard.html \
  --refresh-seconds 10 \
  --watch-seconds 2

# Optional: add a clearly-labelled synthetic planning panel, not for live demos.
python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --synthetic-m4-laptops 10 \
  --out .local/demo-dashboard-planning.html

# 15. Plan a per-peer benchmark sweep without downloading/running models.
python mvp_capabilities/sweep_models.py \
  --peer ~/.bloombee/capabilities/$(hostname -s).json \
  --dry-run
```

Default benchmark is `Qwen/Qwen2.5-0.5B-Instruct` at 128 prefill + 64 decode tokens. On an M4 Pro it downloads in ~10 s and runs end-to-end in under 30 s.

---

## Verified MVP status

As of the current implementation slice:

- Weighted engineering-build status from `mvp_status.py`:
  `████████████░░░░░░░░ 61%` built from the plan, with claim boundary
  `weighted_plan_status_not_demo_proof`. Next gate: Qwen3-8B one-block server
  proof.
- One-block proof harness (`one_block_proof.py`) exists. It emits exact
  Qwen3-8B server/client commands and verifies captured logs before allowing the
  `one_block_server` gate to be marked passed; it is harness-only until a live
  run succeeds.

- Local `evinova` / `Evis-MacBook-Pro`: M4, 16GB unified memory, MPS.
- Remote `evinova-self` / `m4pro`: M4 Pro, 48GB unified memory, verified via `ssh m4pro`.
- Fresh repo-local live scan on 2026-07-03: local M4 has ~2.3GB free and
  m4pro has ~34.5GB free; combined live roster is 2 peers, 64GB total,
  ~36.8GB free.
- Real two-device roster route currently picks `google/gemma-2-9b-it` as a solo M4 Pro route when M4 Pro has enough free memory.
- Synthetic 10-laptop MVP route picks `Qwen/Qwen3-30B-A3B` as the block-parallel candidate.
- Prepared Qwen3-30B-A3B 2507 variants (`Instruct-2507`, `Thinking-2507`)
  are registered as Qwen3-MoE candidates with pending proof; `safe-demo` will
  not auto-select them until `full_generation` passes.
- Proof ladder audit (`proof_ladder.py`) exists. Qwen3-8B and Qwen3-14B have
  passed config-only prescan as `qwen3` dense models, but one-block,
  multi-block, and full-generation gates remain pending; they are experimental,
  not `safe-demo` selectable.
- Join-link and heartbeat foundation (`join_coordinator.py`) exists: shareable
  `bloombee://join?...` offers and token-scoped active heartbeat rosters.
  `join_http_server.py` exposes `/healthz`, `/offer`, `/heartbeat`, and `/active`
  HTTP endpoints with explicit `no_inference_proof` claim boundaries. In the
  Hermes sandbox, dispatch functions are verified without binding a port because
  socket bind is blocked.
- Physical-device join client (`join_client.py`) exists: it parses a
  `bloombee://join?...` offer, loads peer-scan capabilities, and posts a
  heartbeat to the coordinator. Dry-run mode prints the exact request without
  network side effects.
- SVG join-card renderer (`join_card.py`) exists: it embeds the exact join URL
  in text/data attributes and renders a deterministic visual grid. It carries
  `scanner_interop_unproven`; true QR scanner compatibility is still a future
  proof gate.
- Demo dashboard (`demo_dashboard.py`) surfaces `mvp_status.py` progress, next
  gate, remaining percentage, and milestone table beside routes/evidence.
- Proof-state observability (`proof_state.py`) parses retained status/log/cache
  facts from long-running proof prep and feeds the dashboard without promoting
  inference gates.
- Layer planner (`layer_planner.py`) exists: it assigns deterministic contiguous
  layer ranges from a selected model and live/synthetic peer roster. With
  `--include-launch-commands`, it adds exact BloomBee server command runbooks
  and `join_layer_plan.py` can feed active token-scoped coordinator heartbeats
  into those placements.
  while preserving a no-server-started claim boundary.
- Simulation harness (`swarm_simulator.py`) exists: it rehearses variable-device
  rosters and failed-host scenarios, then emits route + layer-plan JSON with an
  explicit simulation-only/no-inference-proof claim boundary.
- Demo dashboard generator (`mvp_capabilities/demo_dashboard.py`) emits a local
  dark HTML dashboard with connected devices, real-swarm route cards, measured
  throughput, inference evidence, real layer-placement metadata, live telemetry
  counters, and claim boundaries. Synthetic 10-laptop planning is hidden by
  default and appears only with `--synthetic-m4-laptops`.
- Real layer-placement proof (2026-07-03): three live BloomBee server processes
  on `m4pro` served TinyLlama layers `0:8`, `8:15`, and `15:22`; a direct client
  call over all `0:22` layers returned finite outputs and gradients
  (`forward_seconds=0.529`, `backward_seconds=0.266`, `grad_finite=true`).
- TinyLlama distributed inference has been verified through two-server,
  two-laptop, three-peer, forward-loop text parity, and cached `.generate()`
  parity evidence, including S2S-enabled cached generation with direct fallback
  as the default correctness path.
- Qwen3-30B-A3B MoE support has passed config/wrapper tests and one live M4 Pro
  block-shard proof: block `0:1` loaded real safetensors and served direct RPC
  forward/backward with finite output and gradient.
- Phone/mobile peer support is at capability-discovery stage: `peer_scan.py`
  now emits a `mobile` profile for Android/Termux devices, but no phone is
  counted as a useful inference worker until it produces throughput evidence and
  successfully serves at least one transformer block in the distributed path.
- Physical 10-laptop showcase remains part of MVP scope. The next hard gates are
  full multi-block Qwen3-30B-A3B distributed serving, two-laptop cached
  `.generate()` with S2S/default fallback, and then the physical showcase.

### Measured M4 Pro bf16 bench (2026-07-02)

Source: `bench_evidence/m4pro_bf16_2026-07-02.jsonl` (one JSON line per model).

| Model | Prefill tok/s | Decode tok/s |
|--:|--:|--:|
| Qwen/Qwen2.5-0.5B-Instruct | 587.2 | 11.4 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 517.1 | 17.7 |
| Qwen/Qwen2.5-1.5B-Instruct | 216.0 | 13.2 |
| Qwen/Qwen2.5-3B-Instruct | 107.4 | 3.6 |
| Qwen/Qwen2.5-7B-Instruct | 65.4 | 2.6 |

Reproduction:
```bash
python mvp_capabilities/bench_matrix.py \
  mvp_capabilities/bench_evidence/m4pro_bf16_2026-07-02.jsonl \
  --default-host m4pro > .local/m4pro-bench-matrix.json
python mvp_capabilities/route_picker.py \
  --cap-dir ~/.bloombee/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json
```

---

## Wiring into BloomBee

These tools **do not import BloomBee** on purpose. The hivemind/DHT stack is heavy and version-sensitive; routing decisions shouldn't pay that cost.

Three reasonable integration points:

1. **Static config dump.** Run `peer_scan.py` once on each peer, commit the JSON files into `bloombee/peer_capabilities/` alongside the swarm config, and let the existing config-loader read them.
2. **CLI dashboard.** A `swarm status` command in the CLI that runs `peer_scan.py` over SSH or Tailscale to each peer and joins the result with the model catalog to render a table: `host | free_gb | best_model | decode_tok/s`.
3. **DHT population.** On startup, each peer pushes a small capability record into the BloomBee DHT (one key per peer), and the route picker queries the DHT for "peers where `min_total_mem_gb <= free_gb` AND `decode_tok_per_s >= threshold`". The benchmark numbers are an offline measurement, not a runtime query — re-benchmark when hardware changes, not per request.

All three patterns consume the same JSON shape that `peer_scan.py` already emits, so swapping mechanisms is a one-file change.

---

## Roofline sanity check

The decode phase of an autoregressive LLM is *memory-bandwidth-bound*: every step you have to read every parameter from VRAM/RAM at least once to produce the next token. The roofline (theoretical maximum, before any overhead) is therefore:

```
decode_tok_per_s_roofline  ≈  mem_bandwidth_GB_s  /  (params_b × dtype_bytes)
```

Example, Qwen2.5-7B in bf16 on an M4 Pro:
- M4 Pro unified-memory bandwidth ≈ **273 GB/s**
- params = 7.62 B, dtype_bytes = 2 → weights = 15.24 GB
- roofline ≈ 273 / 15.24 ≈ **17.9 tok/s**

In practice you will measure **40–70 %** of that — call it **8–12 tok/s** — because:

- the attention KV cache and activations also consume bandwidth per step
- framework allocator overhead (PyTorch caching allocator padding, MPS graph captures)
- the LM head matmul at vocab≈150k is not pure weight-read
- batch=1 wastes compute parallelism that real serving pools via continuous batching

If `bench_throughput.py` reports a number *higher* than the roofline, suspect either a smaller model than you think, or that the GPU isn't fully resident. If it's *much* lower (< 30 % of roofline), suspect CPU offload, an under-spec'd `dtype`, or another process pinning the accelerator.

This formula is the first thing to check when a peer that "should" be fast is slow.