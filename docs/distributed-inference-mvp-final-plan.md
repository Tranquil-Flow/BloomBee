# Distributed Inference MVP — final plan and coherence review

This is the integrated MVP plan for the live BloomBee distributed-inference demo.
The core idea is not "hardcode one model." The core idea is:

> Users join a swarm by link or QR code; the coordinator measures the connected
> hardware, selects the strongest **proven** model that the live swarm can run,
> assigns real layer ranges, runs distributed inference, and shows every device,
> chain, request, metric, and claim boundary on the dashboard.

The demo must remain honest: synthetic peers are allowed for planning and stress
simulation, but the live surface uses real connected devices only.

## North-star demo

1. Coordinator starts a local demo server and dashboard.
2. Participants scan a QR code or open a join link.
3. Each laptop runs a short join command, scans capabilities, and registers with
   the coordinator.
4. Coordinator chooses the best model under live constraints:
   - available memory,
   - measured or estimated throughput,
   - architecture support,
   - proof status,
   - model benchmark/quality rank,
   - requested demo risk level.
5. Coordinator assigns each worker a layer range and launches/prints the exact
   BloomBee server command.
6. Dashboard shows:
   - connected devices,
   - which devices serve which layers,
   - selected model and why it won,
   - stronger models that were considered and why they were rejected,
   - active chains and utilisation,
   - request queue / latency / tok/s,
   - memory pressure,
   - recovery and S2S telemetry,
   - speculative-draft metrics if enabled,
   - explicit claim boundaries.
7. Demo sends real prompts through distributed inference.
8. Multi-request load run proves routing/scheduling behavior.

## Success criteria

The MVP is complete when the following are all true:

- A fresh laptop can join through link/QR without manual bespoke setup.
- Dashboard shows that laptop as a real connected device.
- Coordinator assigns concrete `start:end` layer ranges to real devices.
- At least one distributed generation proof succeeds on a model selected by the
  coordinator from live connected resources.
- Multiple requests are routed through healthy chains with visible utilisation
  counters.
- If the live swarm is larger than the test swarm, the coordinator can select a
  stronger prepared model automatically.
- If a prepared stronger model is not proven enough for public demo, it is shown
  as `experimental` or `blocked`, not silently used as if ready.
- No synthetic device is shown in real-demo mode.

## Current proof ladder

Current weighted engineering-build status from `mvp_capabilities/mvp_status.py`:

```text
██████████████░░░░░░ 71%
```

Claim boundary: `weighted_plan_status_not_demo_proof`. This is plan progress, not
public-demo proof. Next gate: **Qwen3-8B multi-block or full-generation proof**.

Already proven:

- TinyLlama two-server forward/backward.
- TinyLlama two-laptop forward/backward over LAN.
- TinyLlama three-peer forward/backward.
- TinyLlama forward-loop text parity.
- TinyLlama cached `.generate()` parity.
- TinyLlama S2S opportunistic push with client-direct fallback.
- Real dashboard artifact with connected devices, route evidence, weighted MVP
  status/next gate, live proof-prep feed, joined-peer layer-plan runbooks,
  chain-scheduler rehearsal panels, telemetry, and real layer-placement metadata.
- Active join-heartbeat rosters can feed deterministic layer-placement runbooks
  through `join_layer_plan.py`, either from local state or coordinator HTTP
  `/active`; operator-captured seed multiaddrs can now resolve follower launch
  commands before no-execution launch-readiness checklists mark the runbook
  startable.
- Three real BloomBee server processes on `m4pro` serving TinyLlama layer ranges
  `0:8`, `8:15`, and `15:22`.
- Qwen3-30B-A3B MoE one-block live server shard on M4 Pro.
- Model compatibility scanner plus proof-status registry: local `config.json`
  prescan, BloomBee family mapping, unsupported-wrapper blocking, and claim-level
  output are test-covered.
- Proof ladder audit exists: ordered gate reports show next promotion gates for
  prepared models without claiming inference. Qwen3-8B and Qwen3-14B passed
  config-only prescan as supported `qwen3` dense models; Qwen3-8B weights are
  cache-complete on M4 Pro, stale `.incomplete` leftovers are distinguished by
  `proof_state.py`, and Qwen3-8B `one_block_server` is passed from live M4 Pro
  evidence. Multi-block and full-generation gates remain pending.
- One-block proof harness exists: `one_block_proof.py` emits exact server/client
  commands and verifies captured logs for the Qwen3-8B `one_block_server` gate.
  It does not itself prove inference; a live run must still pass.
- Multi-block proof harness exists: `multi_block_proof.py` emits two-or-more
  server runbooks and verifies only when every server records start,
  block-range announcement, RPC evidence, plus a combined direct-client result.
  Qwen3-8B multi-block remains pending; initial M4 Pro attempts started both
  servers but failed at client DHT bootstrap before RPC proof.
- Route picker selector modes are wired: `planning` keeps memory-fit simulation,
  `showcase-attempt` allows experimental supported wrappers while blocking
  missing-wrapper candidates, and `safe-demo` requires `full_generation` proof.
- Prepared Qwen3-30B-A3B 2507 variants are in the registry with fetched config
  metadata and pending proof: Instruct-2507 and Thinking-2507.
- Join-flow foundation exists: shareable join-link offers, token-scoped heartbeat
  state, active-peer filtering, stdlib HTTP `/healthz`, `/offer`, `/heartbeat`,
  `/active`, `/route`, and `/plan` endpoints, proof-aware auto model selection
  for `/plan?model=auto`, physical-device `join_client.py` heartbeat posting,
  SVG join-card rendering, QR scanner dependency preflight, and explicit
  no-inference-proof claim boundaries. SVG visual-grid scanner interoperability
  remains unproven until a generated artifact decodes to the exact join URL.
- Layer planner exists: selected model + live/synthetic roster becomes
  deterministic contiguous layer ranges, with optional exact BloomBee server
  launch command runbooks, operator-captured seed multiaddr substitution,
  launch-readiness checklists, and explicit `launch_commands_only_no_server_started`
  / `launch_multiaddr_resolution_only_no_server_started` / readiness claim
  boundaries; active coordinator heartbeats can be handed into the same planner
  via `join_layer_plan.py` from local state or HTTP `/active`.
- Chain scheduler planning exists: joined layer plans become multi-request waves,
  per-peer scheduled-token estimates, and `planned_no_live_traffic` health
  reports. `request_telemetry.py` summarizes direct-client `[direct] RESULT` logs,
  request success/failure counts, latency, model/block coverage, and blockers;
  `demo_dashboard.py --chain-schedule ... --request-log ...` renders those
  planned waves plus live request telemetry without claiming load proof.
- Simulation harness exists: synthetic/live rosters can be rehearsed with failed
  hosts, route selection, and layer placement while staying simulation-only.

Not yet proven:

- self-serve QR/link laptop join,
- automatic layer assignment across arbitrary joined laptops after server launch
  multiaddrs are captured,
- physical N-laptop showcase,
- full Qwen3-30B-A3B distributed generation,
- Qwen3-30B-A3B 2507 checkpoint proof,
- multi-chain same-model request routing,
- phone as useful inference or draft worker,
- true continuous batching,
- real prefill KV prefix reuse.

## Dynamic best-model selection

The live demo should select the best model from a prepared ladder, not hardcode a
single target. Model choice must be a constrained optimisation problem:

```text
candidate models
  -> filter by architecture support
  -> filter by proof status required for demo risk level
  -> filter by live aggregate memory with safety margin
  -> filter by per-peer layer feasibility
  -> score by quality/benchmark rank, throughput estimate, license, context,
     MoE efficiency, and risk
  -> select best safe model
```

### Model readiness fields

Each candidate model should carry these fields in a registry or generated proof
index:

```yaml
model_id: Qwen/Qwen3-30B-A3B-Instruct-2507
family: qwen3_moe
license: Apache-2.0
total_params_b: 30.5
active_params_b: 3.3
num_layers: 48
recommended_free_gb: 70
context_cap_for_mvp: 32768
quality_rank: ...
architecture_supported: true
prescan_status: pending|passed|failed
one_block_server_status: pending|passed|failed
multi_block_status: pending|passed|failed
full_generation_status: pending|passed|failed
cache_generation_status: pending|passed|failed
measured_throughput: []
claim_level: planning|experimental|demo_safe
```

Hard rule: public real-demo mode may only auto-select models whose `claim_level`
is `demo_safe`. The dashboard may show stronger experimental candidates, but must
label them clearly.

### Demo risk levels

Use explicit risk modes:

| Mode | Allowed proof level | Purpose |
|---|---|---|
| `safe-demo` | full distributed generation proof | public live demo |
| `showcase-attempt` | multi-block or one-block proof plus operator approval | attempt stronger model live |
| `planning` | memory/fit only | synthetic planning, never user-facing as proof |

### Build-ready target ladder

This is the target ladder the coordinator should reason over. Each tier is a
candidate **only after** its proof gate passes for the selected risk mode.

| Tier | Model / class | Output guarantee | Why | Status / gate |
|---|---|---|---|---|
| Infra fallback | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exact TinyLlama | small, fast, already proves the distributed runtime | `demo_safe` for infrastructure, low quality |
| Quality fallback | `Qwen/Qwen3-8B` | exact Qwen3-8B | good dense Qwen3 fallback for small swarms | wrapper exists; proof needed |
| Strong fallback | `Qwen/Qwen3-14B` | exact Qwen3-14B | better fallback if M4 Pro / few laptops can host it | wrapper exists; proof needed |
| Core dream | `Qwen/Qwen3-30B-A3B-Instruct-2507` | exact Qwen3-30B verifier output | best practical blend of quality, Apache license, MoE efficiency, and laptop-swarm fit | same family as proven Qwen3-MoE block; checkpoint proof needed |
| Existing MoE fallback | `Qwen/Qwen3-30B-A3B` | exact Qwen3-30B verifier output | one-block live serving already proven | full generation proof needed |
| Reasoning stretch | `Qwen/Qwen3-30B-A3B-Thinking-2507` | exact Thinking-2507 verifier output | stronger reasoning but longer/slower outputs | proof needed; cap thinking budget |
| High-compute exact | `Qwen/Qwen3-235B-A22B-Instruct-2507` | exact Qwen3-235B verifier output | strongest same-family Qwen3-MoE upgrade if huge memory appears | last-stage only; full proof ladder required |
| Frontier backend experiment | `zai-org/GLM-5.2` / FP8 / quantized variants | exact GLM-5.2 only if GLM-5.2 is verifier | frontier open-weight coding/agentic target | post-core LayerExecutor/quantized-backend path; no native BloomBee wrapper yet |
| Frontier backend experiment | `deepseek-ai/DeepSeek-V4-Flash` | exact V4 Flash only if V4 Flash is verifier | more plausible DeepSeek V4 target than Pro due to smaller total weights | post-core quantized backend path |
| Frontier post-MVP | `deepseek-ai/DeepSeek-V4-Pro`, Kimi K2.x, giant Qwen3-Coder MoEs | exact only if those models are verifiers | highest benchmark ceiling | post-MVP unless very large hardware or expert paging exists |

Gemma 3 / 27B-class models may benchmark well and support multimodal input, but
for this BloomBee MVP they are lower priority unless their exact model family is
wrapped and proven. Qwen3-MoE is the better swarm story: strong quality, Apache
license, and much lower active-parameter compute than dense 27B/32B models.

## Stronger-model readiness before the live demo

Because the live demo may have more compute than today's two-laptop testbed, we
need a prepared model shelf. Each shelf item needs as much proof as possible
before demo day.

### P0 prepared shelf

1. TinyLlama — current safe infrastructure fallback.
2. Qwen3-8B — prove dense Qwen3 path on available hardware.
3. Qwen3-14B — prove if M4 Pro memory allows.
4. Qwen3-30B-A3B — extend from one-block proof to multi-block proof.
5. Qwen3-30B-A3B-Instruct-2507 — prescan and one-block proof; promote only if it
   passes the same gates.

### Last-stage high-compute shelf

These models are **not** prerequisites for the MVP. They should come last, only
after the join flow, layer planner, dashboard, best-model selector, scheduler,
simulation harness, fallback proofs, and Qwen3-30B proof ladder are working. The
reason to include them in the plan is readiness: if demo day unexpectedly brings
far more aggregate memory than our current two-laptop testbed, the coordinator
should know which bigger models are worth attempting and which are blocked.

| Model | Rough weight budget | Why consider it | Current blocker | When to attempt |
|---|---:|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | ~560GB+ bf16/fp16 with margin | strongest same-family Qwen3-MoE upgrade; better benchmark class than 30B | huge memory; not cached/proven | after Qwen3-30B full-generation works and connected swarm has ~29 × 20GB-free laptops or ~12 × 48GB-free laptops |
| `zai-org/GLM-5.2` | ~1.8TB bf16, ~0.9TB FP8, ~0.45TB ideal FP4 floor with margin | frontier open-weight coding/agentic target; 744B total / ~40B active class | no native BloomBee wrapper; needs quantized backend or enormous swarm | after core demo works, via LayerExecutor wrapping vLLM/SGLang/llama.cpp-style backend before native BloomBee block work |
| `deepseek-ai/DeepSeek-V4-Flash` | ~0.68TB bf16, ~0.34TB FP8, ~0.17TB ideal FP4 floor with margin | most plausible DeepSeek V4 family target; much smaller than Pro | no native wrapper; quantized backend path needed | post-core if quantized weights/backend are stable and hardware is large enough |
| `deepseek-ai/DeepSeek-V4-Pro` | ~3.8TB bf16, ~1.9TB FP8, ~0.96TB ideal FP4 floor with margin | frontier benchmark ceiling | no native wrapper; huge; special attention/MoE state likely | post-MVP / backend research only |
| Kimi K2.x / giant Qwen3-Coder MoEs | likely ~1–2.4TB+ depending checkpoint/precision | best long-horizon/agentic/coding ceiling if available | no wrapper; huge; quant/expert-paging required | post-MVP unless hardware pool is far larger than expected |

High-compute rule: these models may appear in the dashboard as `last-stage`,
`blocked-by-wrapper`, `blocked-by-memory`, or `blocked-by-proof`, but they must
not delay the core live demo. If one appears feasible, run the same proof ladder
as every other model. For GLM-5.2 / DeepSeek V4-class models, the first practical
path is a **LayerExecutor backend** that wraps a proven quantized serving backend
(vLLM, SGLang, llama.cpp/KTransformers-style), not a native BloomBee block wrapper.

### Proof ladder for each prepared model

1. `model_compat_scan`: AutoConfig dispatch, wrapper, layer count, block prefix,
   checkpoint index.
2. block wrapper unit tests if new architecture or checkpoint behavior differs.
3. one-block server dry/run proof.
4. multi-block direct RPC proof.
5. full-layer hidden-state finite proof.
6. forward-loop text parity.
7. cached `.generate()` parity.
8. multi-request/load proof.

A model is not `demo_safe` until at least full distributed text generation works.

## What limits which models we can use

1. **Architecture wrapper support.** BloomBee must know how to construct and load
   a single transformer block for the HF `model_type`. Registered families today
   include Bloom, Falcon, Gemma4, Llama, Mixtral, Qwen3 dense, and Qwen3-MoE.
2. **Checkpoint layout.** Each model family uses different parameter prefixes.
   Wrong prefix means per-block safetensor loading fails.
3. **Cache/state layout.** GQA, sliding attention, QK norm, MoE, and recurrent or
   convolutional state all affect the block wrapper and KV contract.
4. **Quantisation.** Current BloomBee MVP path is effectively fp16/bf16; do not
   plan on 4-bit/8-bit memory wins unless that path is separately implemented and
   proven.
5. **KV memory and context.** Long-context claims are dangerous. MVP should cap
   context to a proven budget (for example 4K, 8K, or 32K), not attempt 128K+
   just because a model card supports it.
6. **Network latency.** Pipeline-parallel hidden-state hops can dominate decode
   latency. More devices are not automatically better; chain planner must account
   for pairwise latency and utilisation.
7. **Proof status.** Fit prediction is not inference proof. The selector must not
   auto-promote unproven models in safe-demo mode.

## Coordinator components to build

### 1. Join coordinator

- `GET /join/<token>` shows instructions.
- `GET /join/<token>/install.sh` or equivalent bootstrap script.
- `POST /api/capabilities` registers worker scan.
- `POST /api/heartbeat` updates worker state.
- Dashboard reads coordinator state.

### 2. Model compatibility scanner

- Reads HF config/checkpoint index.
- Reports BloomBee family support and missing wrappers.
- Produces proof-status entries for model selector.

### 3. Best-model selector

Inputs:

- connected worker capabilities,
- measured benchmark matrix,
- model registry,
- proof-status index,
- demo risk mode.

Outputs:

- selected model,
- rejected stronger models and reasons,
- assignment plan,
- claim level.

### 4. Layer planner

- Splits model layers across workers.
- Weights by memory, measured throughput, pairwise latency, and stability.
- Keeps assignments contiguous.
- Avoids critical-memory workers.
- Emits exact BloomBee server commands.

### 5. Chain/request scheduler

- Tracks chain health and utilisation.
- Routes new requests to least-loaded healthy chain.
- Supports interactive vs batch queues.
- Records p50/p95 latency, TTFT, decode tok/s, queue wait, and failure events.

### 6. Simulation harness

Before live demo, simulate variable devices on M4 Pro:

- different worker counts,
- latency matrices,
- memory pressure,
- request bursts,
- worker failures,
- multiple chains,
- phone draft workers.

This lets us validate policies before a real 10-device gathering.

## Variable-device readiness matrix

| Scenario | Expected behavior |
|---|---|
| 1 laptop only | choose best solo proven model; no fake distributed claim |
| 2 laptops | choose best solo or two-device split depending proof/fit |
| 3 laptops | run TinyLlama real split; attempt Qwen3 dense if proven |
| 6+ laptops | attempt Qwen3-30B-A3B multi-block if proof gates pass |
| 10+ laptops | choose best proven prepared model; likely Qwen3-30B-A3B family |
| phones present | show as control-plane/draft candidates unless throughput proof exists |
| high-latency peer | assign fewer/no layers, avoid critical chain positions |
| memory-pressure peer | mark draining; stop new requests; replan if safe |
| unproven high-quality model fits | show as experimental/blocked, not demo-safe |

## Speculative decoding, phones, and output guarantees

Speculative decoding is a speed technique, not a magic quality upgrade. The output
is exactly equivalent to the **verifier/target** model only when every accepted
token is checked by that same verifier.

```text
small draft + Qwen3-30B verifier = exact Qwen3-30B output, faster if acceptance is high
small draft + Qwen3-30B verifier != GLM-5.2 output
small draft + GLM-5.2 verifier = exact GLM-5.2 output, but GLM-5.2 must actually run
```

### Exact verifier mode

Use this for demo claims that say "equivalent to model X":

- `verifier_model_id`: the model whose distribution/output is guaranteed.
- `draft_model_id`: n-gram, Qwen3 small dense, phone model, EAGLE/Medusa/MTP head,
  or another cheap proposal source.
- Every accepted token is validated by the verifier.
- Dashboard shows proposed tokens, accepted tokens, rejected tokens, acceptance
  rate, verifier latency, draft latency, wall-clock speedup, and exact guarantee.

MVP exact target:

```text
verifier: Qwen/Qwen3-30B-A3B-Instruct-2507 or Qwen/Qwen3-30B-A3B
draft: n-gram, Qwen3-1.7B/4B/8B, or Android phone draft provider after proof
guarantee: exact verifier-equivalent output
```

High-compute exact targets after the core demo works:

```text
verifier: Qwen/Qwen3-235B-A22B-Instruct-2507
verifier: GLM-5.2 via LayerExecutor backend
verifier: DeepSeek-V4-Flash via LayerExecutor backend
```

### Approximate teacher mode

Use this only for clearly-labelled experiments:

- A hosted or offline frontier model such as GLM-5.2 / DeepSeek V4 teaches,
  distils, plans, or critiques.
- A smaller local swarm model executes or imitates.
- Output is **not** exact frontier-model equivalent.
- Dashboard claim level must say `approximate-teacher`, `distilled`, or
  `cascade`, never `exact-verifier`.

### Phone role

Phones should not be counted as block workers until a real block-serving proof
exists. Their MVP role is:

1. control-plane participant,
2. capability scan source,
3. async draft provider for speculative decoding after throughput proof,
4. never part of the exact-output claim unless their proposed tokens are verified
   by the target verifier.

Defer Eagle3 chain-mode unless compatible draft assets are already available. The
first speculative implementation should be a simple exact verifier loop with an
n-gram or cheap multi-token draft provider.

## Prefill and caching

For final demo robustness, instrument before optimising:

- prompt tokens,
- prefill seconds,
- decode seconds,
- TTFT,
- cache hits/misses,
- repeated-prefix savings.

Prefill KV cache reuse is high value for shared system prompts and RAG contexts,
but correctness-sensitive. MVP should start with exact prefix hashing and metadata;
real KV reuse can be promoted after proof.

## Final coherent vision

The distributed inference MVP is a **live adaptive swarm**, not a static benchmark.
It should answer in real time:

```text
Who is connected?
What can they actually run?
Which model is strongest under current proof and resource constraints?
Which device runs which layers?
How fast is it?
What failed or recovered?
What stronger model would unlock if more devices joined?
```

The moonlit demo story:

> Scan a code. Join the swarm. Watch your laptop become a layer in a larger mind.
> The coordinator measures the living hardware, chooses the strongest safe model,
> and proves the output with real distributed inference.

## Final review: likely missed risks

- **Model registry overclaiming:** registry must distinguish fit, wrapper support,
  and proof. This is the most important correction to current route tooling.
- **Benchmark score vs serving proof:** public leaderboard quality is not enough;
  every model needs a BloomBee proof ladder.
- **Context length trap:** long-context model cards can imply impossible KV memory.
  MVP must cap context.
- **Dense-model trap:** dense 27B/32B models may fit aggregate memory but be much
  slower than Qwen3-MoE because all parameters are active per token.
- **More devices can hurt:** latency and hop count can dominate. Route planner must
  prefer good chains, not simply more peers.
- **Phones are not block workers yet:** they belong in draft/control-plane roles
  until a real block-serving proof exists.
- **Synthetic planning must stay separate:** useful for readiness, never proof.
- **Safe fallback required:** TinyLlama/Qwen3-8B/Qwen3-14B ladder prevents live demo
  from failing if Qwen3-30B full generation is not ready.

## Immediate next build order

Build in this order. Do not let frontier-model dreams block the core swarm demo.

1. `model_compat_scan.py` + proof-status registry. **Initial slice complete**:
   local config scanning, supported-family mapping, proof merge, and CLI JSON
   output exist. Next slice should feed this into the best-model selector.
2. Add Qwen3-30B-A3B-Instruct-2507 / Thinking-2507 candidates with pending proof.
   **Initial slice complete**: both prepared 2507 variants are in
   `MODEL_REGISTRY.yaml` as Qwen3-MoE candidates with pending proof gates.
3. Best-model selector with `safe-demo`, `showcase-attempt`, and `planning` modes.
   **Initial slice complete**: route picking and explain output include selector
   mode, proof status, claim level, and selectable/blocker metadata.
4. QR/link join coordinator and heartbeat state. **Foundation + HTTP slice
   complete**: join-link offers and heartbeat state exist; stdlib HTTP health,
   offer, heartbeat, and active-roster endpoints exist; physical-device join
   client wiring exists; SVG join-card rendering exists with scanner interop
   explicitly unproven; QR dependency preflight now fails closed when encoder or
   decoder packages are missing. True QR scanner proof and fresh-device showcase
   remain future work.
5. Layer planner from live worker capabilities. **Initial slice complete**:
   `layer_planner.py` emits deterministic contiguous block ranges from a model
   registry entry and peer free-memory capacities. **Launch-command slice
   complete**: optional BloomBee server commands are included with a
   no-server-started boundary. This is a placement/launch plan, not inference proof.
6. M4 Pro simulation harness for variable-device routing/load/failure.
   **Initial slice complete**: `swarm_simulator.py` emits a simulation-only
   route + layer-plan report for synthetic/live rosters and failed-host lists.
7. Qwen3 dense fallback proofs: 8B, then 14B. **Config-only prescan + harness
   slice complete** for both; Qwen3-8B cache snapshot is complete on M4 Pro and
   one-block server proof passed; next gate is Qwen3-8B multi-block/full-generation
   proof or Qwen3-14B one-block proof if memory allows.
8. Qwen3-30B-A3B / Instruct-2507 multi-block and full-generation proof ladder.
9. Multi-request chain scheduler. **Initial planning slice complete**:
   `chain_scheduler.py` emits request waves and per-peer utilization from a joined
   layer plan, with `chain_scheduler_plan_only_no_inference_proof`. Live load
   proof remains future work.
10. Exact speculative verifier loop with cheap draft provider; phones only as async
    draft workers after capability and throughput proof.
11. Qwen3-235B-A22B-Instruct-2507 last-stage same-family attempt, only if the
    connected swarm has enough memory and Qwen3-30B generation already works.
12. LayerExecutor backend interface for quantized frontier serving backends.
13. GLM-5.2 backend experiment after LayerExecutor works.
14. DeepSeek-V4-Flash backend experiment after LayerExecutor works.
15. DeepSeek-V4-Pro / Kimi K2.x / giant Qwen3-Coder MoEs as post-MVP research.

The first build sprint should therefore start at item 1, not at speculative
decoding or frontier-model wrappers.
