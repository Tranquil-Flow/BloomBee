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

Already proven:

- TinyLlama two-server forward/backward.
- TinyLlama two-laptop forward/backward over LAN.
- TinyLlama three-peer forward/backward.
- TinyLlama forward-loop text parity.
- TinyLlama cached `.generate()` parity.
- TinyLlama S2S opportunistic push with client-direct fallback.
- Real dashboard artifact with connected devices, route evidence, telemetry, and
  real layer-placement metadata.
- Three real BloomBee server processes on `m4pro` serving TinyLlama layer ranges
  `0:8`, `8:15`, and `15:22`.
- Qwen3-30B-A3B MoE one-block live server shard on M4 Pro.

Not yet proven:

- self-serve QR/link laptop join,
- automatic layer assignment across arbitrary joined laptops,
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

### Current candidate ladder

| Tier | Model | Why | Status |
|---|---|---|---|
| Proven infra | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Small, fast, full proof ladder | `demo_safe` for infrastructure, low quality |
| Strong fallback | Qwen/Qwen3-8B | Good Qwen3 dense quality, moderate memory | wrapper exists, proof needed |
| Strong fallback | Qwen/Qwen3-14B | Better quality, plausible on M4 Pro / few laptops | wrapper exists, proof needed |
| Primary dream | Qwen/Qwen3-30B-A3B-Instruct-2507 | Best blend of quality, Apache license, MoE efficiency, swarm story | same family as proven Qwen3-MoE block, checkpoint proof needed |
| Existing MoE fallback | Qwen/Qwen3-30B-A3B | One-block proof already exists | full generation proof needed |
| Reasoning stretch | Qwen/Qwen3-30B-A3B-Thinking-2507 | stronger reasoning, longer outputs | proof needed; riskier latency |
| Post-MVP | Qwen/Qwen3-235B-A22B | flagship quality | requires far more memory / quant/backend work |

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

| Model | Rough fp16/bf16 budget | Why consider it | Current blocker | When to attempt |
|---|---:|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | ~560GB+ | strongest same-family Qwen3-MoE upgrade; better benchmark class than 30B | huge memory; not cached/proven | after Qwen3-30B full-generation works and connected swarm has ~29 × 20GB-free laptops or ~12 × 48GB-free laptops |
| `zai-org/GLM-4.5-Air` / FP8 variant | ~255GB bf16, less if FP8 path works | high-upside MoE reasoning/agent model; MIT; smaller than 235B | no BloomBee `glm4_moe` wrapper; FP8 path not integrated | after core demo works, if swarm has ~260GB free and wrapper investigation looks simple |
| `zai-org/GLM-4.5` | ~850GB bf16 | stronger GLM-class model | no wrapper; very large | post-MVP unless hardware is abundant |
| `moonshotai/Kimi-K2-Instruct` / newer Kimi K2.x | ~2.4TB bf16 | frontier open MoE/agentic capability | no wrapper; too large for normal laptop swarm | post-MVP / only with quantized expert paging or huge hardware |
| DeepSeek V3/V4-class MoE | ~1.6TB bf16 for V3-class | strong coding/reasoning class | no wrapper; MLA/DeepSeekMoE-specific state; huge | post-MVP / backend research |
| Qwen3-Coder large MoE class | likely ~1TB+ depending checkpoint | best coding-demo story if available | likely huge; wrapper/proof unknown | post-MVP unless quant/backend path exists |

High-compute rule: these models may appear in the dashboard as `last-stage` or
`blocked-by-wrapper/memory/proof`, but they must not delay the core live demo.
If one appears feasible, run the same proof ladder as every other model: prescan,
one-block server proof, multi-block proof, full distributed generation, then
load/multi-request proof.

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

## Speculative decoding and phones

Speculative decoding is important, but should not block the core join/layer/model
selection MVP. The best MVP path is:

1. Build a verifier loop with exact output parity.
2. Start with n-gram or cheap multi-token draft provider.
3. Add dashboard metrics: proposed, accepted, rejected, acceptance rate, wall-clock delta.
4. Treat Android phones as async draft workers after capability and throughput proof.
5. Keep iOS as control-plane/async-only for now.
6. Defer Eagle3 chain-mode unless compatible draft assets are already available.

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

1. `model_compat_scan.py` + proof-status registry.
2. Add Qwen3-30B-A3B-Instruct-2507 / Thinking-2507 candidates with pending proof.
3. Best-model selector with `safe-demo`, `showcase-attempt`, and `planning` modes.
4. QR/link join coordinator and heartbeat state.
5. Layer planner from live worker capabilities.
6. M4 Pro simulation harness for variable-device routing/load/failure.
7. Qwen3 dense fallback proofs: 8B, then 14B.
8. Qwen3-30B-A3B multi-block/full-generation proof ladder.
9. Multi-request chain scheduler.
10. Speculative verifier + cheap draft provider; phones as async draft workers.
11. Last-stage high-compute shelf, only after 1–10 work: Qwen3-235B-A22B first,
    then GLM-4.5-Air if a `glm4_moe` wrapper looks tractable, with Kimi/DeepSeek
    giant MoEs reserved for post-MVP backend/quantization research.
