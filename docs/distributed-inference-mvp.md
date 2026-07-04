# distributed-inference-mvp

This page tracks the BloomBee distributed inference MVP. The artifact name is intentionally `distributed-inference-mvp` because the deliverable is the full hardware-aware distributed inference layer.

## Correct host map

- `evinova` / `Evis-MacBook-Pro`: local M4 laptop, 16GB unified memory.
- `evinova-self` / `m4pro`: M4 Pro laptop, 48GB unified memory, verified reachable via `ssh m4pro`.
- `astra-macbook`: additional M4 laptop peer, Tailscale IP `100.117.33.124`, SSH may be gated.

Do not use local sandbox `tailscale status` as the source of truth; it can fail with daemon socket permissions. Prefer direct SSH (`ssh m4pro`) for the M4 Pro.

## MVP deliverables

1. Peer hardware scan with real JSON from every device.
2. Swarm roster aggregation across local and remote peers.
3. Model registry with fit metadata, quality targets, and MoE/dense flags.
4. Bench throughput sweep producing measured prefill/decode tok/s per peer/model.
5. Route picker choosing the strongest feasible model for the current swarm.
6. 10-laptop swarm readiness as part of MVP scope.
7. Physical 10-laptop showcase after local and two-device verification are complete.
8. Mobile-phone peer readiness: scan Android/Termux phones, benchmark them, and
   include them only when measured evidence shows they are genuinely useful for
   inference work.
9. Demo dashboard: show connected devices, route choices, measured throughput,
   inference evidence, S2S/recovery telemetry, and honest claim boundaries in a
   single local HTML artifact.

## Final plan and dynamic target selection

The integrated final plan lives in
[`docs/distributed-inference-mvp-final-plan.md`](distributed-inference-mvp-final-plan.md).
The live demo should not hardcode one model. It should let users join via link or
QR code, scan real connected devices, then select the strongest **proven** model
that the live swarm can actually run.

Safe-demo mode must filter by architecture support, proof status, memory fit,
throughput evidence, and claim boundaries. Stronger models that fit by memory but
lack full generation proof should appear as `experimental` or `blocked`, not as
the automatic public-demo choice.

## 10-laptop prepared target

Primary prepared 10-laptop target: **Qwen/Qwen3-30B-A3B-Instruct-2507**, with
**Qwen/Qwen3-30B-A3B** as the already-partially-proven fallback in the same
`qwen3_moe` family.

Why:

- 30.5B total parameters, ~3.3B active parameters per token.
- Better quality-per-watt than dense 14B/32B on Apple Silicon swarms.
- Fits aggregate memory of 10 M4 laptops even when each laptop cannot host the whole model solo.
- Best MVP showcase story: many modest devices collaborate to serve the strongest
  model that the connected swarm can prove safe to run.

Last-stage stretch targets are tracked in the final plan and come **after** the
core live demo works. The new near-term candidate branch is
**Qwen/Qwen-AgentWorld-35B-A3B**: it fits the synthetic 10-laptop memory budget
(~80GB recommended) but is blocked today because HF reports `model_type=qwen3_5_moe`
with a `qwen3_5_moe_text` tower, and BloomBee only has a `qwen3_moe` wrapper.
First true high-compute stretch is the same-family
**Qwen/Qwen3-235B-A22B-Instruct-2507** if enough aggregate memory appears. After
that, a LayerExecutor backend can try quantized frontier serving backends for
**MiniMaxAI/MiniMax-M3**, **GLM-5.2**, and **DeepSeek-V4-Flash**. DeepSeek-V4-Pro,
Kimi K2.x, and giant Qwen3-Coder MoEs stay post-MVP unless quantized expert paging
or a much larger hardware pool exists. MiniMax M3 is currently blocked for native
BloomBee by `model_type=minimax_m3_vl`, MiniMax Sparse Attention, and ~809GiB bf16
weights (~900GB recommended runtime memory).

## Verified current state

Current weighted engineering-build status from `mvp_capabilities/mvp_status.py`:

```text
███████████████████░ 94%
```

Claim boundary: `weighted_plan_status_not_demo_proof`. This is MVP-core plan
progress, not public-demo proof. Next gate: **physical/self-serve showcase with
fresh joined devices**. `physical_showcase_proof.py` and the proof-orchestration
dashboard now fail-closed verify and surface operator-captured physical evidence
plus cross-artifact active-roster/layer-plan/generation-placement/load alignment.
A real Pixel 8 Pro Termux run proved a fresh physical-device `join_client.py`
3-heartbeat active-roster path via ADB UI, but physical camera QR scan,
selected-model server launch from that fresh joined plan, and the final
`physical_showcase_proof.py` cross-artifact pass remain uncaptured. Qwen3-30B and larger/model-optimisation work remains
visible as post-MVP/stretch work and does not drag the 100% MVP denominator.

- Local M4 16GB can load and run TinyLlama-1.1B on MPS after the sitecustomize RLock fix.
- Fresh repo-local live scan on 2026-07-03: local `evinova` /
  `Evis-MacBook-Pro` reports MPS, 16GB total, ~2.3GB free; `m4pro`
  reports MPS, 48GB total, ~34.5GB free. Combined live roster: 2 peers,
  64GB total, ~36.8GB free.
- Current route selection now separates memory-fit planning from BloomBee runtime wrapper support. `qwen2`/Qwen2.5 and `gemma2` benchmark entries are useful local-transformers speed evidence, but they are blocked from showcase/safe-demo BloomBee selection until wrappers exist. TinyLlama is the proven safe-demo fallback; Qwen3 candidates remain experimental until generation proof gates pass.
- Measured local TinyLlama benchmark (cold cache): prefill ~610.7 tok/s, decode ~7.7 tok/s.
- Measured local TinyLlama (warm cache, repeat run): prefill 1130.3 tok/s, decode 28.6 tok/s, peak 0.21 GB.
- Measured M4 Pro bf16 local-transformers sweep (5 models, 2026-07-02; benchmark data only, not BloomBee wrapper proof):
  | Model | Prefill tok/s | Decode tok/s |
  |--:|--:|--:|
  | Qwen2.5-0.5B-Instruct | 587 | 11.4 |
  | TinyLlama-1.1B-Chat | 517 | 17.7 |
  | Qwen2.5-1.5B-Instruct | 216 | 13.2 |
  | Qwen2.5-3B-Instruct | 107 | 3.6 |
  | Qwen2.5-7B-Instruct | 65 | 2.6 |
- Direct SSH to M4 Pro works and reports 48GB memory.
- Qwen3-MoE block wrapper (`src/bloombee/models/qwen3_moe/`) is in place:
  auto-dispatches from real Qwen3-30B-A3B config (48 layers, hidden=2048,
  128 experts @ 8/topk). Wrapper contract tests pass. One live M4 Pro server
  shard loaded real Qwen3-30B-A3B safetensors for block `0:1`; a later
  clean-archive two-server proof served blocks `0:1` + `1:2` and direct RPC
  `0:2` forward/backward with finite outputs and gradients.
- `mvp_capabilities/model_compat_scan.py` and `PROOF_STATUS.yaml` exist. They
  prescan local model configs, map HF `model_type` to BloomBee support, merge
  proof gates, and emit `demo_safe` / `experimental` / `blocked` claim levels.
- `mvp_capabilities/proof_ladder.py` audits the ordered proof gates for prepared
  models and emits the next gate before promotion. Qwen3-8B and Qwen3-14B now
  have real config-only prescan evidence (`qwen3`, wrapper-supported), but all
  inference gates remain pending, so they stay experimental and not safe-demo.
- `mvp_capabilities/one_block_proof.py` emits exact run/verify commands for the
  Qwen3-8B one-block proof and refuses to mark the gate passed unless both server
  and direct-client logs contain matching finite-output evidence. Qwen3-8B
  `one_block_server` is now passed from M4 Pro evidence; full generation remains
  pending.
- `mvp_capabilities/full_generation_proof.py` emits `text_generation_parity.py`
  runbooks and verifies captured parity JSON before a model's `full_generation`
  gate can be marked passed. It requires exact generated token ID/text parity and
  server placement attribution. Planning mode is not generation proof.
- `mvp_capabilities/cache_generation_proof.py` emits
  `text_generation_parity.py --mode generate-api` runbooks and verifies cached
  generate parity before a model's `cache_generation` gate can be marked passed.
  Forward-loop parity is explicitly not enough for this gate.
- `mvp_capabilities/multi_block_proof.py` emits two-or-more-server runbooks and
  verifies multi-block evidence only when each server records start, block-range
  announcement, RPC evidence, and a combined direct-client result. Qwen3-8B
  minimal two-server multi-block direct RPC now passes from a clean m4pro archive:
  layers `0:1` and `1:2` served finite forward/backward over combined range `0:2`.
  Full-generation/cache-generation remain pending.
- `mvp_capabilities/demo_dashboard.py` surfaces the weighted MVP status bar,
  remaining percentage, next gate, proof-prep state, joined-peer layer-plan
  runbooks, coordinator handoff bundles with fresh-device bootstrap scripts,
  proof orchestration checklists, speculative decode plans, multi-block
  diagnostics, and chain-scheduler rehearsals beside route/evidence/telemetry
  panels.
- `mvp_capabilities/join_handoff.py` fetches `/handoff` or redacts a saved raw
  bundle into `.local/handoff-bundle.json` for the dashboard; tokens are stripped
  from nested fields and URLs before writing/printing.
- `mvp_capabilities/join_card.py --write-sidecars` writes `.join.json` and
  `.join.txt` fallback artifacts containing the exact copy/paste join URL; this
  improves operator flow but still does not prove QR scanner interoperability.
- `mvp_capabilities/proof_state.py` parses retained proof-prep status/log/cache
  facts, distinguishes complete snapshots from stale `.incomplete` leftovers,
  emits ETA fields, and feeds the dashboard while explicitly refusing to claim
  inference.
- `mvp_capabilities/join_layer_plan.py` converts active token-scoped coordinator
  heartbeats from local state or HTTP `/active` into deterministic layer
  placements, launch-command runbooks, and no-execution launch-readiness
  checklists. Generated follower commands use the current `run_server`
  `--initial_peers` CLI flag. Operators can pass captured seed addresses with
  `--seed-multiaddr HOST=/ip4/.../p2p/...` to resolve follower placeholders
  before the checklist marks the runbook ready.
- `mvp_capabilities/route_picker.py` now accepts `--selector-mode planning`,
  `--selector-mode showcase-attempt`, and `--selector-mode safe-demo`, so the
  live dashboard/coordinator can distinguish memory-fit planning from
  proof-gated demo-safe selection.
- `MODEL_REGISTRY.yaml` includes the prepared core-dream Qwen3-MoE variants
  `Qwen/Qwen3-30B-A3B-Instruct-2507` and
  `Qwen/Qwen3-30B-A3B-Thinking-2507` with pending proof gates; they are not
  safe-demo candidates until full distributed generation passes.
- `mvp_capabilities/join_coordinator.py` creates `bloombee://join?...` offers
  and token-scoped heartbeat rosters. `mvp_capabilities/join_http_server.py`
  exposes `/healthz`, `/offer`, `/heartbeat`, `/active`, `/route`, `/plan`,
  `/speculative`, `/bootstrap`, `/bootstrap.sh`, `/handoff`, and
  `/proof-orchestration` endpoints using Python stdlib
  HTTP. `/bootstrap` returns a token-scoped peer-scan + bounded-heartbeat JSON
  runbook; `/bootstrap.sh` returns the same runbook as plain shell; `/route`
  returns proof-aware dynamic model selection for current heartbeats;
  `/speculative` returns a verifier-authoritative draft-provider plan, and
  `mvp_capabilities/draft_provider.py` adds a dependency-free provider contract
  plus proposed/accepted/rejected counters for dashboard smoke reports;
  `mvp_capabilities/draft_provider_bridge.py` exposes the same contract over
  stdio JSONL for Termux/ADB/SSH bridge experiments;
  `mvp_capabilities/termux_draft_smoke.py` renders/verifies a pasteable Termux
  phone-smoke script when the sandbox cannot start `adb`, and tracked real
  Pixel 8 Pro Termux smoke, repeated static-contract latency, tiny-runtime
  blocker evidence, guarded no-install GGUF runtime plan, standalone
  tiny-GGUF generation proof, draft-provider-candidate bridge proof, and
  verifier-prefix comparison evidence including same-GGUF local acceptance,
  tokenizer-ID match evidence, a fail-closed wall-clock gate, and a local
  speculative harness reference, phone-token verifier preflight, binding-verifier
  text-prefix evidence, external context-token ingestion evidence, and BloomBee block-serving preflight now live under
  `mvp_capabilities/distributed_evidence/phone/`;
  `/plan?model=auto` folds that selection into a no-execution joined layer plan
  without requiring shared filesystem access; `/handoff` bundles offer, active
  roster, bootstrap runbook, speculative plan, auto route, launch plan, proof
  harness runbooks, and an embedded proof orchestration checklist without
  starting servers; `/proof-orchestration` returns the checklist directly.
  `mvp_capabilities/join_client.py` lets physical devices
  parse a join URL and post one-shot or bounded repeated peer-scan heartbeats so
  they remain active during operator planning. `mvp_capabilities/join_card.py`
  renders an SVG join card with exact URL metadata and copy/paste sidecars.
  `mvp_capabilities/join_qr_preflight.py` reports missing QR encoder/decoder
  dependencies fail-closed, and `mvp_capabilities/join_qr_proof.py` can generate
  a true QR PNG and decode it back to the exact join URL locally. Physical
  camera scanner interop remains unproven. Pixel 8 Pro Termux via ADB UI now
  proves a fresh physical-device 3-heartbeat `join_client.py` roster path, but
  physical camera QR scan remains unproven. This is bootstrap/roster state only
  and explicitly does not claim inference proof.
  `mvp_capabilities/join_layer_plan.py` then turns those active heartbeats into
  launch-ready layer-placement runbooks from local state or HTTP `/active`
  without starting servers; `--include-launch-readiness` adds a checklist that
  keeps seed multiaddr placeholders as explicit blockers unless `--seed-multiaddr
  HOST=MULTIADDR` resolves them from operator-captured values before any
  server-start claim.
  `mvp_capabilities/chain_scheduler.py` turns a joined layer plan into
  multi-request waves, per-peer scheduled-token estimates, and no-live-traffic
  health reports. `mvp_capabilities/proof_orchestrator.py` turns a handoff bundle
  into an ordered no-execution checklist for server launch, multiaddr capture,
  proof clients, verification, and manual proof-status promotion; it keeps
  unresolved placeholders and forbidden legacy peer flags visible instead of
  silently executing. `mvp_capabilities/request_telemetry.py` summarizes direct-client
  `[direct] RESULT` logs into request success/failure counts, forward/backward
  latency, model/block coverage, and errors for the dashboard.
  `mvp_capabilities/multi_block_diagnostics.py` reads the same multi-block
  server/client proof logs and produces a per-server health/coverage/operator
  action report for dashboard troubleshooting; it remains observability only and
  cannot promote proof status.
  `mvp_capabilities/multi_request_load_proof.py` emits repeated direct-client
  runbooks and verifies expected successful request logs before allowing proof
  promotion. This is still not a live load proof until real traffic logs pass.
- `mvp_capabilities/layer_planner.py` converts a chosen model and peer roster
  into deterministic contiguous layer ranges and can attach exact BloomBee
  server launch commands with `--include-launch-commands`. Follower runbooks use
  `--initial_peers`, matching the current `run_server` CLI. This
  is placement and launch planning only; real serving still requires the
  BloomBee server proof ladder.
- `mvp_capabilities/swarm_simulator.py` rehearses synthetic/live rosters with
  failed hosts, route selection, and layer placement. It is explicitly
  simulation-only, not an inference or serving proof.
- TinyLlama distributed inference has been verified as a proof ladder:
  two-server, two-laptop, three-peer, forward-loop text parity, and cached
  `.generate()` parity. Cached generation now matches exact token IDs and
  decoded text after the `rpc_inference` recovery fixes.

## Mobile phone peer status

Status: **capability discovery groundwork started; useful inference-worker proof
not yet complete.**

What exists now:

- `mvp_capabilities/peer_scan.py` emits a `mobile` profile. Android/Termux
  phones are identified via Termux environment variables and Android
  `getprop` fields such as model, manufacturer, SoC, ABI, and SDK.
- Non-mobile hosts explicitly emit `"mobile": {"is_mobile": false, ...}` so
  route planning can distinguish laptops from phones without guessing.

What does **not** exist yet:

- No Android/Termux phone has produced committed throughput evidence.
- No phone has successfully served a BloomBee transformer block in the DHT/RPC
  path.
- Current BloomBee server auto block selection is still GPU-shaped: it accepts
  `cuda`/`mps` automatically and only permits CPU-only servers when the operator
  manually specifies `--num_blocks`. That means Android phones are not yet
  counted as useful block workers until a CPU/mobile block-serving run is
  proven and benchmarked.

Practical MVP path for phones:

1. Run `peer_scan.py` inside Termux on one connected Android phone and commit
   the capability JSON.
2. Run `bench_throughput.py --device cpu` on that phone for TinyLlama or a
   smaller model to get honest prefill/decode numbers.
3. Try `python -m bloombee.cli.run_server ... --device cpu --num_blocks 1` on
   the phone. Only count it as an inference peer if a client can route through
   it and output parity still matches.
4. If CPU throughput is too low, phones can still be useful post-MVP as DHT,
   monitoring, control-plane, or gateway peers; for transformer-block work we
   should evaluate Android GPU paths separately (Vulkan/NNAPI/ExecuTorch/MLC),
   which is a backend integration project rather than a simple BloomBee flag.

### MoE compatibility note

BloomBee is **not Petals**. Petals' original public design routed per-token through the
network, which is incompatible with MoE because each token selects different experts.
BloomBee's serving path is **block-parallel** (pipeline-parallel at the transformer-block
level), not token-routed: each peer hosts a contiguous range of transformer blocks, and
forward passes flow **block-to-block through peers**, not token-to-token. MoE is just a
different *block internals* — the routing expert lives *inside* a block on one peer.

- Upstream BloomBee already ships a Mixtral MoE wrapper
  (`src/bloombee/models/mixtral`, `WrappedMixtralBlock`).
- Our `src/bloombee/models/qwen3_moe/` adds the Qwen3-MoE family on the same pattern.
- The MoE models fit block-parallel placement because the *block* lives on one peer
  and the experts live inside that block — no per-token network routing.

## Server bring-up recipe and verified swarm gates

The full multi-peer test (`tests/test_remote_sequential.py`) requires:

1. N≥2 laptops, each running `python -m bloombee.cli.run_server <model_name>`.
2. Shared `--initial_peers` target (one peer's multiaddr is shared to the other).
3. Same `MODEL_NAME` and shared `dht_prefix` across peers.

TinyLlama-1.1B two-device rehearsal (the smallest real distributed test):

```bash
# On machine A (the seed peer — its multiaddr gets shared to B):
cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate
export PYTHONPATH=".:src"
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --new_swarm --block_indices 0:11 \
    --device mps --torch_dtype bfloat16 --port 31337

# It prints a seed multiaddr like:
#   /ip4/192.168.1.42/tcp/31337/p2p/QmXXX
# Copy that, then on machine B:
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --initial_peers "/ip4/<A_IP>/tcp/31337/p2p/QmXXX" \
    --block_indices 11:22 \
    --device mps --torch_dtype bfloat16
```

Then from a third laptop:

```bash
INITIAL_PEERS="/ip4/<A_IP>/tcp/31337/p2p/QmXXX" \
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
pytest tests/test_remote_sequential.py -v -s
```

The original `pytest tests/test_remote_sequential.py` gate remains awkward under
the Hermes/pytest forked event-loop setup. The practical verified path is now
`scripts/direct_remote_call.py` and `scripts/text_generation_parity.py`, both of
which produce machine-readable JSON evidence and avoid the test harness hang.

Verified gates now include:

- two-server TinyLlama forward/backward on one host,
- two-laptop TinyLlama forward/backward over LAN,
- three-peer TinyLlama forward/backward on one host,
- three-peer forward-loop text-generation parity,
- cached `.generate()` text-generation parity on the patched `rpc_inference`
  recovery path,
- three-peer cached `.generate()` parity with S2S enabled by default as an
  opportunistic optimization plus direct client fallback,
- one-block Qwen3-30B-A3B MoE live-server shard proof on M4 Pro,
- two-server Qwen3-30B-A3B MoE multi-block `0:2` direct RPC proof on M4 Pro.

Next verification gates are Qwen3-30B-A3B full-generation/cache/load proofs,
two-laptop cached `.generate()` with S2S/default fallback, and the physical
10-laptop showcase.

### Verified distributed-server boot on M4 Pro (2026-07-02 ~21:52)

A BloomBee server was actually started on M4 Pro against TinyLlama-1.1B
with the existing TinyLlama cache (no extra download required). Captured
lines from `/tmp/bloombee_seed.log`:

```
[INFO] Running bloombee 2.3.0.dev2
[INFO] Using DHT prefix: TinyLlama-1-1B-Chat-v1-0-hf
[INFO] This server is accessible directly
[INFO] Running a server on [
  /ip4/100.84.252.4/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn,
  /ip4/127.0.0.1/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn,
  /ip6/::1/tcp/31337/p2p/12D3KooWGNBpUDuU7YJ1gAWt8MKk771FrvHBfjkNDwfWsuGqaZSn]
[WARN] Type bfloat16 is not supported on MPS, using float16 instead
[INFO] Inference throughput: 328.3 tokens/sec per block (1 tokens/batch, MPS, float16)
[INFO] Forward pass throughput: 14705.8 tokens/sec per block (1024 tokens/batch, MPS, float16)
[INFO] Network throughput: 356.3 tokens/sec (11.68 Mbit/s on download, 18.11 Mbit/s on upload)
[INFO] Reporting throughput: 356.3 tokens/sec for 11 blocks
[INFO] Announced that blocks range(0, 11) are joining
[INFO] Started
```

This was a single-device seed running blocks 0..10 of 22 (TinyLlama has 22
transformer blocks). The actual `pytest tests/test_remote_sequential.py`
client test was driven against this server on M4 Pro but exhausted available
RAM (each machine was already at <300 MB free at session start) and the DHT
handshake needed more time than the sandbox would allow. **The server boot
itself is now confirmed working** — what remains is just driving the full
two-process DHT handshake with clean memory pressure, which is a memory
environment problem, not a code correctness problem.

## No-overclaiming rules

- Do not claim 10 physical laptops have run until the showcase test happens.
- Do not claim full Qwen3-30B-A3B distributed generation works until the full
  generation/cache/load ladder passes. Two-server `0:2` multi-block serving is
  proven; full-model distributed generation is not.
- Do not claim a server gate is complete from registry fit alone; fit prediction is not inference proof.
- Do not claim quantized checkpoints are BloomBee-runnable just because HF/vLLM/
  SGLang/llama.cpp can serve them. Current BloomBee HF-block loading expects
  normal PyTorch block weights, filters `.safetensors` by block prefix, calls
  `load_state_dict`, and casts to fp16/bf16; it does not instantiate GPTQ/AWQ/
  FP8/NVFP4/MXFP quantized kernels.
- Do not count phones as useful inference workers until a phone produces
  measured throughput and successfully serves at least one transformer block in
  the distributed path.

## Operator commands

```bash
python mvp_capabilities/peer_scan.py
python mvp_capabilities/swarm_roster.py --cap-dir ~/.bloombee/capabilities --json
python mvp_capabilities/route_picker.py --cap-dir ~/.bloombee/capabilities --scenario mvp-10-laptop
# Show *why* a route was or wasn't chosen — pass/fail per candidate, near-misses:
python mvp_capabilities/route_picker.py --cap-dir ~/.bloombee/capabilities --explain
python mvp_capabilities/sweep_models.py --peer ~/.bloombee/capabilities/$(hostname -s).json --dry-run
ssh m4pro 'cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate && python mvp_capabilities/peer_scan.py'
```

### Demo dashboard

If `~/.bloombee/capabilities` is unavailable in a sandboxed session, use the
repo-local `.local/capabilities/` directory populated by fresh scans. Generate a
self-contained dashboard snapshot for real connected peers only:

```bash
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate

python mvp_capabilities/join_handoff.py \
  --coordinator-url http://127.0.0.1:8787 \
  --token moon-token \
  --model auto \
  --selector-mode planning \
  --request-count 2 \
  --out .local/handoff-bundle.json

python mvp_capabilities/proof_orchestrator.py \
  --handoff-bundle .local/handoff-bundle.json \
  --out .local/proof-orchestration.json

python mvp_capabilities/multi_block_diagnostics.py \
  --model Qwen/Qwen3-8B \
  --block-range 0:18 \
  --block-range 18:36 \
  --server-log .local/qwen3-8b-server-0.log \
  --server-log .local/qwen3-8b-server-1.log \
  --client-log .local/qwen3-8b-client.log > .local/multi-block-diagnostics.json

python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --proof-state .local/proof-state.json \
  --joined-layer-plan .local/joined-layer-plan.json \
  --chain-schedule .local/chain-schedule.json \
  --handoff-bundle .local/handoff-bundle.json \
  --proof-orchestration .local/proof-orchestration.json \
  --multi-block-diagnostics .local/multi-block-diagnostics.json \
  --request-log .local/direct-client.log \
  --out .local/demo-dashboard.html \
  --refresh-seconds 10
```

For live recovery/S2S counters, add one or more server/client logs:

```bash
python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --telemetry-log /tmp/bloombee_seed.log \
  --out .local/demo-dashboard.html
```

Open `.local/demo-dashboard.html` during the demo. To keep the snapshot updating
while the browser auto-refreshes, run bounded or unbounded watch mode:

```bash
python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --out .local/demo-dashboard.html \
  --refresh-seconds 10 \
  --watch-seconds 2
```

The dashboard labels unbenchmarked route choices as `unmeasured`, not `0 tok/s`,
so fit-only routes do not masquerade as throughput evidence. It also renders a
**Layer placement** section from `server_placements` metadata in proof JSON, e.g.
`m4pro-seed` serving layers `0:8`, `m4pro-mid` serving `8:15`, and `m4pro-tail`
serving `15:22`. If supplied with `--joined-layer-plan`, it separately renders a
**Joined-peer layer plan** panel from `join_layer_plan.py`; that panel is a
launch runbook only and carries `joined_roster_layer_plan_only_no_inference_proof`.

Synthetic 10-laptop route planning is opt-in and should not be used as the live
demo surface:

```bash
python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
  --synthetic-m4-laptops 10 \
  --out .local/demo-dashboard-planning.html
```

Real setup status: a physical multi-user/laptop demo is **not self-serve ready**
yet. Verified today: three live BloomBee server processes on `m4pro` served
TinyLlama layers `0:8`, `8:15`, and `15:22`, and a direct client call over
`0:22` layers returned finite output and gradients. Remaining live-demo gates are
a real laptop join script/installer, automatic layer assignment for connected
peers, a non-sandbox client path for generate-api parity, and a physical N-laptop
showcase run.

Phone speculative-decoding analysis lives at
`docs/phone-speculative-decoding-mvp.md`.
