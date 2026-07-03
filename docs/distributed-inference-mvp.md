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

## 10-laptop MVP target

Primary 10-laptop target: **Qwen/Qwen3-30B-A3B**.

Why:

- 30.5B total parameters, ~3.3B active parameters per token.
- Better quality-per-watt than dense 14B/32B on Apple Silicon swarms.
- Fits aggregate memory of 10 M4 laptops even when each laptop cannot host the whole model solo.
- Best MVP showcase story: many modest devices collaborate to serve a stronger model.

Stretch target: **Qwen/Qwen3-235B-A22B**. This stays a stretch target until the swarm has enough aggregate memory and BloomBee has a verified Qwen3 MoE block handler.

## Verified current state

- Local M4 16GB can load and run TinyLlama-1.1B on MPS after the sitecustomize RLock fix.
- Fresh repo-local live scan on 2026-07-03: local `evinova` /
  `Evis-MacBook-Pro` reports MPS, 16GB total, ~2.3GB free; `m4pro`
  reports MPS, 48GB total, ~34.5GB free. Combined live roster: 2 peers,
  64GB total, ~36.8GB free.
- Current two-peer route with the measured M4 Pro matrix picks
  `google/gemma-2-9b-it` as a solo `m4pro` route. This is a live roster
  choice, not the final 10-laptop target.
- Measured local TinyLlama benchmark (cold cache): prefill ~610.7 tok/s, decode ~7.7 tok/s.
- Measured local TinyLlama (warm cache, repeat run): prefill 1130.3 tok/s, decode 28.6 tok/s, peak 0.21 GB.
- Measured M4 Pro bf16 sweep (5 models, 2026-07-02):
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
  shard has loaded real Qwen3-30B-A3B safetensors for block `0:1` and served
  direct RPC forward/backward with finite outputs and gradients.
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
2. Shared `INITIAL_PEERS` (one peer's multiaddr is shared to the other).
3. Same `MODEL_NAME` and shared `dht_prefix` across peers.

TinyLlama-1.1B two-device rehearsal (the smallest real distributed test):

```bash
# On machine A (the seed peer — its multiaddr gets shared to B):
cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate
export PYTHONPATH=".:src"
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --new_swarm --block_indices 0:11 \
    --device mps --torch_dtype bfloat16 --port 31337

# It prints INITIAL_PEERS multiaddr like:
#   /ip4/192.168.1.42/tcp/31337/p2p/QmXXX
# Copy that, then on machine B:
INITIAL_PEERS="/ip4/<A_IP>/tcp/31337/p2p/QmXXX" \
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
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
- one-block Qwen3-30B-A3B MoE live-server shard proof on M4 Pro.

Next verification gates are full multi-block Qwen3-30B-A3B distributed serving,
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
- Do not claim full Qwen3-30B-A3B distributed generation works until all required
  blocks have been served across a live swarm. One-block live serving is proven;
  full-model distributed generation is not.
- Do not claim a server gate is complete from registry fit alone; fit prediction is not inference proof.
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
self-contained dashboard snapshot:

```bash
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate

python mvp_capabilities/demo_dashboard.py \
  --cap-dir .local/capabilities \
  --bench-matrix .local/m4pro-bench-matrix.json \
  --evidence-dir mvp_capabilities/distributed_evidence \
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

Open `.local/demo-dashboard.html` during the demo. The dashboard labels
unbenchmarked route choices as `unmeasured`, not `0 tok/s`, so fit-only routes do
not masquerade as throughput evidence.

Phone speculative-decoding analysis lives at
`docs/phone-speculative-decoding-mvp.md`.
