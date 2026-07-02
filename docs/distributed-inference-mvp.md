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
- Qwen3-MoE block wrapper (`src/bloombee/models/qwen3_moe/`) is in place: auto-dispatches from real Qwen3-30B-A3B config (48 layers, hidden=2048, 128 experts @ 8/topk). Wrapper contract tests pass. Has not yet been exercised with full 30B safetensors in a live server.

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

## Server bring-up recipe (BLOCKED on real swarm)

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
    --new_swarm --block_indices 0 10 \
    --device mps --torch_dtype bfloat16 --port 31337

# It prints INITIAL_PEERS multiaddr like:
#   /ip4/192.168.1.42/tcp/31337/p2p/QmXXX
# Copy that, then on machine B:
INITIAL_PEERS="/ip4/<A_IP>/tcp/31337/p2p/QmXXX" \
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --block_indices 10 22 \
    --device mps --torch_dtype bfloat16
```

Then from a third laptop:

```bash
INITIAL_PEERS="/ip4/<A_IP>/tcp/31337/p2p/QmXXX" \
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
pytest tests/test_remote_sequential.py -v -s
```

Gating this end-to-end test is **the** outstanding MVP verification: I need 2+
laptops running the BloomBee server simultaneously with this environment
(`PYTHONPATH=.:src`, the sitecustomize RLock fix in `.venv/`). The M4 Pro is
reachable via `ssh m4pro` and its `.venv` shares the same fix; both Macs
together are the next gate before physical 10-laptop rehearsal.

## No-overclaiming rules

- Do not claim 10 physical laptops have run until the showcase test happens.
- Do not claim Qwen3 MoE serving works until a BloomBee MoE wrapper is implemented and tested.
- Do not claim a server gate is complete from registry fit alone; fit prediction is not inference proof.

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
