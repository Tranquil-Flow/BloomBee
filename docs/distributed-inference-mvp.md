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
- Measured local TinyLlama benchmark: prefill ~610.7 tok/s, decode ~7.7 tok/s.
- Direct SSH to M4 Pro works and reports 48GB memory.

## No-overclaiming rules

- Do not claim 10 physical laptops have run until the showcase test happens.
- Do not claim Qwen3 MoE serving works until a BloomBee MoE wrapper is implemented and tested.
- Do not claim a server gate is complete from registry fit alone; fit prediction is not inference proof.

## Operator commands

```bash
python mvp_capabilities/peer_scan.py
python mvp_capabilities/swarm_roster.py --cap-dir ~/.bloombee/capabilities --json
python mvp_capabilities/route_picker.py --cap-dir ~/.bloombee/capabilities --scenario mvp-10-laptop
python mvp_capabilities/sweep_models.py --peer ~/.bloombee/capabilities/$(hostname -s).json --dry-run
ssh m4pro 'cd ~/Projects/distributed-inference-mvp && source .venv/bin/activate && python mvp_capabilities/peer_scan.py'
```
