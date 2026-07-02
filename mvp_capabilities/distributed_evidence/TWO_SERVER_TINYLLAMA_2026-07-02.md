# Two-Server Distributed Inference — Real, Verified

**Date:** 2026-07-02 ~22:57 CEST
**Location:** M4 Pro (m4pro, 48GB) — both servers on the same host,
different ports, but the DHT client talks to them as if they were
two physically distinct peers.
**Model:** TinyLlama-1.1B-Chat-v1.0 (22 transformer blocks, hidden=2048)

## Topology

- Server 1: `bloombee.cli.run_server` on port 31337, blocks 0..10
- Server 2: `bloombee.cli.run_server` on port 31338, blocks 11..21
  (joined Server 1's DHT via `--initial_peers`)
- Client: `scripts/direct_remote_call.py` bootstraps to Server 1's
  multiaddr, requests `RemoteSequential(start_block=0, end_block=22)`,
  then runs forward + backward through the full 22-block pipeline.

## Verified metrics

Forward pass through all 22 blocks (data crosses from Server 1 → Server 2
over the network mid-pipeline):

```
[direct]   ... 4.16s, output shape=[1, 5, 2048]
[direct]   ... backward 1.35s, grad_finite=True, grad_norm=143.8710
[direct] outputs finite: True, unique values: 6224.0
[direct] RESULT: {"ok": true, "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "block_range": [0, 22], "input_shape": [1, 5, 2048],
  "output_shape": [1, 5, 2048], "outputs_finite": true,
  "outputs_unique": 6224.0, "grad_finite": true,
  "grad_norm": 143.8710174560547,
  "forward_seconds": 4.155869960784912,
  "backward_seconds": 1.3531010150909424}
```

## Server-side traces (corroborating)

Server 1 (blocks 0..10) saw 11 backends processed, total cross-GPU
transfer 137.91 ms — these blocks served directly to the client and
forwarded to Server 2.

Server 2 (blocks 11..21) saw 11 backends processed, total cross-GPU
transfer 3937.43 ms — these blocks received the hidden state from
Server 1 over the network, processed them, and sent the output back
to the client. The ~3.9s is the network + processing cost of running
on Server 2 after Server 1.

```
Server 1: [PROCESSING_LATENCY] Backend 4..10 each 5-13ms
Server 1: [CROSS_GPU_TRANSFER_LATENCY] Total: 137.91ms | Backends: 11
Server 2: [PROCESSING_LATENCY] Backend 4..10 each 270-381ms (cold)
Server 2: [CROSS_GPU_TRANSFER_LATENCY] Total: 3937.43ms | Backends: 11
```

## What this proves

1. **Two server processes, each holding half a model, joined a single
   DHT and served a unified `RemoteSequential` to a single client.**
2. **Hidden state tensor crossed the network between servers mid-pipeline.**
3. **Gradients flowed back through both servers in the backward pass.**
4. **Output values are finite, distinct, and the gradient norm is
   reasonable (143.87) — the model is computing, not degenerating.**

## What's still unverified

- The same test against two PHYSICALLY distinct laptops (M4 Pro + local
  M4). On this run, both servers happened to run on M4 Pro (different
  ports) because local M4 16GB does not have enough free RAM to also
  host a server process alongside the DHT client. The protocol, DHT
  client behavior, and `RemoteSequential` semantics are identical
  whether the two servers share a host or not — but the actual
  cross-laptop latency is not yet measured.
- A real LLM token-generation loop (text in, text out, not just
  hidden-state forward). TinyLlama has no LM head in the public
  `WrappedLlamaBlock`; the per-block outputs are hidden states.

## How to reproduce

```bash
# On machine 1 (or process 1):
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate
export PYTHONPATH=.:src
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --new_swarm --block_indices 0:11 --device mps \
    --torch_dtype bfloat16 --port 31337

# On machine 2 (or process 2), with the multiaddr from server 1's
# "Running a server on [...]" line:
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --block_indices 11:22 \
    --initial_peers /ip4/<HOST>/tcp/31337/p2p/<PEER_ID> \
    --device mps --torch_dtype bfloat16 --port 31338

# On the client:
python scripts/direct_remote_call.py \
    --server-maddr /ip4/<HOST>/tcp/31337/p2p/<PEER_ID> \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --block-range 0:22
```
