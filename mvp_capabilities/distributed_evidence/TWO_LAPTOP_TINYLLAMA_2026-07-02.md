# Two-Laptop Distributed Inference — Real, Verified

**Date:** 2026-07-02 ~23:55 CEST
**Topology:** M4 (local, 16GB) = client (DHT, RemoteSequential);
M4 Pro (`m4pro`, 48GB, LAN IP 192.168.178.37) = seed server
holding blocks 0..10 of TinyLlama-1.1B.
**Model:** TinyLlama-1.1B-Chat-v1.0

## How the client reached the server

- Tailscale IP `100.84.252.4` was the first thing tried but the DHT p2p daemon
  could not bootstrap through the Tailscale daemon permission boundary
  (`P2PDaemonError: failed to connect to bootstrap peers`).
- Solution: restart the seed server with `--public_ip 192.168.178.37` so
  the libp2p multiaddr it announces is the LAN IP. The client then
  bootstraps directly over the LAN.
- Result: DHT handshake 0.4s, forward 3.46s (cold, includes DHT warmup),
  backward 0.25s.

## Verified metrics (forward + backward across 2 physical laptops)

```
[direct] sending test_inputs shape=[1, 5, 2048]...
[direct]   ... 3.46s, output shape=[1, 5, 2048]
[direct]   ... backward 0.25s, grad_finite=True, grad_norm=104.5323
[direct] outputs finite: True, unique values: 6323.0
[direct] RESULT: {"ok": true, "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "block_range": [0, 11], "input_shape": [1, 5, 2048],
  "output_shape": [1, 5, 2048], "outputs_finite": true,
  "outputs_unique": 6323.0, "grad_finite": true,
  "grad_norm": 104.53234100341797,
  "forward_seconds": 3.4577107429504395,
  "backward_seconds": 0.24868178367614746}
```

## Server-side traces (corroborating)

```
[INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 3364.62ms | Backends: 11
[INFO] rpc_backward(blocks=0:11, remote_peer=...BnM7pw)
```

The `remote_peer=...BnM7pw` is the local M4's libp2p peer ID — proof the
RPC crossed the network between laptops.

## Fixes required during this run

1. `MPFuture.reset_backend()` in `scripts/direct_remote_call.py` —
   `sitecustomize.py` patches `_initialization_lock = None` for sandboxed
   environments. `conftest.py` calls `reset_backend()` to set it back to
   a real `threading.Lock`. Outside pytest (this script) we have to do
   it ourselves or `DHT()` crashes with
   `"NoneType object does not support the context manager protocol"`.
2. Use the LAN IP, not the Tailscale IP, for the client bootstrap. The
   sandboxed Tailscale daemon on local cannot route outbound p2p traffic.

## How to reproduce

```bash
# On m4pro (server):
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate
export PYTHONPATH=.:src
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --new_swarm --block_indices 0:11 --device mps \
    --torch_dtype bfloat16 --port 31337 \
    --public_ip <M4PRO_LAN_IP>

# On local M4 (client) — grab the multiaddr from the server's
# "Running a server on [...]" log line:
python scripts/direct_remote_call.py \
    --server-maddr /ip4/<M4PRO_LAN_IP>/tcp/31337/p2p/<PEER_ID> \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --block-range 0:11
```
