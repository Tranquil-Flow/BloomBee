# Text Generation Parity: TinyLlama-1.1B local vs 3-peer distributed

**Date:** 2026-07-02 / 2026-07-03 CEST  
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Mode:** `forward-loop` greedy decode (full-prefix recompute each token)  
**Status:** ✅ **PASS — exact generated token IDs and decoded text match**

This proof verifies that using BloomBee distributed inference does **not** change greedy text output for the tested prompt/checkpoint. The local baseline is a full Hugging Face TinyLlama model on M4 Pro/MPS fp16. The distributed path uses a BloomBee client with client-side embeddings + final norm + LM head, and three remote BloomBee server processes holding the transformer blocks.

---

## Swarm layout

| Peer | Multiaddr | Blocks |
|---|---|---|
| peer 1 | `/ip4/192.168.178.37/tcp/31337/p2p/12D3KooWBx9vaNYWbH3Evh8kLgU7PKEaK1jDx2qnx8vB138Z8GVz` | `0:8` |
| peer 2 | `/ip4/192.168.178.37/tcp/31338/p2p/12D3KooWCjZQ2SNtnZchWGVeqcuhWq3e59C4FnZk2pqq6y2rD45v` | `8:15` |
| peer 3 | `/ip4/192.168.178.37/tcp/31339/p2p/12D3KooWPeUfUzBQPgxRVUu9QDRaw7SaRYLDdem549RihEHKP85h` | `15:22` |

Total: 22 transformer blocks, split 8 / 7 / 7 across 3 separate BloomBee server processes.

---

## Command

```bash
python scripts/text_generation_parity.py \
  --server-maddr '/ip4/192.168.178.37/tcp/31337/p2p/12D3KooWBx9vaNYWbH3Evh8kLgU7PKEaK1jDx2qnx8vB138Z8GVz' \
  --server-maddr '/ip4/192.168.178.37/tcp/31338/p2p/12D3KooWCjZQ2SNtnZchWGVeqcuhWq3e59C4FnZk2pqq6y2rD45v' \
  --server-maddr '/ip4/192.168.178.37/tcp/31339/p2p/12D3KooWPeUfUzBQPgxRVUu9QDRaw7SaRYLDdem549RihEHKP85h' \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt 'The capital of France is' \
  --max-new-tokens 6 \
  --reference-device mps \
  --reference-dtype float16 \
  --distributed-dtype float16 \
  --mode forward-loop \
  --out mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_TINYLLAMA_2026-07-02.json
```

---

## Result summary

Prompt:

```text
The capital of France is
```

Input IDs:

```json
[1, 450, 7483, 310, 3444, 338]
```

Local full-model generated IDs:

```json
[1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889]
```

Distributed generated IDs:

```json
[1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889]
```

Decoded text (both):

```text
The capital of France is Paris.

2.
```

Metrics:

| Metric | Value |
|---|---:|
| `ok` | `true` |
| `generated_ids_match` | `true` |
| `generated_text_match` | `true` |
| `next_token_match` | `true` |
| first next token | `3681` (`Paris`) |
| logits max abs diff | `0.016242504119873047` |
| logits mean abs diff | `0.0030047385953366756` |
| local full-model decode time | `1.80s` |
| distributed forward-loop decode time | `39.18s` |

The small logit deltas are expected fp16/MPS numerics; top-token and full greedy output remain identical.

Full machine-readable evidence: `TEXT_GEN_PARITY_TINYLLAMA_2026-07-02.json`.

---

## Server-side evidence

All three peers processed the growing prefixes for the same client peer (`remote_peer=...xrQQWF`). This proves text-generation parity used the actual distributed transformer block chain, not a local shortcut.

### Peer 1 — blocks `0:8`

```text
Jul 03 00:57:45.290 [INFO] rpc_forward(blocks=0:8, remote_peer=...xrQQWF)
Jul 03 00:57:45.375 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 83.85ms | Backends: 8 | Output Shape: torch.Size([1, 6, 2048])
Jul 03 00:58:16.345 [INFO] rpc_forward(blocks=0:8, remote_peer=...xrQQWF)
Jul 03 00:58:18.972 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 2623.94ms | Backends: 8 | Output Shape: torch.Size([1, 11, 2048])
```

### Peer 2 — blocks `8:15`

```text
Jul 03 00:57:45.383 [INFO] rpc_forward(blocks=8:15, remote_peer=...xrQQWF)
Jul 03 00:57:45.453 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 68.28ms | Backends: 7 | Output Shape: torch.Size([1, 6, 2048])
Jul 03 00:58:18.980 [INFO] rpc_forward(blocks=8:15, remote_peer=...xrQQWF)
Jul 03 00:58:21.368 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 2385.78ms | Backends: 7 | Output Shape: torch.Size([1, 11, 2048])
```

### Peer 3 — blocks `15:22`

```text
Jul 03 00:57:45.461 [INFO] rpc_forward(blocks=15:22, remote_peer=...xrQQWF)
Jul 03 00:57:45.539 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 75.29ms | Backends: 7 | Output Shape: torch.Size([1, 6, 2048])
Jul 03 00:58:21.375 [INFO] rpc_forward(blocks=15:22, remote_peer=...xrQQWF)
Jul 03 00:58:24.018 [INFO] [CROSS_GPU_TRANSFER_LATENCY] Total: 2640.11ms | Backends: 7 | Output Shape: torch.Size([1, 11, 2048])
```

---

## Important note: cached `.generate()` path

This document originally recorded a cached `RemoteGenerationMixin.generate()` failure:

```text
P2PHandlerError: Failed to call handler `TransformerConnectionHandler.rpc_inference` ...
BH mismatch: src span 256 > dst 32, cannot write
```

That issue is now fixed in commit `2117eca` (`fix(distributed): repair cached generate recovery`). The root causes were dropped full-batch KV metadata and stale cache-position accounting after recovery. See `CACHED_GENERATE_API_TINYLLAMA_2026-07-03.md` and `TEXT_GEN_PARITY_GENERATE_API_6TOK_TINYLLAMA_2026-07-03.json` for the updated cached `.generate()` evidence.

Current status: `--mode forward-loop` remains a useful math-isolation proof, and `--mode generate-api` now also passes exact TinyLlama token/text parity for the tested 6-token prompt on the patched path.
