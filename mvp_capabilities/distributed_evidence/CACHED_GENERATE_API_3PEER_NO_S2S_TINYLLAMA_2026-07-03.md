# Cached generate-api parity: TinyLlama three-peer, no S2S push

**Date:** 2026-07-03 CEST  
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Mode:** cached `RemoteGenerationMixin.generate()` / `rpc_inference`  
**Topology:** three BloomBee server processes on m4pro, block split `0:8`, `8:15`, `15:22`  
**Transport:** client-orchestrated stages (`--no-server-to-server`, `server_to_server=false`)  
**Prompt:** `The capital of France is`  
**New tokens:** 6

## Result

Both same-host client and physical local-M4-to-m4pro client match local full-model generation exactly.

### Same-host m4pro client -> m4pro three-peer swarm

Evidence JSON:

```text
mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_3PEER_NO_S2S_TINYLLAMA_2026-07-03.json
```

Key fields:

```json
{
  "ok": true,
  "mode": "generate-api",
  "server_to_server": false,
  "max_new_tokens": 6,
  "generated_ids_match": true,
  "generated_text_match": true,
  "next_token_match": true,
  "distributed_seconds": 47.25556802749634
}
```

Exact generated IDs:

```text
[1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889]
```

Exact decoded text:

```text
The capital of France is Paris.

2.
```

### Physical local M4 client -> m4pro three-peer swarm

Evidence JSON:

```text
mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_LOCAL_TO_M4PRO_3PEER_NO_S2S_TINYLLAMA_2026-07-03.json
```

Key fields:

```json
{
  "ok": true,
  "mode": "generate-api",
  "server_to_server": false,
  "max_new_tokens": 6,
  "generated_ids_match": true,
  "generated_text_match": true,
  "next_token_match": true,
  "reference_seconds": 1.7451050281524658,
  "distributed_seconds": 6.503264904022217
}
```

Exact generated IDs:

```text
[1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889]
```

Exact decoded text:

```text
The capital of France is Paris.

2.
```

Server logs show the physical local client hitting all three m4pro server processes in the same run. The `remote_peer` suffix `...aXYQcB` is the local client for the 6-token run:

```text
# /tmp/bloombee_c3_p1.log
Jul 03 12:30:36.508 [INFO] rpc_forward(blocks=0:8, remote_peer=...aXYQcB)
Jul 03 12:30:37.027 [INFO] rpc_inference.open(blocks=0:8, remote_peer=...aXYQcB)
...

# /tmp/bloombee_c3_p2.log
Jul 03 12:30:36.606 [INFO] rpc_forward(blocks=8:15, remote_peer=...aXYQcB)
Jul 03 12:30:37.213 [INFO] rpc_inference.open(blocks=8:15, remote_peer=...aXYQcB)
...

# /tmp/bloombee_c3_p3.log
Jul 03 12:30:36.702 [INFO] rpc_forward(blocks=15:22, remote_peer=...aXYQcB)
Jul 03 12:30:37.383 [INFO] rpc_inference.open(blocks=15:22, remote_peer=...aXYQcB)
...
```

## Code changes needed to run this proof

- `scripts/text_generation_parity.py`
  - Added `--no-server-to-server` so the harness can isolate split-inference correctness from direct server-to-server `rpc_push` optimization.
  - Added sandbox-safe `MPFuture.reset_backend()` and `SharedBytes` heap fallback, mirroring `scripts/direct_remote_call.py`, so the harness can run from Hermes local shells.
- `src/bloombee/client/inference_session.py`
  - Added trimming of recovered full-history hidden states before already-warm downstream stages. Replacement downstream stages still receive full history so their KV caches can rebuild.
- `tests/test_remote_generation_cache_position.py`
  - Added regression tests for downstream trimming vs replacement-session full-history rebuild.

## Verification commands run

Local:

```bash
source .venv/bin/activate
pytest tests/test_remote_generation_cache_position.py tests/test_rpc_inference_metadata.py -q
# 10 passed
python -m py_compile scripts/text_generation_parity.py src/bloombee/client/inference_session.py
```

m4pro:

```bash
source .venv/bin/activate
pytest tests/test_remote_generation_cache_position.py tests/test_rpc_inference_metadata.py -q
# 10 passed
```

JSON evidence validation:

```bash
python - <<'PY'
import json
paths = [
  'mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_3PEER_NO_S2S_TINYLLAMA_2026-07-03.json',
  'mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_LOCAL_TO_M4PRO_1TOK_SMOKE_TINYLLAMA_2026-07-03.json',
  'mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_LOCAL_TO_M4PRO_3PEER_NO_S2S_TINYLLAMA_2026-07-03.json',
]
for p in paths:
    d = json.load(open(p))
    assert d['ok'] and d['generated_ids_match'] and d['generated_text_match'] and d['next_token_match'], p
PY
```

## Important caveat for F

This proof disables direct server-to-server push. That is intentional: it proves cached split inference does not degrade/change output when the client orchestrates the stages.

Default S2S push remains a hardening item. With `server_to_server=true`, the same three-peer cached run still timed out in the S2S `rpc_push`/recovery path after MPS placeholder recovery. This belongs under F: health/error taxonomy + S2S recovery hardening. It should not block the correctness claim above, but it should block any claim that S2S push is robust.
