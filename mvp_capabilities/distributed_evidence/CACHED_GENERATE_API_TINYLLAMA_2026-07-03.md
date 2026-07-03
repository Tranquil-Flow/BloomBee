# Cached generate-api parity: TinyLlama-1.1B

**Date:** 2026-07-03 CEST  
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Mode:** BloomBee cached `.generate()` / `rpc_inference`  
**Server:** one patched BloomBee server on m4pro, blocks `0:22`, MPS fp16  
**Client:** m4pro, same repo checkout, patched client/server code

## Bug reproduced

Cached text generation previously failed before producing output:

```text
BH mismatch: src span 256 > dst 32, cannot write
```

That happened during prefill with logical batch size 1 and prompt length > 1. The server-side full-batch `rpc_inference` path dropped `full_batch_size` / `micro_batch_size` metadata, so KV-cache write code fell back to KV-head inference and interpreted a prompt-prefill tensor as multiple batch rows.

After the metadata fix, multi-token cached generation completed but initially produced wrong output:

```text
reference:   The capital of France is Paris.\n\n2.
distributed: The capital of France is Paris.....
```

That exposed two recovery/accounting bugs after an MPS transient:

1. `DistributedLlamaModel.forward()` updated `RemotePastKeyValues` from the sliced output tensor length. After a recovered decode step, the tensor may be `[B, 1, H]` even though the remote session has cached 7+ tokens.
2. `_ServerInferenceSession.step()` recreated failed server sessions by sending full history, but advanced the replacement server session by only the current-token count. The next decode step then used `prefix=1` repeatedly.

## Fixes

- `src/bloombee/server/block_functions.py`
  - Added `_fullbatch_metadata_kwargs(...)`.
  - Propagates `batch_offset`, `full_batch_size`, and `micro_batch_size` into non-microbatch full-batch `InferenceMetadata` builders.

- `src/bloombee/models/llama/model.py`
  - Added `_remote_seen_tokens_after_forward(...)`.
  - `RemotePastKeyValues` now advertises the active remote session position when available, not the sliced local output length.

- `src/bloombee/client/inference_session.py`
  - Added `_server_session_tokens_to_advance(...)`.
  - Replacement server sessions now advance by the full history length they actually sent during recovery.

## Regression tests

Local M4:

```text
pytest tests/test_rpc_inference_metadata.py tests/test_remote_generation_cache_position.py -q
7 passed, 5 warnings in 4.47s
```

M4 Pro:

```text
pytest tests/test_rpc_inference_metadata.py tests/test_remote_generation_cache_position.py -q
7 passed, 5 warnings in 1.96s
```

Syntax:

```text
python -m py_compile \
  src/bloombee/client/inference_session.py \
  src/bloombee/server/block_functions.py \
  src/bloombee/models/llama/model.py \
  scripts/text_generation_parity.py
py_compile_ok
```

## Live parity proof

Command shape:

```bash
python scripts/text_generation_parity.py \
  --server-maddr '/ip4/192.168.178.37/tcp/31337/p2p/12D3KooWHEt7jcqo36iv4YuvzEpv8Yv9vTPJjdqgA31VGFBRSyvQ' \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt 'The capital of France is' \
  --max-new-tokens 6 \
  --reference-device mps --reference-dtype float16 \
  --distributed-dtype float16 --mode generate-api \
  --out mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_6TOK_TINYLLAMA_2026-07-03.json
```

Result:

```json
{
  "ok": true,
  "generated_ids_match": true,
  "generated_text_match": true,
  "next_token_match": true,
  "reference_ids": [1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889],
  "distributed_ids": [1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889],
  "reference_text": "The capital of France is Paris.\n\n2.",
  "distributed_text": "The capital of France is Paris.\n\n2.",
  "distributed_seconds": 45.12113308906555,
  "logits_max_abs_diff": 0.016242504119873047,
  "logits_mean_abs_diff": 0.0030047385953366756
}
```

Server recovery evidence shows prefixes now advance correctly after each MPS placeholder recovery:

```text
prefix=6
prefix=7
prefix=8
prefix=9
prefix=10
```

Before the client recovery fix, those retries stayed at `prefix=1`, causing repeated `.` tokens.

## Caveat

The MPS server still emits transient `Placeholder storage has not been allocated on MPS device!` on decode attempts and then recovers by rebuilding full history. This fix makes recovery semantically correct. Post-MVP performance work should remove the transient MPS placeholder path so cached generation avoids repeated rebuilds and runs much faster.
