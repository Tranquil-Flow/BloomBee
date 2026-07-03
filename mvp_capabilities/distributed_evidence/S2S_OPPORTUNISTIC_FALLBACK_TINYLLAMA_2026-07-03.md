# S2S hardening: opportunistic push with direct fallback

**Date:** 2026-07-03 CEST  
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Mode:** cached `generate-api` / `rpc_inference`  
**Topology:** 3 peers on m4pro, split `0:8`, `8:15`, `15:22`  
**Key change:** `BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE` is no longer enabled by default.

## Root cause

The correctness path was already proven by the no-S2S cached parity test, but the remaining hardening failure came from the default **push-only downstream decode** mode:

```text
client -> upstream server
upstream server -> downstream server via rpc_push
client waits for downstream pushed response
```

That mode treats server-to-server `rpc_push` as the only delivery path after warmup. If `rpc_push` fails, stalls, or races with MPS placeholder recovery, the client can hang or repeatedly rebuild sessions even though the client already has the upstream hidden states needed to call the downstream stage directly.

The robust policy is:

```text
S2S push = opportunistic optimization
client direct downstream request = correctness fallback/default
push-only wait = explicit opt-in only
```

## TDD gate

Added:

```text
tests/test_client_config_defaults.py
```

RED failure before the fix:

```text
pytest tests/test_client_config_defaults.py::test_push_only_downstream_decode_defaults_to_direct_fallback -q
F
E       AssertionError: assert True is False
```

GREEN after the fix:

```text
pytest tests/test_client_config_defaults.py -q
3 passed
```

Behavior now:

- env unset -> `ClientConfig().push_only_downstream_decode is False`
- `BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE=1|true|yes|on` -> opt into push-only mode
- `BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE=0|false|no|off` -> explicit direct-fallback mode

## Live proofs

First, the live proof used the old explicit env switch to validate the intended new default behavior before changing config:

```bash
env PYTHONUNBUFFERED=1 PYTHONPATH=.:src \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  BLOOMBEE_MAX_RETRIES=2 \
  BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE=0 \
  python -u scripts/text_generation_parity.py \
    --server-maddr "/ip4/192.168.178.37/tcp/31337/p2p/12D3KooWFPf53Btcu4HTs9BcrC8z5zkZfJg6RrXerhmbsqnoELRe" \
    --server-maddr "/ip4/192.168.178.37/tcp/31338/p2p/12D3KooWJ2EbCoSH6W2XtyvBzSrotx6VAMHsi6KTkvXjNhT1ew5W" \
    --server-maddr "/ip4/192.168.178.37/tcp/31339/p2p/12D3KooWGVt6YxN91vR4cpTcP43DGyeHVHfBUUivbwVd3fjSmLct" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "The capital of France is" \
    --max-new-tokens 6 \
    --reference-device mps \
    --reference-dtype float16 \
    --distributed-dtype float16 \
    --mode generate-api \
    --out mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_OPPORTUNISTIC_TINYLLAMA_2026-07-03.json
```

Then the patched default was synced to m4pro and run with `BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE` unset:

```bash
env -u BLOOMBEE_PUSH_ONLY_DOWNSTREAM_DECODE \
  PYTHONUNBUFFERED=1 PYTHONPATH=.:src \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  BLOOMBEE_MAX_RETRIES=2 \
  python -u scripts/text_generation_parity.py \
    --server-maddr "/ip4/192.168.178.37/tcp/31337/p2p/12D3KooWFPf53Btcu4HTs9BcrC8z5zkZfJg6RrXerhmbsqnoELRe" \
    --server-maddr "/ip4/192.168.178.37/tcp/31338/p2p/12D3KooWJ2EbCoSH6W2XtyvBzSrotx6VAMHsi6KTkvXjNhT1ew5W" \
    --server-maddr "/ip4/192.168.178.37/tcp/31339/p2p/12D3KooWGVt6YxN91vR4cpTcP43DGyeHVHfBUUivbwVd3fjSmLct" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "The capital of France is" \
    --max-new-tokens 6 \
    --reference-device mps \
    --reference-dtype float16 \
    --distributed-dtype float16 \
    --mode generate-api \
    --out mvp_capabilities/distributed_evidence/TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA_2026-07-03.json
```

Results:

```text
TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_OPPORTUNISTIC_TINYLLAMA_2026-07-03.json True True True True True 5.511
TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA_2026-07-03.json True True True True True 4.313
```

Default-mode JSON summary:

```json
{
  "ok": true,
  "server_to_server": true,
  "generated_ids_match": true,
  "generated_text_match": true,
  "next_token_match": true,
  "distributed_seconds": 4.313472032546997
}
```

Exact output:

```text
[1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 13, 29906, 29889]
The capital of France is Paris.

2.
```

## Recovery evidence

The live run still hit the MPS placeholder recovery path repeatedly:

```text
Placeholder storage has not been allocated on MPS device!
Final stage returned full-history hidden states after session recovery; slicing seq_len from 7 to current_step_tokens=1
Final stage returned full-history hidden states after session recovery; slicing seq_len from 8 to current_step_tokens=1
Final stage returned full-history hidden states after session recovery; slicing seq_len from 9 to current_step_tokens=1
Final stage returned full-history hidden states after session recovery; slicing seq_len from 10 to current_step_tokens=1
Final stage returned full-history hidden states after session recovery; slicing seq_len from 11 to current_step_tokens=1
```

But with direct fallback enabled, the cached generation completed and matched exactly. This is the important hardening result: recovery noise no longer invalidates the user-facing output in the tested path.

Compact log counts from the three server logs:

```text
/tmp/f_s2s_p1.log: rpc_push=17, Placeholder storage=35, Maximum length=0
/tmp/f_s2s_p2.log: rpc_push=17, Placeholder storage=35, Maximum length=0
/tmp/f_s2s_p3.log: rpc_push=12, Placeholder storage=35, Maximum length=0
```

## Code change

Changed:

```text
src/bloombee/client/config.py
```

Before:

```python
DEFAULT_PUSH_ONLY_DOWNSTREAM_DECODE = env not in {"0", "false", "no", "off"} if env else True
```

After:

```python
DEFAULT_PUSH_ONLY_DOWNSTREAM_DECODE = env in {"1", "true", "yes", "on"} if env else False
```

This preserves the push-only performance experiment as an explicit opt-in while making the correctness-preserving fallback path the default.

## Remaining F work

This change makes S2S push opportunistic by default, but does **not** claim push-only S2S itself is robust. Remaining hardening:

1. Add clearer client/server telemetry taxonomy:
   - push scheduled
   - push acked
   - push failed
   - direct fallback used
   - late push skipped
   - recovery session rebuilt
2. Fix or suppress macOS/MPS placeholder recovery at its source.
3. Make S2S push failures surface as structured counters instead of noisy tracebacks.
4. Add a short timeout/direct-fallback path if push-only mode remains available for performance experiments.

Safe wording now:

> Default cached distributed generation uses S2S push opportunistically while preserving client-direct downstream delivery as the correctness fallback.

Still unsafe:

> Push-only S2S decode is robust.
