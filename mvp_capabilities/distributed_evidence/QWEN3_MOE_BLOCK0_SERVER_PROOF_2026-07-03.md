# Qwen3-30B-A3B MoE block-0 server proof

**Date:** 2026-07-03 CEST  
**Host:** `m4pro` / `evinova-self` / 48GB Apple Silicon  
**Clean proof tree:** `~/Projects/distributed-inference-mvp-proof` synced from local `git archive HEAD`  
**Model:** `Qwen/Qwen3-30B-A3B`  
**Block range served:** `0:1`  
**Device:** `mps`  
**Requested dtype:** `bfloat16`; server warned MPS does not support bf16 and used `float16`  
**Cache:** `~/.cache/huggingface/hub`, offline mode

## Why this proof matters

This is the first live BloomBee server proof that the Qwen3-MoE adapter can serve a real Qwen3-30B-A3B block from cached safetensors and answer remote RPC calls. Earlier gates only proved config dispatch and wrapper contract tests.

## Preflight evidence

Host identity:

```text
user=evinova-self
host=m4pro
mem=51539607552
project=yes
```

HF cache scan:

```text
TARGET Qwen/Qwen3-30B-A3B
hits 1
repo Qwen/Qwen3-30B-A3B model 61.1G
rev ad44e777bcd1 files 26 size 61.1G
incomplete_count 0
lock_count 0
```

Config/wrapper prescan from clean proof tree:

```text
[prescan] loading AutoConfig for Qwen/Qwen3-30B-A3B
[prescan] model_type='qwen3_moe'
[prescan] num_hidden_layers=48
[prescan] hidden_size=2048 num_experts=128
[prescan] num_experts_per_tok=8
[prescan] block_class=WrappedQwen3MoeBlock
[prescan] attn_class=Qwen3MoeAttention
[prescan] block_prefix=model.layers
[prescan] PASS
```

Remote clean test:

```text
pytest tests/test_qwen3_moe_block_parity.py -q
6 passed
```

## Fixes required before live serving

The live server attempt exposed and fixed three real integration gaps:

1. `tensor_parallel` is Linux-only / missing on macOS. Fixed by making `tensor_parallel` and `pynvml` imports optional in:
   - `src/bloombee/server/backend.py`
   - `src/bloombee/utils/convert_block.py`

2. `qwen3_moe` wrapper existed but was not imported by `bloombee.models`, so server startup failed with:

   ```text
   ValueError: BloomBee does not support model type qwen3_moe
   ```

   Fixed by importing `bloombee.models.qwen3_moe` in `src/bloombee/models/__init__.py`.

3. Qwen3-MoE blocks were not routed through the standard HF block loader. Server fell into the FlexGen/Llama path, then failed first with `skip_init_weights` and later with meta-vs-MPS tensors. Fixed by:
   - adding `WrappedQwen3MoeBlock` to `get_model_block` in `src/bloombee/server/block_utils.py`, and
   - adding `WrappedQwen3MoeBlock` to the HF-model tuple in `src/bloombee/server/from_pretrained.py`.

Commits:

```text
d2789a0 fix(distributed): make tensor parallel optional
2e272fc fix(distributed): register qwen3 moe models
da5e157 fix(distributed): load qwen3 moe blocks
7552e6e fix(distributed): treat qwen3 moe as hf block
```

## Server command

```bash
cd ~/Projects/distributed-inference-mvp-proof
source ~/Projects/distributed-inference-mvp/.venv/bin/activate
env PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BLOOMBEE_VERBOSE_KV_LOGS=1 \
  python -m bloombee.cli.run_server Qwen/Qwen3-30B-A3B \
    --new_swarm \
    --block_indices 0:1 \
    --device mps \
    --torch_dtype bfloat16 \
    --cache_dir ~/.cache/huggingface/hub \
    --port 31437 \
    --public_ip 192.168.178.37
```

Important: explicit `--cache_dir ~/.cache/huggingface/hub` was required. Without it, offline HF lookup used BloomBee's default cache dir and could not find `model.safetensors.index.json` despite the model being present in the normal HF hub cache.

## Server startup evidence

```text
Jul 03 12:45:13.398 [INFO] Running a server on ['/ip4/192.168.178.37/tcp/31437/p2p/12D3KooWRTkqybfmcXMTKDEtwHcFmdiNsyXVkyNW4EkoKE4hwMHk']
Jul 03 12:45:13.398 [WARN] Type bfloat16 is not supported on MPS, using float16 instead
Jul 03 12:45:13.429 [INFO] Loading HF weights for model.layers.0. from Qwen/Qwen3-30B-A3B
Jul 03 12:45:15.380 [INFO] Inference throughput: 115.6 tokens/sec per block (1 tokens/batch, MPS, float16)
Jul 03 12:45:18.636 [INFO] Forward pass throughput: 5801.7 tokens/sec per block (1024 tokens/batch, MPS, float16)
Jul 03 12:45:42.856 [INFO] Network throughput: 610.3 tokens/sec (34.12 Mbit/s on download, 20.00 Mbit/s on upload)
Jul 03 12:45:42.874 [INFO] Announced that blocks range(0, 1) are joining
Jul 03 12:45:42.920 [INFO] Loading HF weights for model.layers.0. from Qwen/Qwen3-30B-A3B
Jul 03 12:45:43.183 [INFO] Started
```

Non-fatal noise: FlexGen copy-worker threads still attempt `torch.cuda.set_device()` on macOS and log `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`. The server continues, loads weights, announces the block, and serves RPCs. This should be cleaned up later but did not block the proof.

## Direct RPC proof

Command:

```bash
cd ~/Projects/distributed-inference-mvp-proof
source ~/Projects/distributed-inference-mvp/.venv/bin/activate
env PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/direct_remote_call.py \
    --server-maddr "/ip4/192.168.178.37/tcp/31437/p2p/12D3KooWRTkqybfmcXMTKDEtwHcFmdiNsyXVkyNW4EkoKE4hwMHk" \
    --model Qwen/Qwen3-30B-A3B \
    --hidden-dim 2048 \
    --block-range 0:1
```

Result:

```json
{
  "ok": true,
  "model": "Qwen/Qwen3-30B-A3B",
  "block_range": [0, 1],
  "input_shape": [1, 5, 2048],
  "output_shape": [1, 5, 2048],
  "outputs_finite": true,
  "outputs_unique": 6331.0,
  "grad_finite": true,
  "grad_norm": 101.02727508544922,
  "forward_seconds": 0.6200950145721436,
  "backward_seconds": 1.1640067100524902
}
```

## Status

D preflight gate is now **passed for one live Qwen3-30B-A3B MoE block shard**:

- Real config dispatch works.
- Wrapper contract tests pass.
- Full 61.1GB cache is present and complete.
- One real block server starts from clean HEAD.
- Server loads real Qwen3-MoE block weights from safetensors.
- Direct remote RPC forward/backward through the live block succeeds with finite output and gradient.

Still not proven yet:

- Multi-block Qwen3-MoE server.
- Full 48-layer Qwen3-30B-A3B distributed generation.
- 10-laptop physical Qwen3 showcase.
- Robust S2S push/recovery path from F.
