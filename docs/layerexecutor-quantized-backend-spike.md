# LayerExecutor / Quantized Backend Feasibility Spike

**Status:** research spike only. No runnable backend proof. No MVP-core status change.

Evidence artifact:

```text
mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json
```

## Claim boundary

This spike answers a planning question: which frontier open-weight families are plausible post-MVP targets for a LayerExecutor or quantized-backend path, and what proof would be required before any route/demo claim?

It does **not** prove:

- native BloomBee support for any frontier target below;
- external quantized runtime serving;
- route-picker eligibility;
- user-facing demo safety;
- any change to MVP-core 100%.

## Current BloomBee boundary

The current compatibility scanner recognizes these BloomBee model families:

```text
bloom, falcon, gemma4, llama, mixtral, qwen3, qwen3_moe
```

Everything below is blocked today because its HF `model_type` is outside that set and/or requires quantized/sparse-attention runtime support not present in the existing BloomBee block wrappers.

## Target findings

| Model | Config / public facts | Current blocker | Minimal next proof |
|---|---|---|---|
| `MiniMaxAI/MiniMax-M3` | model_type `minimax_m3_vl`; 60 layers; hidden 6144; experts 128×4; context 1048576; quant `None` | No BloomBee block wrapper registered for model_type=minimax_m3_vl; Research sources describe MiniMax Sparse Attention and multimodal VLM inputs; current BloomBee wrappers do not expose MSA/multimodal text-tower serving contracts. | Config-only prescan plus text-tower wrapper scout. Do not attempt native BloomBee serving until minimax_m3_vl/minimax_m3_vl_text attention and multimodal wrapper assumptions are mapped. External-runtime smoke should be separate from BloomBee block-serving proof. |
| `zai-org/GLM-5.2` | model_type `glm_moe_dsa`; 78 layers; hidden 6144; experts 256×8; context 1048576; quant `None` | No BloomBee block wrapper registered for model_type=glm_moe_dsa; Research sources describe MLA + Dynamic Sparse Attention / TileLang DSA; current BloomBee wrappers do not implement glm_moe_dsa attention state or sparse kernels. | Add a wrapper-scout artifact for glm_moe_dsa config fields and DSA attention contract before coding. A runnable proof likely needs a GPU sparse-kernel runtime; macOS/MPS laptop proof is not a credible first target. |
| `deepseek-ai/DeepSeek-V4-Flash` | model_type `deepseek_v4`; 43 layers; hidden 4096; experts 256×6; context 1048576; quant `fp8` | No BloomBee block wrapper registered for model_type=deepseek_v4; Quantized HF checkpoint declares quantization_config=fp8; current BloomBee HF-block loader instantiates fp16/bf16 PyTorch blocks and does not build GPTQ/AWQ/FP8/NVFP4/MXFP quantized Linear kernels; Research sources describe DeepSeek-V4 hybrid sparse/compressed attention; current BloomBee wrappers do not implement deepseek_v4 attention state or quantized kernels. | Treat as quantized-backend work, not a native BloomBee wrapper first. Minimal proof is external vLLM/SGLang/TensorRT-LLM config/runtime smoke on suitable NVIDIA hardware, followed by a narrow LayerExecutor adapter proposal. |
| `moonshotai/Kimi-K2-Instruct` | model_type `kimi_k2`; 61 layers; hidden 7168; experts 384×8; context 131072; quant `fp8` | No BloomBee block wrapper registered for model_type=kimi_k2; Quantized HF checkpoint declares quantization_config=fp8; current BloomBee HF-block loader instantiates fp16/bf16 PyTorch blocks and does not build GPTQ/AWQ/FP8/NVFP4/MXFP quantized Linear kernels; HF AutoConfig requires trust_remote_code; planning scan used local config.json only and did not execute remote model code. | Avoid trust_remote_code during planning. Use local config.json scans only until a wrapper/backend scout explicitly audits custom code. Quantized external-runtime smoke is a separate opt-in proof gate. |

## Recommendation

Do **not** spend the next scarce proof window on these frontier backends before base Qwen3-30B-A3B full-generation/cache/load. The base Qwen3-30B path is already cached on `m4pro` and has prescan + one-block + multi-block proof. These frontier targets require new wrappers, custom attention contracts, quantized kernels, or external GPU runtimes before a meaningful BloomBee claim exists.

If a research lane continues anyway, keep it narrow:

1. Build a `LayerExecutor` boundary document/API shape, not a full wrapper.
2. Pick one target only.
3. Start with config-only and external-runtime smoke proof.
4. Keep the evidence claim as `external_runtime_smoke_only` until a BloomBee block-level call path exists.
5. Never promote route selection from this spike alone.

## External-runtime smoke planner

`mvp_capabilities/frontier_backend_smoke_plan.py` turns the research spike into
a concrete, fail-closed operator plan. It emits claim boundary
`frontier_external_runtime_smoke_plan_only_no_bloombee_route_claim`, keeps native
BloomBee support false, and keeps route/demo promotion false.

Example first target for quantized form:

```bash
python mvp_capabilities/frontier_backend_smoke_plan.py \
  --spike-artifact mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json \
  --target deepseek-ai/DeepSeek-V4-Flash \
  --out .local/frontier/deepseek-v4-flash-smoke-plan.json
```

The generated plan prefers `deepseek-ai/DeepSeek-V4-Flash` when requested because
its config advertises `fp8`, but this still requires suitable NVIDIA GPU runtime
support such as vLLM/SGLang/TensorRT-LLM. It is not a macOS/MPS proof and not a
native BloomBee route.

## Route-report visibility

The route picker now exposes unsupported frontier rows as planning visibility,
without selecting them for showcase or demo serving. Use `route_picker.py --report`
when an operator wants to inspect a pinned frontier request and the fallback that
would actually be served:

```bash
.venv/bin/python mvp_capabilities/route_picker.py --report \
  --selector-mode showcase-attempt \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --synthetic-m4-laptops 20 \
  --synthetic-total-gb 128 \
  --synthetic-free-gb 100 || true
```

The JSON includes `blocked_frontier_candidates` under claim boundary
`blocked_frontier_candidates_no_serving_proof`. Entries in that array set
`can_update_proof_status=false` and `inference_proven=false`; they are evidence
that the planner knows why the target is blocked, not evidence that the target
runs.

## Sources

- MiniMax M3 official blog: <https://www.minimax.io/blog/minimax-m3>
- NVIDIA MiniMax M3 deployment blog: <https://developer.nvidia.com/blog/deploy-long-context-reasoning-and-agentic-workflows-with-minimax-m3-on-nvidia-accelerated-infrastructure>
- MiniMax M3 Hugging Face model card: <https://huggingface.co/MiniMaxAI/MiniMax-M3>
- NVIDIA NeMo GLM-5/5.1/5.2 coverage: <https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/glm-5-moe-dsa>
- DeepSeek-V4-Flash NVIDIA NIM model card: <https://build.nvidia.com/deepseek-ai/deepseek-v4-flash/modelcard>
- DeepSeek-V4-Flash Hugging Face model card: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Kimi K2 Instruct Hugging Face model card: <https://huggingface.co/moonshotai/Kimi-K2-Instruct>

## Verification

```bash
.venv/bin/python -m json.tool mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json >/dev/null
.venv/bin/python -m pytest tests/test_mvp_capabilities.py::test_layerexecutor_quantized_backend_spike_artifact_is_conservative -q
.venv/bin/python -m pytest tests/test_frontier_backend_smoke_plan.py -q
```
