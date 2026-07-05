"""
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from enum import Enum
from typing import Optional, Sequence

import numpy as np
try:
    import tensor_parallel as tp
    from tensor_parallel.slicing_configs import get_bloom_config
except ImportError:
    tp = None
    get_bloom_config = None
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig
try:
    from pynvml import *
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
from bloombee.utils.debug import dprint
from bloombee.utils.memory_usage import see_memory_usage, log_mem
try:
    from bloombee.server.flexgen_tensor_parallel import FlexgenLlamaTensorParallel
except ImportError:
    FlexgenLlamaTensorParallel = None

logger = get_logger(__name__)


def _get_choice(cur_percent, percents, choices):
    """Return which device a parameter belongs to based on its cumulative position.

    Mirrors LLaMA's get_choice in flex_llama.py / from_pretrained.py.

    Args:
        cur_percent: Midpoint percentage (0-100) of this parameter in the full model.
        percents:    Allocation percentages [disk%, cpu%, gpu%] that must sum to 100.
        choices:     Corresponding device choices.
    """
    cum = np.cumsum(percents)
    for i, boundary in enumerate(cum):
        if cur_percent < boundary:
            return choices[i]
    return choices[-1]


def _assign_param_devices(module, policy, gpu_device):
    """Assign each named parameter to CPU or GPU using the same cumulative-midpoint
    logic that LLaMA's FlexGen system uses (init_weight_list in flex_llama.py).

    With policy.w_gpu_percent=50 / w_cpu_percent=50 the first ~50% of parameters
    (by element count) are placed on CPU, the remaining ~50% on GPU.

    Disk offload is not supported for standard HF modules; w_disk_percent is merged
    into the CPU allocation instead.

    Returns:
        dict mapping parameter name → torch.device
    """
    cpu_device = torch.device('cpu')

    param_list = list(module.named_parameters())
    if not param_list:
        return {}

    sizes = np.array([p.numel() for _, p in param_list], dtype=np.float64)
    sizes_cumsum = np.cumsum(sizes)
    total = sizes_cumsum[-1]

    # Merge disk% into CPU% (disk offload not implemented for HF blocks)
    effective_cpu = getattr(policy, 'w_cpu_percent', 0) + getattr(policy, 'w_disk_percent', 0)
    effective_gpu = getattr(policy, 'w_gpu_percent', 100)
    # percents must sum to 100; the first bucket (0%) is a placeholder so that the
    # cumulative sum aligns with [effective_cpu, 100] boundaries.
    dev_percents = [0.0, float(effective_cpu), float(effective_gpu)]
    dev_choices  = [cpu_device, cpu_device, gpu_device]

    param_devices = {}
    for i, (name, _) in enumerate(param_list):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / total * 100
        param_devices[name] = _get_choice(mid_percent, dev_percents, dev_choices)
    return param_devices


class QuantType(Enum):
    """
    Quantization type enum for FlexGen compression.
    Note: bitsandbytes quantization is not used. This enum only controls FlexGen's group-wise quantization.
    """
    NONE = 0
    INT8 = 1  # 8-bit group-wise quantization for FlexGen
    NF4 = 2  # 4-bit group-wise quantization for FlexGen


def convert_block(
    block: nn.Module,
    block_index: int,
    config: PretrainedConfig,
    tensor_parallel_devices: Sequence[torch.device],
    output_device: torch.device,
    quant_type: QuantType,
    freeze: bool = True,
    adapters: Optional[Sequence[str]] = None,
    policy=None,
    **kwargs,
) -> "tp.TensorParallel":
    """
    Optimize a transformer block for use in a Petals server with FlexGen.
    
    Note: Quantization is handled by FlexGen's weight loading system, not here.
    The quant_type parameter is passed through but not used in this function.

    :note: some optimizations will modify the input block in-place!
    :param block: a single transformer block, either pre-trained or newly initialized
    :param config: HF transformers config for the full model
    :param tensor_parallel_devices: if specified, use tensor parallelism to split the model between these devices
    :note: if there is only a single device, model wil still be wrapped with TensorParallel (for uniformity)
    :param output_device: if tensor_parallel_devices is True, output
    :param quant_type: quantization type (used by FlexGen compression, not applied here)
    :param freeze: if True (default), make all module parameters non-trainable
    :return: a module that acts like the original block, but runs with all specified optimizations

    """
    if freeze:
        block.requires_grad_(False)
    if len(tensor_parallel_devices) > 1 and config.model_type == "llama":
        return make_tensor_parallel(
            block,
            config,
            tensor_parallel_devices,
            output_device,
            policy=policy,
        )
    if len(tensor_parallel_devices) > 1:
        # Only the FlexGen-native LLaMA path implements tensor parallelism.
        # Silently running single-device while the operator believes TP is on
        # would corrupt any scaling experiment, so say it loudly.
        logger.warning(
            "--tensor_parallel_devices is only implemented for LLaMA (FlexGen-native TP); "
            f"model_type={config.model_type!r} will run on a single device and the TP "
            f"request across {len(tensor_parallel_devices)} devices is ignored."
        )

    # Skip tensor parallelism for FlexGen blocks - they manage their own weights and devices
    log_prefix = f"[convert_block:{block_index}]"
    # log_mem(f"{log_prefix} skipping tensor parallelism - FlexGen manages weights directly")

    # FlexGen-native blocks (meta params) compress during their own weight
    # loading; HF blocks get weight-only quantization here or nowhere.
    first_param = next(iter(block.parameters()), None)
    is_hf_block_with_weights = first_param is not None and first_param.device.type != "meta"
    if quant_type != QuantType.NONE and is_hf_block_with_weights:
        effective_cpu = getattr(policy, "w_cpu_percent", 0) + getattr(policy, "w_disk_percent", 0)
        if policy is not None and effective_cpu > 0:
            raise ValueError(
                "quantized HF blocks do not support per-parameter CPU offload "
                f"(w_cpu+w_disk={effective_cpu}%): the offload path moves param.data and "
                "would corrupt packed quantized tensors. Quantization shrinks the block "
                "instead; use w_gpu_percent=100 or quant_type=none."
            )
        quantize_hf_block(block, quant_type=quant_type, model_type=config.model_type)
        if freeze:
            # quanto swaps nn.Linear modules after the initial freeze above;
            # those fresh QLinear parameters default to requires_grad=True.
            # Serving backends require all parameters frozen.
            block.requires_grad_(False)

    # Create a simple wrapper that provides TensorParallel interface for pipeline parallelism
    # but uses FlexGen's forward method directly
    class PipelineParallelWrapper:
        def __init__(self, module, devices, output_device, block_index=0, policy=None):
            self._module = module
            self.devices = devices
            self.output_device = output_device
            self.output_device_index = 0  # Single device in pipeline parallelism
            self.module_shards = [module]  # Single shard per pipeline stage

            # Fine-grained per-parameter CPU/GPU split, mirroring LLaMA's FlexGen approach.
            # FlexGen blocks use meta-device initially — skip them.
            first_param = next(iter(module.parameters()), None)
            is_hf_block = (
                first_param is not None
                and first_param.device.type != 'meta'
                and output_device is not None
            )

            self._param_devices = {}   # name → torch.device
            self._cpu_offload = False

            if is_hf_block and policy is not None:
                effective_cpu = (
                    getattr(policy, 'w_cpu_percent', 0)
                    + getattr(policy, 'w_disk_percent', 0)
                )
                if effective_cpu > 0:
                    # Assign each parameter individually using cumulative-midpoint logic.
                    # Buffers (e.g. rotary embedding inv_freq, lazy cos_cached/sin_cached)
                    # are kept on GPU at all times to avoid device-mismatch issues with
                    # lazily-registered buffers (registered during the first forward call).
                    self._param_devices = _assign_param_devices(module, policy, output_device)
                    pin = getattr(policy, 'pin_weight', False) and output_device.type == 'cuda'

                    # Move parameters to their assigned device
                    for name, param in module.named_parameters():
                        target = self._param_devices[name]
                        if target.type == 'cpu':
                            if pin:
                                param.data = param.data.cpu().pin_memory()
                            else:
                                param.data = param.data.cpu()
                        else:
                            param.data = param.data.to(output_device)

                    # Always keep buffers on GPU so that lazily-registered buffers
                    # (like Falcon's cos_cached / sin_cached) are created on GPU too.
                    for buf_name, buf in list(module.named_buffers()):
                        if buf is not None and buf.device.type != output_device.type:
                            # Navigate to the submodule that owns this buffer and re-register
                            parts = buf_name.split('.')
                            submod = module
                            for part in parts[:-1]:
                                submod = getattr(submod, part)
                            submod.register_buffer(parts[-1], buf.to(output_device), persistent=False)

                    self._cpu_offload = any(
                        d.type == 'cpu' for d in self._param_devices.values()
                    )
                    n_cpu = sum(1 for d in self._param_devices.values() if d.type == 'cpu')
                    n_gpu = len(self._param_devices) - n_cpu
                    logger.info(
                        f"[block {block_index}] Per-parameter CPU offload: "
                        f"{n_cpu}/{len(self._param_devices)} params on CPU, "
                        f"{n_gpu}/{len(self._param_devices)} params on GPU "
                        f"(w_gpu={getattr(policy,'w_gpu_percent',100)}%, "
                        f"w_cpu={getattr(policy,'w_cpu_percent',0)}%)"
                    )
                else:
                    # All weights on GPU
                    module.to(output_device)
            elif is_hf_block:
                module.to(output_device)

        def forward(self, *args, **kwargs):
            if self._cpu_offload:
                # Move CPU-resident parameters to GPU before forward.
                # Buffers stay on GPU permanently (see __init__).
                for name, param in self._module.named_parameters():
                    if self._param_devices.get(name, self.output_device).type == 'cpu':
                        param.data = param.data.to(self.output_device, non_blocking=True)
                if self.output_device.type == 'cuda':
                    torch.cuda.synchronize(self.output_device)

                result = self._module.forward(*args, **kwargs)

                # Restore CPU params asynchronously after forward
                for name, param in self._module.named_parameters():
                    if self._param_devices.get(name, self.output_device).type == 'cpu':
                        param.data = param.data.to('cpu', non_blocking=True)
                return result
            return self._module.forward(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
            
        def parameters(self, *args, **kwargs):
            return self._module.parameters(*args, **kwargs)

        def named_parameters(self, *args, **kwargs):
            return self._module.named_parameters(*args, **kwargs)

        def parameters(self, *args, **kwargs):
            return self._module.parameters(*args, **kwargs)
            
        def named_buffers(self, *args, **kwargs):
            return self._module.named_buffers(*args, **kwargs)

        def buffers(self, *args, **kwargs):
            return self._module.buffers(*args, **kwargs)
        
        def rms_norm(self, *args, **kwargs):
            if hasattr(self._module, 'rms_norm'):
                return self._module.rms_norm(*args, **kwargs)
            return None

        def load_lm_head(self, *args, **kwargs):
            if hasattr(self._module, 'load_lm_head'):
                return self._module.load_lm_head(*args, **kwargs)
            # No-op for non-FlexGen models (Falcon, Mixtral)

        def lm_head_forward(self, *args, **kwargs):
            if hasattr(self._module, 'lm_head_forward'):
                return self._module.lm_head_forward(*args, **kwargs)
            return None
    
    tp_block = PipelineParallelWrapper(block, tensor_parallel_devices, output_device, block_index=block_index, policy=policy)
    # log_mem(f"{log_prefix} created PipelineParallel wrapper")
    
    dprint('quant_type ', quant_type)
    dprint('adapters ', adapters )
    if adapters:
        
        from bloombee.utils.peft import add_adapter_to_block, create_lora_adapter, load_peft

        create_lora_adapter(tp_block)
        for adapter_name in adapters:
            adapter_config, adapter_state_dict = load_peft(
                adapter_name,
                block_idx=block_index,
                **kwargs,
            )
            add_adapter_to_block(tp_block, block_index, adapter_name, adapter_config, adapter_state_dict)

    return tp_block


def quantize_hf_block(block: nn.Module, *, quant_type: QuantType, model_type: str) -> dict:
    """Weight-only quantization for standard HF blocks (qwen3, qwen3_moe, ...).

    FlexGen-native blocks keep their own group-wise compression; this path is
    for the HF-module families that previously had no quantization at all.

    Mapping (fail-closed, never silently falls back to fp16):
      - INT8 + dense block:     optimum-quanto qint8 on all Linears
      - INT8 + qwen3_moe block: custom int8 fused-expert swap, then quanto
                                qint8 on the remaining Linears
      - NF4  + qwen3_moe block: packed int4 fused-expert swap, then quanto
                                qint8 on the remaining Linears (experts are
                                ~97% of block bytes; attention stays int8 so
                                serving never depends on quanto's JIT-built
                                qint4 C++/MPS extension)
      - NF4  + dense block:     blocked until a deterministic dense int4
                                path exists

    The router Linear (``*gate``) is excluded from quanto on MoE blocks:
    routing is an argmax over its logits, and keeping it fp16 keeps expert
    selection bit-identical to the fp16 reference.
    """
    stats = {
        "quant_type": quant_type.name.lower(),
        "model_type": model_type,
        "applied": False,
        "moe_expert_swap": None,
    }
    if quant_type == QuantType.NONE:
        return stats

    is_moe = model_type == "qwen3_moe"
    if quant_type == QuantType.NF4 and not is_moe:
        raise NotImplementedError(
            f"NF4 for dense HF blocks (model_type={model_type!r}) is blocked: quanto qint4 "
            "needs a JIT-built C++/MPS extension we refuse to depend on in the serving path. "
            "Use INT8, or extend the packed-int4 path beyond fused MoE experts."
        )

    try:
        from optimum.quanto import freeze, qint8, quantize
    except ImportError as exc:  # fail closed: operator asked for quantization
        raise RuntimeError(
            f"quant_type={quant_type.name} requires optimum-quanto for HF blocks; "
            "install it or use --quant_type none"
        ) from exc

    def _bytes(module: nn.Module) -> int:
        return sum(
            v.numel() * v.element_size() for v in module.state_dict().values() if isinstance(v, torch.Tensor)
        )

    before_bytes = _bytes(block)
    exclude = []
    if is_moe:
        from bloombee.utils.moe_expert_quant import quantize_qwen3_moe_block_experts

        expert_scheme = "int4" if quant_type == QuantType.NF4 else "int8"
        stats["moe_expert_swap"] = quantize_qwen3_moe_block_experts(block, scheme=expert_scheme)
        exclude = ["*gate"]  # router logits must stay fp16-exact

    quantize(block, weights=qint8, exclude=exclude)
    freeze(block)
    after_bytes = _bytes(block)

    stats.update(
        applied=True,
        linear_weights="quanto qint8",
        router_excluded=is_moe,
        weight_bytes_before=before_bytes,
        weight_bytes_after=after_bytes,
        compression_ratio=round(before_bytes / after_bytes, 3) if after_bytes else None,
    )
    logger.info(
        f"Quantized HF block ({model_type}, {quant_type.name}): "
        f"{before_bytes / 1e6:.1f}MB -> {after_bytes / 1e6:.1f}MB "
        f"({stats['compression_ratio']}x)"
    )
    return stats


def make_tensor_parallel(
    block: nn.Module,
    model_config: PretrainedConfig,
    devices: Sequence[torch.device],
    output_device: torch.device,
    policy=None,
) -> nn.Module:
    if model_config.model_type == "llama":
        num_kv_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
        if (
            not hasattr(block, "env")
            or not hasattr(block, "path")
            or num_kv_heads != model_config.num_attention_heads
            or model_config.num_attention_heads % len(devices) != 0
            or model_config.intermediate_size % len(devices) != 0
        ):
            raise ValueError(
                "BloomBee only supports FlexGen-native LLaMA tensor parallelism. "
                f"Unsupported config: num_attention_heads={model_config.num_attention_heads}, "
                f"num_key_value_heads={num_kv_heads}, intermediate_size={model_config.intermediate_size}, "
                f"tp_world_size={len(devices)}"
            )

        return FlexgenLlamaTensorParallel(
            block,
            model_config,
            devices,
            output_device,
            policy=policy,
        )

    if model_config.model_type == "bloom":
        tp_config = get_bloom_config(model_config, devices)
        del tp_config.state_rules[re.compile(".*word_embeddings.weight$")]
    else:
        if len(devices) > 1:
            logger.warning("Tensor parallelism is not tested for models other than BLOOM yet, proceed with caution")
        tp_config = None
    tp_block = tp.TensorParallel(block, devices, config=tp_config, output_device=output_device, delay_init=True)
    # print('make_tensor_parallel: tp_block ', tp_block)
    # import pdb; pdb.set_trace()
    total_heads = 0
    for tp_shard in tp_block.module_shards:
        for submodule in tp_shard.modules():
            # print("flex_llama.LlamaAttention ", flex_llama.LlamaAttention)
            # print("submodule ", submodule)
            if isinstance(submodule, model_config.attn_class):
                total_heads += submodule.num_heads
    if model_config.model_type == "bloom":
        assert total_heads == model_config.num_attention_heads
    return tp_block


def check_device_balance(devices: Sequence[torch.device]):
    if not all(device.type == "cuda" for device in devices):
        logger.warning("Running tensor parallelism on non-GPU devices; proceed at your own risk")
        return
    unique_device_capabilities = set(map(torch.cuda.get_device_capability, devices))
    if len(unique_device_capabilities) > 1:
        logger.warning(
            f"Found GPUs with uneven capabilities: {unique_device_capabilities}. "
            f"Using GPUs with different performance will cause the server to wait for the slowest GPU."
        )

    memory_per_device = tuple(torch.cuda.get_device_properties(device).total_memory for device in devices)
    used_memory = min(memory_per_device) * len(memory_per_device)
    wasted_memory_rate = (sum(memory_per_device) - used_memory) / sum(memory_per_device)
    if wasted_memory_rate > 0.05:
        logger.warning(
            f"GPU devices have highly uneven memory, {wasted_memory_rate * 100:.2f}% memory is wasted. "
            f"Consider running high-memory GPUs in a separate server."
        )
