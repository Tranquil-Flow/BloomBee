"""Int8 weight-only quantization for fused 3D MoE expert tensors.

Why this exists: transformers 5.x stores Qwen3-MoE expert weights as fused 3D
``nn.Parameter`` tensors on ``Qwen3MoeExperts`` (``gate_up_proj`` of shape
``[num_experts, 2*intermediate, hidden]`` and ``down_proj`` of shape
``[num_experts, hidden, intermediate]``), not as per-expert ``nn.Linear``
modules. Off-the-shelf weight-only quantizers (optimum-quanto, torchao) walk
``nn.Linear`` and therefore skip ~97% of a Qwen3-30B-A3B block's bytes.

This module provides a drop-in replacement that stores the expert tensors as
int8 with per-output-channel fp16 scales and dequantizes only the experts hit
by the router in each forward. For decode (top-8 of 128 experts) that is a
small, bounded dequant per token. Attention/router linears are intentionally
left alone so a standard Linear quantizer can handle them independently.

Quantization scheme: symmetric per-output-channel int8 over the input
dimension (the last dim of each expert matrix), i.e. one fp16 scale per
``[expert, out_channel]``. No zero points. Dequant is ``q.to(dtype) * scale``.

The forward mirrors ``Qwen3MoeExperts.forward`` exactly (same routing-mask
construction, same ``index_add_`` accumulation) so generated tokens differ
from fp16 only by quantization rounding, never by routing logic.

Backward: expert weights are frozen buffers (no grad); gradients flow to the
activations through the dequantized matmuls, which BloomBee's backward RPC
path requires.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def quantize_per_out_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization with one scale per leading [..., out, in] row.

    Accepts ``[E, out, in]`` (or ``[out, in]``) and returns ``(q, scale)`` where
    ``q`` is int8 with the same shape and ``scale`` is fp16 shaped like
    ``weight`` with the last dim reduced to 1, so ``q.to(fp16) * scale``
    reconstructs the weight.
    """
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs.clamp(min=1e-8) / 127.0).to(torch.float16)
    q = torch.round(weight / scale.to(weight.dtype)).clamp(-127, 127).to(torch.int8)
    return q, scale


class QuantizedQwen3MoeExperts(nn.Module):
    """Drop-in int8 replacement for ``Qwen3MoeExperts``.

    Construct via :meth:`from_experts`. Keeps the same forward contract:
    ``forward(hidden_states_2d, top_k_index, top_k_weights)``.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: nn.Module,
        gate_up_q: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_q: torch.Tensor,
        down_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.register_buffer("gate_up_proj_q", gate_up_q)
        self.register_buffer("gate_up_proj_scale", gate_up_scale)
        self.register_buffer("down_proj_q", down_q)
        self.register_buffer("down_proj_scale", down_scale)

    @classmethod
    def from_experts(cls, experts: nn.Module) -> "QuantizedQwen3MoeExperts":
        """Quantize a ``Qwen3MoeExperts``-shaped module (needs ``gate_up_proj``,
        ``down_proj``, ``act_fn``, ``num_experts``, ``hidden_dim``,
        ``intermediate_dim`` attributes)."""
        gate_up = experts.gate_up_proj.data
        down = experts.down_proj.data
        gate_up_q, gate_up_scale = quantize_per_out_channel_int8(gate_up)
        down_q, down_scale = quantize_per_out_channel_int8(down)
        return cls(
            num_experts=int(experts.num_experts),
            hidden_dim=int(experts.hidden_dim),
            intermediate_dim=int(experts.intermediate_dim),
            act_fn=experts.act_fn,
            gate_up_q=gate_up_q,
            gate_up_scale=gate_up_scale,
            down_q=down_q,
            down_scale=down_scale,
        )

    def _dequant(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return q.to(dtype) * scale.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        dtype = hidden_states.dtype
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_w = self._dequant(self.gate_up_proj_q[expert_idx], self.gate_up_proj_scale[expert_idx], dtype)
            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_w = self._dequant(self.down_proj_q[expert_idx], self.down_proj_scale[expert_idx], dtype)
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def module_weight_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def quantize_qwen3_moe_block_experts(block: nn.Module) -> dict[str, Any]:
    """Swap every ``*.experts`` fused-3D-expert module under ``block`` for a
    :class:`QuantizedQwen3MoeExperts` and return before/after byte stats.

    Fail-closed: raises ``ValueError`` if no expert module with fused 3D
    parameters is found, so callers cannot silently believe a block was
    quantized when the architecture did not match.
    """
    swapped: list[str] = []
    before_bytes = module_weight_bytes(block)
    for name, module in list(block.named_modules()):
        if not name.endswith("experts"):
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if not (isinstance(gate_up, torch.Tensor) and gate_up.dim() == 3):
            continue
        if not (isinstance(down, torch.Tensor) and down.dim() == 3):
            continue
        quantized = QuantizedQwen3MoeExperts.from_experts(module)
        quantized = quantized.to(gate_up.device)
        parent_name, _, attr = name.rpartition(".")
        parent = block.get_submodule(parent_name) if parent_name else block
        setattr(parent, attr, quantized)
        swapped.append(name)
    if not swapped:
        raise ValueError(
            "no fused 3D MoE expert modules found under block; "
            "refusing to report a no-op as quantization"
        )
    after_bytes = module_weight_bytes(block)
    return {
        "swapped_modules": swapped,
        "weight_bytes_before": before_bytes,
        "weight_bytes_after": after_bytes,
        "compression_ratio": round(before_bytes / after_bytes, 3) if after_bytes else None,
        "scheme": "int8_symmetric_per_out_channel",
    }
