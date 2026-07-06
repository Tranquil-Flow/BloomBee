"""BloomBee block wrapper for MiniMax-M2 / M2.7 REAP.

Claim boundary: this wrapper-level contract is not a live model proof. It wires
MiniMaxM2DecoderLayer into BloomBee's existing block/cache interface so exact
one-block server proof can be attempted separately.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.minimax_m2 import MiniMaxM2Config as _BaseBlockConfig
from transformers.models.minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2DecoderLayer as _BaseDecoderLayer,
    MiniMaxM2RotaryEmbedding as _RotaryEmbedding,
)

from bloombee.utils.cache_compat import make_empty_kv_cache, make_past_kv_cache, read_kv_from_cache


def build_mtp_contract(config: _BaseBlockConfig) -> dict[str, object]:
    use_mtp = bool(getattr(config, "use_mtp", False))
    guard_mode = getattr(config, "bloombee_minimax_m2_proof_scope", None)
    if not use_mtp:
        guard_mode = "not_required"
    return {
        "use_mtp": use_mtp,
        "num_mtp_modules": int(getattr(config, "num_mtp_modules", 0) or 0),
        "mtp_transformer_layers": int(getattr(config, "mtp_transformer_layers", 0) or 0),
        "guard_mode": guard_mode,
        "base_decoder_supported": True,
        "mtp_modules_supported": False,
        "requires_explicit_guard": use_mtp,
    }


def validate_mtp_contract(config: _BaseBlockConfig) -> dict[str, object]:
    contract = build_mtp_contract(config)
    if contract["requires_explicit_guard"] and contract["guard_mode"] != "base_decoder_only":
        raise ValueError(
            "MiniMax-M2 MTP use_mtp=True requires explicit base_decoder_only proof scope; "
            "BloomBee supports only the base decoder block contract here. Set "
            "bloombee_minimax_m2_proof_scope='base_decoder_only' "
            "for proof harnesses that intentionally exclude MTP modules."
        )
    return contract


class WrappedMiniMaxM2Block(_BaseDecoderLayer):
    def __init__(self, config: _BaseBlockConfig, layer_idx: int):
        mtp_contract = validate_mtp_contract(config)
        super().__init__(config, layer_idx)
        self.mtp_contract = mtp_contract
        self._attn_implementation = config._attn_implementation
        self.sliding_window = getattr(config, "sliding_window", None)
        self.layer_idx = layer_idx
        self._rotary_emb = _RotaryEmbedding(config)

        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        rot = getattr(self, "_rotary_emb", None)
        if rot is not None:
            for name in ("inv_freq", "original_inv_freq"):
                buf = getattr(rot, name, None)
                if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                    rot.register_buffer(name, buf.float(), persistent=False)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0
        past_key_value = layer_past

        if past_key_value is not None:
            pk, pv = past_key_value
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value = (pk, pv)
            past_key_values_length = past_key_value[0].shape[2]
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = make_past_kv_cache(
                _past_key_value[0],
                _past_key_value[1],
                layer_idx=self.layer_idx,
                seen_tokens=past_key_values_length,
            )
        elif use_cache:
            past_key_value = make_empty_kv_cache(self.layer_idx)

        if attention_mask is None:
            total_len = past_key_values_length + seq_length
            neg_inf = torch.finfo(hidden_states.dtype).min
            causal = torch.full(
                (seq_length, total_len),
                neg_inf,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if total_len > 0:
                causal = torch.triu(causal, diagonal=past_key_values_length + 1)
            attention_mask = causal.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(hidden_states, position_ids=position_ids)
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            dtype=torch.long,
            device=hidden_states.device,
        )

        skip_keys = {
            "position_ids",
            "attention_mask",
            "use_cache",
            "rotary_position_ids",
            "position_embeddings",
            "past_key_value",
            "past_key_values",
            "cache_position",
        }
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **extra_kwargs,
        )
        output_hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        if use_cache and past_key_value is not None:
            pk, pv = read_kv_from_cache(past_key_value, self.layer_idx)
            if pk is not None:
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
                return (output_hidden, present_key_value)

        return (output_hidden, None)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        if key_states.dim() == 4:
            nkv = self.self_attn.num_key_value_heads
            return key_states[:, :nkv, :, :], value_states[:, :nkv, :, :]
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim)
        value_states = value_states.view(*key_states.shape)
        return key_states, value_states

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape).permute(0, 2, 1)
        return key_states, value_states
