"""BloomBee block wrapper for Qwen3.5-MoE text tower.

This is a contract-only post-MVP wrapper slice for the language model tower
(``Qwen3_5MoeDecoderLayer`` / ``Qwen3_5MoeTextConfig``). It mirrors the proven
Qwen3-MoE cache/causal-mask adapter while handling Qwen3.5's explicit M-RoPE
``position_embeddings`` requirement.
"""

from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _BaseTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer as _BaseDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding as _RotaryEmbedding,
)

from bloombee.utils.cache_compat import make_empty_kv_cache, make_past_kv_cache, read_kv_from_cache


class WrappedQwen3_5MoeTextBlock(_BaseDecoderLayer):
    def __init__(self, config: _BaseTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self._attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.sliding_window = getattr(config, "sliding_window", None)
        self.layer_idx = layer_idx
        self._rotary_emb = _RotaryEmbedding(config)

        if hasattr(self, "self_attn"):
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

        if self.layer_type == "linear_attention":
            if layer_past is not None or use_cache:
                past_key_value = self._linear_cache_from_bloom(layer_past, hidden_states)
            # The linear mixer consumes only a 2D padding mask. BloomBee's
            # generated causal masks are for full-attention KV layers, so do
            # not synthesize or forward them here.
            if attention_mask is not None and attention_mask.dim() > 2:
                attention_mask = None
        elif past_key_value is not None:
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

        if self.layer_type != "linear_attention" and attention_mask is None:
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
        elif attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(hidden_states, position_ids)
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

        output_hidden = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **extra_kwargs,
        )

        if use_cache and past_key_value is not None and self.layer_type == "linear_attention":
            present_state = self._linear_cache_to_bloom(past_key_value, hidden_states)
            if present_state is not None:
                return (output_hidden, present_state)

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
        nkv = self.self_attn.num_key_value_heads
        if key_states.dim() == 4:
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(batch_size, nkv, seq_length, self.self_attn.head_dim)
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        nkv = self.self_attn.num_key_value_heads
        head_dim = self.self_attn.head_dim
        value_states = value_states.reshape(batch_size * nkv, seq_length, head_dim)
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)

    def _linear_cache_from_bloom(
        self,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        hidden_states: torch.Tensor,
    ) -> DynamicCache:
        cache = DynamicCache(config=self.config)
        if state is None:
            return cache
        conv_state, recurrent_state = state
        conv_state = conv_state.to(device=hidden_states.device, dtype=hidden_states.dtype)
        recurrent_state = recurrent_state.to(device=hidden_states.device, dtype=hidden_states.dtype)
        cache.update_conv_state(conv_state, self.layer_idx)
        cache.update_recurrent_state(recurrent_state, self.layer_idx)
        return cache

    def _linear_cache_to_bloom(
        self,
        cache: DynamicCache,
        hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.layer_idx >= len(cache.layers):
            return None
        layer = cache.layers[self.layer_idx]
        conv_state = getattr(layer, "conv_states", None)
        recurrent_state = getattr(layer, "recurrent_states", None)
        if conv_state is None or recurrent_state is None:
            return None
        return (
            conv_state.to(device=hidden_states.device, dtype=hidden_states.dtype),
            recurrent_state.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
