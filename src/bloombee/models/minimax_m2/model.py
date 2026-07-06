"""Distributed MiniMax-M2 model wrappers for BloomBee native block serving."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.minimax_m2 import (
    MiniMaxM2ForCausalLM as _BaseCausalLM,
    MiniMaxM2Model as _BaseModel,
    MiniMaxM2PreTrainedModel as _BasePreTrained,
)

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.minimax_m2.config import DistributedMiniMaxM2Config
from bloombee.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


class DistributedMiniMaxM2Model(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, _BaseModel):
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]
    config_class = DistributedMiniMaxM2Config

    def __init__(self, config: DistributedMiniMaxM2Config, *, dht: Optional[DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer
        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)
        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert attention_mask is None or (attention_mask == 1).all(), f"Custom attention masks are not supported, {attention_mask=}"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all(), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"
        assert not output_router_logits, f"{output_router_logits=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)
        if not isinstance(past_key_values, RemotePastKeyValues):
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))
        hidden_states = self.layers(hidden_states, prompts=intermediate_prompts, hypo_ids=past_key_values.hypo_ids)
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            router_logits=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.norm


class DistributedMiniMaxM2ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, _BaseCausalLM):
    _keys_to_ignore_on_load_missing = DistributedMiniMaxM2Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedMiniMaxM2Model._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True
    config_class = DistributedMiniMaxM2Config

    def __init__(self, config: DistributedMiniMaxM2Config):
        _BasePreTrained.__init__(self, config)
        self.model = DistributedMiniMaxM2Model(config)
        self.lm_head = LMHead(config)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.001)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "_seen_tokens", cache_length)
                if hasattr(past_key_values, "get_max_length"):
                    max_cache_length = past_key_values.get_max_length()
                elif hasattr(past_key_values, "get_max_cache_shape"):
                    max_cache_length = past_key_values.get_max_cache_shape()
                else:
                    max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
