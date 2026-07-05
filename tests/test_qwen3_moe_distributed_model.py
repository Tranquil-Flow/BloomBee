from __future__ import annotations

import torch

from bloombee.models.qwen3_moe.config import DistributedQwen3MoeConfig
from bloombee.models.qwen3_moe import model as qwen3_moe_model


class _FakeRemoteSequential(torch.nn.Module):
    def __init__(self, config, dht=None):
        super().__init__()
        self.config = config
        self.position = 0

    def forward(self, hidden_states, prompts=None, hypo_ids=None):
        return hidden_states


def _tiny_config() -> DistributedQwen3MoeConfig:
    cfg = DistributedQwen3MoeConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
        tie_word_embeddings=True,
        initializer_range=0.02,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_distributed_qwen3_moe_accepts_default_router_logits_kwarg(monkeypatch):
    monkeypatch.setattr(qwen3_moe_model, "RemoteSequential", _FakeRemoteSequential)
    model = qwen3_moe_model.DistributedQwen3MoeModel(_tiny_config()).eval()

    out = model(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        output_router_logits=False,
    )

    assert out.last_hidden_state.shape == (1, 3, _tiny_config().hidden_size)


def test_distributed_qwen3_moe_for_causal_lm_default_forward_keeps_logits(monkeypatch):
    monkeypatch.setattr(qwen3_moe_model, "RemoteSequential", _FakeRemoteSequential)
    model = qwen3_moe_model.DistributedQwen3MoeForCausalLM(_tiny_config()).eval()

    out = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert out.logits.shape == (1, 3, _tiny_config().vocab_size)


def test_distributed_qwen3_moe_rejects_requested_router_logits(monkeypatch):
    monkeypatch.setattr(qwen3_moe_model, "RemoteSequential", _FakeRemoteSequential)
    model = qwen3_moe_model.DistributedQwen3MoeModel(_tiny_config()).eval()

    try:
        model(torch.tensor([[1, 2, 3]], dtype=torch.long), output_router_logits=True)
    except AssertionError as exc:
        assert "output_router_logits" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("router-logit requests must remain fail-closed")
