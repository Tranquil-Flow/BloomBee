"""Contract tests for BloomBee's Qwen3.5-MoE text wrapper.

Qwen/Qwen-AgentWorld-35B-A3B exposes an outer ``qwen3_5_moe`` config with a
language-model text tower using ``qwen3_5_moe_text``. The first safe post-MVP
step is import/config/block-contract support for that text tower only — no live
server or demo promotion claim.
"""

import subprocess
import sys

import pytest
import torch

qwen3_5_moe = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    reason="transformers qwen3_5_moe implementation required",
)

from bloombee.utils.auto_config import _CLASS_MAPPING


def _make_text_config():
    from bloombee.models.qwen3_5_moe.config import DistributedQwen3_5MoeTextConfig

    cfg = DistributedQwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=1024,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        max_position_embeddings=64,
        rope_scaling={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        rms_norm_eps=1e-6,
        attn_implementation="eager",
        initializer_range=0.02,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_qwen3_5_moe_text_registers_config_and_block_without_serving_claim():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import bloombee.models; "
                "from bloombee.utils.auto_config import _CLASS_MAPPING; "
                "assert 'qwen3_5_moe_text' in _CLASS_MAPPING; "
                "m=_CLASS_MAPPING['qwen3_5_moe_text']; "
                "assert m.model_for_causal_lm is None"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr + probe.stdout

    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock
    from bloombee.models.qwen3_5_moe.config import DistributedQwen3_5MoeTextConfig

    assert "qwen3_5_moe_text" in _CLASS_MAPPING
    mapping = _CLASS_MAPPING["qwen3_5_moe_text"]
    assert mapping.config is DistributedQwen3_5MoeTextConfig
    assert mapping.model_for_causal_lm is None
    assert DistributedQwen3_5MoeTextConfig.block_class is WrappedQwen3_5MoeTextBlock


def test_qwen3_5_moe_text_full_attention_block_prefill_shape_and_kv_contract():
    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock

    torch.manual_seed(0)
    cfg = _make_text_config()
    block = WrappedQwen3_5MoeTextBlock(cfg, layer_idx=3).eval()
    h = torch.randn(1, 4, cfg.hidden_size)

    out, kv = block(h, attention_mask=None, use_cache=True)

    assert out.shape == h.shape
    pk, pv = kv
    assert pk.shape == (cfg.num_key_value_heads, cfg.head_dim, 4)
    assert pv.shape == (cfg.num_key_value_heads, 4, cfg.head_dim)


def test_qwen3_5_moe_text_full_attention_block_decode_advances_one_token():
    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock

    torch.manual_seed(1)
    cfg = _make_text_config()
    block = WrappedQwen3_5MoeTextBlock(cfg, layer_idx=3).eval()
    h = torch.randn(1, 3, cfg.hidden_size)
    _, (pk, pv) = block(h, attention_mask=None, use_cache=True)

    h_next = torch.randn(1, 1, cfg.hidden_size)
    out, (pk2, pv2) = block(h_next, layer_past=(pk, pv), attention_mask=None, use_cache=True)

    assert out.shape == h_next.shape
    assert pk2.shape[-1] == 1
    assert pv2.shape[-2] == 1


def test_qwen3_5_moe_text_linear_attention_no_cache_forward_is_supported():
    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock

    torch.manual_seed(2)
    cfg = _make_text_config()
    block = WrappedQwen3_5MoeTextBlock(cfg, layer_idx=0).eval()
    h = torch.randn(1, 2, cfg.hidden_size)

    out, kv = block(h, attention_mask=None, use_cache=False)

    assert out.shape == h.shape
    assert kv is None


def test_qwen3_5_moe_text_linear_attention_cache_state_roundtrip():
    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock

    torch.manual_seed(3)
    cfg = _make_text_config()
    block = WrappedQwen3_5MoeTextBlock(cfg, layer_idx=0).eval()
    h = torch.randn(1, 2, cfg.hidden_size)

    out, state = block(h, attention_mask=None, use_cache=True)

    assert out.shape == h.shape
    conv_state, recurrent_state = state
    assert conv_state.shape == (1, 8192, cfg.linear_conv_kernel_dim)
    assert recurrent_state.shape == (1, 32, 128, 128)

    h_next = torch.randn(1, 1, cfg.hidden_size)
    out2, state2 = block(h_next, layer_past=state, attention_mask=None, use_cache=True)

    assert out2.shape == h_next.shape
    conv_state2, recurrent_state2 = state2
    assert conv_state2.shape == conv_state.shape
    assert recurrent_state2.shape == recurrent_state.shape
    assert not torch.equal(conv_state2, conv_state)


def test_qwen3_5_moe_text_rotary_buffers_remain_fp32_after_cast():
    from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock

    cfg = _make_text_config()
    block = WrappedQwen3_5MoeTextBlock(cfg, layer_idx=3).eval()
    block.to(torch.float16)

    assert block._rotary_emb.inv_freq.dtype == torch.float32
    original = getattr(block._rotary_emb, "original_inv_freq", None)
    if original is not None:
        assert original.dtype == torch.float32
