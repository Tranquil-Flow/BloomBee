import pytest
import torch

try:
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2DecoderLayer as _HFDecoderLayer  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("minimax_m2 not available in this transformers", allow_module_level=True)

from bloombee.models.minimax_m2.block import WrappedMiniMaxM2Block
from bloombee.models.minimax_m2.config import DistributedMiniMaxM2Config


def _make_config() -> DistributedMiniMaxM2Config:
    cfg = DistributedMiniMaxM2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        shared_intermediate_size=0,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        rope_theta=5_000_000,
        rotary_dim=16,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        router_jitter_noise=0.0,
        router_aux_loss_coef=0.001,
        qk_norm_type="per_layer",
        scoring_func="sigmoid",
        use_qk_norm=True,
        use_routing_bias=True,
        use_cache=True,
        use_mtp=False,
        tie_word_embeddings=True,
        initializer_range=0.02,
        attn_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_stable_block(cfg: DistributedMiniMaxM2Config, *, layer_idx: int = 0) -> WrappedMiniMaxM2Block:
    block = WrappedMiniMaxM2Block(cfg, layer_idx=layer_idx).eval()
    with torch.no_grad():
        for name, param in block.named_parameters():
            if not param.dtype.is_floating_point:
                continue
            param.zero_()
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                param.fill_(1.0)
    return block


def test_minimax_m2_prefill_shape_and_kv_contract():
    torch.manual_seed(0)
    cfg = _make_config()
    block = _make_stable_block(cfg)
    h = torch.randn(1, 4, cfg.hidden_size)

    out, kv = block(h, use_cache=True)

    assert out.shape == h.shape
    pk, pv = kv
    assert pk.shape == (cfg.num_key_value_heads, cfg.head_dim, 4)
    assert pv.shape == (cfg.num_key_value_heads, 4, cfg.head_dim)


def test_minimax_m2_decode_step_advances_kv_length():
    torch.manual_seed(0)
    cfg = _make_config()
    block = _make_stable_block(cfg)
    _, (pk, pv) = block(torch.randn(1, 5, cfg.hidden_size), use_cache=True)

    _, (pk2, pv2) = block(torch.randn(1, 1, cfg.hidden_size), layer_past=(pk, pv), use_cache=True)

    assert pk2.shape[-1] == 1
    assert pv2.shape[-2] == 1


def test_minimax_m2_rotary_inv_freq_stays_fp32_under_fp16_cast():
    cfg = _make_config()
    block = _make_stable_block(cfg)

    block.to(torch.float16)

    assert block._rotary_emb.inv_freq.dtype == torch.float32
    original = getattr(block._rotary_emb, "original_inv_freq", None)
    if original is not None:
        assert original.dtype == torch.float32


def test_minimax_m2_config_wires_block_class_and_model_type():
    cfg = _make_config()
    assert cfg.model_type == "minimax_m2"
    assert cfg.block_class is WrappedMiniMaxM2Block
    assert cfg.block_prefix == "model.layers"
