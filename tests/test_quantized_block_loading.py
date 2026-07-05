"""Server-side quantized loading for HF blocks (handover Task 2).

Contract under test (no network, tiny configs):

- ``QuantType.INT8`` + dense HF block  -> optimum-quanto qint8 weights
- ``QuantType.INT8`` + qwen3_moe block -> custom int8 fused-expert swap plus
  quanto qint8 on remaining Linears, router ``mlp.gate`` left fp16 so routing
  decisions stay exact
- ``QuantType.NF4``  + qwen3_moe block -> packed int4 fused-expert swap plus
  quanto qint8 on remaining Linears (experts dominate bytes; attention stays
  int8 to keep the serving path free of quanto's JIT-built qint4 extension)
- ``QuantType.NF4``  + dense HF block  -> fail-closed (blocked until a
  deterministic dense int4 path exists)
- FlexGen-native blocks (meta params)  -> untouched by this lane
- ``get_block_size(..., quant_type=...)`` -> compressed byte estimates
"""
import pytest
import torch

from bloombee.utils.convert_block import QuantType, convert_block, quantize_hf_block
from bloombee.utils.moe_expert_quant import (
    QuantizedQwen3MoeExperts,
    QuantizedQwen3MoeExpertsInt4,
)

pytest.importorskip("optimum.quanto", reason="optimum-quanto required for the quantized loading lane")
qwen3_moe = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
    reason="transformers qwen3_moe implementation required",
)


def _state_dict_bytes(module: torch.nn.Module) -> int:
    """quanto freeze() packs weights into QTensor subclasses; walking the
    state_dict is the honest byte count (scales live in separate tensors)."""
    total = 0
    for value in module.state_dict().values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _moe_config():
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dense_config():
    from transformers.models.qwen3 import Qwen3Config

    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _init_(block: torch.nn.Module, seed: int) -> torch.nn.Module:
    # Standalone HF layer construction uses torch.empty and never initializes;
    # init explicitly so byte/parity math is meaningful (known gotcha).
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in block.parameters():
            torch.nn.init.normal_(param, std=0.02)
    return block.to(torch.float16).eval()


def _moe_layer(seed: int = 61):
    return _init_(qwen3_moe.Qwen3MoeDecoderLayer(_moe_config(), layer_idx=0), seed)


def _dense_layer(seed: int = 67):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    return _init_(Qwen3DecoderLayer(_dense_config(), layer_idx=0), seed)


def _forward(block, config, seed: int = 71):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

    torch.manual_seed(seed)
    seq_len = 4
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float16) * 0.5
    rotary = Qwen3MoeRotaryEmbedding(config)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)
    with torch.no_grad():
        out = block(hidden, attention_mask=None, position_embeddings=position_embeddings)
    return out[0] if isinstance(out, tuple) else out


def test_int8_moe_block_swaps_experts_quantizes_linears_and_spares_router():
    from optimum.quanto.nn import QLinear

    block = _moe_layer()
    before = _state_dict_bytes(block)
    stats = quantize_hf_block(block, quant_type=QuantType.INT8, model_type="qwen3_moe")
    after = _state_dict_bytes(block)

    assert isinstance(block.mlp.experts, QuantizedQwen3MoeExperts)
    assert stats["moe_expert_swap"]["swapped_modules"] == ["mlp.experts"]
    # attention linears got quanto qint8
    assert isinstance(block.self_attn.q_proj, QLinear)
    # the router decides expert selection; it must stay unquantized fp16
    # (transformers 5.x uses Qwen3MoeTopKRouter, older versions a plain
    # nn.Linear — either way it must not become a quanto QLinear)
    assert not isinstance(block.mlp.gate, QLinear)
    assert all(p.dtype == torch.float16 for p in block.mlp.gate.parameters())
    assert before / after >= 1.7
    assert stats["compression_ratio"] >= 1.7

    out = _forward(block, _moe_config())
    assert torch.isfinite(out).all()


def test_nf4_moe_block_uses_packed_int4_experts():
    block_int8 = _moe_layer(seed=73)
    block_nf4 = _moe_layer(seed=73)
    quantize_hf_block(block_int8, quant_type=QuantType.INT8, model_type="qwen3_moe")
    stats = quantize_hf_block(block_nf4, quant_type=QuantType.NF4, model_type="qwen3_moe")

    assert isinstance(block_nf4.mlp.experts, QuantizedQwen3MoeExpertsInt4)
    assert "int4" in stats["moe_expert_swap"]["scheme"]
    assert _state_dict_bytes(block_nf4) < _state_dict_bytes(block_int8)

    out = _forward(block_nf4, _moe_config())
    assert torch.isfinite(out).all()


def test_int8_dense_block_quantizes_all_linears():
    from optimum.quanto.nn import QLinear

    block = _dense_layer()
    before = _state_dict_bytes(block)
    stats = quantize_hf_block(block, quant_type=QuantType.INT8, model_type="qwen3")
    after = _state_dict_bytes(block)

    assert any(isinstance(m, QLinear) for m in block.modules())
    assert stats["moe_expert_swap"] is None
    assert before / after >= 1.7

    out = _forward(block, _dense_config())
    assert torch.isfinite(out).all()


def test_nf4_dense_block_fails_closed():
    block = _dense_layer(seed=79)
    with pytest.raises(NotImplementedError, match="NF4"):
        quantize_hf_block(block, quant_type=QuantType.NF4, model_type="qwen3")


def test_quant_none_is_a_recorded_noop():
    block = _dense_layer(seed=83)
    before = _state_dict_bytes(block)
    stats = quantize_hf_block(block, quant_type=QuantType.NONE, model_type="qwen3")
    assert stats["applied"] is False
    assert _state_dict_bytes(block) == before


def test_convert_block_applies_int8_to_hf_moe_block():
    device = torch.device("cpu")
    config = _moe_config()
    block = _moe_layer(seed=89)
    wrapped = convert_block(
        block,
        0,
        config,
        [device],
        device,
        QuantType.INT8,
        freeze=True,
    )
    inner = wrapped.module_shards[0]
    assert isinstance(inner.mlp.experts, QuantizedQwen3MoeExperts)


def test_convert_block_quant_plus_cpu_offload_fails_closed():
    class _Policy:
        w_gpu_percent = 50
        w_cpu_percent = 50
        w_disk_percent = 0
        pin_weight = False

    device = torch.device("cpu")
    block = _moe_layer(seed=97)
    with pytest.raises(ValueError, match="offload"):
        convert_block(
            block,
            0,
            _moe_config(),
            [device],
            device,
            QuantType.INT8,
            freeze=True,
            policy=_Policy(),
        )


def test_get_block_size_returns_compressed_bytes_for_quantized_modes():
    from bloombee.models.qwen3_moe.config import DistributedQwen3MoeConfig
    from bloombee.server.block_utils import get_block_size

    cfg = DistributedQwen3MoeConfig(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
    )
    cfg._attn_implementation = "eager"

    fp16 = get_block_size(cfg, "memory", None, None, dtype=torch.float16, quant_type=QuantType.NONE)
    int8 = get_block_size(cfg, "memory", None, None, dtype=torch.float16, quant_type=QuantType.INT8)
    nf4 = get_block_size(cfg, "memory", None, None, dtype=torch.float16, quant_type=QuantType.NF4)

    # int8 roughly halves; int4 experts shrink further; both keep norms/router fp16
    assert 0.40 * fp16 < int8 < 0.65 * fp16
    assert nf4 < int8


def test_get_block_size_nf4_dense_fails_closed():
    from bloombee.models.qwen3.config import DistributedQwen3Config
    from bloombee.server.block_utils import get_block_size

    cfg = DistributedQwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    cfg._attn_implementation = "eager"

    with pytest.raises(NotImplementedError, match="NF4"):
        get_block_size(cfg, "memory", None, None, dtype=torch.float16, quant_type=QuantType.NF4)
