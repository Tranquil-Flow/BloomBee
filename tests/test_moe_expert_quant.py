"""Default-suite tests for int8 fused-expert quantization (no network, no swarm).

Uses a tiny random-weight Qwen3-MoE config so the suite stays fast. Real-dim
memory evidence lives in the quantized-block spike artifact under
mvp_capabilities/distributed_evidence/stretch/.
"""
import pytest
import torch

from bloombee.utils.moe_expert_quant import (
    QuantizedQwen3MoeExperts,
    QuantizedQwen3MoeExpertsInt4,
    dequant_group_int4,
    module_weight_bytes,
    quantize_group_int4,
    quantize_per_out_channel_int8,
    quantize_qwen3_moe_block_experts,
)

qwen3_moe = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
    reason="transformers qwen3_moe implementation required",
)


def _tiny_config(**overrides):
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    kwargs = dict(
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
    kwargs.update(overrides)
    return Qwen3MoeConfig(**kwargs)


def _tiny_sparse_block(seed: int = 7, **config_overrides):
    torch.manual_seed(seed)
    block = qwen3_moe.Qwen3MoeSparseMoeBlock(_tiny_config(**config_overrides))
    # Qwen3MoeExperts allocates with torch.empty and relies on the full model's
    # weight init; standalone construction leaves uninitialized (possibly NaN)
    # memory, so initialize explicitly.
    with torch.no_grad():
        for param in block.parameters():
            torch.nn.init.normal_(param, std=0.02)
    return block.to(torch.float16).eval()


def test_per_out_channel_int8_roundtrip_error_bounded():
    torch.manual_seed(0)
    weight = torch.randn(4, 6, 32, dtype=torch.float16)
    q, scale = quantize_per_out_channel_int8(weight)
    assert q.dtype == torch.int8 and q.shape == weight.shape
    assert scale.dtype == torch.float16 and scale.shape == (4, 6, 1)
    recon = q.to(torch.float16) * scale
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
    # symmetric int8 rounding error is at most half a quantization step
    assert torch.all((weight - recon).abs() <= max_abs / 127.0)


def test_quantized_experts_parity_and_routing_match_reference():
    block = _tiny_sparse_block()
    torch.manual_seed(11)
    hidden = torch.randn(1, 5, 64, dtype=torch.float16) * 0.5

    with torch.no_grad():
        reference = block(hidden)

    quantized_experts = QuantizedQwen3MoeExperts.from_experts(block.experts)
    block.experts = quantized_experts
    with torch.no_grad():
        candidate = block(hidden)

    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0)
    assert torch.isfinite(cand).all()
    assert float(cosine) > 0.999
    assert float((ref - cand).abs().max()) < 0.05


def test_backward_to_input_grad_finite_through_quantized_experts():
    block = _tiny_sparse_block(seed=13)
    block.experts = QuantizedQwen3MoeExperts.from_experts(block.experts)
    hidden = (torch.randn(1, 4, 64, dtype=torch.float16) * 0.5).requires_grad_(True)
    out = block(hidden)
    out.float().sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_block_swap_helper_reports_compression():
    block = _tiny_sparse_block(seed=17)
    before = module_weight_bytes(block)
    stats = quantize_qwen3_moe_block_experts(block)
    assert stats["swapped_modules"] == ["experts"]
    assert stats["weight_bytes_before"] == before
    assert stats["weight_bytes_after"] < before
    # experts dominate this tiny block less than a real 30B block, but int8
    # must still roughly halve the expert bytes
    assert stats["compression_ratio"] > 1.4
    assert isinstance(block.experts, QuantizedQwen3MoeExperts)


def test_block_swap_helper_fails_closed_on_non_moe_module():
    dense = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(ValueError, match="refusing to report a no-op"):
        quantize_qwen3_moe_block_experts(dense)


def test_quantized_experts_state_dict_roundtrip():
    block = _tiny_sparse_block(seed=19)
    quantized = QuantizedQwen3MoeExperts.from_experts(block.experts)
    state = quantized.state_dict()
    assert state["gate_up_proj_q"].dtype == torch.int8
    assert state["down_proj_q"].dtype == torch.int8

    clone = QuantizedQwen3MoeExperts(
        num_experts=quantized.num_experts,
        hidden_dim=quantized.hidden_dim,
        intermediate_dim=quantized.intermediate_dim,
        act_fn=quantized.act_fn,
        gate_up_q=torch.zeros_like(state["gate_up_proj_q"]),
        gate_up_scale=torch.zeros_like(state["gate_up_proj_scale"]),
        down_q=torch.zeros_like(state["down_proj_q"]),
        down_scale=torch.zeros_like(state["down_proj_scale"]),
    )
    clone.load_state_dict(state)
    torch.manual_seed(23)
    hidden = torch.randn(6, 64, dtype=torch.float16)
    top_k = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), 0.5, dtype=torch.float16)
    with torch.no_grad():
        a = quantized(hidden, top_k, weights)
        b = clone(hidden, top_k, weights)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Packed int4 experts (the NF4 serving lane)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (4, 6, 128),  # exact single group
        (2, 3, 100),  # partial last group, even length
        (2, 3, 129),  # partial last group, odd length (nibble pad)
        (5, 260),     # 2D, two full groups + partial
    ],
)
def test_group_int4_roundtrip_error_bounded(shape):
    torch.manual_seed(0)
    group_size = 128
    weight = torch.randn(*shape, dtype=torch.float16)
    packed, scale = quantize_group_int4(weight, group_size=group_size)

    in_dim = shape[-1]
    n_groups = -(-in_dim // group_size)
    assert packed.dtype == torch.uint8
    assert packed.shape == (*shape[:-1], (in_dim + 1) // 2)
    assert scale.dtype == torch.float16
    assert scale.shape == (*shape[:-1], n_groups)

    recon = dequant_group_int4(packed, scale, in_dim, torch.float16, group_size=group_size)
    assert recon.shape == weight.shape
    # per-group symmetric int4 bound from the handover spec:
    # |W - dequant(pack(W))| <= group_max_abs / 7 elementwise
    for g in range(n_groups):
        sl = slice(g * group_size, min((g + 1) * group_size, in_dim))
        w_g = weight[..., sl]
        err = (w_g - recon[..., sl]).abs()
        bound = w_g.abs().amax(dim=-1, keepdim=True) / 7.0
        assert torch.all(err <= bound + 1e-6)


def test_int4_expert_compression_at_least_3_5x():
    # Group-128 scales need input dims >= group size to amortize; use dims in
    # that regime (gate_up in=256, down in=128), still tiny enough for CI.
    block = _tiny_sparse_block(
        seed=29, hidden_size=256, moe_intermediate_size=128, num_experts=4
    )
    experts_before = module_weight_bytes(block.experts)
    quantized = QuantizedQwen3MoeExpertsInt4.from_experts(block.experts)
    experts_after = module_weight_bytes(quantized)
    assert experts_before / experts_after >= 3.5


def test_int4_experts_parity_and_routing_match_reference():
    import copy

    block = _tiny_sparse_block(seed=31)
    torch.manual_seed(37)
    hidden = torch.randn(1, 5, 64, dtype=torch.float16) * 0.5

    with torch.no_grad():
        reference = block(hidden)

    # The strong parity property: the packed forward must be BIT-EXACT with
    # running the reference module on dequant(pack(w)) weights. Any packing,
    # scale-layout, or routing bug breaks this; inherent int4 rounding noise
    # does not (it affects both sides equally).
    deq_block = copy.deepcopy(block)
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(deq_block.experts, name).data
        packed, scale = quantize_group_int4(w)
        getattr(deq_block.experts, name).data = dequant_group_int4(
            packed, scale, w.shape[-1], w.dtype
        )

    block.experts = QuantizedQwen3MoeExpertsInt4.from_experts(block.experts)
    with torch.no_grad():
        dequant_reference = deq_block(hidden)
        candidate = block(hidden)

    assert torch.isfinite(candidate).all()
    assert torch.equal(candidate, dequant_reference)

    # Loose sanity vs fp16: int4 rounding at the residual-free sparse-MoE
    # module level on tiny random weights is inherently noisy (~0.985 cosine
    # here); whole-decoder-layer parity with residual is the real quality
    # gate and lives in the proof ladder (handover Task 4/5).
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0)
    assert float(cosine) > 0.97


def test_int4_backward_to_input_grad_finite():
    block = _tiny_sparse_block(seed=41)
    block.experts = QuantizedQwen3MoeExpertsInt4.from_experts(block.experts)
    hidden = (torch.randn(1, 4, 64, dtype=torch.float16) * 0.5).requires_grad_(True)
    out = block(hidden)
    out.float().sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_block_swap_helper_int4_scheme_and_fail_closed():
    block_int8 = _tiny_sparse_block(seed=43)
    block_int4 = _tiny_sparse_block(seed=43)

    stats_int8 = quantize_qwen3_moe_block_experts(block_int8)
    stats_int4 = quantize_qwen3_moe_block_experts(block_int4, scheme="int4")

    assert isinstance(block_int4.experts, QuantizedQwen3MoeExpertsInt4)
    assert stats_int4["scheme"] == "int4_packed_symmetric_group128"
    assert stats_int4["weight_bytes_after"] < stats_int8["weight_bytes_after"]
    assert stats_int4["compression_ratio"] > stats_int8["compression_ratio"]

    with pytest.raises(ValueError, match="unknown quantization scheme"):
        quantize_qwen3_moe_block_experts(_tiny_sparse_block(seed=43), scheme="int2")


def test_int4_state_dict_roundtrip():
    block = _tiny_sparse_block(seed=47)
    quantized = QuantizedQwen3MoeExpertsInt4.from_experts(block.experts)
    state = quantized.state_dict()
    assert state["gate_up_proj_packed"].dtype == torch.uint8
    assert state["down_proj_packed"].dtype == torch.uint8
    assert state["gate_up_proj_scale"].dtype == torch.float16

    clone = QuantizedQwen3MoeExpertsInt4(
        num_experts=quantized.num_experts,
        hidden_dim=quantized.hidden_dim,
        intermediate_dim=quantized.intermediate_dim,
        act_fn=quantized.act_fn,
        group_size=quantized.group_size,
        gate_up_packed=torch.zeros_like(state["gate_up_proj_packed"]),
        gate_up_scale=torch.zeros_like(state["gate_up_proj_scale"]),
        down_packed=torch.zeros_like(state["down_proj_packed"]),
        down_scale=torch.zeros_like(state["down_proj_scale"]),
    )
    clone.load_state_dict(state)
    torch.manual_seed(53)
    hidden = torch.randn(6, 64, dtype=torch.float16)
    top_k = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), 0.5, dtype=torch.float16)
    with torch.no_grad():
        a = quantized(hidden, top_k, weights)
        b = clone(hidden, top_k, weights)
    assert torch.equal(a, b)
