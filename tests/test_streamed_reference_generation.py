from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "streamed_reference_generation.py"


def _load_streamed_reference_module():
    spec = importlib.util.spec_from_file_location("streamed_reference_generation_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tiny_qwen3_dense_config():
    from transformers.models.qwen3 import Qwen3Config

    return Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        initializer_range=0.02,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _tiny_qwen3_moe_config():
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        initializer_range=0.02,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _make_stable_tiny_qwen3_moe_model():
    """Create a tiny Qwen3-MoE fixture with finite deterministic weights.

    This test covers streamed BloomBee block loading/execution, not HF random
    tiny-MoE initialization. Full-suite order exposed all-NaN logits from raw
    and small-random tiny-MoE fixtures, so remove random router/expert numerics:
    norm weights stay 1.0 and all other floating weights are zero.
    """
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM(_tiny_qwen3_moe_config()).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.dtype.is_floating_point:
                continue
            param.zero_()
            if name.endswith("norm.weight"):
                param.fill_(1.0)
    return model


@torch.inference_mode()
def _full_model_greedy_ids(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    output = input_ids.detach().cpu().clone().to(torch.long)
    for _ in range(max_new_tokens):
        logits = model(output).logits[:, -1, :].detach().float().cpu()
        output = torch.cat([output, logits.argmax(dim=-1, keepdim=True).to(torch.long)], dim=1)
    return output


def test_full_generation_plan_can_emit_streamed_reference_route_command():
    from mvp_capabilities.full_generation_proof import build_full_generation_plan

    plan = build_full_generation_plan(
        model_id="Qwen/Qwen3-30B-A3B@int8",
        checkpoint_model="Qwen/Qwen3-30B-A3B",
        server_maddrs=["/ip4/192.168.178.37/tcp/31347/p2p/server"],
        server_placements=["m4pro-full=0:48"],
        prompt="The moon is",
        max_new_tokens=1,
        mode="forward-loop",
        reference_mode="streamed-blocks",
        reference_cache_dir="/Volumes/Seagate Portable Drive/huggingface/hub",
        reference_local_files_only=True,
        evidence_path="mvp_capabilities/distributed_evidence/post_mvp/qwen30b-int8-full-generation.json",
    )

    command = plan["parity_command"]
    assert plan["model_id"] == "Qwen/Qwen3-30B-A3B@int8"
    assert plan["checkpoint_model"] == "Qwen/Qwen3-30B-A3B"
    assert plan["reference_mode"] == "streamed-blocks"
    assert "--model Qwen/Qwen3-30B-A3B@int8" in command
    assert "--checkpoint-model 'Qwen/Qwen3-30B-A3B'" in command
    assert "--reference-mode streamed-blocks" in command
    assert "--reference-cache-dir '/Volumes/Seagate Portable Drive/huggingface/hub'" in command
    assert "--reference-local-files-only" in command
    assert "--server-placement 'm4pro-full=0:48'" in command


def test_checkpoint_model_for_route_strips_known_quantized_suffixes():
    streamed = _load_streamed_reference_module()

    assert streamed.checkpoint_model_for_route("Qwen/Qwen3-30B-A3B@int8") == "Qwen/Qwen3-30B-A3B"
    assert streamed.checkpoint_model_for_route("Qwen/Qwen3-30B-A3B@nf4") == "Qwen/Qwen3-30B-A3B"
    assert streamed.checkpoint_model_for_route("route@int8", "checkpoint") == "checkpoint"
    assert streamed.checkpoint_model_for_route("TinyLlama/TinyLlama-1.1B-Chat-v1.0") == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_streamed_qwen3_dense_reference_matches_full_model_greedy_ids(tmp_path: Path):
    from transformers.models.qwen3 import Qwen3ForCausalLM

    streamed = _load_streamed_reference_module()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_tiny_qwen3_dense_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    expected = _full_model_greedy_ids(model, input_ids, max_new_tokens=1)

    actual, steps = streamed.streamed_greedy_generate_ids(
        str(tmp_path),
        input_ids,
        max_new_tokens=1,
        dtype=torch.float32,
        device="cpu",
        local_files_only=True,
    )

    assert actual.tolist() == expected.tolist()
    assert [step["block_count"] for step in steps] == [2]
    assert [step["next_token_id"] for step in steps] == [int(expected[0, input_ids.shape[1]])]


def test_streamed_qwen3_moe_local_loader_packs_split_expert_weights(tmp_path: Path):
    streamed = _load_streamed_reference_module()
    model = _make_stable_tiny_qwen3_moe_model()
    with torch.no_grad():
        source_experts = model.model.layers[0].mlp.experts
        for expert_idx in range(source_experts.num_experts):
            source_experts.gate_up_proj[expert_idx].fill_(0.125 * (expert_idx + 1))
            source_experts.down_proj[expert_idx].fill_(0.25 * (expert_idx + 1))
        expected_gate_up = source_experts.gate_up_proj.detach().clone()
        expected_down = source_experts.down_proj.detach().clone()
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = streamed.AutoDistributedConfig.from_pretrained(str(tmp_path), local_files_only=True)

    block = streamed._load_streamed_block(
        str(tmp_path),
        0,
        config=config,
        dtype=torch.float32,
        cache_dir=None,
        local_files_only=True,
    )

    experts = block.mlp.experts
    torch.testing.assert_close(experts.gate_up_proj, expected_gate_up)
    torch.testing.assert_close(experts.down_proj, expected_down)


def test_streamed_qwen3_moe_reference_emits_finite_bloombee_block_trace(tmp_path: Path):
    streamed = _load_streamed_reference_module()
    model = _make_stable_tiny_qwen3_moe_model()
    model.save_pretrained(tmp_path, safe_serialization=True)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits, timings = streamed.streamed_forward_logits(
        str(tmp_path),
        input_ids,
        dtype=torch.float32,
        device="cpu",
        local_files_only=True,
        record_block_timings=True,
    )

    assert torch.isfinite(logits).all()
    assert logits.shape == (1, _tiny_qwen3_moe_config().vocab_size)
    assert len(timings) == 1
    assert timings[0]["layer_idx"] == 0
