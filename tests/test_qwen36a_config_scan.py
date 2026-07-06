import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def _write_qwen36a_config(path: Path) -> Path:
    layer_types = []
    for _ in range(10):
        layer_types.extend(["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    payload = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10]},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qwen36a_exact_config_scan_is_native_target_but_fail_closed(tmp_path):
    from mvp_capabilities.qwen36a_config_scan import build_qwen36a_report

    config_path = _write_qwen36a_config(tmp_path / "config.json")

    report = build_qwen36a_report(config_path)

    assert report["claim_boundary"] == "post_mvp_qwen36a_exact_config_scan_no_runtime_proof_no_demo_promotion"
    assert report["model_id"] == MODEL_ID
    assert report["native_bloombee_distributed_path_target"] is True
    assert report["exact_config_scan"] == "passed"
    assert report["config_scan"]["hf_model_type"] == "qwen3_5_moe"
    assert report["config_scan"]["hf_text_model_type"] == "qwen3_5_moe_text"
    assert report["config_scan"]["architecture_supported"] is False
    assert report["config_scan"]["claim_level"] == "blocked"
    assert report["layer_type_counts"] == {"linear_attention": 30, "full_attention": 10}
    assert report["qwen35b_family_match"] == "qwen3_5_moe_text_linear_attention_family"
    assert report["wrapper_code_written_for_exact_model"] is False
    assert report["one_block_server_proven"] is False
    assert report["can_update_route_status"] is False
    assert report["can_update_demo_status"] is False
    assert report["can_update_mvp_status"] is False
    assert any("linear-state cache" in reason for reason in report["blocked_reasons"])
    assert report["recommended_next_step"] == "map_qwen36a_to_qwen3_5_moe_text_backend_state_cache_then_one_block_proof"


def test_qwen36a_config_scan_cli_writes_report(tmp_path):
    config_path = _write_qwen36a_config(tmp_path / "config.json")
    out_path = tmp_path / "qwen36a-report.json"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "mvp_capabilities/qwen36a_config_scan.py"),
            "--config",
            str(config_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["model_id"] == MODEL_ID
    assert payload["layer_type_counts"]["linear_attention"] == 30
    assert payload["layer_type_counts"]["full_attention"] == 10
    assert payload["config_scan"]["num_layers"] == 40
    assert payload["config_scan"]["hidden_size"] == 2048
