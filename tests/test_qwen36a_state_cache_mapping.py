import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/qwen36a/qwen36a-config-20260706.raw.json"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def test_qwen36a_exact_config_maps_to_native_state_cache_descriptors():
    from mvp_capabilities.qwen36a_state_cache_mapping import build_qwen36a_state_cache_mapping_report

    report = build_qwen36a_state_cache_mapping_report(CONFIG_PATH)

    assert report["claim_boundary"] == "post_mvp_qwen36a_state_cache_mapping_no_server_proof_no_demo_promotion"
    assert report["model_id"] == MODEL_ID
    assert report["native_bloombee_distributed_path_target"] is True
    assert report["text_config_loaded"] is True
    assert report["text_model_type"] == "qwen3_5_moe_text"
    assert report["mapping_status"] == "passed_descriptor_contract_no_live_server"
    assert report["linear_attention_descriptor_contract"]["block_index"] == 0
    assert report["linear_attention_descriptor_contract"]["layer_type"] == "linear_attention"
    assert report["linear_attention_descriptor_contract"]["descriptors"] == [
        {"type": "LinearStateTensorDescriptor", "kind": "qwen3_5_linear_conv", "shape": [2, 8192, 4]},
        {"type": "LinearStateTensorDescriptor", "kind": "qwen3_5_linear_recurrent", "shape": [2, 32, 128, 128]},
    ]
    assert report["full_attention_descriptor_contract"]["block_index"] == 3
    assert report["full_attention_descriptor_contract"]["layer_type"] == "full_attention"
    assert report["full_attention_descriptor_contract"]["descriptors"] == [
        {"type": "TensorDescriptor", "kind": None, "shape": [2, 16, 256, 16]}
    ]
    assert report["wrapper_code_written_for_exact_model"] is False
    assert report["one_block_server_proven"] is False
    assert report["can_update_route_status"] is False
    assert report["can_update_demo_status"] is False
    assert report["recommended_next_step"] == "run_exact_qwen36a_one_block_server_proof_on_suitable_memory"


def test_qwen36a_state_cache_mapping_cli_writes_report(tmp_path):
    out_path = tmp_path / "mapping.json"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "mvp_capabilities/qwen36a_state_cache_mapping.py"),
            "--config",
            str(CONFIG_PATH),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mapping_status"] == "passed_descriptor_contract_no_live_server"
    assert payload["proof_status_update"] == {}
    assert payload["one_block_server_proven"] is False
