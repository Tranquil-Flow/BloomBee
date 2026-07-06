import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B"
CONFIG_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/minimax-m27-reap-config-20260706.raw.json"


def test_minimax_m27_reap_exact_config_native_contract_scan_fails_closed():
    from mvp_capabilities.minimax_m27_native_contract_scan import build_minimax_m27_native_contract_report

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    report = build_minimax_m27_native_contract_report(config, model_id=MODEL_ID)

    assert report["claim_boundary"] == "minimax_m27_reap_native_contract_scan_no_live_inference"
    assert report["model_id"] == MODEL_ID
    assert report["native_bloombee_target"] is True
    assert report["model_type"] == "minimax_m2"
    assert report["architecture"] == "MiniMaxM2ForCausalLM"
    assert report["num_hidden_layers"] == 62
    assert report["hidden_size"] == 3072
    assert report["num_attention_heads"] == 48
    assert report["num_key_value_heads"] == 8
    assert report["head_dim"] == 128
    assert report["num_local_experts"] == 154
    assert report["num_experts_per_tok"] == 8
    assert report["attn_type_counts"] == {"1": 62}
    assert report["exact_config_has_sparse_attention_flag"] is False
    assert report["use_mtp"] is True
    assert report["num_mtp_modules"] == 3
    assert report["use_qk_norm"] is True
    assert report["use_routing_bias"] is True
    assert report["scoring_func"] == "sigmoid"
    assert report["native_wrapper_package_present"] is True
    assert report["mtp_base_decoder_guard_available"] is True
    assert report["state_cache_contract"]["attention_cache_kind"] == "dynamic_kv_per_layer"
    assert report["state_cache_contract"]["kv_layers"] == 62
    assert report["state_cache_contract"]["moe_router_top_k"] == 8
    assert report["native_bloombee_support_proven"] is False
    assert report["live_run_attempted"] is False
    assert report["one_block_server_proven"] is False
    assert report["can_update_proof_status"] is False
    assert "bloombee_minimax_m2_wrapper_missing" not in report["remaining_blockers"]
    assert "minimax_m2_moe_router_real_weight_proof_missing" in report["remaining_blockers"]
    assert "minimax_m2_mtp_contract_unimplemented" not in report["remaining_blockers"]
    assert "minimax_m2_mtp_real_weight_or_full_module_proof_missing" in report["remaining_blockers"]


def test_minimax_m27_native_contract_scan_cli_writes_json(tmp_path: Path):
    out = tmp_path / "minimax-m27-native-contract.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/minimax_m27_native_contract_scan.py",
            "--config",
            str(CONFIG_PATH),
            "--out",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == json.loads(proc.stdout)
    assert payload["model_id"] == MODEL_ID
    assert payload["native_bloombee_support_proven"] is False
