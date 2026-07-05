import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPIKE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/stretch/layerexecutor-feasibility-20260704.json"


def _spike_payload() -> dict:
    return json.loads(SPIKE_PATH.read_text(encoding="utf-8"))


def test_frontier_backend_smoke_plan_prefers_quantized_deepseek_flash_without_route_claim():
    from mvp_capabilities.frontier_backend_smoke_plan import build_frontier_backend_smoke_plan

    plan = build_frontier_backend_smoke_plan(
        _spike_payload(),
        preferred_targets=["deepseek-ai/DeepSeek-V4-Flash", "zai-org/GLM-5.2", "MiniMaxAI/MiniMax-M3"],
    )

    assert plan["claim_boundary"] == "frontier_external_runtime_smoke_plan_only_no_bloombee_route_claim"
    assert plan["selected_target"]["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert plan["selected_target"]["quantization_method"] == "fp8"
    assert plan["selected_target"]["hf_model_type"] == "deepseek_v4"
    assert plan["native_bloombee_support_proven"] is False
    assert plan["external_runtime_smoke_proven"] is False
    assert plan["can_update_route_status"] is False
    assert plan["can_update_demo_status"] is False
    assert "No BloomBee block wrapper registered for model_type=deepseek_v4" in plan["selected_target"]["blocked_reasons"]
    assert any("vllm serve deepseek-ai/DeepSeek-V4-Flash" in command for command in plan["smoke_commands"])
    assert any("--quantization fp8" in command for command in plan["smoke_commands"])
    assert "nvidia_gpu_required" in plan["required_hardware"]


def test_frontier_backend_smoke_plan_can_emit_minimax_but_marks_native_blocked():
    from mvp_capabilities.frontier_backend_smoke_plan import build_frontier_backend_smoke_plan

    plan = build_frontier_backend_smoke_plan(
        _spike_payload(),
        preferred_targets=["MiniMaxAI/MiniMax-M3"],
    )

    assert plan["selected_target"]["model_id"] == "MiniMaxAI/MiniMax-M3"
    assert plan["selected_target"]["quantization_method"] is None
    assert plan["status"] == "external_runtime_smoke_planned_native_bloombee_blocked"
    assert plan["native_bloombee_support_proven"] is False
    assert "minimax_m3_vl" in " ".join(plan["selected_target"]["blocked_reasons"])
    assert any("MiniMaxAI/MiniMax-M3" in command for command in plan["smoke_commands"])


def test_frontier_backend_smoke_plan_cli_writes_json(tmp_path: Path):
    out_path = tmp_path / "frontier-smoke-plan.json"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/frontier_backend_smoke_plan.py",
            "--spike-artifact",
            str(SPIKE_PATH),
            "--target",
            "deepseek-ai/DeepSeek-V4-Flash",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selected_target"]["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert payload["external_runtime_smoke_proven"] is False
