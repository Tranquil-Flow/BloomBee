#!/usr/bin/env python3
"""Plan a narrow frontier-model external-runtime smoke without route claims.

MiniMax/GLM/DeepSeek/Kimi frontier models are not native BloomBee-runnable today.
This planner turns the conservative LayerExecutor feasibility artifact into an
operator checklist for a future external-runtime smoke (for example vLLM/SGLang
on suitable NVIDIA hardware) while keeping all BloomBee route/demo flags false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

CLAIM_BOUNDARY = "frontier_external_runtime_smoke_plan_only_no_bloombee_route_claim"
SOURCE = "frontier_backend_smoke_plan.py"
DEFAULT_TARGET_ORDER = (
    "deepseek-ai/DeepSeek-V4-Flash",
    "zai-org/GLM-5.2",
    "MiniMaxAI/MiniMax-M3",
    "moonshotai/Kimi-K2-Instruct",
)


def _targets_by_id(spike_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(target.get("model_id")): target
        for target in spike_payload.get("target_models", [])
        if target.get("model_id")
    }


def _quantization_method(target: dict[str, Any]) -> str | None:
    facts = target.get("config_facts") or {}
    value = facts.get("quantization_method")
    return str(value) if value else None


def _smoke_commands(model_id: str, quantization: str | None) -> list[str]:
    commands = [
        "# Run only on a suitable NVIDIA GPU host with explicit operator approval; this is not a macOS/MPS proof.",
    ]
    if quantization == "fp8":
        commands.append(
            f"vllm serve {model_id} --trust-remote-code --quantization fp8 --max-model-len 4096 --port 8000"
        )
    else:
        commands.append(
            f"vllm serve {model_id} --trust-remote-code --max-model-len 4096 --port 8000"
        )
    commands.extend(
        [
            "curl -s http://127.0.0.1:8000/v1/models | python -m json.tool",
            "curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say one moonlit sentence.\"}],\"max_tokens\":16}' | python -m json.tool",
        ]
    )
    return [command.replace("MODEL_ID", model_id) for command in commands]


def build_frontier_backend_smoke_plan(
    spike_payload: dict[str, Any],
    *,
    preferred_targets: Iterable[str] = DEFAULT_TARGET_ORDER,
) -> dict[str, Any]:
    targets = _targets_by_id(spike_payload)
    selected: dict[str, Any] | None = None
    missing: list[str] = []
    for model_id in preferred_targets:
        target = targets.get(model_id)
        if target is None:
            missing.append(str(model_id))
            continue
        selected = target
        break
    if selected is None:
        raise ValueError(f"none of the preferred targets exist in the spike artifact: {missing}")

    model_id = str(selected["model_id"])
    quantization = _quantization_method(selected)
    blocked_reasons = [str(reason) for reason in selected.get("blocked_reasons") or []]
    config_facts = selected.get("config_facts") or {}
    selected_target = {
        "model_id": model_id,
        "hf_model_type": selected.get("hf_model_type"),
        "hf_text_model_type": selected.get("hf_text_model_type"),
        "claim_level": selected.get("claim_level", "blocked"),
        "quantization_method": quantization,
        "hidden_size": config_facts.get("hidden_size"),
        "num_layers": config_facts.get("num_layers"),
        "num_experts": config_facts.get("num_experts"),
        "experts_per_token": config_facts.get("experts_per_token"),
        "blocked_reasons": blocked_reasons,
        "minimal_proof": selected.get("minimal_proof"),
    }
    return {
        "source": SOURCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "external_runtime_smoke_planned_native_bloombee_blocked",
        "selected_target": selected_target,
        "preferred_targets": list(preferred_targets),
        "missing_preferred_targets": missing,
        "required_hardware": [
            "nvidia_gpu_required",
            "external_runtime_required",
            "operator_supplied_credentials_if_model_is_gated",
        ],
        "smoke_commands": _smoke_commands(model_id, quantization),
        "native_bloombee_support_proven": False,
        "external_runtime_smoke_proven": False,
        "route_picker_eligible": False,
        "can_update_route_status": False,
        "can_update_demo_status": False,
        "mvp_core_status_unchanged": True,
        "operator_next_steps": [
            "run the smoke commands only on an approved NVIDIA GPU host with a runtime that supports the target quantization/attention contract",
            "save request/response/log evidence as a separate external_runtime_smoke artifact",
            "design a LayerExecutor adapter boundary only after external runtime smoke passes",
            "do not mark native BloomBee support, route eligibility, or demo safety from this plan alone",
        ],
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-artifact", required=True)
    parser.add_argument("--target", action="append", default=[], help="Preferred target model ID; repeat for fallback order")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = build_frontier_backend_smoke_plan(
        _read_json(args.spike_artifact),
        preferred_targets=args.target or DEFAULT_TARGET_ORDER,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
