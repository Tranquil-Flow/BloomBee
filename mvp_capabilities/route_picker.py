#!/usr/bin/env python3
"""Pick the strongest feasible model for a BloomBee peer swarm.

The router is conservative: it separates solo fit from aggregate swarm fit,
labels block-parallel candidates honestly, and treats measured benchmark data
as stronger evidence than estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, PROOF_KEYS, SUPPORTED_FAMILIES, load_proof_status
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/route_picker.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.model_compat_scan import DEFAULT_PROOF_STATUS, PROOF_KEYS, SUPPORTED_FAMILIES, load_proof_status
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster


DEFAULT_REGISTRY = Path(__file__).with_name("MODEL_REGISTRY.yaml")
MVP_MODEL_ID = "Qwen/Qwen3-30B-A3B"
STRETCH_MODEL_ID = "Qwen/Qwen3-235B-A22B"
SELECTOR_MODES = ("planning", "showcase-attempt", "safe-demo")


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = payload.get("models") or []
    enriched: list[dict[str, Any]] = []
    for rank, model in enumerate(models):
        model = _annotate_architecture_support(dict(model))
        model.setdefault("quality_rank", _default_quality_rank(model, rank))
        model["mvp_target"] = model.get("model_id") == MVP_MODEL_ID or bool(model.get("mvp_target"))
        model["stretch_target"] = model.get("model_id") == STRETCH_MODEL_ID or bool(model.get("stretch_target"))
        enriched.append(model)
    return enriched


def _infer_hf_model_type(model: dict[str, Any]) -> str | None:
    explicit = model.get("hf_model_type") or model.get("model_type")
    if explicit:
        return str(explicit)

    model_id = str(model.get("model_id") or "").lower()
    if not model_id:
        return None
    if "qwen3.5" in model_id or "qwen3_5" in model_id or "qwen3-5" in model_id:
        return "qwen3_5_moe"
    if "minimax-m3" in model_id or "minimax/m3" in model_id:
        return "minimax_m3_vl"
    if "qwen3" in model_id and ("a3b" in model_id or "a22b" in model_id):
        return "qwen3_moe"
    if "qwen3" in model_id:
        return "qwen3"
    if "qwen2" in model_id or "deepseek-r1-distill-qwen" in model_id:
        return "qwen2"
    if "mixtral" in model_id:
        return "mixtral"
    if "mistral" in model_id:
        return "mistral"
    if "tinyllama" in model_id or "llama" in model_id:
        return "llama"
    if "bloom" in model_id:
        return "bloom"
    if "falcon" in model_id:
        return "falcon"
    if "gemma-4" in model_id or "gemma4" in model_id:
        return "gemma4"
    if "gemma-2" in model_id or "gemma2" in model_id:
        return "gemma2"
    if "glm" in model_id:
        return "glm5_moe"
    return None


def _annotate_architecture_support(model: dict[str, Any]) -> dict[str, Any]:
    model = dict(model)
    hf_model_type = _infer_hf_model_type(model)
    if hf_model_type:
        model.setdefault("hf_model_type", hf_model_type)

    if model.get("architecture_supported") is None and hf_model_type:
        model["architecture_supported"] = hf_model_type in SUPPORTED_FAMILIES
    elif model.get("architecture_supported") is not None:
        model["architecture_supported"] = bool(model.get("architecture_supported"))

    if model.get("architecture_supported") and hf_model_type in SUPPORTED_FAMILIES:
        model.setdefault("bloombee_family", SUPPORTED_FAMILIES[hf_model_type]["bloombee_family"])
        model.setdefault("block_prefix", SUPPORTED_FAMILIES[hf_model_type]["block_prefix"])

    blocked_reasons = list(model.get("blocked_reasons") or [])
    if model.get("architecture_supported") is False:
        reason = f"No BloomBee block wrapper registered for model_type={hf_model_type or 'unknown'}"
        if reason not in blocked_reasons:
            blocked_reasons.append(reason)
    if model.get("quantization_supported") is False or model.get("quantization_method"):
        method = model.get("quantization_method") or "unknown"
        reason = (
            f"Quantized checkpoint ({method}) is not native-selectable: current BloomBee "
            "HF block loading builds fp16/bf16 PyTorch blocks, not GPTQ/AWQ/FP8/NVFP4/MXFP kernels"
        )
        if reason not in blocked_reasons:
            blocked_reasons.append(reason)
    if blocked_reasons:
        model["blocked_reasons"] = blocked_reasons
    return model


def _default_quality_rank(model: dict[str, Any], fallback_rank: int) -> float:
    # Quality proxy: total params, with a boost for MoE active efficiency.
    params = float(model.get("params_b") or 0.0)
    active = float(model.get("active_params_b") or params or 0.0)
    moe_bonus = 10.0 if model.get("supports_moe") else 0.0
    # Keep deterministic ordering for same-sized models.
    return params + moe_bonus + fallback_rank / 1000.0 + active / 10000.0


def _peer_free_gb(peer: dict[str, Any]) -> float:
    memory = peer.get("memory") or {}
    accelerator = peer.get("accelerator") or {}
    free = memory.get("free_gb")
    if free is None and accelerator.get("unified_memory"):
        free = accelerator.get("vram_free_gb")
    try:
        return float(free or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _model_required_gb(model: dict[str, Any]) -> float:
    try:
        return float(model.get("recommended_min_free_mem_gb") or model.get("min_total_mem_gb") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bench_score(model: dict[str, Any], bench_matrix: dict[str, Any] | None) -> float:
    if not bench_matrix:
        return 0.0
    values: list[float] = []
    model_id = model.get("model_id")
    for peer_result in bench_matrix.values() if isinstance(bench_matrix, dict) else []:
        if not isinstance(peer_result, dict):
            continue
        entry = peer_result.get(model_id) or peer_result.get("models", {}).get(model_id)
        if isinstance(entry, dict):
            try:
                values.append(float(entry.get("decode_tok_per_s") or 0.0))
            except (TypeError, ValueError):
                pass
    return max(values) if values else 0.0


def _default_proof_status() -> dict[str, str]:
    return {key: "pending" for key in PROOF_KEYS}


def _proof_status_for(
    model: dict[str, Any],
    proof_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    status = _default_proof_status()
    model_status = model.get("proof_status")
    if isinstance(model_status, dict):
        status.update({str(key): str(value) for key, value in model_status.items()})
    model_id = model.get("model_id")
    if model_id and proof_status and model_id in proof_status:
        status.update(proof_status[model_id])
    return status


def _status_is_blocked(value: object) -> bool:
    return str(value).lower().startswith("blocked")


def _claim_level_for(model: dict[str, Any], status: dict[str, str]) -> str:
    explicit = model.get("claim_level")
    if explicit:
        return str(explicit)
    if model.get("architecture_supported") is False:
        return "blocked"
    if model.get("blocked_reasons"):
        return "blocked"
    if any(_status_is_blocked(value) for value in status.values()):
        return "blocked"
    if status.get("full_generation") == "passed":
        return "demo_safe"
    return "experimental"


def _selector_allowed(selector_mode: str, claim_level: str) -> tuple[bool, str | None]:
    if selector_mode == "planning":
        return True, None
    if selector_mode == "showcase-attempt":
        if claim_level == "blocked":
            return False, "blocked by missing wrapper or proof gate"
        return True, None
    if selector_mode == "safe-demo":
        if claim_level != "demo_safe":
            return False, "safe-demo requires full_generation proof"
        return True, None
    raise ValueError(f"unknown selector_mode={selector_mode!r}; expected one of {SELECTOR_MODES}")


def evaluate_model(
    peers: list[dict[str, Any]],
    model: dict[str, Any],
    bench_matrix: dict[str, Any] | None = None,
    *,
    proof_status: dict[str, dict[str, str]] | None = None,
    selector_mode: str = "planning",
) -> dict[str, Any]:
    model = _annotate_architecture_support(model)
    required_gb = _model_required_gb(model)
    free_by_peer = [(peer.get("hostname", "unknown"), _peer_free_gb(peer)) for peer in peers]
    solo_hosts = [host for host, free in free_by_peer if free >= required_gb]
    total_free = sum(free for _, free in free_by_peer)
    memory_fit = False
    placement = "unsupported"

    if solo_hosts:
        memory_fit = True
        placement = "solo" if len(solo_hosts) == 1 else "replicated"
        reason = f"{len(solo_hosts)} peer(s) have >= {required_gb:.1f}GB free"
    elif peers and total_free >= required_gb:
        memory_fit = True
        placement = "block_parallel_candidate"
        reason = f"aggregate swarm free memory {total_free:.1f}GB >= {required_gb:.1f}GB required"
    else:
        reason = f"requires {required_gb:.1f}GB free; swarm has {total_free:.1f}GB"

    architecture_supported = model.get("architecture_supported") is not False
    runtime_supported = memory_fit and architecture_supported
    model_proof_status = _proof_status_for(model, proof_status)
    claim_level = _claim_level_for(model, model_proof_status)
    selector_allowed, selector_blocked_reason = _selector_allowed(selector_mode, claim_level)

    return {
        "model_id": model.get("model_id"),
        "supported": memory_fit,
        "memory_fit": memory_fit,
        "architecture_supported": architecture_supported,
        "runtime_supported": runtime_supported,
        "hf_model_type": model.get("hf_model_type"),
        "bloombee_family": model.get("bloombee_family"),
        "blocked_reasons": model.get("blocked_reasons") or [],
        "placement": placement,
        "reason": reason,
        "required_free_gb": required_gb,
        "swarm_free_gb": round(total_free, 2),
        "solo_hosts": solo_hosts,
        "mvp_target": bool(model.get("mvp_target")),
        "stretch_target": bool(model.get("stretch_target")),
        "supports_moe": bool(model.get("supports_moe")),
        "quality_rank": float(model.get("quality_rank") or 0.0),
        "measured_decode_tok_per_s": _bench_score(model, bench_matrix),
        "proof_status": model_proof_status,
        "claim_level": claim_level,
        "selector_mode": selector_mode,
        "selector_allowed": selector_allowed,
        "selector_blocked_reason": selector_blocked_reason,
    }


def synthetic_m4_laptops(
    *,
    count: int = 10,
    total_gb: float = 24.0,
    free_gb: float = 20.0,
    prefix: str = "m4-laptop",
) -> list[dict[str, Any]]:
    """Create deterministic synthetic peers for pre-showcase planning.

    Synthetic peers let the MVP route picker treat the 10-laptop swarm as a
    first-class target before the physical showcase. They are explicitly marked
    so they cannot be mistaken for evidence from real hardware.
    """
    return [
        {
            "hostname": f"{prefix}-{index:02d}",
            "synthetic": True,
            "memory": {"total_gb": float(total_gb), "free_gb": float(free_gb)},
            "accelerator": {
                "device": "mps",
                "unified_memory": True,
                "vram_total_gb": float(total_gb),
                "vram_free_gb": float(free_gb),
            },
            "network": {"tailscale_ip": None},
        }
        for index in range(1, count + 1)
    ]


def choose_best_route(
    peers: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    scenario: str | None = None,
    requested_model: str | None = None,
    bench_matrix: dict[str, Any] | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
    selector_mode: str = "planning",
) -> dict[str, Any]:
    if requested_model:
        for model in registry:
            if model.get("model_id") == requested_model:
                return evaluate_model(
                    peers,
                    model,
                    bench_matrix,
                    proof_status=proof_status,
                    selector_mode=selector_mode,
                )
        return {
            "model_id": requested_model,
            "supported": False,
            "placement": "unsupported",
            "reason": "model not found in registry",
            "mvp_target": requested_model == MVP_MODEL_ID,
            "stretch_target": requested_model == STRETCH_MODEL_ID,
            "selector_mode": selector_mode,
            "selector_allowed": False,
            "selector_blocked_reason": "model not found in registry",
        }

    candidates = [
        evaluate_model(
            peers,
            model,
            bench_matrix,
            proof_status=proof_status,
            selector_mode=selector_mode,
        )
        for model in registry
    ]
    supported = [
        candidate for candidate in candidates
        if candidate["supported"] and candidate.get("selector_allowed", True)
    ]
    if not supported:
        return {
            "model_id": None,
            "supported": False,
            "placement": "unsupported",
            "reason": f"no selectable model fits current swarm for selector_mode={selector_mode}",
            "selector_mode": selector_mode,
            "selector_allowed": False,
        }

    if scenario == "mvp-10-laptop":
        for candidate in supported:
            if candidate["model_id"] == MVP_MODEL_ID:
                return candidate

    return max(
        supported,
        key=lambda item: (
            1 if item.get("mvp_target") else 0,
            item.get("quality_rank") or 0.0,
            item.get("measured_decode_tok_per_s") or 0.0,
            -float(item.get("required_free_gb") or 0.0),
        ),
    )


def explain_route(
    peers: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    scenario: str | None = None,
    requested_model: str | None = None,
    bench_matrix: dict[str, Any] | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
    selector_mode: str = "planning",
) -> dict[str, Any]:
    """Return the picked route plus full evidence for every candidate.

    Consumers (UI, tests, logs) get to see WHY the router picked what it did:
    which candidates supported, which placed how, which fell short by how much,
    whether measured bench data changed the outcome, and which proof gate made a
    candidate selectable for the requested mode.
    """
    candidates: list[dict[str, Any]] = [
        evaluate_model(
            peers,
            model,
            bench_matrix,
            proof_status=proof_status,
            selector_mode=selector_mode,
        )
        for model in registry
    ]
    if requested_model:
        for candidate in candidates:
            if candidate.get("model_id") == requested_model:
                return {
                    "picked": candidate,
                    "scenario": scenario,
                    "selector_mode": selector_mode,
                    "peer_summary": _peer_summary(peers),
                    "candidates": candidates,
                }
        return {
            "picked": {
                "model_id": requested_model,
                "supported": False,
                "placement": "unsupported",
                "reason": "model not found in registry",
                "mvp_target": requested_model == MVP_MODEL_ID,
                "stretch_target": requested_model == STRETCH_MODEL_ID,
                "selector_mode": selector_mode,
                "selector_allowed": False,
                "selector_blocked_reason": "model not found in registry",
            },
            "scenario": scenario,
            "selector_mode": selector_mode,
            "peer_summary": _peer_summary(peers),
            "candidates": candidates,
        }

    picked = choose_best_route(
        peers,
        registry,
        scenario=scenario,
        bench_matrix=bench_matrix,
        proof_status=proof_status,
        selector_mode=selector_mode,
    )

    supported = [c for c in candidates if c["supported"]]
    selectable = [c for c in supported if c.get("selector_allowed", True)]
    near_miss = [
        c for c in candidates
        if not c["supported"]
        and c.get("swarm_free_gb", 0) >= 0.5 * (c.get("required_free_gb") or 0)
    ]

    return {
        "picked": picked,
        "scenario": scenario,
        "selector_mode": selector_mode,
        "peer_summary": _peer_summary(peers),
        "supported_count": len(supported),
        "selectable_count": len(selectable),
        "near_miss": near_miss,
        "candidates": candidates,
    }


def route_report(
    peers: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    scenario: str | None = None,
    requested_model: str | None = None,
    bench_matrix: dict[str, Any] | None = None,
    proof_status: dict[str, dict[str, str]] | None = None,
    selector_mode: str = "planning",
) -> dict[str, Any]:
    """Return auto-pick plus requested-model serving/refusal metadata.

    ``explain_route`` answers "what would the router pick?". This report also
    answers "what did the operator pin, can we serve that pin under the selected
    mode, and if not what will actually be served?" No inference is run here.
    """
    auto = explain_route(
        peers,
        registry,
        scenario=scenario,
        bench_matrix=bench_matrix,
        proof_status=proof_status,
        selector_mode=selector_mode,
    )
    best_available = dict(auto.get("picked") or {})
    requested_evaluation: dict[str, Any] | None = None
    normalized_request = requested_model if requested_model not in (None, "", "auto") else None
    serving = best_available
    override_active = False
    override_refused = False
    override_reason = "auto / none"

    if normalized_request:
        requested = explain_route(
            peers,
            registry,
            scenario=scenario,
            requested_model=normalized_request,
            bench_matrix=bench_matrix,
            proof_status=proof_status,
            selector_mode=selector_mode,
        )
        requested_evaluation = dict(requested.get("picked") or {})
        requested_supported = requested_evaluation.get("supported") is True
        requested_allowed = requested_evaluation.get("selector_allowed") is True
        if requested_supported and requested_allowed:
            serving = requested_evaluation
            if serving.get("model_id") != best_available.get("model_id"):
                override_active = True
                override_reason = f"serving requested model; auto-pick would be {best_available.get('model_id')}"
            else:
                override_reason = "requested model matches best_available"
        else:
            override_refused = True
            if not requested_supported:
                override_reason = str(requested_evaluation.get("reason") or "requested model does not fit current swarm")
            else:
                override_reason = str(requested_evaluation.get("selector_blocked_reason") or "requested model is disallowed by selector mode")

    report = dict(auto)
    report.update(
        {
            "picked": serving,
            "serving": serving,
            "best_available": best_available,
            "requested_model": normalized_request,
            "requested_evaluation": requested_evaluation,
            "override_active": override_active,
            "override_refused": override_refused,
            "override_reason": override_reason,
            "inference_proven": False,
            "can_update_proof_status": False,
        }
    )
    return report


def _peer_summary(peers: list[dict[str, Any]]) -> dict[str, Any]:
    free_by_host = {
        peer.get("hostname", "unknown"): _peer_free_gb(peer) for peer in peers
    }
    return {
        "peer_count": len(peers),
        "swarm_free_gb": round(sum(free_by_host.values()), 2),
        "free_by_host": free_by_host,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--cap-dir", action="append", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model", dest="requested_model", default=None)
    parser.add_argument("--bench-matrix", default=None)
    parser.add_argument("--proof-status", default=str(DEFAULT_PROOF_STATUS))
    parser.add_argument(
        "--selector-mode",
        choices=SELECTOR_MODES,
        default="planning",
        help="planning ignores proof gates; showcase-attempt blocks missing wrappers; safe-demo requires full_generation proof.",
    )
    parser.add_argument("--synthetic-m4-laptops", type=int, default=0, help="Append N synthetic M4 laptop peers")
    parser.add_argument("--synthetic-total-gb", type=float, default=24.0)
    parser.add_argument("--synthetic-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print full candidate evidence (peer summary, near-misses, all placements).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print route report with best_available, requested-model override/refusal, and serving model.",
    )
    args = parser.parse_args(argv)

    bench_matrix = None
    if args.bench_matrix:
        bench_matrix = json.loads(Path(args.bench_matrix).expanduser().read_text(encoding="utf-8"))
    proof_status = load_proof_status(args.proof_status)
    peers = load_roster(args.cap_dir or [DEFAULT_CAP_DIR])
    if args.synthetic_m4_laptops:
        peers.extend(
            synthetic_m4_laptops(
                count=args.synthetic_m4_laptops,
                total_gb=args.synthetic_total_gb,
                free_gb=args.synthetic_free_gb,
            )
        )
    registry = load_registry(args.registry)

    if args.report:
        report = route_report(
            peers,
            registry,
            scenario=args.scenario,
            requested_model=args.requested_model,
            bench_matrix=bench_matrix,
            proof_status=proof_status,
            selector_mode=args.selector_mode,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.explain:
        explainable = explain_route(
            peers,
            registry,
            scenario=args.scenario,
            requested_model=args.requested_model,
            bench_matrix=bench_matrix,
            proof_status=proof_status,
            selector_mode=args.selector_mode,
        )
        print(json.dumps(explainable, indent=2, sort_keys=True))
        return 0

    print(
        json.dumps(
            choose_best_route(
                peers,
                registry,
                scenario=args.scenario,
                requested_model=args.requested_model,
                bench_matrix=bench_matrix,
                proof_status=proof_status,
                selector_mode=args.selector_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
