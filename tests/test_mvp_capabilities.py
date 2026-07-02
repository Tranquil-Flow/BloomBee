from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
REGISTRY_PATH = PROJECT_ROOT / "mvp_capabilities" / "MODEL_REGISTRY.yaml"


def _write_peer(path: Path, *, hostname: str, total_gb: float, free_gb: float, device: str = "mps") -> Path:
    payload = {
        "hostname": hostname,
        "platform": {"system": "Darwin", "machine": "arm64"},
        "memory": {"total_gb": total_gb, "free_gb": free_gb},
        "accelerator": {
            "device": device,
            "unified_memory": device == "mps",
            "vram_total_gb": total_gb if device == "mps" else None,
            "vram_free_gb": free_gb if device == "mps" else None,
            "gpus": [],
        },
        "network": {"tailscale_ip": None},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_roster_from_capability_dir(tmp_path: Path):
    from mvp_capabilities.swarm_roster import load_roster, summarize_roster

    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _write_peer(cap_dir / "zeta.json", hostname="zeta", total_gb=24, free_gb=18)
    _write_peer(cap_dir / "alpha.json", hostname="alpha", total_gb=16, free_gb=7)

    roster = load_roster([cap_dir])
    assert [peer["hostname"] for peer in roster] == ["alpha", "zeta"]

    summary = summarize_roster(roster)
    assert summary["peer_count"] == 2
    assert summary["total_memory_gb"] == 40
    assert summary["free_memory_gb"] == 25
    assert summary["accelerators"] == {"mps": 2}


def test_load_roster_deduplicates_same_hostname(tmp_path: Path):
    from mvp_capabilities.swarm_roster import load_roster, summarize_roster

    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _write_peer(cap_dir / "evinova.json", hostname="evinova", total_gb=16, free_gb=2)
    _write_peer(cap_dir / "evinova.stdout.json", hostname="evinova", total_gb=16, free_gb=3)

    roster = load_roster([cap_dir])
    assert [peer["hostname"] for peer in roster] == ["evinova"]
    assert roster[0]["memory"]["free_gb"] == 3
    assert summarize_roster(roster)["total_memory_gb"] == 16


def test_route_picker_recommends_qwen3_30b_for_10_laptops():
    from mvp_capabilities.route_picker import choose_best_route, load_registry, synthetic_m4_laptops

    peers = synthetic_m4_laptops(count=10, total_gb=24, free_gb=20)
    registry = load_registry(REGISTRY_PATH)

    route = choose_best_route(peers, registry, scenario="mvp-10-laptop")

    assert route["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert route["placement"] in {"block_parallel_candidate", "replicated"}
    assert route["mvp_target"] is True
    assert route["supported"] is True
    assert route["reason"]


def test_synthetic_m4_laptops_have_stable_hostnames():
    from mvp_capabilities.route_picker import synthetic_m4_laptops

    peers = synthetic_m4_laptops(count=3, total_gb=24, free_gb=20, prefix="showcase")

    assert [peer["hostname"] for peer in peers] == ["showcase-01", "showcase-02", "showcase-03"]
    assert all(peer["accelerator"]["device"] == "mps" for peer in peers)


def test_bench_matrix_feeds_measured_decode_tok_per_s_into_router(tmp_path: Path):
    from mvp_capabilities.bench_matrix import build_matrix
    from mvp_capabilities.route_picker import choose_best_route, load_registry

    bench_path = tmp_path / "bench.jsonl"
    bench_path.write_text(
        '{"model": "Qwen/Qwen2.5-7B-Instruct", "device": "mps", "dtype": "bf16", '
        '"decode_tok_per_s": 2.61, "prefill_tok_per_s": 65.4, "params_b": 7.62, "host": "m4pro"}\n'
        '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "mps", "dtype": "bf16", '
        '"decode_tok_per_s": 11.4, "prefill_tok_per_s": 587.0, "params_b": 0.49, "host": "m4pro"}\n'
    )

    matrix = build_matrix([bench_path], default_host="m4pro")
    assert "m4pro" in matrix
    assert matrix["m4pro"]["models"]["Qwen/Qwen2.5-7B-Instruct"]["decode_tok_per_s"] == 2.61

    peers = [
        {
            "hostname": "m4pro",
            "memory": {"total_gb": 48, "free_gb": 30},
            "accelerator": {"device": "mps", "unified_memory": True},
        }
    ]
    registry = load_registry(REGISTRY_PATH)
    candidates = [choose_best_route(peers, [m], bench_matrix=matrix) for m in registry]
    seven_b = next(c for c in candidates if c["model_id"] == "Qwen/Qwen2.5-7B-Instruct")
    assert seven_b["measured_decode_tok_per_s"] == 2.61


def test_route_picker_marks_235b_as_stretch_until_memory_fits():
    from mvp_capabilities.route_picker import choose_best_route, load_registry

    peers = [
        {
            "hostname": f"m4-laptop-{i:02d}",
            "memory": {"total_gb": 24, "free_gb": 20},
            "accelerator": {"device": "mps", "unified_memory": True},
        }
        for i in range(10)
    ]
    registry = load_registry(REGISTRY_PATH)

    route = choose_best_route(peers, registry, requested_model="Qwen/Qwen3-235B-A22B")

    assert route["model_id"] == "Qwen/Qwen3-235B-A22B"
    assert route["supported"] is False
    assert route["stretch_target"] is True
    assert "requires" in route["reason"].lower()


def test_sweep_models_dry_run_selects_feasible_models(tmp_path: Path):
    from mvp_capabilities.sweep_models import build_sweep_plan

    peer_path = _write_peer(tmp_path / "peer.json", hostname="tiny-peer", total_gb=8, free_gb=6)
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]

    plan = build_sweep_plan(peer_path, registry, max_models=20)
    model_ids = [item["model_id"] for item in plan["models"]]

    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in model_ids
    assert "Qwen/Qwen2.5-7B-Instruct" not in model_ids
    assert plan["peer"]["hostname"] == "tiny-peer"
    assert all("command" in item for item in plan["models"])


def test_docs_use_distributed_inference_mvp_name_not_bloombee_mvp():
    doc_path = PROJECT_ROOT / "docs" / "distributed-inference-mvp.md"
    assert doc_path.exists()
    text = doc_path.read_text(encoding="utf-8")
    assert "distributed-inference-mvp" in text


def test_explain_route_returns_picked_plus_full_candidate_evidence():
    from mvp_capabilities.route_picker import explain_route, load_registry

    peers = [
        {
            "hostname": f"m4-laptop-{i:02d}",
            "memory": {"total_gb": 24, "free_gb": 20},
            "accelerator": {"device": "mps", "unified_memory": True},
        }
        for i in range(10)
    ]
    registry = load_registry(REGISTRY_PATH)

    result = explain_route(peers, registry, scenario="mvp-10-laptop")

    assert "picked" in result
    assert result["picked"]["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert result["picked"]["mvp_target"] is True
    assert result["scenario"] == "mvp-10-laptop"
    assert result["peer_summary"]["peer_count"] == 10
    assert result["peer_summary"]["swarm_free_gb"] == 200.0
    assert result["supported_count"] >= 1
    assert isinstance(result["near_miss"], list)
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) == len(registry)


def test_explain_route_surfaces_near_miss_when_swarm_is_short():
    from mvp_capabilities.route_picker import explain_route, load_registry

    # Single 16 GB M4 — Qwen3-235B won't fit solo, but the explain view
    # still shows the candidate and tells us how far short we are.
    peers = [
        {
            "hostname": "evinova",
            "memory": {"total_gb": 16, "free_gb": 14},
            "accelerator": {"device": "mps", "unified_memory": True},
        }
    ]
    registry = load_registry(REGISTRY_PATH)

    result = explain_route(peers, registry)

    assert result["picked"]["model_id"] != "Qwen/Qwen3-235B-A22B"
    stretch = next(
        c for c in result["candidates"]
        if c["model_id"] == "Qwen/Qwen3-235B-A22B"
    )
    assert stretch["supported"] is False
    assert stretch["stretch_target"] is True
    assert "swarm_free_gb" in stretch
    assert "required_free_gb" in stretch
    assert stretch["swarm_free_gb"] < stretch["required_free_gb"]


def test_cli_explain_flag_runs_without_error():
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "mvp_capabilities/route_picker.py",
        "--cap-dir",
        "/tmp/mvp_demo_caps",
        "--explain",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "candidates" in payload
    assert "peer_summary" in payload
    assert "near_miss" in payload
    # Picked payload should NOT collide with top-level keys.
    assert "picked" in payload
    assert isinstance(payload["picked"], dict)
    assert "candidates" not in payload["picked"]  # no nested duplication
