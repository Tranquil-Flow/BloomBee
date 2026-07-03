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


def test_swarm_simulator_filters_failed_hosts_and_replans_layers():
    from mvp_capabilities.swarm_simulator import simulate_swarm

    result = simulate_swarm(
        scenario="mvp-10-laptop",
        model_id="Qwen/Qwen3-30B-A3B",
        synthetic_m4_laptops=10,
        synthetic_total_gb=24,
        synthetic_free_gb=20,
        failed_hosts=["m4-laptop-01", "m4-laptop-02", "ghost-host"],
        request_count=3,
    )

    assert result["claim_boundary"] == "simulation_only_no_inference_proof"
    assert result["scenario"] == "mvp-10-laptop"
    assert result["input_peer_count"] == 10
    assert result["active_peer_count"] == 8
    assert result["failed_hosts"] == ["m4-laptop-01", "m4-laptop-02", "ghost-host"]
    assert result["missing_failed_hosts"] == ["ghost-host"]
    assert result["request_count"] == 3
    assert result["route"]["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert result["route"]["supported"] is True
    assert result["layer_plan"]["supported"] is True
    assert result["layer_plan"]["assigned_layers"] == result["layer_plan"]["num_layers"] == 48
    assert result["layer_plan"]["claim_boundary"] == "placement_plan_only_no_inference_proof"


def test_swarm_simulator_cli_outputs_failure_scenario_json():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/swarm_simulator.py",
            "--scenario",
            "mvp-10-laptop",
            "--model",
            "Qwen/Qwen3-30B-A3B",
            "--synthetic-m4-laptops",
            "10",
            "--fail-host",
            "m4-laptop-01",
            "--request-count",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["active_peer_count"] == 9
    assert payload["request_count"] == 2
    assert payload["route"]["supported"] is True
    assert payload["layer_plan"]["assigned_layers"] == 48
    assert payload["claim_boundary"] == "simulation_only_no_inference_proof"


def test_swarm_simulator_pure_synthetic_ignores_default_capability_dir(tmp_path: Path, monkeypatch):
    from mvp_capabilities import swarm_simulator

    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _write_peer(cap_dir / "real-peer.json", hostname="real-peer", total_gb=48, free_gb=48)
    monkeypatch.setattr(swarm_simulator, "DEFAULT_CAP_DIR", cap_dir)

    result = swarm_simulator.simulate_swarm(
        scenario="mvp-10-laptop",
        model_id="Qwen/Qwen3-30B-A3B",
        synthetic_m4_laptops=2,
        synthetic_total_gb=24,
        synthetic_free_gb=20,
        failed_hosts=[],
    )

    assert result["input_peer_count"] == 2
    assert result["active_peer_count"] == 2
    assert "real-peer" not in {peer["hostname"] for peer in result["layer_plan"]["assignments"]}


def test_proof_ladder_reports_ordered_gates_and_next_pending_step():
    from mvp_capabilities.proof_ladder import build_proof_ladder

    proof_status = {
        "Qwen/Qwen3-8B": {
            "prescan": "passed",
            "one_block_server": "pending",
            "multi_block": "pending",
            "full_generation": "pending",
        }
    }

    report = build_proof_ladder("Qwen/Qwen3-8B", proof_status=proof_status)

    assert report["model_id"] == "Qwen/Qwen3-8B"
    assert report["claim_boundary"] == "proof_ladder_audit_only_no_inference_proof"
    assert report["claim_level"] == "experimental"
    assert report["next_gate"] == "one_block_server"
    assert [gate["name"] for gate in report["gates"]] == [
        "prescan",
        "one_block_server",
        "multi_block",
        "full_generation",
        "cache_generation",
        "multi_request_load",
    ]
    assert report["gates"][0]["status"] == "passed"
    assert report["gates"][1]["status"] == "pending"


def test_proof_ladder_safe_demo_requires_full_generation_passed():
    from mvp_capabilities.proof_ladder import build_proof_ladder

    report = build_proof_ladder(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        proof_status={
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "prescan": "passed",
                "one_block_server": "passed",
                "multi_block": "passed",
                "full_generation": "passed",
                "cache_generation": "pending",
                "multi_request_load": "pending",
            }
        },
    )

    assert report["claim_level"] == "demo_safe"
    assert report["safe_demo_selectable"] is True
    assert report["next_gate"] == "cache_generation"


def test_proof_ladder_cli_outputs_fallback_ladder_json():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/proof_ladder.py",
            "--fallback-ladder",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "proof_ladder_audit_only_no_inference_proof"
    assert [item["model_id"] for item in payload["models"]][:3] == [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
    ]
    assert all("next_gate" in item for item in payload["models"])


def test_qwen3_dense_fallbacks_have_prescan_only_not_safe_demo():
    from mvp_capabilities.model_compat_scan import load_proof_status
    from mvp_capabilities.proof_ladder import build_proof_ladder

    proof = load_proof_status(PROJECT_ROOT / "mvp_capabilities" / "PROOF_STATUS.yaml")

    for model_id in ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B"):
        report = build_proof_ladder(model_id, proof_status=proof)
        assert report["proof_status"]["prescan"] == "passed"
        assert report["proof_status"]["one_block_server"] == "pending"
        assert report["claim_level"] == "experimental"
        assert report["safe_demo_selectable"] is False
        assert report["next_gate"] == "one_block_server"


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


def test_registry_includes_prepared_qwen3_30b_2507_variants_with_pending_proof():
    from mvp_capabilities.model_compat_scan import load_proof_status
    from mvp_capabilities.route_picker import choose_best_route, load_registry, synthetic_m4_laptops

    registry = load_registry(REGISTRY_PATH)
    by_id = {model["model_id"]: model for model in registry}
    expected_ids = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
    ]

    for model_id in expected_ids:
        model = by_id[model_id]
        assert model["supports_moe"] is True
        assert model["hf_model_type"] == "qwen3_moe"
        assert model["num_layers"] == 48
        assert model["hidden_size"] == 2048
        assert model["num_experts"] == 128
        assert model["num_experts_per_tok"] == 8
        assert model["context_length"] == 262144
        assert model["recommended_min_free_mem_gb"] == 70

        route = choose_best_route(
            synthetic_m4_laptops(count=10, total_gb=24, free_gb=20),
            registry,
            requested_model=model_id,
            proof_status=load_proof_status(PROJECT_ROOT / "mvp_capabilities" / "PROOF_STATUS.yaml"),
            selector_mode="safe-demo",
        )
        assert route["supported"] is True
        assert route["claim_level"] == "experimental"
        assert route["selector_allowed"] is False
        assert route["proof_status"]["full_generation"] == "pending"


def test_layer_planner_assigns_contiguous_ranges_by_peer_capacity():
    from mvp_capabilities.layer_planner import plan_layer_placement

    model = {
        "model_id": "test/TwelveLayer",
        "num_layers": 12,
        "recommended_min_free_mem_gb": 24,
    }
    peers = [
        {"hostname": "alpha", "memory": {"free_gb": 10}, "accelerator": {"device": "mps"}},
        {"hostname": "bravo", "memory": {"free_gb": 8}, "accelerator": {"device": "mps"}},
        {"hostname": "charlie", "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
    ]

    plan = plan_layer_placement(peers, model)

    assert plan["supported"] is True
    assert plan["model_id"] == "test/TwelveLayer"
    assert plan["num_layers"] == 12
    assert plan["per_layer_required_gb"] == 2.0
    assert plan["claim_boundary"] == "placement_plan_only_no_inference_proof"
    assert plan["assignments"] == [
        {"hostname": "alpha", "start_layer": 0, "end_layer": 5, "layer_count": 5, "free_gb": 10.0, "capacity_layers": 5},
        {"hostname": "bravo", "start_layer": 5, "end_layer": 9, "layer_count": 4, "free_gb": 8.0, "capacity_layers": 4},
        {"hostname": "charlie", "start_layer": 9, "end_layer": 12, "layer_count": 3, "free_gb": 6.0, "capacity_layers": 3},
    ]


def test_layer_planner_reports_missing_capacity_without_overclaiming():
    from mvp_capabilities.layer_planner import plan_layer_placement

    model = {
        "model_id": "test/FourLayer",
        "num_layers": 4,
        "recommended_min_free_mem_gb": 40,
    }
    peers = [
        {"hostname": "tiny-a", "memory": {"free_gb": 10}, "accelerator": {}},
        {"hostname": "tiny-b", "memory": {"free_gb": 10}, "accelerator": {}},
    ]

    plan = plan_layer_placement(peers, model)

    assert plan["supported"] is False
    assert plan["reason"] == "capacity covers 2/4 layers; missing 2"
    assert plan["assigned_layers"] == 2
    assert plan["missing_layers"] == 2
    assert plan["claim_boundary"] == "placement_plan_only_no_inference_proof"


def test_layer_planner_cli_outputs_json_for_synthetic_swarm():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/layer_planner.py",
            "--model",
            "Qwen/Qwen3-30B-A3B",
            "--synthetic-m4-laptops",
            "10",
            "--synthetic-total-gb",
            "24",
            "--synthetic-free-gb",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert payload["supported"] is True
    assert payload["num_layers"] == 48
    assert payload["assigned_layers"] == 48
    assert payload["assignments"][0]["start_layer"] == 0
    assert payload["assignments"][-1]["end_layer"] == 48


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


def test_peer_scan_marks_non_mobile_hosts_explicitly(monkeypatch):
    from mvp_capabilities import peer_scan

    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.setattr(peer_scan.sys, "platform", "darwin")

    assert peer_scan.detect_mobile_profile() == {"is_mobile": False, "kind": None, "runtime": None}


def test_peer_scan_identifies_termux_android(monkeypatch):
    from mvp_capabilities import peer_scan

    props = {
        "ro.product.model": "Pixel 8 Pro",
        "ro.product.manufacturer": "Google",
        "ro.soc.model": "Tensor G3",
        "ro.product.cpu.abi": "arm64-v8a",
        "ro.build.version.sdk": "35",
    }

    def fake_getprop(name: str):
        return props.get(name)

    monkeypatch.setattr(peer_scan.sys, "platform", "linux")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(peer_scan, "_android_getprop", fake_getprop)

    profile = peer_scan.detect_mobile_profile()

    assert profile["is_mobile"] is True
    assert profile["kind"] == "android"
    assert profile["runtime"] == "termux"
    assert profile["model"] == "Pixel 8 Pro"
    assert profile["soc"] == "Tensor G3"
    assert profile["cpu_abi"] == "arm64-v8a"


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


def _write_hf_config(model_dir: Path, **fields) -> Path:
    model_dir.mkdir()
    path = model_dir / "config.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _tiny_route_registry() -> list[dict]:
    return [
        {
            "model_id": "demo-safe/Tiny-Proven",
            "recommended_min_free_mem_gb": 4,
            "min_total_mem_gb": 4,
            "params_b": 1.0,
            "active_params_b": 1.0,
            "quality_rank": 1.0,
        },
        {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "recommended_min_free_mem_gb": 70,
            "min_total_mem_gb": 63,
            "params_b": 30.5,
            "active_params_b": 3.3,
            "quality_rank": 50.0,
            "supports_moe": True,
        },
        {
            "model_id": "zai-org/GLM-5.2",
            "recommended_min_free_mem_gb": 70,
            "min_total_mem_gb": 63,
            "params_b": 744.0,
            "active_params_b": 40.0,
            "quality_rank": 200.0,
            "supports_moe": True,
        },
    ]


def _proof_status_fixture() -> dict[str, dict[str, str]]:
    return {
        "demo-safe/Tiny-Proven": {
            "prescan": "passed",
            "one_block_server": "passed",
            "multi_block": "passed",
            "full_generation": "passed",
        },
        "Qwen/Qwen3-30B-A3B": {
            "prescan": "passed",
            "one_block_server": "passed",
            "multi_block": "pending",
            "full_generation": "pending",
        },
        "zai-org/GLM-5.2": {
            "prescan": "pending",
            "one_block_server": "blocked-by-wrapper",
            "multi_block": "blocked-by-wrapper",
            "full_generation": "blocked-by-wrapper",
        },
    }


def test_safe_demo_selector_picks_only_full_generation_proven_model():
    from mvp_capabilities.route_picker import choose_best_route

    peers = [{"hostname": "swarm", "memory": {"free_gb": 100}, "accelerator": {}}]

    route = choose_best_route(
        peers,
        _tiny_route_registry(),
        proof_status=_proof_status_fixture(),
        selector_mode="safe-demo",
    )

    assert route["model_id"] == "demo-safe/Tiny-Proven"
    assert route["selector_mode"] == "safe-demo"
    assert route["claim_level"] == "demo_safe"
    assert route["selector_allowed"] is True


def test_showcase_attempt_allows_experimental_but_blocks_wrapper_blocked():
    from mvp_capabilities.route_picker import explain_route

    peers = [{"hostname": "swarm", "memory": {"free_gb": 100}, "accelerator": {}}]

    result = explain_route(
        peers,
        _tiny_route_registry(),
        proof_status=_proof_status_fixture(),
        selector_mode="showcase-attempt",
    )

    assert result["picked"]["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert result["picked"]["claim_level"] == "experimental"
    blocked = next(c for c in result["candidates"] if c["model_id"] == "zai-org/GLM-5.2")
    assert blocked["claim_level"] == "blocked"
    assert blocked["selector_allowed"] is False
    assert "wrapper" in blocked["selector_blocked_reason"].lower()


def test_planning_mode_keeps_memory_fit_candidates_even_when_unproven():
    from mvp_capabilities.route_picker import choose_best_route

    peers = [{"hostname": "swarm", "memory": {"free_gb": 100}, "accelerator": {}}]

    route = choose_best_route(
        peers,
        _tiny_route_registry(),
        proof_status=_proof_status_fixture(),
        selector_mode="planning",
    )

    assert route["model_id"] == "zai-org/GLM-5.2"
    assert route["claim_level"] == "blocked"
    assert route["selector_allowed"] is True
    assert route["selector_mode"] == "planning"


def test_cli_safe_demo_mode_uses_proof_status_before_selecting_model():
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "mvp_capabilities/route_picker.py",
        "--cap-dir",
        "/tmp/mvp_demo_caps",
        "--scenario",
        "mvp-10-laptop",
        "--synthetic-m4-laptops",
        "10",
        "--selector-mode",
        "safe-demo",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["selector_mode"] == "safe-demo"
    assert payload["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert payload["claim_level"] == "demo_safe"


def test_model_compat_scan_marks_qwen3_moe_as_bloombee_supported(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import scan_model_config

    model_dir = tmp_path / "qwen3_moe"
    _write_hf_config(
        model_dir,
        model_type="qwen3_moe",
        num_hidden_layers=48,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts=128,
        num_experts_per_tok=8,
        architectures=["Qwen3MoeForCausalLM"],
    )

    result = scan_model_config(model_dir, model_id="Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert result["model_id"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert result["hf_model_type"] == "qwen3_moe"
    assert result["architecture_supported"] is True
    assert result["bloombee_family"] == "qwen3_moe"
    assert result["block_prefix"] == "model.layers"
    assert result["num_layers"] == 48
    assert result["hidden_size"] == 2048
    assert result["num_experts"] == 128
    assert result["experts_per_token"] == 8
    assert result["proof_status"]["prescan"] == "passed"
    assert result["claim_level"] == "experimental"


def test_model_compat_scan_marks_unknown_frontier_family_blocked(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import scan_model_config

    model_dir = tmp_path / "glm52"
    _write_hf_config(
        model_dir,
        model_type="glm5_moe",
        num_hidden_layers=128,
        hidden_size=8192,
        num_attention_heads=64,
        num_experts=256,
        num_experts_per_tok=8,
    )

    result = scan_model_config(model_dir, model_id="zai-org/GLM-5.2")

    assert result["hf_model_type"] == "glm5_moe"
    assert result["architecture_supported"] is False
    assert result["claim_level"] == "blocked"
    assert "wrapper" in result["blocked_reasons"][0].lower()


def test_model_compat_scan_merges_proof_status_registry(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import load_proof_status, scan_model_config

    model_dir = tmp_path / "qwen3_moe"
    _write_hf_config(
        model_dir,
        model_type="qwen3_moe",
        num_hidden_layers=48,
        hidden_size=2048,
    )
    proof_path = tmp_path / "proof.yaml"
    proof_path.write_text(
        "models:\n"
        "  Qwen/Qwen3-30B-A3B:\n"
        "    one_block_server: passed\n"
        "    multi_block: pending\n"
        "    full_generation: pending\n",
        encoding="utf-8",
    )

    proof = load_proof_status(proof_path)
    result = scan_model_config(model_dir, model_id="Qwen/Qwen3-30B-A3B", proof_status=proof)

    assert result["proof_status"]["prescan"] == "passed"
    assert result["proof_status"]["one_block_server"] == "passed"
    assert result["proof_status"]["multi_block"] == "pending"
    assert result["claim_level"] == "experimental"


def test_join_offer_builds_shareable_link_with_expiry():
    from mvp_capabilities.join_coordinator import create_join_offer

    offer = create_join_offer(
        coordinator="http://m4pro.local:8787",
        token="moon-token",
        now=1_000,
        ttl_seconds=120,
    )

    assert offer["token"] == "moon-token"
    assert offer["created_at"] == 1_000
    assert offer["expires_at"] == 1_120
    assert offer["coordinator"] == "http://m4pro.local:8787"
    assert offer["join_url"] == "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token"
    assert offer["claim_boundary"] == "link_offer_only_no_inference_proof"


def test_join_heartbeat_state_filters_stale_and_wrong_token_peers(tmp_path: Path):
    from mvp_capabilities.join_coordinator import load_active_heartbeats, record_heartbeat

    state_dir = tmp_path / "join-state"
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="fresh-peer",
        capabilities={"memory": {"free_gb": 20}, "accelerator": {"device": "mps"}},
        now=1_000,
    )
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="stale-peer",
        capabilities={"memory": {"free_gb": 20}},
        now=900,
    )
    record_heartbeat(
        state_dir,
        token="other-token",
        peer_id="wrong-token-peer",
        capabilities={"memory": {"free_gb": 20}},
        now=1_000,
    )

    active = load_active_heartbeats(state_dir, token="moon-token", now=1_030, max_age_seconds=60)

    assert [peer["peer_id"] for peer in active] == ["fresh-peer"]
    assert active[0]["capabilities"]["memory"]["free_gb"] == 20


def test_join_coordinator_cli_offer_outputs_json():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_coordinator.py",
            "offer",
            "--coordinator",
            "http://m4pro.local:8787",
            "--token",
            "moon-token",
            "--now",
            "1000",
            "--ttl-seconds",
            "120",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["join_url"].startswith("bloombee://join?")
    assert payload["expires_at"] == 1120
