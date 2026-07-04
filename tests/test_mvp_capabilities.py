from __future__ import annotations

import json
import subprocess
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


def test_qwen3_dense_fallbacks_track_one_block_without_safe_demo():
    from mvp_capabilities.model_compat_scan import load_proof_status
    from mvp_capabilities.proof_ladder import build_proof_ladder

    proof = load_proof_status(PROJECT_ROOT / "mvp_capabilities" / "PROOF_STATUS.yaml")

    expected = {
        "Qwen/Qwen3-8B": ("passed", "multi_block"),
        "Qwen/Qwen3-14B": ("pending", "one_block_server"),
    }
    for model_id, (one_block_status, next_gate) in expected.items():
        report = build_proof_ladder(model_id, proof_status=proof)
        assert report["proof_status"]["prescan"] == "passed"
        assert report["proof_status"]["one_block_server"] == one_block_status
        assert report["claim_level"] == "experimental"
        assert report["safe_demo_selectable"] is False
        assert report["next_gate"] == next_gate


def test_mvp_status_report_has_weighted_progress_bar():
    from mvp_capabilities.mvp_status import build_status_report

    report = build_status_report()
    assert report["claim_boundary"] == "weighted_plan_status_not_demo_proof"
    assert report["total_weight"] == 100
    assert report["overall_percent"] == 76
    assert report["overall_bar"] == "███████████████░░░░░ 76%"
    assert report["remaining_percent"] == 24
    assert report["next_gate"] == "Qwen3-8B multi-block or full-generation proof"
    assert any(item["id"] == "qwen3_30b_proof_ladder" for item in report["milestones"])
    assert report["task_summary"] == {"complete": 5, "partial": 7, "pending": 3, "blocked": 2, "total": 17}
    tasks = {item["id"]: item for item in report["planned_tasks"]}
    assert tasks["tinyllama_distributed_generation"]["done"] is True
    assert tasks["qwen3_8b_proof"]["status"] == "partial"
    assert tasks["qwen35b_candidate"]["status"] == "blocked"
    assert tasks["physical_showcase"]["done"] is False


def test_mvp_status_markdown_contains_status_bar_and_next_gate():
    from mvp_capabilities.mvp_status import build_status_report, render_markdown

    text = render_markdown(build_status_report())
    assert "Distributed Inference MVP status" in text
    assert "███████████████░░░░░ 76%" in text
    assert "Qwen3-8B multi-block or full-generation proof" in text
    assert "weighted_plan_status_not_demo_proof" in text
    assert "## Planned tasks" in text
    assert "TinyLlama distributed fallback generation proof | complete | yes" in text
    assert "Physical/self-serve N-laptop showcase | pending | no" in text


def test_mvp_status_cli_outputs_json():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/mvp_status.py", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["overall_percent"] == 76
    assert payload["overall_bar"].endswith("76%")
    assert payload["next_gate"] == "Qwen3-8B multi-block or full-generation proof"
    assert payload["task_summary"]["blocked"] == 2
    assert any(task["id"] == "minimax_m3_candidate" and task["status"] == "blocked" for task in payload["planned_tasks"])


def test_one_block_proof_plan_generates_qwen3_8b_commands():
    from mvp_capabilities.one_block_proof import build_one_block_plan
    from mvp_capabilities.route_picker import load_registry

    plan = build_one_block_plan(
        "Qwen/Qwen3-8B",
        registry=load_registry(REGISTRY_PATH),
        port=31337,
        device="mps",
        dtype="float16",
    )

    assert plan["model_id"] == "Qwen/Qwen3-8B"
    assert plan["claim_boundary"] == "proof_harness_only_no_live_inference"
    assert plan["hidden_size"] == 4096
    assert plan["block_range"] == "0:1"
    assert "--block_indices 0:1" in plan["server_command"]
    assert "--hidden-dim 4096" in plan["client_command"]
    assert plan["proof_status_on_success"] == "one_block_server: passed"


def test_one_block_proof_verifier_requires_server_and_client_evidence():
    from mvp_capabilities.one_block_proof import verify_one_block_evidence

    server_log = "[INFO] Announced that blocks range(0, 1) are joining\n[INFO] Started\n"
    client_log = '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], "outputs_finite": true, "grad_finite": true}\n'

    result = verify_one_block_evidence(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        server_log=server_log,
        client_log=client_log,
    )

    assert result["claim_boundary"] == "verified_one_block_server_evidence"
    assert result["proof_gate"] == "one_block_server"
    assert result["status"] == "passed"
    assert result["can_update_proof_status"] is True


def test_one_block_proof_verifier_blocks_partial_evidence():
    from mvp_capabilities.one_block_proof import verify_one_block_evidence

    result = verify_one_block_evidence(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        server_log="[INFO] Started\n",
        client_log='[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], "outputs_finite": true, "grad_finite": true}\n',
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert "server did not announce requested block range" in result["failed_checks"]


def test_one_block_proof_cli_plan_outputs_json():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/one_block_proof.py", "plan", "--model", "Qwen/Qwen3-8B"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model_id"] == "Qwen/Qwen3-8B"
    assert payload["claim_boundary"] == "proof_harness_only_no_live_inference"
    assert payload["server_command"].startswith("PYTHONPATH=.:src")


def test_multi_block_proof_plan_generates_two_server_runbook():
    from mvp_capabilities.multi_block_proof import build_multi_block_plan
    from mvp_capabilities.route_picker import load_registry

    plan = build_multi_block_plan(
        "Qwen/Qwen3-8B",
        registry=load_registry(REGISTRY_PATH),
        block_ranges=["0:1", "1:2"],
        ports=[31337, 31338],
    )

    assert plan["claim_boundary"] == "multi_block_proof_harness_only_no_live_inference"
    assert plan["proof_gate"] == "multi_block"
    assert plan["combined_block_range"] == "0:2"
    assert len(plan["server_commands"]) == 2
    assert "--new_swarm" in plan["server_commands"][0]
    assert "--initial_peers '<PASTE_SEED_MULTIADDR>'" in plan["server_commands"][1]
    assert "BLOOMBEE_INITIAL_PEERS" not in plan["server_commands"][1]
    assert "--server-maddr '<PASTE_SERVER_0_MULTIADDR>'" in plan["client_command"]
    assert "--server-maddr '<PASTE_SERVER_1_MULTIADDR>'" in plan["client_command"]
    assert plan["proof_status_on_success"] == "multi_block: passed"


def test_multi_block_proof_verifier_requires_each_server_and_combined_client():
    from mvp_capabilities.multi_block_proof import verify_multi_block_evidence

    server_logs = [
        "[INFO] Announced that blocks range(0, 1) are joining\n[INFO] Started\n[INFO] rpc_forward(blocks=0:1, remote_peer=...abc)\n",
        "[INFO] Announced that blocks range(1, 2) are joining\n[INFO] Started\n[INFO] rpc_backward(blocks=1:2, remote_peer=...abc)\n",
    ]
    client_log = '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 2], "outputs_finite": true, "grad_finite": true}\n'

    result = verify_multi_block_evidence(
        model_id="Qwen/Qwen3-8B",
        block_ranges=["0:1", "1:2"],
        server_logs=server_logs,
        client_log=client_log,
    )

    assert result["status"] == "passed"
    assert result["claim_boundary"] == "verified_multi_block_server_evidence"
    assert result["proof_gate"] == "multi_block"
    assert result["combined_block_range"] == "0:2"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"multi_block": "passed"}


def test_multi_block_proof_verifier_blocks_missing_server_rpc():
    from mvp_capabilities.multi_block_proof import verify_multi_block_evidence

    result = verify_multi_block_evidence(
        model_id="Qwen/Qwen3-8B",
        block_ranges=["0:1", "1:2"],
        server_logs=[
            "[INFO] Announced that blocks range(0, 1) are joining\n[INFO] Started\n[INFO] rpc_forward(blocks=0:1, remote_peer=...abc)\n",
            "[INFO] Announced that blocks range(1, 2) are joining\n[INFO] Started\n",
        ],
        client_log='[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 2], "outputs_finite": true, "grad_finite": true}\n',
    )

    assert result["status"] == "failed"
    assert any("server 1 did not record rpc evidence" in item for item in result["failed_checks"])


def test_request_telemetry_parses_direct_client_results_and_errors(tmp_path: Path):
    from mvp_capabilities.request_telemetry import build_request_telemetry

    success_log = tmp_path / "direct-success.log"
    success_log.write_text(
        "[direct] model=Qwen/Qwen3-8B\n"
        '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
        '"forward_seconds": 0.08, "backward_seconds": 0.20, "outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )
    failure_log = tmp_path / "direct-failure.log"
    failure_log.write_text(
        "[direct] model=Qwen/Qwen3-8B\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: DHT bootstrap failed before RPC\n",
        encoding="utf-8",
    )

    report = build_request_telemetry([success_log, failure_log])

    assert report["claim_boundary"] == "request_telemetry_observability_only_no_load_proof"
    assert report["live_requests_seen"] is True
    assert report["load_proof_claimed"] is False
    assert report["request_counts"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert report["models"] == {"Qwen/Qwen3-8B": 1}
    assert report["block_ranges"] == {"0:1": 1}
    assert report["latency_seconds"]["forward"]["avg"] == 0.08
    assert report["latency_seconds"]["backward"]["max"] == 0.2
    assert report["errors"][0]["message"] == "RuntimeError: DHT bootstrap failed before RPC"


def test_request_telemetry_treats_zero_latency_as_unmeasured(tmp_path: Path):
    from mvp_capabilities.request_telemetry import build_request_telemetry

    log = tmp_path / "direct-zero-latency.log"
    log.write_text(
        '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
        '"forward_seconds": 0, "backward_seconds": 0, "outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )

    report = build_request_telemetry([log])

    assert report["request_counts"] == {"total": 1, "succeeded": 1, "failed": 0}
    assert report["latency_seconds"]["forward"]["count"] == 0
    assert report["latency_seconds"]["forward"]["avg"] is None
    assert report["latency_seconds"]["forward"]["unmeasured_count"] == 1
    assert report["latency_seconds"]["backward"]["count"] == 0
    assert report["latency_seconds"]["backward"]["unmeasured_count"] == 1


def test_request_telemetry_cli_outputs_json(tmp_path: Path):
    log = tmp_path / "direct.log"
    log.write_text(
        '[direct] RESULT: {"ok": true, "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", '
        '"block_range": [0, 2], "forward_seconds": 1.5, "backward_seconds": 2.5, '
        '"outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/request_telemetry.py",
            "--request-log",
            str(log),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["request_counts"]["succeeded"] == 1
    assert payload["latency_seconds"]["forward"]["max"] == 1.5


def test_multi_request_load_proof_plan_generates_repeated_client_runbook():
    from mvp_capabilities.multi_request_load_proof import build_multi_request_load_plan

    plan = build_multi_request_load_plan(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        server_maddrs=["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
        request_count=3,
        hidden_dim=4096,
    )

    assert plan["claim_boundary"] == "multi_request_load_harness_only_no_live_traffic"
    assert plan["proof_gate"] == "multi_request_load"
    assert plan["request_count"] == 3
    assert len(plan["client_commands"]) == 3
    assert "--server-maddr '/ip4/100.64.0.20/tcp/31337/p2p/seed'" in plan["client_commands"][0]
    assert "tee .local/load-client-000.log" in plan["client_commands"][0]
    assert "multi_request_load_proof.py verify" in plan["verify_command"]
    assert plan["proof_status_on_success"] == "multi_request_load: passed"


def test_multi_request_load_proof_verifier_accepts_successful_live_request_logs(tmp_path: Path):
    from mvp_capabilities.multi_request_load_proof import verify_multi_request_load_evidence

    logs: list[Path] = []
    for index in range(3):
        log = tmp_path / f"load-client-{index:03d}.log"
        log.write_text(
            '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
            f'"forward_seconds": {0.1 + index / 100:.2f}, "backward_seconds": 0.2, '
            '"outputs_finite": true, "grad_finite": true}\n',
            encoding="utf-8",
        )
        logs.append(log)

    result = verify_multi_request_load_evidence(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        request_logs=logs,
        expected_request_count=3,
    )

    assert result["status"] == "passed"
    assert result["claim_boundary"] == "verified_multi_request_load_evidence"
    assert result["proof_gate"] == "multi_request_load"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"multi_request_load": "passed"}
    assert result["telemetry"]["request_counts"] == {"total": 3, "succeeded": 3, "failed": 0}
    assert result["telemetry"]["latency_seconds"]["forward"]["max"] == 0.12


def test_multi_request_load_proof_verifier_blocks_failed_or_missing_requests(tmp_path: Path):
    from mvp_capabilities.multi_request_load_proof import verify_multi_request_load_evidence

    ok_log = tmp_path / "load-client-000.log"
    ok_log.write_text(
        '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
        '"forward_seconds": 0.1, "backward_seconds": 0.2, "outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )
    failed_log = tmp_path / "load-client-001.log"
    failed_log.write_text("RuntimeError: DHT bootstrap failed before RPC\n", encoding="utf-8")

    result = verify_multi_request_load_evidence(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        request_logs=[ok_log, failed_log],
        expected_request_count=3,
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert any("expected 3 successful requests, saw 1" in item for item in result["failed_checks"])
    assert any("request telemetry recorded 1 failed request" in item for item in result["failed_checks"])


def test_multi_request_load_proof_verifier_blocks_unmeasured_zero_latency(tmp_path: Path):
    from mvp_capabilities.multi_request_load_proof import verify_multi_request_load_evidence

    log = tmp_path / "load-client-000.log"
    log.write_text(
        '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
        '"forward_seconds": 0, "backward_seconds": 0, "outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )

    result = verify_multi_request_load_evidence(
        model_id="Qwen/Qwen3-8B",
        block_range="0:1",
        request_logs=[log],
        expected_request_count=1,
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert result["telemetry"]["latency_seconds"]["forward"]["unmeasured_count"] == 1
    assert any("forward latency measured for 0/1 requests" in item for item in result["failed_checks"])
    assert any("0 means unmeasured" in item for item in result["failed_checks"])


def test_full_generation_proof_plan_emits_text_generation_parity_runbook():
    from mvp_capabilities.full_generation_proof import build_full_generation_plan

    plan = build_full_generation_plan(
        model_id="Qwen/Qwen3-8B",
        server_maddrs=["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
        server_placements=["m4pro=0:36"],
        prompt="The moon is",
        max_new_tokens=4,
        evidence_path=".local/qwen3-full-generation.json",
    )

    assert plan["claim_boundary"] == "full_generation_proof_harness_only_no_live_generation"
    assert plan["proof_gate"] == "full_generation"
    assert "scripts/text_generation_parity.py" in plan["parity_command"]
    assert "--server-placement 'm4pro=0:36'" in plan["parity_command"]
    assert "--out .local/qwen3-full-generation.json" in plan["parity_command"]
    assert "full_generation_proof.py verify" in plan["verify_command"]
    assert plan["proof_status_on_success"] == "full_generation: passed"


def test_full_generation_proof_verifier_accepts_matching_parity_evidence(tmp_path: Path):
    from mvp_capabilities.full_generation_proof import verify_full_generation_evidence

    evidence = tmp_path / "qwen3-full-generation.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "generate-api",
                "model": "Qwen/Qwen3-8B",
                "prompt": "The moon is",
                "max_new_tokens": 4,
                "generated_ids_match": True,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_ids": [1, 2, 3, 4],
                "reference_ids": [1, 2, 3, 4],
                "distributed_text": "The moon is bright",
                "reference_text": "The moon is bright",
                "distributed_seconds": 12.0,
                "reference_seconds": 2.0,
                "server_maddrs": ["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
                "server_placements": [
                    {"host": "m4pro", "layers": [0, 36], "server_maddr": "/ip4/100.64.0.20/tcp/31337/p2p/seed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_full_generation_evidence(
        evidence_path=evidence,
        model_id="Qwen/Qwen3-8B",
        min_new_tokens=4,
        require_server_placements=True,
    )

    assert result["status"] == "passed"
    assert result["claim_boundary"] == "verified_full_generation_evidence"
    assert result["proof_gate"] == "full_generation"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"full_generation": "passed"}
    assert result["evidence_summary"]["generated_ids_match"] is True
    assert result["evidence_summary"]["server_count"] == 1


def test_full_generation_proof_verifier_blocks_mismatch_or_missing_placements(tmp_path: Path):
    from mvp_capabilities.full_generation_proof import verify_full_generation_evidence

    evidence = tmp_path / "bad-full-generation.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "generate-api",
                "model": "Qwen/Qwen3-8B",
                "max_new_tokens": 4,
                "generated_ids_match": False,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_ids": [1, 2, 9],
                "reference_ids": [1, 2, 3],
                "server_maddrs": ["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
                "server_placements": [],
            }
        ),
        encoding="utf-8",
    )

    result = verify_full_generation_evidence(
        evidence_path=evidence,
        model_id="Qwen/Qwen3-8B",
        min_new_tokens=4,
        require_server_placements=True,
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert any("generated token IDs did not match" in item for item in result["failed_checks"])
    assert any("server placements are required" in item for item in result["failed_checks"])


def test_cache_generation_proof_plan_uses_generate_api_runbook():
    from mvp_capabilities.cache_generation_proof import build_cache_generation_plan

    plan = build_cache_generation_plan(
        model_id="Qwen/Qwen3-8B",
        server_maddrs=["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
        server_placements=["m4pro=0:36"],
        prompt="The moon is",
        max_new_tokens=4,
        evidence_path=".local/qwen3-cache-generation.json",
    )

    assert plan["claim_boundary"] == "cache_generation_proof_harness_only_no_live_generation"
    assert plan["proof_gate"] == "cache_generation"
    assert "scripts/text_generation_parity.py" in plan["parity_command"]
    assert "--mode generate-api" in plan["parity_command"]
    assert "--out .local/qwen3-cache-generation.json" in plan["parity_command"]
    assert "cache_generation_proof.py verify" in plan["verify_command"]
    assert plan["proof_status_on_success"] == "cache_generation: passed"


def test_cache_generation_proof_verifier_accepts_generate_api_parity(tmp_path: Path):
    from mvp_capabilities.cache_generation_proof import verify_cache_generation_evidence

    evidence = tmp_path / "qwen3-cache-generation.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "generate-api",
                "model": "Qwen/Qwen3-8B",
                "prompt": "The moon is",
                "max_new_tokens": 4,
                "generated_ids_match": True,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_ids": [1, 2, 3, 4],
                "reference_ids": [1, 2, 3, 4],
                "distributed_text": "The moon is bright",
                "reference_text": "The moon is bright",
                "distributed_steps": [],
                "reference_steps": [],
                "distributed_seconds": 12.0,
                "reference_seconds": 2.0,
                "server_maddrs": ["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
                "server_placements": [
                    {"host": "m4pro", "layers": [0, 36], "server_maddr": "/ip4/100.64.0.20/tcp/31337/p2p/seed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_cache_generation_evidence(
        evidence_path=evidence,
        model_id="Qwen/Qwen3-8B",
        min_new_tokens=4,
        require_server_placements=True,
    )

    assert result["status"] == "passed"
    assert result["claim_boundary"] == "verified_cache_generation_evidence"
    assert result["proof_gate"] == "cache_generation"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"cache_generation": "passed"}
    assert result["evidence_summary"]["mode"] == "generate-api"


def test_cache_generation_proof_verifier_blocks_forward_loop_evidence(tmp_path: Path):
    from mvp_capabilities.cache_generation_proof import verify_cache_generation_evidence

    evidence = tmp_path / "forward-loop-generation.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "forward-loop",
                "model": "Qwen/Qwen3-8B",
                "max_new_tokens": 4,
                "generated_ids_match": True,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_ids": [1, 2, 3, 4],
                "reference_ids": [1, 2, 3, 4],
                "distributed_text": "The moon is bright",
                "reference_text": "The moon is bright",
                "distributed_steps": [{"step": 0}],
                "reference_steps": [{"step": 0}],
                "server_maddrs": ["/ip4/100.64.0.20/tcp/31337/p2p/seed"],
                "server_placements": [
                    {"host": "m4pro", "layers": [0, 36], "server_maddr": "/ip4/100.64.0.20/tcp/31337/p2p/seed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_cache_generation_evidence(
        evidence_path=evidence,
        model_id="Qwen/Qwen3-8B",
        min_new_tokens=4,
        require_server_placements=True,
    )

    assert result["status"] == "failed"
    assert result["can_update_proof_status"] is False
    assert any("cache_generation requires mode=generate-api" in item for item in result["failed_checks"])


def test_proof_state_parses_retained_download_logs_and_cache_stats(tmp_path: Path):
    from mvp_capabilities.proof_state import build_proof_state

    status = tmp_path / "download.status"
    status.write_text("EXIT_CODE=0\n", encoding="utf-8")
    log = tmp_path / "download.log"
    log.write_text(
        "START=2026-07-03T16:33:24Z\n"
        "HOST=m4pro\n"
        "MODEL=Qwen/Qwen3-8B\n"
        "TOKEN_FILE_PRESENT True\n"
        "\rFetching 15 files:  40%|████      | 6/15 [00:01<00:01,  6.97it/s]\n"
        "SNAPSHOT_PATH /Users/evi/.cache/hub/models--Qwen--Qwen3-8B/snapshots/abc\n"
        "SECONDS 123.45\n"
        "WEIGHT_FILES=5\n"
        "24G\t/Users/evi/.cache/hub/models--Qwen--Qwen3-8B\n"
        "END=2026-07-03T16:40:00Z\n",
        encoding="utf-8",
    )

    payload = build_proof_state(
        model="Qwen/Qwen3-8B",
        gate="one_block_server",
        status_file=status,
        log_file=log,
        cache_bytes=24 * 1024**3,
        weight_files=5,
    )

    assert payload["claim_boundary"] == "proof_state_observability_only_no_inference_proof"
    assert payload["model"] == "Qwen/Qwen3-8B"
    assert payload["gate"] == "one_block_server"
    assert payload["download_status"] == "complete"
    assert payload["exit_code"] == 0
    assert payload["host"] == "m4pro"
    assert payload["token_file_present"] is True
    assert payload["fetch_progress"]["percent"] == 40
    assert payload["fetch_progress"]["completed_files"] == 6
    assert payload["fetch_progress"]["total_files"] == 15
    assert payload["cache"]["weight_files"] == 5
    assert payload["cache"]["bytes"] == 24 * 1024**3
    assert payload["inference_proven"] is False


def test_proof_state_marks_complete_snapshot_despite_stale_incomplete_blobs(tmp_path: Path):
    from mvp_capabilities.proof_state import build_proof_state

    cache = tmp_path / "models--Qwen--Qwen3-8B"
    snapshot = cache / "snapshots" / "abc123"
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    for index in range(1, 6):
        (snapshot / f"model-0000{index}-of-00005.safetensors").write_bytes(b"x" * index)
    (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (blobs / "old-partial.incomplete").write_bytes(b"stale")
    log = tmp_path / "download.log"
    log.write_text(
        f"SNAPSHOT_PATH={snapshot}\n"
        "Fetching 15 files: 100%|██████████| 15/15 [00:00<00:00, 2000it/s]\n",
        encoding="utf-8",
    )

    payload = build_proof_state(
        model="Qwen/Qwen3-8B",
        gate="one_block_server",
        log_file=log,
        cache_dir=cache,
    )

    assert payload["download_status"] == "complete"
    assert payload["cache"]["snapshot_complete"] is True
    assert payload["cache"]["snapshot_weight_files"] == 5
    assert payload["cache"]["stale_incomplete_files"] == 1
    assert payload["eta_seconds"] == 0
    assert payload["eta_reason"] == "snapshot_complete"
    assert payload["inference_proven"] is False


def test_proof_state_cli_outputs_json(tmp_path: Path):
    status = tmp_path / "download.status"
    status.write_text("EXIT_CODE=1\n", encoding="utf-8")
    log = tmp_path / "download.log"
    log.write_text("HOST=m4pro\nMODEL=Qwen/Qwen3-8B\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "mvp_capabilities" / "proof_state.py"),
            "--model",
            "Qwen/Qwen3-8B",
            "--gate",
            "one_block_server",
            "--status-file",
            str(status),
            "--log-file",
            str(log),
            "--weight-files",
            "0",
            "--cache-bytes",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["download_status"] == "failed"
    assert payload["exit_code"] == 1
    assert payload["inference_proven"] is False


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


def test_registry_includes_qwen35b_candidate_branch_but_blocks_showcase_until_wrapper():
    from mvp_capabilities.model_compat_scan import load_proof_status
    from mvp_capabilities.route_picker import choose_best_route, load_registry, synthetic_m4_laptops

    registry = load_registry(REGISTRY_PATH)
    by_id = {model["model_id"]: model for model in registry}
    model = by_id["Qwen/Qwen-AgentWorld-35B-A3B"]

    assert model["candidate_branch"] == "qwen35b"
    assert model["hf_model_type"] == "qwen3_5_moe"
    assert model["hf_text_model_type"] == "qwen3_5_moe_text"
    assert model["num_layers"] == 40
    assert model["hidden_size"] == 2048
    assert model["num_experts"] == 256
    assert model["num_experts_per_tok"] == 8
    assert model["recommended_min_free_mem_gb"] == 80
    assert model["architecture_supported"] is False

    peers = synthetic_m4_laptops(count=10, total_gb=24, free_gb=20)
    proof = load_proof_status(PROJECT_ROOT / "mvp_capabilities" / "PROOF_STATUS.yaml")
    planning = choose_best_route(
        peers,
        registry,
        requested_model="Qwen/Qwen-AgentWorld-35B-A3B",
        proof_status=proof,
        selector_mode="planning",
    )
    assert planning["memory_fit"] is True
    assert planning["runtime_supported"] is False
    assert planning["claim_level"] == "blocked"
    assert planning["selector_allowed"] is True

    showcase = choose_best_route(
        peers,
        registry,
        requested_model="Qwen/Qwen-AgentWorld-35B-A3B",
        proof_status=proof,
        selector_mode="showcase-attempt",
    )
    assert showcase["selector_allowed"] is False
    assert "wrapper" in showcase["selector_blocked_reason"].lower()
    assert showcase["proof_status"]["one_block_server"] == "blocked-by-wrapper"


def test_registry_tracks_minimax_m3_as_high_compute_blocked_candidate():
    from mvp_capabilities.route_picker import choose_best_route, load_registry, synthetic_m4_laptops

    registry = load_registry(REGISTRY_PATH)
    by_id = {model["model_id"]: model for model in registry}
    model = by_id["MiniMaxAI/MiniMax-M3"]

    assert model["hf_model_type"] == "minimax_m3_vl"
    assert model.get("hf_text_model_type") is None
    assert model["params_b"] == 428.0
    assert model["active_params_b"] == 23.0
    assert model["num_layers"] == 60
    assert model["hidden_size"] == 6144
    assert model["supports_sparse_attention"] is True
    assert model["recommended_min_free_mem_gb"] == 900
    assert model["architecture_supported"] is False

    route = choose_best_route(
        synthetic_m4_laptops(count=10, total_gb=24, free_gb=20),
        registry,
        requested_model="MiniMaxAI/MiniMax-M3",
        selector_mode="planning",
    )
    assert route["memory_fit"] is False
    assert route["runtime_supported"] is False
    assert route["claim_level"] == "blocked"
    assert any("sparse attention" in reason.lower() for reason in route["blocked_reasons"])


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


def test_layer_planner_can_attach_exact_bloombee_server_commands():
    from mvp_capabilities.layer_planner import attach_launch_commands, plan_layer_placement

    model = {
        "model_id": "test/TwelveLayer",
        "num_layers": 12,
        "recommended_min_free_mem_gb": 24,
    }
    peers = [
        {"hostname": "alpha", "memory": {"free_gb": 10}, "accelerator": {"device": "mps"}},
        {"hostname": "bravo", "memory": {"free_gb": 14}, "accelerator": {"device": "mps"}},
    ]
    plan = attach_launch_commands(
        plan_layer_placement(peers, model),
        device="mps",
        dtype="float16",
        base_port=31337,
        dht_prefix="demo-prefix",
    )

    assert plan["launch_commands_claim_boundary"] == "launch_commands_only_no_server_started"
    assert plan["assignments"][0]["port"] == 31337
    assert plan["assignments"][1]["port"] == 31338
    assert "--block_indices 0:5" in plan["assignments"][0]["launch_command"]
    assert "--block_indices 5:12" in plan["assignments"][1]["launch_command"]
    assert "--dht_prefix demo-prefix" in plan["assignments"][0]["launch_command"]
    assert "--initial_peers '<SEED_MULTIADDR_FROM_alpha>'" in plan["assignments"][1]["launch_command"]
    assert "BLOOMBEE_INITIAL_PEERS" not in plan["assignments"][1]["launch_command"]


def test_layer_planner_cli_can_include_launch_commands():
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
            "--include-launch-commands",
            "--dht-prefix",
            "mvp-demo",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["launch_commands_claim_boundary"] == "launch_commands_only_no_server_started"
    assert payload["assignments"][0]["launch_command"].startswith("PYTHONPATH=.:src python -m bloombee.cli.run_server")
    assert "--new_swarm" in payload["assignments"][0]["launch_command"]
    assert payload["assignments"][1]["launch_command"].startswith("PYTHONPATH=.:src python -m bloombee.cli.run_server")
    assert "--initial_peers '<SEED_MULTIADDR_FROM_" in payload["assignments"][1]["launch_command"]
    assert "BLOOMBEE_INITIAL_PEERS" not in payload["assignments"][1]["launch_command"]


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


def test_model_compat_scan_extracts_nested_qwen35b_text_config_and_blocks_wrapper(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import scan_model_config

    model_dir = tmp_path / "qwen35"
    _write_hf_config(
        model_dir,
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        language_model_only=True,
        text_config={
            "model_type": "qwen3_5_moe_text",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 262144,
        },
    )

    result = scan_model_config(model_dir, model_id="Qwen/Qwen-AgentWorld-35B-A3B")

    assert result["hf_model_type"] == "qwen3_5_moe"
    assert result["hf_text_model_type"] == "qwen3_5_moe_text"
    assert result["num_layers"] == 40
    assert result["hidden_size"] == 2048
    assert result["num_key_value_heads"] == 2
    assert result["num_experts"] == 256
    assert result["experts_per_token"] == 8
    assert result["max_position_embeddings"] == 262144
    assert result["architecture_supported"] is False
    assert result["claim_level"] == "blocked"
    assert "qwen3_5_moe" in result["blocked_reasons"][0]


def test_model_compat_scan_marks_quantized_qwen35b_checkpoint_blocked(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import scan_model_config

    model_dir = tmp_path / "qwen35_gptq"
    _write_hf_config(
        model_dir,
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        quantization_config={"quant_method": "gptq", "bits": 4},
        text_config={
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
        },
    )

    result = scan_model_config(model_dir, model_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")

    assert result["quantization_method"] == "gptq"
    assert result["quantization_supported"] is False
    assert result["claim_level"] == "blocked"
    assert any("quantized" in reason.lower() for reason in result["blocked_reasons"])


def test_model_compat_scan_extracts_minimax_m3_text_config_and_blocks_sparse_wrapper(tmp_path: Path):
    from mvp_capabilities.model_compat_scan import scan_model_config

    model_dir = tmp_path / "minimax_m3"
    _write_hf_config(
        model_dir,
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config={
            "architectures": ["MiniMaxM3SparseForCausalLM"],
            "num_hidden_layers": 60,
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "max_position_embeddings": 1048576,
            "sparse_attention_config": {"use_sparse_attention": True},
        },
    )

    result = scan_model_config(model_dir, model_id="MiniMaxAI/MiniMax-M3")

    assert result["hf_model_type"] == "minimax_m3_vl"
    assert result["hf_text_model_type"] is None
    assert result["num_layers"] == 60
    assert result["hidden_size"] == 6144
    assert result["num_experts"] == 128
    assert result["experts_per_token"] == 4
    assert result["max_position_embeddings"] == 1048576
    assert result["uses_sparse_attention"] is True
    assert result["architecture_supported"] is False
    assert result["claim_level"] == "blocked"
    assert any("sparse_attention" in reason for reason in result["blocked_reasons"])


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


def test_route_picker_blocks_registry_models_without_bloombee_wrapper():
    from mvp_capabilities.route_picker import choose_best_route, load_registry

    peers = [{"hostname": "m4pro", "memory": {"free_gb": 25}, "accelerator": {"device": "mps"}}]
    registry = load_registry(REGISTRY_PATH)

    qwen25 = choose_best_route(
        peers,
        registry,
        requested_model="Qwen/Qwen2.5-7B-Instruct",
        selector_mode="showcase-attempt",
    )

    assert qwen25["hf_model_type"] == "qwen2"
    assert qwen25["architecture_supported"] is False
    assert qwen25["memory_fit"] is True
    assert qwen25["runtime_supported"] is False
    assert qwen25["claim_level"] == "blocked"
    assert qwen25["selector_allowed"] is False
    assert "wrapper" in qwen25["selector_blocked_reason"].lower()

    route = choose_best_route(peers, registry, selector_mode="showcase-attempt")

    assert route["model_id"].startswith("Qwen/Qwen3-")
    assert route["hf_model_type"] == "qwen3"
    assert route["architecture_supported"] is True
    assert route["runtime_supported"] is True
    assert route["claim_level"] == "experimental"


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


def test_join_http_server_health_offer_heartbeat_and_active(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get, handle_post

    state_dir = tmp_path / "join-http-state"

    status, health = handle_get("/healthz", state_dir=state_dir, coordinator="http://127.0.0.1:8787")
    assert status == 200
    assert health == {"ok": True, "claim_boundary": "coordinator_health_only_no_inference_proof"}

    status, offer = handle_get(
        "/offer?token=moon-token&ttl_seconds=120",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
    )
    assert status == 200
    assert offer["token"] == "moon-token"
    assert offer["claim_boundary"] == "link_offer_only_no_inference_proof"
    assert offer["join_url"].startswith("bloombee://join?")

    heartbeat_body = json.dumps(
        {
            "token": "moon-token",
            "peer_id": "fresh-peer",
            "capabilities": {"hostname": "fresh-peer", "memory": {"free_gb": 20}, "accelerator": {"device": "mps"}},
        }
    ).encode("utf-8")
    status, heartbeat = handle_post("/heartbeat", body=heartbeat_body, state_dir=state_dir)
    assert status == 200
    assert heartbeat["peer_id"] == "fresh-peer"
    assert heartbeat["claim_boundary"] == "heartbeat_only_no_inference_proof"

    status, active = handle_get("/active?token=moon-token&max_age_seconds=60", state_dir=state_dir, coordinator="http://127.0.0.1:8787")
    assert status == 200
    assert active["claim_boundary"] == "heartbeat_roster_only_no_inference_proof"
    assert [peer["peer_id"] for peer in active["active_peers"]] == ["fresh-peer"]


def test_join_http_server_plan_endpoint_builds_launch_ready_plan(tmp_path: Path):
    from urllib.parse import quote

    from mvp_capabilities.join_http_server import handle_get, handle_post

    state_dir = tmp_path / "join-http-state"
    for peer in ("peer-a", "peer-b"):
        body = json.dumps(
            {
                "token": "moon-token",
                "peer_id": peer,
                "capabilities": {"hostname": peer, "memory": {"free_gb": 12}, "accelerator": {"device": "mps"}},
                "now": 100,
            }
        ).encode("utf-8")
        status, heartbeat = handle_post("/heartbeat", body=body, state_dir=state_dir)
        assert status == 200
        assert heartbeat["claim_boundary"] == "heartbeat_only_no_inference_proof"

    seed = "/ip4/100.64.0.10/tcp/41000/p2p/12D3KooWseed"
    status, plan = handle_get(
        "/plan?"
        "token=moon-token"
        "&model=Qwen%2FQwen3-8B"
        "&now=110"
        "&max_age_seconds=60"
        "&include_launch_commands=1"
        "&include_launch_readiness=1"
        "&base_port=41000"
        f"&seed_multiaddr={quote(f'peer-a={seed}', safe='')}",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
    )

    assert status == 200
    assert plan["claim_boundary"] == "joined_roster_layer_plan_only_no_inference_proof"
    assert plan["source"] == "coordinator_http_plan_endpoint"
    assert plan["active_peer_count"] == 2
    assert plan["placement"]["launch_commands_claim_boundary"] == "launch_commands_only_no_server_started"
    assert plan["placement"]["multiaddr_resolution_claim_boundary"] == "launch_multiaddr_resolution_only_no_server_started"
    assert plan["launch_readiness"]["ready_to_start"] is True
    assert seed in plan["placement"]["assignments"][1]["launch_command"]
    assert plan["inference_proven"] is False


def test_join_http_server_plan_endpoint_requires_token_and_model(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get

    status, missing_token = handle_get(
        "/plan?model=Qwen%2FQwen3-8B",
        state_dir=tmp_path / "join-http-state",
        coordinator="http://127.0.0.1:8787",
    )
    status_model, missing_model = handle_get(
        "/plan?token=moon-token",
        state_dir=tmp_path / "join-http-state",
        coordinator="http://127.0.0.1:8787",
    )

    assert status == 400
    assert missing_token == {"error": "missing token", "claim_boundary": "coordinator_error_no_inference_proof"}
    assert status_model == 400
    assert missing_model == {"error": "missing model", "claim_boundary": "coordinator_error_no_inference_proof"}


def test_join_http_server_route_endpoint_picks_best_joined_model(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get, handle_post

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
models:
  - model_id: test/Tiny
    params_b: 1
    num_layers: 2
    recommended_min_free_mem_gb: 4
  - model_id: test/Bigger
    params_b: 3
    num_layers: 6
    recommended_min_free_mem_gb: 12
""".strip(),
        encoding="utf-8",
    )
    state_dir = tmp_path / "join-http-state"
    body = json.dumps(
        {
            "token": "moon-token",
            "peer_id": "joined-peer",
            "capabilities": {"hostname": "joined-peer", "memory": {"free_gb": 14}, "accelerator": {"device": "mps"}},
            "now": 100,
        }
    ).encode("utf-8")
    status, _ = handle_post("/heartbeat", body=body, state_dir=state_dir)
    assert status == 200

    status, route = handle_get(
        "/route?token=moon-token&now=110&max_age_seconds=60&selector_mode=planning",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
        registry=registry,
    )

    assert status == 200
    assert route["claim_boundary"] == "coordinator_route_only_no_inference_proof"
    assert route["source"] == "coordinator_http_route_endpoint"
    assert route["picked"]["model_id"] == "test/Bigger"
    assert route["peer_summary"]["peer_count"] == 1
    assert route["selector_mode"] == "planning"
    assert route["inference_proven"] is False


def test_join_http_server_plan_endpoint_can_auto_select_model(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get, handle_post

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
models:
  - model_id: test/Tiny
    params_b: 1
    num_layers: 2
    recommended_min_free_mem_gb: 4
  - model_id: test/Bigger
    params_b: 3
    num_layers: 6
    recommended_min_free_mem_gb: 12
""".strip(),
        encoding="utf-8",
    )
    state_dir = tmp_path / "join-http-state"
    body = json.dumps(
        {
            "token": "moon-token",
            "peer_id": "joined-peer",
            "capabilities": {"hostname": "joined-peer", "memory": {"free_gb": 14}, "accelerator": {"device": "mps"}},
            "now": 100,
        }
    ).encode("utf-8")
    status, _ = handle_post("/heartbeat", body=body, state_dir=state_dir)
    assert status == 200

    status, plan = handle_get(
        "/plan?token=moon-token&model=auto&now=110&max_age_seconds=60&selector_mode=planning&include_launch_commands=1",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
        registry=registry,
    )

    assert status == 200
    assert plan["model_id"] == "test/Bigger"
    assert plan["route_decision"]["picked"]["model_id"] == "test/Bigger"
    assert plan["route_decision"]["claim_boundary"] == "coordinator_route_only_no_inference_proof"
    assert plan["placement"]["supported"] is True
    assert plan["placement"]["assignments"][0]["block_range"] == "0:6"
    assert plan["inference_proven"] is False


def test_join_http_server_handoff_endpoint_bundles_operator_runbook(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get, handle_post

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
models:
  - model_id: test/Tiny
    params_b: 1
    num_layers: 2
    hidden_size: 64
    recommended_min_free_mem_gb: 4
  - model_id: test/Bigger
    params_b: 3
    num_layers: 6
    hidden_size: 128
    recommended_min_free_mem_gb: 12
""".strip(),
        encoding="utf-8",
    )
    state_dir = tmp_path / "join-http-state"
    for peer in ("peer-a", "peer-b"):
        body = json.dumps(
            {
                "token": "moon-token",
                "peer_id": peer,
                "capabilities": {"hostname": peer, "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
                "now": 100,
            }
        ).encode("utf-8")
        status, _ = handle_post("/heartbeat", body=body, state_dir=state_dir)
        assert status == 200

    status, handoff = handle_get(
        "/handoff?token=moon-token&model=auto&now=110&max_age_seconds=60&selector_mode=planning&base_port=41000&request_count=2",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
        registry=registry,
    )

    assert status == 200
    assert handoff["claim_boundary"] == "coordinator_handoff_bundle_only_no_server_started"
    assert handoff["source"] == "coordinator_http_handoff_endpoint"
    assert handoff["offer"]["claim_boundary"] == "link_offer_only_no_inference_proof"
    assert handoff["active"]["claim_boundary"] == "heartbeat_roster_only_no_inference_proof"
    assert handoff["route_decision"]["picked"]["model_id"] == "test/Bigger"
    assert handoff["plan"]["model_id"] == "test/Bigger"
    assert handoff["plan"]["launch_readiness"]["ready_to_start"] is False
    assert "--initial_peers" in handoff["plan"]["placement"]["assignments"][1]["launch_command"]
    assert "BLOOMBEE_INITIAL_PEERS" not in handoff["plan"]["placement"]["assignments"][1]["launch_command"]
    assert handoff["proof_runbooks"]["multi_block"]["claim_boundary"] == "multi_block_proof_harness_only_no_live_inference"
    assert handoff["proof_runbooks"]["full_generation"]["claim_boundary"] == "full_generation_proof_harness_only_no_live_generation"
    assert handoff["proof_runbooks"]["cache_generation"]["claim_boundary"] == "cache_generation_proof_harness_only_no_live_generation"
    assert handoff["proof_runbooks"]["multi_request_load"]["claim_boundary"] == "multi_request_load_harness_only_no_live_traffic"
    assert handoff["proof_runbooks"]["multi_request_load"]["request_count"] == 2
    assert handoff["proof_runbooks"]["multi_request_load"]["hidden_dim"] == 128
    assert handoff["bootstrap_runbook"]["claim_boundary"] == "coordinator_bootstrap_runbook_only_no_server_started"
    assert "join_client.py" in handoff["bootstrap_runbook"]["shell_script"]
    assert handoff["proof_orchestration"]["claim_boundary"] == "proof_orchestration_plan_only_no_live_inference"
    assert handoff["proof_orchestration"]["source"] == "coordinator_handoff_embedded_proof_orchestration"
    assert handoff["proof_orchestration"]["phase_order"] == [
        "start_servers",
        "capture_server_multiaddrs",
        "run_proof_clients",
        "verify_then_promote_manually",
    ]
    assert handoff["proof_orchestration"]["launch_steps"][1]["role"] == "follower"
    assert "--initial_peers" in handoff["proof_orchestration"]["launch_steps"][1]["command"]
    assert "BLOOMBEE_INITIAL_PEERS" not in json.dumps(handoff["proof_orchestration"])
    assert handoff["proof_orchestration"]["summary"]["ready_for_proof_clients"] is False
    assert "<PASTE_SERVER_0_MULTIADDR>" in handoff["proof_orchestration"]["summary"]["unresolved_placeholders"]
    assert handoff["inference_proven"] is False
    assert handoff["can_update_proof_status"] is False


def test_join_http_server_proof_orchestration_endpoint_returns_operator_plan(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get, handle_post

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
models:
  - model_id: test/Bigger
    params_b: 3
    num_layers: 6
    hidden_size: 128
    recommended_min_free_mem_gb: 12
""".strip(),
        encoding="utf-8",
    )
    state_dir = tmp_path / "join-http-state"
    for peer in ("peer-a", "peer-b"):
        status, _ = handle_post(
            "/heartbeat",
            body=json.dumps(
                {
                    "token": "moon-token",
                    "peer_id": peer,
                    "capabilities": {"hostname": peer, "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
                    "now": 100,
                }
            ).encode("utf-8"),
            state_dir=state_dir,
        )
        assert status == 200

    status, plan = handle_get(
        "/proof-orchestration?token=moon-token&model=auto&now=110&max_age_seconds=60&selector_mode=planning&request_count=2",
        state_dir=state_dir,
        coordinator="http://127.0.0.1:8787",
        registry=registry,
    )

    assert status == 200
    assert plan["source"] == "coordinator_http_proof_orchestration_endpoint"
    assert plan["claim_boundary"] == "proof_orchestration_plan_only_no_live_inference"
    assert plan["model_id"] == "test/Bigger"
    assert [step["proof_gate"] for step in plan["proof_steps"]] == [
        "multi_block",
        "full_generation",
        "cache_generation",
        "multi_request_load",
    ]
    assert plan["summary"]["server_count"] == 2
    assert plan["summary"]["available_proof_gates"] == ["multi_block", "full_generation", "cache_generation", "multi_request_load"]
    assert plan["inference_proven"] is False
    assert plan["can_update_proof_status"] is False


def test_proof_orchestrator_blocks_ignored_bloombee_initial_peers_env():
    from mvp_capabilities.proof_orchestrator import build_proof_orchestration_plan

    handoff = {
        "claim_boundary": "coordinator_handoff_bundle_only_no_server_started",
        "source": "coordinator_http_handoff_endpoint",
        "token": "moon-token",
        "plan": {
            "model_id": "test/SixLayer",
            "claim_boundary": "joined_roster_layer_plan_only_no_inference_proof",
            "placement": {
                "assignments": [
                    {
                        "hostname": "seed",
                        "block_range": "0:3",
                        "launch_command": "PYTHONPATH=.:src python -m bloombee.cli.run_server test/SixLayer --new_swarm --block_indices 0:3",
                    },
                    {
                        "hostname": "tail",
                        "block_range": "3:6",
                        "launch_command": "PYTHONPATH=.:src BLOOMBEE_INITIAL_PEERS=/ip4/1/tcp/2/p2p/seed python -m bloombee.cli.run_server test/SixLayer --block_indices 3:6",
                    },
                ]
            },
        },
        "proof_runbooks": {},
    }

    plan = build_proof_orchestration_plan(handoff)

    assert plan["summary"]["forbidden_flags"] == ["launch step tail uses ignored BLOOMBEE_INITIAL_PEERS"]
    assert plan["launch_steps"][1]["ready"] is False
    assert "--initial_peers" in plan["launch_steps"][1]["fix_hint"]
    assert "BLOOMBEE_INITIAL_PEERS" not in " ".join(plan["operator_next_steps"])


def test_proof_orchestrator_cli_writes_no_execution_plan_without_tokens(tmp_path: Path):
    handoff = {
        "claim_boundary": "coordinator_handoff_bundle_only_no_server_started",
        "source": "coordinator_http_handoff_endpoint",
        "token": "moon-token",
        "plan": {
            "model_id": "test/SixLayer",
            "launch_readiness": {"ready_to_start": True, "claim_boundary": "launch_readiness_checklist_only_no_server_started"},
            "placement": {
                "assignments": [
                    {
                        "hostname": "seed",
                        "block_range": "0:6",
                        "launch_command": "PYTHONPATH=.:src python -m bloombee.cli.run_server test/SixLayer --new_swarm --block_indices 0:6",
                    }
                ]
            },
        },
        "proof_runbooks": {
            "full_generation": {
                "claim_boundary": "full_generation_proof_harness_only_no_live_generation",
                "proof_gate": "full_generation",
                "parity_command": "PYTHONPATH=.:src python scripts/text_generation_parity.py --server-maddr '<PASTE_SERVER_0_MULTIADDR>' --out .local/full.json",
                "verify_command": "python mvp_capabilities/full_generation_proof.py verify --evidence .local/full.json --model test/SixLayer",
            }
        },
    }
    handoff_path = tmp_path / "handoff.json"
    out_path = tmp_path / "proof-orchestration.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/proof_orchestrator.py",
            "--handoff-bundle",
            str(handoff_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["claim_boundary"] == "proof_orchestration_plan_only_no_live_inference"
    assert written["live_commands_executed"] is False
    assert "moon-token" not in proc.stdout
    assert "moon-token" not in out_path.read_text(encoding="utf-8")


def test_distributed_inference_docs_server_recipe_uses_initial_peers_cli_flag():
    text = (PROJECT_ROOT / "docs" / "distributed-inference-mvp.md").read_text(encoding="utf-8")

    assert "--initial_peers" in text
    assert "BLOOMBEE_INITIAL_PEERS" not in text


def test_join_handoff_cli_builds_redacted_dashboard_artifact(tmp_path: Path):
    from mvp_capabilities.join_handoff import build_handoff_url, redact_handoff_bundle

    url = build_handoff_url(
        "http://127.0.0.1:8787/",
        token="moon-token",
        model="auto",
        selector_mode="planning",
        max_age_seconds=60,
        include_launch_commands=True,
        include_launch_readiness=True,
        request_count=2,
    )
    assert url == (
        "http://127.0.0.1:8787/handoff?"
        "token=moon-token&model=auto&selector_mode=planning&max_age_seconds=60"
        "&include_launch_commands=1&include_launch_readiness=1&request_count=2"
    )

    raw = {
        "claim_boundary": "coordinator_handoff_bundle_only_no_server_started",
        "source": "coordinator_http_handoff_endpoint",
        "token": "moon-token",
        "offer": {"token": "moon-token", "join_url": "bloombee://join?coordinator=http%3A%2F%2F127.0.0.1%3A8787&token=moon-token"},
        "active": {"active_peers": [{"peer_id": "peer-a", "token": "moon-token"}]},
        "bootstrap_runbook": {
            "shell_script": "python mvp_capabilities/join_client.py --join-url 'bloombee://join?coordinator=http%3A%2F%2F127.0.0.1%3A8787&token=moon-token' --count 180",
        },
        "proof_runbooks": {"multi_block": {"claim_boundary": "multi_block_proof_harness_only_no_live_inference"}},
        "inference_proven": False,
        "can_update_proof_status": False,
    }
    redacted = redact_handoff_bundle(raw, fetched_url=url)
    assert redacted["claim_boundary"] == "coordinator_handoff_bundle_only_no_server_started"
    assert redacted["handoff_fetch_claim_boundary"] == "join_handoff_fetch_only_no_server_started"
    assert redacted["token"] == "***"
    assert redacted["offer"]["token"] == "***"
    assert redacted["bootstrap_runbook"]["shell_script"].count("token=%2A%2A%2A") + redacted["bootstrap_runbook"]["shell_script"].count("token=***") >= 1
    assert "moon-token" not in json.dumps(redacted)
    assert redacted["proof_runbooks"]["multi_block"]["claim_boundary"] == "multi_block_proof_harness_only_no_live_inference"

    raw_path = tmp_path / "raw-handoff.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "handoff-bundle.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "mvp_capabilities" / "join_handoff.py"),
            "--input-json",
            str(raw_path),
            "--fetched-url",
            url,
            "--out",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["source"] == "coordinator_http_handoff_endpoint"
    assert written["handoff_fetch_source"] == "join_handoff_cli"
    assert "moon-token" not in out.read_text(encoding="utf-8")
    assert "moon-token" not in result.stdout


def test_join_layer_plan_builds_launch_runbook_from_active_heartbeats(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat
    from mvp_capabilities.join_layer_plan import build_join_layer_plan

    state_dir = tmp_path / "join-state"
    model = {
        "model_id": "test/SixLayer",
        "num_layers": 6,
        "recommended_min_free_mem_gb": 12,
    }
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="peer-b",
        capabilities={"hostname": "peer-b", "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
        now=100,
    )
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="peer-a",
        capabilities={"hostname": "peer-a", "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
        now=100,
    )
    record_heartbeat(
        state_dir,
        token="other-token",
        peer_id="wrong-token",
        capabilities={"hostname": "wrong-token", "memory": {"free_gb": 48}},
        now=100,
    )
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="stale-peer",
        capabilities={"hostname": "stale-peer", "memory": {"free_gb": 48}},
        now=1,
    )

    plan = build_join_layer_plan(
        state_dir=state_dir,
        token="moon-token",
        model=model,
        now=110,
        max_age_seconds=30,
        include_launch_commands=True,
        base_port=41000,
        dht_prefix="moon-swarm",
    )

    assert plan["claim_boundary"] == "joined_roster_layer_plan_only_no_inference_proof"
    assert plan["heartbeat_claim_boundary"] == "heartbeat_roster_only_no_inference_proof"
    assert plan["placement"]["claim_boundary"] == "placement_plan_only_no_inference_proof"
    assert plan["placement"]["launch_commands_claim_boundary"] == "launch_commands_only_no_server_started"
    assert plan["token"] == "moon-token"
    assert plan["active_peer_count"] == 2
    assert [peer["peer_id"] for peer in plan["active_heartbeats"]] == ["peer-a", "peer-b"]
    assert plan["placement"]["supported"] is True
    assert [(item["hostname"], item["block_range"]) for item in plan["placement"]["assignments"]] == [
        ("peer-a", "0:3"),
        ("peer-b", "3:6"),
    ]
    assert "--block_indices 0:3" in plan["placement"]["assignments"][0]["launch_command"]
    assert "--new_swarm" in plan["placement"]["assignments"][0]["launch_command"]
    assert "<SEED_MULTIADDR_FROM_peer-a>" in plan["placement"]["assignments"][1]["launch_command"]
    assert plan["inference_proven"] is False


def test_join_layer_plan_materializes_launch_readiness_checklist(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat
    from mvp_capabilities.join_layer_plan import build_join_layer_plan, materialize_launch_readiness

    state_dir = tmp_path / "join-state"
    model = {"model_id": "test/SixLayer", "num_layers": 6, "recommended_min_free_mem_gb": 12}
    for peer in ("peer-a", "peer-b"):
        record_heartbeat(
            state_dir,
            token="moon-token",
            peer_id=peer,
            capabilities={"hostname": peer, "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
            now=100,
        )
    plan = build_join_layer_plan(
        state_dir=state_dir,
        token="moon-token",
        model=model,
        now=110,
        max_age_seconds=30,
        include_launch_commands=True,
        base_port=41000,
    )

    readiness = materialize_launch_readiness(plan)

    assert readiness["claim_boundary"] == "launch_readiness_checklist_only_no_server_started"
    assert readiness["ready_to_start"] is False
    assert readiness["server_count"] == 2
    assert readiness["unresolved_placeholders"] == ["<SEED_MULTIADDR_FROM_peer-a>"]
    assert readiness["operator_steps"][0] == "start seed server peer-a and capture its announced multiaddr"
    assert readiness["operator_steps"][1] == "replace <SEED_MULTIADDR_FROM_peer-a> in peer-b command before starting it"
    assert readiness["server_checks"][0]["hostname"] == "peer-a"
    assert readiness["server_checks"][0]["ready"] is True
    assert readiness["server_checks"][0]["role"] == "seed"
    assert readiness["server_checks"][1]["hostname"] == "peer-b"
    assert readiness["server_checks"][1]["ready"] is False
    assert readiness["server_checks"][1]["blocked_by"] == ["unresolved_multiaddr_placeholder"]
    assert readiness["inference_proven"] is False


def test_join_layer_plan_resolves_seed_multiaddrs_before_readiness(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat
    from mvp_capabilities.join_layer_plan import build_join_layer_plan

    state_dir = tmp_path / "join-state"
    model = {"model_id": "test/SixLayer", "num_layers": 6, "recommended_min_free_mem_gb": 12}
    for peer in ("peer-a", "peer-b"):
        record_heartbeat(
            state_dir,
            token="moon-token",
            peer_id=peer,
            capabilities={"hostname": peer, "memory": {"free_gb": 6}, "accelerator": {"device": "mps"}},
            now=100,
        )

    seed_multiaddr = "/ip4/100.64.0.10/tcp/41000/p2p/12D3KooWmoonseed"
    plan = build_join_layer_plan(
        state_dir=state_dir,
        token="moon-token",
        model=model,
        now=110,
        max_age_seconds=30,
        include_launch_commands=True,
        include_launch_readiness=True,
        base_port=41000,
        seed_multiaddrs={"peer-a": seed_multiaddr},
    )

    follower_command = plan["placement"]["assignments"][1]["launch_command"]
    assert "<SEED_MULTIADDR_FROM_peer-a>" not in follower_command
    assert seed_multiaddr in follower_command
    assert plan["placement"]["multiaddr_resolution_claim_boundary"] == "launch_multiaddr_resolution_only_no_server_started"
    assert plan["placement"]["resolved_multiaddr_hosts"] == ["peer-a"]
    assert plan["placement"]["unresolved_multiaddr_placeholders"] == []
    assert plan["launch_readiness"]["ready_to_start"] is True
    assert plan["launch_readiness"]["unresolved_placeholders"] == []
    assert plan["launch_readiness"]["server_checks"][1]["ready"] is True
    assert plan["launch_readiness"]["inference_proven"] is False


def test_join_layer_plan_embeds_launch_readiness_when_requested(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat
    from mvp_capabilities.join_layer_plan import build_join_layer_plan

    state_dir = tmp_path / "join-state"
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="peer-a",
        capabilities={"hostname": "peer-a", "memory": {"free_gb": 48}, "accelerator": {"device": "mps"}},
        now=100,
    )

    plan = build_join_layer_plan(
        state_dir=state_dir,
        token="moon-token",
        model={"model_id": "test/SixLayer", "num_layers": 6, "recommended_min_free_mem_gb": 12},
        now=110,
        max_age_seconds=30,
        include_launch_commands=True,
        include_launch_readiness=True,
        base_port=41000,
    )

    assert plan["launch_readiness"]["ready_to_start"] is True
    assert plan["launch_readiness"]["unresolved_placeholders"] == []
    assert plan["launch_readiness"]["operator_steps"] == ["start seed server peer-a and capture its announced multiaddr"]


def test_join_layer_plan_cli_can_emit_launch_readiness(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat

    state_dir = tmp_path / "join-state"
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="m4pro-a",
        capabilities={"hostname": "m4pro-a", "memory": {"free_gb": 12}, "accelerator": {"device": "mps"}},
        now=100,
    )
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="m4pro-b",
        capabilities={"hostname": "m4pro-b", "memory": {"free_gb": 12}, "accelerator": {"device": "mps"}},
        now=100,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_layer_plan.py",
            "--state-dir",
            str(state_dir),
            "--token",
            "moon-token",
            "--model",
            "Qwen/Qwen3-8B",
            "--now",
            "110",
            "--include-launch-commands",
            "--include-launch-readiness",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["launch_readiness"]["claim_boundary"] == "launch_readiness_checklist_only_no_server_started"
    assert payload["launch_readiness"]["ready_to_start"] is False
    assert payload["launch_readiness"]["unresolved_placeholders"] == ["<SEED_MULTIADDR_FROM_m4pro-a>"]
    assert payload["launch_readiness"]["inference_proven"] is False


def test_join_layer_plan_cli_accepts_seed_multiaddr_for_ready_runbook(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat

    state_dir = tmp_path / "join-state"
    for peer in ("m4pro-a", "m4pro-b"):
        record_heartbeat(
            state_dir,
            token="moon-token",
            peer_id=peer,
            capabilities={"hostname": peer, "memory": {"free_gb": 12}, "accelerator": {"device": "mps"}},
            now=100,
        )
    seed_multiaddr = "/ip4/100.64.0.20/tcp/31337/p2p/12D3KooWm4proseed"

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_layer_plan.py",
            "--state-dir",
            str(state_dir),
            "--token",
            "moon-token",
            "--model",
            "Qwen/Qwen3-8B",
            "--now",
            "110",
            "--include-launch-commands",
            "--include-launch-readiness",
            "--seed-multiaddr",
            f"m4pro-a={seed_multiaddr}",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["placement"]["multiaddr_resolution_claim_boundary"] == "launch_multiaddr_resolution_only_no_server_started"
    assert payload["placement"]["resolved_multiaddr_hosts"] == ["m4pro-a"]
    assert payload["placement"]["unresolved_multiaddr_placeholders"] == []
    assert payload["launch_readiness"]["ready_to_start"] is True
    assert seed_multiaddr in payload["placement"]["assignments"][1]["launch_command"]


def test_join_layer_plan_builds_from_http_active_payload():
    from mvp_capabilities.join_layer_plan import build_join_layer_plan_from_active_payload, fetch_active_heartbeats

    captured_urls: list[str] = []

    def fake_fetcher(url: str) -> dict[str, object]:
        captured_urls.append(url)
        return {
            "token": "moon-token",
            "active_peers": [
                {
                    "peer_id": "peer-http-b",
                    "token": "moon-token",
                    "timestamp": 100,
                    "capabilities": {"hostname": "peer-http-b", "memory": {"free_gb": 6}},
                    "claim_boundary": "heartbeat_only_no_inference_proof",
                },
                {
                    "peer_id": "peer-http-a",
                    "token": "moon-token",
                    "timestamp": 100,
                    "capabilities": {"hostname": "peer-http-a", "memory": {"free_gb": 6}},
                    "claim_boundary": "heartbeat_only_no_inference_proof",
                },
            ],
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }

    active_payload = fetch_active_heartbeats(
        "http://coordinator.local:8787",
        token="moon-token",
        now=110,
        max_age_seconds=30,
        fetcher=fake_fetcher,
    )
    plan = build_join_layer_plan_from_active_payload(
        active_payload,
        model={"model_id": "test/SixLayer", "num_layers": 6, "recommended_min_free_mem_gb": 12},
        include_launch_commands=True,
        base_port=41000,
    )

    assert captured_urls == ["http://coordinator.local:8787/active?token=moon-token&max_age_seconds=30&now=110"]
    assert plan["source"] == "coordinator_http_active"
    assert plan["claim_boundary"] == "joined_roster_layer_plan_only_no_inference_proof"
    assert plan["heartbeat_claim_boundary"] == "heartbeat_roster_only_no_inference_proof"
    assert [peer["peer_id"] for peer in plan["active_heartbeats"]] == ["peer-http-a", "peer-http-b"]
    assert [(item["hostname"], item["block_range"]) for item in plan["placement"]["assignments"]] == [
        ("peer-http-a", "0:3"),
        ("peer-http-b", "3:6"),
    ]
    assert plan["inference_proven"] is False


def test_join_layer_plan_cli_accepts_coordinator_url(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    from mvp_capabilities import join_layer_plan

    def fake_fetch_active_heartbeats(coordinator_url: str, *, token: str, now: int | None = None, max_age_seconds: int = 30, fetcher=None):
        assert coordinator_url == "http://coordinator.local:8787"
        assert token == "moon-token"
        assert max_age_seconds == 30
        return {
            "token": token,
            "active_peers": [
                {
                    "peer_id": "http-peer",
                    "token": token,
                    "timestamp": 100,
                    "capabilities": {"hostname": "http-peer", "memory": {"free_gb": 48}, "accelerator": {"device": "mps"}},
                    "claim_boundary": "heartbeat_only_no_inference_proof",
                }
            ],
            "claim_boundary": "heartbeat_roster_only_no_inference_proof",
        }

    monkeypatch.setattr(join_layer_plan, "fetch_active_heartbeats", fake_fetch_active_heartbeats)
    rc = join_layer_plan.main(
        [
            "--coordinator-url",
            "http://coordinator.local:8787",
            "--token",
            "moon-token",
            "--model",
            "Qwen/Qwen3-8B",
            "--max-age-seconds",
            "30",
            "--include-launch-commands",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "coordinator_http_active"
    assert payload["claim_boundary"] == "joined_roster_layer_plan_only_no_inference_proof"
    assert payload["active_peer_count"] == 1
    assert payload["placement"]["assignments"][0]["hostname"] == "http-peer"
    assert payload["inference_proven"] is False


def test_join_layer_plan_cli_reads_registry_model_and_active_state(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat

    state_dir = tmp_path / "join-state"
    record_heartbeat(
        state_dir,
        token="moon-token",
        peer_id="m4pro",
        capabilities={"hostname": "m4pro", "memory": {"free_gb": 48}, "accelerator": {"device": "mps"}},
        now=100,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_layer_plan.py",
            "--state-dir",
            str(state_dir),
            "--token",
            "moon-token",
            "--model",
            "Qwen/Qwen3-8B",
            "--now",
            "110",
            "--max-age-seconds",
            "30",
            "--include-launch-commands",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "joined_roster_layer_plan_only_no_inference_proof"
    assert payload["model_id"] == "Qwen/Qwen3-8B"
    assert payload["active_peer_count"] == 1
    assert payload["placement"]["assignments"][0]["hostname"] == "m4pro"
    assert payload["placement"]["launch_commands_claim_boundary"] == "launch_commands_only_no_server_started"
    assert payload["inference_proven"] is False


def test_join_http_server_rejects_bad_heartbeat_json(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_post

    status, payload = handle_post("/heartbeat", body=b'{"token":"moon-token"}', state_dir=tmp_path / "state")
    assert status == 400
    assert payload["error"] == "missing required heartbeat fields"
    assert payload["claim_boundary"] == "coordinator_error_no_inference_proof"


def test_join_http_server_bootstrap_runbook_keeps_fresh_laptop_heartbeating(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get

    status, payload = handle_get(
        "/bootstrap?token=moon-token&count=180&interval_seconds=10&now=1000&ttl_seconds=600",
        state_dir=tmp_path / "state",
        coordinator="http://m4pro.local:8787",
    )

    script = payload["shell_script"]
    assert status == 200
    assert payload["source"] == "coordinator_http_bootstrap_endpoint"
    assert payload["claim_boundary"] == "coordinator_bootstrap_runbook_only_no_server_started"
    assert payload["offer"]["join_url"] == "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token"
    assert payload["heartbeat_loop"]["count"] == 180
    assert payload["heartbeat_loop"]["interval_seconds"] == 10.0
    assert "peer_scan.py --out" in script
    assert "join_client.py" in script
    assert "--count 180" in script
    assert "--interval-seconds 10" in script
    assert "bloombee.cli.run_server" not in script
    assert payload["inference_proven"] is False
    assert payload["can_update_proof_status"] is False


def test_join_http_server_bootstrap_requires_token(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get

    status, payload = handle_get("/bootstrap", state_dir=tmp_path / "state", coordinator="http://m4pro.local:8787")

    assert status == 400
    assert payload["error"] == "missing token"
    assert payload["claim_boundary"] == "coordinator_error_no_inference_proof"


def test_join_http_server_bootstrap_sh_returns_plain_shell_script(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get_text

    status, content_type, body = handle_get_text(
        "/bootstrap.sh?token=moon-token&count=2&interval_seconds=0&now=1000&ttl_seconds=600",
        state_dir=tmp_path / "state",
        coordinator="http://m4pro.local:8787",
    )

    script = body.decode("utf-8")
    assert status == 200
    assert content_type == "text/x-shellscript; charset=utf-8"
    assert script.startswith("#!/usr/bin/env bash")
    assert "# claim_boundary: coordinator_bootstrap_runbook_only_no_server_started" in script
    assert "# inference_proven: false" in script
    assert "peer_scan.py --out" in script
    assert "join_client.py" in script
    assert "--count 2" in script
    assert "--interval-seconds 0" in script
    assert "bloombee.cli.run_server" not in script


def test_join_http_server_bootstrap_sh_requires_token(tmp_path: Path):
    from mvp_capabilities.join_http_server import handle_get_text

    status, content_type, body = handle_get_text("/bootstrap.sh", state_dir=tmp_path / "state", coordinator="http://m4pro.local:8787")

    assert status == 400
    assert content_type == "text/plain; charset=utf-8"
    assert "missing token" in body.decode("utf-8")
    assert "coordinator_error_no_inference_proof" in body.decode("utf-8")


def test_draft_provider_contract_counts_accept_reject_and_fallback():
    from mvp_capabilities.draft_provider import StaticDraftProvider, build_draft_provider_report

    report = build_draft_provider_report(
        provider=StaticDraftProvider((10, 11, 12), provider_id="phone-fake"),
        prompt_tokens=(1, 2, 3),
        verifier_tokens=(10, 99, 100),
        max_draft_tokens=3,
    )

    assert report["claim_boundary"] == "draft_provider_contract_only_no_generation_proof"
    assert report["provider"]["phone_compatible_interface"] is True
    assert report["provider"]["can_serve_transformer_blocks"] is False
    assert report["proposal"]["draft_tokens"] == [10, 11, 12]
    assert report["verdict"]["accepted_tokens"] == [10]
    assert report["verdict"]["rejected_tokens"] == [11, 12]
    assert report["verdict"]["verifier_fallback_token"] == 99
    assert report["verdict"]["committed_tokens"] == [10, 99]
    assert report["dashboard_counters"] == {"proposed": 3, "accepted": 1, "rejected": 2, "acceptance_rate": 0.333333}
    assert report["generation_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_proof_status"] is False


def test_draft_provider_hash_fake_is_deterministic():
    from mvp_capabilities.draft_provider import DeterministicHashDraftProvider

    provider = DeterministicHashDraftProvider(vocab_size=128, seed="moon")
    first = provider.propose((4, 5, 6), 4)
    second = provider.propose((4, 5, 6), 4)

    assert first.draft_tokens == second.draft_tokens
    assert len(first.draft_tokens) == 4
    assert all(0 <= token < 128 for token in first.draft_tokens)
    assert first.provider_kind == "deterministic_hash_fake"


def test_draft_provider_cli_outputs_dashboard_counters():
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/draft_provider.py",
            "--prompt-tokens",
            "1,2,3",
            "--draft-tokens",
            "5,6,7",
            "--verifier-tokens",
            "5,6,8",
            "--max-draft-tokens",
            "3",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source"] == "draft_provider.py"
    assert payload["dashboard_counters"] == {"proposed": 3, "accepted": 2, "rejected": 1, "acceptance_rate": 0.666667}
    assert payload["verdict"]["verifier_fallback_token"] == 8
    assert payload["generation_proven"] is False


def test_draft_provider_bridge_handles_static_request():
    from mvp_capabilities.draft_provider_bridge import handle_draft_bridge_request

    response = handle_draft_bridge_request(
        {
            "request_id": "req-1",
            "prompt_tokens": [1, 2, 3],
            "draft_tokens": [5, 6, 7],
            "verifier_tokens": [5, 9, 10],
            "max_draft_tokens": 3,
            "provider_id": "phone-stdio-fake",
        }
    )

    assert response["ok"] is True
    assert response["request_id"] == "req-1"
    assert response["claim_boundary"] == "draft_provider_stdio_bridge_only_no_generation_proof"
    assert response["draft_provider_claim_boundary"] == "draft_provider_contract_only_no_generation_proof"
    assert response["dashboard_counters"] == {"proposed": 3, "accepted": 1, "rejected": 2, "acceptance_rate": 0.333333}
    assert response["report"]["provider"]["provider_id"] == "phone-stdio-fake"
    assert response["generation_proven"] is False
    assert response["can_update_proof_status"] is False


def test_draft_provider_bridge_rejects_malformed_payload():
    from mvp_capabilities.draft_provider_bridge import handle_draft_bridge_request

    response = handle_draft_bridge_request({"request_id": "bad", "prompt_tokens": "1,2,3"})

    assert response["ok"] is False
    assert response["request_id"] == "bad"
    assert response["claim_boundary"] == "draft_provider_stdio_bridge_error_no_generation_proof"
    assert "prompt_tokens must be a JSON list" in response["error"]
    assert response["inference_proven"] is False


def test_draft_provider_bridge_serve_stdio_outputs_jsonl():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/draft_provider_bridge.py", "serve-stdio"],
        input=json.dumps(
            {
                "request_id": "stdio-1",
                "prompt_tokens": [1],
                "draft_tokens": [2, 3],
                "verifier_tokens": [2, 4],
                "max_draft_tokens": 2,
            }
        )
        + "\n",
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["ok"] is True
    assert lines[0]["request_id"] == "stdio-1"
    assert lines[0]["dashboard_counters"] == {"acceptance_rate": 0.5, "accepted": 1, "proposed": 2, "rejected": 1}


def test_speculative_decode_plan_keeps_verifier_authoritative_and_phones_draft_only():
    from mvp_capabilities.speculative_decode_plan import build_speculative_decode_plan

    route_decision = {
        "picked": {
            "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "claim_level": "experimental",
            "selector_mode": "showcase-attempt",
            "proof_status": {"full_generation": "pending"},
        }
    }
    peers = [
        {"hostname": "phone-a", "mobile": {"is_mobile": True, "runtime": "termux"}, "memory": {"free_gb": 3}},
        {"hostname": "m4pro", "accelerator": {"device": "mps"}, "memory": {"free_gb": 34}},
    ]

    plan = build_speculative_decode_plan(
        verifier_route=route_decision,
        peers=peers,
        draft_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_draft_tokens=4,
        acceptance_window=2,
    )

    assert plan["claim_boundary"] == "speculative_decode_plan_only_no_generation_proof"
    assert plan["verifier"]["model_id"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert plan["verifier"]["authoritative"] is True
    assert plan["draft"]["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert plan["draft"]["phone_candidates"][0]["hostname"] == "phone-a"
    assert plan["phone_policy"]["phones_as_block_workers"] is False
    assert plan["correctness_contract"]["accepted_tokens_require_verifier_match"] is True
    assert plan["execution_plan"][0]["stage"] == "draft_propose"
    assert plan["inference_proven"] is False
    assert plan["can_update_proof_status"] is False


def test_speculative_decode_plan_cli_outputs_json(tmp_path: Path):
    route_path = tmp_path / "route.json"
    peers_path = tmp_path / "peers.json"
    route_path.write_text(json.dumps({"picked": {"model_id": "Qwen/Qwen3-8B", "claim_level": "experimental"}}), encoding="utf-8")
    peers_path.write_text(json.dumps([{"hostname": "phone-a", "mobile": {"is_mobile": True}}]), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/speculative_decode_plan.py",
            "--route-json",
            str(route_path),
            "--peers-json",
            str(peers_path),
            "--draft-model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--max-draft-tokens",
            "3",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "speculative_decode_plan_only_no_generation_proof"
    assert payload["draft"]["max_draft_tokens"] == 3
    assert payload["draft"]["phone_candidates"][0]["hostname"] == "phone-a"
    assert payload["verifier"]["authoritative"] is True


def test_join_http_server_speculative_endpoint_and_handoff_include_no_generation_plan(tmp_path: Path):
    from mvp_capabilities.join_coordinator import record_heartbeat
    from mvp_capabilities.join_http_server import handle_get

    state = tmp_path / "state"
    record_heartbeat(
        state,
        token="moon-token",
        peer_id="phone-a",
        capabilities={"hostname": "phone-a", "mobile": {"is_mobile": True, "runtime": "termux"}, "memory": {"free_gb": 3}},
        now=1_000,
    )
    record_heartbeat(
        state,
        token="moon-token",
        peer_id="m4pro",
        capabilities={"hostname": "m4pro", "accelerator": {"device": "mps"}, "memory": {"free_gb": 80}},
        now=1_000,
    )

    status, payload = handle_get(
        "/speculative?token=moon-token&model=auto&selector_mode=showcase-attempt&draft_model=TinyLlama/TinyLlama-1.1B-Chat-v1.0&max_draft_tokens=4&now=1000",
        state_dir=state,
        coordinator="http://m4pro.local:8787",
        registry=REGISTRY_PATH,
    )

    assert status == 200
    assert payload["claim_boundary"] == "coordinator_speculative_plan_only_no_generation_proof"
    assert payload["speculative_plan"]["claim_boundary"] == "speculative_decode_plan_only_no_generation_proof"
    assert payload["speculative_plan"]["draft"]["phone_candidates"][0]["hostname"] == "phone-a"
    assert payload["speculative_plan"]["verifier"]["authoritative"] is True
    assert payload["inference_proven"] is False

    handoff_status, handoff = handle_get(
        "/handoff?token=moon-token&model=auto&selector_mode=showcase-attempt&now=1000",
        state_dir=state,
        coordinator="http://m4pro.local:8787",
        registry=REGISTRY_PATH,
    )
    assert handoff_status == 200
    assert handoff["speculative_plan"]["claim_boundary"] == "speculative_decode_plan_only_no_generation_proof"
    assert handoff["speculative_plan"]["correctness_contract"]["accepted_tokens_require_verifier_match"] is True


def test_join_client_parses_join_url_and_builds_heartbeat_request(tmp_path: Path):
    from mvp_capabilities.join_client import build_heartbeat_request, build_heartbeat_payload, parse_join_url

    capabilities_path = tmp_path / "fresh-peer.json"
    capabilities_path.write_text(
        json.dumps({"hostname": "fresh-peer", "memory": {"free_gb": 20}, "accelerator": {"device": "mps"}}),
        encoding="utf-8",
    )

    join = parse_join_url("bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token")
    payload = build_heartbeat_payload(join, capabilities_path=capabilities_path, now=1_000)
    request = build_heartbeat_request(join, payload)

    assert join == {"coordinator": "http://m4pro.local:8787", "token": "moon-token"}
    assert payload["token"] == "moon-token"
    assert payload["peer_id"] == "fresh-peer"
    assert payload["capabilities"]["memory"]["free_gb"] == 20
    assert payload["claim_boundary"] == "join_client_request_only_no_inference_proof"
    assert request.full_url == "http://m4pro.local:8787/heartbeat"
    assert request.get_method() == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert json.loads(request.data.decode("utf-8"))["peer_id"] == "fresh-peer"


def test_join_client_rejects_malformed_join_url():
    from mvp_capabilities.join_client import parse_join_url

    with pytest.raises(ValueError, match="join URL must include coordinator and token"):
        parse_join_url("bloombee://join?token=moon-token")


def test_join_client_post_heartbeat_uses_injectable_urlopen(tmp_path: Path):
    from mvp_capabilities.join_client import post_heartbeat

    capabilities_path = tmp_path / "fresh-peer.json"
    capabilities_path.write_text(json.dumps({"hostname": "fresh-peer"}), encoding="utf-8")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"peer_id":"fresh-peer","claim_boundary":"heartbeat_only_no_inference_proof"}'

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    result = post_heartbeat(
        "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
        capabilities_path=capabilities_path,
        timeout=7,
        urlopen_fn=fake_urlopen,
    )

    assert captured["url"] == "http://m4pro.local:8787/heartbeat"
    assert captured["body"]["token"] == "moon-token"
    assert captured["timeout"] == 7
    assert result["server_response"]["peer_id"] == "fresh-peer"
    assert result["claim_boundary"] == "join_client_post_only_no_inference_proof"


def test_join_client_run_heartbeat_loop_reposts_with_sleep_hook(tmp_path: Path):
    from mvp_capabilities.join_client import run_heartbeat_loop

    capabilities_path = tmp_path / "fresh-peer.json"
    capabilities_path.write_text(json.dumps({"hostname": "fresh-peer"}), encoding="utf-8")
    captured_bodies = []
    slept = []
    now_values = iter([1_000, 1_010, 1_020])

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok":true,"claim_boundary":"heartbeat_only_no_inference_proof"}'

    def fake_urlopen(request, timeout=0):
        captured_bodies.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    report = run_heartbeat_loop(
        "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
        capabilities_path=capabilities_path,
        count=3,
        interval_seconds=2.5,
        timeout=7,
        now_fn=lambda: next(now_values),
        sleep_fn=slept.append,
        urlopen_fn=fake_urlopen,
    )

    assert [body["now"] for body in captured_bodies] == [1_000, 1_010, 1_020]
    assert slept == [2.5, 2.5]
    assert report["heartbeat_count"] == 3
    assert len(report["results"]) == 3
    assert report["results"][0]["server_response"]["ok"] is True
    assert report["claim_boundary"] == "join_client_heartbeat_loop_only_no_inference_proof"
    assert report["inference_proven"] is False


def test_join_client_cli_dry_run_outputs_request_json(tmp_path: Path):
    import subprocess
    import sys

    capabilities_path = tmp_path / "fresh-peer.json"
    capabilities_path.write_text(json.dumps({"hostname": "fresh-peer"}), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_client.py",
            "--join-url",
            "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
            "--capabilities",
            str(capabilities_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["url"] == "http://m4pro.local:8787/heartbeat"
    assert payload["body"]["peer_id"] == "fresh-peer"
    assert payload["claim_boundary"] == "join_client_dry_run_only_no_inference_proof"


def test_join_card_renders_svg_with_metadata_and_claim_boundary():
    from mvp_capabilities.join_card import render_join_card_svg

    join_url = "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token"
    svg = render_join_card_svg(join_url, title="BloomBee Join", expires_at=1120)

    assert svg.startswith("<svg ")
    assert "BloomBee Join" in svg
    assert "bloombee://join?" in svg
    assert "moon-token" in svg
    assert "join_card_visual_only_no_inference_proof" in svg
    assert "scanner_interop_unproven" in svg
    assert "data-join-url=" in svg
    assert svg.count("<rect") > 40


def test_join_card_sidecar_exposes_exact_copy_paste_join_artifacts():
    from mvp_capabilities.join_card import render_join_card_sidecar, render_join_card_sidecar_text

    join_url = "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token"
    sidecar = render_join_card_sidecar(join_url, title="BloomBee Join", expires_at=1120)
    text = render_join_card_sidecar_text(sidecar)

    assert sidecar["claim_boundary"] == "join_card_sidecar_exact_url_no_scanner_proof"
    assert sidecar["scanner_status"] == "scanner_interop_unproven"
    assert sidecar["join_url"] == join_url
    assert sidecar["url_text_copyable"] is True
    assert sidecar["scanner_interop_proven"] is False
    assert sidecar["inference_proven"] is False
    assert "join_client.py" in sidecar["join_client_command"]
    assert join_url in sidecar["join_client_command"]
    assert join_url in text
    assert "Visual grid is not a proven QR code" in text


def test_join_card_cli_writes_svg_file(tmp_path: Path):
    import subprocess
    import sys

    out = tmp_path / "join-card.svg"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_card.py",
            "--join-url",
            "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
            "--title",
            "Moonlit Join",
            "--expires-at",
            "1120",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["out"] == str(out)
    assert payload["claim_boundary"] == "join_card_visual_only_no_inference_proof"
    text = out.read_text(encoding="utf-8")
    assert "Moonlit Join" in text
    assert "scanner_interop_unproven" in text


def test_join_card_cli_can_write_json_and_text_sidecars(tmp_path: Path):
    import subprocess
    import sys

    out = tmp_path / "join-card.svg"
    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/join_card.py",
            "--join-url",
            "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
            "--title",
            "Moonlit Join",
            "--expires-at",
            "1234",
            "--out",
            str(out),
            "--write-sidecars",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["sidecar_json"].endswith("join-card.join.json")
    assert payload["sidecar_text"].endswith("join-card.join.txt")
    sidecar = json.loads(Path(payload["sidecar_json"]).read_text(encoding="utf-8"))
    sidecar_text = Path(payload["sidecar_text"]).read_text(encoding="utf-8")
    assert sidecar["claim_boundary"] == "join_card_sidecar_exact_url_no_scanner_proof"
    assert sidecar["join_url"] in sidecar_text
    assert sidecar["expires_at"] == 1234
    assert sidecar["scanner_interop_proven"] is False


def test_join_qr_preflight_reports_missing_dependencies_fail_closed():
    from mvp_capabilities.join_qr_preflight import check_qr_scanner_readiness

    report = check_qr_scanner_readiness(availability={"qrcode": False, "PIL": False, "cv2": False, "pyzbar": False, "segno": False})

    assert report["claim_boundary"] == "qr_scanner_preflight_only_no_scanner_proof"
    assert report["scanner_status"] == "scanner_interop_blocked_missing_dependencies"
    assert report["ready_for_scanner_proof"] is False
    assert report["can_replace_visual_grid"] is False
    assert report["missing_encoder_options"] == ["qrcode+PIL", "segno"]
    assert report["missing_decoder_options"] == ["cv2", "pyzbar"]
    assert report["inference_proven"] is False


def test_join_qr_preflight_marks_ready_only_when_encoder_and_decoder_available():
    from mvp_capabilities.join_qr_preflight import check_qr_scanner_readiness

    report = check_qr_scanner_readiness(availability={"qrcode": True, "PIL": True, "cv2": True, "pyzbar": False, "segno": False})

    assert report["scanner_status"] == "scanner_interop_preflight_ready_no_scan_yet"
    assert report["ready_for_scanner_proof"] is True
    assert report["can_replace_visual_grid"] is False
    assert report["next_step"] == "generate a true QR artifact, decode it with an installed scanner library, and compare the decoded URL exactly"


def test_join_qr_preflight_cli_outputs_fail_closed_json():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/join_qr_preflight.py", "--json"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "qr_scanner_preflight_only_no_scanner_proof"
    assert isinstance(payload["ready_for_scanner_proof"], bool)
    assert payload["scanner_status"] in {
        "scanner_interop_blocked_missing_dependencies",
        "scanner_interop_preflight_ready_no_scan_yet",
    }
    if payload["ready_for_scanner_proof"]:
        assert payload["scanner_status"] == "scanner_interop_preflight_ready_no_scan_yet"
        assert payload["missing_encoder_options"] == []
        assert payload["missing_decoder_options"] == []
    else:
        assert payload["scanner_status"] == "scanner_interop_blocked_missing_dependencies"
    assert payload["inference_proven"] is False


def test_join_qr_proof_reports_missing_dependencies_fail_closed(tmp_path: Path):
    from mvp_capabilities.join_qr_proof import run_qr_artifact_proof

    report = run_qr_artifact_proof(
        "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
        tmp_path / "join.png",
        availability={"qrcode": False, "PIL": False, "cv2": False, "pyzbar": False, "segno": False},
    )

    assert report["claim_boundary"] == "qr_artifact_exact_decode_proof_no_physical_scanner_no_inference"
    assert report["scanner_status"] == "scanner_interop_blocked_missing_dependencies"
    assert report["local_exact_decode_proven"] is False
    assert report["physical_scanner_interop_proven"] is False
    assert report["scanner_interop_proven"] is False
    assert report["can_update_proof_status"] is False
    assert not (tmp_path / "join.png").exists()


def test_join_qr_proof_exact_decode_redacts_token_but_matches_hash(tmp_path: Path):
    from mvp_capabilities.join_qr_proof import run_qr_artifact_proof

    join_url = "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token"
    out = tmp_path / "join.png"

    def fake_encoder(value: str, artifact: Path) -> None:
        artifact.write_text(value, encoding="utf-8")

    def fake_decoder(artifact: Path) -> str:
        return artifact.read_text(encoding="utf-8")

    report = run_qr_artifact_proof(join_url, out, encoder=fake_encoder, decoder=fake_decoder)

    assert report["scanner_status"] == "local_qr_exact_decode_proven"
    assert report["exact_match"] is True
    assert report["local_exact_decode_proven"] is True
    assert report["physical_scanner_interop_proven"] is False
    assert report["expected_url_redacted"].endswith("token=%2A%2A%2A")
    assert report["decoded_url_redacted"] == report["expected_url_redacted"]
    assert report["expected_url_sha256"] == report["decoded_url_sha256"]
    assert "moon-token" not in json.dumps(report)


def test_join_qr_proof_detects_decoder_mismatch(tmp_path: Path):
    from mvp_capabilities.join_qr_proof import run_qr_artifact_proof

    def fake_encoder(value: str, artifact: Path) -> None:
        artifact.write_text(value, encoding="utf-8")

    report = run_qr_artifact_proof(
        "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
        tmp_path / "join.png",
        encoder=fake_encoder,
        decoder=lambda _artifact: "bloombee://join?token=wrong",
    )

    assert report["scanner_status"] == "local_qr_exact_decode_failed"
    assert report["exact_match"] is False
    assert report["local_exact_decode_proven"] is False
    assert report["decoded_url_sha256"] != report["expected_url_sha256"]


def test_chain_scheduler_builds_multi_request_waves_from_joined_plan():
    from mvp_capabilities.chain_scheduler import build_chain_schedule

    joined_plan = {
        "claim_boundary": "joined_roster_layer_plan_only_no_inference_proof",
        "model_id": "Qwen/Qwen3-8B",
        "placement": {
            "supported": True,
            "assignments": [
                {"hostname": "m4pro-seed", "block_range": "0:18", "assigned_layers": 18, "port": 41000},
                {"hostname": "m4pro-tail", "block_range": "18:36", "assigned_layers": 18, "port": 41001},
            ],
        },
    }

    schedule = build_chain_schedule(joined_plan, request_count=5, max_parallel_per_peer=2, prompt_tokens=32, max_new_tokens=16)

    assert schedule["claim_boundary"] == "chain_scheduler_plan_only_no_inference_proof"
    assert schedule["scheduler_status"] == "ready_to_rehearse_no_live_requests"
    assert schedule["model_id"] == "Qwen/Qwen3-8B"
    assert schedule["request_count"] == 5
    assert schedule["stage_count"] == 2
    assert [wave["request_ids"] for wave in schedule["waves"]] == [["req-000", "req-001"], ["req-002", "req-003"], ["req-004"]]
    assert schedule["request_chains"][0]["stages"][0]["hostname"] == "m4pro-seed"
    assert schedule["request_chains"][0]["stages"][1]["block_range"] == "18:36"
    assert schedule["peer_health"]["m4pro-seed"]["scheduled_requests"] == 5
    assert schedule["peer_health"]["m4pro-tail"]["peak_parallel_requests"] == 2
    assert schedule["peer_health"]["m4pro-tail"]["utilization_fraction"] == 0.83
    assert schedule["token_budget"]["tokens_per_request"] == 48
    assert schedule["inference_proven"] is False
    assert schedule["live_requests_sent"] is False


def test_chain_scheduler_blocks_unplaceable_joined_plan():
    from mvp_capabilities.chain_scheduler import build_chain_schedule

    schedule = build_chain_schedule(
        {"model_id": "Qwen/Qwen3-8B", "placement": {"supported": False, "assignments": []}},
        request_count=3,
    )

    assert schedule["scheduler_status"] == "blocked_no_supported_layer_plan"
    assert schedule["waves"] == []
    assert schedule["request_chains"] == []
    assert schedule["inference_proven"] is False
    assert schedule["next_step"] == "produce a supported joined layer plan before scheduling live requests"


def test_chain_scheduler_cli_reads_joined_plan_json(tmp_path: Path):
    joined_plan = {
        "model_id": "Qwen/Qwen3-8B",
        "placement": {
            "supported": True,
            "assignments": [
                {"hostname": "joined-a", "block_range": "0:12", "assigned_layers": 12},
                {"hostname": "joined-b", "block_range": "12:24", "assigned_layers": 12},
                {"hostname": "joined-c", "block_range": "24:36", "assigned_layers": 12},
            ],
        },
    }
    plan_path = tmp_path / "joined-layer-plan.json"
    plan_path.write_text(json.dumps(joined_plan), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/chain_scheduler.py",
            "--joined-layer-plan",
            str(plan_path),
            "--request-count",
            "4",
            "--max-parallel-per-peer",
            "2",
            "--prompt-tokens",
            "10",
            "--max-new-tokens",
            "5",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "chain_scheduler_plan_only_no_inference_proof"
    assert payload["stage_count"] == 3
    assert payload["wave_count"] == 2
    assert payload["peer_health"]["joined-c"]["scheduled_tokens"] == 60
    assert payload["can_update_proof_status"] is False


# ── multi_block_diagnostics ──────────────────────────────────────────


def test_multi_block_diagnostics_surfaces_per_server_state_and_coverage():
    from mvp_capabilities.multi_block_diagnostics import build_multi_block_diagnostics

    server_logs = [
        "[INFO] Announced that blocks range(0, 3) are joining\n[INFO] Started\n[INFO] rpc_forward(blocks=0:3, remote_peer=...abc)\n",
        "[INFO] Announced that blocks range(3, 6) are joining\n[INFO] Started\n[INFO] rpc_backward(blocks=3:6, remote_peer=...abc)\n",
    ]
    client_log = (
        '[direct] RESULT: {"ok": true, "model": "test/SixLayer", "block_range": [0, 6], '
        '"forward_seconds": 0.47, "backward_seconds": 0.43, '
        '"outputs_finite": true, "grad_finite": true}\n'
    )

    report = build_multi_block_diagnostics(
        model_id="test/SixLayer",
        block_ranges=["0:3", "3:6"],
        server_logs=server_logs,
        client_log=client_log,
    )

    assert report["claim_boundary"] == "multi_block_diagnostics_observability_only_no_inference_proof"
    assert report["model_id"] == "test/SixLayer"
    assert report["combined_block_range"] == "0:6"
    assert report["server_count"] == 2
    assert report["coverage"]["covered_layers"] == 6
    assert report["coverage"]["missing_layers"] == 0
    assert report["coverage"]["full_coverage"] is True

    assert report["servers"][0]["block_range"] == "0:3"
    assert report["servers"][0]["started"] is True
    assert report["servers"][0]["announced_block_range"] is True
    assert report["servers"][0]["has_rpc_evidence"] is True
    assert report["servers"][0]["health"] == "healthy"

    assert report["servers"][1]["block_range"] == "3:6"
    assert report["servers"][1]["health"] == "healthy"

    assert report["client_result"]["ok"] is True
    assert report["client_result"]["block_range"] == [0, 6]
    assert report["client_result"]["forward_seconds"] == 0.47

    assert report["summary"]["healthy_servers"] == 2
    assert report["summary"]["unhealthy_servers"] == 0
    assert report["summary"]["status"] == "all_servers_healthy_client_passed"
    assert report["inference_proven"] is False
    assert report["can_update_proof_status"] is False


def test_multi_block_diagnostics_flags_missing_server_and_gapped_coverage():
    from mvp_capabilities.multi_block_diagnostics import build_multi_block_diagnostics

    # Only one server log for "0:3" — "3:6" is missing entirely.
    server_logs = [
        "[INFO] Announced that blocks range(0, 3) are joining\n[INFO] Started\n[INFO] rpc_forward(blocks=0:3, remote_peer=...abc)\n",
    ]
    client_log = (
        '[direct] RESULT: {"ok": true, "model": "test/SixLayer", "block_range": [0, 6], '
        '"forward_seconds": 1.2, "backward_seconds": 0.0, '
        '"outputs_finite": true, "grad_finite": true}\n'
    )

    report = build_multi_block_diagnostics(
        model_id="test/SixLayer",
        block_ranges=["0:3", "3:6"],
        server_logs=server_logs,
        client_log=client_log,
    )

    assert report["server_count"] == 2
    assert report["servers"][0]["health"] == "healthy"
    assert report["servers"][1]["health"] == "unhealthy"
    assert report["servers"][1]["errors"] == [
        "server did not reach Started state",
        "server did not announce block range 3:6",
        "server did not record rpc evidence for 3:6",
    ]
    assert report["summary"]["healthy_servers"] == 1
    assert report["summary"]["unhealthy_servers"] == 1
    assert report["summary"]["status"] == "unhealthy_servers_detected"
    assert report["coverage"]["covered_layers"] == 3
    assert report["coverage"]["missing_layers"] == 3
    assert report["coverage"]["full_coverage"] is False
    assert report["inference_proven"] is False
    assert len(report["operator_actions"]) >= 1
    assert "server 1 (3:6)" in report["operator_actions"][0].lower()


def test_multi_block_diagnostics_handles_failed_client_result():
    from mvp_capabilities.multi_block_diagnostics import build_multi_block_diagnostics

    server_logs = [
        "[INFO] Announced that blocks range(0, 3) are joining\n[INFO] Started\n",
        "[INFO] Announced that blocks range(3, 6) are joining\n[INFO] Started\n",
    ]
    client_log = "RuntimeError: DHT bootstrap failed before RPC\n"

    report = build_multi_block_diagnostics(
        model_id="test/SixLayer",
        block_ranges=["0:3", "3:6"],
        server_logs=server_logs,
        client_log=client_log,
    )

    assert report["servers"][0]["has_rpc_evidence"] is False
    assert report["servers"][1]["has_rpc_evidence"] is False
    assert report["client_result"] is None
    assert report["summary"]["status"] == "client_connection_failed"
    assert report["coverage"]["full_coverage"] is False


def test_multi_block_diagnostics_cli_outputs_operator_report(tmp_path: Path):
    client_log = tmp_path / "client.log"
    client_log.write_text(
        '[direct] RESULT: {"ok": true, "model": "test/TwelveLayer", "block_range": [0, 4], '
        '"forward_seconds": 0.5, "backward_seconds": 0.3, '
        '"outputs_finite": true, "grad_finite": true}\n',
        encoding="utf-8",
    )
    server_a = tmp_path / "server-0.log"
    server_a.write_text(
        "[INFO] Announced that blocks range(0, 2) are joining\n[INFO] Started\n[INFO] rpc_forward(blocks=0:2, remote_peer=...x)\n",
        encoding="utf-8",
    )
    server_b = tmp_path / "server-1.log"
    server_b.write_text(
        "[INFO] Announced that blocks range(2, 4) are joining\n[INFO] Started\n[INFO] rpc_backward(blocks=2:4, remote_peer=...y)\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "mvp_capabilities/multi_block_diagnostics.py",
            "--model",
            "test/TwelveLayer",
            "--block-range",
            "0:2",
            "--block-range",
            "2:4",
            "--server-log",
            str(server_a),
            "--server-log",
            str(server_b),
            "--client-log",
            str(client_log),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["claim_boundary"] == "multi_block_diagnostics_observability_only_no_inference_proof"
    assert payload["server_count"] == 2
    assert payload["summary"]["status"] == "all_servers_healthy_client_passed"
    assert payload["coverage"]["full_coverage"] is True
    assert payload["inference_proven"] is False
