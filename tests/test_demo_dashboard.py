from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _write_peer(path: Path, *, hostname: str, total_gb: float, free_gb: float, device: str = "mps") -> None:
    path.write_text(
        json.dumps(
            {
                "hostname": hostname,
                "memory": {"total_gb": total_gb, "free_gb": free_gb},
                "accelerator": {
                    "device": device,
                    "unified_memory": device == "mps",
                    "vram_total_gb": total_gb if device == "mps" else None,
                    "vram_free_gb": free_gb if device == "mps" else None,
                    "gpus": [],
                },
                "network": {"tailscale_ip": "100.64.0.1" if hostname == "m4pro" else None},
                "mobile": {"is_mobile": False, "kind": None, "runtime": None},
            }
        ),
        encoding="utf-8",
    )


def _write_bench_matrix(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "m4pro": {
                    "summary": {"hostname": "m4pro"},
                    "models": {
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                            "decode_tok_per_s": 17.66,
                            "prefill_tok_per_s": 517.1,
                            "device": "mps",
                            "dtype": "bf16",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "generate-api",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "server_to_server": True,
                "generated_ids_match": True,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_seconds": 4.313,
                "server_placements": [
                    {"host": "m4pro-seed", "layers": [0, 8], "server_maddr": "/ip4/192.168.178.37/tcp/31337/p2p/seed"},
                    {"host": "m4pro-mid", "layers": [8, 15], "server_maddr": "/ip4/192.168.178.37/tcp/31338/p2p/mid"},
                    {"host": "m4pro-tail", "layers": [15, 22], "server_maddr": "/ip4/192.168.178.37/tcp/31339/p2p/tail"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_proof_state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "proof_state_observability_only_no_inference_proof",
                "model": "Qwen/Qwen3-8B",
                "gate": "one_block_server",
                "download_status": "complete",
                "host": "m4pro",
                "fetch_progress": {"percent": 100, "completed_files": 15, "total_files": 15},
                "cache": {
                    "weight_files": 5,
                    "bytes": 36_374_890_321,
                    "human": "33.9G",
                    "snapshot_complete": True,
                    "stale_incomplete_files": 4,
                },
                "eta_seconds": 0,
                "eta_reason": "snapshot_complete",
                "inference_proven": False,
            }
        ),
        encoding="utf-8",
    )


def _write_joined_layer_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "joined_roster_layer_plan_only_no_inference_proof",
                "source": "coordinator_http_active",
                "token": "moon-token",
                "model_id": "Qwen/Qwen3-8B",
                "active_peer_count": 2,
                "inference_proven": False,
                "placement": {
                    "supported": True,
                    "reason": "capacity covers all 36 layers across 2 peer(s)",
                    "num_layers": 36,
                    "assigned_layers": 36,
                    "missing_layers": 0,
                    "claim_boundary": "placement_plan_only_no_inference_proof",
                    "launch_commands_claim_boundary": "launch_commands_only_no_server_started",
                    "assignments": [
                        {
                            "hostname": "joined-peer-a",
                            "block_range": "0:18",
                            "start_layer": 0,
                            "end_layer": 18,
                            "layer_count": 18,
                            "port": 41000,
                            "launch_command": "python -m bloombee.cli.run_server Qwen/Qwen3-8B --block_indices 0:18 --new_swarm",
                        },
                        {
                            "hostname": "joined-peer-b",
                            "block_range": "18:36",
                            "start_layer": 18,
                            "end_layer": 36,
                            "layer_count": 18,
                            "port": 41001,
                            "launch_command": "PYTHONPATH=.:src python -m bloombee.cli.run_server Qwen/Qwen3-8B --block_indices 18:36 --initial_peers '<SEED_MULTIADDR_FROM_joined-peer-a>'",
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_chain_schedule(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "chain_scheduler_plan_only_no_inference_proof",
                "scheduler_status": "ready_to_rehearse_no_live_requests",
                "model_id": "Qwen/Qwen3-8B",
                "request_count": 5,
                "stage_count": 2,
                "wave_count": 3,
                "waves": [
                    {"wave_index": 0, "request_ids": ["req-000", "req-001"], "parallel_request_count": 2},
                    {"wave_index": 1, "request_ids": ["req-002", "req-003"], "parallel_request_count": 2},
                    {"wave_index": 2, "request_ids": ["req-004"], "parallel_request_count": 1},
                ],
                "peer_health": {
                    "joined-peer-a": {
                        "hostname": "joined-peer-a",
                        "block_range": "0:18",
                        "scheduled_requests": 5,
                        "scheduled_tokens": 240,
                        "peak_parallel_requests": 2,
                        "utilization_fraction": 0.83,
                        "health_status": "planned_no_live_traffic",
                    },
                    "joined-peer-b": {
                        "hostname": "joined-peer-b",
                        "block_range": "18:36",
                        "scheduled_requests": 5,
                        "scheduled_tokens": 240,
                        "peak_parallel_requests": 2,
                        "utilization_fraction": 0.83,
                        "health_status": "planned_no_live_traffic",
                    },
                },
                "token_budget": {"tokens_per_request": 48, "scheduled_tokens": 240},
                "inference_proven": False,
                "live_requests_sent": False,
            }
        ),
        encoding="utf-8",
    )


def _write_handoff_bundle(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "coordinator_handoff_bundle_only_no_server_started",
                "source": "coordinator_http_handoff_endpoint",
                "token": "moon-token",
                "inference_proven": False,
                "can_update_proof_status": False,
                "route_decision": {"picked": {"model_id": "Qwen/Qwen3-8B"}},
                "plan": {
                    "model_id": "Qwen/Qwen3-8B",
                    "launch_readiness": {
                        "ready_to_start": False,
                        "claim_boundary": "launch_readiness_checklist_only_no_server_started",
                    },
                },
                "bootstrap_runbook": {
                    "claim_boundary": "coordinator_bootstrap_runbook_only_no_server_started",
                    "heartbeat_loop": {"count": 180, "interval_seconds": 10.0},
                    "shell_script": "python mvp_capabilities/peer_scan.py --out \"$CAP_PATH\"\npython mvp_capabilities/join_client.py --join-url 'bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=%2A%2A%2A' --capabilities \"$CAP_PATH\" --count 180 --interval-seconds 10",
                },
                "proof_runbooks": {
                    "multi_block": {"claim_boundary": "multi_block_proof_harness_only_no_live_inference", "proof_gate": "multi_block"},
                    "full_generation": {"claim_boundary": "full_generation_proof_harness_only_no_live_generation", "proof_gate": "full_generation"},
                    "cache_generation": {"claim_boundary": "cache_generation_proof_harness_only_no_live_generation", "proof_gate": "cache_generation"},
                    "multi_request_load": {
                        "claim_boundary": "multi_request_load_harness_only_no_live_traffic",
                        "proof_gate": "multi_request_load",
                        "request_count": 2,
                    },
                },
                "proof_orchestration": {
                    "claim_boundary": "proof_orchestration_plan_only_no_live_inference",
                    "source": "coordinator_handoff_embedded_proof_orchestration",
                    "model_id": "Qwen/Qwen3-8B",
                    "phase_order": [
                        "start_servers",
                        "capture_server_multiaddrs",
                        "run_proof_clients",
                        "verify_then_promote_manually",
                    ],
                    "summary": {
                        "server_count": 2,
                        "ready_to_start_servers": False,
                        "ready_for_proof_clients": False,
                        "unresolved_placeholders": ["<SEED_MULTIADDR_FROM_joined-peer-a>", "<PASTE_SERVER_0_MULTIADDR>"],
                        "available_proof_gates": ["multi_block", "full_generation", "cache_generation", "multi_request_load"],
                    },
                    "launch_steps": [
                        {"hostname": "joined-peer-a", "role": "seed", "block_range": "0:18", "ready": True},
                        {"hostname": "joined-peer-b", "role": "follower", "block_range": "18:36", "ready": False},
                    ],
                    "proof_steps": [
                        {"proof_gate": "multi_block", "ready": False, "command_count": 2},
                        {"proof_gate": "full_generation", "ready": False, "command_count": 2},
                        {"proof_gate": "cache_generation", "ready": False, "command_count": 2},
                        {"proof_gate": "multi_request_load", "ready": False, "command_count": 3},
                    ],
                    "inference_proven": False,
                    "can_update_proof_status": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_speculative_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "speculative_decode_plan_only_no_generation_proof",
                "source": "speculative_decode_plan.py",
                "verifier": {"model_id": "Qwen/Qwen3-8B", "authoritative": True, "claim_level": "experimental"},
                "draft": {
                    "mode": "async_draft_provider",
                    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "max_draft_tokens": 4,
                    "phone_candidates": [{"hostname": "phone-a", "runtime": "termux"}],
                },
                "correctness_contract": {"accepted_tokens_require_verifier_match": True},
                "phone_policy": {"phones_as_block_workers": False, "phones_as_draft_providers_only": True},
                "inference_proven": False,
                "can_update_proof_status": False,
            }
        ),
        encoding="utf-8",
    )


def _write_draft_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "draft_provider_contract_only_no_generation_proof",
                "source": "draft_provider.py",
                "provider": {
                    "provider_id": "phone-fake",
                    "provider_kind": "deterministic_fake",
                    "phone_compatible_interface": True,
                    "can_serve_transformer_blocks": False,
                },
                "proposal": {
                    "provider_id": "phone-fake",
                    "provider_kind": "deterministic_fake",
                    "draft_tokens": [10, 11, 12],
                    "draft_token_count": 3,
                    "elapsed_ms": 0.04,
                },
                "verdict": {
                    "accepted_tokens": [10],
                    "rejected_tokens": [11, 12],
                    "accepted_count": 1,
                    "rejected_count": 2,
                    "proposed_count": 3,
                    "acceptance_rate": 0.333333,
                    "verifier_fallback_token": 99,
                    "committed_tokens": [10, 99],
                    "verifier_authoritative": True,
                },
                "dashboard_counters": {"proposed": 3, "accepted": 1, "rejected": 2, "acceptance_rate": 0.333333},
                "generation_proven": False,
                "speedup_proven": False,
                "inference_proven": False,
            }
        ),
        encoding="utf-8",
    )


def _write_request_log(path: Path) -> None:
    path.write_text(
        "[direct] model=Qwen/Qwen3-8B\n"
        '[direct] RESULT: {"ok": true, "model": "Qwen/Qwen3-8B", "block_range": [0, 1], '
        '"forward_seconds": 0.08, "backward_seconds": 0.20, "outputs_finite": true, "grad_finite": true}\n'
        "RuntimeError: DHT bootstrap failed before RPC\n",
        encoding="utf-8",
    )


def _write_multi_block_diagnostics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claim_boundary": "multi_block_diagnostics_observability_only_no_inference_proof",
                "source": "multi_block_diagnostics.py",
                "model_id": "Qwen/Qwen3-8B",
                "combined_block_range": "0:36",
                "server_count": 2,
                "summary": {
                    "healthy_servers": 1,
                    "unhealthy_servers": 1,
                    "status": "unhealthy_servers_detected",
                },
                "coverage": {
                    "covered_layers": 18,
                    "total_layers": 36,
                    "missing_layers": 18,
                    "full_coverage": False,
                },
                "servers": [
                    {"server_index": 0, "block_range": "0:18", "health": "healthy", "started": True, "announced_block_range": True, "has_rpc_evidence": True},
                    {
                        "server_index": 1,
                        "block_range": "18:36",
                        "health": "unhealthy",
                        "started": False,
                        "announced_block_range": False,
                        "has_rpc_evidence": False,
                        "errors": ["server did not reach Started state"],
                    },
                ],
                "operator_actions": ["server 1 (18:36): server did not reach Started state. Check server logs for crash/port/block-index details."],
                "inference_proven": False,
                "can_update_proof_status": False,
            }
        ),
        encoding="utf-8",
    )


def test_dashboard_data_surfaces_devices_routes_benchmarks_and_evidence(tmp_path: Path):
    from mvp_capabilities.demo_dashboard import build_dashboard_document, render_dashboard_html

    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _write_peer(cap_dir / "evinova.json", hostname="evinova", total_gb=16, free_gb=2.5)
    _write_peer(cap_dir / "m4pro.json", hostname="m4pro", total_gb=48, free_gb=34.5)
    bench_matrix = tmp_path / "bench-matrix.json"
    _write_bench_matrix(bench_matrix)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    _write_evidence(evidence_dir / "TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA.json")
    proof_state = tmp_path / "proof-state.json"
    _write_proof_state(proof_state)
    joined_layer_plan = tmp_path / "joined-layer-plan.json"
    _write_joined_layer_plan(joined_layer_plan)
    chain_schedule = tmp_path / "chain-schedule.json"
    _write_chain_schedule(chain_schedule)
    handoff_bundle = tmp_path / "handoff-bundle.json"
    _write_handoff_bundle(handoff_bundle)
    speculative_plan = tmp_path / "speculative-plan.json"
    _write_speculative_plan(speculative_plan)
    draft_report = tmp_path / "draft-report.json"
    _write_draft_report(draft_report)
    multi_block_diagnostics = tmp_path / "multi-block-diagnostics.json"
    _write_multi_block_diagnostics(multi_block_diagnostics)
    request_log = tmp_path / "direct-client.log"
    _write_request_log(request_log)

    doc = build_dashboard_document(
        cap_dirs=[cap_dir],
        bench_matrix_path=bench_matrix,
        evidence_dir=evidence_dir,
        proof_state_path=proof_state,
        joined_layer_plan_path=joined_layer_plan,
        chain_schedule_path=chain_schedule,
        handoff_bundle_path=handoff_bundle,
        speculative_plan_path=speculative_plan,
        draft_report_path=draft_report,
        multi_block_diagnostics_path=multi_block_diagnostics,
        request_logs=[request_log],
        synthetic_m4_laptops=10,
        synthetic_total_gb=24,
        synthetic_free_gb=20,
    )
    html = render_dashboard_html(doc, refresh_seconds=15)

    assert doc["real_route"]["picked"]["model_id"]
    assert doc["synthetic_10_laptop_route"]["picked"]["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert doc["roster"]["summary"]["peer_count"] == 2
    assert doc["benchmarks"]["m4pro"]["models"]["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]["decode_tok_per_s"] == 17.66
    assert doc["evidence"][0]["generated_text_match"] is True
    assert doc["layer_placements"][0]["host"] == "m4pro-seed"
    assert doc["layer_placements"][0]["layers"] == [0, 8]
    assert doc["layer_placements"][2]["host"] == "m4pro-tail"
    assert doc["mvp_status"]["overall_percent"] == 77
    assert doc["mvp_status"]["next_gate"] == "Qwen3-8B full-generation or cache-generation proof"
    assert doc["mvp_status"]["task_summary"]["total"] == 17
    assert doc["mvp_status"]["task_summary"]["blocked"] == 2
    assert any(task["id"] == "physical_showcase" and task["done"] is False for task in doc["mvp_status"]["planned_tasks"])
    assert doc["proof_state"]["download_status"] == "complete"
    assert doc["proof_state"]["inference_proven"] is False
    assert doc["joined_layer_plan"]["source"] == "coordinator_http_active"
    assert doc["joined_layer_plan"]["active_peer_count"] == 2
    assert doc["chain_schedule"]["scheduler_status"] == "ready_to_rehearse_no_live_requests"
    assert doc["chain_schedule"]["peer_health"]["joined-peer-b"]["utilization_fraction"] == 0.83
    assert doc["handoff_bundle"]["claim_boundary"] == "coordinator_handoff_bundle_only_no_server_started"
    assert doc["handoff_bundle"]["bootstrap_runbook"]["claim_boundary"] == "coordinator_bootstrap_runbook_only_no_server_started"
    assert doc["handoff_bundle"]["proof_runbooks"]["multi_block"]["proof_gate"] == "multi_block"
    assert doc["proof_orchestration"]["claim_boundary"] == "proof_orchestration_plan_only_no_live_inference"
    assert doc["proof_orchestration"]["summary"]["ready_for_proof_clients"] is False
    assert doc["proof_orchestration"]["proof_steps"][0]["proof_gate"] == "multi_block"
    assert doc["speculative_plan"]["claim_boundary"] == "speculative_decode_plan_only_no_generation_proof"
    assert doc["speculative_plan"]["verifier"]["authoritative"] is True
    assert doc["draft_report"]["claim_boundary"] == "draft_provider_contract_only_no_generation_proof"
    assert doc["draft_report"]["dashboard_counters"] == {"proposed": 3, "accepted": 1, "rejected": 2, "acceptance_rate": 0.333333}
    assert doc["multi_block_diagnostics"]["summary"]["status"] == "unhealthy_servers_detected"
    assert doc["multi_block_diagnostics"]["coverage"]["missing_layers"] == 18
    assert doc["request_telemetry"]["request_counts"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert doc["request_telemetry"]["latency_seconds"]["forward"]["avg"] == 0.08
    assert "evinova" in html
    assert "m4pro" in html
    assert "Qwen/Qwen3-30B-A3B" in html
    assert "MVP build status" in html
    assert "███████████████░░░░░ 77%" in html
    assert "Qwen3-8B full-generation or cache-generation proof" in html
    assert "weighted_plan_status_not_demo_proof" in html
    assert "Planned tasks" in html
    assert "Task summary: 5 complete, 7 partial, 3 pending, 2 blocked" in html
    assert "TinyLlama distributed fallback generation proof" in html
    assert "Physical/self-serve N-laptop showcase" in html
    assert "Qwen35B candidate branch" in html
    assert "Live proof-prep state" in html
    assert "Snapshot" in html
    assert "stale partials 4" in html
    assert "ETA" in html
    assert "Qwen/Qwen3-8B" in html
    assert "proof_state_observability_only_no_inference_proof" in html
    assert "100%" in html
    assert "inference not proven" in html
    assert "unmeasured" in html
    assert "Layer placement" in html
    assert "m4pro-seed" in html
    assert "layers 0:8" in html
    assert "m4pro-tail" in html
    assert "Joined-peer layer plan" in html
    assert "coordinator_http_active" in html
    assert "joined-peer-a" in html
    assert "layers 0:18" in html
    assert "launch_commands_only_no_server_started" in html
    assert "joined_roster_layer_plan_only_no_inference_proof" in html
    assert "Chain scheduler rehearsal" in html
    assert "chain_scheduler_plan_only_no_inference_proof" in html
    assert "ready_to_rehearse_no_live_requests" in html
    assert "req-000, req-001" in html
    assert "planned_no_live_traffic" in html
    assert "Operator handoff bundle" in html
    assert "coordinator_http_handoff_endpoint" in html
    assert "coordinator_handoff_bundle_only_no_server_started" in html
    assert "Fresh-device bootstrap" in html
    assert "coordinator_bootstrap_runbook_only_no_server_started" in html
    assert "peer_scan.py" in html
    assert "join_client.py" in html
    assert "heartbeat count 180" in html
    assert "multi_block_proof_harness_only_no_live_inference" in html
    assert "multi_request_load_harness_only_no_live_traffic" in html
    assert "Proof orchestration" in html
    assert "proof_orchestration_plan_only_no_live_inference" in html
    assert "capture_server_multiaddrs" in html
    assert "ready for proof clients: no" in html
    assert "&lt;PASTE_SERVER_0_MULTIADDR&gt;" in html
    assert "Speculative decode plan" in html
    assert "speculative_decode_plan_only_no_generation_proof" in html
    assert "Verifier authoritative" in html
    assert "phone-a" in html
    assert "phones as draft providers only" in html
    assert "Draft-provider contract smoke" in html
    assert "draft_provider_contract_only_no_generation_proof" in html
    assert "Proposed / accepted / rejected" in html
    assert "3 / 1 / 2" in html
    assert "0.333333" in html
    assert "Multi-block diagnostics" in html
    assert "multi_block_diagnostics_observability_only_no_inference_proof" in html
    assert "unhealthy_servers_detected" in html
    assert "coverage 18/36 layers" in html
    assert "server 1 (18:36)" in html
    assert "Live request telemetry" in html
    assert "request_telemetry_observability_only_no_load_proof" in html
    assert "succeeded 1 / failed 1" in html
    assert "forward avg 0.08s" in html
    assert "DHT bootstrap failed before RPC" in html
    assert "[S2S_PUSH_EVENT]" in html
    assert "TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA.json" in html


def test_dashboard_request_telemetry_formats_zero_latency_as_unmeasured():
    from mvp_capabilities.demo_dashboard import _request_telemetry_panel

    html = _request_telemetry_panel(
        {
            "claim_boundary": "request_telemetry_observability_only_no_load_proof",
            "request_counts": {"total": 1, "succeeded": 1, "failed": 0},
            "latency_seconds": {
                "forward": {"count": 0, "avg": None, "min": None, "max": None, "p95": None, "unmeasured_count": 1},
                "backward": {"count": 0, "avg": None, "min": None, "max": None, "p95": None, "unmeasured_count": 1},
            },
            "models": {"Qwen/Qwen3-8B": 1},
            "block_ranges": {"0:1": 1},
            "load_proof_claimed": False,
            "errors": [],
            "next_step": "Zero latency means unmeasured, not a fast request.",
        }
    )

    assert "forward avg unmeasured" in html
    assert "backward avg unmeasured" in html
    assert "Unmeasured latency" in html
    assert "forward 1 / backward 1" in html
    assert "0.00s" not in html


def test_dashboard_cli_writes_html_artifact(tmp_path: Path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _write_peer(cap_dir / "m4pro.json", hostname="m4pro", total_gb=48, free_gb=34.5)
    bench_matrix = tmp_path / "bench-matrix.json"
    _write_bench_matrix(bench_matrix)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    _write_evidence(evidence_dir / "TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA.json")
    proof_state = tmp_path / "proof-state.json"
    _write_proof_state(proof_state)
    joined_layer_plan = tmp_path / "joined-layer-plan.json"
    _write_joined_layer_plan(joined_layer_plan)
    chain_schedule = tmp_path / "chain-schedule.json"
    _write_chain_schedule(chain_schedule)
    handoff_bundle = tmp_path / "handoff-bundle.json"
    _write_handoff_bundle(handoff_bundle)
    speculative_plan = tmp_path / "speculative-plan.json"
    _write_speculative_plan(speculative_plan)
    draft_report = tmp_path / "draft-report.json"
    _write_draft_report(draft_report)
    multi_block_diagnostics = tmp_path / "multi-block-diagnostics.json"
    _write_multi_block_diagnostics(multi_block_diagnostics)
    out = tmp_path / "dashboard.html"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "mvp_capabilities" / "demo_dashboard.py"),
            "--cap-dir",
            str(cap_dir),
            "--bench-matrix",
            str(bench_matrix),
            "--evidence-dir",
            str(evidence_dir),
            "--proof-state",
            str(proof_state),
            "--joined-layer-plan",
            str(joined_layer_plan),
            "--chain-schedule",
            str(chain_schedule),
            "--handoff-bundle",
            str(handoff_bundle),
            "--speculative-plan",
            str(speculative_plan),
            "--draft-report",
            str(draft_report),
            "--multi-block-diagnostics",
            str(multi_block_diagnostics),
            "--out",
            str(out),
            "--refresh-seconds",
            "10",
            "--watch-ticks",
            "1",
            "--watch-seconds",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "BloomBee Distributed Inference Demo Dashboard" in text
    assert "m4pro" in text
    assert "MVP build status" in text
    assert "███████████████░░░░░ 77%" in text
    assert "Live proof-prep state" in text
    assert "Joined-peer layer plan" in text
    assert "joined-peer-b" in text
    assert "layers 18:36" in text
    assert "Chain scheduler rehearsal" in text
    assert "Operator handoff bundle" in text
    assert "Proof orchestration" in text
    assert "proof_orchestration_plan_only_no_live_inference" in text
    assert "Speculative decode plan" in text
    assert "speculative_decode_plan_only_no_generation_proof" in text
    assert "Draft-provider contract smoke" in text
    assert "draft_provider_contract_only_no_generation_proof" in text
    assert "Multi-block diagnostics" in text
    assert "multi_block_diagnostics_observability_only_no_inference_proof" in text
    assert "coordinator_handoff_bundle_only_no_server_started" in text
    assert "joined-peer-a" in text
    assert "240" in text
    assert "proof_state_observability_only_no_inference_proof" in text
    assert "auto-refreshes every 10 seconds" in text
    assert "Synthetic 10-laptop target route" not in text


def test_phone_speculative_decoding_mvp_analysis_sets_honest_claim_boundaries():
    text = (PROJECT_ROOT / "docs" / "phone-speculative-decoding-mvp.md").read_text(encoding="utf-8")

    assert "MVP verdict" in text
    assert "draft model" in text.lower()
    assert "speculative decoding" in text.lower()
    assert "not count phones as transformer-block workers" in text.lower()
    assert "proof gates" in text.lower()
