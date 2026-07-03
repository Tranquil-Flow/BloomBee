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
                "download_status": "running",
                "host": "m4pro",
                "fetch_progress": {"percent": 40, "completed_files": 6, "total_files": 15},
                "cache": {"weight_files": 0, "bytes": 3_006_477_312, "human": "2.8G"},
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
                            "launch_command": "python -m bloombee.cli.run_server Qwen/Qwen3-8B --block_indices 18:36 --initial_peers '<SEED_MULTIADDR_FROM_joined-peer-a>'",
                        },
                    ],
                },
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

    doc = build_dashboard_document(
        cap_dirs=[cap_dir],
        bench_matrix_path=bench_matrix,
        evidence_dir=evidence_dir,
        proof_state_path=proof_state,
        joined_layer_plan_path=joined_layer_plan,
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
    assert doc["mvp_status"]["overall_percent"] == 63
    assert doc["mvp_status"]["next_gate"] == "Qwen3-8B one-block server proof"
    assert doc["proof_state"]["download_status"] == "running"
    assert doc["proof_state"]["inference_proven"] is False
    assert doc["joined_layer_plan"]["source"] == "coordinator_http_active"
    assert doc["joined_layer_plan"]["active_peer_count"] == 2
    assert "evinova" in html
    assert "m4pro" in html
    assert "Qwen/Qwen3-30B-A3B" in html
    assert "MVP build status" in html
    assert "█████████████░░░░░░░ 63%" in html
    assert "Qwen3-8B one-block server proof" in html
    assert "weighted_plan_status_not_demo_proof" in html
    assert "Live proof-prep state" in html
    assert "Qwen/Qwen3-8B" in html
    assert "proof_state_observability_only_no_inference_proof" in html
    assert "40%" in html
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
    assert "[S2S_PUSH_EVENT]" in html
    assert "TEXT_GEN_PARITY_GENERATE_API_3PEER_S2S_DEFAULT_TINYLLAMA.json" in html


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
    assert "█████████████░░░░░░░ 63%" in text
    assert "Live proof-prep state" in text
    assert "Joined-peer layer plan" in text
    assert "joined-peer-b" in text
    assert "layers 18:36" in text
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
