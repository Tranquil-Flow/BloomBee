from __future__ import annotations

import json
from pathlib import Path

from mvp_capabilities.cache_generation_proof import build_cache_generation_plan, verify_cache_generation_evidence


def test_cache_generation_proof_plan_can_use_streamed_reference_for_quantized_route():
    plan = build_cache_generation_plan(
        model_id="Qwen/Qwen3-30B-A3B@int8",
        checkpoint_model="Qwen/Qwen3-30B-A3B",
        server_maddrs=["/ip4/100.84.252.4/tcp/31484/p2p/seed"],
        server_placements=["m4pro-full=0:48"],
        prompt="The capital of France is",
        max_new_tokens=1,
        evidence_path="mvp_capabilities/distributed_evidence/post_mvp/qwen30b-int8-cache-generation.json",
        reference_mode="streamed-blocks",
        reference_cache_dir="/Volumes/Seagate Portable Drive/huggingface/hub",
        reference_local_files_only=True,
    )

    command = plan["parity_command"]
    assert plan["proof_gate"] == "cache_generation"
    assert plan["checkpoint_model"] == "Qwen/Qwen3-30B-A3B"
    assert plan["reference_mode"] == "streamed-blocks"
    assert "--mode generate-api" in command
    assert "--model Qwen/Qwen3-30B-A3B@int8" in command
    assert "--checkpoint-model 'Qwen/Qwen3-30B-A3B'" in command
    assert "--reference-mode streamed-blocks" in command
    assert "--reference-cache-dir '/Volumes/Seagate Portable Drive/huggingface/hub'" in command
    assert "--reference-local-files-only" in command
    assert "cache_generation_proof.py verify" in plan["verify_command"]


def test_cache_generation_proof_verifier_accepts_streamed_reference_steps(tmp_path: Path):
    evidence = tmp_path / "qwen30b-int8-cache-generation.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "generate-api",
                "reference_mode": "streamed-blocks",
                "reference_generation_path": "streamed-forward-loop-correctness-fallback",
                "model": "Qwen/Qwen3-30B-A3B@int8",
                "checkpoint_model": "Qwen/Qwen3-30B-A3B",
                "prompt": "The capital of France is",
                "max_new_tokens": 1,
                "generated_ids_match": True,
                "generated_text_match": True,
                "next_token_match": True,
                "distributed_ids": [785, 6722, 315, 9625, 374, 20763],
                "reference_ids": [785, 6722, 315, 9625, 374, 20763],
                "distributed_text": "The capital of France is disaster",
                "reference_text": "The capital of France is disaster",
                "distributed_steps": [],
                "reference_steps": [{"step": 0, "next_token_id": 20763}],
                "distributed_seconds": 7.0,
                "reference_seconds": 653.0,
                "server_maddrs": ["/ip4/100.84.252.4/tcp/31484/p2p/seed"],
                "server_placements": [
                    {"host": "m4pro-full", "layers": [0, 48], "server_maddr": "/ip4/100.84.252.4/tcp/31484/p2p/seed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_cache_generation_evidence(
        evidence_path=evidence,
        model_id="Qwen/Qwen3-30B-A3B@int8",
        min_new_tokens=1,
        require_server_placements=True,
    )

    assert result["status"] == "passed"
    assert result["can_update_proof_status"] is True
    assert result["proof_status_update"] == {"cache_generation": "passed"}
