"""Auto-deploy pipeline coordinator fixes (2026-07-10 session).

Covers the coordinator half of the zero-touch deploy contract:
  - /deploy jobs carry pipeline-position facts (num_layers / is_first /
    is_last) so peers can prefetch exactly the shards for their layers
  - /deploy resets stale peer statuses from a previous run
  - /active (wildcard) no longer deletes heartbeat files the instant they
    cross the display threshold — quiet-but-alive peers stay matchable
  - /weights-needed hands boundary shards to the peers at the layer-order
    ends of the pipeline, not the alphabetical ends
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

from mvp_capabilities.join_coordinator import record_heartbeat
from mvp_capabilities.join_http_server import (
    _handle_deploy,
    _peer_status_dir,
    _weights_needed,
    handle_get,
)
from mvp_capabilities.route_picker import DEFAULT_REGISTRY

MODEL_ID = "Qwen/Qwen3-8B"  # registry: 36 layers, 23 GB recommended


def _register_peer(state_dir: Path, hostname: str, total_gb: float = 48.0) -> None:
    record_heartbeat(
        state_dir,
        token="*",
        peer_id=f"{hostname}-abc123",
        capabilities={
            "hostname": hostname,
            "memory": {"total_gb": total_gb, "available_gb": total_gb * 0.8},
            "platform": "darwin",
        },
        now=int(time.time()),
    )


def _deploy(state_dir: Path):
    status, payload = _handle_deploy(
        {"model_id": [MODEL_ID], "token": ["*"]},
        state_dir=state_dir,
        registry=DEFAULT_REGISTRY,
    )
    assert status == 200, payload
    return payload


def test_deploy_jobs_carry_pipeline_position(tmp_path):
    _register_peer(tmp_path, "Evis-MacBook-Pro")
    _register_peer(tmp_path, "m4pro")

    deployment = _deploy(tmp_path)
    jobs = deployment["jobs"]
    assert len(jobs) == 2

    by_start = sorted(jobs.values(), key=lambda j: j["start_layer"])
    first, last = by_start[0], by_start[-1]
    assert all(j["num_layers"] == 36 for j in jobs.values())
    assert first["start_layer"] == 0 and first["is_first"] is True
    assert first["is_last"] is False
    assert last["end_layer"] == 36 and last["is_last"] is True
    assert last["is_first"] is False


def test_deploy_clears_stale_peer_statuses(tmp_path):
    _register_peer(tmp_path, "Evis-MacBook-Pro")
    _register_peer(tmp_path, "m4pro")

    status_dir = _peer_status_dir(tmp_path)
    status_dir.mkdir(parents=True)
    stale = status_dir / "Evis-MacBook-Pro-abc123.json"
    stale.write_text(
        json.dumps({"peer_id": "Evis-MacBook-Pro-abc123", "status": "serving",
                    "updated_at": time.time()}),
        encoding="utf-8",
    )

    _deploy(tmp_path)
    # The pre-deploy "serving" claim belonged to the previous run; keeping it
    # would make /infer and the dashboard report readiness that isn't real.
    assert not stale.exists()


def test_deploy_planning_window_survives_slow_heartbeat(tmp_path):
    """A peer whose last heartbeat is 45s old (QR default interval was 30s;
    jitter happens) must still be part of the deployment plan."""
    now = int(time.time())
    record_heartbeat(
        tmp_path, token="*", peer_id="Evis-MacBook-Pro-abc123",
        capabilities={"hostname": "Evis-MacBook-Pro",
                      "memory": {"total_gb": 48.0, "available_gb": 40.0}},
        now=now - 45,
    )
    record_heartbeat(
        tmp_path, token="*", peer_id="m4pro-def456",
        capabilities={"hostname": "m4pro",
                      "memory": {"total_gb": 48.0, "available_gb": 40.0}},
        now=now,
    )
    deployment = _deploy(tmp_path)
    assert set(deployment["jobs"]) == {"Evis-MacBook-Pro", "m4pro"}


def test_active_wildcard_keeps_quiet_heartbeats_on_disk(tmp_path):
    """A peer that has gone quiet for 2 minutes (blocked in a weight
    download) drops out of the *displayed* roster but its heartbeat file
    must survive — /job matching and the deploy planner rely on it."""
    now = int(time.time())
    quiet = tmp_path / "quiet-peer.json"
    quiet.write_text(
        json.dumps({"ok": True, "peer_id": "quiet-peer", "token": "*",
                    "timestamp": now - 120, "capabilities": {"hostname": "quiet-peer"}}),
        encoding="utf-8",
    )
    ancient = tmp_path / "ancient-peer.json"
    ancient.write_text(
        json.dumps({"ok": True, "peer_id": "ancient-peer", "token": "*",
                    "timestamp": now - 7200, "capabilities": {"hostname": "ancient-peer"}}),
        encoding="utf-8",
    )

    status, payload = handle_get(
        "/active?max_age_seconds=30",
        state_dir=tmp_path,
        coordinator="http://127.0.0.1:8787",
    )
    assert status == 200
    peer_ids = {p.get("peer_id") for p in payload["active_peers"]}
    assert "quiet-peer" not in peer_ids  # not displayed…
    assert quiet.exists()                # …but retained on disk
    assert not ancient.exists()          # >1h old → cleaned up


def test_weights_needed_boundary_shards_follow_layer_order(tmp_path, monkeypatch):
    """The embedding/lm_head shards must go to the peers at the layer-order
    ends of the pipeline. Hostnames are chosen so the MIDDLE peer sorts
    alphabetically first — the old sorted(jobs.keys()) logic hands it the
    boundary shards and starves the actual first peer."""
    deployment = {
        "model_id": MODEL_ID,
        "created_at": time.time(),
        "jobs": {
            "bravo-first": {"start_layer": 0, "end_layer": 9},
            "aardvark-mid": {"start_layer": 9, "end_layer": 20},
            "charlie-last": {"start_layer": 20, "end_layer": 36},
        },
    }
    (tmp_path / "deployment.json").write_text(json.dumps(deployment), encoding="utf-8")

    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "model.layers.0.a": "model-00001.safetensors",
            "model.layers.9.a": "model-00002.safetensors",
            "model.layers.20.a": "model-00003.safetensors",
            "model.layers.35.a": "model-00005.safetensors",
            "lm_head.weight": "model-00005.safetensors",
        }
    }

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=15: _FakeResponse(json.dumps(index).encode("utf-8")),
    )

    result = _weights_needed(tmp_path)
    assert result["ok"] is True
    peers = result["peers"]
    non_layer = {"model-00001.safetensors", "model-00005.safetensors"}
    assert non_layer <= set(peers["bravo-first"]["shards"])
    assert non_layer <= set(peers["charlie-last"]["shards"])
    assert set(peers["aardvark-mid"]["shards"]) == {"model-00002.safetensors"}
