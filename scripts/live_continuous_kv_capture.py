#!/usr/bin/env python3
"""Live continuous-batching + KV-prefix-reuse capture through a real BloomBee server.

Starts a single-node BloomBee server (TinyLlama-1.1B), then runs:

Phase A — Continuous batching:
  1. Two requests sequentially (baseline), capture tokens + wall-clock
  2. Two requests same-arrival with BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING=1
  3. Server log must show [LIVE_CONTINUOUS_BATCHING] JSON lines

Phase B — KV prefix reuse:
  1. Two requests with shared prefix, no opt-in (baseline)
  2. Two requests with shared prefix + BLOOMBEE_ENABLE_KV_PREFIX_REUSE=1
  3. Server must observe KV prefix metadata

All evidence is written to .local/live-capture-evidence/
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / ".local" / "live-capture-evidence"
SERVER_LOG = EVIDENCE_DIR / "server.log"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DHT_PORT = 31337
SERVER_PORT = 31338
BLOCKED_CLAIM_BOUNDARY = "live_continuous_kv_capture_blocked_no_live_server_proof"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live-capture")


def build_server_command(
    *,
    model_id: str,
    initial_peer: str,
    num_blocks: int,
    port: int,
    batch_size: int,
    device: str | None = None,
) -> list[str]:
    """Return a run_server command wired to the private DHT bootstrap peer."""
    command = [
        sys.executable,
        "-m",
        "bloombee.cli.run_server",
        model_id,
        "--num_blocks",
        str(num_blocks),
        "--port",
        str(port),
        "--batch_size",
        str(batch_size),
        "--initial_peers",
        initial_peer,
    ]
    if device:
        command.extend(["--device", device])
    return command


def build_blocked_live_capture_evidence(*, reason: str, detail: str, model_id: str = MODEL_ID) -> dict:
    """Build fail-closed evidence when the live capture runner cannot proceed."""
    return {
        "model": model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "claim_boundary": BLOCKED_CLAIM_BOUNDARY,
        "source": "scripts/live_continuous_kv_capture.py",
        "blocked": True,
        "blocker": reason,
        "detail": detail,
        "continuous_batching": {
            "live_server_late_arrival_parity_proven": False,
            "live_server_proven": False,
            "speedup_proven": False,
            "wallclock_speedup_proven": False,
        },
        "kv_prefix_reuse": {
            "live_kv_cache_reuse_proven": False,
            "server_observed_kv_cache_reuse": False,
            "speedup_proven": False,
        },
        "can_update_demo_status": False,
        "can_update_proof_status": False,
    }


def write_blocked_live_capture_evidence(*, reason: str, detail: str, model_id: str = MODEL_ID) -> Path:
    """Write fail-closed blocker evidence next to raw capture artifacts."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_blocked_live_capture_evidence(reason=reason, detail=detail, model_id=model_id)
    path = EVIDENCE_DIR / "live-capture-blocked.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def reset_capture_workspace() -> None:
    """Remove stale readiness files before a live-capture attempt."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (SERVER_LOG, EVIDENCE_DIR / "dht_ready.marker", EVIDENCE_DIR / "dht_peer.txt"):
        if path.exists():
            path.unlink()


def run_cmd(cmd: list[str], env: dict | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """Run a command, return (exit_code, stdout, stderr)."""
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=merged_env)
    return proc.returncode, proc.stdout, proc.stderr


def wait_for_dht(host: str = "127.0.0.1", port: int = DHT_PORT, timeout: float = 30) -> bool:
    """Wait for DHT to accept connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                log.info(f"DHT reachable at {host}:{port}")
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(1)
    return False


def wait_for_server_blocks(timeout: float = 240, *, server_proc=None, poll_interval: float = 3.0) -> bool:
    """Wait for the BloomBee server to announce blocks, failing fast if it exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            log.error(f"Server process exited before announcing blocks (exit={server_proc.poll()})")
            return False
        if SERVER_LOG.exists():
            text = SERVER_LOG.read_text(errors="replace")
            if "Started" in text:
                log.info("Server reached 'Started' state")
                return True
            # BloomBee actually says "blocks [...] are joining" or "announced"
            if "announced that blocks" in text.lower() or "are joining" in text.lower():
                log.info("Server announced blocks, ready")
                return True
            if any(k in text for k in ["Server started", "serve blocks", "now serving"]):
                log.info("Server appears started")
                return True
        time.sleep(poll_interval)
    return False


def run_inference(prompt: str, *, max_new_tokens: int = 5, env_flags: dict | None = None) -> dict:
    """Run a single inference request through the BloomBee server."""
    env = dict(os.environ)
    if env_flags:
        env.update(env_flags)

    # Read the DHT peer ID from the server log (we wrote it after DHT startup)
    dht_peer_file = EVIDENCE_DIR / "dht_peer.txt"
    if not dht_peer_file.exists():
        return {"error": "DHT peer ID not yet discovered", "prompt": prompt}
    dht_peer_id = dht_peer_file.read_text().strip()
    initial_peer = f"/ip4/127.0.0.1/tcp/{DHT_PORT}/p2p/{dht_peer_id}"

    script = f"""
import time, json, sys, torch
sys.path.insert(0, "src")

from bloombee import AutoDistributedModelForCausalLM

start = time.time()
try:
    model = AutoDistributedModelForCausalLM.from_pretrained(
        "{MODEL_ID}",
        initial_peers=["{initial_peer}"],
        torch_dtype=torch.float32,
        request_timeout=120,
        max_retries=5,
    )
except Exception as e:
    import traceback
    print(json.dumps({{"error": f"connect failed: {{repr(e)}}", "traceback": traceback.format_exc()[-500:], "prompt": {json.dumps(prompt)}}}))
    sys.exit(1)

tokenizer = model.config.tokenizer or None
if tokenizer is None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("{MODEL_ID}")

inputs = tokenizer({json.dumps(prompt)}, return_tensors="pt")
gen_start = time.time()
output = model.generate(
    inputs["input_ids"],
    max_new_tokens={max_new_tokens},
    do_sample=False,
)
gen_elapsed = time.time() - gen_start
generated_ids = output[0][inputs["input_ids"].shape[1]:].tolist()
print(json.dumps({{
    "prompt": {json.dumps(prompt)},
    "generated_token_ids": generated_ids,
    "generated_text": tokenizer.decode(generated_ids),
    "elapsed_seconds": round(gen_elapsed, 4),
    "connect_seconds": round(gen_start - start, 4),
    "num_blocks": getattr(model.transformer.h, "num_blocks", lambda: 0)(),
}}))
"""

    code, stdout, stderr = run_cmd(
        [sys.executable, "-c", script],
        env=env,
        timeout=300,
    )
    if code != 0:
        return {"error": f"exit={code}: {stderr[-1000:]}", "prompt": prompt, "stdout_tail": stdout[-300:]}
    try:
        lines = [l for l in stdout.strip().split("\n") if l.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"error": f"no json in output: {stdout[-500:]}", "prompt": prompt}
    except (json.JSONDecodeError, IndexError):
        return {"error": f"parse failure: {stdout[-500:]}", "prompt": prompt}


def phase_a_continuous_batching() -> dict:
    """Phase A: Continuous batching live capture."""
    log.info("=== Phase A: Continuous batching ===")

    # Baseline: two sequential requests
    log.info("Running baseline (sequential) requests...")
    baseline_a = run_inference("The capital of France is", max_new_tokens=3)
    time.sleep(1)
    baseline_b = run_inference("The capital of Japan is", max_new_tokens=3)

    # Continuous: same prompt structure, with opt-in flag
    log.info("Running continuous-batching requests (BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING=1)...")
    continuous_a = run_inference(
        "The capital of France is", max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING": "1"},
    )
    time.sleep(1)
    continuous_b = run_inference(
        "The capital of Japan is", max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING": "1"},
    )

    return {
        "phase": "continuous_batching",
        "baseline": {"req-a": baseline_a, "req-b": baseline_b},
        "continuous": {"req-a": continuous_a, "req-b": continuous_b},
    }


def phase_b_kv_prefix_reuse() -> dict:
    """Phase B: KV prefix reuse live capture."""
    log.info("=== Phase B: KV prefix reuse ===")

    shared_prefix = "You are a helpful assistant. Please answer the following question concisely."
    suffix_a = "What is 2+2?"
    suffix_b = "What is 3+3?"

    # Baseline: no opt-in
    log.info("Running baseline KV prefix requests (no opt-in)...")
    baseline_a = run_inference(shared_prefix + " " + suffix_a, max_new_tokens=3)
    time.sleep(0.5)
    baseline_b = run_inference(shared_prefix + " " + suffix_b, max_new_tokens=3)

    # Reuse: with opt-in
    log.info("Running KV prefix reuse requests (BLOOMBEE_ENABLE_KV_PREFIX_REUSE=1)...")
    reuse_a = run_inference(
        shared_prefix + " " + suffix_a,
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_KV_PREFIX_REUSE": "1"},
    )
    time.sleep(0.5)
    reuse_b = run_inference(
        shared_prefix + " " + suffix_b,
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_KV_PREFIX_REUSE": "1"},
    )

    return {
        "phase": "kv_prefix_reuse",
        "shared_prefix": shared_prefix,
        "baseline": {"suffix-a": baseline_a, "suffix-b": baseline_b},
        "reuse": {"suffix-a": reuse_a, "suffix-b": reuse_b},
    }


def parse_server_log_for_continuous_batching() -> dict:
    """Extract [LIVE_CONTINUOUS_BATCHING] JSON lines from server log."""
    observations = []
    if SERVER_LOG.exists():
        for line in SERVER_LOG.read_text(errors="replace").splitlines():
            if "[LIVE_CONTINUOUS_BATCHING]" in line:
                json_str = line.split("[LIVE_CONTINUOUS_BATCHING]")[-1].strip()
                try:
                    observations.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
    return {"observations": observations, "count": len(observations)}


def main():
    reset_capture_workspace()

    log.info(f"Starting DHT on port {DHT_PORT}...")
    # Start DHT and capture peer ID to file for clients
    log_marker_path = EVIDENCE_DIR / "dht_ready.marker"

    # Start DHT
    dht_script = f"""
import os, sys, time
sys.path.insert(0, "src")
os.makedirs({str(EVIDENCE_DIR)!r}, exist_ok=True)
from hivemind import DHT
dht = DHT(start=True, host_maddrs=["/ip4/0.0.0.0/tcp/{DHT_PORT}"])
peer_id = str(dht.peer_id)
with open({str(EVIDENCE_DIR / "dht_peer.txt")!r}, "w") as f:
    f.write(peer_id)
with open({str(log_marker_path)!r}, "w") as f:
    f.write("ready")
print(f"DHT_READY {{peer_id}}", flush=True)
while True:
    time.sleep(60)
"""
    dht_proc = subprocess.Popen(
        [sys.executable, "-c", dht_script],
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    log.info(f"DHT PID: {dht_proc.pid}")

    if not wait_for_dht():
        log.error("DHT failed to start")
        blocked_path = write_blocked_live_capture_evidence(
            reason="dht_start_failed",
            detail="DHT did not become reachable on the local capture port; inspect server.log for sandbox bind or daemon errors.",
        )
        log.error(f"Fail-closed blocker evidence written to {blocked_path}")
        dht_proc.kill()
        return 1

    # Wait for DHT peer ID to be captured
    deadline = time.time() + 30
    peer_id = ""
    while time.time() < deadline:
        peer_file = EVIDENCE_DIR / "dht_peer.txt"
        if peer_file.exists():
            peer_id = peer_file.read_text().strip()
            if peer_id:
                log.info(f"DHT peer_id: {peer_id}")
                break
        time.sleep(1)
    else:
        log.error("DHT peer ID not captured")
        blocked_path = write_blocked_live_capture_evidence(
            reason="dht_peer_id_missing",
            detail="DHT marker appeared unavailable or did not write peer_id before timeout.",
        )
        log.error(f"Fail-closed blocker evidence written to {blocked_path}")
        dht_proc.kill()
        return 1

    # Start BloomBee server
    log.info(f"Starting BloomBee server for {MODEL_ID}...")
    server_env = dict(os.environ)
    server_env["BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"] = "1"
    server_env["BLOOMBEE_ENABLE_KV_PREFIX_REUSE"] = "1"

    initial_peer = f"/ip4/127.0.0.1/tcp/{DHT_PORT}/p2p/{peer_id}"
    server_proc = subprocess.Popen(
        build_server_command(
            model_id=MODEL_ID,
            initial_peer=initial_peer,
            num_blocks=4,
            port=SERVER_PORT,
            batch_size=2,
            device="mps",
        ),
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=server_env,
    )
    log.info(f"Server PID: {server_proc.pid}")

    if not wait_for_server_blocks(timeout=120, server_proc=server_proc):
        log.error("Server failed to start. Dumping last 50 lines of log:")
        log_tail = ""
        if SERVER_LOG.exists():
            lines = SERVER_LOG.read_text(errors="replace").splitlines()
            log_tail = "\n".join(lines[-50:])
            for line in lines[-50:]:
                log.error(f"  {line}")
        blocked_path = write_blocked_live_capture_evidence(
            reason="server_start_failed",
            detail=log_tail or "BloomBee server did not announce blocks before timeout.",
        )
        log.error(f"Fail-closed blocker evidence written to {blocked_path}")
        dht_proc.kill()
        server_proc.kill()
        return 1

    log.info("Server ready! Running capture phases...")

    # Phase A: Continuous batching
    phase_a = phase_a_continuous_batching()

    # Phase B: KV prefix reuse
    phase_b = phase_b_kv_prefix_reuse()

    # Parse server log for observations
    cb_observations = parse_server_log_for_continuous_batching()

    # Write evidence
    evidence = {
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "continuous_batching": phase_a,
        "kv_prefix_reuse": phase_b,
        "server_observations": {
            "live_continuous_batching": cb_observations,
        },
    }

    evidence_path = EVIDENCE_DIR / "live-capture-raw.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    log.info(f"Evidence written to {evidence_path}")

    # Cleanup
    log.info("Shutting down server and DHT...")
    server_proc.kill()
    dht_proc.kill()

    # Summary
    print("\n" + "="*60)
    print("LIVE CAPTURE SUMMARY")
    print("="*60)

    print("\n--- Continuous Batching ---")
    for phase in ["baseline", "continuous"]:
        print(f"  {phase}:")
        for req_id, data in evidence["continuous_batching"][phase].items():
            if "error" in data:
                print(f"    {req_id}: ERROR - {data['error'][:100]}")
            else:
                print(f"    {req_id}: tokens={data.get('generated_token_ids', 'N/A')}, time={data.get('elapsed_seconds', 'N/A')}s")

    print(f"  Server observations: {cb_observations['count']} [LIVE_CONTINUOUS_BATCHING] lines")

    print("\n--- KV Prefix Reuse ---")
    for phase in ["baseline", "reuse"]:
        print(f"  {phase}:")
        for req_id, data in evidence["kv_prefix_reuse"][phase].items():
            if "error" in data:
                print(f"    {req_id}: ERROR - {data['error'][:100]}")
            else:
                print(f"    {req_id}: tokens={data.get('generated_token_ids', 'N/A')}, time={data.get('elapsed_seconds', 'N/A')}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
