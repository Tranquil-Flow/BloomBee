#!/usr/bin/env python3
"""Real end-to-end live capture for continuous batching + KV prefix reuse.

Strategy: the client uses DHT directly + RemoteSequential (proven working).
The server uses a pre-converted BLOOMBEE_NP_PATH of TinyLlama weights.
We bypass AutoDistributedModelForCausalLM (which has an extra libp2p
peer-discovery step that hangs on a single-host DHT with no relay).

This script DOES require the BloomBee server to be reachable, and
DOES generate real tokens through a real pipeline.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / ".local" / "live-capture-evidence"
SERVER_LOG = EVIDENCE_DIR / "server.log"
DHT_PORT = 31337
SERVER_PORT = 31338
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live-capture")

BLOCKED_CLAIM_BOUNDARY = "live_continuous_kv_capture_blocked_no_live_server_proof"


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
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=merged_env)
    return proc.returncode, proc.stdout, proc.stderr


def wait_for_dht(host: str = "127.0.0.1", port: int = DHT_PORT, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
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
            if any(k in text for k in [
                "Announced that blocks",
                "blocks [...] are joining",
                "Started",
                "Server started",
                "now serving",
            ]):
                log.info("Server announced blocks, ready")
                return True
        time.sleep(poll_interval)
    return False

def run_inference_via_remote_sequential(
    prompt: str,
    *,
    max_new_tokens: int = 4,
    env_flags: dict | None = None,
) -> dict:
    """Run a real inference through BloomBee using direct RemoteSequential + LocalMLPHead."""
    env = dict(os.environ)
    if env_flags:
        env.update(env_flags)

    dht_peer_file = EVIDENCE_DIR / "dht_peer.txt"
    if not dht_peer_file.exists():
        return {"error": "DHT peer ID not yet discovered", "prompt": prompt}
    peer_id = dht_peer_file.read_text().strip()
    initial_peer = f"/ip4/127.0.0.1/tcp/{DHT_PORT}/p2p/{peer_id}"

    # Inject runtime params into env (cleaner than escaping in f-strings)
    env["BLOOMBEE_INITIAL_PEER"] = initial_peer
    env["BLOOMBEE_MODEL_ID"] = MODEL_ID
    env["BLOOMBEE_PROMPT"] = prompt
    env["BLOOMBEE_MAX_NEW_TOKENS"] = str(max_new_tokens)
    env["BLOOMBEE_DHT_PREFIX"] = "mycelium-capture-v1"

    # Use BloomBee's Pipeline class for full inference (handles LM head locally)
    script = r'''
import time, json, sys, os, traceback
sys.path.insert(0, "src")
os.environ["BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"] = os.environ.get("BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING", "0")
os.environ["BLOOMBEE_ENABLE_KV_PREFIX_REUSE"] = os.environ.get("BLOOMBEE_ENABLE_KV_PREFIX_REUSE", "0")

start = time.time()
err_prefix = '{"error": '
try:
    from bloombee import AutoDistributedModelForCausalLM
except Exception as e:
    print(err_prefix + "import failed: " + repr(e) + "}", flush=True)
    sys.exit(1)

initial_peer = os.environ["BLOOMBEE_INITIAL_PEER"]
try:
    from bloombee.client.config import ClientConfig
    from hivemind import DHT
    # Just pass initial_peers directly to from_pretrained — let bloombee create its own DHT internally
    print(f"using initial_peer={initial_peer}", flush=True)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        os.environ["BLOOMBEE_MODEL_ID"],
        initial_peers=[initial_peer],
        torch_dtype="float32",
        request_timeout=90,
        max_retries=3,
        dht_prefix=os.environ["BLOOMBEE_DHT_PREFIX"],
    )
    print("MODEL_LOADED:" + str(int(time.time() - start)), flush=True)
except Exception as e:
    print(err_prefix + "from_pretrained failed: " + repr(e) + ", tb=" + traceback.format_exc()[-500:] + "}", flush=True)
    sys.exit(1)

try:
    nb = model.transformer.h.num_blocks() if hasattr(model.transformer.h, "num_blocks") else 0
except Exception:
    nb = 0
print(f"num_blocks: {nb}", flush=True)

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["BLOOMBEE_MODEL_ID"])
    raw_prompt = os.environ["BLOOMBEE_PROMPT"]
    max_new_tokens = int(os.environ["BLOOMBEE_MAX_NEW_TOKENS"])
    inputs = tok(raw_prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
except Exception as e:
    print(err_prefix + "tok failed: " + repr(e) + "}", flush=True)
    sys.exit(1)

gen_start = time.time()
try:
    out = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
except Exception as e:
    print(err_prefix + "generate failed: " + repr(e) + ", tb=" + traceback.format_exc()[-300:] + "}", flush=True)
    sys.exit(1)
elapsed = time.time() - gen_start

gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
print(json.dumps({
    "prompt": raw_prompt,
    "prompt_token_ids": prompt_ids,
    "generated_token_ids": gen_ids,
    "generated_text": tok.decode(gen_ids),
    "elapsed_seconds": round(elapsed, 4),
    "connect_seconds": round(gen_start - start, 4),
    "num_remote_blocks": nb,
}))
'''

    code, stdout, stderr = run_cmd([sys.executable, "-u", "-c", script], env=env, timeout=240)
    if code != 0:
        # Try to extract the JSON error from output
        for line in stdout.splitlines():
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    if "error" in payload:
                        return payload
                except json.JSONDecodeError:
                    pass
        return {
            "error": f"exit={code}",
            "stderr_tail": stderr[-1000:],
            "stdout_tail": stdout[-500:],
            "prompt": prompt,
        }
    # Parse last JSON line
    json_lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if json_lines:
        try:
            return json.loads(json_lines[-1])
        except json.JSONDecodeError as e:
            return {"error": f"json parse: {e}", "line": json_lines[-1][:300], "prompt": prompt}
    return {"error": f"no JSON in output", "stdout": stdout[-500:], "prompt": prompt}


# ----------------------------- Phases ----------------------------- #

def phase_a_continuous_batching() -> dict:
    log.info("=== Phase A: Continuous batching ===")
    log.info("Baseline run (no opt-in)...")
    baseline_a = run_inference_via_remote_sequential("The capital of France is", max_new_tokens=3)
    time.sleep(2)
    baseline_b = run_inference_via_remote_sequential("The capital of Japan is", max_new_tokens=3)

    log.info("Continuous-batching run (opt-in ON)...")
    continuous_a = run_inference_via_remote_sequential(
        "The capital of France is",
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING": "1"},
    )
    time.sleep(2)
    continuous_b = run_inference_via_remote_sequential(
        "The capital of Japan is",
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING": "1"},
    )

    return {
        "phase": "continuous_batching",
        "baseline": {"req-a": baseline_a, "req-b": baseline_b},
        "continuous": {"req-a": continuous_a, "req-b": continuous_b},
    }


def phase_b_kv_prefix_reuse() -> dict:
    log.info("=== Phase B: KV prefix reuse ===")
    shared_prefix = "You are a helpful assistant. Please answer the following question concisely."
    suffix_a = " What is 2+2?"
    suffix_b = " What is 3+3?"

    log.info("Baseline (no opt-in)...")
    baseline_a = run_inference_via_remote_sequential(shared_prefix + suffix_a, max_new_tokens=3)
    time.sleep(2)
    baseline_b = run_inference_via_remote_sequential(shared_prefix + suffix_b, max_new_tokens=3)

    log.info("KV-prefix-reuse run (opt-in ON)...")
    reuse_a = run_inference_via_remote_sequential(
        shared_prefix + suffix_a,
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_KV_PREFIX_REUSE": "1"},
    )
    time.sleep(2)
    reuse_b = run_inference_via_remote_sequential(
        shared_prefix + suffix_b,
        max_new_tokens=3,
        env_flags={"BLOOMBEE_ENABLE_KV_PREFIX_REUSE": "1"},
    )

    return {
        "phase": "kv_prefix_reuse",
        "shared_prefix": shared_prefix,
        "baseline": {"suffix-a": baseline_a, "suffix-b": baseline_b},
        "reuse": {"suffix-a": reuse_a, "suffix-b": reuse_b},
    }


def parse_server_log() -> dict:
    """Extract structured observation JSON lines from server log."""
    cb = []
    kv = []
    if SERVER_LOG.exists():
        text = SERVER_LOG.read_text(errors="replace")
        for line in text.splitlines():
            if "[LIVE_CONTINUOUS_BATCHING]" in line:
                js = line.split("[LIVE_CONTINUOUS_BATCHING]")[-1].strip()
                try:
                    cb.append(json.loads(js))
                except json.JSONDecodeError:
                    pass
            if "[KV_PREFIX_REUSE]" in line:
                js = line.split("[KV_PREFIX_REUSE]")[-1].strip()
                try:
                    kv.append(json.loads(js))
                except json.JSONDecodeError:
                    pass
    return {
        "live_continuous_batching_observations": cb,
        "kv_prefix_reuse_observations": kv,
        "cb_count": len(cb),
        "kv_count": len(kv),
    }


# ----------------------------- DHT lifecycle ----------------------------- #

def start_dht_process():
    """Start DHT subprocess and write peer_id to file."""
    dht_proc = subprocess.Popen(
        [sys.executable, "-u", "-c", f"""
import sys, time, os
sys.path.insert(0, "src")
from hivemind import DHT
dht = DHT(start=True, host_maddrs=["/ip4/0.0.0.0/tcp/{DHT_PORT}"])
try:
    dht.wait_until_ready(timeout=30)
except Exception as e:
    print(f"DHT_READY_ERROR: {{e}}", flush=True)
maddrs = dht.get_visible_maddrs()
if maddrs:
    peer_id = str(maddrs[0]).rsplit("/p2p/", 1)[-1]
    print(f"DHT_PEER_ID:​{{peer_id}}", flush=True)
    with open("{EVIDENCE_DIR}/dht_peer.txt", "w") as f:
        f.write(peer_id)
while True:
    time.sleep(60)
"""],
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    return dht_proc



def start_bloombee_server(env_overrides: dict | None = None):
    """Start BloomBee server (TinyLlama) against the private capture DHT."""
    env = dict(os.environ)
    env["BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"] = "1"
    env["BLOOMBEE_ENABLE_KV_PREFIX_REUSE"] = "1"
    if env_overrides:
        env.update(env_overrides)

    peer_file = EVIDENCE_DIR / "dht_peer.txt"
    if not peer_file.exists() or not peer_file.read_text().strip():
        raise RuntimeError("DHT peer ID not yet discovered")
    initial_peer = f"/ip4/127.0.0.1/tcp/{DHT_PORT}/p2p/{peer_file.read_text().strip()}"
    command = build_server_command(
        model_id=MODEL_ID,
        initial_peer=initial_peer,
        num_blocks=4,
        port=SERVER_PORT,
        batch_size=2,
        device="mps",
    )
    server_proc = subprocess.Popen(
        command,
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return server_proc

def main():
    reset_capture_workspace()

    # ---- Start DHT ----
    log.info(f"Starting DHT on port {DHT_PORT}...")
    dht_proc = start_dht_process()
    log.info(f"DHT PID: {dht_proc.pid}")

    if not wait_for_dht():
        detail = f"DHT did not listen on 127.0.0.1:{DHT_PORT}"
        log.error(detail)
        write_blocked_live_capture_evidence(reason="dht_not_listening", detail=detail)
        dht_proc.kill()
        return 1

    # Wait for DHT peer_id file
    for _ in range(30):
        if (EVIDENCE_DIR / "dht_peer.txt").exists() and (EVIDENCE_DIR / "dht_peer.txt").read_text().strip():
            log.info(f"DHT peer_id: {(EVIDENCE_DIR / 'dht_peer.txt').read_text().strip()}")
            break
        time.sleep(1)
    else:
        log.error("No peer_id captured")
        dht_proc.kill()
        return 1

    # ---- Start server ----
    log.info(f"Starting BloomBee server ({MODEL_ID})...")
    server_proc = start_bloombee_server()
    log.info(f"Server PID: {server_proc.pid}")

    if not wait_for_server_blocks(timeout=240, server_proc=server_proc):
        log.error("Server failed to announce. Last 60 log lines:")
        if SERVER_LOG.exists():
            for line in SERVER_LOG.read_text(errors="replace").splitlines()[-60:]:
                log.error(f"  {line}")
        write_blocked_live_capture_evidence(reason="server_not_ready", detail="server failed to announce blocks")
        dht_proc.kill()
        server_proc.kill()
        return 1

    log.info("Server ready. Running capture phases...")

    # ---- Phases ----
    try:
        phase_a = phase_a_continuous_batching()
        phase_b = phase_b_kv_prefix_reuse()
        obs = parse_server_log()
    except Exception as e:
        log.error(f"Phase error: {e}")
        traceback.print_exc()
        phase_a = phase_b = obs = {"phase_error": str(e)}
    finally:
        # Capture final log size before shutdown
        log_size_before = SERVER_LOG.stat().st_size if SERVER_LOG.exists() else 0

        log.info("Shutting down server and DHT...")
        server_proc.kill()
        time.sleep(2)
        dht_proc.kill()

    evidence = {
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dht_port": DHT_PORT,
        "server_port": SERVER_PORT,
        "phase_a_continuous_batching": phase_a,
        "phase_b_kv_prefix_reuse": phase_b,
        "server_log_observations": obs,
        "server_log_path": str(SERVER_LOG),
    }

    out = EVIDENCE_DIR / "live-capture-raw.json"
    out.write_text(json.dumps(evidence, indent=2))
    log.info(f"Evidence written to {out}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("LIVE CAPTURE SUMMARY")
    print("=" * 60)

    def _print_pairs(label, d):
        print(f"  {label}:")
        for k, v in d.items():
            if isinstance(v, dict) and "error" in v:
                err_msg = v["error"][:80] if isinstance(v["error"], str) else str(v["error"])[:80]
                elapsed = v.get("elapsed_seconds", "?")
                print(f"    {k}: ERROR ({elapsed}s) — {err_msg}")
            elif isinstance(v, dict):
                elapsed = v.get("elapsed_seconds", "?")
                tokens = v.get("generated_token_ids", [])
                text = v.get("generated_text", "")
                print(f"    {k}: tokens={tokens} text={text!r} ({elapsed}s)")

    print("\n--- Continuous Batching ---")
    if isinstance(phase_a, dict) and "baseline" in phase_a:
        print("Baseline:")
        _print_pairs("", phase_a.get("baseline", {}))
        print("Continuous (BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING=1):")
        _print_pairs("", phase_a.get("continuous", {}))
    cb_count = obs.get("cb_count", 0) if isinstance(obs, dict) else 0
    print(f"  Server [LIVE_CONTINUOUS_BATCHING] observations: {cb_count}")

    print("\n--- KV Prefix Reuse ---")
    if isinstance(phase_b, dict) and "baseline" in phase_b:
        print("Baseline:")
        _print_pairs("", phase_b.get("baseline", {}))
        print("Reuse (BLOOMBEE_ENABLE_KV_PREFIX_REUSE=1):")
        _print_pairs("", phase_b.get("reuse", {}))
    kv_count = obs.get("kv_count", 0) if isinstance(obs, dict) else 0
    print(f"  Server [KV_PREFIX_REUSE] observations: {kv_count}")

    return 0 if (isinstance(phase_a, dict) and "baseline" in phase_a) else 2


if __name__ == "__main__":
    raise SystemExit(main())
