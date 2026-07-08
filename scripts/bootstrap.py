#!/usr/bin/env python3
"""Self-contained bootstrap script for joining a BloomBee distributed inference swarm.

No dependencies outside Python stdlib. Run this on any machine to:
  1. Scan hardware capabilities (CPU, RAM, GPU, disk, network)
  2. Send a heartbeat to the coordinator with those capabilities
  3. Keep heartbeating to stay in the active roster

Usage:
  # From a join URL (scan the QR):
  python3 bootstrap.py --join-url "bloombee://join?coordinator=http%3A%2F%2F192.168.1.100%3A8787&token=abc123"

  # Download and run in one command:
  curl -s http://COORDINATOR:8787/bootstrap.py | python3 - --join-url "bloombee://join?..."

  # Keep heartbeating every 30 seconds:
  python3 bootstrap.py --join-url "..." --loop --interval 30

The script is intentionally dependency-free — pure Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import re
import uuid
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


# ── peer scan ────────────────────────────────────────────────────────────────

def scan_capabilities() -> dict[str, Any]:
    """Detect hardware capabilities of this machine. No ML deps required."""
    hostname = socket.gethostname().split(".")[0]
    # Android/Termux often returns 'localhost' — try to get real device name
    if hostname in ("", "localhost", "127.0.0.1", "::1"):
        try:
            out = subprocess.run(["getprop", "ro.product.model"], capture_output=True, text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                hostname = out.stdout.strip().replace(" ", "-")
        except Exception:
            pass
    if hostname in ("", "localhost", "127.0.0.1", "::1"):
        hostname = f"android-{uuid.getnode():x}"[:24]

    # CPU
    cpu_model = _platform.processor() or "unknown"
    cpu_cores = os.cpu_count() or 1

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 1)
        ram_available_gb = round(mem.available / (1024**3), 1)
    except ImportError:
        # Fallback: sysctl on macOS, /proc/meminfo on Linux, free command on Android
        ram_total_gb = 0.0
        ram_available_gb = 0.0
        try:
            if sys.platform == "darwin":
                out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
                ram_total_gb = round(int(out.stdout.strip()) / (1024**3), 1)
            elif sys.platform == "linux":
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if "MemTotal" in line:
                                ram_total_gb = round(int(line.split()[1]) / (1024**2), 1)
                                break
                except (OSError, PermissionError):
                    pass
                # /proc/meminfo failed? Try 'free' command (works on Termux)
                if not ram_total_gb:
                    try:
                        out = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=3)
                        for line in out.stdout.split("\n"):
                            if line.startswith("Mem:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    ram_total_gb = round(int(parts[1]) / (1024**3), 1)
                                break
                    except Exception:
                        pass
        except Exception:
            pass
        if not ram_total_gb:
            print("   ⚠️  Could not detect RAM — reporting 0 GB", file=sys.stderr)
        ram_available_gb = ram_total_gb  # best guess

    # GPU
    gpu_info: dict[str, Any] = {"available": False, "name": "none"}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {"available": True, "name": torch.cuda.get_device_name(0), "backend": "cuda"}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = {"available": True, "name": "Apple Metal (MPS)", "backend": "mps"}
    except ImportError:
        pass

    # Disk
    disk_total_gb = 0
    disk_free_gb = 0
    try:
        usage = shutil.disk_usage(Path.home())
        disk_total_gb = round(usage.total / (1024**3), 1)
        disk_free_gb = round(usage.free / (1024**3), 1)
    except Exception:
        pass

    # Network — detect local IP
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("1.1.1.1", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    return {
        "hostname": hostname,
        "peer_id": f"{hostname}-{uuid.getnode():x}",
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "has_python": shutil.which("python") is not None,
        "has_python3": shutil.which("python3") is not None,
        "cpu": {"model": cpu_model, "cores": cpu_cores},
        "memory": {"total_gb": ram_total_gb, "available_gb": ram_available_gb},
        "gpu": gpu_info,
        "disk": {"total_gb": disk_total_gb, "free_gb": disk_free_gb},
        "network": {"local_ip": local_ip},
        "scanned_at": int(time.time()),
    }


# ── join client ───────────────────────────────────────────────────────────────

def parse_join_url(join_url: str) -> dict[str, str]:
    """Parse a bloombee://join?... URL into {coordinator, token}."""
    parsed = urlparse(join_url)
    query = parse_qs(parsed.query)
    coordinator = (query.get("coordinator") or [None])[0]
    token = (query.get("token") or [None])[0]
    if not coordinator or not token:
        raise ValueError(f"Invalid join URL — missing coordinator or token: {join_url}")
    return {"coordinator": coordinator, "token": token}


def send_heartbeat(coordinator: str, token: str, capabilities: dict[str, Any]) -> dict[str, Any]:
    """POST a heartbeat to the coordinator. Returns the response JSON."""
    payload = json.dumps({
        "token": token,
        "peer_id": capabilities["peer_id"],
        "capabilities": capabilities,
    }, sort_keys=True).encode("utf-8")

    url = coordinator.rstrip("/") + "/heartbeat"
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── auto-serve ───────────────────────────────────────────────────────────────

def poll_job_for_peer(coordinator: str, peer_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET /job?peer_id=X — returns job dict or {'job': None}."""
    url = f"{coordinator.rstrip('/')}/job?peer_id={peer_id}"
    req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e), "job": None}


def post_peer_status(
    coordinator: str,
    peer_id: str,
    *,
    status: str,
    progress: float | None = None,
    message: str | None = None,
    job_port: int | None = None,
    model_id: str | None = None,
) -> None:
    """POST /peer-status — report live state to the coordinator.

    Best-effort: any network error is swallowed so a flaky status channel
    doesn't kill the server. The coordinator reads this to drive the
    dashboard progress bar / status pills.
    """
    payload = {"peer_id": peer_id, "status": status}
    if progress is not None:
        payload["progress"] = float(progress)
    if message:
        payload["message"] = message
    if job_port is not None:
        payload["job_port"] = int(job_port)
    if model_id:
        payload["model_id"] = model_id
    try:
        req = Request(
            f"{coordinator.rstrip('/')}/peer-status",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=3).read()
    except Exception:
        pass  # best-effort


def post_seed_multiaddr(
    coordinator: str,
    *,
    hostname: str,
    peer_id: str,
    multiaddrs: list[str],
    job_port: int | None = None,
    model_id: str | None = None,
) -> None:
    """POST /seed-multiaddr — publish visible libp2p multiaddrs.

    Best-effort. Followers poll /job until this appears, then coordinator
    substitutes <SEED_MULTIADDR_FROM_HOST> into their launch command.
    """
    payload: dict[str, Any] = {
        "hostname": hostname,
        "peer_id": peer_id,
        "multiaddrs": multiaddrs,
    }
    if job_port is not None:
        payload["job_port"] = int(job_port)
    if model_id:
        payload["model_id"] = model_id
    try:
        req = Request(
            f"{coordinator.rstrip('/')}/seed-multiaddr",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=3).read()
    except Exception:
        pass


_MULTIADDR_RE = re.compile(r"(/(?:ip4|ip6|dns4|dns6|p2p-circuit)[^\s,'\"\]]*/p2p/[A-Za-z0-9]+)")


def extract_multiaddrs(text: str) -> list[str]:
    """Extract libp2p multiaddrs from BloomBee/Hivemind log text."""
    found: list[str] = []
    for match in _MULTIADDR_RE.findall(text):
        if match not in found:
            found.append(match)
    return found


# The hivemind Runtime logs exactly "Started" immediately after it sets the
# `ready` event ("set iff server is currently running and ready to accept
# batches"). That is the ONLY authoritative readiness signal: it fires *after*
# every block's weights are loaded. Earlier lines like "Running a server on
# ..." (server prints its multiaddr) and "Announced that blocks ... are
# joining" happen before weights load — treating them as "serving" is the bug
# that showed stuck, still-downloading peers as green on the dashboard.
_SERVER_READY_RE = re.compile(r"(?:^|\])\s*Started\s*$")


def is_server_ready_line(line: str) -> bool:
    """True only when a server log line proves the server finished loading its
    blocks and is ready to serve requests (hivemind Runtime's "Started")."""
    return bool(_SERVER_READY_RE.search(line.rstrip()))


def model_weights_cached(model_id: str, cache_dir: "str | Path | None" = None) -> bool:
    """Return True iff the HF cache holds real weight files for ``model_id``.

    A config-only cache entry (just ``config.json`` / tokenizer / a bare
    ``*.index.json``) is NOT enough — the server would hang for a long time
    downloading the actual shards. Preflighting this lets the bootstrap fail
    fast with an actionable message instead of appearing to "serve" forever.
    """
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = Path(cache_dir)

    model_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
    if not model_dir.is_dir():
        return False

    for pattern in ("*.safetensors", "*.bin"):
        for weight_file in model_dir.rglob(pattern):
            # rglob("*.safetensors") never matches "*.safetensors.index.json"
            # (that ends in .json), so any hit here is a real weight file.
            try:
                if weight_file.stat().st_size > 0:  # stat() follows symlinks (HF blobs)
                    return True
            except OSError:
                continue
    return False


def _heartbeat_status_thread(
    coordinator: str,
    peer_id: str,
    job_port: int | None,
    model_id: str | None,
    stop_event,
) -> None:
    """Background thread: re-posts status: 'serving' every 4s while the
    server runs, so the dashboard sees a fresh timestamp."""
    while not stop_event.is_set():
        post_peer_status(
            coordinator,
            peer_id,
            status="serving",
            progress=100.0,
            job_port=job_port,
            model_id=model_id,
        )
        stop_event.wait(4.0)


def execute_job_command(
    command: str,
    *,
    cwd: str | None = None,
    coordinator: str | None = None,
    peer_id: str | None = None,
    hostname: str | None = None,
    job_port: int | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Execute a BloomBee server command and stream status to coordinator.

    Uses Popen (not subprocess.run) so we can read server logs while the
    long-running server is still alive. As soon as BloomBee prints its
    visible libp2p multiaddr, the seed peer POSTs it to /seed-multiaddr;
    follower peers keep polling /job until the coordinator substitutes it.
    """
    import threading as _threading
    import os as _os

    coord: str | None = coordinator
    pid: str | None = peer_id
    host = hostname or (peer_id.split("-", 1)[0] if peer_id else "")

    # ── Resolve project root for PYTHONPATH ─────────────────────────
    # The command may contain "PYTHONPATH=.:src" which assumes cwd is
    # the project root. If the bootstrap is running from ~/ or /tmp,
    # ".:src" won't find bloombee. Walk up from the effective cwd to
    # locate src/bloombee/cli/run_server.py, then set PYTHONPATH so the
    # subprocess can import bloombee regardless of working directory.
    effective_cwd = cwd or _os.getcwd()
    repo_root = effective_cwd
    found = False
    for _ in range(6):
        if (_os.path.isdir(_os.path.join(repo_root, "src", "bloombee", "cli"))
                and _os.path.isfile(_os.path.join(repo_root, "src", "bloombee", "cli", "run_server.py"))):
            found = True
            break
        # Also check immediate subdirectories at this level, so if the
        # bootstrap is run from ~/Projects (parent of the actual repo) we
        # still discover ~/Projects/distributed-inference-mvp.
        try:
            for entry in _os.listdir(repo_root):
                sub = _os.path.join(repo_root, entry)
                if _os.path.isdir(sub) and _os.path.isdir(_os.path.join(sub, "src", "bloombee", "cli")):
                    if _os.path.isfile(_os.path.join(sub, "src", "bloombee", "cli", "run_server.py")):
                        repo_root = sub
                        found = True
                        break
        except PermissionError:
            pass
        if found:
            break
        parent = _os.path.dirname(repo_root)
        if parent == repo_root:
            repo_root = effective_cwd  # give up, fall back to cwd
            break
        repo_root = parent
    resolved_pythonpath = f"{repo_root}:{_os.path.join(repo_root, 'src')}"
    # Strip any hardcoded PYTHONPATH= prefix from the command itself;
    # we set it via the subprocess env instead so cwd doesn't matter.
    import re as _re
    clean_command = _re.sub(r"^PYTHONPATH=[^ ]+\s+", "", command)
    # Detect the virtualenv Python so dependencies (hivemind, torch,
    # transformers) are importable. Falls back to the python3 on PATH.
    venv_python = _os.path.join(repo_root, ".venv", "bin", "python")
    if not _os.path.isfile(venv_python):
        venv_python = _os.path.join(repo_root, ".venv", "bin", "python3")
    if _os.path.isfile(venv_python):
        # Replace 'python3' first; only try bare 'python' if 'python3'
        # wasn't in the command, to avoid double-replacing the venv path
        # (e.g. .../venv/bin/python → .../venv/bin/.../venv/bin/python).
        if "python3 " in clean_command:
            clean_command = clean_command.replace("python3 ", f"{venv_python} ", 1)
        elif "python " in clean_command:
            clean_command = clean_command.replace("python ", f"{venv_python} ", 1)
    have_status = bool(coord and pid)
    # ────────────────────────────────────────────────────────────────

    print(f"   🚀 Executing: {clean_command}", file=sys.stderr)

    # ── Preflight: the model weights must actually be in the HF cache ─
    # A config-only cache entry makes run_server hang for a long time
    # pulling ~GBs of shards; the old flow then reported that stuck server
    # as "serving" the instant it printed its multiaddr. Fail fast with an
    # actionable message instead of hanging while masquerading as green.
    if model_id and not model_weights_cached(model_id):
        msg = (
            f"Model weights for '{model_id}' are not downloaded (HF cache has "
            f"config only). Pre-download them on this machine before serving: "
            f"huggingface-cli download {model_id}"
        )
        print(f"   ❌ {msg}", file=sys.stderr)
        if have_status:
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status="error", progress=None,
                job_port=job_port, model_id=model_id,
                message=msg,
            )
        return {
            "command": command,
            "exit_code": 2,
            "stdout_tail": "",
            "stderr_tail": msg,
            "weights_missing": True,
        }

    if have_status:
        post_peer_status(
            coord, pid,  # type: ignore[arg-type]
            status="downloading", progress=0.0,
            job_port=job_port, model_id=model_id,
            message="Model download / server startup beginning",
        )

    stop_progress = _threading.Event()
    stop_serving = _threading.Event()
    serving_thread_started = False

    def _progress_pump() -> None:
        if not have_status:
            return
        start = time.time()
        stage = "downloading"
        posted_loading = False
        while not stop_progress.is_set():
            elapsed = time.time() - start
            if not posted_loading and elapsed >= 30.0:
                stage = "loading"
                post_peer_status(
                    coord, pid,  # type: ignore[arg-type]
                    status="loading", progress=50.0,
                    job_port=job_port, model_id=model_id,
                    message="Loading model into memory / starting server",
                )
                posted_loading = True
            if not posted_loading:
                pct = min(45.0, (elapsed / 30.0) * 45.0)
            else:
                pct = 50.0 + min(40.0, ((elapsed - 30.0) / 60.0) * 40.0)
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status=stage, progress=pct,
                job_port=job_port, model_id=model_id,
            )
            stop_progress.wait(2.0)

    if have_status:
        _threading.Thread(target=_progress_pump, daemon=True).start()

    output_lines: list[str] = []
    posted_multiaddrs: set[str] = set()
    detected_serving = False
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            clean_command,
            shell=True,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={
                **_os.environ,
                "PYTHONPATH": resolved_pythonpath,
                "PATH": f"{_os.path.join(repo_root, '.venv', 'bin')}:{_os.environ.get('PATH', '')}",
                "VIRTUAL_ENV": _os.path.join(repo_root, ".venv"),
            },
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            if len(output_lines) > 400:
                output_lines = output_lines[-400:]
            # Surface server logs in the peer terminal for operator debugging.
            print("   │ " + line.rstrip(), file=sys.stderr)

            multiaddrs = extract_multiaddrs(line)
            new_multiaddrs = [addr for addr in multiaddrs if addr not in posted_multiaddrs]
            if coord and pid and host and new_multiaddrs:
                posted_multiaddrs.update(new_multiaddrs)
                post_seed_multiaddr(
                    coord,
                    hostname=host,
                    peer_id=pid,
                    multiaddrs=sorted(posted_multiaddrs),
                    job_port=job_port,
                    model_id=model_id,
                )
                post_peer_status(
                    coord, pid,
                    status="loading", progress=90.0,
                    job_port=job_port, model_id=model_id,
                    message="published seed multiaddr; followers can start",
                )

            # Only flip to "serving" on the real readiness marker (hivemind
            # Runtime's "Started"), which fires AFTER every block's weights
            # are loaded — not on "Running a server on ..." (printed before
            # weight load), which used to mask stuck, still-downloading peers
            # as green. The multiaddr POST above still fires early (status
            # "loading") so followers can start bootstrapping meanwhile.
            if is_server_ready_line(line):
                detected_serving = True
                stop_progress.set()
                if have_status:
                    post_peer_status(
                        coord, pid,  # type: ignore[arg-type]
                        status="serving", progress=100.0,
                        job_port=job_port, model_id=model_id,
                        message="server is running",
                    )
                    if not serving_thread_started:
                        _threading.Thread(
                            target=_heartbeat_status_thread,
                            args=(coord, pid, job_port, model_id, stop_serving),  # type: ignore[arg-type]
                            daemon=True,
                        ).start()
                        serving_thread_started = True
        return_code = proc.wait()
    except Exception as exc:
        stop_progress.set()
        stop_serving.set()
        if have_status:
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status="error", progress=None,
                job_port=job_port, model_id=model_id,
                message=f"launcher error: {exc}",
            )
        return {"command": command, "exit_code": 127, "stdout_tail": "", "stderr_tail": str(exc)}
    finally:
        stop_progress.set()
        stop_serving.set()

    full_output = "".join(output_lines)
    if have_status:
        if return_code == 0 and detected_serving:
            post_peer_status(coord, pid, status="serving", progress=100.0, job_port=job_port, model_id=model_id)  # type: ignore[arg-type]
        elif return_code == 0:
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status="serving", progress=100.0,
                job_port=job_port, model_id=model_id,
                message="(launcher exited; server may still be running in background)",
            )
        else:
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status="error", progress=None,
                job_port=job_port, model_id=model_id,
                message=f"exit code {return_code}",
            )

    return {
        "command": command,
        "exit_code": return_code,
        "stdout_tail": full_output[-2000:],
        "stderr_tail": "",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BloomBee distributed inference — bootstrap & join",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bootstrap.py --join-url "bloombee://join?coordinator=http%3A%2F%2F192.168.1.100%3A8787&token=abc123"
  python3 bootstrap.py --join-url "bloombee://join?..." --loop --interval 30
  python3 bootstrap.py --join-url "bloombee://join?..." --scan-only
        """,
    )
    parser.add_argument("--join-url", required=True, help="Join URL from QR code or share link")
    parser.add_argument("--loop", action="store_true", help="Keep sending heartbeats")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between heartbeats (default: 30)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan capabilities, don't join")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--auto-serve", action="store_true",
                        help="After heartbeat, poll /job for a deployment assignment and execute it. "
                             "Used with --loop for zero-touch deploy.")
    args = parser.parse_args()

    # Step 1: Scan
    if not args.json:
        print("🔍 Scanning hardware capabilities...", file=sys.stderr)
    capabilities = scan_capabilities()

    if args.json:
        print(json.dumps(capabilities, indent=2))
    else:
        print(f"   Hostname: {capabilities['hostname']}", file=sys.stderr)
        print(f"   Peer ID:  {capabilities['peer_id']}", file=sys.stderr)
        print(f"   CPU:      {capabilities['cpu']['model']} ({capabilities['cpu']['cores']} cores)", file=sys.stderr)
        print(f"   RAM:      {capabilities['memory']['total_gb']} GB", file=sys.stderr)
        print(f"   GPU:      {capabilities['gpu']['name']}", file=sys.stderr)
        print(f"   Platform: {capabilities['platform']}", file=sys.stderr)

    if args.scan_only:
        return

    # Step 2: Parse join URL
    join = parse_join_url(args.join_url)
    coordinator = join["coordinator"]
    token = join["token"]

    if not args.json:
        print(f"\n🔗 Joining swarm at {coordinator}...", file=sys.stderr)

    # Step 3: Send initial heartbeat
    response = send_heartbeat(coordinator, token, capabilities)
    if response.get("ok"):
        if not args.json:
            print(f"   ✅ Connected! Peer '{capabilities['peer_id']}' registered.", file=sys.stderr)
    else:
        error = response.get("error", "unknown error")
        print(f"   ❌ Failed to join: {error}", file=sys.stderr)
        if not args.loop:
            sys.exit(1)

    # Step 4: Loop
    if args.loop:
        if not args.json:
            extra = " + auto-serve" if args.auto_serve else ""
            print(f"   💓 Heartbeating every {args.interval}s{extra}... (Ctrl+C to stop)", file=sys.stderr)
        job_executed = False
        try:
            while True:
                time.sleep(args.interval)
                capabilities["scanned_at"] = int(time.time())
                response = send_heartbeat(coordinator, token, capabilities)
                if not args.json:
                    status = "✅" if response.get("ok") else "❌"
                    print(f"   {status} Heartbeat at {time.strftime('%H:%M:%S')}", file=sys.stderr)

                # Auto-serve: poll for a deployment job and execute it
                if args.auto_serve and not job_executed:
                    poll = poll_job_for_peer(coordinator, capabilities["peer_id"])
                    job = poll.get("job")
                    if job and job.get("status") == "waiting_for_seed":
                        if not args.json:
                            print("   ⏳ Job assigned; waiting for seed multiaddr...", file=sys.stderr)
                        post_peer_status(
                            coordinator, capabilities["peer_id"],
                            status="waiting_for_seed", progress=None,
                            job_port=job.get("port"),
                            model_id=poll.get("deployed_model"),
                            message=job.get("message") or "waiting for seed multiaddr",
                        )
                        continue
                    if job and not job.get("command"):
                        post_peer_status(
                            coordinator, capabilities["peer_id"],
                            status="queued", progress=None,
                            job_port=job.get("port"),
                            model_id=poll.get("deployed_model"),
                            message="job assigned but command not ready yet",
                        )
                        continue
                    if job and job.get("command"):
                        cmd = job["command"]
                        # Defensive fallback for older coordinators: never run
                        # raw shell placeholders, because <...> is redirection
                        # syntax and fails as exit 127.
                        if "<SEED_MULTIADDR_FROM_" in cmd or "<PASTE_SERVER_" in cmd:
                            print(
                                f"\n   ⏳ Job for {poll.get('deployed_model','?')} still has an unsubstituted "
                                f"seed multiaddr placeholder; waiting for coordinator substitution.",
                                file=sys.stderr,
                            )
                            post_peer_status(
                                coordinator, capabilities["peer_id"],
                                status="waiting_for_seed", progress=None,
                                job_port=job.get("port"),
                                model_id=poll.get("deployed_model"),
                                message="waiting for seed multiaddr substitution",
                            )
                            continue
                        print(f"\n   📦 Got job for model: {poll.get('deployed_model', '?')}", file=sys.stderr)
                        print(f"   🎯 Assigned: {job.get('role', '?')} {job.get('block_range', '')} port {job.get('port', '?')}", file=sys.stderr)
                        result = execute_job_command(
                            cmd,
                            coordinator=coordinator,
                            peer_id=capabilities["peer_id"],
                            hostname=capabilities["hostname"],
                            job_port=job.get("port"),
                            model_id=poll.get("deployed_model"),
                        )
                        print(f"   🏁 Server exited with code {result.get('exit_code')}", file=sys.stderr)
                        job_executed = True  # only run once per session
        except KeyboardInterrupt:
            if not args.json:
                print("\n👋 Disconnected.", file=sys.stderr)

    if args.json:
        print(json.dumps({"ok": True, "peer_id": capabilities["peer_id"], "coordinator": coordinator}))


if __name__ == "__main__":
    main()
