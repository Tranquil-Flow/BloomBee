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
from urllib.request import ProxyHandler, Request, build_opener, install_opener, urlopen


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
    """Parse a bloombee://join?... URL into {coordinator, token, coordinators}.

    The single primary ``coordinator`` is preserved as-is for backwards
    compatibility. Any numbered ``coordinator_2`` / ``coordinator_3`` /
    ``..._N`` query parameters are collected in ``coordinators`` (extra
    candidates the bootstrap should try if the primary is unreachable) —
    this is how the offer endpoint signals "the QR has multiple URLs,
    try each one". Multi-IP offers use this so a peer whose en0 IP has
    changed (e.g. from home LAN to coffee-shop wifi) still finds the
    coordinator.

    Reserved join-URL parameter names:
        coordinator, token, coordinator_2 .. coordinator_N
    All other parameters are ignored for now.
    """
    parsed = urlparse(join_url)
    query = parse_qs(parsed.query)
    primary = (query.get("coordinator") or [None])[0]
    token = (query.get("token") or [None])[0]
    if not primary or not token:
        raise ValueError(f"Invalid join URL — missing coordinator or token: {join_url}")
    # Collect numbered fallbacks in numeric order (coordinator_2, _3, ...).
    # We deliberately don't wildcard-match "coordinator_*" because we
    # want stable, reviewable behaviour.
    extras: list[str] = []
    indexed_extras: list[tuple[int, str]] = []
    for key, values in query.items():
        if key.startswith("coordinator_") and key[len("coordinator_"):].isdigit():
            for val in values:
                if val:
                    indexed_extras.append((int(key[len("coordinator_"):]), val))
    for _, val in sorted(indexed_extras):
        if val not in extras and val != primary:
            extras.append(val)
    return {
        "coordinator": primary,
        "coordinators": [primary] + extras,
        "token": token,
    }


def pick_reachable_coordinator(
    candidates: list[str],
    *,
    timeout_seconds: float = 1.5,
) -> str | None:
    """Return the first candidate whose ``/healthz`` answers 200, or None.

    This is a connectivity probe, not a security check — we just hit the
    well-known health endpoint and time out aggressively so the bootstrap
    doesn't spend forever on the wrong URL when the right one is 30 ms
    further down the list. Used by ``main()`` when ``coordinators`` has
    more than one entry: the join URL from a multi-IP offer should try
    them in priority order.

    Stdlib-only; relies on the bootstrap's earlier ``install_proxy_bypass``
    call so all probes go direct, not through any system HTTP_PROXY.
    """
    import urllib.request as _ur
    import urllib.error as _ue
    for url in candidates:
        try:
            req = _ur.Request(url.rstrip("/") + "/healthz", method="GET")
            with _ur.urlopen(req, timeout=timeout_seconds) as resp:
                # Any 2xx counts as reachable. We don't parse the body —
                # we only care that the TCP + HTTP round-trip succeeded,
                # which is enough to know the URL is routable.
                if 200 <= getattr(resp, "status", 200) < 300:
                    return url
        except (OSError, _ue.URLError, _ue.HTTPError, TimeoutError):
            continue
    return None


# Install a global proxy-bypassing opener. The bootstrap is meant to talk
# directly to the LAN coordinator (port 8787) — NOT through any HTTP proxy
# the user happens to have set in their environment (mitmproxy, corporate
# proxies, etc.). urllib's default opener respects HTTP_PROXY even for LAN
# addresses because proxy_bypass() only does hostname matching, not CIDR.
# Without this, requests are routed through the proxy, which then rewrites
# the request line to absolute-URI form and the bootstrap sees
# "Remote end closed connection without response" on every heartbeat.
_proxy_bypass_opener = build_opener(ProxyHandler({}))
install_opener(_proxy_bypass_opener)


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

_LAN_RE = re.compile(r"/(?:ip4/(?:192\.168\.|10\.|172\.(?:1[6-9]|2\d|3[01])\.)[^/]+|ip6/(?:fc|fd)[^/]+)")


def _sort_multiaddrs_lan_first(addrs: list[str]) -> list[str]:
    """Sort multiaddrs so LAN addresses come first, Tailscale/overlay second,
    loopback last. This ensures followers on the physical LAN get a reachable
    bootstrap peer instead of a Tailscale-only address."""
    def _key(addr: str) -> tuple[int, str]:
        if _LAN_RE.search(addr):  # RFC 1918 / RFC 4193 (LAN / Unique Local)
            return (0, addr)
        if "/ip4/100." in addr:
            return (1, addr)  # Tailscale / CGNAT overlay
        if "127.0.0.1" in addr or "::1" in addr:
            return (3, addr)  # loopback
        return (2, addr)

    return sorted(addrs, key=_key)


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


def _shards_needed_for_layers(
    model_id: str,
    start_layer: int,
    end_layer: int,
    is_first_peer: bool = False,
    is_last_peer: bool = False,
    cache_dir: "str | Path | None" = None,
    timeout: int = 10,
) -> list[str]:
    """Return the list of safetensors shard filenames needed to cover
    ``[start_layer, end_layer)`` of ``model_id``.

    Uses the model's ``model.safetensors.index.json`` to map each layer to
    its shard. First/last peers also get the non-layer shards (embeddings,
    lm_head, layer norms) needed at pipeline boundaries. Falls back to an
    empty list if the index is unavailable.
    """
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = Path(cache_dir)
    model_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
    if not model_dir.is_dir():
        return []
    # The index is small — just load it locally rather than hitting HF
    for index_file in model_dir.rglob("model.safetensors.index.json"):
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight_map = data.get("weight_map", {})
        needed: set[str] = set()
        for key, filename in weight_map.items():
            if "model.layers." in key:
                try:
                    layer_num = int(key.split("model.layers.")[1].split(".")[0])
                    if start_layer <= layer_num < end_layer:
                        needed.add(filename)
                except (ValueError, IndexError):
                    pass
            elif is_first_peer or is_last_peer:
                # Embeddings, lm_head, final layer_norm live outside the
                # transformer block range; first peer needs embeddings,
                # last peer needs lm_head.
                if is_first_peer and ("embed" in key.lower() or "wte" in key.lower()):
                    needed.add(filename)
                elif is_last_peer and ("lm_head" in key.lower() or "embed_out" in key.lower()):
                    needed.add(filename)
        return sorted(needed)
    return []


def _hf_hub_cache_dir(cache_dir: "str | Path | None" = None) -> Path:
    """Resolve the HF hub cache root (respects HF_HOME)."""
    if cache_dir is not None:
        return Path(cache_dir)
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"


def _model_index_cached(model_id: str, cache_dir: "str | Path | None" = None) -> bool:
    """True iff model.safetensors.index.json for ``model_id`` is in the HF cache.

    Without the index we cannot map layers to shards, so a cold-cache peer
    would fall back to downloading the ENTIRE model instead of just its
    assigned layers. The fix is to fetch the (tiny) metadata files first,
    then compute the per-layer shard list.
    """
    model_dir = _hf_hub_cache_dir(cache_dir) / ("models--" + model_id.replace("/", "--"))
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("model.safetensors.index.json"))


def _run_hf_download(
    download_args: list[str],
    *,
    env: dict[str, str],
    idle_timeout_s: int = 300,
    total_timeout_s: int = 1800,
    on_tick=None,
) -> int:
    """Run ``hf download <args>`` with idle-kill + wall-clock budget.

    - Kills the process when stdout is silent for ``idle_timeout_s`` (the
      xet/HTTP mid-stream stall signature) or after ``total_timeout_s``.
    - Falls back to ``huggingface-cli download`` when the newer ``hf``
      entry point is not installed.
    - Calls ``on_tick(elapsed_seconds)`` roughly every 10s so the caller can
      keep the coordinator heartbeat/status fresh during multi-GB downloads.

    Returns the process exit code; raises TimeoutError on stall/budget kill.

    The stdout is drained by a background thread into a queue. This matters:
    a blocking ``readline()`` in the main loop never returns while a stalled
    connection produces zero bytes, so the idle/total timeout checks after it
    would be unreachable in exactly the situation they exist for.
    """
    import queue as _queue
    import subprocess as _sp
    import threading as _threading

    start = time.time()
    try:
        proc = _sp.Popen(
            ["hf", "download"] + download_args,
            stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
            env=env, bufsize=1,
        )
    except FileNotFoundError:
        # Older huggingface_hub installs ship `huggingface-cli`, not `hf`.
        proc = _sp.Popen(
            ["huggingface-cli", "download"] + download_args,
            stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
            env=env, bufsize=1,
        )
    assert proc.stdout is not None
    lines: "_queue.Queue[str | None]" = _queue.Queue()

    def _pump() -> None:
        try:
            for raw in proc.stdout:  # universal newlines: \r progress → lines
                lines.put(raw)
        except (OSError, ValueError):
            pass
        lines.put(None)  # EOF sentinel

    _threading.Thread(target=_pump, daemon=True).start()

    def _kill_and_raise(reason: str) -> None:
        proc.kill()
        try:
            proc.wait(timeout=10)  # reap — the retry loop must not stack zombies
        except Exception:
            pass
        raise TimeoutError(reason)

    last_io = time.time()
    last_tick = time.time()
    eof = False
    while not eof:
        try:
            item = lines.get(timeout=1.0)
        except _queue.Empty:
            item = ""
        now = time.time()
        if on_tick is not None and now - last_tick >= 10.0:
            last_tick = now
            try:
                on_tick(now - start)
            except Exception:
                pass
        if item is None:
            eof = True
            continue
        # The wall-clock budget applies even while output is flowing — a
        # download crawling along at dial-up pace keeps printing progress
        # lines and must still be bounded.
        if now - start > total_timeout_s:
            _kill_and_raise(f"download exceeded {total_timeout_s}s budget")
        if item:
            last_io = now
            text = item.rstrip()
            if text:
                print("   │ " + text, file=sys.stderr)
            continue
        if now - last_io > idle_timeout_s:
            _kill_and_raise(
                f"no progress for {idle_timeout_s}s — likely xet/HTTP stall"
            )
    return proc.wait()


def model_weights_cached(
    model_id: str,
    cache_dir: "str | Path | None" = None,
    required_shards: "list[str] | None" = None,
) -> bool:
    """Return True iff the HF cache holds real weight files for ``model_id``.

    A config-only cache entry (just ``config.json`` / tokenizer / a bare
    ``*.index.json``) is NOT enough — the server would hang for a long time
    downloading the actual shards. Preflighting this lets the bootstrap fail
    fast with an actionable message instead of appearing to "serve" forever.

    If ``required_shards`` is provided, only those specific shard files are
    checked (matches the peer's assigned layer range). Otherwise any
    safetensors/.bin file in the cache counts.
    """
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = Path(cache_dir)

    model_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
    if not model_dir.is_dir():
        return False

    if required_shards:
        # Per-shard preflight: check each specific shard exists and is non-empty
        for shard_name in required_shards:
            # Shards can be in snapshots/<hash>/<filename> or in blobs/
            found = False
            for path in model_dir.rglob(shard_name):
                try:
                    if path.stat().st_size > 0:
                        found = True
                        break
                except OSError:
                    continue
            if not found:
                return False
        return True

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
    block_range: str | None = None,
    num_layers: int | None = None,
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
    # When the job specifies a block range, we check only the shards that
    # cover those layers — not the full model. This is the per-shard
    # download flow: each peer pre-fetches only its assigned layers.
    preflight_shards: list[str] | None = None
    preflight_block_range: str | None = block_range
    if model_id and not preflight_block_range:
        # Fallback for older coordinators that don't send block_indices in
        # the job payload: derive the range from the command line itself.
        for token in clean_command.split():
            if ":" in token and token.replace(":", "").isdigit():
                preflight_block_range = token
                break
    start: int | None = None
    end: int | None = None
    if model_id and preflight_block_range:
        try:
            start_str, end_str = preflight_block_range.split(":", 1)
            start, end = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            start = end = None
    # Prepare the download environment once — the metadata prefetch and the
    # shard download both need the venv PATH and the xet kill-switch.
    dl_env = dict(_os.environ)
    if _os.path.isfile(venv_python):
        dl_env["PATH"] = f"{_os.path.dirname(venv_python)}:{dl_env.get('PATH', '')}"
    # Disable hf_xet: the xet endpoint regularly stalls on multi-GB shards
    # (the Evis-2026-07-08 incident: 161 MB / 4 GB then frozen for 17 min).
    # The plain HTTP path is slower per MB but never wedges and
    # `.incomplete` files survive restarts either way.
    dl_env.setdefault("HF_HUB_DISABLE_XET", "1")
    if model_id and start is not None and end is not None:
        # On a cold cache the safetensors index isn't local yet, so the
        # layer→shard mapping is empty and the peer would fall back to a
        # full-model download. Fetch the small metadata files (config,
        # tokenizer, index — all *.json) first, then compute the shard list.
        if not _model_index_cached(model_id):
            print(f"   📄 Fetching model metadata (*.json) for {model_id}...", file=sys.stderr)
            if have_status:
                post_peer_status(
                    coord, pid,
                    status="downloading", progress=2.0,
                    job_port=job_port, model_id=model_id,
                    message="fetching model metadata (config/tokenizer/index)",
                )
            try:
                _run_hf_download(
                    [model_id, "--include", "*.json"],
                    env=dl_env, idle_timeout_s=120, total_timeout_s=300,
                )
            except Exception as exc:
                # Non-fatal: single-file models have no index, and the shard
                # download / server launch will surface real network errors.
                print(f"   ⚠️  metadata prefetch failed: {exc}", file=sys.stderr)
        preflight_shards = _shards_needed_for_layers(
            model_id, start, end,
            is_first_peer=(start == 0),
            is_last_peer=(num_layers is not None and end == int(num_layers)),
        )
    did_download = False
    if model_id and not model_weights_cached(
        model_id, required_shards=preflight_shards
    ):
        # ── Auto-download missing shards (zero-touch) ──
        # Instead of exiting with an error, download the required
        # shards automatically and then proceed to launch the server.
        shard_list = " ".join(preflight_shards) if preflight_shards else ""
        download_args = [model_id] + (preflight_shards or [])
        download_cmd = f"hf download {' '.join(download_args)}"

        # Clear stale HF lock/incomplete state from a prior killed downloader.
        # HuggingFace Hub uses file locks to coordinate concurrent writers;
        # when a downloader is killed (timeout, xet stall, OOM), the lock
        # file lingers and blocks subsequent resumes. Removing it is safe —
        # the .incomplete file is the real "what's been downloaded so far"
        # state and `hf download` will resume from it.
        hf_home = _os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        hub_root = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
        try:
            locks_dir = hub_root / ".locks"
            if locks_dir.is_dir():
                for lock in locks_dir.iterdir():
                    try:
                        lock.unlink()
                    except OSError:
                        pass
                print(f"   🧹 cleared stale HF locks in {locks_dir}", file=sys.stderr)
            for inc in hub_root.rglob("*.incomplete"):
                # Leave .incomplete files alone — those ARE the resume state.
                # We only printed them so the operator can see what's partial.
                sz = inc.stat().st_size if inc.exists() else 0
                print(f"   📌 partial shard resume: {inc.name} ({sz/1e6:.1f} MB)", file=sys.stderr)
        except OSError as exc:
            # Don't fail the bootstrap over a permission glitch — the
            # download attempt itself will surface any real issue.
            print(f"   ⚠️  lock-scan skipped: {exc}", file=sys.stderr)

        if have_status:
            post_peer_status(
                coord, pid,
                status="downloading", progress=5.0,
                job_port=job_port, model_id=model_id,
                message=f"Auto-downloading {len(preflight_shards or [])} shard(s) for layers {preflight_block_range or 'all'}...",
            )

        print(f"   📥 Downloading weights: {download_cmd}", file=sys.stderr)
        dl_start = time.time()

        def _download_tick(elapsed: float) -> None:
            # Keeps the coordinator's roster fresh during multi-GB downloads:
            # /peer-status posts also refresh the heartbeat timestamp, so the
            # peer doesn't fall off the dashboard while its main loop is
            # blocked inside this download.
            if have_status:
                post_peer_status(
                    coord, pid,
                    status="downloading",
                    progress=min(18.0, 5.0 + elapsed / 60.0),
                    job_port=job_port, model_id=model_id,
                    message=f"downloading weights... {elapsed:.0f}s elapsed",
                )

        try:
            # Idle-kill after 5 silent minutes (stall signature); wall-clock
            # budget scales with how much we actually have to fetch — a
            # full-model fallback gets a bigger budget than two shards.
            IDLE_TIMEOUT_S = 300
            if preflight_shards:
                TOTAL_TIMEOUT_S = 900 + 900 * len(preflight_shards)
            else:
                TOTAL_TIMEOUT_S = 3600
            download_rc = _run_hf_download(
                download_args,
                env=dl_env,
                idle_timeout_s=IDLE_TIMEOUT_S,
                total_timeout_s=TOTAL_TIMEOUT_S,
                on_tick=_download_tick,
            )
            dl_duration = time.time() - dl_start
        except Exception as exc:
            dl_duration = time.time() - dl_start
            msg = f"Download failed after {dl_duration:.0f}s: {exc}"
            print(f"   ❌ {msg}", file=sys.stderr)
            if have_status:
                post_peer_status(
                    coord, pid,
                    status="error", progress=None,
                    job_port=job_port, model_id=model_id,
                    message=msg,
                )
            return {
                "command": command, "exit_code": 2,
                "stdout_tail": "", "stderr_tail": msg,
                "weights_missing": True,
                "required_shards": preflight_shards or [],
            }

        if download_rc != 0:
            msg = (
                f"weight download exited with code {download_rc} "
                f"after {dl_duration:.0f}s — see download log lines above"
            )
            print(f"   ❌ {msg}", file=sys.stderr)
            if have_status:
                post_peer_status(
                    coord, pid,
                    status="error", progress=None,
                    job_port=job_port, model_id=model_id,
                    message=msg,
                )
            return {
                "command": command, "exit_code": 2,
                "stdout_tail": "", "stderr_tail": msg,
                "weights_missing": True,
                "required_shards": preflight_shards or [],
            }

        print(f"   ✅ Weights downloaded in {dl_duration:.0f}s", file=sys.stderr)
        if have_status:
            post_peer_status(
                coord, pid,
                status="loading", progress=20.0,
                job_port=job_port, model_id=model_id,
                message=f"Downloaded {len(preflight_shards or [])} shards in {dl_duration:.0f}s; launching server...",
            )

        # Verify the shards are now cached — if still missing, something is wrong
        if not model_weights_cached(model_id, required_shards=preflight_shards):
            msg = (
                f"Weights downloaded but still not found in HF cache. "
                f"Check HF_HOME / disk space."
            )
            print(f"   ❌ {msg}", file=sys.stderr)
            if have_status:
                post_peer_status(
                    coord, pid,
                    status="error", progress=None,
                    job_port=job_port, model_id=model_id,
                    message=msg,
                )
            return {
                "command": command, "exit_code": 2,
                "stdout_tail": "", "stderr_tail": msg,
                "weights_missing": True,
                "required_shards": preflight_shards or [],
            }
        did_download = True

    if have_status and not did_download:
        # Weights were already cached — the run starts at the download
        # stage from the dashboard's point of view. When the preflight just
        # downloaded them, skip this: resetting to "downloading 0%" right
        # after posting "loading 20%" walks the progress bar backwards.
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
        # If the preflight just fetched the shards there is nothing left to
        # download — the synthetic progress starts at the loading stage.
        stage = "loading" if did_download else "downloading"
        posted_loading = did_download
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
                # The server self-downloads any shard the preflight missed
                # (bloombee.server.from_pretrained → huggingface_hub). That
                # in-process download has no idle-killer, so make sure it
                # never takes the stall-prone xet path.
                "HF_HUB_DISABLE_XET": _os.environ.get("HF_HUB_DISABLE_XET", "1"),
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
                    multiaddrs=_sort_multiaddrs_lan_first(list(posted_multiaddrs)),
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
            # Launcher exited cleanly but hivemind never printed its "Started"
            # marker — the server is not actually serving. Reporting "serving"
            # would lie to the dashboard (this was the Evis false-positive bug).
            post_peer_status(
                coord, pid,  # type: ignore[arg-type]
                status="error", progress=None,
                job_port=job_port, model_id=model_id,
                message=(
                    "server exited without 'Started' marker — "
                    "check stdout above; HF cache may be partial. "
                    "Try: HF_HUB_DISABLE_XET=1 to bypass xet stalls."
                ),
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
        "detected_serving": detected_serving,
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

    # Step 2: Parse join URL. The join URL may carry multiple candidate
    # coordinator URLs (coordinator, coordinator_2, coordinator_3, ...) —
    # the offer endpoint emits these for /hotspot-wifi and /stale-QR
    # robustness. We probe each candidate and use the first reachable one
    # as the actual coordinator.
    join = parse_join_url(args.join_url)
    candidates = join.get("coordinators") or [join["coordinator"]]
    token = join["token"]

    if len(candidates) > 1 and not args.json:
        print(f"\n🔗 Joining swarm ({len(candidates)} candidate URL{'s' if len(candidates) != 1 else ''}):", file=sys.stderr)
        for i, url in enumerate(candidates, 1):
            print(f"     [{i}] {url}", file=sys.stderr)
    elif not args.json:
        print(f"\n🔗 Joining swarm at {candidates[0]}...", file=sys.stderr)

    # Probe for reachability. Fast (1.5 s timeout each), so a peer with
    # three candidates spends at most ~5 s here before falling back or
    # giving up. If only one candidate is present this is a single
    # round-trip to the same host.
    reachable = pick_reachable_coordinator(candidates) or candidates[0]
    if reachable != candidates[0]:
        if not args.json:
            print(f"   ↪ primary {candidates[0]} unreachable, using {reachable}", file=sys.stderr)
    coordinator = reachable

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
        # Retry accounting per deployment: keyed by the deployment's
        # created_at timestamp so a re-deploy from the dashboard resets the
        # budget automatically (the old `job_executed` boolean permanently
        # disabled auto-serve after ONE attempt — a single transient download
        # failure meant a human had to restart the bootstrap, and a re-deploy
        # was never picked up at all).
        MAX_JOB_ATTEMPTS = 3
        job_attempts: dict[str, int] = {}
        try:
            while True:
                time.sleep(args.interval)
                capabilities["scanned_at"] = int(time.time())
                response = send_heartbeat(coordinator, token, capabilities)
                if not args.json:
                    status = "✅" if response.get("ok") else "❌"
                    print(f"   {status} Heartbeat at {time.strftime('%H:%M:%S')}", file=sys.stderr)

                # Auto-serve: poll for a deployment job and execute it
                if args.auto_serve:
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
                        deploy_key = str(poll.get("deployed_at") or poll.get("deployed_model") or "default")
                        attempts = job_attempts.get(deploy_key, 0)
                        if attempts >= MAX_JOB_ATTEMPTS:
                            # Budget exhausted for THIS deployment; a fresh
                            # Deploy click (new created_at) resets it.
                            continue
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
                        print(
                            f"   🎯 Assigned: {job.get('role', '?')} "
                            f"{job.get('block_indices') or job.get('block_range', '')} "
                            f"port {job.get('port', '?')} (attempt {attempts + 1}/{MAX_JOB_ATTEMPTS})",
                            file=sys.stderr,
                        )
                        job_attempts[deploy_key] = attempts + 1
                        result = execute_job_command(
                            cmd,
                            coordinator=coordinator,
                            peer_id=capabilities["peer_id"],
                            hostname=capabilities["hostname"],
                            job_port=job.get("port"),
                            model_id=poll.get("deployed_model"),
                            block_range=job.get("block_indices") or job.get("block_range"),
                            num_layers=job.get("num_layers"),
                        )
                        print(f"   🏁 Server exited with code {result.get('exit_code')}", file=sys.stderr)
                        if job_attempts.get(deploy_key, 0) >= MAX_JOB_ATTEMPTS:
                            print(
                                f"   🛑 {MAX_JOB_ATTEMPTS} attempts used for this deployment; "
                                "waiting for a new Deploy before retrying.",
                                file=sys.stderr,
                            )
        except KeyboardInterrupt:
            if not args.json:
                print("\n👋 Disconnected.", file=sys.stderr)

    if args.json:
        print(json.dumps({"ok": True, "peer_id": capabilities["peer_id"], "coordinator": coordinator}))


if __name__ == "__main__":
    main()
