#!/usr/bin/env python3
"""
peer_scan.py — hardware capability probe for a BloomBee peer node.

Emits a single JSON document describing the host's compute, memory,
accelerator, network reachability and disk state. Intended to be run on
every node in the swarm so a scheduler/UI can route requests to a peer
whose model fits comfortably.

No BloomBee import — pure stdlib + psutil + torch. Safe to run from
cron, a CLI dashboard, or the bootstrap step of a peer.

Output:
    stdout (JSON), and a copy at ~/.bloombee/capabilities/<hostname>.json
    (mkdir -p as needed; the dir lives under the user's home).

Exit code: always 0 — this script is informational. Callers should parse
the JSON and decide; nothing here is fatal.

Usage:
    python peer_scan.py
    python peer_scan.py --peers m4-pro,m4-laptop,node3.tail.ts.net
    python peer_scan.py --out /tmp/peer.json --peers host1,host2
"""

from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def detect_hostname() -> str:
    return socket.gethostname().split(".")[0]


def detect_tailscale_ip() -> str | None:
    # Tailscale ships `tailscale ip -4` on macOS/Linux. Missing binary is
    # fine — node just has no mesh IP yet.
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            ip = out.stdout.strip().splitlines()[0].strip()
            return ip or None
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass
    return None


def detect_cpu() -> dict[str, Any]:
    info: dict[str, Any] = {"model": None, "logical": None, "physical": None}
    info["model"] = _platform.processor() or None

    try:
        import psutil  # type: ignore

        info["logical"] = psutil.cpu_count(logical=True)
        info["physical"] = psutil.cpu_count(logical=False)
        return info
    except ImportError:
        pass

    info["logical"] = os.cpu_count()
    # /proc/cpuinfo is Linux-only and gives "cpu cores" per physical package
    if sys.platform.startswith("linux"):
        try:
            cores = set()
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("cpu cores"):
                        cores.add(_safe_int(line.split(":", 1)[1]))
            if cores:
                info["physical"] = sum(cores)
        except OSError:
            pass
    return info


def detect_memory() -> dict[str, Any]:
    total_gb: float | None = None
    free_gb: float | None = None

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024**3), 2)
        free_gb = round(vm.available / (1024**3), 2)
    except ImportError:
        if sys.platform == "darwin":
            try:
                out = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                total_gb = round(int(out.stdout.strip()) / (1024**3), 2)
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
        elif sys.platform.startswith("linux"):
            try:
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        k, _, v = line.partition(":")
                        meminfo[k.strip()] = v.strip()
                if "MemTotal" in meminfo:
                    total_gb = round(int(meminfo["MemTotal"].split()[0]) / (1024**2), 2)
                if "MemAvailable" in meminfo:
                    free_gb = round(int(meminfo["MemAvailable"].split()[0]) / (1024**2), 2)
            except (OSError, ValueError, IndexError):
                pass

    return {"total_gb": total_gb, "free_gb": free_gb}


def detect_accelerator() -> dict[str, Any]:
    """MPS / CUDA / CPU-only. Apple Silicon uses unified memory — we
    report the same physical RAM as both ram_total_gb AND vram_total_gb
    and flag unified_memory=True so callers don't double-count."""
    info: dict[str, Any] = {
        "device": "cpu",
        "unified_memory": False,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "gpus": [],
    }

    try:
        import torch  # type: ignore
    except ImportError:
        return info

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["unified_memory"] = True
        # MPS shares system RAM; mirror what detect_memory found.
        mem = detect_memory()
        info["vram_total_gb"] = mem["total_gb"]
        info["vram_free_gb"] = mem["free_gb"]
        return info

    # CUDA path — try pynvml first (cleaner), then nvidia-smi, then give up.
    if torch.cuda.is_available():
        info["device"] = "cuda"
        gpus: list[dict[str, Any]] = []

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                gpus.append(
                    {
                        "index": i,
                        "name": name,
                        "vram_total_gb": round(mem.total / (1024**3), 2),
                        "vram_free_gb": round(mem.free / (1024**3), 2),
                    }
                )
            pynvml.nvmlShutdown()
        except (ImportError, Exception):  # noqa: BLE001 — pynvml raises RuntimeError when no driver
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    for line in out.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            gpus.append(
                                {
                                    "index": _safe_int(parts[0]),
                                    "name": parts[1],
                                    "vram_total_gb": round(int(parts[2]) / 1024, 2),
                                    "vram_free_gb": round(int(parts[3]) / 1024, 2),
                                }
                            )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        info["gpus"] = gpus
        if gpus:
            info["vram_total_gb"] = sum(g["vram_total_gb"] for g in gpus if g["vram_total_gb"])
            info["vram_free_gb"] = sum(g["vram_free_gb"] for g in gpus if g["vram_free_gb"])
        return info

    return info


def ping_peers(peers: list[str], timeout_s: float = 1.5) -> dict[str, dict[str, Any]]:
    """Send 3 pings per host, report median RTT in ms and packet loss.
    Missing `ping` binary or unreachable host → reported with null rtt."""
    results: dict[str, dict[str, Any]] = {}
    if not peers:
        return results

    count_flag = "-c"
    timeout_flag = "-W" if not sys.platform.startswith("darwin") else "-W"  # busybox on linux vs mac
    # macOS ping uses `-W` for milliseconds too, but `-c` for count — same shape, good.

    for host in peers:
        host = host.strip()
        if not host:
            continue
        cmd = ["ping", count_flag, "3", timeout_flag, str(int(timeout_s * 1000)), host]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s * 4 + 2)
            text = out.stdout or ""
            # Extract "time=12.3 ms" tokens, average them
            rtts: list[float] = []
            for tok in text.replace("\n", " ").split():
                if tok.startswith("time=") or tok.startswith("time<"):
                    num = tok.split("=", 1)[1].rstrip("ms").strip()
                    try:
                        rtts.append(float(num))
                    except ValueError:
                        pass
            results[host] = {
                "reachable": bool(rtts) and out.returncode == 0,
                "rtt_ms_median": round(sorted(rtts)[len(rtts) // 2], 2) if rtts else None,
                "rtt_ms_avg": round(sum(rtts) / len(rtts), 2) if rtts else None,
            }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            results[host] = {"reachable": False, "rtt_ms_median": None, "rtt_ms_avg": None}
    return results


def disk_free(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        # shutil.disk_usage requires the path to exist; if it doesn't, try parent
        target = p if p.exists() else p.parent
        usage = shutil.disk_usage(target)
        return {
            "path": str(p),
            "free_gb": round(usage.free / (1024**3), 2),
            "total_gb": round(usage.total / (1024**3), 2),
        }
    except OSError as e:
        return {"path": str(p), "error": str(e)}


def torch_version() -> str | None:
    try:
        import torch  # type: ignore

        return torch.__version__
    except ImportError:
        return None


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def scan(peers: list[str]) -> dict[str, Any]:
    mem = detect_memory()
    acc = detect_accelerator()
    return {
        "schema_version": 1,
        "scanned_at": time.time(),
        "hostname": detect_hostname(),
        "tailscale_ip": detect_tailscale_ip(),
        "platform": "darwin" if sys.platform == "darwin" else sys.platform.split("-")[0],
        "python": python_version(),
        "torch": torch_version(),
        "cpu": detect_cpu(),
        "memory": mem,
        "accelerator": acc,
        # Unified-memory note: if MPS, ram == vram. Keep both for clarity,
        # consumers should treat vram_free_gb as the binding constraint.
        "vram_total_gb": acc.get("vram_total_gb"),
        "vram_free_gb": acc.get("vram_free_gb"),
        "peers": ping_peers(peers),
        "huggingface_cache": disk_free("~/.cache/huggingface"),
    }


def write_out(doc: dict[str, Any], out_path: Path | None) -> None:
    if out_path is None:
        out_dir = Path.home() / ".bloombee" / "capabilities"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc['hostname']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local hardware capabilities for BloomBee peers.")
    parser.add_argument("--peers", default="", help="comma-separated hostnames to ping for RTT")
    parser.add_argument("--out", default=None, help="explicit output path (default ~/.bloombee/capabilities/<host>.json)")
    args = parser.parse_args()

    peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    doc = scan(peers)

    out_path = Path(args.out).expanduser() if args.out else None
    try:
        write_out(doc, out_path)
    except OSError as e:
        # Write failure shouldn't kill the JSON-on-stdout path.
        doc["_write_error"] = str(e)

    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())