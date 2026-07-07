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


def _run_text(cmd: list[str], timeout_s: float = 3.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text or None


def _android_getprop(name: str) -> str | None:
    value = _run_text(["getprop", name], timeout_s=1.5)
    if value:
        return value
    # Some Termux setups do not expose getprop on PATH but keep Android's
    # system binary accessible.
    return _run_text(["/system/bin/getprop", name], timeout_s=1.5)


def detect_mobile_profile() -> dict[str, Any]:
    """Detect phone/tablet-specific traits without requiring Android APIs.

    The scanner still reports CPU/CUDA/MPS separately. This mobile profile is a
    routing hint: phones are potential peers only after they produce measured
    throughput evidence, and current BloomBee block servers still need explicit
    CPU/mobile validation before being counted as useful inference workers.
    """
    prefix = os.environ.get("PREFIX", "")
    android_root = os.environ.get("ANDROID_ROOT", "")
    termux = "com.termux" in prefix or bool(os.environ.get("TERMUX_VERSION"))
    android = termux or bool(android_root) or sys.platform.startswith("android")

    # A cheap probe catches Android shells that do not set Termux-style env.
    model = _android_getprop("ro.product.model") if sys.platform.startswith("linux") or android else None
    if model:
        android = True

    if not android:
        return {"is_mobile": False, "kind": None, "runtime": None}

    manufacturer = _android_getprop("ro.product.manufacturer")
    soc = (
        _android_getprop("ro.soc.model")
        or _android_getprop("ro.board.platform")
        or _android_getprop("ro.hardware")
    )
    return {
        "is_mobile": True,
        "kind": "android",
        "runtime": "termux" if termux else "android-linux",
        "model": model,
        "manufacturer": manufacturer,
        "soc": soc,
        "cpu_abi": _android_getprop("ro.product.cpu.abi"),
        "sdk": _android_getprop("ro.build.version.sdk"),
    }


def _parse_meminfo_total_kb(value: str) -> int | None:
    """Parse a /proc/meminfo line value.

    Handles all four observed formats across Android/Termux/Linux:
      - "MemTotal:       11899328 kB"
      - "MemTotal:        11899328"
      - "MemTotal:         12345678 KB"
      - "MemTotal:    11899328 bytes"   (rare but seen)
    Returns kB as int, or None on parse failure.
    """
    if not value:
        return None
    parts = value.split()
    if not parts:
        return None
    try:
        n = int(parts[0])
    except ValueError:
        return None
    if len(parts) >= 2:
        unit = parts[1].lower()
        if unit in ("bytes", "b"):
            return n // 1024
        if unit in ("mb", "m"):
            return n * 1024
        # "kb" / "k" / "kib" → already kB, no conversion
    return n


def detect_cpu() -> dict[str, Any]:
    info: dict[str, Any] = {"model": None, "logical": None, "physical": None}

    # 1. platform.processor() works on macOS/Linux glibc but returns "" on
    #    Termux/Android. Try it first as a fast path.
    info["model"] = _platform.processor() or None

    # 2. Android/Termux: /proc/cpuinfo uses "Hardware" (Tensor G3) and
    #    "model name" rather than the glibc "cpu cores" / "model name"
    #    combo the original code expected. Parse Android-style first.
    if sys.platform.startswith("linux") and not info["model"]:
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            for line in cpuinfo.splitlines():
                low = line.lower()
                if low.startswith("hardware") and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        info["model"] = val
                elif low.startswith("model name") and ":" in line and not info["model"]:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        info["model"] = val
                elif low.startswith("processor") and ":" in line:
                    # "Processor : ARMv8 Processor rev 0 (v8l)" — generic
                    val = line.split(":", 1)[1].strip()
                    if val and not info["model"]:
                        info["model"] = val
                elif low.startswith("cpu part") and ":" in line and not info["model"]:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        info["model"] = f"ARM CPU part {val}"
                elif low.startswith("cpu implementer") and ":" in line and not info["model"]:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        info["model"] = f"ARM implementer 0x{val}"
        except OSError:
            pass

    # 3. On Android, getprop gives the *commercial* model (e.g. "Pixel 8 Pro")
    #    which is much more useful than raw SoC part numbers. Prefer it.
    if not info["model"] or info["model"] in ("unknown", "", "ARMv8"):
        product = _android_getprop("ro.product.model")
        if product:
            info["model"] = product
        if not info["model"]:
            soc = (
                _android_getprop("ro.soc.model")
                or _android_getprop("ro.board.platform")
                or _android_getprop("ro.hardware")
            )
            if soc:
                info["model"] = soc

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
            cores: set[int] = set()
            with open("/proc/cpuinfo") as f:
                for line in f:
                    # Both "cpu cores" (glibc) and "CPU part" (Android) — only
                    # the cores line should populate physical count.
                    low = line.lower()
                    if low.startswith("cpu cores") and ":" in line:
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
        total_gb = round(vm.total / (1024**3), 2) if vm.total > 0 else None
        free_gb = round(vm.available / (1024**3), 2) if vm.available > 0 else None
    except ImportError:
        pass

    # Platform-specific fallbacks when psutil returns 0 or is unavailable
    if total_gb is None or total_gb == 0:
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
                # Use the unit-tolerant parser: handles "11899328 kB", "11899328 KB",
                # "11899328" (unitless), and rare "bytes" / "MB" variants seen on
                # some Android kernels and Termux builds.
                if "MemTotal" in meminfo:
                    kb = _parse_meminfo_total_kb(meminfo["MemTotal"])
                    if kb is not None:
                        total_gb = round(kb / (1024**2), 2)
                if "MemAvailable" in meminfo:
                    kb = _parse_meminfo_total_kb(meminfo["MemAvailable"])
                    if kb is not None:
                        free_gb = round(kb / (1024**2), 2)
                elif "MemFree" in meminfo:
                    # Some Android kernels omit MemAvailable entirely. MemFree alone
                    # understates available RAM (excludes reclaimable cache), but is
                    # still better than 0.0.
                    kb = _parse_meminfo_total_kb(meminfo["MemFree"])
                    if kb is not None:
                        free_gb = round(kb / (1024**2), 2)
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
    mobile = detect_mobile_profile()
    return {
        "schema_version": 1,
        "scanned_at": time.time(),
        "hostname": detect_hostname(),
        "tailscale_ip": detect_tailscale_ip(),
        "platform": "darwin" if sys.platform == "darwin" else sys.platform.split("-")[0],
        "mobile": mobile,
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