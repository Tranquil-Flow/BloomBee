#!/usr/bin/env python3
"""Physical-device join client for the distributed-inference MVP.

Parses a `bloombee://join?...` offer, loads a peer capability JSON, and posts a
heartbeat to the coordinator. This is bootstrap/roster state only; it does not
start BloomBee servers or claim inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


JOIN_COORDINATOR_FALLBACK_PREFIX = "coordinator_"

REQUEST_CLAIM_BOUNDARY = "join_client_request_only_no_inference_proof"
DRY_RUN_CLAIM_BOUNDARY = "join_client_dry_run_only_no_inference_proof"
POST_CLAIM_BOUNDARY = "join_client_post_only_no_inference_proof"
LOOP_CLAIM_BOUNDARY = "join_client_heartbeat_loop_only_no_inference_proof"
JOB_POLL_CLAIM_BOUNDARY = "join_client_job_poll_only_no_inference_proof"


def parse_join_url(join_url: str) -> dict[str, Any]:
    parsed = urlparse(join_url)
    query = parse_qs(parsed.query)
    coordinator = (query.get("coordinator") or [None])[0]
    token = (query.get("token") or [None])[0]
    if parsed.scheme != "bloombee" or parsed.netloc != "join" or not coordinator or not token:
        raise ValueError("join URL must include coordinator and token")

    coordinators = [coordinator]
    indexed_candidates: list[tuple[int, str]] = []
    for key, values in query.items():
        if not key.startswith(JOIN_COORDINATOR_FALLBACK_PREFIX) or not values:
            continue
        suffix = key[len(JOIN_COORDINATOR_FALLBACK_PREFIX):]
        if not suffix.isdigit():
            continue
        indexed_candidates.append((int(suffix), values[0]))
    seen = {coordinator}
    for _, candidate in sorted(indexed_candidates):
        if candidate and candidate not in seen:
            coordinators.append(candidate)
            seen.add(candidate)
    return {"coordinator": coordinator, "coordinators": coordinators, "token": token}


def _load_capabilities(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def build_heartbeat_payload(
    join: dict[str, str],
    *,
    capabilities_path: str | Path,
    peer_id: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    capabilities = _load_capabilities(capabilities_path)
    resolved_peer_id = peer_id or capabilities.get("hostname") or capabilities.get("peer_id")
    if not resolved_peer_id:
        raise ValueError("peer_id is required when capabilities JSON has no hostname")
    payload: dict[str, Any] = {
        "token": join["token"],
        "peer_id": str(resolved_peer_id),
        "capabilities": capabilities,
        "claim_boundary": REQUEST_CLAIM_BOUNDARY,
    }
    if now is not None:
        payload["now"] = int(now)
    return payload


def _heartbeat_url(coordinator: str) -> str:
    return coordinator.rstrip("/") + "/heartbeat"


def build_heartbeat_request(join: dict[str, str], payload: dict[str, Any]) -> Request:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return Request(
        _heartbeat_url(join["coordinator"]),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def dry_run_report(join_url: str, *, capabilities_path: str | Path, peer_id: str | None = None, now: int | None = None) -> dict[str, Any]:
    join = parse_join_url(join_url)
    payload = build_heartbeat_payload(join, capabilities_path=capabilities_path, peer_id=peer_id, now=now)
    request = build_heartbeat_request(join, payload)
    return {
        "dry_run": True,
        "url": request.full_url,
        "method": request.get_method(),
        "headers": dict(request.headers),
        "body": payload,
        "claim_boundary": DRY_RUN_CLAIM_BOUNDARY,
    }


def post_heartbeat(
    join_url: str,
    *,
    capabilities_path: str | Path,
    peer_id: str | None = None,
    timeout: float = 5.0,
    now: int | None = None,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    join = parse_join_url(join_url)
    payload = build_heartbeat_payload(join, capabilities_path=capabilities_path, peer_id=peer_id, now=now)
    request = build_heartbeat_request(join, payload)
    with urlopen_fn(request, timeout=timeout) as response:
        server_response = json.loads(response.read().decode("utf-8"))
    return {
        "url": request.full_url,
        "peer_id": payload["peer_id"],
        "server_response": server_response,
        "claim_boundary": POST_CLAIM_BOUNDARY,
    }


def run_heartbeat_loop(
    join_url: str,
    *,
    capabilities_path: str | Path,
    peer_id: str | None = None,
    count: int = 1,
    interval_seconds: float = 10.0,
    timeout: float = 5.0,
    now: int | None = None,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], Any] = time.sleep,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Post repeated heartbeats so a joined laptop stays active in the roster.

    This is still coordinator/roster wiring only. It does not start BloomBee
    servers, run inference, or update proof status. ``count`` is intentionally
    finite so operators can choose a bounded live-demo window and tests can run
    deterministically.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be >= 0")

    results: list[dict[str, Any]] = []
    for index in range(count):
        heartbeat_now = int(now if now is not None else now_fn())
        result = post_heartbeat(
            join_url,
            capabilities_path=capabilities_path,
            peer_id=peer_id,
            timeout=timeout,
            now=heartbeat_now,
            urlopen_fn=urlopen_fn,
        )
        result["iteration"] = index + 1
        results.append(result)
        if index < count - 1:
            sleep_fn(interval_seconds)

    return {
        "heartbeat_count": len(results),
        "requested_count": count,
        "interval_seconds": interval_seconds,
        "results": results,
        "claim_boundary": LOOP_CLAIM_BOUNDARY,
        "inference_proven": False,
        "can_update_proof_status": False,
    }


def _job_url(coordinator: str) -> str:
    return coordinator.rstrip("/") + "/job"


def poll_job(
    join_url: str,
    *,
    peer_id: str,
    timeout: float = 5.0,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Poll GET /job?peer_id=X — return the job assignment or null."""
    join = parse_join_url(join_url)
    url = _job_url(join["coordinator"]) + f"?peer_id={peer_id}"
    with urlopen_fn(url, timeout=timeout) as response:
        job_response = json.loads(response.read().decode("utf-8"))
    return {
        "url": url,
        "peer_id": peer_id,
        "job_response": job_response,
        "claim_boundary": JOB_POLL_CLAIM_BOUNDARY,
    }


def _bootstrap_executor():
    """Return scripts.bootstrap.execute_job_command when importable, else None.

    The bootstrap executor is the maintained auto-serve path: weight-cache
    preflight, per-shard auto-download with stall killing, live /peer-status
    reporting, seed-multiaddr publishing, and readiness gating on hivemind's
    "Started" marker. join_client (the repo-clone path) must behave like the
    QR path, so it delegates instead of duplicating that logic.
    """
    try:
        import sys as _sys
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.bootstrap import execute_job_command as _exec  # type: ignore
        return _exec
    except Exception:
        return None


def execute_job_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a shell command for serving. Does NOT return until the server exits.

    Fallback path only — main() prefers the bootstrap executor via
    ``_bootstrap_executor()`` so repo-clone peers get the same preflight and
    status reporting as QR peers."""
    start_time = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            env={**__import__("os").environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=None,  # server runs indefinitely; caller controls lifecycle
        )
        elapsed = time.time() - start_time
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout,
            "stderr": proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
            "elapsed_seconds": elapsed,
            "claim_boundary": JOB_POLL_CLAIM_BOUNDARY,
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        return {
            "command": command,
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "claim_boundary": JOB_POLL_CLAIM_BOUNDARY,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-url", required=True)
    parser.add_argument("--capabilities", required=True, help="Path to peer capability JSON from peer_scan.py")
    parser.add_argument("--peer-id", default=None, help="Override peer id; defaults to capabilities.hostname")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--now", type=int, default=None)
    parser.add_argument("--count", type=int, default=1, help="Number of heartbeats to post; default is one-shot")
    parser.add_argument("--interval-seconds", type=float, default=10.0, help="Seconds to sleep between repeated heartbeats")
    parser.add_argument("--dry-run", action="store_true", help="Print request payload without sending it")
    parser.add_argument("--poll-job", action="store_true", help="After heartbeat, poll /job for deployment assignment")
    parser.add_argument("--auto-serve", action="store_true", help="Poll job and auto-execute serve command (--poll-job + execution)")
    parser.add_argument("--repo-dir", default=None, help="Working directory for auto-serve command execution")
    args = parser.parse_args(argv)

    # Resolve peer_id early
    capabilities = _load_capabilities(args.capabilities)
    resolved_peer_id = args.peer_id or capabilities.get("hostname") or capabilities.get("peer_id")

    if args.dry_run:
        payload = dry_run_report(args.join_url, capabilities_path=args.capabilities, peer_id=args.peer_id, now=args.now)
    elif args.auto_serve:
        # Heartbeat first, then poll job
        hb = post_heartbeat(
            args.join_url,
            capabilities_path=args.capabilities,
            peer_id=args.peer_id,
            timeout=args.timeout,
            now=args.now or int(time.time()),
        )
        # Poll for job
        poll_result = poll_job(args.join_url, peer_id=resolved_peer_id, timeout=args.timeout)
        job_response = poll_result.get("job_response") or {}
        job = job_response.get("job")
        if job and job.get("command"):
            print(json.dumps({"heartbeat": hb, "poll": poll_result}, indent=2, sort_keys=True), file=sys.stderr)
            # Auto-execute the serve command — prefer the bootstrap executor
            # (preflight + per-shard download + live status) so this path
            # behaves exactly like the QR bootstrap.
            bootstrap_exec = _bootstrap_executor()
            if bootstrap_exec is not None:
                join = parse_join_url(args.join_url)
                exec_result = bootstrap_exec(
                    job["command"],
                    cwd=args.repo_dir or None,
                    coordinator=join["coordinator"],
                    peer_id=str(resolved_peer_id),
                    hostname=str(capabilities.get("hostname") or resolved_peer_id),
                    job_port=job.get("port"),
                    model_id=job_response.get("deployed_model"),
                    block_range=job.get("block_indices") or job.get("block_range"),
                    num_layers=job.get("num_layers"),
                )
            else:
                exec_result = execute_job_command(job["command"], cwd=args.repo_dir or None)
            payload = {"auto_serve": True, "heartbeat": hb, "job": job, "execution": exec_result, "claim_boundary": JOB_POLL_CLAIM_BOUNDARY}
        else:
            payload = {"auto_serve": True, "heartbeat": hb, "poll": poll_result, "no_job_assigned": True, "claim_boundary": JOB_POLL_CLAIM_BOUNDARY}
    elif args.poll_job:
        poll_result = poll_job(args.join_url, peer_id=resolved_peer_id, timeout=args.timeout)
        payload = poll_result
    elif args.count != 1:
        payload = run_heartbeat_loop(
            args.join_url,
            capabilities_path=args.capabilities,
            peer_id=args.peer_id,
            count=args.count,
            interval_seconds=args.interval_seconds,
            timeout=args.timeout,
            now=args.now,
        )
    else:
        payload = post_heartbeat(
            args.join_url,
            capabilities_path=args.capabilities,
            peer_id=args.peer_id,
            timeout=args.timeout,
            now=args.now or int(time.time()),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
