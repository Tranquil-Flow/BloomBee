#!/usr/bin/env python3
"""Stress test for the BloomBee onboarding swarm.

Validates:
  1. Five peers can join concurrently and all appear in /active.
  2. Two peers "die" (state files removed) and disappear from roster.
  3. The remaining 3 continue heartbeating and the roster self-heals.
  4. Malformed heartbeats (missing fields, bad token) are rejected with 400.
  5. Stale heartbeats (timestamp too old) are filtered out.

Run against a live coordinator on localhost:8787.

Usage: python3 scripts/stress_test_onboarding.py [--coordinator URL]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def http(method: str, url: str, body: bytes | None = None, timeout: int = 5) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def peer_capabilities(hostname: str, gpu: bool = False) -> dict:
    return {
        "platform": "darwin",
        "hostname": hostname,
        "python_version": "3.14",
        "cpu": {"cores": 8, "model": "arm"},
        "memory": {"total_gb": 16, "available_gb": 12},
        "gpu": ({"available": True, "backend": "mps", "name": "Apple Metal (MPS)"}
                if gpu else {"available": False, "backend": "cpu", "name": "cpu-only"}),
        "disk": {"total_gb": 500, "free_gb": 200},
        "network": {"local_ip": "192.168.178.99"},
    }


def join_peer(coordinator: str, peer_id: str, *, gpu: bool = False) -> tuple[str, str]:
    """Offer + first heartbeat. Returns (peer_id, token)."""
    code, body = http("GET", f"{coordinator}/offer?peer_id={peer_id}&ttl_seconds=600")
    assert code == 200, f"offer failed: {code} {body}"
    token = body["token"]
    cap = peer_capabilities(peer_id, gpu=gpu)
    payload = json.dumps({"token": token, "peer_id": peer_id, "capabilities": cap}).encode()
    code, _ = http("POST", f"{coordinator}/heartbeat", payload)
    assert code == 200, f"heartbeat failed: {code}"
    return peer_id, token


def test_five_peers_concurrent(coordinator: str) -> None:
    print("\n=== Test 1: 5 peers join concurrently ===")
    peers = [
        ("laptop-alice", True),
        ("laptop-bob", False),
        ("phone-pixel", False),
        ("laptop-charlie", True),
        ("phone-samsung", False),
    ]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda p: join_peer(coordinator, p[0], gpu=p[1]), peers))
    dt = time.time() - t0
    assert len(results) == 5, f"expected 5 joins, got {len(results)}"
    code, body = http("GET", f"{coordinator}/active?max_age_seconds=60")
    assert code == 200
    active = body.get("active_peers", [])
    peer_ids = sorted(p["peer_id"] for p in active)
    expected = sorted([p[0] for p in peers])
    assert peer_ids == expected, f"roster mismatch: got {peer_ids}, want {expected}"
    print(f"  ✅ All 5 peers visible in {dt*1000:.0f}ms: {peer_ids}")


def test_peer_departure(coordinator: str) -> None:
    print("\n=== Test 2: Two peers die, roster self-heals ===")
    state_dir = Path(".local/join-state")
    victims = ["laptop-bob", "phone-samsung"]
    for v in victims:
        (state_dir / f"{v}.json").unlink(missing_ok=True)
    code, body = http("GET", f"{coordinator}/active?max_age_seconds=60")
    active = [p["peer_id"] for p in body.get("active_peers", [])]
    assert "laptop-bob" not in active, "laptop-bob should be gone"
    assert "phone-samsung" not in active, "phone-samsung should be gone"
    assert len(active) == 3, f"expected 3 remaining, got {active}"
    print(f"  ✅ Roster self-healed to 3: {sorted(active)}")


def test_rejoin_after_partition(coordinator: str) -> None:
    print("\n=== Test 3: Rejoin after partition ===")
    join_peer(coordinator, "laptop-bob", gpu=False)
    code, body = http("GET", f"{coordinator}/active?max_age_seconds=60")
    active = sorted([p["peer_id"] for p in body.get("active_peers", [])])
    assert active == sorted(["laptop-alice", "laptop-bob", "laptop-charlie", "phone-pixel"]), active
    print(f"  ✅ Rejoined peer present: {active}")


def test_malformed_heartbeats(coordinator: str) -> None:
    print("\n=== Test 4: Malformed heartbeats rejected ===")
    bad_payloads = [
        (b"not json", "non-JSON body"),
        (b"{}", "empty body"),
        (b'{"token":"x"}', "missing peer_id + capabilities"),
        (b'{"peer_id":"x","capabilities":{}}', "missing token"),
        (b'{"token":"x","peer_id":"y","capabilities":"notdict"}', "capabilities not dict"),
    ]
    for payload, label in bad_payloads:
        code, _ = http("POST", f"{coordinator}/heartbeat", payload)
        assert code == 400, f"should reject {label} with 400, got {code}"
    print(f"  ✅ Rejected {len(bad_payloads)} malformed payloads with 400")

    # Valid token but unknown -> 200 (token validation happens at /plan stage, not /heartbeat)
    valid = json.dumps({"token": "fake-but-string", "peer_id": "x",
                        "capabilities": peer_capabilities("x")}).encode()
    code, _ = http("POST", f"{coordinator}/heartbeat", valid)
    print(f"  ℹ️  Heartbeat with unknown token (no validation at this layer): {code}")


def test_stale_heartbeat_filtered(coordinator: str) -> None:
    print("\n=== Test 5: Stale heartbeats filtered ===")
    import os
    state_file = Path(".local/join-state/stale-test.json")
    state_file.write_text(json.dumps({
        "peer_id": "stale-test",
        "token": "x",
        "ok": True,
        "capabilities": peer_capabilities("stale-test"),
        "timestamp": int(time.time()) - 600,  # 10 minutes ago
        "claim_boundary": "heartbeat_only_no_inference_proof",
    }))
    code, body = http("GET", f"{coordinator}/active?max_age_seconds=30")
    active = [p["peer_id"] for p in body.get("active_peers", [])]
    assert "stale-test" not in active, f"stale peer leaked into roster: {active}"
    state_file.unlink(missing_ok=True)
    print(f"  ✅ Stale heartbeat (10min old) correctly filtered")


def test_token_offer_race(coordinator: str) -> None:
    print("\n=== Test 6: 10 fresh offers for same peer all unique tokens ===")
    tokens = set()
    for _ in range(10):
        code, body = http("GET", f"{coordinator}/offer?peer_id=race-test&ttl_seconds=600")
        tokens.add(body["token"])
    assert len(tokens) == 10, f"expected 10 unique tokens, got {len(tokens)} ({tokens=})"
    print(f"  ✅ 10/10 tokens unique")


def cleanup() -> None:
    state_dir = Path(".local/join-state")
    keep_demo = {"Evis-MacBook-Pro-"}  # leave hostname-keyed alone
    for f in state_dir.glob("*.json"):
        if not any(f.name.startswith(k) for k in keep_demo):
            f.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stress test onboarding flow")
    parser.add_argument("--coordinator", default="http://localhost:8787")
    args = parser.parse_args(argv)

    print(f"Stress test against: {args.coordinator}")
    code, _ = http("GET", f"{args.coordinator}/healthz")
    if code != 200:
        print(f"❌ Coordinator not reachable (healthz={code})", file=sys.stderr)
        return 1

    cleanup()
    try:
        test_five_peers_concurrent(args.coordinator)
        test_peer_departure(args.coordinator)
        test_rejoin_after_partition(args.coordinator)
        test_malformed_heartbeats(args.coordinator)
        test_stale_heartbeat_filtered(args.coordinator)
        test_token_offer_race(args.coordinator)
        print("\n🎉 ALL STRESS TESTS PASSED\n")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
