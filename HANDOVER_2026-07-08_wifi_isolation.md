# HANDOVER — same-wifi connectivity (2026-07-08 ~10:45 CEST)

**Repo:** `~/Projects/distributed-inference-mvp/`
**Branch:** `main`, last commit `3fff2d5` (locally; +292 on top of origin, awaiting DNS to push)
**Previous handover:** `HANDOVER.md` (867e068, ~10:25 CEST) — still accurate as overall context
**This handover ends at:** a non-container Hermes agent that can actually see the user's wifi.

---

## TL;DR — what happened

User is on **public wifi** with both Evis-MacBook-Pro and m4pro connected. Dashboard
shows "Cannot reach coordinator. Check the URL above." User's strict requirement
removing options: **no Tailscale**, must be same-wifi only.

Container-side agent (Moon / me) cannot resolve this — no `ip`, no `ifconfig`,
no wifi visibility, no SSH into either Mac. The next agent runs **directly on
one of the Macs** where it can `lsof`, `ping`, `curl` across the LAN, edit
files in place, restart services, and observe the dashboard. *All this handover
needs from the next agent is 60 seconds of diagnostics before touching code.*

**Do not start implementing any "same-wifi auto-discovery" feature yet.** The
error message tells us a join URL is wrong. There are four causes that produce
exactly that error — three are one-line config fixes, one is a hard network
constraint that no code can solve. See `## Step 1 — diagnostics` below. Pick the
right one before writing anything.

---

## Step 1 — diagnostics (run these first; ~30 seconds)

Run the three blocks and report findings to the user verbatim. **Do not skip this
even if the user re-asks for code.** Refusing to skip is itself the value you
add here — Moon already learned this once the hard way.

### On Evis (the coordinator laptop)

```bash
# What IP did Evis get on this wifi? (was 192.168.178.48 at home)
ipconfig getifaddr en0
# Fallback if that fails: ifconfig | grep "inet " | grep -v 127.0.0.1

# Is the coordinator actually listening right now? On what address?
lsof -nP -iTCP:8787 -sTCP:LISTEN

# Coordinator reachable from itself?
curl -s http://127.0.0.1:8787/healthz
```

### On m4pro (the seed laptop — was healthy at 100.84.252.4/192.168.178.52)

```bash
# Same network check
ipconfig getifaddr en0

# THE actual test: can m4pro reach Evis? Times out = AP isolation.
# If the user reports they're on public/coffee-shop/edu wifi, expect this to fail.
ping -c 3 -W 2 $(ssh evis "ipconfig getifaddr en0" 2>/dev/null) 2>&1 | tail -10
# If no SSH key setup, ask the user to read the IP off Evis and paste it.
```

### What the answers mean

| en0 IPs | ping result | diagnosis | action |
|---|---|---|---|
| Same subnet (both `192.168.1.x`, both `192.168.0.x`, etc.) and both `127.0.0.1:8787/healthz` returns `ok: true` | ping works | **Case 4: stale join token.** Bootstrap is using `bloombee://join?coordinator=...192.168.178.48...` from when the user was at home. | Restart `join_http_server` on Evis so it re-detects `en0`'s IP; generate a fresh `/offer`; restart both bootstraps with the new token. |
| Different subnets, but ping works | ping works | **Case 1: AP routed subnets.** User is at a hotel or campus where the AP uses 10.x or another corporate range; cross-peer routing is allowed. | Same fix as above — the auto-detect has already produced the right IP, just the token is stale. |
| Same/different subnet, ping fails with 100% packet loss | ping fails | **Case 3: AP client isolation.** No code can fix this. Public/cafe wifi almost always does this. | Tell the user honestly. Options: (a) hotspot-tether m4pro to Evis's phone, (b) switch to a wifi without isolation (home router, mobile hotspot, most "trusted" networks do allow client-to-client), (c) defer the demo to a less hostile network. **Don't ship a custom same-wifi discovery feature as a workaround for this case — the issue isn't discovery, it's isolation.** |
| `127.0.0.1:8787/healthz` fails | any | **Case 2: coordinator not listening.** Stale process or wrong bind. | `pkill -9 -f join_http_server; PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server --host 0.0.0.0 --port 8787 &` (see HANDOVER.md §"How to Restart Everything" for full recipe). |

The user's requirement "no Tailscale, same-wifi only" is **only satisfiable if
the answer lands in row 1, 2, or 4 above** (which together reduce to "restart
with the right URL"). If it lands in row 3, the user's constraint conflicts
with physics and we owe them honesty over engineering theatre.

---

## Step 2 — actual fix once diagnosis is in hand

The 60-second restart recipe if the answer is "stale URL":

**On Evis:**
```bash
cd ~/Projects/distributed-inference-mvp
pkill -9 -f join_http_server 2>/dev/null
sleep 1
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server \
  --host 0.0.0.0 --port 8787 &
sleep 3
EIP=$(ipconfig getifaddr en0)
echo "Coordinator advertising http://$EIP:8787"
curl -s http://$EIP:8787/healthz   # expect ok: true
curl -s "http://$EIP:8787/offer?peer_id=evis&ttl_seconds=1800" | tee /tmp/join.json
# Extract join_url (Python: jq, or python -c "import json,sys;print(json.load(open('/tmp/join.json'))['join_url'])")
# Hand that to m4pro via iMessage / clipboard / termux — easiest: copy it.
```

**On m4pro:**
```bash
pkill -9 -f "bootstrap.py.*bloombee" 2>/dev/null
sleep 1
curl -s "http://$EIP:8787/bootstrap.py" | python3 -u - \
  --join-url "<the fresh bloombee://join?... URL from /tmp/join.json>" \
  --loop --interval 15 --auto-serve
```

If the bootstrap's first heartbeat succeeds, the dashboard turns green and we're
back in business. **Try this before any code change.**

---

## Step 3 — only if the user wants the *code* improved regardless

If cases 1/2/4 are confirmed, also consider landing the multi-IP-offer
improvement as a tiny patch — it makes the next coffee-shop reboot a one-liner
without going through Step 2 again:

**Patch sketch** for `mvp_capabilities/join_http_server.py`:

```python
# In _detect_lan_ip (~line 204), add a Tailscale-preferred prologue before
# the en0 macOS path. Honours "no Tailscale" constraint: if Tailscale is not
# installed OR not running, the call returns empty stderr and we fall through
# to the existing en0 logic unchanged.
import subprocess as _sp
try:
    out = _sp.check_output(
        ["tailscale", "ip", "-4"], timeout=2, text=True, stderr=_sp.DEVNULL
    ).strip()
    if out:
        first = out.splitlines()[0].strip()
        if _looks_like_ipv4(first):
            return first
except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired):
    pass
```

Wait — the user has **prohibited Tailscale**. So patching Tailscale-preferred
auto-detect is out. What we *can* do legitimately: include **both IPs** in the
offer, in priority order, so a single token survives switching networks:

- `en0` LAN IP → primary (no infra needed)
- phone USB tether IP (`192.168.0.x` from `en1`/`bridge100`) → fallback if user
  later wants to tether m4pro to Evis's iPhone for a true same-network demo

Sketch: extend `/offer` to return `{"join_urls": [primary, fallback...], ...}`
and `bootstrap.py` to try them in order. Smaller patch, lower test surface, and
it survives the case where en0's IP changes between QR-scan and follower's
bootstrap launch (~30 s of clock skew while user types / SSHes).

But again — **do not implement this until Step 1's diagnostics confirm we
aren't in case 3 (AP isolation)**. Otherwise we'd be shipping a feature that
doesn't fix the user's actual problem.

---

## Step 4 — context the next agent needs

### Code state

- `main` is at `3fff2d5` (local), ahead of origin by ~292 commits. Last real
  code change before that was `76914a6` (HTTP_PROXY bypass). Network DNS is
  not back yet, so `git push tranquil-flow main` does not work in this session.
- `scripts/bootstrap.py` last touched by `3fff2d5` — auto-download hardened
  against xet stalls + false-positive `serving` status fixed. See commit
  message for full diff narrative.
- `mvp_capabilities/join_http_server.py` UNTOUCHED in `3fff2d5`. Still has the
  single-IP `_detect_lan_ip()` from the original HANDOVER.md context.

### Open issues the next agent inherits from HANDOVER.md

- **Evis-MacBook-Pro's `hf download` of shards 00001+00002** — last seen hung
  at 161 MB / 4 GB on 2026-07-08 ~10:08 UTC. Commit `3fff2d5` SHOULD have
  unblocked this on the next bootstrap restart (`HF_HUB_DISABLE_XET=1` +
  5-min idle-kill + lock cleanup). But it's untested in production — confirm
  by re-launching the bootstrap on Evis and watching for the "downloading" →
  "loading" → "serving" state transitions in the dashboard.
- **290+ committed-but-unpushed commits** — `git push tranquil-flow main`
  the moment DNS recovers.
- **Inline inference** in dashboard Generate tab is still a no-op stub (per
  HANDOVER.md "Known Limitations"). Out of scope here.

### Container-accessible skills the next agent should already know

If the next agent is on one of the Macs, none of the container-only skills
apply. They should have a `.venv` from prior sessions (per HANDOVER.md bootstrap
recipes), the `~/Projects/distributed-inference-mvp/` layout is the same, and
the coordinator command line is identical:

```
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server \
    --host 0.0.0.0 --port 8787
```

The "easy way to get a bootstrap going" still works:

```
curl -s http://<coord_ip>:8787/bootstrap.py | python3 -u - \
    --join-url "<join_url>" --loop --interval 15 --auto-serve
```

### What NOT to do

- **Don't add Tailscale.** The user has explicitly removed this option.
- **Don't implement a "LAN discovery via mDNS/Bonjour" workaround.** If AP
  isolation is the cause, mDNS goes nowhere either (it's still L2/L3 multicast
  within the subnet, which is exactly what AP isolation blocks).
- **Don't write a custom WebRTC mesh / hole-punching layer for a demo.**
- **Don't restart the swarm and call it "fixed" without verifying the
  dashboard flips to "serving".** Confirming with a `transport probe`
  (`scripts/direct_remote_call.py`, see HANDOVER.md §"Verify end-to-end
  inference") is the real proof.

---

## What I'd do next, in this exact order

1. **Read this whole handover first**, including the table in §"Step 1".
2. Run the three diagnostic blocks on Evis and m4pro, ~30 s total.
3. Match the answers to the table. Diagnose before coding.
4. If the diagnosis is "stale URL" (cases 1/2/4): restart with the §"Step 2" recipe
   and confirm the dashboard turns green. That's it. No commit needed.
5. If the diagnosis is "AP isolation" (case 3): be honest. Suggest hotspot-tether
   or a less hostile network. Do not bolt on a custom mesh.
6. If the user explicitly wants the multi-IP-offer code change from §"Step 3"
   *in addition to* a successful restart: write that patch as `3fff2d6`, run
   the existing test suite (`pytest tests/test_bootstrap_readiness.py -q`
   should be ~14 passing), commit with a clear message.
7. Push to `tranquil-flow main` when DNS recovers.

Good luck. The code is in good shape; this is a deployment-environment issue.
Get the diagnosis right before reaching for git. ✨🌙
