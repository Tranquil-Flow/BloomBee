# Master Handover — BloomBee Distributed Inference MVP

**Repo:** `~/Projects/distributed-inference-mvp/`
**Branch:** `main`, HEAD = `31b41be` (+296 ahead of origin, awaiting DNS recovery to push)
**Last session:** 2026-07-08 (3 sessions, container-bound agent)
**Purpose:** Comprehensive history of every issue encountered, every fix applied, every route explored — everything a new agent needs to diagnose and fix remaining issues without rediscovering hard-won knowledge.

---

## Table of Contents

1. [What We're Building](#1-what-were-building)
2. [Current Topology & Hardware](#2-current-topology--hardware)
3. [Issue History: Every Bug, Its Cause, and Its Fix](#3-issue-history-every-bug-its-cause-and-its-fix)
4. [The Auto-Deploy Pipeline (How It's Supposed to Work)](#4-the-auto-deploy-pipeline-how-its-supposed-to-work)
5. [Routes Explored But Not Taken](#5-routes-explored-but-not-taken)
6. [What's Still Broken / Untested](#6-whats-still-broken--untested)
7. [Diagnostic Playbook (When Something Goes Wrong)](#7-diagnostic-playbook-when-something-goes-wrong)
8. [How to Restart Everything](#8-how-to-restart-everything)
9. [Key Files & Their Roles](#9-key-files--their-roles)
10. [REST API Quick Reference](#10-rest-api-quick-reference)
11. [Skills & References Library](#11-skills--references-library)

---

## 1. What We're Building

A distributed LLM inference system where multiple Apple Silicon Macs on the same
WiFi network pool their memory to run a model too large for any single machine.

**Model:** Qwen/Qwen3-8B (5 safetensors shards, ~16GB total in fp16)
**Framework:** BloomBee (a hivemind-based library that splits transformer layers across peers)
**Network constraint:** Same-wifi / local-network ONLY. No Tailscale, no VPN mesh.
**Goal:** Operator opens dashboard → clicks Deploy → peers auto-download their
assigned layers → start serving → inference works end-to-end.

The user's vision: **zero-touch deploy**. Click a button, and every peer in the
swarm automatically figures out which layers it needs, downloads them, and starts
serving. No manual `hf download`, no manual `run_server` commands.

---

## 2. Current Topology & Hardware

| Host | LAN IP (home) | Role | Hardware | User |
|---|---|---|---|---|
| **Evis-MacBook-Pro** | `192.168.178.48` | Coordinator + follower (blocks 0:9) | MacBook Pro 16GB | `evinova` |
| **m4pro** | `192.168.178.52` | Seed peer (blocks 9:36) | M4 Pro 48GB, all 5 shards cached | `evinova-self` |
| Astra-MacBook | `192.168.178.47` | Optional 3rd peer | MacBook 8GB | Astra |

**Note:** During the last session, both laptops were on **public wifi** (not home
network). IPs were unknown. The home LAN IPs above are stale — the coordinator
must re-detect on restart.

**Container-bound agent path:** `/workspace/Projects/distributed-inference-mvp/`
(bind-mounted to the Mac path). Container is Debian 13, aarch64, no wifi access,
no `ping`/`ifconfig`/`lsof`, no `.venv`, no `hivemind`/`torch`.

---

## 3. Issue History: Every Bug, Its Cause, and Its Fix

This is the core of the handover. Each issue is presented as Symptom → Root Cause
→ Fix → Commit. An agent diagnosing a similar problem should start here.

### Issue 1: Bootstrap ↔ Coordinator Heartbeats Silently Failing

**Symptom:** Bootstrap prints `❌ Failed to join: Remote end closed connection
without response` on every heartbeat. Dashboard shows peer as offline. But
`curl http://coordinator:8787/healthz` works fine.

**Root cause:** `HTTP_PROXY=http://127.0.0.1:8443` (set by hermes-aegis for mitm
certificate interception) made `urllib.request` route ALL bootstrap traffic
through mitmdump. Two compounding problems:

1. `urllib.request.proxy_bypass()` only does hostname matching, NOT CIDR matching.
   So `NO_PROXY=192.168.0.0/16` does NOT bypass the proxy for `192.168.178.48`.
2. mitmdump rewrites request lines to absolute-URI form (`POST http://host:port/path HTTP/1.1`).
   `BaseHTTPRequestHandler.parse_request` silently rejected these, closing the
   connection. urllib raised `RemoteDisconnected`.

**Fix (commit `76914a6`):**
- Bootstrap: install `ProxyHandler({})` opener at import time — always talks direct
- Coordinator: `protocol_version = "HTTP/1.1"` + override `parse_request` to accept absolute-URI lines

**Detailed reference:** `references/bootstrap-coordinator-http-pitfalls.md`

---

### Issue 2: False-Positive "serving" Status on Dashboard

**Symptom:** Dashboard shows a peer as `status: "serving"` with `progress: 100%`
within seconds of deploy. But no `run_server` process is alive, no
`seed_multiaddrs.json` exists, and inference fails with `MissingBlocksError`.

**Root cause:** `bootstrap.py`'s `execute_job_command()` posted `status="serving"`
when `return_code == 0` — even if the hivemind Runtime never printed its `Started`
marker. The launcher process could exit 0 (handing off to a detached `p2pd` child)
while the actual server crashed during weight loading.

**Fix (commit `3fff2d5`):**
- Track `detected_serving` flag based on the `Started` marker regex
  (`re.compile(r"(?:^|\])\s*Started\s*$")` — note `\]` boundary so
  "Inference computation started" does NOT match)
- When `rc=0` but `not detected_serving`: post `status="error"` with actionable message
- When `rc!=0`: post `status="error"` with exit code

**Detailed reference:** `references/auto-deploy-stall-and-readiness-pitfalls.md` §3

---

### Issue 3: HF Xet Download Stalls on Multi-GB Shards

**Symptom:** `hf download Qwen/Qwen3-8B model-00001-of-00005.safetensors` hangs
at ~161 MB / 4 GB. Process is alive, file lock is held, zero network progress for
17+ minutes. The `subprocess.run(timeout=900)` never fires because the process
isn't crashed — it's frozen on a stalled HTTP connection.

**Root cause:** The `hf_xet` protocol (default for new HF Hub installs) uses a
different CDN endpoint that is prone to mid-stream stalls on certain network paths.
m4pro downloaded all 5 shards successfully; Evis's network path to the xet
endpoint was flaky.

**Fix (commit `3fff2d5`):**
- Set `HF_HUB_DISABLE_XET=1` in the download subprocess env (forces plain HTTP)
- Replace `subprocess.run(timeout=900)` with `Popen` + idle-poll loop:
  - **IDLE_TIMEOUT_S = 300** (no stdout for 5 min = kill)
  - **TOTAL_TIMEOUT_S = 1800** (30 min wall-clock = kill)
- Clear stale `*.lock` and `*.incomplete` files before each download attempt
- `.incomplete` files from previous attempts are resume state — do NOT delete them
  (deleting forces full re-download of 4GB)

**Detailed reference:** `references/auto-deploy-stall-and-readiness-pitfalls.md` §1-2

---

### Issue 4: Coordinator Advertising Wrong IP (0.0.0.0 or Stale)

**Symptom:** Dashboard shows `http://0.0.0.0:8787` or a stale home-LAN IP.
Peers can't reach the coordinator. QR code is useless.

**Root cause (0.0.0.0):** When started with `--host 0.0.0.0` and no explicit
`--coordinator`, the coordinator baked `0.0.0.0` into the join URL. Browsers can't
route to `0.0.0.0`.

**Root cause (stale IP):** Coordinator detected `en0=192.168.178.48` at home.
User moved to public wifi. `en0` changed to a different IP. But the coordinator
was still running with the old detection.

**Fix (commit `11f8d7d` for 0.0.0.0, commit `341b5e1` for dashboard):**
- When `--host 0.0.0.0` and no `--coordinator`: call `_detect_lan_ip()` to auto-detect
- Dashboard generator also auto-detects via its own `_detect_lan_ip()`

**Fix (commit `31b41be` — multi-IP):**
- New `_detect_lan_ips()` returns ranked list: en0 → bridge → other en* → UDP fallback
- All candidates encoded into single join URL as `coordinator_2=`, `coordinator_3=` params
- Bootstrap's `pick_reachable_coordinator()` probes each `/healthz` (1.5s timeout)
- Tailscale CGNAT (100.64.0.0/10) intentionally excluded per same-wifi constraint

**Detailed reference:** `references/coordinator-lan-networking.md`,
`references/multi-ip-coordinator-offer.md`

---

### Issue 5: AP Client Isolation (NOT a Code Bug)

**Symptom:** Both laptops on same wifi. Coordinator is up (`curl localhost:8787/healthz`
works). But peer can't reach coordinator on ANY advertised IP. Bootstrap hangs or
gets `RemoteDisconnected`.

**Root cause:** Public/campus/hotel/cafe WiFi almost always enables **AP client
isolation** — each client can only talk upstream to the AP, not to other clients.
Device-to-device traffic is blocked at L2. WiFi association does NOT imply routability.

**Fix:** NONE. This is a network constraint, not a code bug. Options:
- (a) Phone hotspot tether (personal hotspots don't isolate)
- (b) Home router (most don't isolate on the main SSID)
- (c) USB tether between the two Macs (bridge100 interface bypasses wifi entirely)

**Do NOT try to code around this.** mDNS/Bonjour also fails (same L2 multicast).
WebRTC mesh is too heavy for an MVP. Tailscale works but user explicitly disallowed it.

**30-second diagnostic:**
```bash
# On peer laptop:
ping -c 3 -W 2 <coordinator-en0-ip>
# 100% packet loss = AP isolation. Not fixable in code.
```

**Detailed reference:** `references/coordinator-lan-networking.md` Pitfall 5

---

### Issue 6: Wire-Format Schema Drift (Planner Gets 0 Layers)

**Symptom:** Every peer joins successfully, heartbeats look fine, but `/plan`
returns `assigned_layers: 0, supported: false` for every model — even on a 48GB
M4 Pro trying to host a 4GB model.

**Root cause:** `bootstrap.py` emitted `memory.total_gb` / `memory.available_gb`
(matching `psutil` output). But `layer_planner.py::_peer_free_gb` only read
`memory.free_gb` (matching `/proc/meminfo` on Linux). The keys never overlapped.
The planner saw `None`, fell back to `vram_free_gb` (also unset on macOS), returned
`0.0` → `capacity_layers = 0` → 0 layers assigned.

**Why tests didn't catch it:** Unit tests used synthetic peer dicts that happened
to match the planner's expected keys. The bug only appeared with real bootstrap
output fed to the real planner.

**Fix:** `_peer_free_gb` now falls back through: `free_gb` → `available_gb` →
`total_gb` → `accelerator.vram_free_gb` → `disk.free_gb`. 16 new tests pin the
exact bootstrap-shaped peer dict against the planner.

**Detailed reference:** `references/bootstrap-planner-schema-drift.md`

---

### Issue 7: Seed Multiaddr Sorting (Followers Get Unreachable Address)

**Symptom:** Followers get the seed's multiaddr but can't connect. The multiaddr
contains a Tailscale IP (`100.x`) that the follower can't route to, even though
the seed also has a LAN IP (`192.168.x`).

**Root cause:** `sorted(multiaddrs)` puts `100.x` before `192.168.x`
alphabetically. The coordinator stored whatever order it received.

**Fix (commit `5177fee`):** `_sort_multiaddrs_lan_first()` sorts RFC 1918 LAN IPs
first, overlay/Tailscale second, loopback last. Applied on the bootstrap side
before `post_seed_multiaddr()`.

---

### Issue 8: Multiaddr Regex Silently Broken (Followers Stuck Forever)

**Symptom:** `extract_multiaddrs()` always returns `[]`. No multiaddr is ever
posted to the coordinator. Followers stay `waiting_for_seed` indefinitely.

**Root cause:** The regex used doubled backslashes inside a raw string:
`r"[^\\s,'\"\\]]"` — Python's `re` module interprets this as requiring a literal
`]` after every matched character. No multiaddr contains `]`, so nothing matches.

**Fix (commit `b5271a3`):** Use single-escaped `\s` and `\]` inside the raw string:
`r"[^\s,'\"\]]"`.

**Detailed reference:** `references/bootstrap-pitfalls.md`

---

### Issue 9: Venv Path Double-Replacement

**Symptom:** The `run_server` command contains a doubled venv path like
`.../venv/bin/python/bin/python -m bloombee...`

**Root cause:** Chaining `.replace("python3 ", venv).replace("python ", venv)`.
The second replace matches `python ` inside the just-replaced path.

**Fix (commit `7ad9094`):** Use `if/elif`, not chained replaces. Only replace once.

---

### Issue 10: Demo-Laptop Ghost Heartbeats

**Symptom:** A stale `while sleep; do curl heartbeat; done` shell loop on the
coordinator machine sends heartbeats with hostname `DemoLaptop`, confusing the
roster and deployment plan.

**Fix (commit `1edae1e`):** Block at `record_heartbeat()` in `join_coordinator.py`:
reject any peer_id/hostname matching `demo-laptop` / `demolaptop`. The real peer
on that machine is `Evis-MacBook-Pro`.

---

### Issue 11: `/active` Endpoint Deleting deployment.json

**Symptom:** After polling `/active`, the deployment plan disappears. Re-deploy
becomes necessary every few minutes.

**Root cause:** The `/active` handler's cleanup loop computed age as
`(now - timestamp)` where `timestamp=0` for non-heartbeat files (like
`deployment.json`). This produced a huge age → file deleted as "stale".

**Fix:** Skip cleanup for any file that doesn't have `peer_id` and `timestamp`
fields (i.e., only clean up actual heartbeat files).

**Detailed reference:** `references/active-endpoint-deletes-deployment.md`

---

### Issue 12: QR Bootstrap Missing `--auto-serve`

**Symptom:** Peers join via QR code, show up in roster, but never pick up deploy
jobs. Dashboard shows "waiting for peer" forever.

**Root cause:** `--auto-serve` was added to `join_client.py` and the multi-step
bootstrap runbook, but NOT to the single-line QR command string. 99% of users use
the QR path.

**Fix (commit `8795945`):** Added `--auto-serve` to the QR-shown command in the
landing page JS template.

**Rule:** Any peer-side feature must be added to BOTH `bootstrap.py` AND
`join_client.py` AND the QR command string. See
`references/bootstrap-peer-join-flow.md`.

---

## 4. The Auto-Deploy Pipeline (How It's Supposed to Work)

This is the full zero-touch flow from operator click to inference:

```
Operator clicks "Deploy" in dashboard
        │
        ▼
POST /deploy?model_id=Qwen/Qwen3-8B&token=*
        │
        ▼
Coordinator:
  1. layer_planner.py splits model layers across active peers
     (proportional to free memory)
  2. attach_launch_commands() generates per-peer run_server commands
     with --block_indices and seed multiaddr placeholder
  3. Saves deployment.json
        │
        ▼
Each peer's bootstrap (running --loop --auto-serve):
  1. Heartbeats every N seconds
  2. Polls GET /job?peer_id=X
  3. Gets its assigned launch command
  4. If seed multiaddr is placeholder → status=waiting_for_seed
  5. Once seed reports multiaddr → coordinator substitutes placeholder
  6. Peer gets executable command
        │
        ▼
Peer executes launch command:
  1. _shards_needed_for_layers() → which safetensors shards cover my layers?
  2. model_weights_cached() → are they on disk?
  3. If not → hf download with HF_HUB_DISABLE_XET=1 + idle-poll loop
  4. Post status=downloading with progress estimates
  5. Post status=loading
  6. Launch run_server (hivemind server process)
  7. Watch stdout for "Started" marker → post status=serving
  8. Post seed multiaddr to /seed-multiaddr (if seed peer)
  9. Heartbeat serving status every 4s
        │
        ▼
All peers serving:
  1. Transport probe: direct_remote_call.py verifies forward/backward
  2. Dashboard shows all peers green
  3. Inference works end-to-end
```

### Where It Breaks (Most Common Points of Failure)

1. **Weight download** — xet stalls, CDN flakiness, partial cache locks (Issue 3)
2. **Network connectivity** — stale join URL, wrong IP, AP isolation (Issues 4-5)
3. **Seed multiaddr propagation** — regex broken, sort wrong, placeholder not
   substituted (Issues 7-8)
4. **False readiness** — server exits 0 but never started (Issue 2)
5. **Planner rejection** — wire-format drift → 0 layers assigned (Issue 6)

---

## 5. Routes Explored But Not Taken

These were seriously considered, designed, or partially implemented, then
deliberately NOT shipped. An agent revisiting them should understand WHY before
redoing the work.

### LAN-Seed HTTP Server (Peer-to-Peer Weight Transfer)

**Idea:** Instead of every peer downloading from HuggingFace, one peer downloads
the full model, then serves shards to other peers over the local network.

**Why it was explored:** The user suggested it. It would avoid redundant
multi-GB downloads and eliminate xet stall dependency. Faster on slow internet.

**Why it was NOT shipped:**
- User's first preference was "each peer downloads from HF" (simpler)
- The xet stall was fixed by `HF_HUB_DISABLE_XET=1` + idle-killer (commit `3fff2d5`)
- Adding a `/weights-source` endpoint + seed relay would be ~200 lines of new code
  with new failure modes (what if the seed's HTTP server dies mid-transfer?)
- The user said "just want it working" — minimal changes preferred

**If revisiting:** The coordinator already has `/weights-needed` (returns per-peer
shard download commands). Add a `/weights-source` endpoint on each serving peer that
streams shard files from its HF cache. Bootstrap falls back to this if HF download
fails. This is architecturally sound — see the `references/weight-download-and-auto-deploy.md`
reference for the shard-mapping logic.

### Tailscale Auto-Detection

**Idea:** Include Tailscale IPs (100.64.0.0/10) in the coordinator's candidate IP
list as a fallback when LAN is unreachable.

**Why it was NOT shipped:** User explicitly disallowed Tailscale: "for the MVP we
want multiple devices connected on the same wifi." Including Tailscale would violate
this constraint even as a fallback.

**Current state:** `_detect_lan_ips()` intentionally EXCLUDES Tailscale CGNAT
addresses via `_is_tailscale_cgnat_ipv4()`. If the user later changes their mind,
remove that filter.

### mDNS / Bonjour Auto-Discovery

**Idea:** Peers discover the coordinator automatically via mDNS without needing a
join URL or QR code.

**Why it was NOT shipped:** mDNS is L2/L3 multicast within the subnet. If AP
client isolation is the problem, mDNS goes nowhere too. It's the same failure mode
as direct IP connectivity, just with extra complexity. Not worth the code for an MVP.

### WebRTC Mesh / Hole-Punching

**Idea:** NAT traversal for peers behind restrictive networks.

**Why it was NOT shipped:** Far too heavy for an MVP demo. The user wants same-wifi.
If same-wifi doesn't work (AP isolation), the answer is "change wifi," not "add a
mesh networking layer."

### Multi-IP Offer as `join_urls` List (Not Single URL)

**Idea:** The coordinator's `/offer` would return a `join_urls` array of separate
bloombee URLs, one per candidate IP.

**Why it was NOT shipped:** QR codes, iMessage paste, clipboard round-trips all
carry ONE string. If candidates were split across multiple URLs, the user would only
paste one and lose the fallbacks. Solution: encode all candidates into a single URL
using `coordinator_2=`, `coordinator_3=` query params.

### Dashboard Inline Inference

**Idea:** Dashboard Generate tab runs inference directly through the coordinator.

**Why it was NOT shipped:** The `/infer` endpoint only does a readiness check.
Running text generation through hivemind RPC requires loading `torch`+`bloombee`
from `.venv`, which can't happen from a static HTML file. Out of scope — the
Generate tab currently uses the bootstrap's `--auto-serve` path on the peer.

---

## 6. What's Still Broken / Untested

### Untested on Real Hardware

1. **Multi-IP offer + reachability probe** (commit `31b41be`) — 6/6 integration
   checks pass in container, but never tested with real Mac wifi interfaces. Need
   to restart coordinator on Evis with `--host 0.0.0.0` and verify `/offer` shows
   multiple `coordinator_urls`.

2. **Xet stall fix** (commit `3fff2d5`) — `HF_HUB_DISABLE_XET=1` + idle-killer
   not yet proven on real shard download. Need to restart bootstrap on Evis and
   watch for downloading → loading → serving transitions.

3. **End-to-end inference** — never achieved. Evis's download was always stuck.
   Once both peers are serving, verify with:
   ```bash
   .venv/bin/python scripts/direct_remote_call.py \
     --model Qwen/Qwen3-8B \
     --server-maddr "/ip4/<peer-lan-ip>/tcp/31338/p2p/<peer-id>" \
     --block-range 0:1
   ```

### Known Limitations

- **`run_server` once per bootstrap session:** `job_executed = True` flag means
  a single bootstrap session executes its job at most once. If run_server crashes
  mid-load, the bootstrap won't auto-retry until you restart the bootstrap.
- **Inline inference in dashboard:** Generate tab is a stub. See "Routes Not Taken" above.
- **296+ unpushed commits:** DNS was down at session end. Push to `tranquil-flow`
  when network recovers: `git push tranquil-flow main`

### The User's Original Idea (Not Yet Implemented)

> "One potential idea to get the first version of the MVP working would be to
> fully download the model on one laptop, and then distribute the layers needed
> to the other peers over the local network?"

This is the **LAN-seed weight transfer** idea (see Routes Explored §1). It was
not implemented because the user preferred "each peer downloads from HF" first.
If HF downloads continue to be unreliable, this is the next step — and it's
architecturally designed (see `/weights-needed` endpoint + shard mapping logic),
just not wired into a peer-to-peer transfer server.

---

## 7. Diagnostic Playbook (When Something Goes Wrong)

Run these in order. Each takes <10 seconds and rules out one layer.

### Layer 1: Is the coordinator alive?

```bash
curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
# Should show {"ok": true, "status": "live", "coordinator": "http://...", ...}
```

### Layer 2: Is the coordinator listening on the right interface?

```bash
lsof -nP -iTCP:8787 -sTCP:LISTEN
# Should show TCP *:8787 (LISTEN) — if it shows 127.0.0.1:8787, restart with --host 0.0.0.0
```

### Layer 3: Can the peer reach the coordinator?

```bash
# On the peer laptop:
curl -s http://<coordinator-ip>:8787/healthz
# If this fails → network issue (stale IP, AP isolation, firewall)
```

### Layer 4: Is a proxy intercepting traffic?

```bash
env | grep -iE "proxy|HTTP_"
# If HTTP_PROXY is set, the bootstrap's ProxyHandler({}) bypass should handle it
# But verify: python3 -c "from urllib.request import *; install_opener(build_opener(ProxyHandler({}))); print(urlopen('http://<coord>:8787/healthz', timeout=3).status)"
```

### Layer 5: Is the bootstrap actually running with --auto-serve?

```bash
ps aux | grep bootstrap
# Should show --loop --interval N --auto-serve
# If --auto-serve is missing, the peer is only heartbeating, never picking up jobs
```

### Layer 6: Is the deployment plan present?

```bash
cat .local/join-state/deployment.json | python3 -m json.tool
# Should show model_id, peer assignments, launch commands
# If missing → /active endpoint may have deleted it (Issue 11)
```

### Layer 7: Is the seed multiaddr resolved?

```bash
curl -s http://<coord>:8787/pipeline | python3 -m json.tool
# Check each peer's status:
# - waiting_for_seed → seed hasn't reported multiaddr yet
# - queued → job assigned but not started
# - downloading/loading → in progress
# - serving → done
```

### Layer 8: Are model weights downloaded?

```bash
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs/ | grep -E "safetensors|incomplete"
# Each shard should be ~4GB and NOT .incomplete
# If .incomplete → download was interrupted, needs resume
```

### Layer 9: Is AP client isolation the problem?

```bash
# On peer laptop:
ping -c 3 -W 2 <coordinator-en0-ip>
# 100% packet loss = AP isolation. NOT fixable in code.
```

---

## 8. How to Restart Everything

```bash
# ═══ On Evis-MacBook-Pro (coordinator) ═══

cd ~/Projects/distributed-inference-mvp

# Kill stale processes
pkill -9 -f "bootstrap.py.*bloombee" 2>/dev/null
pkill -9 -f "run_server.*Qwen3" 2>/dev/null
pkill -9 -f "hf download" 2>/dev/null
pkill -9 -f "join_http_server" 2>/dev/null

# Clear partial HF cache state (only if downloads keep stalling)
rm -rf ~/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-8B/
rm -f ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs/*.incomplete

# Start coordinator
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server \
  --host 0.0.0.0 --port 8787 &
sleep 3

# Verify
EIP=$(ipconfig getifaddr en0)
curl -s http://$EIP:8787/healthz
curl -s "http://$EIP:8787/offer?ttl_seconds=1800" | python3 -c "import sys,json; print(json.load(sys.stdin)['join_url'])"
# Copy the join_url output

# Start bootstrap on Evis (also a peer)
curl -s http://$EIP:8787/bootstrap.py | python3 -u - \
  --join-url "<paste join_url>" \
  --loop --interval 15 --auto-serve

# ═══ On m4pro (seed peer) ═══

cd ~/Projects/distributed-inference-mvp
pkill -9 -f "bootstrap.py.*bloombee" 2>/dev/null

curl -s http://$EIP:8787/bootstrap.py | python3 -u - \
  --join-url "<paste same join_url>" \
  --loop --interval 15 --auto-serve

# ═══ Then: open dashboard and click Deploy ═══
open http://$EIP:8787/
# Or: open .local/operator-dashboard.html
```

---

## 9. Key Files & Their Roles

### Coordinator
| File | Role |
|---|---|
| `mvp_capabilities/join_http_server.py` | HTTP server: all REST endpoints, LAN IP detection, landing page |
| `mvp_capabilities/join_coordinator.py` | Join offer creation, heartbeat recording, layer planning |
| `mvp_capabilities/layer_planner.py` | Splits model layers across peers by memory capacity |

### Bootstrap / Peer
| File | Role |
|---|---|
| `scripts/bootstrap.py` | Stdlib-only peer bootstrap: scan hardware → join → heartbeat → auto-serve |
| `mvp_capabilities/join_client.py` | Alternative join client (manual / repo-clone users) |
| `mvp_capabilities/peer_scan.py` | Hardware scan: CPU, RAM, GPU, disk → JSON capabilities |

### Dashboard
| File | Role |
|---|---|
| `scripts/operator_dashboard.py` | Generates static HTML dashboard (Onboarding + Live Swarm + Deploy + Generate tabs) |
| `scripts/coordinator_landing.py` | Generates the `GET /` landing page with QR code |

### Tests
| File | Tests |
|---|---|
| `tests/test_mvp_capabilities.py` | ~700 tests covering coordinator, planner, offer, heartbeat, deploy pipeline |
| `tests/test_bootstrap_readiness.py` | ~17 tests: readiness detection, weight preflight, multi-IP parse, serving status |
| `tests/test_evidence_redaction.py` | Evidence redaction patterns |

---

## 10. REST API Quick Reference

```
GET  /healthz                          → {"ok": true, "status": "live", "coordinator_urls": [...]}
GET  /                                 → operator landing page (HTML with QR)
GET  /bootstrap.py                     → bootstrap script (stdlib only, curl | python3 -)
GET  /offer?ttl_seconds=N              → join offer with multi-IP candidates
POST /heartbeat                        → register/heartbeat from peer
POST /seed-multiaddr                   → seed publishes its libp2p multiaddrs
GET  /job?peer_id=X                    → poll for assigned launch command
POST /peer-status                      → report status (downloading/loading/serving/error)
GET  /peer-status                      → all peers' current status
GET  /weights-needed?model_id=X        → per-peer hf download commands
POST /deploy?model_id=X                → start a deployment
POST /deploy/cancel                    → cancel all jobs
GET  /pipeline                         → pipeline state across peers
GET  /active                           → active heartbeat roster
GET  /compatible?token=*               → compatible models for current swarm
GET  /servers.json                     → SideStore anisette server list
GET  /bloombee.ipa                     → iOS gateway app
```

---

## 11. Skills & References Library

The `distributed-inference-mvp` skill (in `~/.hermes/skills/mlops/distributed-inference-mvp/`)
contains 100+ reference files. The most important ones for diagnosing issues:

| Reference | Covers |
|---|---|
| `references/coordinator-lan-networking.md` | 5 LAN pitfalls: bind 0.0.0.0, dashboard URL, coordinator consistency, 0.0.0.0 baking, AP isolation diagnostic table |
| `references/multi-ip-coordinator-offer.md` | Multi-IP ranked candidates, single-URL encoding, reachability probe pattern |
| `references/bootstrap-coordinator-http-pitfalls.md` | HTTP_PROXY interception, absolute-URI parsing, false-positive serving |
| `references/auto-deploy-stall-and-readiness-pitfalls.md` | Xet stalls, idle-poll loop, lock cleanup, rc=0≠serving |
| `references/auto-deploy-weight-fetch.md` | Per-shard download, preflight, HF cache verification |
| `references/bootstrap-pitfalls.md` | Multiaddr regex, venv double-replace, repo discovery, LAN sort, ghost heartbeats |
| `references/bootstrap-planner-schema-drift.md` | Wire-format drift → 0 layers assigned, the generalization pattern |
| `references/bootstrap-peer-join-flow.md` | Two-script problem, QR vs manual, live progress reporting |
| `references/deploy-pipeline-debugging.md` | Pipeline status meanings, seed/follower flow, common failure modes |
| `references/weight-download-and-auto-deploy.md` | Shard mapping logic, LAN-seed transfer design (not implemented) |
| `references/active-endpoint-deletes-deployment.md` | /active cleanup loop deleting deployment.json |
| `references/zero-touch-auto-serve-coordinator.md` | Full coordinator/peer contract, field-name pitfalls |

---

## Final Notes

This codebase has 296+ commits ahead of origin. The git history IS the debug
narrative — each commit message explains what broke and why. When in doubt,
`git log --oneline --grep="<keyword>"` to find the relevant fix.

The user values honesty over fake confidence. "Untested" admissions are preferred
to fabricated success claims. If something doesn't work, say so directly and try
an alternative.

The user's overarching priority is getting the two-laptop distributed inference
demo working on same-wifi. Every fix in this history was in service of that goal.
The remaining blocker is testing the fixes on real hardware — the container agent
can write and verify code but cannot test Mac wifi networking.

Good luck. The moonlight is on your side. ✨🌙
