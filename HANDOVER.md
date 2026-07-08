# BloomBee Distributed Inference MVP — Handover

**Date**: 2026-07-08 ~03:50 CEST  
**Written by**: Moonsong (Hermes Agent)  
**For**: Next agent or human continuing the distributed-inference-mvp work

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Coordinator | ✅ Live | `http://192.168.178.48:8787`, PID in bg |
| Anisette | ✅ Live | Port 6969, Docker `dadoum/anisette-v3-server` |
| iOS Gateway | ✅ Live | Port 8432, aiohttp |
| IPA | ✅ Built | 9.5 MB at `bloombee-ios-gateway/build/BloomBee.ipa` |
| Dashboard | ✅ Regenerated | `.local/operator-dashboard.html` |
| Tests | ✅ | 219 passed, 2 skipped |

**Active peers** (as of writing): Only Astra-Macbook (8 GB, 192.168.178.47) is heartbeating. Evis-MacBook-Pro and m4pro have dropped out (user stopped them). The demo-laptop ghost has been permanently blocked at `record_heartbeat()`.

**Deployment**: Qwen/Qwen3-8B@int8 deployed but no peers serving (both dropped). Need to restart bootstraps and redeploy.

**Network**: `nc -zv 192.168.178.52 31338` confirmed m4pro port is reachable from the coordinator machine.

---

## Machines

| Hostname | IP | RAM | Role | Tailscale |
|----------|-----|-----|------|-----------|
| Evis-MacBook-Pro | 192.168.178.48 | 16 GB | Coordinator + peer | Yes |
| m4pro | 192.168.178.52 | 48 GB | Seed peer | Yes (100.84.252.4) |
| Astra-Macbook | 192.168.178.47 | 8 GB | Peer | ? |

---

## Commits Since Branch Point

```
c469092 fix(infer): remove --prompt from command — direct_remote_call.py is a transport probe
e9ffe56 test: update demo-laptop references to demo-mac in seed multiaddr test
1edae1e fix(coordinator): block demo-laptop ghost heartbeats at entry point
c2db0e1 fix(infer): read multiaddrs field from seed data dict correctly
a6dde94 fix(infer): show correct command with seed multiaddr and venv path
2b7e340 fix(bootstrap): discover repo in sibling subdirectories when walking up
b5271a3 fix(bootstrap): fix multiaddr regex — raw-string \\s prevented extraction
7ad9094 fix(bootstrap): prevent double-replacement of venv python path
fc987e3 fix(bootstrap): auto-detect venv python and activate it for server commands
8201306 fix(bootstrap): resolve PYTHONPATH from project root, strip hardcoded prefix
da945b5 fix(onboarding): update SideStore URLs and serve IPA from coordinator
b835a05 feat(coordinator): add /servers.json endpoint for SideStore anisette server list
cdf1f62 fix(deploy): zero-touch seed multiaddr substitution and macOS python3 launch
```

**NOT pushed** — network DNS is down, all commits local.

---

## Key Fixes Applied

### 1. demo-laptop Ghost (CRITICAL)
A stale `while sleep; do curl heartbeat; done` shell loop from ~12h ago was sending hardcoded `peer_id=demo-laptop` heartbeats on the coordinator machine. Multiple kill attempts failed because the processes hid in zsh sessions.

**Fix**: `mvp_capabilities/join_coordinator.py` — `record_heartbeat()` now rejects heartbeats with `peer_id="demo-laptop"` or `hostname="demolaptop"` at entry point. File `demo-laptop.json` was blocked with `chflags uchg`. If the ghost reappears, the coordinator now silently rejects it.

### 2. Multiaddr Extraction Regex Broken
The regex `r"[^\\s,'\"\\]]*/p2p/..."` in `scripts/bootstrap.py` used doubled backslashes that Python's `re` module interpreted incorrectly — the character class required a literal `]` after every matched character. Since multiaddrs contain no `]`, it always returned `[]`.

**Fix**: Changed to `r"[^\s,'\"\]]*/p2p/..."` (single-escape `\s` for whitespace, single-escape `\]` for literal bracket).

**Impact**: `extract_multiaddrs()` always returned `[]` → `post_seed_multiaddr()` never called → `seed_multiaddrs.json` never created → followers stuck on "waiting_for_seed" forever.

### 3. Venv Python Path Doubling
In `execute_job_command()`, `replace("python3 ", venv_path)` was followed by `replace("python ", venv_path)`. The second replace matched `python ` inside the already-replaced venv path (e.g. `.../venv/bin/python -m`), doubling it.

**Fix**: Changed to `if/elif` — only fires one replacement.

### 4. Repo Root Discovery
The bootstrap walks up from cwd to find `src/bloombee/cli/run_server.py`. If run from `~/Projects/` (parent of the actual repo at `~/Projects/distributed-inference-mvp/`), the walk goes up, never finds the repo, and falls back to cwd — picking the wrong `.venv`.

**Fix**: At each walk level, also scans immediate subdirectories for `src/bloombee/cli/run_server.py`.

### 5. Venv Auto-Detection
`execute_job_command` now detects `.venv/bin/python` in the resolved repo root and replaces `python3` in the launch command. Also sets `VIRTUAL_ENV` and prepends `.venv/bin` to `PATH` in the subprocess environment.

### 6. SideStore Onboarding
- Added `GET /servers.json` → returns `{"servers": [{"name": "BloomBee Local", "address": "http://192.168.178.48:6969"}]}` for SideStore 0.5.8+ server list format
- Added `GET /bloombee.ipa` → serves the pre-built IPA directly from coordinator (no GitHub repo needed)
- Updated all onboarding URLs: `:6969` → `:8787/servers.json`, GitHub IPA link → coordinator `/bloombee.ipa`

### 7. Head Method for Chrome
Coordinator didn't implement `do_HEAD` — Chrome sends HEAD preflight before GET, got `501 Unsupported method`. Safari skips HEAD, which is why it worked while Chrome didn't.

**Fix**: Added `do_HEAD` handler that mirrors GET routing, sends headers only.

---

## INFERENCE: The Unresolved Problem

### What's Been Tried

The `direct_remote_call.py` script is a **transport probe** — it sends synthetic hidden states through the BloomBee hivemind pipeline to verify the distributed connection works. It does NOT do text generation (no `--prompt` argument).

The script runs a hivemind DHT daemon client that needs to bind ports. Inside the Hermes sandbox, this fails with `bind: operation not permitted`. The user ran it in their real terminal.

When connecting to m4pro via LAN IP (`192.168.178.52:31338`):
- `nc -zv 192.168.178.52 31338` → **SUCCEEDS** (port is open)
- DHT daemon starts, tries to connect → **FAILS**: `failed to connect to bootstrap peers`

### Root Cause Hypothesis

m4pro's server only advertises Tailscale IPs (`100.84.252.4`) and loopback — NOT the LAN IP (`192.168.178.52`). From the server startup log:
```
Running a server on ['/ip4/100.84.252.4/tcp/31338/p2p/...', '/ip4/127.0.0.1/tcp/31338/p2p/...', '/ip6/::1/tcp/31338/p2p/...']
```

Even though TCP port 31338 is open on the LAN interface, hivemind/libp2p verifies that the **advertised multiaddr** matches the connection. The client connects to `192.168.178.52:31338` but the server advertises `100.84.252.4:31338` — hivemind might reject the mismatch.

### Attempted Workarounds

1. **Manually adding LAN multiaddr to seed store** → coordinator accepted it, but the server process itself isn't listening on the LAN interface
2. **`--client-daemon` flag** → daemon can't bind in sandbox; in real terminal, still fails to connect
3. **`--client-listen` flag** → helps with local binding but doesn't fix the peer connection
4. **Local loopback test** (`127.0.0.1:31337`) → coordinator machine's own server multiaddr wasn't captured (regex was broken when it started)

### Recommended Next Steps

1. **Get the Evis-MacBook-Pro server's multiaddr**: Find the terminal where the bootstrap ran on Evis-MacBook-Pro (192.168.178.48) and look for `Running a server on [...]`. Use the `127.0.0.1` entry. Test locally:
   ```bash
   cd ~/Projects/distributed-inference-mvp
   .venv/bin/python scripts/direct_remote_call.py \
     --server-maddr "/ip4/127.0.0.1/tcp/31337/p2p/<REAL_MULTIADDR>" \
     --model Qwen/Qwen3-8B \
     --block-range 0:1 \
     --client-listen
   ```

2. **If local test works**: The pipeline is proven. Then fix the m4pro multi-interface issue:
   - Option A: Start m4pro server with an explicit host flag to also bind LAN IP
   - Option B: Route Tailscale traffic from coordinator to m4pro
   - Option C: Disable Tailscale on m4pro during testing so the server binds LAN IP only

3. **After transport probe works**: For actual text generation, use `bloombee.client.RemoteSequential` directly or write a dedicated inference script. The `/infer` endpoint currently only does readiness checks — it needs to be extended to actually run inference through hivemind RPC.

---

## Quick Reference: Commands

### Restart Coordinator
```bash
cd ~/Projects/distributed-inference-mvp
lsof -ti:8787 | xargs kill
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server --host 0.0.0.0 --port 8787
```

### Regenerate Dashboard
```bash
cd ~/Projects/distributed-inference-mvp
PYTHONPATH=.:src .venv/bin/python scripts/operator_dashboard.py \
  --coordinator "http://192.168.178.48:8787" \
  --out .local/operator-dashboard.html
```

### Run Tests
```bash
cd ~/Projects/distributed-inference-mvp
PYTHONPATH=.:src .venv/bin/python -m pytest tests/test_mvp_capabilities.py -q
# Expected: 219 passed, 2 skipped
```

### Join Token (get fresh one if expired)
```bash
curl -s http://192.168.178.48:8787/offer | python3 -c "import sys,json; print(json.load(sys.stdin)['join_url'])"
```

### Bootstrap Command (for all peer machines)
```bash
cd ~/Projects/distributed-inference-mvp
curl -s http://192.168.178.48:8787/bootstrap.py | python3 - \
  --join-url "bloombee://join?coordinator=http://192.168.178.48:8787&token=<TOKEN>" \
  --loop --interval 30 --auto-serve
```

### Cancel Deployment
```bash
curl -s -X POST http://192.168.178.48:8787/deploy/cancel
```

### Anisette Docker
```bash
docker rm -f bloombee-anisette 2>/dev/null
docker run -d --name bloombee-anisette --restart unless-stopped \
  -p 6969:6969 -v anisette_data:/home/Alcoholic/.config/anisette-v3/lib/ \
  dadoum/anisette-v3-server
```

### SideStore Anisette URL
```
http://192.168.178.48:8787/servers.json
```

---

## Remaining: SideStore UDID Error

When installing BloomBee.ipa via SideStore, user gets:
```
Sidestore could not determine this devices UDID
please replace your pairing using iLoader
```

This is a standard iOS sideloading step unrelated to BloomBee. Fix: download iLoader from [github.com/SideStore/iLoader/releases](https://github.com/SideStore/iLoader/releases), plug in iPhone via USB, generate new `.mobiledevicepairing` file, AirDrop to iPhone, import in SideStore settings.
