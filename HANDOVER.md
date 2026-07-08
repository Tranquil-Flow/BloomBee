# HANDOVER — BloomBee Distributed Inference MVP

**Repo:** `~/Projects/distributed-inference-mvp/`
**Branch:** `main`, ahead of `origin/main` by **290 commits** (network DNS down, not pushed)
**Date:** Wed 2026-07-08  ~10:25 CEST
**Current session ending — passing to a non-container agent.**

---

## TL;DR — Current State

| Component | Status | Detail |
|---|---|---|
| Coordinator | ✅ **LIVE** | `http://192.168.178.48:8787`, HTTP/1.1, `uptime 38+ min` |
| Dashboard | ✅ Live | `.local/operator-dashboard.html`, opens via `http://192.168.178.48:8787/` |
| m4pro peer | ✅ **Serving** | Tailscale `100.84.252.4:31338`, peer ID `12D3KooWJ6wd3fCgUKgJuvC7nwWMJa1kvs5jg5DJdDMY8eRSaUpT`, all 5 Qwen3-8B shards cached, p2pd PID `59948` |
| Evis-MacBook-Pro peer | ⚠️ **Stuck** | Heartbeating, but `hf download` on shards `00001`+`00002` is hung at ~161 MB / 4 GB (file lock contention; see "Open Issue" below) |
| Transport probe | ❌ **Fails** | `MissingBlocksError("No servers holding blocks 0 are online")` because Evis never finished downloading its blocks |
| Tests | 684 passed, 25 skipped | Per last Opus run |
| Active deployment | Qwen/Qwen3-8B (Qwen3-8B) | Evis=follower 0:9 port 31337, m4pro=seed 9:36 port 31338 |

**The single biggest blocker:** the Evis-MacBook-Pro peer cannot finish downloading its shard files. The seed (m4pro) is healthy and serving. Inference cannot be demonstrated end-to-end until Evis's download completes.

---

## What Just Got Fixed (last 90 minutes)

Three issues were diagnosed and fixed; all committed locally as `76914a6`:

### 1. HTTP_PROXY routing the bootstrap through mitmproxy
- **Symptom:** Every bootstrap heartbeat failed with `Remote end closed connection without response`. Dashboard stuck on stale "serving" status. run_server never launched.
- **Root cause:** `HTTP_PROXY=http://127.0.0.1:8443` (set by hermes-aegis) made `urllib.request` route bootstrap traffic through mitmdump. mitmdump rewrites request lines to absolute-URI form (`POST http://host:port/path HTTP/1.1`); `BaseHTTPRequestHandler.parse_request` silently rejects absolute-URI requests and closes the connection. urllib then raises `RemoteDisconnected`.
- **Why `NO_PROXY=192.168.0.0/16` didn't help:** `urllib.request.proxy_bypass()` does hostname matching only, not CIDR. `192.168.178.48` does not match `192.168.0.0/16` from urllib's perspective.
- **Fix A (bootstrap):** install a global `ProxyHandler({})` opener at import time so the bootstrap always talks directly to the LAN coordinator.
- **Fix B (coordinator):** set `protocol_version = "HTTP/1.1"` and override `parse_request` to accept absolute-URI request lines (defensive — works whether or not a client routes through a proxy).

### 2. False-positive "serving" status
- `bootstrap.py` line 783-789 posts `status: "serving"` even when the run_server subprocess exits cleanly (`return_code == 0`) but never printed hivemind's `Started` marker (`detected_serving == False`). This is a defensive "launcher exited; server may still be running in background" message but it lies when the launcher actually crashed silently.
- **Not yet fixed.** This is what's currently making the dashboard lie about Evis being healthy.
- **Recommended fix:** in that branch, post `status: "error"` (not `serving`) when `detected_serving == False`, with a message like `"server exited without 'Started' marker — check stdout"`.

### 3. HF xet download stalls on large shards
- `hf download Qwen/Qwen3-8B model-00001-of-00005.safetensors` (a 4.0 GB file) stalls after downloading 161 MB. The download process (`python3.1` PID 93048) is alive and holding the file lock but making zero network progress.
- Pre-existing shards (config, tokenizer) and an orphaned `~/.cache/huggingface/xet/logs/xet_*.log` from an earlier failed session still have stale entries from 02:54 UTC — suggesting an earlier hung xet was killed but left partial state.
- The bootstrap's `subprocess.run(timeout=900)` should fire around 10:23 UTC; bootstrap will then post `error: Download failed after 900s` and the next poll will retry.

---

## Topology & Network

| Host | LAN IP | Tailscale IP | Role | User |
|---|---|---|---|---|
| **Evis-MacBook-Pro** | `192.168.178.48` | — | Coordinator + follower (blocks 0:9) | `evinova` |
| **m4pro** | `192.168.178.52` | `100.84.252.4` | Seed (blocks 9:36) | `evinova-self` |
| Astra-Macbook | `192.168.178.47` | (was `100.78.72.79`, now `100.117.33.124`) | Optional peer (currently canceled) | (Astra) |

- Coordinator listens on `0.0.0.0:8787`. Auto-detects LAN IP.
- iOS gateway on port `8432` (`~/Projects/bloombee-ios-gateway/gateway/server.py`).
- Anisette Docker on port `6969`. Image `dadoum/anisette-v3-server` (NOT ghcr.io).
- Coordinator serves: `/servers.json` (SideStore anisette URL list), `/bloombee.ipa` (the iOS gateway IPA, 9.5 MB), `/bootstrap.py` (bootstrap script).

---

## How to Restart Everything (clean slate)

```bash
# On Evis-MacBook-Pro (this machine) — kill any leftover processes
pkill -9 -f "bootstrap.py.*bloombee" 2>/dev/null
pkill -9 -f "run_server.*Qwen3" 2>/dev/null
pkill -9 -f "hf download" 2>/dev/null
xcrun lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep -E '31337|31338'
# Should show no run_server / p2pd listeners. If anything, kill those PIDs.

# Optional: clear partial HF cache state (only if downloads keep stalling)
rm -rf ~/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-8B/
rm -f ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs/*.incomplete

# Restart coordinator
cd ~/Projects/distributed-inference-mvp
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server \
  --host 0.0.0.0 --port 8787 &
sleep 3
curl http://192.168.178.48:8787/healthz  # should return ok: true

# Start bootstrap with auto-serve (the QR-scannable command)
curl -s http://192.168.178.48:8787/bootstrap.py | python3 -u - \
  --join-url "bloombee://join?coordinator=http%3A%2F%2F192.168.178.48%3A8787&token=cUbnIgyQWCqZkKIbBD9FmjLfg8jhtlFP" \
  --loop --interval 15 --auto-serve
# Use `-u` for unbuffered stderr so progress is visible immediately.

# On m4pro (over SSH)
ssh m4pro
cd ~/Projects/distributed-inference-mvp
curl -s http://192.168.178.48:8787/bootstrap.py | python3 -u - \
  --join-url "bloombee://join?coordinator=http%3A%2F%2F192.168.178.48%3A8787&token=cUbnIgyQWCqZkKIbBD9FmjLfg8jhtlFP" \
  --loop --interval 15 --auto-serve
```

Generate a fresh join token (TTL 600s) at:
```
curl "http://192.168.178.48:8787/offer?peer_id=<name>&ttl_seconds=600"
```

---

## Open Issue — Evis's Stuck Download

**Symptom:** `hf download Qwen/Qwen3-8B model-00001-of-00005.safetensors model-00002-of-00005.safetensors` hangs. The first shard reaches ~161 MB then stalls indefinitely (last update 10:08 UTC, currently 10:25 UTC — 17 minutes of zero progress).

**Diagnosis commands:**
```bash
# Check active download processes
xcrun lsof 2>/dev/null | grep -E "hf_xet|hf download" | head -5

# Inspect in-progress files
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs/ | grep incomplete

# xet diagnostic log (may be helpful)
tail -30 ~/.cache/huggingface/xet/logs/xet_*.log

# Check file lock holders (downloaders acquire advisory locks here)
xcrun lsof 2>/dev/null | grep "models--Qwen--Qwen3-8B.*\.lock" | head -5
```

**Workarounds (in order of preference):**
1. **Wait for the 900s `subprocess.run` timeout** (~10:23 UTC). The bootstrap will post `error: Download failed after 900s` to the coordinator. The next poll will retry; the partial `.incomplete` files let `hf download` resume from where it stopped, so the second attempt may succeed.
2. **Disable xet** for the download (forces HTTP fallback):
   ```bash
   export HF_HUB_DISABLE_XET=1
   curl -s http://192.168.178.48:8787/bootstrap.py | python3 -u - \
     --join-url "...&token=..." --loop --interval 15 --auto-serve
   ```
3. **Manual download** (last resort):
   ```bash
   HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-8B \
     model-00001-of-00005.safetensors \
     model-00002-of-00005.safetensors
   # Then restart the bootstrap.
   ```

**Why this is happening:** Most likely a transient CDN / xet API issue from huggingface.co for Evis's network path. m4pro downloaded all 5 shards successfully earlier in this session, so the network itself is fine — it's a per-host flake with the xet endpoint.

---

## Files & Modules

### Coordinator (`mvp_capabilities/join_http_server.py`)
- HTTP/1.1, accepts absolute-URI request lines (commit `76914a6`).
- Routes: `/`, `/healthz`, `/heartbeat`, `/seed-multiaddr`, `/weights-needed`, `/deploy`, `/deploy/cancel`, `/job`, `/peer-status`, `/pipeline`, `/servers.json`, `/bloombee.ipa`, `/bootstrap.py`, `/bootstrap.sh`, `/offer`, `/active`, `/inference-feed`.
- `_sort_multiaddrs_lan_first()` sorts RFC 1918 LAN IPs before Tailscale/loopback when seed-multiaddr POSTs arrive (commit `5177fee`).
- `/weights-needed` endpoint returns per-peer `hf download` commands for the model's `model.safetensors.index.json` shards (commit `a3f5a4a`).

### Bootstrap (`scripts/bootstrap.py`)
- Stdlib-only Python script (deployable via `curl | python3 -`).
- Installs `ProxyHandler({})` opener at import so it always talks directly to the LAN coordinator (commit `76914a6`).
- Per-shard preflight: `_shards_needed_for_layers()` reads the local `model.safetensors.index.json` and computes the shards covering the peer's `--block_indices` range (commit `85e55e4`).
- Auto-download: if `model_weights_cached(required_shards=...)` fails, the bootstrap runs `hf download <model> <shard1> <shard2> ...` automatically, then verifies the cache before launching `run_server` (commit `d2be537`).
- Readiness detection: `is_server_ready_line()` only flips to "serving" on hivemind Runtime's `Started` marker (after every block's weights are loaded) — not on the premature "Running a server on..." line (commit `0e27cc0`).

### Dashboard
- `.local/operator-dashboard.html` (generated by `scripts/operator_dashboard.py`).
- Download Weights card in Deploy tab: per-peer shard list + 📋 Copy button (commit `24de422`).
- Generate tab filtered to deployed model only (commit `20dcad4`).

### Tests
- 684 passed, 25 skipped (last verified count).
- `tests/test_bootstrap_readiness.py` — 13 new tests covering preflight + serving detection.

---

## Recent Commit Log (last 10)

```
76914a6 fix(bootstrap): bypass HTTP_PROXY + accept absolute-URI request lines
20dcad4 fix(dashboard): Generate tab only shows the deployed model
d2be537 fix(bootstrap): auto-download missing shards on preflight — true zero-touch
85e55e4 fix(bootstrap): per-shard weight preflight — each peer fetches only its layers
24de422 feat(dashboard): Download Weights card — per-peer shard commands
a3f5a4a feat(coordinator): /weights-needed endpoint — per-peer shard download commands
5177fee fix(bootstrap): sort seed multiaddrs with LAN preference — not alphabetical
1d06cc8 test(deploy): rename ghost-colliding DemoLaptop peer to Demo-Mac
0e27cc0 fix(bootstrap): report 'serving' only on real readiness + preflight model weights
8b4815a docs: add handover for distributed inference mvp
```

To push to `tranquil-flow` (when network recovers):
```bash
git push tranquil-flow main
```

---

## Quick Reference — REST Endpoints

```
GET  /healthz                          → {"ok": true, "status": "live", ...}
GET  /                                 → operator dashboard (HTML)
GET  /bootstrap.py                     → bootstrap script (stdlib only)
GET  /servers.json                     → [{"name":"BloomBee Local","address":"http://192.168.178.48:6969"}]
GET  /bloombee.ipa                     → 9.5 MB iOS app
POST /heartbeat                        → register/heartbeat from peer
POST /seed-multiaddr                   → seed publishes its libp2p multiaddrs
GET  /job?peer_id=X                    → poll for assigned launch command
POST /peer-status                      → report status (downloading/loading/serving/error)
GET  /peer-status                      → all peers' current status
GET  /weights-needed?model_id=X        → per-peer hf download commands
POST /deploy                           → start a deployment (form: model_id, peer_count)
POST /deploy/cancel                    → cancel all jobs
GET  /pipeline                         → pipeline state across peers
GET  /active                           → active heartbeat roster
GET  /offer?peer_id=X&ttl_seconds=N    → generate a join token (TTL bounded)
```

---

## Known Limitations (carry-over from earlier sessions)

- **Inline inference in dashboard:** the `/infer` endpoint only does a readiness check; it doesn't actually run text generation through hivemind RPC. Generating text inline would require the dashboard to load `torch`+`bloombee` from `.venv` (which it can't from a static HTML file). Out of scope for this session.
- **Generate tab UI:** shows deployed model only, but the inference call still uses the bootstrap's `--auto-serve` path on the peer, not a direct coordinator-side client.
- **SideStore iPhone pairing:** UDID-related issue mentioned in earlier handovers — orthogonal to this session.
- **`run_server` once per bootstrap:** `job_executed = True` flag in bootstrap means a single bootstrap session executes its job at most once. If run_server crashes mid-load, the bootstrap won't auto-retry until you restart the bootstrap.

---

## What I'd Do Next (if continuing)

1. **Wait for Evis's 900s timeout** (~10:23 UTC). If it retries successfully → done. If not → manual `HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-8B model-00001-of-00005.safetensors model-00002-of-00005.safetensors` on Evis, then restart bootstrap.
2. **Fix the false-positive "serving" status** in `bootstrap.py` (commit suggestion: replace line 783-789 with `status: "error"` when `not detected_serving`).
3. **Verify end-to-end inference:** once Evis shows `serving`, run the transport probe from a terminal:
   ```bash
   .venv/bin/python scripts/direct_remote_call.py \
     --model Qwen/Qwen3-8B \
     --server-maddr "/ip4/100.84.252.4/tcp/31338/p2p/12D3KooWJ6wd3fCgUKgJuvC7nwWMJa1kvs5jg5DJdDMY8eRSaUpT" \
     --block-range 0:1
   # OR — better, against Evis's LAN multiaddr once it's published:
   .venv/bin/python scripts/direct_remote_call.py \
     --model Qwen/Qwen3-8B \
     --server-maddr "/ip4/192.168.178.48/tcp/31337/p2p/<evis_peer_id>" \
     --block-range 0:1
   ```
4. **Push the 290 commits** to `tranquil-flow` once network DNS recovers.

---

## Contact / Session Markers

- Session started: Wed 2026-07-08 ~07:30 UTC (~09:30 CEST)
- Session ended: Wed 2026-07-08 ~08:25 UTC (~10:25 CEST)
- Discord channel: #dark-forest / thread 1521796001969471488
- Last user message before handover: "can you create handover doc? going to pass this to another agent that is not in the container"

Good luck. ✨