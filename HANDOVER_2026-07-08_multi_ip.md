# Handover: Multi-IP Coordinator Offer + Bootstrap Reachability

**Date**: 2026-07-08 (session 3)
**Agent**: Moon (container-bound, no wifi/Mac access)
**Base**: `b313a08` → this commit
**Status**: Code complete, all integration checks pass, NOT yet tested on real hardware

---

## TL;DR

The coordinator now advertises MULTIPLE candidate LAN IPs in every join offer,
and the bootstrap tries each one until it finds one that's reachable. This
fixes the "stale join URL" / "wrong interface" class of connectivity failures
that were blocking the two-laptop MVP demo.

**What this does NOT fix**: AP client isolation (public/hotel/cafe wifi that
blocks device-to-device traffic). That's a network-level problem, not a code
problem. The 30-second diagnostic in `references/coordinator-lan-networking.md`
Pitfall 5 distinguishes the two cases.

---

## What Changed (6 files, +404/-38)

### `mvp_capabilities/join_http_server.py` (+195/-38)

1. **`_detect_lan_ips()`** (NEW, ~line 219) — Returns ranked list of candidate
   IPv4 addresses: en0 (WiFi) → bridge100 (USB tether) → other en* → UDP-route
   fallback. Tailscale CGNAT (100.64.0.0/10) is EXCLUDED via
   `_is_tailscale_cgnat_ipv4()` per the "same-wifi only" MVP constraint.

2. **`_detect_lan_ip()`** (line 204) — Now a thin shim: `return _detect_lan_ips()[0]`.
   All existing callers unchanged.

3. **`handle_get()`** (line 1962) — Takes `coordinator_urls` kwarg. `/offer`
   passes them to `create_join_offer()`. `/healthz` exposes them in the response
   body for dashboard consumption.

4. **`JoinCoordinatorHTTPServer.__init__`** (line 2246) — Stores
   `coordinator_urls` list. Primary is always forced to position 0.

5. **`create_server()`** (line 2380+) — Takes `coordinator_urls` kwarg,
   passes to server constructor.

6. **`main()`** (line 2553+) — When `--host 0.0.0.0` and no `--coordinator`:
   calls `_detect_lan_ips()` to build the full candidate list, sets
   `args.coordinator_urls`, passes through to `create_server()`.

### `mvp_capabilities/join_coordinator.py` (+44/-1)

7. **`create_join_offer()`** (line 28) — New `coordinator_urls` param. The
   `coordinator` arg is always forced to position 0. All candidates encoded
   into ONE `bloombee://join?...` URL via `coordinator_2=`, `coordinator_3=`,
   etc. Also returns `coordinator_urls` (list) and `join_urls` (backwards-compat).

### `mvp_capabilities/join_client.py` (+22/-1)

8. **`parse_join_url()`** (line 31) — Now collects numbered
   `coordinator_2`/`coordinator_3`/...`coordinator_N` params. NUMERIC sort
   (not lexical) so `_10` comes after `_9`. Returns `coordinators` list
   alongside the legacy `coordinator`/`token`.

### `scripts/bootstrap.py` (+98/-6)

9. **`parse_join_url()`** (line 153) — Same multi-IP parsing as join_client.
   Returns `{coordinator, coordinators, token}`.

10. **`pick_reachable_coordinator()`** (NEW, line 191) — Probes each
    candidate's `/healthz` with 1.5s timeout. Returns first 2xx or None.

11. **`main()`** (line 995+) — Parses join URL → gets `coordinators` list →
    calls `pick_reachable_coordinator()` → uses the first reachable one.
    Falls back to `candidates[0]` if all probes fail (so heartbeat produces
    a real error, not a silent death).

### Tests (+83/-3)

12. **`tests/test_mvp_capabilities.py`** —
    - Updated `test_join_offer_builds_shareable_link_with_expiry` to assert
      new `coordinator_urls` and `join_urls` fields
    - NEW: `test_join_offer_encodes_ranked_multi_ip_fallbacks_in_single_url`
    - Updated `test_join_client_parses_join_url_and_builds_heartbeat_request`
      for new `coordinators` key
    - NEW: `test_join_client_parses_ranked_coordinator_fallbacks`

13. **`tests/test_bootstrap_readiness.py`** —
    - NEW: `test_parse_join_url_preserves_legacy_single_coordinator`
    - NEW: `test_parse_join_url_collects_ranked_coordinator_fallbacks`

---

## Integration Verification (all passed in container)

```
1. legacy single-coordinator parse: OK
2. multi-IP parse (3 candidates, numeric order): OK
3. pick_reachable unreachable → None: OK
4. live /offer multi-IP: OK
5. live /healthz exposes coordinator_urls: OK
6. pick_reachable skips dead, finds live: OK
```

---

## Architecture Decision: Single URL vs Multiple URLs

**Decision**: Encode ALL candidates into ONE `bloombee://join?...` URL using
`coordinator_2=`, `coordinator_3=` query params.

**Why NOT a `join_urls` list**: QR codes, iMessage paste, clipboard round-trips
all carry ONE string. If candidates were split across multiple URLs, the user
would only paste one and lose the fallbacks. Single URL = the full candidate
set survives every transport.

The offer payload DOES include `join_urls` (list of single-coordinator URLs)
for backwards compatibility, but the bootstrap and join_client only use the
primary `join_url` field.

---

## What the Next Agent Should Do

### Step 1: Commit (if not already committed)

```bash
cd ~/Projects/distributed-inference-mvp
git add -A
git commit -m "feat(coordinator): multi-IP offer + bootstrap reachability probe

Coordinator advertises ranked candidate LAN IPs in /offer and /healthz.
Bootstrap parses coordinator_2/_3/_N fallbacks, probes each /healthz,
uses first reachable. Fixes stale-URL and wrong-interface connectivity.

Tailscale CGNAT excluded per same-wifi MVP constraint.
6 integration checks pass. Not yet tested on real hardware."
```

### Step 2: Test on real hardware (REQUIRES non-container agent or Evi)

This is the critical next step. The code works in the container's integration
test, but has NOT been tested with real Mac wifi interfaces.

**On the coordinator laptop (Evis-MacBook-Pro):**

```bash
cd ~/Projects/distributed-inference-mvp
pkill -9 -f join_http_server  # kill stale coordinator
PYTHONPATH=.:src .venv/bin/python -m mvp_capabilities.join_http_server \
  --host 0.0.0.0 --port 8787
```

Check the startup log — it should show the detected LAN IPs. Then verify:

```bash
curl -s http://127.0.0.1:8787/offer | python3 -c "
import sys, json
o = json.load(sys.stdin)
print('coordinator:', o['coordinator'])
print('coordinator_urls:', o['coordinator_urls'])
print('join_url:', o['join_url'][:120], '...')
"
```

The `coordinator_urls` list should show en0's IP first, then any bridge/other
interfaces. NOT 100.64.x.x (Tailscale filtered).

**On the peer laptop (m4pro):**

```bash
cd ~/Projects/distributed-inference-mvp
# Get the join URL from the coordinator's /offer endpoint, then:
PYTHONPATH=.:src .venv/bin/python scripts/bootstrap.py \
  --join-url "<paste join_url from /offer>" --loop --interval 30 --auto-serve
```

Watch for:
- `🔗 Joining swarm (N candidate URLs)` — confirms multi-IP parsing
- `↪ primary X unreachable, using Y` — confirms reachability probe working
- `✅ Connected!` — heartbeat succeeded

### Step 3: Run the AP isolation diagnostic (if still stuck)

If BOTH laptops are on the same wifi but the bootstrap can't reach the
coordinator on ANY candidate, run the 30-second diagnostic from
`references/coordinator-lan-networking.md` Pitfall 5:

```bash
# On Evis (coordinator)
ipconfig getifaddr en0
lsof -nP -iTCP:8787 -sTCP:LISTEN

# On m4pro (peer)
ipconfig getifaddr en0
ping -c 3 -W 2 <evis-en0-ip>
```

If ping fails with 100% loss → AP client isolation. NOT a code bug.
Switch to phone hotspot or a wifi without isolation.

### Step 4: Push (once DNS recovers)

```bash
git push origin main  # or tranquil-flow depending on remote name
```

Currently +295 commits ahead of origin (DNS was down at session end).

---

## Related Work in This Session

### Prior commits (already committed before this handover)

- `3fff2d5` — fix(bootstrap): survive xet stalls + stop false-positive 'serving' status
  - HF_HUB_DISABLE_XET=1, Popen+idle-killer (300s no stdout = kill), lock cleanup
  - False-positive serving → error status when run_server exits without "Started"
- `b313a08` — docs: handover for same-wifi connectivity (the prior handover)
- `76914a6` — fix(bootstrap): bypass HTTP_PROXY + accept absolute-URI request lines

### Skill library updated

- `references/multi-ip-coordinator-offer.md` — NEW reference in the
  `distributed-inference-mvp` skill documenting this entire pattern

---

## Environment Notes for Next Agent

- **Container path**: `/workspace/Projects/distributed-inference-mvp/`
- **Mac path**: `~/Projects/distributed-inference-mvp/` (bind-mounted, same files)
- **Container has**: pyyaml installed (pip install), no `.venv`, no `hivemind`, no `torch`
- **Container cannot**: ping, ifconfig, lsof, access real Mac network interfaces
- **Tests**: Can't run full pytest (missing deps). Integration checks run via
  bare `python3` with `PYTHONPATH=.`
- **Git**: `moonsong <moonsong@tranquil.flow>`, branch `main`

---

## Uncommitted State

All 6 modified files are staged and ready to commit. No untracked files except
`.claude/` (editor metadata, not relevant).
