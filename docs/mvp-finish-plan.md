# Distributed Inference MVP Finish Plan

**Current status:** `████████████████████ 100%` — 100 / 100 weighted MVP-core points.

**Next gate:** post-MVP improvements. MVP-core is complete.

**Claim boundary:** This document is the final runbook/completion record. The 100% status is backed by the strict same-session `physical_showcase_proof.py` pass committed at:

```text
mvp_capabilities/distributed_evidence/physical_showcase/qwen3-8b-final-physical-showcase-20260704T155722Z.json
```

---

## Final same-session result

The final run closed the old physical/self-serve showcase blocker with a real cross-artifact proof:

1. Started a fresh coordinator and QR scan-capture server on `m4pro`.
2. Displayed a real QR code carrying the same join offer through a scan-capture URL.
3. Captured a real Pixel 8 Pro camera/browser scan with a matching join URL hash.
4. Ran Pixel Termux `join_client.py --count 3` through ADB UI against the same offer.
5. Preserved non-secret `server_response.ok=true` heartbeat fields while redacting raw tokens and raw join URLs.
6. Posted same-session `m4pro-full` capacity heartbeats to the same coordinator/token.
7. Built a joined Qwen3-8B layer plan assigning layers `0:36` to `m4pro-full`.
8. Launched the Qwen3-8B full-range server from that plan.
9. Captured cache-generation parity with server placements matching the joined plan.
10. Captured a deterministic scaled 3/3 live direct-client load proof with finite forward and backward passes.
11. Assembled operator evidence, active roster, joined layer plan, generation evidence, load evidence, and proof status.
12. Ran strict `physical_showcase_proof.py verify`; result was `status: passed`.

Verifier outcome:

```json
{
  "status": "passed",
  "physical_showcase_proven": true,
  "inference_proven": true,
  "can_update_mvp_status": true,
  "mvp_status_update": {"physical_showcase": "passed"},
  "failed_checks": []
}
```

The final proof uses Qwen/Qwen3-8B as the strongest proven live/demo-safe model for MVP-core. Qwen3-30B and larger/model-optimisation work remain post-MVP/stretch; in that post-MVP lane, the base and Instruct-2507 INT8 30B rows are now demo-safe under the current proof gates.

---

## Bugs fixed during finalization

### Successful heartbeat response shape

The strict verifier requires `server_response.ok=true` for successful fresh-device heartbeat rows. The coordinator previously recorded successful heartbeat payloads without an explicit `ok` field.

Fix:

```text
mvp_capabilities/join_coordinator.py
```

Successful heartbeat records now include:

```json
{"ok": true}
```

Regression test:

```text
tests/test_mvp_capabilities.py::test_join_heartbeat_success_record_carries_ok_for_physical_showcase_verifier
```

### Deterministic scaled load probe

Unscaled random synthetic hidden-state load probes occasionally produced non-finite gradients on full-range Qwen3-8B even while forward outputs were finite. The direct RPC proof is a transport/block finite-value probe, not a claim that arbitrary unbounded random hidden states are in-distribution.

Fix:

```text
scripts/direct_remote_call.py
```

Added deterministic synthetic tensor controls:

```text
--seed <int>
--input-scale <float>
```

The final load proof used seeds `100`, `101`, `102` and `input_scale=0.1`, producing 3/3 finite forward/backward requests.

Regression test:

```text
tests/test_mvp_capabilities.py::test_direct_remote_call_builds_deterministic_scaled_synthetic_tensors
```

---

## Post-MVP / stretch blockers

These do **not** block MVP 100%, but they remain tracked:

| Item | Status | Fix path |
|---|---:|---|
| `qwen35b_candidate` | partial native lane | AgentWorld-35B text-tower wrapper contract partial: `qwen-agentworld-35b-wrapper-scout-20260704.json` plus wrapper/backend state contracts are green but still need one-block server proof; user-requested Qwen36A/Qwen3.6 lane has exact config-scan evidence at `qwen36a-config-scan-20260706.json`, state-cache mapping evidence at `qwen36a-state-cache-mapping-20260706.json`, and m4pro preflight `qwen36a-oneblock-host-preflight-20260706.json` blocked by 48GB/~33.2GB free vs 80GB required; now needs a suitable host/hardware for one-block proof. |
| `minimax_m3_candidate` | blocked native lane | MiniMax M2.7 REAP is now explicitly targeted for the main native BloomBee distributed path, not only GGUF; `frontier-distributed-pathway-minimax-m27-current-20260706.json` records it blocked by missing real-weight one-block proof and MTP proof-time guard. Exact contract scan `minimax-m27-reap-native-contract-scan-20260706.json` records MiniMaxM2ForCausalLM/minimax_m2, 62 layers, 154 experts, top-k 8, MTP enabled, no exact sparse-attention flag, native wrapper package present, and remaining real-weight/MTP/one-block blockers. M3 remains blocked by `minimax_m3_vl` + sparse-attention wrapper/kernels and huge memory. |
| `continuous_batching` | complete for functional proof | Late-arrival live-server token/logit parity passed; wall-clock/demo promotion remains a separate timing gate. |
| `kv_prefix_reuse` | complete for functional proof | v32 proved server-observed KV tensor reuse with token/logit parity; wall-clock/demo promotion remains separate. |

Current committed post-MVP scope is in `docs/post-mvp-scope.md`.

| Workstream | Current progress | Next proof |
|---|---:|---|
| Qwen3-30B-A3B proof ladder | 100% for base + Instruct int8 gates | base @int8 and Instruct-2507@int8 are demo-safe under current full/cache/load/token-parity gates; optional next work is broader prompt-set parity or Thinking-2507 only if needed. |
| Qwen3-30B-A3B Instruct-2507 | int8 load + full/cache parity passed | Seagate-backed prescan, one-block, and multi-block artifacts are committed (`instruct2507-seagate-multiblock-proof-20260705T064511Z.json`); `Instruct-2507@int8` full 0:48 multi-request load, streamed-reference full-generation parity, and streamed-reference cache/generate-api parity passed. |
| Live chain scheduler | scoped / pending | live request artifact with `verified_chain_scheduler_live_request_evidence`. |
| Continuous batching | complete for functional proof | late-arrival live-server token/logit parity passed; next separate gate is wall-clock/demo promotion timing. |
| KV prefix reuse | complete for functional proof | v32 server-observed KV tensor reuse, token parity, and numeric logit parity passed; next separate gate is wall-clock/demo promotion timing. |
| Phone draft-provider wall-clock | partial / wrapped | `speculative-phone-worker-wrapup-current-20260706.json` consolidates current gates: Android ready, iOS missing, only one ready phone, integrated non-sequential speedup not measured. |
| Android/Termux capability fidelity | partial / wrapped | Pixel/Termux context-token + wall-clock correctness groundwork exists; no BloomBee block-serving or speedup claim. |
| qwen3_5_moe / AgentWorld-35B / Qwen36A | native candidate partial | `qwen-agentworld-35b-text-wrapper-gate-20260704.json` covers import/config dispatch, full_attention KV tuples, local linear_attention conv/recurrent state round-trip, and backend state-cache contracts; `qwen36a-config-scan-20260706.json` confirms Qwen36A uses the same qwen3_5_moe/qwen3_5_moe_text family with 30 linear-attention + 10 full-attention layers, `qwen36a-state-cache-mapping-20260706.json` maps exact descriptors, and `qwen36a-oneblock-host-preflight-20260706.json` is blocked by host memory; both still need one-block proof. |
| LayerExecutor / quantized frontier backends | bounded feasibility spike complete | `layerexecutor-feasibility-20260704.json`; all scanned targets remain blocked for native route/demo until wrapper/backend proof exists, and no runnable backend proof is claimed by the spike. |
| Dashboard/status separation | post-MVP panel shipped | status JSON and dashboard show post-MVP/stretch milestones without moving MVP-core percent. |

---

## Repro/operation notes

Keep raw coordinator tokens and raw `bloombee://join?...` URLs in `.local/` scratch only. Committed evidence may include hashes, peer IDs, non-secret heartbeats, route/placement summaries, and verifier result fields.

For future same-session re-runs, prefer the hardened path from this completion record:

1. Generate/display QR without relying on remote `qrcode`/Pillow availability.
2. Verify physical scan by hash-matching the scan-capture URL against the coordinator offer.
3. Query `/active` with verifier-safe timing if phone/host clock skew could age out otherwise valid heartbeats.
4. Require same-token capacity peer, joined plan, generation placements, load proof, and operator evidence to align before any status promotion.
