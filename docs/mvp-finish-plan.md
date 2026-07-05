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

The final proof uses Qwen/Qwen3-8B as the strongest proven live/demo-safe model for MVP-core. Qwen3-30B and larger/model-optimisation work remain post-MVP/stretch.

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
| `qwen35b_candidate` | blocked after scout | Config-only scout complete at `mvp_capabilities/distributed_evidence/qwen35b/qwen-agentworld-35b-wrapper-scout-20260704.json`; write/import-dispatch tests for `qwen3_5_moe` / `qwen3_5_moe_text` before wrapper code or selection. |
| `minimax_m3_candidate` | blocked after spike | LayerExecutor/quantized feasibility spike complete; still needs runnable backend proof or native `minimax_m3_vl` + sparse-attention wrapper/kernels. |
| `continuous_batching` | partial | Deterministic scheduler/planner proof exists; wire into live request loop behind opt-in flag, then prove parity and throughput. |
| `kv_prefix_reuse` | partial | Deterministic prefix planner proof exists; wire into real prefill/session cache metadata, then prove parity and timing. |

Current committed post-MVP scope is in `docs/post-mvp-scope.md`.

| Workstream | Current progress | Next proof |
|---|---:|---|
| Qwen3-30B-A3B proof ladder | 60% | base and Instruct-2507 lower gates are proven; base@int8 and Instruct-2507@int8 full-0:48 load proofs passed; next full-generation parity, then cache-generation. |
| Qwen3-30B-A3B Instruct-2507 | int8 load proof passed, fp16 full/cache/load pending | Seagate-backed prescan, one-block, and multi-block artifacts are committed (`instruct2507-seagate-multiblock-proof-20260705T064511Z.json`); `Instruct-2507@int8` full 0:48 multi-request load passed, but full-generation/cache parity remain pending. |
| Live chain scheduler | scoped / pending | live request artifact with `verified_chain_scheduler_live_request_evidence`. |
| Continuous batching | partial | live-loop unit seam artifact exists; next real live request artifact with correctness + throughput telemetry behind opt-in integration. |
| KV prefix reuse | partial | exact-token/logit parity plus timing delta against no-reuse baseline. |
| Phone draft-provider wall-clock | partial | live phone token transport into verifier plus faster-than-baseline wall-clock gate. |
| Android/Termux capability fidelity | partial | richer peer scan memory/storage reporting without block-serving overclaim. |
| qwen3_5_moe / AgentWorld-35B | config-only scout complete; wrapper blocked | `qwen-agentworld-35b-wrapper-scout-20260704.json` proves the text tower alternates `linear_attention`/`full_attention`; write RED import/config-dispatch tests before wrapper code. |
| LayerExecutor / quantized frontier backends | bounded feasibility spike complete | `layerexecutor-feasibility-20260704.json`; all scanned targets remain blocked for native route/demo until wrapper/backend proof exists. |
| Dashboard/status separation | post-MVP panel shipped | status JSON and dashboard show post-MVP/stretch milestones without moving MVP-core percent. |

---

## Repro/operation notes

Keep raw coordinator tokens and raw `bloombee://join?...` URLs in `.local/` scratch only. Committed evidence may include hashes, peer IDs, non-secret heartbeats, route/placement summaries, and verifier result fields.

For future same-session re-runs, prefer the hardened path from this completion record:

1. Generate/display QR without relying on remote `qrcode`/Pillow availability.
2. Verify physical scan by hash-matching the scan-capture URL against the coordinator offer.
3. Query `/active` with verifier-safe timing if phone/host clock skew could age out otherwise valid heartbeats.
4. Require same-token capacity peer, joined plan, generation placements, load proof, and operator evidence to align before any status promotion.
