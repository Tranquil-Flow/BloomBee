# Distributed Inference MVP Finish Plan

**Current status:** `███████████████████░ 96%` — 95.6 / 100 weighted MVP-core points.

**Next gate:** `physical/self-serve showcase with fresh joined devices`.

**Claim boundary:** This document is a finish/runbook plan, not proof. MVP reaches 100% only after `physical_showcase_proof.py` passes with real cross-artifact evidence.

---

## What is actually blocked now?

There are two categories of blockers.

### MVP-core blocker: final physical/self-serve showcase

The remaining MVP-core blocker is not Qwen3-8B correctness. Qwen3-8B already passed:

- prescan
- one-block server
- multi-block server
- full-generation parity
- cache-generation parity
- multi-request load
- Pixel physical QR scan + Termux heartbeat subclaim

The final blocker is **cross-artifact alignment**:

1. A fresh capacity-reporting peer must join through the self-serve coordinator.
2. `join_layer_plan.py` must assign Qwen3-8B layers to that fresh peer.
3. Qwen3-8B cache-generation evidence must list `server_placements` matching that joined plan.
4. Qwen3-8B multi-request load evidence must pass for the same selected model/range.
5. Operator evidence must include physical QR scan, repeated heartbeats, and dashboard observation.
6. `physical_showcase_proof.py verify` must return `status: passed`.

The previous real Pixel QR run failed the strict verifier with:

```text
joined layer plan is not supported
joined layer plan does not cover every model layer
generation server placements do not match joined layer plan
load evidence did not pass multi_request_load gate
```

Root cause: the Pixel 8 Pro proved QR scan + heartbeat, but the Pixel Termux peer did not provide enough Qwen3-8B layer capacity to be the selected-model server host.

### Post-MVP / stretch blockers

These do **not** block MVP 100%, but they are tracked:

| Item | Status | Fix path |
|---|---:|---|
| `qwen35b_candidate` | blocked | Add/prove native `qwen3_5_moe` / `qwen3_5_moe_text` wrapper before selection. |
| `minimax_m3_candidate` | blocked | Needs LayerExecutor/quantized backend or native `minimax_m3_vl` + sparse-attention wrapper/kernels. |
| `continuous_batching` | pending | Build live batching proof harness after MVP correctness. |
| `kv_prefix_reuse` | pending | Build prefix-cache correctness/timing proof after cache-generation remains stable. |

---

## Important verifier finding

A local diagnostic bundle under `.local/mvp-finish-attempt-*` showed:

- The real previous Pixel QR evidence fails strict cross-artifact verification for the reasons above.
- A **hypothetical** fresh `m4pro-full` active roster + joined plan matching the already-proven Qwen3-8B generation/load evidence passes the strict verifier.

That diagnostic proves the remaining gap is the real-world capture/run protocol, not missing proof code.

Do **not** promote MVP status from the hypothetical pass. The real final run must capture a fresh same-session capacity peer and matching artifacts.

---

## Exact path to finish MVP-core

### Option A — strict same-session finish, preferred

Use this if we want the cleanest 100% claim.

1. Start a fresh coordinator and QR scan-capture server on m4pro.
2. User scans QR with Pixel; scan-capture event records hash match.
3. Pixel runs `join_client.py --count 3` for operator physical QR evidence.
4. m4pro also joins the same coordinator as a capacity-reporting peer with hostname/peer label matching the intended server placement, e.g. `m4pro-full`.
5. Build joined layer plan for `Qwen/Qwen3-8B`; it must assign `0:36` to `m4pro-full`.
6. Launch Qwen3-8B server from that plan with explicit HF cache:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src \
.venv/bin/python -m bloombee.cli.run_server Qwen/Qwen3-8B \
  --new_swarm \
  --block_indices 0:36 \
  --device mps \
  --torch_dtype float16 \
  --port 31359 \
  --throughput 1 \
  --num_handlers 2 \
  --inference_max_length 64 \
  --attn_cache_tokens 64 \
  --max_batch_size 64 \
  --max_chunk_size_bytes 16777216 \
  --cache_dir /Users/evinova-self/.cache/huggingface/hub
```

7. Rerun cache-generation parity with `server_placements` equal to the joined plan.
8. Rerun multi-request load with 3 direct-client requests.
9. Normalize load evidence to the nested `.verify` object if the committed file is a wrapper envelope.
10. Run strict verifier:

```bash
.venv/bin/python -m mvp_capabilities.physical_showcase_proof verify \
  --active-roster .local/<run>/active.json \
  --joined-layer-plan .local/<run>/joined-layer-plan.json \
  --generation-evidence .local/<run>/qwen3-8b-cache-generation.json \
  --load-evidence .local/<run>/qwen3-8b-load-verify.json \
  --operator-evidence .local/<run>/operator-evidence.json \
  --proof-status mvp_capabilities/PROOF_STATUS.yaml \
  --min-joined-peers 1 \
  --min-heartbeat-results 3 \
  --max-heartbeat-age-seconds 120 \
  --now <same-session-now>
```

Expected final pass:

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

Then update `mvp_status.py` / docs/tests from 96% to 100%, run focused tests, commit, and push.

### Option B — capacity-peer-only dry run, not enough for 100%

A fresh m4pro capacity heartbeat plus old Qwen3-8B evidence can demonstrate that the verifier contract is satisfiable, but it must remain a dry run unless generation/load are rerun from the fresh joined plan in the same session.

Use only for debugging the runbook, not for final MVP promotion.

---

## Fixes before the next same-session run

1. Harden `.local/scripts/start_physical_qr_run.sh` so m4pro QR rendering does not require remote `qrcode`/Pillow. Generate QR locally or embed a base64 PNG in a self-contained page.
2. Add a small m4pro self-join helper that:
   - creates a capability JSON with hostname `m4pro-full`,
   - reports current memory/free capacity,
   - posts 3 heartbeats to the same token/coordinator,
   - writes `active.json` and `joined-layer-plan.json`.
3. Add a final-showcase assembly helper that writes:
   - operator evidence from Pixel scan/heartbeats,
   - active roster,
   - joined layer plan,
   - cache-generation evidence,
   - load `.verify` evidence,
   - final strict verifier report.
4. Keep all raw tokens and raw join URLs in `.local/` only; committed evidence may include hashes and non-secret `server_response.ok` fields.

---

## Post-MVP improvements and current progress

Current committed post-MVP scope is in `docs/post-mvp-scope.md`.

| Workstream | Current progress | Next proof |
|---|---:|---|
| Qwen3-30B-A3B proof ladder | 35% | full-generation parity, then cache-generation, then multi-request load. |
| Qwen3-30B-A3B 2507 variants | scoped / pending | prescan + one-block proof per exact model ID. |
| Live chain scheduler | scoped / pending | live request artifact with `verified_chain_scheduler_live_request_evidence`. |
| Continuous batching | pending | concurrent request proof with correctness + throughput telemetry. |
| KV prefix reuse | pending | exact-token/logit parity plus timing delta against no-reuse baseline. |
| Phone draft-provider wall-clock | partial | live phone token transport into verifier plus faster-than-baseline wall-clock gate. |
| Android/Termux capability fidelity | partial | richer peer scan memory/storage reporting without block-serving overclaim. |
| qwen3_5_moe / AgentWorld-35B | blocked | wrapper feasibility + one-block proof. |
| LayerExecutor / quantized frontier backends | research | bounded feasibility spike for MiniMax M3, GLM-5.2, DeepSeek-V4-Flash, Kimi/giant Qwen-Coder. |
| Dashboard/status separation | scoped | post-MVP panel that cannot move MVP-core percent. |

---

## Current recommended next action

Run Option A as one fresh same-session proof. The implementation work left is mostly orchestration and evidence assembly, not new model code.

If the QR display opens without visible QR again, use the already-proven fallback: convert QR to RGB and open raw PNG or embed it as `data:image/png;base64,...` in a self-contained page.
