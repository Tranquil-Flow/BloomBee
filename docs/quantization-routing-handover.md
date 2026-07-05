# Quantization + Best-Model/Override Routing Handover

> **For the build agent:** Fable did the research spikes, the hard modules,
> and the routing-contract design in this handover. **Update 2026-07-05:**
> Tasks 2, 3, 4 and 6 are DONE and committed with tests (server-side
> quantized loading, quant-aware route memory math, quantized proof rows +
> token-parity policy, packed int4 experts); Task 1 (TinyLlama load gate) was
> closed by Moonsong. Task 5 is partially executed on m4pro: base 30B@int8
> one-block, 0:2 multi-block, and full 0:48 multi-request load passed; exact
> Instruct-2507@int8 also has a full 0:48 multi-request load proof. Full/cache
> generation and token parity remain fail-closed pending for both quantized
> rows. Task 7 coordinator/dashboard wiring is done: `/route` and `/handoff`
> accept quantized `requested_model` pins, preserve override/refusal metadata,
> and dashboard route cards show serving model + quantization. What remains is
> the parity/reference problem for quantized 30B demo-safe promotion.
> Work task-by-task, RED tests first where specified, commit after each task. Do
> not move the MVP-core denominator; everything here is post-MVP.

**Branch state:** committed on `main` through
`808b21f feat(quant): surface quantized route pins in coordinator dashboard`;
current working slice closes Task 8 docs/status coherence. Baseline suite before
this slice: `487 passed, 23 skipped, 4 warnings`.

**Related lane built by Moonsong meanwhile:**
`mvp_capabilities/quantized_route_lane.py` (+ its evidence artifact and
tests) — a claim-bounded planning report for `Qwen/Qwen3-30B-A3B@int8` that
encoded the `model_id@int8` proof-row keying and int8 memory math before Fable
generalized it into `route_picker`.

---

## 1. What is already researched, built, and proven

### 1.1 The quantization wall, located precisely

- BloomBee removed bitsandbytes. `QuantType INT8/NF4` only feeds FlexGen
  group-wise compression (`src/bloombee/flexgen_utils/compression.py`), which
  lives inside FlexGen's `TorchTensor` infrastructure and **only applies to
  the FlexGen-native LLaMA path**. Standard HF blocks (qwen3, qwen3_moe,
  falcon, mixtral) load unquantized. `convert_block.quantize_module` is an
  explicit no-op.
- transformers 5.x stores Qwen3-MoE experts as **fused 3D `nn.Parameter`
  tensors** on `Qwen3MoeExperts` (`gate_up_proj [128, 1536, 2048]`,
  `down_proj [128, 2048, 768]`), NOT per-expert `nn.Linear`. Off-the-shelf
  weight-only quantizers (optimum-quanto, torchao) walk `nn.Linear` and skip
  ~97% of a Qwen3-30B-A3B block's bytes. Measured: quanto-only on the real-dim
  30B block compresses **1.015×** (attention only).

### 1.2 Spike evidence (all runs passed, CPU + MPS)

Artifact: `mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json`
Script: `scripts/quantized_block_spike.py` (deterministic, seeded, JSON out)

| Block | Mode | Compression | Cosine vs fp16 | Backward-to-input grad |
|---|---|---:|---:|---|
| TinyLlama layer0 (REAL weights) | quanto qint8 | 1.998× | ≥ 0.9999972 | finite |
| TinyLlama layer0 (REAL weights) | quanto qint4 | 3.764× | ≥ 0.9999957 | finite |
| Qwen3-30B-A3B MoE layer (real dims, random weights) | quanto qint8 only | 1.015× | — (proves the gap) | finite |
| Qwen3-30B-A3B MoE layer (real dims, random weights) | **custom int8 experts + quanto attn** | **1.996×** | ≥ 0.9999982 | finite |
| Qwen3-30B-A3B MoE layer | quanto qint4 (attn only) | 1.023× | — (proves int4-experts gap) | finite |

Caveat recorded in the artifact: MoE parity on random ~N(0, 0.02) weights is
weak evidence (expert contributions are unrealistically small vs the
residual). Real-weight parity is a REQUIRED gate before any serving claim
(Task 5).

### 1.3 The hard module: fused-expert int8 (BUILT, TESTED)

- `src/bloombee/utils/moe_expert_quant.py`
  - `QuantizedQwen3MoeExperts` — drop-in replacement for `Qwen3MoeExperts`;
    int8 storage + per-output-channel fp16 scales as registered buffers;
    forward mirrors upstream exactly (same routing mask + `index_add_`), and
    **dequantizes only router-hit experts** (top-8 of 128 at decode).
  - `quantize_qwen3_moe_block_experts(block)` — swaps every fused-3D
    `*.experts` module under a block, returns byte stats, and **fails closed**
    (raises) if no fused experts found, so a wrong architecture can never
    silently report itself quantized.
- `tests/test_moe_expert_quant.py` — 6 default-suite tests (no network):
  round-trip error bound, parity vs reference (cosine > 0.999), backward
  gradient finiteness, swap-helper stats + fail-closed, state_dict roundtrip.
- Gotcha discovered: standalone HF layer construction uses `torch.empty` and
  never initializes → NaN garbage weights. Any harness constructing bare
  blocks MUST init explicitly (see `_tiny_sparse_block` in the test file).

### 1.4 Memory math this unlocks (why we care)

Qwen3-30B-A3B: 48 layers ≈ 61 GB fp16 (m4pro has ~37 GB free → blocked today).

- int8 (built, servable via `--quant_type INT8`): ≈ **30.5 GB** → 30B fits on
  m4pro alone; fp16 KV cache is unchanged and small at demo batch sizes.
- int4 experts (built, servable via `--quant_type NF4` on qwen3_moe): ≈
  **16 GB** → 30B on a 24 GB-class device. Dense int4 stays blocked.

### 1.5 Routing: strict demo_safe + detect-best/route-to-choice (BUILT, TESTED)

- `route_picker.DEMO_SAFE_GATES = (full_generation, cache_generation,
  multi_request_load)` — `demo_safe` now requires ALL three (was:
  full_generation alone, which violated the project guardrail; TinyLlama was
  demo_safe with load unproven). `proof_ladder._claim_level` aligned.
  `PROOF_STATUS.yaml` header updated.
  - Consequence: TinyLlama is now `experimental` (its `multi_request_load` is
    `pending`). See Task 1.
- `route_picker.route_report()` + CLI `--report`: always computes
  `best_available` (what we COULD serve), evaluates the operator pin
  (`--model`), serves the pin only when the selector mode allows it,
  otherwise `override_refused` + reason and picks the best allowed model.
  `override_active` marks "serving X, auto-pick would be Y". Verified live:
  safe-demo pin of unproven 30B → refused, picked Qwen3-8B; exit code 1 on
  refusal.
- New tests in `tests/test_mvp_capabilities.py`: full-generation-only is NOT
  demo_safe; report detects best + serves allowed pin; disallowed pin refused
  fail-closed.

### 1.6 Server-side quantized loading (BUILT, TESTED — was Task 2)

`convert_block` no longer treats `quant_type` as a FlexGen-only passenger.
`bloombee.utils.convert_block.quantize_hf_block(block, quant_type=..., model_type=...)`
implements the mapping and `convert_block()` calls it for any HF block with
real (non-meta) weights; FlexGen-native LLaMA blocks are untouched.

| quant_type | dense HF block | qwen3_moe block |
|---|---|---|
| `INT8` | quanto qint8 on all Linears | int8 expert swap + qint8 Linears |
| `NF4` | **fail-closed** `NotImplementedError` | packed-int4 expert swap + qint8 Linears |
| `NONE` | recorded no-op (`applied: false`) | recorded no-op |

Decisions baked in (do not silently change):

- **Router stays fp16.** Expert selection is an argmax over router logits;
  quantizing it risks flipped expert picks and breaks the exact-token-parity
  policy. transformers 5.x's `Qwen3MoeTopKRouter` is not an `nn.Linear` so
  quanto skips it anyway; the `exclude=["*gate"]` guard covers older
  versions where the router was a plain Linear.
- **NF4 attention stays qint8**, not qint4: experts are ~97% of block bytes,
  and quanto qint4 needs a JIT-built C++/MPS extension we refuse to depend on
  in the serving path.
- **Quantization + per-parameter CPU offload fails closed** (ValueError): the
  offload path moves `param.data` and would corrupt packed tensors. The block
  is ~2–3.8× smaller quantized, so offload need drops anyway.
- `get_block_size(..., location="memory", quant_type=INT8/NF4)` now returns
  compressed byte estimates (mirrors the load mapping incl. fp16 router/norm
  exclusions) instead of raising — server memory planning and throughput math
  no longer assume fp16. `get_dtype_name` reports "quantized to int8/nf4".

Tests: `tests/test_quantized_block_loading.py` (9 tests, default suite).
`server_info.quant_type` already advertises the mode (`server.py:414`) — no
change was needed there.

### 1.7 Packed int4 experts (BUILT, TESTED — was Task 6)

`moe_expert_quant.py` now has the int4 lane:

- `quantize_group_int4` / `dequant_group_int4` — symmetric int4, two nibbles
  per byte along the input dim, group-wise fp16 scales (group=128). Partial
  last groups get their own scale over the actual elements; odd input dims
  pad a single zero nibble. Pure torch ops, CPU+MPS safe.
- `QuantizedQwen3MoeExpertsInt4` — same routing forward as the int8 class
  (shared `_RoutedExpertsForwardMixin`, mirrors upstream exactly).
- `quantize_qwen3_moe_block_experts(block, scheme="int4")` — same fail-closed
  swap helper; unknown schemes raise.

Measured on the test fixtures: expert-tensor compression **≥3.5×** vs fp16 at
group-amortized dims (analytically 3.88× at real 30B dims → experts ~16 GB).
The packed forward is **bit-exact** vs running the reference module on
`dequant(pack(w))` weights — that is the strong regression property; any
packing/scale-layout/routing bug breaks it, inherent rounding noise cannot.

Honest calibration note: at the residual-free sparse-MoE-module level on tiny
random weights, int4-vs-fp16 cosine is ~0.985 (int4 rounding is ~18× coarser
than int8). This is expected, not a bug; whole-decoder-layer parity with
residual is the meaningful quality gate and belongs to the ladder (Tasks 4/5),
where exact greedy token-ID parity remains the demo_safe bar.

### 1.8 Environment facts you will need

- `optimum-quanto` is now declared as the `quant` extra in both `setup.cfg`
  and `setup.py` (`pip install -e .[quant]` / `uv pip install -e .[quant]`);
  the serving path fails closed with a clear error if quantization is requested
  without it. `torchao==0.17.0` and `ninja` are still only in `.venv`: torchao is unused (drop it unless picked
  up deliberately) and ninja is only needed to rebuild the quanto qint4
  extension for spikes — the serving path deliberately never uses qint4.
- quanto qint4 lazily builds C++/MPS extensions via torch cpp_extension and
  needs `ninja` **on PATH as an absolute entry** (subprocess lookup):
  `PATH="$PWD/.venv/bin:$PATH"`. First build is slow; cached afterwards under
  the quanto package dir. Without ninja the failure is a confusing
  dlopen/"Ninja is required" error.
- Fleet: m4pro ≈ 37 GB free / 48 GB; MacBook 16 GB (OOMs easily — run heavy
  proofs on m4pro via `ssh m4pro`). Base 30B weights are already cached on
  m4pro (~61 GB, snapshot ad44e777).

---

## 2. Build tasks, in order

### Task 1: restore TinyLlama demo-safe fallback — **DONE (Moonsong)**
`PROOF_STATUS.yaml` TinyLlama `multi_request_load: passed`; the safe-demo
fallback ladder has TinyLlama and Qwen3-8B again.

### Task 2: server-side quantized loading for HF blocks — **DONE (Fable, 2026-07-05)**
See §1.6. Commit `587e27c`; tests in `tests/test_quantized_block_loading.py`.
One intentional deviation from the original spec: `NF4` on qwen3_moe blocks
is **not** blocked anymore — it maps to packed-int4 experts + qint8 Linears
(Task 6 landed first). `NF4` on dense blocks remains fail-closed.

### Task 3: quantization-aware route memory math — **DONE (Fable, 2026-07-05)**
Commit `beaf477`; tests in `tests/test_quantized_proof_routing.py`.
`route_picker.derive_quantized_variant` / `expand_quantized_variants` build
`model_id@int8` (fp16 required ÷ 2.0, any supported family) and `@nf4`
(÷ 3.5, qwen3_moe only — dense nf4 variants are *blocked*, matching the
fail-closed loader) candidates; registry entries may override with
`int8_min_free_mem_gb` / `nf4_min_free_mem_gb`. Variants rank just below
their fp16 base (fp16 preferred at equal fit, int8 over nf4) and are
first-class in `route_report` candidates and pins by default
(`include_quantized=True`; `choose_best_route`/`explain_route` keep it
opt-in for backward compatibility). Verified: fp16 30B does not fit an
m4pro-class peer (70 GB) but `@int8` does (35 GB ≤ 37 GB free).
Follow-up idea (not blocking): derive required bytes from
`get_block_size(..., quant_type=...)` instead of divisors once route
planning has model configs at hand.

### Task 4: proof gates for quantized serving — **DONE (Fable, 2026-07-05)**
Commit `beaf477`. The policy is now code, centralized in
`model_compat_scan`: `split_route_id`, `is_demo_safe(status, quant_type=...)`,
`DEMO_SAFE_GATES`, `TOKEN_PARITY_KEY`. Rules enforced everywhere
(`route_picker`, `proof_ladder`, `model_compat_scan` — whose own weak
full_generation-only gate is also fixed — and `quantized_route_lane`):

- proof rows keyed `model_id@int8` / `@nf4`; plain id == fp16; absent
  quantized row == all pending (never inherits fp16 gates)
- quantized `demo_safe` = all three gates passed **AND** `token_parity: exact`
  in the row; `diverged`/absent parity caps below demo_safe, no exceptions
- unknown `@suffix` is not a quant marker — a typo becomes an unknown
  (all-pending) row, never an alias of the fp16 row
- `PROOF_STATUS.yaml` documents the policy and carries the explicit
  `Qwen/Qwen3-30B-A3B@int8` row. After the m4pro Task 5 partial run, that row
  has only the proven int8 gates marked passed (`prescan`, `one_block_server`,
  `multi_block`, `multi_request_load`); `full_generation`, `cache_generation`,
  and exact `token_parity` remain pending/fail-closed.
- `proof_ladder` reports `quant_type`/`token_parity` and names
  `token_parity` as the next gate when only parity is missing;
  `Qwen/Qwen3-30B-A3B@int8` is in the fallback ladder

**For the ladder runner (Task 5):** when the quantized full/cache gates pass,
write `token_parity: exact` or `token_parity: diverged` into the `@int8` row
alongside the gate results, and record first-divergence position + text
sample in the evidence artifact on divergence.

### Task 5: Qwen3-30B-A3B int8 ladder on m4pro — **PARTIAL (Moonsong, 2026-07-05)**
The serving path is real now: `--quant_type INT8` on a qwen3_moe server
actually quantizes blocks at load (§1.6). Base 30B weights are cached on the
m4pro Seagate snapshot.

Verified on m4pro from clean proof tree `7830e76` for base 30B, then clean
proof tree `3d915db` for exact Instruct-2507:

- `qwen30b-int8-oneblock-20260705T131443Z.json` — one-block INT8 server +
  direct RPC forward/backward passed; block compressed 1246.2 MB → 624.3 MB
  (1.996×); claim boundary `qwen30b_int8_one_block_server_direct_rpc_only`.
- `qwen30b-int8-multiblock-0-2-20260705T131529Z.json` — `0:2` INT8
  multi-block direct RPC forward/backward passed; two blocks quantized.
- `qwen30b-int8-full-load-0-48-20260705T131803Z.json` — full `0:48` INT8
  server loaded all 48 blocks and passed `multi_request_load_proof.py verify`
  for 3 seeded/scaled direct-client requests. Forward latencies: avg 2.645s;
  backward avg 4.566s; 0 failures.
- `instruct2507-int8-full-load-0-48-20260705T133853Z.json` — exact
  `Qwen/Qwen3-30B-A3B-Instruct-2507@int8` full `0:48` INT8 server loaded all
  48 blocks from the complete Seagate cache and passed the same 3-request load
  verifier. Forward latencies: avg 2.467s; backward avg 4.544s; 0 failures.

Proof row updates are quantized-row-only:
`Qwen/Qwen3-30B-A3B@int8` and
`Qwen/Qwen3-30B-A3B-Instruct-2507@int8` have `prescan`,
`one_block_server`, `multi_block`, and `multi_request_load` marked `passed`.
`full_generation` and `cache_generation` stay `pending`, and
`token_parity: not_evaluated_reference_fp16_exceeds_m4pro_memory` keeps both
rows below demo_safe.

Remaining Task 5 work: exact full/cache generation parity. The current
`text_generation_parity.py` verifier loads a full local fp16 HF reference
while the distributed server is live. The fp16 30B requirement (route planner:
70 GB class) exceeds m4pro's ~48 GB unified memory / ~37 GB free, so this
must remain fail-closed until there is a credible reference path (for example a
larger host, a separately recorded fp16 reference trace, or a test-backed
streaming/offloaded reference mode that still represents the fp16 baseline). Do
not write `token_parity: exact` or mark `full_generation`/`cache_generation`
passed from the load-only artifacts above.

### Task 6: int4 expert packing — **DONE (Fable, 2026-07-05)**
See §1.7. Commit `78a152a`; tests in `tests/test_moe_expert_quant.py` (15
total). The NF4 mapping in Task 2 is already unblocked for qwen3_moe blocks.
One spec correction learned doing it: the original "cosine > 0.999 on the
tiny fixture" bar is not achievable for int4 at the residual-free module
level (~0.985 is inherent rounding noise); the enforced properties are the
elementwise round-trip bound and bit-exact packed-forward-vs-dequantized-
weights equality. Real-weight decoder-layer parity remains the Task 4/5 gate.

### Task 7: wire route_report into coordinator + dashboard — **DONE (Moonsong, 2026-07-05)**
- `join_http_server` `/route` (and `/handoff`'s embedded route decision) now
  accept both `requested_model` and legacy `model` pins plus `selector_mode`,
  and return the backward-compatible `route_report` payload with
  `best_available`, `serving`/`picked`, `requested_evaluation`,
  `override_active`, `override_refused`, and refusal reason.
- Quantized route ids like `Qwen/Qwen3-30B-A3B@int8` are first-class pins. In
  `safe-demo`, partially proven int8 rows are evaluated and refused with their
  quantized proof status visible; the served fallback remains demo-safe. In
  planning mode, `/handoff` can plan the quantized route id while launch
  commands use the base HF model id plus `--quant_type INT8`.
- `demo_dashboard.py` route cards render the actual serving model and
  `quant_type`/`fp16 / none`, preserving override/refusal visibility so a
  quantized route is never shown without its quant marker.
- Regression coverage: `test_layer_planner_quantized_route_uses_base_model_and_quant_flag`,
  `test_join_http_server_route_requested_model_alias_handles_quantized_safe_demo_pin`,
  `test_join_http_server_handoff_surfaces_quantized_requested_model_refusal`,
  `test_join_http_server_handoff_planning_quantized_pin_emits_quantized_launch_command`,
  and `test_dashboard_route_card_surfaces_serving_quantization_and_refused_pin`.


### Task 8: docs + status coherence — **DONE (Moonsong, 2026-07-05)**
- `docs/distributed-inference-mvp.md` now states the current safe-demo ladder
  precisely: TinyLlama and Qwen3-8B are proven fallbacks, while Qwen3-30B-A3B,
  Instruct-2507, and their int8 route IDs remain post-MVP/experimental until
  full-generation, cache-generation, and exact token-parity gates pass for the
  exact route row.
- `mvp_status.py` has the post-MVP quantization milestone updated to
  `int8_load_proven_route_dashboard_wired` with evidence for base +
  Instruct-2507 int8 load proofs, coordinator/dashboard route-pin handling,
  quantized launch commands, and fail-closed full/cache/token-parity blockers.
- Docs coherence is guarded by
  `tests/test_mvp_capabilities.py::test_docs_post_mvp_status_rows_match_completed_scouts`
  plus the status/dashboard assertions that now reject the stale
  `route_lane_committed` state.


---

## 3. Related must-fixes from Fable's earlier review (separate lane, small)

1. `physical_showcase_proof.py`: record verifier params (`now`,
   `max_heartbeat_age_seconds`, minimums) in the report; assert
   `token_sha256` consistency across offer/heartbeats/operator evidence.
2. `_load_report`/`multi_request_load_proof.py`: require + bound
   seed/input_scale metadata per request row for Qwen-class load artifacts.
3. `join_http_server`: stop honoring client-supplied `now` in `/heartbeat`
   (server clock only, or test-only flag); enforce offer TTL.
4. Default-suite redaction test over
   `mvp_capabilities/distributed_evidence/**.json`: forbid `bloombee://join`,
   `token=`, raw `"token":` (allow `*_sha256` and
   `raw_join_url_recorded_in_scratch_only`).

## 4. Guardrails (non-negotiable)

- MVP-core stays 100%; nothing here touches its denominator.
- No `demo_safe` for any quantized route without exact greedy token-ID parity
  + cache parity + load proof for that exact `(model_id, quant_type)`.
- Quantized evidence artifacts always record the quant scheme; never let a
  quantized proof update the fp16 proof row.
- Fail-closed everywhere: unknown architecture → refuse to quantize (the
  expert-swap helper already raises); refused pin → never served.
- No raw tokens/join URLs in committed artifacts.

## 5. Verification commands

```bash
source .venv/bin/activate
# current baseline before the m4pro int8 evidence/status slice
.venv/bin/python -m pytest -q                                        # 481 passed, 23 skipped, 4 warnings
.venv/bin/python -m pytest tests/test_moe_expert_quant.py -q         # 15 passed (int8 + int4)
.venv/bin/python -m pytest tests/test_quantized_block_loading.py -q  # 10 passed (convert_block/get_block_size/freeze)
.venv/bin/python -m pytest tests/test_quantized_proof_routing.py -q  # 11 passed (proof rows + quant routing + int8 artifacts)
# quantized pin end-to-end (planning serves it; safe-demo still refuses until full/cache + exact token parity)
.venv/bin/python -m mvp_capabilities.route_picker --report --selector-mode planning \
  --model Qwen/Qwen3-30B-A3B@int8 --synthetic-m4-laptops 2 \
  --synthetic-total-gb 48 --synthetic-free-gb 37
.venv/bin/python -m mvp_capabilities.proof_ladder --model Qwen/Qwen3-30B-A3B@int8
# spike re-run (slow; MoE block needs ~4GB free)
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/quantized_block_spike.py --devices cpu --skip-moe
# routing contract
.venv/bin/python -m mvp_capabilities.route_picker --report --selector-mode safe-demo \
  --model Qwen/Qwen3-30B-A3B --synthetic-m4-laptops 10   # override_refused=true, picked=Qwen3-8B
# quantized 30B planning lane (Moonsong)
.venv/bin/python -m mvp_capabilities.quantized_route_lane
.venv/bin/python -m json.tool mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json >/dev/null
```
