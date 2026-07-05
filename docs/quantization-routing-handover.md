# Quantization + Best-Model/Override Routing Handover

> **For the build agent:** Fable did the research spikes, the hard modules,
> and the routing-contract design in this handover. **Update 2026-07-05:**
> Fable also completed the two hardest build tasks — server-side quantized
> loading (old Task 2) and packed int4 expert quantization (old Task 6). Both
> are committed with tests. Your job is now proof-ladder execution and the
> remaining integration tasks below. Work task-by-task, RED tests first where
> specified, commit after each task. Do not move the MVP-core denominator;
> everything here is post-MVP.

**Branch state:** committed on `main` through
`587e27c feat(quant): wire server-side quantized loading for HF blocks`.
Baseline suite: `466 passed, 23 skipped`.

**Related lane built by Moonsong meanwhile:**
`mvp_capabilities/quantized_route_lane.py` (+ its evidence artifact and
tests) — a claim-bounded planning report for `Qwen/Qwen3-30B-A3B@int8` that
already encodes the `model_id@int8` proof-row keying and int8 memory math.
Task 3 below should generalize that lane into `route_picker` rather than
duplicating it.

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

- `optimum-quanto` is now declared as the `quant` extra in `setup.cfg`
  (`pip install -e .[quant]`); the serving path fails closed with a clear
  error if quantization is requested without it. `torchao==0.17.0` and
  `ninja` are still only in `.venv`: torchao is unused (drop it unless picked
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

### Task 1 (trivial, do first): restore TinyLlama demo-safe fallback
Run the existing `multi_request_load` proof for TinyLlama (same harness as
Qwen3-8B: 3× seeded `scripts/direct_remote_call.py --seed N --input-scale
0.1` over its full block range), commit evidence, flip
`PROOF_STATUS.yaml` TinyLlama `multi_request_load: passed`. Until then the
safe-demo fallback ladder has only Qwen3-8B.

### Task 2: server-side quantized loading for HF blocks — **DONE (Fable, 2026-07-05)**
See §1.6. Commit `587e27c`; tests in `tests/test_quantized_block_loading.py`.
One intentional deviation from the original spec: `NF4` on qwen3_moe blocks
is **not** blocked anymore — it maps to packed-int4 experts + qint8 Linears
(Task 6 landed first). `NF4` on dense blocks remains fail-closed.

### Task 3: quantization-aware route memory math
`route_picker._model_required_gb` assumes fp16 registry numbers. Add registry
fields (e.g. `int8_min_free_mem_gb`) OR compute `required_gb / 2` (int8) and
`/ 3.7` (int4, later) when evaluating a quantized route variant. A quantized
route candidate must carry `quant_type` in its evaluation payload so
`route_report` output distinguishes `Qwen3-30B-A3B@int8` from fp16.
RED test: with m4pro-like peer (37 GB free), fp16 30B is not memory_fit but
int8 30B is.

**Update:** Moonsong's `mvp_capabilities/quantized_route_lane.py` already
does exactly this for the single `Qwen/Qwen3-30B-A3B@int8` lane (memory math
via measured 1.996× ratio, `@int8` proof-row keying, demo-safe gating).
Generalize that into `route_picker.evaluate_model`/`route_report` — do not
build a second parallel path. Prefer deriving required bytes from
`get_block_size(..., quant_type=...)` over hardcoded ratios where a config is
available; keep the lane's guardrail flags.

### Task 4: proof gates for quantized serving (policy — read carefully)
Quantized routes must NOT inherit fp16 proof status. Key the proof registry by
`(model_id, quant_type)` — suggested YAML shape:
`Qwen/Qwen3-30B-A3B@int8:` alongside the existing plain ids (plain id ==
fp16). `route_picker`/`proof_ladder` treat an absent quantized entry as all
`pending` (fail-closed, they already default pending).
Parity policy for the quantized `full_generation`/`cache_generation` gates:
- exact greedy token-ID match vs the fp16 reference for the demo prompt set →
  eligible for `demo_safe`
- if IDs diverge, the quantized route caps at `showcase-attempt` and the
  artifact must record first-divergence position + text sample. No "close
  enough" demo_safe promotion. This keeps guardrail #2 intact.

### Task 5: Qwen3-30B-A3B int8 ladder on m4pro (the payoff — now fully unblocked)
The serving path is real now: `--quant_type INT8` on a qwen3_moe server
actually quantizes blocks at load (§1.6). Base 30B weights are cached on the
m4pro Seagate snapshot (see `scripts/instruct2507_cache_readiness.py` lane
for the cache layout conventions).

Order: quantize-at-load smoke (one block, measure RSS; compare against
`get_block_size(..., quant_type=INT8)` estimate) → one_block_server →
multi_block 0:2 → full_generation (greedy parity vs fp16 single-host
reference where feasible; if fp16 reference cannot run on the fleet, record
that the reference is the HF CPU implementation on m4pro with short prompts)
→ cache_generation → multi_request_load (seeded, scale-bounded — and record
seed/input_scale per request; the physical-showcase verifier currently
ignores this metadata, which is a known gap).
Every artifact records: quant scheme (`int8_symmetric_per_out_channel` +
`quanto qint8 attn`), source commit, block range, device. Use
`ssh m4pro` directly; MacBook will OOM.

### Task 6: int4 expert packing — **DONE (Fable, 2026-07-05)**
See §1.7. Commit `78a152a`; tests in `tests/test_moe_expert_quant.py` (15
total). The NF4 mapping in Task 2 is already unblocked for qwen3_moe blocks.
One spec correction learned doing it: the original "cosine > 0.999 on the
tiny fixture" bar is not achievable for int4 at the residual-free module
level (~0.985 is inherent rounding noise); the enforced properties are the
elementwise round-trip bound and bit-exact packed-forward-vs-dequantized-
weights equality. Real-weight decoder-layer parity remains the Task 4/5 gate.

### Task 7: wire route_report into coordinator + dashboard
- `join_http_server` `/route` (and `/handoff`'s embedded route decision):
  accept `requested_model` + `selector_mode`, return `route_report` payload
- `demo_dashboard.py`: render "Serving: X (operator override — auto-pick: Y)"
  when `override_active`, and the refusal reason when `override_refused`
- keep the existing `/route` response shape backward compatible (add fields,
  don't rename)
RED tests: HTTP handler returns `best_available` + `picked`; refused pin
never appears as `picked`.

### Task 8: docs + status coherence
- `docs/distributed-inference-mvp.md:107` still calls TinyLlama "the proven
  safe-demo fallback" and Qwen3 "experimental" — stale on both ends now.
- Add a post-MVP milestone entry for the quantization lane in
  `mvp_status.py` `POST_MVP_MILESTONES` (status `in_progress`, spike
  evidence path above). Do NOT touch `MILESTONES` or weights.
- `docs/mvp-finish-plan.md` has a docs-coherence regression test
  (`test_docs_post_mvp_status_rows_match_completed_scouts`) — run it after
  editing.

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
# current baseline (all green at handoff)
.venv/bin/python -m pytest -q                                        # 466 passed, 23 skipped
.venv/bin/python -m pytest tests/test_moe_expert_quant.py -q         # 15 passed (int8 + int4)
.venv/bin/python -m pytest tests/test_quantized_block_loading.py -q  # 9 passed (convert_block/get_block_size)
# spike re-run (slow; MoE block needs ~4GB free)
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/quantized_block_spike.py --devices cpu --skip-moe
# routing contract
.venv/bin/python -m mvp_capabilities.route_picker --report --selector-mode safe-demo \
  --model Qwen/Qwen3-30B-A3B --synthetic-m4-laptops 10   # override_refused=true, picked=Qwen3-8B
# quantized 30B planning lane (Moonsong)
.venv/bin/python -m mvp_capabilities.quantized_route_lane
.venv/bin/python -m json.tool mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json >/dev/null
```
