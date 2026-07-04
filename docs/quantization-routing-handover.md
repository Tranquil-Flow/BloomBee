# Quantization + Best-Model/Override Routing Handover

> **For the build agent:** Fable did the research spikes, the hard module, and
> the routing-contract design in this handover. Your job is integration,
> proof-ladder execution, and the one remaining hard sub-task (int4 expert
> packing). Work task-by-task, RED tests first where specified, commit after
> each task. Do not move the MVP-core denominator; everything here is
> post-MVP.

**Branch state at handoff:** uncommitted work on `main` (Evi reviews before
commit). Baseline suite: `395 passed, 23 skipped`.

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

- int8 (built): ≈ **30.5 GB** → 30B fits on m4pro alone; fp16 KV cache is
  unchanged and small at demo batch sizes.
- int4 experts (Task 6): ≈ **16 GB** → 30B on a 24 GB-class device; dense
  Qwen3-8B at int4 ≈ 4 GB.

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

### 1.6 Environment facts you will need

- New dev deps installed in `.venv` (NOT yet in packaging metadata):
  `optimum-quanto==0.2.7`, `torchao==0.17.0`, `ninja`. Task 2 decides where
  they belong (`quant` extra suggested). torchao is unused so far — drop it
  unless you pick it up deliberately.
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

### Task 2: server-side quantized loading for HF blocks
Integration point: `src/bloombee/server/from_pretrained.py::load_pretrained_block`
(after state-dict load) or `convert_block.convert_block` (replace the no-op) —
prefer `convert_block`, it already receives `quant_type`.

Mapping (reuse the existing `QuantType` enum and `--quant_type` flag; do NOT
invent a parallel flag):
- `INT8` + dense HF block → quanto `quantize(block, weights=qint8); freeze`
- `INT8` + qwen3_moe block → `quantize_qwen3_moe_block_experts(block)` then
  quanto qint8 on remaining Linears (this is exactly the spike's
  `moe_int8_experts+qint8_attn` mode, 1.996× proven)
- `NF4` → blocked with a clear error until Task 6 (do not silently fall back)
- FlexGen-native LLaMA path → untouched (keeps its own compression)

RED tests first:
- loading a tiny qwen3_moe block with `quant_type=INT8` produces a block whose
  experts module is `QuantizedQwen3MoeExperts` and whose weight bytes shrink
  ≥1.9× (tiny-config analog)
- `quant_type=NF4` on an HF block raises (fail-closed) until Task 6
- `server_info.quant_type` advertises the applied mode so peers/dashboard see
  it (field already exists in `data_structures.py`)
- `get_block_size(..., quant_type=INT8)` returns compressed bytes so
  throughput/memory planning stops assuming fp16 (check
  `server/block_utils.py` + `server/throughput.py`)

### Task 3: quantization-aware route memory math
`route_picker._model_required_gb` assumes fp16 registry numbers. Add registry
fields (e.g. `int8_min_free_mem_gb`) OR compute `required_gb / 2` (int8) and
`/ 3.7` (int4, later) when evaluating a quantized route variant. A quantized
route candidate must carry `quant_type` in its evaluation payload so
`route_report` output distinguishes `Qwen3-30B-A3B@int8` from fp16.
RED test: with m4pro-like peer (37 GB free), fp16 30B is not memory_fit but
int8 30B is.

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

### Task 5: Qwen3-30B-A3B int8 ladder on m4pro (the payoff)
Order: quantize-at-load smoke (one block, measure RSS) → one_block_server →
multi_block 0:2 → full_generation (greedy parity vs fp16 single-host
reference where feasible; if fp16 reference cannot run on the fleet, record
that the reference is the HF CPU implementation on m4pro with short prompts)
→ cache_generation → multi_request_load (seeded, scale-bounded — and record
seed/input_scale per request; the physical-showcase verifier currently
ignores this metadata, which is a known gap).
Every artifact records: quant scheme (`int8_symmetric_per_out_channel` +
`quanto qint8 attn`), source commit, block range, device. Use
`ssh m4pro` directly; MacBook will OOM.

### Task 6 (the remaining hard one): int4 expert packing
Extend `moe_expert_quant.py` with a packed int4 mode:
- pack two nibbles/byte along the input dim; group-wise scales (group=128,
  fp16) instead of per-channel — per-channel int4 loses too much precision
- unpack in `_dequant` for hit experts only (pure torch ops so CPU+MPS both
  work; no custom kernels required for correctness — speed later)
- target ≥3.5× vs fp16 on the expert tensors; parity gate: cosine > 0.999 on
  the tiny fixture AND real-weight ladder per Task 4/5 before any route use
- then unblock `NF4` mapping from Task 2
This is genuinely fiddly (odd input dims, group boundaries, scale dtype
drift). Write the round-trip property test first: for random W,
`|W - dequant(pack(W))| <= group_max_abs / 7` elementwise.

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
.venv/bin/python -m pytest -q                                    # 395 passed, 23 skipped
.venv/bin/python -m pytest tests/test_moe_expert_quant.py -q     # 6 passed
# spike re-run (slow; MoE block needs ~4GB free)
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/quantized_block_spike.py --devices cpu --skip-moe
# routing contract
.venv/bin/python -m mvp_capabilities.route_picker --report --selector-mode safe-demo \
  --model Qwen/Qwen3-30B-A3B --synthetic-m4-laptops 10   # override_refused=true, picked=Qwen3-8B
.venv/bin/python -m json.tool mvp_capabilities/distributed_evidence/stretch/quantized-block-spike-20260704T203500Z.json >/dev/null
```
