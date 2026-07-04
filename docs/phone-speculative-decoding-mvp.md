# Phone peers and speculative decoding for the distributed-inference MVP

## MVP verdict

Phones can probably become useful MVP components as **draft model** or control-plane
peers, but we should **not count phones as transformer-block workers** for the
MVP until a real phone produces throughput evidence and successfully serves at
least one verified BloomBee block.

The shortest plausible path is not "phone hosts Qwen/TinyLlama transformer
blocks." The shortest plausible path is **speculative decoding**:

```text
phone runs small draft model / token proposer
laptop swarm runs verifier model
accepted draft tokens reduce verifier decode steps
rejected draft tokens fall back to normal verifier decode
```

That lets a phone contribute useful work even if it cannot hold large block
weights or sustain laptop-class memory bandwidth.

## Why ordinary block serving is unlikely for phones in this MVP

Current BloomBee block serving expects a Python/PyTorch runtime and benefits from
MPS/CUDA-style accelerator memory. Android/Termux phones are usually constrained
by:

- lower sustained memory bandwidth than M-series laptops,
- thermal throttling under long decode loops,
- limited RAM headroom after Android background services,
- no MPS backend and inconsistent PyTorch/accelerator support,
- higher setup risk for libp2p/DHT/PyTorch wheels in Termux,
- lower value-per-engineering-hour than laptop/M4 Pro gates already on the MVP path.

A phone can still be valuable as DHT/control-plane/monitoring immediately, but
that is not the same as useful transformer-block inference.

## Why speculative decoding is the better phone role

Speculative decoding is a verifier/draft split:

1. A fast cheap draft model proposes several candidate tokens.
2. The main model verifies those tokens in one larger forward pass.
3. Accepted tokens are committed; rejected tokens fall back to the verifier's next token.

For phones, this is attractive because the draft task can be small:

- a TinyLlama-class draft if the phone can run it,
- a sub-500M model through llama.cpp/MLC/ExecuTorch if PyTorch is awkward,
- a prompt/cache-aware n-gram proposer as a trivial baseline.

The verifier remains the laptop swarm, so quality and correctness still come from
the main model.

## MVP feasibility estimate

**Feasible for MVP if scoped as a demo extension:** yes, with a narrow target.

Recommended MVP phone target:

```text
Phone = draft-token proposer over HTTP/WebSocket/ADB/SSH
Laptop swarm = verifier/generator
Dashboard = shows proposed/accepted/rejected token counts and acceptance rate
```

Do **not** try to make Android a full BloomBee block worker before the 10-laptop
showcase unless a phone already has a clean Python/PyTorch + server path. That is
likely to consume more time than it saves.

## Proof gates

A phone speculative-decoding contribution counts only after these proof gates pass:

1. **Capability scan:** `peer_scan.py` runs on the phone and records Android/Termux
   model, SoC, ABI, SDK, RAM, and network identity.
2. **Draft throughput:** the phone emits measured draft tokens/sec for the chosen
   draft path with thermal/run duration noted.
3. **Verifier integration:** a laptop-side verifier accepts draft candidates and
   preserves exact output parity with baseline greedy decode.
4. **Useful acceptance:** dashboard reports accepted/rejected/proposed counts;
   acceptance rate is non-zero and total wall-clock is not worse than baseline
   after network overhead.
5. **Failure isolation:** if the phone disconnects or produces bad drafts, the
   laptop swarm falls back to normal verifier decode without changing output.

## Suggested implementation slices

### Slice 1: phone draft protocol without Android dependency — built

Build a deterministic draft-provider interface and tests on the laptop first:

```text
DraftProvider.propose(prompt_tokens, max_draft_tokens) -> draft_tokens
Verifier accepts/rejects tokens
Dashboard displays proposed/accepted/rejected/acceptance_rate
```

`mvp_capabilities/draft_provider.py` now provides the dependency-free contract,
a deterministic hash fake, a static fake for RED/GREEN tests, verifier-prefix
accept/reject accounting, CLI JSON output, and dashboard counters. It still does
not prove live generation speedup or phone transport.

### Slice 2: remote phone transport — stdio groundwork built

Expose the same provider over a tiny HTTP or stdio bridge. For the user's
workflow, prefer agent-run Termux commands via ADB/SSH/bridge rather than asking
the user to type long commands.

`mvp_capabilities/draft_provider_bridge.py` now exposes the same provider
contract over stdio JSONL, which is the lowest-friction path for Termux/ADB/SSH
experiments. A real phone run, latency measurement, and acceptance-rate proof are
still pending.

Because Hermes cannot start the local `adb` daemon in this sandbox
(`could not install *smartsocket* listener: Operation not permitted`),
`mvp_capabilities/termux_draft_smoke.py` now renders a self-contained shell
script that can be pasted directly into Termux and later verified from its JSON
output. This is still only draft-contract smoke, not useful phone compute.

2026-07-04 update: with the phone connected to `m4pro`, ADB detected a
Pixel 8 Pro (`Tensor G3`, Android SDK 36). The agent pushed the smoke script to
`/sdcard/Download`, typed the short Termux command
`sh /sdcard/Download/bloombee-run.sh`, pulled the JSON output back, and verified
it with `termux_draft_smoke.py verify`. Tracked evidence:
`mvp_capabilities/distributed_evidence/phone/termux-draft-smoke-20260704T095557Z.json`.
The proof boundary remains `termux_draft_provider_smoke_verifier_only_no_generation_proof`:
it proves Termux can run the draft-provider contract smoke, not speedup,
transformer-block serving, or useful phone inference yet.

2026-07-04 latency update: `mvp_capabilities/termux_draft_latency.py` measured
50 static draft-contract iterations on the same Pixel 8 Pro through Termux. The
verified report is tracked at
`mvp_capabilities/distributed_evidence/phone/termux-draft-latency-20260704T100644Z.json`.
It recorded proposed=150, accepted=100, rejected=50, p95=0.001669 ms for the
static in-process contract loop. Boundary:
`termux_draft_latency_verifier_only_no_generation_proof`. This is not a real
tiny-model benchmark, not a network round-trip bridge benchmark, and not a
speedup proof.

2026-07-04 feasibility update: `mvp_capabilities/termux_tiny_model_probe.py`
verified the phone environment without installing or downloading anything.
Tracked report:
`mvp_capabilities/distributed_evidence/phone/termux-tiny-model-probe-20260704T101232Z.json`.
Facts from the Pixel 8 Pro: 11.851 GB total RAM, 2.557 GB available RAM,
28.425 GB free storage, Termux Python 3.13.13, clang/cmake/make/git/pip/pkg
present. Genuine blockers for real tiny-model draft or BloomBee block serving:
`torch`, `transformers`, `tokenizers`, `llama_cpp`, and `bloombee` are not
installed. The likely next path is a tiny GGUF/llama.cpp-style draft runtime;
Python/PyTorch BloomBee block serving is not ready on this phone.

The guarded follow-up plan is tracked at
`mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-plan-20260704T101232Z.json`.
It does **not** execute installs or downloads; it only records that a guarded
`llama-cpp-python`/GGUF draft-runtime install attempt is plausible because
Termux, Python, pip, pkg, clang, cmake, make, git, storage, and memory gates pass.
Any install remains a side-effecting operator decision.

2026-07-04 approved install/generation update: using `m4pro` ADB into Termux,
the phone now has Termux `llama.cpp` CLI tools available (`llama-cli`,
`llama-bench`, `llama-server`) and downloaded
`ggml-org/tiny-llamas/stories15M.gguf` (98,357,920 bytes,
SHA256 `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`).
Tracked proof:
`mvp_capabilities/distributed_evidence/phone/termux-gguf-runtime-generation-20260704T104506Z.json`.
`llama-cli` generated `One day, a little girl named Lucy` from prompt
`Once upon a time` in 0.347524s, and `llama-bench` emitted JSON successfully.
This proves standalone tiny-GGUF phone generation, not BloomBee block serving and
not speculative decoding speedup.

Follow-up bridge proof:
`mvp_capabilities/termux_gguf_draft_bridge.py` rendered a phone-side JSON bridge
around `llama-cli`; the Pixel returned a draft-provider-candidate envelope with
prompt `Once upon a time`, `n_predict=8`, generated text
`One day, a little girl named Lucy`, and elapsed time 0.565503s. Tracked proof:
`mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-bridge-20260704T105400Z.json`.
This advances the phone from standalone tiny-GGUF generation to a bridgeable
draft-provider candidate, but verifier acceptance and speedup remain unproven.

Verifier-comparison update: `mvp_capabilities/phone_draft_verifier_compare.py`
compares the phone draft text as exact UTF-8 byte tokens against authoritative
verifier text. Tracked positive-control evidence accepted 33/33 bytes:
`mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-positive-control-20260704T110000Z.json`.
Tracked live verifier evidence used m4pro `Qwen/Qwen2.5-0.5B-Instruct`, which
generated `In the vast and mysterious universe of the` for the same prompt and
accepted 0/33 phone-draft bytes:
`mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-qwen05-20260704T110000Z.json`.
This proves comparison machinery and a real verifier mismatch; it still does not
prove tokenizer-level speculative decoding or speedup.

Same-GGUF verifier update: Hugging Face re-download of the tiny GGUF was flaky,
so the exact proven phone model was copied from Termux to `/sdcard/Download/`,
pulled through m4pro ADB, and run locally with `/opt/homebrew/bin/llama-cli`.
The local verifier artifact
`mvp_capabilities/distributed_evidence/phone/local-stories15M-phone-exact-verifier-20260704T111215Z.json`
has the same SHA256
`61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`
and generated the same text, `One day, a little girl named Lucy`. The comparison
artifact
`mvp_capabilities/distributed_evidence/phone/termux-gguf-draft-verifier-same-gguf-20260704T111215Z.json`
accepted 33/33 UTF-8 bytes. This is nonzero independent verifier acceptance for
the same GGUF, but tokenizer IDs and wall-clock speedup are still unproven.

Tokenizer-ID update: Termux and local `/opt/homebrew/bin/llama-tokenize` now both
tokenize the same prompt/draft using the exact phone-copied GGUF. Tracked phone
tokenizer evidence:
`mvp_capabilities/distributed_evidence/phone/termux-tokenizer-ids-20260704T111800Z.json`.
Tracked local-vs-phone comparison:
`mvp_capabilities/distributed_evidence/phone/termux-local-tokenizer-id-compare-20260704T111800Z.json`.
Prompt IDs match as `[9038, 2501, 263, 931]`; draft IDs match as
`[3118, 2462, 29892, 263, 2217, 7826, 4257, 28846]`; accepted draft token IDs:
8/8. This closes the same-GGUF tokenizer-match gate, but not wall-clock speedup.

Wall-clock gate update: `mvp_capabilities/phone_speculative_wallclock_gate.py`
consumes the phone draft bridge, local same-GGUF verifier, and tokenizer-ID
comparison artifacts. The tracked report
`mvp_capabilities/distributed_evidence/phone/termux-same-gguf-wallclock-gate-20260704T112500Z.json`
shows sequential phone-draft+verifier is slower than verifier-only:
0.565503s + 1.837976s = 2.403479s vs 1.837976s. Therefore
`speedup_proven=false`; the next real speedup gate requires an integrated
verifier path that validates draft token IDs without rerunning full verifier-only
decode.

Local integrated harness reference: local `/opt/homebrew/bin/llama-speculative`
can run `stories15M.gguf` as both draft and target. Tracked artifact:
`mvp_capabilities/distributed_evidence/phone/local-same-gguf-llama-speculative-harness-20260704T113600Z.json`.
It accepted 8/8 draft tokens, proving the local llama.cpp integrated speculative
harness is available. It explicitly does **not** involve the phone and does not
prove phone-backed speedup.

Phone-token integrated verifier preflight:
`mvp_capabilities/distributed_evidence/phone/phone-integrated-verifier-preflight-20260704T114000Z.json`
records the next implementation boundary. Local `llama_cpp` Python bindings are
missing, and the available `llama-speculative` CLI supports `--model-draft` but
does not expose an option to ingest external phone-provided draft token IDs. The
next real implementation therefore needs a custom llama.cpp binding path or a
CLI extension before phone-backed speedup can be measured honestly.

Binding verifier follow-up: `llama-cpp-python` was installed locally after
preinstalling build deps (`cmake`, `ninja`, `scikit-build-core`) with build
isolation disabled. `mvp_capabilities/phone_llama_cpp_binding_verifier.py`
renders the exact CLI chat template observed via `llama-cli --verbose-prompt`
and verifies the phone draft text bytes against context-generated target tokens.
Tracked artifact:
`mvp_capabilities/distributed_evidence/phone/phone-llama-cpp-binding-verifier-20260704T120000Z.json`.
It accepts 33/33 UTF-8 bytes and 8 context-generated target tokens
(`[6716, 2462, 29892, 263, 2217, 7826, 4257, 28846]`).
Important tokenization pitfall: standalone tokenization of the draft text starts
with `3118`, while retokenizing `rendered_prompt + draft_text` in verifier
context starts with `6716`; the binding artifact records both and marks the
standalone/context mismatch explicitly.

External context-token follow-up:
`mvp_capabilities/distributed_evidence/phone/termux-context-token-ids-20260704T121646Z.json`
was emitted by Pixel 8 Pro Termux `llama-tokenize` over the exact rendered prompt
and draft suffix. The binding verifier then consumed those external phone context
token IDs and accepted 8/8 in
`mvp_capabilities/distributed_evidence/phone/phone-context-token-id-verifier-20260704T121646Z.json`.
This proves token-ID ingestion/comparison mechanics for this same-GGUF path, but
still does **not** prove live phone-backed speedup.

BloomBee block-serving preflight:
`mvp_capabilities/distributed_evidence/phone/phone-bloombee-block-serving-preflight-20260704T121500Z.json`
consumes the Termux probe and records that GGUF draft generation is not BloomBee
block serving. It remains fail-closed because Termux is missing `torch`,
`transformers`, and the `bloombee` Python package, so one-block server/client
proof cannot be claimed from phone evidence yet.

### Slice 3: real phone smoke

Run capability scan + draft throughput + verifier parity on one connected phone.
If it fails to improve wall-clock, keep phones in the dashboard as
`control-plane/draft-experimental`, not as inference workers.

## Dashboard implication

The dashboard should show phones separately from laptop block workers:

```text
Connected devices
- block workers: laptops/M4 Pro with served block ranges and throughput
- draft workers: phones with proposed/accepted/rejected tokens/sec
- control-plane: peers that help discovery/monitoring but do not affect inference output
```

That keeps the demo honest: phones can be part of the swarm without pretending
they already run large transformer blocks.
