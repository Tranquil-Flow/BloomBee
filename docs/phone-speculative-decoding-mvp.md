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
