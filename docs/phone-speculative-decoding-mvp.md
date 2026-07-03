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

### Slice 1: phone draft protocol without Android dependency

Build a deterministic draft-provider interface and tests on the laptop first:

```text
DraftProvider.propose(prompt_tokens, max_draft_tokens) -> draft_tokens
Verifier accepts/rejects tokens
Dashboard displays proposed/accepted/rejected/acceptance_rate
```

Use a fake deterministic draft provider for RED/GREEN tests, then a local tiny
model provider.

### Slice 2: remote phone transport

Expose the same provider over a tiny HTTP or stdio bridge. For the user's
workflow, prefer agent-run Termux commands via ADB/SSH/bridge rather than asking
the user to type long commands.

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
