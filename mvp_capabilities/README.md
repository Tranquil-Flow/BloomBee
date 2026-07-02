# mvp_capabilities — BloomBee MVP capability & routing tooling

Small, independent tools that together answer:
**"Given a swarm of BloomBee peers, who can serve which model, and at
what speed?"**

They are deliberately decoupled from BloomBee itself — no `import
bloombee` anywhere. The JSON they produce can be hand-fed into a
scheduler, printed in a CLI dashboard, or pushed into the DHT that
BloomBee's runtime already maintains.

---

## The MVP layers

| Layer | File | What it does | Output |
|------:|------|--------------|--------|
| 1. Hardware | `peer_scan.py` | Probes the local node: hostname, Tailscale IP, CPU model & counts, RAM, MPS/CUDA VRAM, ping latency to peers, free disk on `~/.cache/huggingface`. | JSON to stdout AND `~/.bloombee/capabilities/<hostname>.json` |
| 2. Catalog | `MODEL_REGISTRY.yaml` | Static footprint + arch metadata for ~20 candidate models (TinyLlama through Qwen3-235B-A22B, dense and MoE). | YAML, loaded by the scheduler |
| 3. Benchmark | `bench_throughput.py` | Loads a model with transformers, runs prefill + autoregressive decode, prints `prefill_tok_per_s` and `decode_tok_per_s` plus peak memory. | Single JSON line on stdout |
| 4. Roster | `swarm_roster.py` | Aggregates one or more capability JSON directories, de-duplicates hosts, and prints a swarm summary. | JSON or table |
| 5. Route picker | `route_picker.py` | Chooses the strongest feasible model for the current roster or synthetic 10-laptop MVP scenario. | JSON route decision |
| 6. Sweep planner | `sweep_models.py` | Builds or executes a benchmark sweep for all models that fit a peer. | Dry-run commands or measured JSON |

Layer 1 says *what the hardware is*. Layer 2 says *what models exist and how big they are*. Layer 3 says *what each model actually achieves on this hardware*.

A naive router is `peer_free_gb >= model.min_total_mem_gb`. A better router is `peer_decode_tok_s >= request.min_decode_tok_s`, using the benchmark numbers instead of the catalog alone.

---

## Quickstart

```bash
cd ~/Projects/distributed-inference-mvp
source .venv/bin/activate

# 1. Probe this machine. Writes ~/.bloombee/capabilities/<host>.json.
python mvp_capabilities/peer_scan.py

#    With peer ping list:
python mvp_capabilities/peer_scan.py --peers m4-pro,m4-laptop,node3.tail.ts.net

# 2. Inspect the model catalog (no code needed; it's data).
cat mvp_capabilities/MODEL_REGISTRY.yaml

# 3. Benchmark the default small model on this machine (MPS, bf16).
python mvp_capabilities/bench_throughput.py

#    A bigger target — same shape of output:
python mvp_capabilities/bench_throughput.py --model Qwen/Qwen2.5-3B-Instruct --max-new-tokens 128

#    Force a specific device/dtype (handy for the M4 laptop when you're
#    remote-debugging on a CUDA box):
python mvp_capabilities/bench_throughput.py --device cuda --dtype fp16 --model Qwen/Qwen2.5-7B-Instruct

# 4. Aggregate real peer scans.
python mvp_capabilities/swarm_roster.py --cap-dir ~/.bloombee/capabilities --json

# 5. Pick the strongest feasible route for real devices.
python mvp_capabilities/route_picker.py --cap-dir ~/.bloombee/capabilities

# 6. Plan the 10-laptop MVP showcase route before physical showcase day.
python mvp_capabilities/route_picker.py \
  --cap-dir ~/.bloombee/capabilities \
  --scenario mvp-10-laptop \
  --synthetic-m4-laptops 10 \
  --synthetic-total-gb 24 \
  --synthetic-free-gb 20

# 7. Plan a per-peer benchmark sweep without downloading/running models.
python mvp_capabilities/sweep_models.py \
  --peer ~/.bloombee/capabilities/$(hostname -s).json \
  --dry-run
```

Default benchmark is `Qwen/Qwen2.5-0.5B-Instruct` at 128 prefill + 64 decode tokens. On an M4 Pro it downloads in ~10 s and runs end-to-end in under 30 s.

---

## Verified MVP status

As of the current implementation slice:

- Local `evinova` / `Evis-MacBook-Pro`: M4, 16GB unified memory, MPS.
- Remote `evinova-self` / `m4pro`: M4 Pro, 48GB unified memory, verified via `ssh m4pro`.
- Real two-device roster route currently picks `google/gemma-2-9b-it` as a solo M4 Pro route when M4 Pro has ~28.5GB free.
- Synthetic 10-laptop MVP route picks `Qwen/Qwen3-30B-A3B` as the block-parallel candidate.
- Physical 10-laptop showcase remains part of MVP scope but must happen after local + two-device verification and MoE block serving are complete.

---

## Wiring into BloomBee

These tools **do not import BloomBee** on purpose. The hivemind/DHT stack is heavy and version-sensitive; routing decisions shouldn't pay that cost.

Three reasonable integration points:

1. **Static config dump.** Run `peer_scan.py` once on each peer, commit the JSON files into `bloombee/peer_capabilities/` alongside the swarm config, and let the existing config-loader read them.
2. **CLI dashboard.** A `swarm status` command in the CLI that runs `peer_scan.py` over SSH or Tailscale to each peer and joins the result with the model catalog to render a table: `host | free_gb | best_model | decode_tok/s`.
3. **DHT population.** On startup, each peer pushes a small capability record into the BloomBee DHT (one key per peer), and the route picker queries the DHT for "peers where `min_total_mem_gb <= free_gb` AND `decode_tok_per_s >= threshold`". The benchmark numbers are an offline measurement, not a runtime query — re-benchmark when hardware changes, not per request.

All three patterns consume the same JSON shape that `peer_scan.py` already emits, so swapping mechanisms is a one-file change.

---

## Roofline sanity check

The decode phase of an autoregressive LLM is *memory-bandwidth-bound*: every step you have to read every parameter from VRAM/RAM at least once to produce the next token. The roofline (theoretical maximum, before any overhead) is therefore:

```
decode_tok_per_s_roofline  ≈  mem_bandwidth_GB_s  /  (params_b × dtype_bytes)
```

Example, Qwen2.5-7B in bf16 on an M4 Pro:
- M4 Pro unified-memory bandwidth ≈ **273 GB/s**
- params = 7.62 B, dtype_bytes = 2 → weights = 15.24 GB
- roofline ≈ 273 / 15.24 ≈ **17.9 tok/s**

In practice you will measure **40–70 %** of that — call it **8–12 tok/s** — because:

- the attention KV cache and activations also consume bandwidth per step
- framework allocator overhead (PyTorch caching allocator padding, MPS graph captures)
- the LM head matmul at vocab≈150k is not pure weight-read
- batch=1 wastes compute parallelism that real serving pools via continuous batching

If `bench_throughput.py` reports a number *higher* than the roofline, suspect either a smaller model than you think, or that the GPU isn't fully resident. If it's *much* lower (< 30 % of roofline), suspect CPU offload, an under-spec'd `dtype`, or another process pinning the accelerator.

This formula is the first thing to check when a peer that "should" be fast is slow.