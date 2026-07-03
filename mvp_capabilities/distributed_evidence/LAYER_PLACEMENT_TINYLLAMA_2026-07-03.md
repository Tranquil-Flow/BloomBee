# TinyLlama 3-peer layer-placement proof — 2026-07-03

## What was verified

Three real BloomBee server processes were started on `m4pro` from a disposable
proof tree copied from the current working tree. They served the full 22-layer
TinyLlama transformer stack as contiguous block ranges:

| Server label | Port | Layers |
|---|---:|---:|
| `m4pro-seed` | 31337 | `0:8` |
| `m4pro-mid` | 31338 | `8:15` |
| `m4pro-tail` | 31339 | `15:22` |

A direct BloomBee client call over requested block range `0:22` returned finite
outputs and finite gradients through all three live servers.

Evidence JSON:

```text
mvp_capabilities/distributed_evidence/DIRECT_REMOTE_CALL_3PEER_LAYER_PLACEMENT_TINYLLAMA_2026-07-03.json
```

Key result:

```json
{
  "ok": true,
  "block_range": [0, 22],
  "input_shape": [1, 5, 2048],
  "output_shape": [1, 5, 2048],
  "outputs_finite": true,
  "grad_finite": true,
  "forward_seconds": 0.5287899971008301,
  "backward_seconds": 0.26585912704467773
}
```

The dashboard now reads `server_placements` from evidence JSON and renders this
as a **Layer placement** table.

## Important caveat

This was a real multi-process BloomBee proof, but the client ran on `m4pro` as
well because the local Hermes sandbox currently blocks libp2p daemon/control
socket binds (`listen unix /tmp/hivemind-p2pd-...sock: bind: operation not
permitted`). Earlier committed evidence still covers local-M4-to-m4pro TinyLlama
client/server runs, but this specific layer-placement metadata proof is same-host
client against three live server processes.

Do not claim a physical 10-laptop or self-serve user-laptop demo from this proof.
The remaining gates are a real laptop join script/installer, automatic layer
assignment for connected peers, non-sandbox generate-api parity with placement
metadata, and a physical N-laptop showcase run.
