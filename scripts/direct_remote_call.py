"""Direct, minimal distributed call test against a live BloomBee server.

This bypasses the full test_remote_sequential.py scaffolding (which
includes loading all 22 blocks in a forked pytest process, doing a full
backward pass, etc.) and just exercises the round-trip we care about:
send hidden states to the server, get them back through one remote block,
verify shape and finite values.

Run from inside the bloombee venv on the machine that ALSO has the seed
BloomBee server running with --new_swarm.

Usage:
  python scripts/direct_remote_call.py \\
      --server-maddr "/ip4/127.0.0.1/tcp/31337/p2p/12D3KooW..." \\
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Make the src-layout importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-maddr", required=True,
                   help="One of the multiaddrs printed by the BloomBee server")
    p.add_argument("--model", required=True)
    p.add_argument("--hidden-dim", type=int, default=2048,
                   help="Override hidden_size for the test tensor (defaults to model)")
    p.add_argument("--block-range", default="0:1",
                   help="Block range to request from the server (start:end slice)")
    args = p.parse_args()

    print(f"[direct] model={args.model}")
    print(f"[direct] bootstrap peer: {args.server_maddr}")
    print(f"[direct] requesting block range {args.block_range}")

    from bloombee import AutoDistributedConfig
    from bloombee.client import RemoteSequential
    from bloombee.utils.hivemind_compat import DHT

    print("[direct] AutoDistributedConfig.from_pretrained...")
    t0 = time.time()
    config = AutoDistributedConfig.from_pretrained(
        args.model, initial_peers=[args.server_maddr]
    )
    print(f"[direct]   ... {time.time() - t0:.1f}s, hidden_size={config.hidden_size}, "
          f"num_layers={config.num_hidden_layers}")

    print("[direct] DHT(client_mode=True, start=True)...")
    t0 = time.time()
    dht = DHT(
        initial_peers=[args.server_maddr],
        client_mode=True,
        start=True,
    )
    print(f"[direct]   ... {time.time() - t0:.1f}s")

    print("[direct] RemoteSequential(config, dht=dht)...")
    t0 = time.time()
    start_block, end_block = [int(s) for s in args.block_range.split(":")]
    sequential = RemoteSequential(
        config, dht=dht, start_block=start_block, end_block=end_block,
    )
    print(f"[direct]   ... {time.time() - t0:.1f}s, "
          f"requested layers [{start_block}:{end_block}]")

    hidden = config.hidden_size if args.hidden_dim is None else args.hidden_dim
    test_inputs = torch.randn(1, 5, hidden, requires_grad=True)
    grad_proj = torch.randn(1, 5, hidden)
    print(f"[direct] sending test_inputs shape={list(test_inputs.shape)}...")

    t0 = time.time()
    outputs = sequential(test_inputs)
    forward_s = time.time() - t0
    print(f"[direct]   ... {forward_s:.2f}s, output shape={list(outputs.shape)}")

    t0 = time.time()
    (outputs * grad_proj).sum().backward()
    backward_s = time.time() - t0
    has_grad = test_inputs.grad is not None
    grad_finite = bool(has_grad and torch.isfinite(test_inputs.grad).all().item())
    print(f"[direct]   ... backward {backward_s:.2f}s, grad_finite={grad_finite}, "
          f"grad_norm={float(test_inputs.grad.norm()):.4f}")

    finite = torch.isfinite(outputs).all().item()
    n_unique = float(outputs.unique().numel())
    print(f"[direct] outputs finite: {finite}, unique values: {n_unique}")

    result = {
        "ok": bool(finite and grad_finite),
        "model": args.model,
        "block_range": [start_block, end_block],
        "input_shape": list(test_inputs.shape),
        "output_shape": list(outputs.shape),
        "outputs_finite": finite,
        "outputs_unique": n_unique,
        "grad_finite": grad_finite,
        "grad_norm": float(test_inputs.grad.norm()) if has_grad else None,
        "forward_seconds": forward_s,
        "backward_seconds": backward_s,
    }
    print("[direct] RESULT:", json.dumps(result))
    return 0 if (finite and grad_finite) else 1


if __name__ == "__main__":
    raise SystemExit(main())
