#!/usr/bin/env python3
"""Render and verify repeated Termux draft-provider latency smoke evidence.

This module measures only a deterministic/static draft-provider contract loop on
Termux. It is intentionally not a model benchmark: it does not prove speculative
speedup, live generation, or BloomBee block serving. The goal is to turn the next
phone milestone from "one smoke JSON exists" into "repeated phone-side contract
latency and acceptance counters are captured and auditable".
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import stat
from pathlib import Path
from typing import Any

SOURCE = "termux_draft_latency.py"
SCRIPT_SOURCE = "termux_draft_latency.sh"
CLAIM_BOUNDARY = "termux_draft_latency_static_contract_only_no_generation_proof"
ERROR_CLAIM_BOUNDARY = "termux_draft_latency_error_no_generation_proof"
VERIFY_CLAIM_BOUNDARY = "termux_draft_latency_verifier_only_no_generation_proof"
EXPECTED_COUNTERS_PER_ITERATION = {"proposed": 3, "accepted": 2, "rejected": 1, "acceptance_rate": 0.666667}


def render_termux_latency_script(*, iterations: int = 50, warmup: int = 5) -> str:
    """Return a pasteable Termux shell script for repeated latency smoke."""
    iterations = max(1, int(iterations))
    warmup = max(0, int(warmup))
    template = '''#!/usr/bin/env sh
# BloomBee Termux draft-provider repeated-latency smoke.
# claim_boundary: {CLAIM_BOUNDARY}
# This measures a deterministic static contract loop only.
set -u
PY_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PY_BIN" ]; then
  printf '%s\n' '{"ok":false,"source":"{SCRIPT_SOURCE}","claim_boundary":"{ERROR_CLAIM_BOUNDARY}","error":"python_missing_install_with_pkg_install_python","generation_proven":false,"speedup_proven":false,"inference_proven":false}'
  exit 2
fi
"$PY_BIN" - <<'PY'
import json
import os
import platform
import statistics
import subprocess
import time

CLAIM_BOUNDARY = "{CLAIM_BOUNDARY}"
ITERATIONS = {ITERATIONS}
WARMUP = {WARMUP}

def getprop(key):
    try:
        return subprocess.check_output(["getprop", key], text=True, stderr=subprocess.DEVNULL, timeout=1).strip() or None
    except Exception:
        return None

def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac

def one_iteration():
    prompt_tokens = [1, 2, 3]
    draft_tokens = [5, 6, 7]
    verifier_tokens = [5, 6, 8]
    start = time.perf_counter()
    accepted = []
    for index, token in enumerate(draft_tokens):
        if index >= len(verifier_tokens) or verifier_tokens[index] != token:
            break
        accepted.append(token)
    accepted_count = len(accepted)
    rejected = draft_tokens[accepted_count:]
    fallback_token = verifier_tokens[accepted_count] if rejected and accepted_count < len(verifier_tokens) else None
    committed = list(accepted)
    if fallback_token is not None:
        committed.append(fallback_token)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "proposed": len(draft_tokens),
        "accepted": accepted_count,
        "rejected": len(rejected),
        "acceptance_rate": round(accepted_count / len(draft_tokens), 6),
        "committed_tokens": committed,
    }

for _ in range(WARMUP):
    one_iteration()

results = [one_iteration() for _ in range(ITERATIONS)]
elapsed_values = [item["elapsed_ms"] for item in results]
total_proposed = sum(item["proposed"] for item in results)
total_accepted = sum(item["accepted"] for item in results)
total_rejected = sum(item["rejected"] for item in results)
prefix = os.environ.get("PREFIX")
is_termux = bool(prefix and prefix.startswith("/data/data/com.termux/"))
payload = {
    "ok": True,
    "source": "{SCRIPT_SOURCE}",
    "claim_boundary": CLAIM_BOUNDARY,
    "measurement_kind": "termux_static_draft_contract_loop",
    "phone_runtime": {
        "is_mobile": True,
        "kind": "android",
        "runtime": "termux" if is_termux else "unknown_non_termux",
        "is_termux": is_termux,
        "prefix": prefix,
        "android_model": getprop("ro.product.model"),
        "android_manufacturer": getprop("ro.product.manufacturer"),
        "android_sdk": getprop("ro.build.version.sdk"),
        "soc": getprop("ro.soc.model") or getprop("ro.hardware"),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "provider": {
        "provider_id": "termux-static-draft-provider",
        "provider_kind": "termux_static_fake",
        "phone_compatible_interface": True,
        "can_serve_transformer_blocks": False,
    },
    "iterations": ITERATIONS,
    "warmup_iterations": WARMUP,
    "per_iteration_expected_counters": {
        "proposed": 3,
        "accepted": 2,
        "rejected": 1,
        "acceptance_rate": 0.666667,
    },
    "aggregate_counters": {
        "proposed": total_proposed,
        "accepted": total_accepted,
        "rejected": total_rejected,
        "acceptance_rate": round(total_accepted / total_proposed, 6) if total_proposed else 0.0,
    },
    "latency_ms": {
        "min": round(min(elapsed_values), 6),
        "mean": round(statistics.fmean(elapsed_values), 6),
        "median": round(statistics.median(elapsed_values), 6),
        "p95": round(percentile(elapsed_values, 0.95), 6),
        "max": round(max(elapsed_values), 6),
        "samples": [round(value, 6) for value in elapsed_values],
    },
    "generation_proven": False,
    "speedup_proven": False,
    "inference_proven": False,
    "can_update_proof_status": False,
    "operator_next_steps": [
        "compare repeated phone latency against a real verifier-only baseline before claiming speedup",
        "replace the static provider with a real tiny draft model only after this contract remains stable",
        "do not count this phone as a transformer-block worker without a separate block-serving proof",
    ],
}
print(json.dumps(payload, sort_keys=True))
PY
'''
    return (
        template.replace("{CLAIM_BOUNDARY}", CLAIM_BOUNDARY)
        .replace("{SCRIPT_SOURCE}", SCRIPT_SOURCE)
        .replace("{ERROR_CLAIM_BOUNDARY}", ERROR_CLAIM_BOUNDARY)
        .replace("{ITERATIONS}", str(iterations))
        .replace("{WARMUP}", str(warmup))
    )


def write_termux_latency_script(path: str | Path, *, iterations: int = 50, warmup: int = 5) -> dict[str, object]:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_termux_latency_script(iterations=iterations, warmup=warmup), encoding="utf-8")
    out.chmod(out.stat().st_mode | stat.S_IXUSR)
    return {
        "source": SOURCE,
        "claim_boundary": "termux_draft_latency_script_written_no_phone_run",
        "out": str(out),
        "iterations": max(1, int(iterations)),
        "warmup_iterations": max(0, int(warmup)),
        "local_command": f"sh {shlex.quote(str(out))}",
        "phone_latency_smoke_proven": False,
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if "\n" in text:
        candidates = [line for line in text.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
        if len(candidates) == 1:
            text = candidates[0]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Termux latency evidence must be a JSON object")
    return payload


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def verify_termux_latency_evidence(
    path: str | Path,
    *,
    require_termux: bool = True,
    min_iterations: int = 20,
) -> dict[str, object]:
    try:
        payload = _load_json(path)
    except Exception as exc:
        return {
            "source": SOURCE,
            "claim_boundary": VERIFY_CLAIM_BOUNDARY,
            "verification_status": "failed",
            "ok": False,
            "error": f"could not parse evidence: {type(exc).__name__}: {exc}",
            "phone_latency_smoke_proven": False,
            "generation_proven": False,
            "speedup_proven": False,
            "inference_proven": False,
            "can_update_proof_status": False,
        }

    errors: list[str] = []
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("unexpected claim_boundary")
    if payload.get("measurement_kind") != "termux_static_draft_contract_loop":
        errors.append("unexpected measurement_kind")
    if payload.get("ok") is not True:
        errors.append("payload ok is not true")
    iterations = payload.get("iterations")
    if not isinstance(iterations, int) or iterations < int(min_iterations):
        errors.append(f"iterations must be >= {min_iterations}")
    if payload.get("per_iteration_expected_counters") != EXPECTED_COUNTERS_PER_ITERATION:
        errors.append("per-iteration counters do not match expected static smoke")
    aggregate = payload.get("aggregate_counters") or {}
    if isinstance(iterations, int):
        expected_aggregate = {
            "proposed": iterations * 3,
            "accepted": iterations * 2,
            "rejected": iterations,
            "acceptance_rate": 0.666667,
        }
        if aggregate != expected_aggregate:
            errors.append("aggregate counters do not match iterations")
    latency = payload.get("latency_ms") or {}
    for field in ("min", "mean", "median", "p95", "max"):
        if not _finite_nonnegative(latency.get(field)):
            errors.append(f"latency_ms.{field} must be finite and non-negative")
    samples = latency.get("samples")
    if isinstance(iterations, int):
        if not isinstance(samples, list) or len(samples) != iterations:
            errors.append("latency_ms.samples length must match iterations")
        elif not all(_finite_nonnegative(sample) for sample in samples):
            errors.append("latency_ms.samples must be finite and non-negative")
    runtime = payload.get("phone_runtime") or {}
    if require_termux and not (isinstance(runtime, dict) and runtime.get("is_termux") is True):
        errors.append("evidence was not captured from Termux")
    for flag in ("generation_proven", "speedup_proven", "inference_proven", "can_update_proof_status"):
        if payload.get(flag) is not False:
            errors.append(f"{flag} must be false")

    passed = not errors
    return {
        "source": SOURCE,
        "claim_boundary": VERIFY_CLAIM_BOUNDARY,
        "verification_status": "passed" if passed else "failed",
        "ok": passed,
        "errors": errors,
        "evidence": payload,
        "iterations": iterations,
        "aggregate_counters": aggregate,
        "latency_ms": latency,
        "termux_detected": bool(isinstance(runtime, dict) and runtime.get("is_termux") is True),
        "phone_latency_smoke_proven": passed,
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": "replace static draft loop with a real tiny-model draft provider or compare against verifier-only baseline" if passed else "rerun the Termux latency script and verify the captured JSON",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Write or print the Termux repeated-latency shell script")
    render.add_argument("--out", default=None, help="Optional path to write; omit to print script")
    render.add_argument("--iterations", type=int, default=50)
    render.add_argument("--warmup", type=int, default=5)
    render.add_argument("--json", action="store_true", help="When --out is used, print JSON metadata")

    verify = sub.add_parser("verify", help="Verify captured Termux latency JSON")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--allow-non-termux", action="store_true", help="Allow local/non-Termux smoke evidence; tests only")
    verify.add_argument("--min-iterations", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "render":
        if args.out:
            payload = write_termux_latency_script(args.out, iterations=args.iterations, warmup=args.warmup)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(payload["local_command"])
        else:
            print(render_termux_latency_script(iterations=args.iterations, warmup=args.warmup), end="")
        return 0
    if args.command == "verify":
        report = verify_termux_latency_evidence(
            args.evidence,
            require_termux=not args.allow_non_termux,
            min_iterations=args.min_iterations,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
