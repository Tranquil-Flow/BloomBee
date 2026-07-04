#!/usr/bin/env python3
"""Render and verify a self-contained Termux draft-provider smoke script.

The rendered shell script is meant to be pasted into Termux on a connected phone.
It uses only a local Python interpreter in Termux and emits one JSON object with
proposed/accepted/rejected draft-token counters. This proves only the phone can
run the draft-provider contract smoke; it does not prove live speculative speedup,
BloomBee block serving, or generation correctness.
"""

from __future__ import annotations

import argparse
import json
import shlex
import stat
from pathlib import Path
from typing import Any

SOURCE = "termux_draft_smoke.py"
SCRIPT_SOURCE = "termux_draft_smoke.sh"
CLAIM_BOUNDARY = "termux_draft_provider_smoke_only_no_generation_proof"
ERROR_CLAIM_BOUNDARY = "termux_draft_provider_smoke_error_no_generation_proof"
VERIFY_CLAIM_BOUNDARY = "termux_draft_provider_smoke_verifier_only_no_generation_proof"
EXPECTED_COUNTERS = {"proposed": 3, "accepted": 2, "rejected": 1, "acceptance_rate": 0.666667}


def render_termux_smoke_script() -> str:
    """Return a pasteable Termux shell script for the draft-provider smoke."""
    template = '''#!/usr/bin/env sh
# BloomBee Termux draft-provider smoke.
# claim_boundary: {CLAIM_BOUNDARY}
# This does not prove generation, speedup, or block serving.
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
import subprocess
import time

CLAIM_BOUNDARY = "{CLAIM_BOUNDARY}"

def getprop(key):
    try:
        return subprocess.check_output(["getprop", key], text=True, stderr=subprocess.DEVNULL, timeout=1).strip() or None
    except Exception:
        return None

start = time.perf_counter()
prompt_tokens = [1, 2, 3]
draft_tokens = [5, 6, 7]
verifier_tokens = [5, 6, 8]
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
elapsed_ms = (time.perf_counter() - start) * 1000
prefix = os.environ.get("PREFIX")
is_termux = bool(prefix and prefix.startswith("/data/data/com.termux/"))
payload = {
    "ok": True,
    "source": "{SCRIPT_SOURCE}",
    "claim_boundary": CLAIM_BOUNDARY,
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
    "prompt_tokens": prompt_tokens,
    "verifier_tokens": verifier_tokens,
    "proposal": {
        "draft_tokens": draft_tokens,
        "draft_token_count": len(draft_tokens),
        "max_draft_tokens": 3,
        "elapsed_ms": round(elapsed_ms, 3),
    },
    "verdict": {
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "proposed_count": len(draft_tokens),
        "acceptance_rate": round(accepted_count / len(draft_tokens), 6),
        "verifier_fallback_token": fallback_token,
        "committed_tokens": committed,
        "verifier_authoritative": True,
        "accepted_tokens_require_verifier_match": True,
    },
    "dashboard_counters": {
        "proposed": len(draft_tokens),
        "accepted": accepted_count,
        "rejected": len(rejected),
        "acceptance_rate": round(accepted_count / len(draft_tokens), 6),
    },
    "generation_proven": False,
    "speedup_proven": False,
    "inference_proven": False,
    "can_update_proof_status": False,
    "operator_next_steps": [
        "copy this JSON back to the laptop and verify with termux_draft_smoke.py verify",
        "measure repeated latency before claiming useful phone draft speedup",
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
    )


def write_termux_smoke_script(path: str | Path) -> dict[str, object]:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_termux_smoke_script(), encoding="utf-8")
    out.chmod(out.stat().st_mode | stat.S_IXUSR)
    return {
        "source": SOURCE,
        "claim_boundary": "termux_draft_smoke_script_written_no_phone_run",
        "out": str(out),
        "paste_hint": "copy this script into Termux or run it on the phone checkout; capture the single JSON line it prints",
        "local_command": f"sh {shlex.quote(str(out))}",
        "phone_smoke_proven": False,
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if "\n" in text:
        # Accept copied terminal output as long as exactly one JSON-looking line exists.
        candidates = [line for line in text.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
        if len(candidates) == 1:
            text = candidates[0]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Termux smoke evidence must be a JSON object")
    return payload


def verify_termux_smoke_evidence(path: str | Path, *, require_termux: bool = True) -> dict[str, object]:
    try:
        payload = _load_json(path)
    except Exception as exc:
        return {
            "source": SOURCE,
            "claim_boundary": VERIFY_CLAIM_BOUNDARY,
            "verification_status": "failed",
            "ok": False,
            "error": f"could not parse evidence: {type(exc).__name__}: {exc}",
            "phone_smoke_proven": False,
            "generation_proven": False,
            "speedup_proven": False,
            "inference_proven": False,
            "can_update_proof_status": False,
        }

    errors: list[str] = []
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("unexpected claim_boundary")
    if payload.get("ok") is not True:
        errors.append("payload ok is not true")
    if payload.get("dashboard_counters") != EXPECTED_COUNTERS:
        errors.append("dashboard counters do not match expected static smoke")
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
        "dashboard_counters": payload.get("dashboard_counters"),
        "termux_detected": bool(isinstance(runtime, dict) and runtime.get("is_termux") is True),
        "phone_smoke_proven": passed,
        "generation_proven": False,
        "speedup_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
        "next_step": "measure repeated phone bridge latency and compare against verifier-only baseline" if passed else "rerun the Termux smoke script on the phone and capture the single JSON line",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Write or print the Termux smoke shell script")
    render.add_argument("--out", default=None, help="Optional path to write; omit to print script")
    render.add_argument("--json", action="store_true", help="When --out is used, print JSON metadata")

    verify = sub.add_parser("verify", help="Verify captured Termux smoke JSON")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--allow-non-termux", action="store_true", help="Allow local/non-Termux smoke evidence; tests only")

    args = parser.parse_args(argv)
    if args.command == "render":
        if args.out:
            payload = write_termux_smoke_script(args.out)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(payload["local_command"])
        else:
            print(render_termux_smoke_script(), end="")
        return 0
    if args.command == "verify":
        report = verify_termux_smoke_evidence(args.evidence, require_termux=not args.allow_non_termux)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
