#!/usr/bin/env python3
"""Render and verify a Termux llama.cpp GGUF draft-provider bridge smoke.

This is intentionally a draft-provider *bridge* proof only: Termux runs a tiny
GGUF model through `llama-cli` and returns generated text in a JSON envelope.
It does not prove that an authoritative verifier accepted those tokens, does not
prove speculative speedup, and does not prove BloomBee block serving.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "termux_tiny_gguf_draft_bridge_smoke_no_verifier_acceptance_no_speedup_claim"
MODEL_ID = "ggml-org/tiny-llamas/stories15M.gguf"
MODEL_SHA256 = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"
DEFAULT_MODEL_PATH = "/data/data/com.termux/files/home/bloombee-models/stories15M.gguf"
DEFAULT_OUTPUT = "/sdcard/Download/bloombee-gguf-draft-bridge-output.json"


def render_termux_gguf_draft_bridge_script(
    *,
    prompt: str = "Once upon a time",
    n_predict: int = 8,
    model_path: str = DEFAULT_MODEL_PATH,
    output_path: str = DEFAULT_OUTPUT,
) -> str:
    """Return a pasteable/pushable Termux script for one GGUF draft response."""
    if n_predict <= 0:
        raise ValueError("n_predict must be positive")
    prompt_q = shlex.quote(prompt)
    model_q = shlex.quote(model_path)
    out_q = shlex.quote(output_path)
    script = """#!/data/data/com.termux/files/usr/bin/sh
set -eu
CLAIM_BOUNDARY=__CLAIM_BOUNDARY__
PROMPT=__PROMPT__
N_PREDICT=__N_PREDICT__
MODEL_PATH=__MODEL_PATH__
OUT=__OUT__
STDOUT="$OUT.stdout.txt"
STDERR="$OUT.stderr.txt"
STATUS="$OUT.status"
echo running > "$STATUS"
START=$(python - <<'PY'
import time
print(time.perf_counter())
PY
)
RC=0
timeout 90s llama-cli -m "$MODEL_PATH" -p "$PROMPT" -n "$N_PREDICT" --ctx-size 64 --threads 4 --temp 0 --no-display-prompt --single-turn --simple-io --no-repack --log-disable > "$STDOUT" 2> "$STDERR" || RC=$?
END=$(python - <<'PY'
import time
print(time.perf_counter())
PY
)
python - "$OUT" "$MODEL_PATH" "$PROMPT" "$N_PREDICT" "$RC" "$START" "$END" "$STDOUT" "$STDERR" "$CLAIM_BOUNDARY" <<'PY'
import hashlib, json, os, platform, re, subprocess, sys
out, model_path, prompt, n_predict, rc, start, end, stdout_path, stderr_path, claim_boundary = sys.argv[1:]

def prop(name):
    try:
        return subprocess.check_output(["getprop", name], text=True, timeout=5).strip()
    except Exception:
        return ""

def sha256(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
stdout=open(stdout_path, encoding='utf-8', errors='replace').read() if os.path.exists(stdout_path) else ''
stderr=open(stderr_path, encoding='utf-8', errors='replace').read() if os.path.exists(stderr_path) else ''
rc_int=int(rc)
elapsed=float(end)-float(start)
match = re.search(r'> ' + re.escape(prompt) + r'\\s+(.+?)\\s+\\[ Prompt:', stdout, re.S)
generated = match.group(1).strip() if match else ''
if not generated:
    # Fall back to last non-banner content while keeping raw stdout for audit.
    lines=[line.strip() for line in stdout.splitlines() if line.strip() and not line.startswith(('▄▄','██','build','model','modalities','available commands','/','>','[ Prompt','Exiting','Loading model'))]
    generated='\\n'.join(lines[-4:])
passed = rc_int == 0 and bool(generated)
report={
  "source":"termux_gguf_draft_bridge.sh",
  "claim_boundary": claim_boundary,
  "verification_status":"passed" if passed else "failed",
  "phone_runtime": {"runtime":"termux", "is_termux": os.environ.get('PREFIX','').startswith('/data/data/com.termux'), "android_model": prop('ro.product.model'), "android_sdk": prop('ro.build.version.sdk'), "soc": prop('ro.soc.model'), "python": platform.python_version(), "machine": platform.machine()},
  "model": {"id":"ggml-org/tiny-llamas/stories15M.gguf", "path": model_path, "exists": os.path.exists(model_path), "size_bytes": os.path.getsize(model_path) if os.path.exists(model_path) else 0, "sha256": sha256(model_path) if os.path.exists(model_path) else None},
  "draft_request": {"prompt": prompt, "n_predict": int(n_predict), "role":"draft_provider_candidate"},
  "draft_response": {"returncode": rc_int, "elapsed_s": round(elapsed, 6), "generated_text": generated, "stdout": stdout, "stderr_tail": stderr[-4000:]},
  "generation_proven": passed,
  "verifier_acceptance_proven": False,
  "speedup_proven": False,
  "bloombee_block_serving_proven": False,
  "can_update_speculative_speedup_status": False,
  "can_update_bloombee_block_worker_status": False,
}
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, sort_keys=True)
print(json.dumps({"verification_status": report["verification_status"], "generated_text": generated}, sort_keys=True))
PY
echo finished > "$STATUS"
"""
    return (
        script.replace("__CLAIM_BOUNDARY__", shlex.quote(CLAIM_BOUNDARY))
        .replace("__PROMPT__", prompt_q)
        .replace("__N_PREDICT__", str(int(n_predict)))
        .replace("__MODEL_PATH__", model_q)
        .replace("__OUT__", out_q)
    )


def verify_termux_gguf_draft_bridge_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify a phone GGUF draft bridge payload without overclaiming speedup."""
    failed: list[str] = []
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        failed.append("claim boundary mismatch")
    runtime = payload.get("phone_runtime") or {}
    if runtime.get("is_termux") is not True:
        failed.append("payload is not from Termux")
    model = payload.get("model") or {}
    if model.get("id") != MODEL_ID:
        failed.append("model id mismatch")
    if model.get("exists") is not True:
        failed.append("model file missing")
    if model.get("sha256") != MODEL_SHA256:
        failed.append("model sha256 mismatch")
    request = payload.get("draft_request") or {}
    if not request.get("prompt"):
        failed.append("missing draft prompt")
    if int(request.get("n_predict") or 0) <= 0:
        failed.append("missing positive n_predict")
    response = payload.get("draft_response") or {}
    if response.get("returncode") != 0:
        failed.append("llama-cli return code was not zero")
    if not str(response.get("generated_text") or "").strip():
        failed.append("generated text was empty")
    if payload.get("generation_proven") is not True:
        failed.append("generation_proven was not true")
    if payload.get("verifier_acceptance_proven") is not False:
        failed.append("verifier acceptance must remain false")
    if payload.get("speedup_proven") is not False:
        failed.append("speedup must remain false")
    if payload.get("bloombee_block_serving_proven") is not False:
        failed.append("BloomBee block serving must remain false")

    status = "passed" if not failed else "failed"
    return {
        "source": "termux_gguf_draft_bridge.py",
        "claim_boundary": "termux_tiny_gguf_draft_bridge_verifier_no_speedup_claim",
        "verification_status": status,
        "failed_checks": failed,
        "phone_tiny_gguf_draft_bridge_proven": status == "passed",
        "generated_text": (payload.get("draft_response") or {}).get("generated_text"),
        "can_update_speculative_speedup_status": False,
        "can_update_bloombee_block_worker_status": False,
        "speedup_proven": False,
        "bloombee_block_serving_proven": False,
        "evidence": payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("--out", required=True)
    render.add_argument("--prompt", default="Once upon a time")
    render.add_argument("--n-predict", type=int, default=8)
    render.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    render.add_argument("--output-path", default=DEFAULT_OUTPUT)
    verify = sub.add_parser("verify")
    verify.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    if args.command == "render":
        script = render_termux_gguf_draft_bridge_script(
            prompt=args.prompt,
            n_predict=args.n_predict,
            model_path=args.model_path,
            output_path=args.output_path,
        )
        Path(args.out).expanduser().write_text(script, encoding="utf-8")
        print(json.dumps({"claim_boundary": CLAIM_BOUNDARY, "out": args.out, "generation_proven": False, "speedup_proven": False}, indent=2, sort_keys=True))
    else:
        payload = json.loads(Path(args.evidence).expanduser().read_text(encoding="utf-8"))
        print(json.dumps(verify_termux_gguf_draft_bridge_evidence(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
