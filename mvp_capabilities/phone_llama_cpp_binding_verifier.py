#!/usr/bin/env python3
"""llama-cpp-python verifier for phone GGUF draft text.

This validates phone draft *text bytes* against the target model under the same
chat-template prompt shape that local/Termux `llama-cli --single-turn` rendered
in evidence. It intentionally does not claim that external phone token IDs were
injected into llama.cpp; the current public CLI path has no such input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

CLAIM_BOUNDARY = (
    "phone_draft_llama_cpp_binding_text_prefix_verifier_"
    "no_external_token_ingest_no_speedup_claim"
)
EXTERNAL_TOKEN_CLAIM_BOUNDARY = "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim"
PROMPT_TEMPLATE = "llama_cpp_chat_template_im_start_end"


def render_llama_cpp_chat_prompt(user_prompt: str) -> str:
    """Render the prompt shape observed from `llama-cli --verbose-prompt`."""

    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_from_bytes(chunks: Iterable[bytes]) -> str:
    return b"".join(chunks).decode("utf-8", errors="replace")


def build_text_prefix_verifier_report(
    *,
    prompt: str,
    draft_text: str,
    generated_token_ids: Sequence[int],
    generated_token_bytes: Sequence[bytes],
    context_retokenized_token_ids: Sequence[int] | None = None,
    standalone_draft_token_ids: Sequence[int] | None = None,
    elapsed_s: float,
    model_sha256: str,
    llama_cpp_python_version: str,
    model_id: str = "ggml-org/tiny-llamas/stories15M.gguf",
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed verifier report from context-generated target tokens."""

    proposed = draft_text.encode("utf-8")
    accepted = bytearray()
    accepted_tokens = 0
    mismatch: dict[str, Any] | None = None

    for index, piece in enumerate(generated_token_bytes):
        candidate = bytes(accepted) + piece
        if proposed.startswith(candidate):
            accepted[:] = candidate
            accepted_tokens += 1
            if len(accepted) == len(proposed):
                break
            continue

        common = 0
        for a, b in zip(candidate, proposed):
            if a != b:
                break
            common += 1
        accepted[:] = candidate[:common]
        mismatch = {
            "token_index": index,
            "token_id": int(generated_token_ids[index]),
            "token_text": piece.decode("utf-8", errors="replace"),
            "common_prefix_bytes_after_token": common,
        }
        break

    accepted_text = bytes(accepted).decode("utf-8", errors="replace")
    proven = len(accepted) == len(proposed)
    generated_ids = [int(token) for token in generated_token_ids]
    context_ids = [
        int(token) for token in (context_retokenized_token_ids or generated_token_ids)
    ]
    standalone_ids = (
        [int(token) for token in standalone_draft_token_ids]
        if standalone_draft_token_ids is not None
        else None
    )
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "prompt": prompt,
        "rendered_prompt": render_llama_cpp_chat_prompt(prompt),
        "prompt_template": PROMPT_TEMPLATE,
        "model_id": model_id,
        "model_path": model_path,
        "model_sha256": model_sha256,
        "llama_cpp_python_version": llama_cpp_python_version,
        "draft_text": draft_text,
        "generated_prefix_text": _text_from_bytes(generated_token_bytes),
        "accepted_text": accepted_text,
        "generated_context_token_ids": generated_ids,
        "context_retokenized_draft_token_ids": context_ids,
        "standalone_draft_token_ids": standalone_ids,
        "context_retokenization_matches_generated": context_ids == generated_ids,
        "standalone_draft_token_ids_known_mismatch": (
            standalone_ids is not None and standalone_ids != context_ids
        ),
        "generated_context_token_texts": [
            token.decode("utf-8", errors="replace") for token in generated_token_bytes
        ],
        "accepted_utf8_byte_count": len(accepted),
        "proposed_utf8_byte_count": len(proposed),
        "rejected_utf8_byte_count": len(proposed) - len(accepted),
        "accepted_generated_token_count": accepted_tokens,
        "generated_token_count": len(generated_token_ids),
        "text_prefix_acceptance_proven": proven,
        "mismatch": mismatch,
        "phone_external_token_ids_ingested": False,
        "phone_integrated_verifier_proven": proven,
        "speedup_proven": False,
        "bloombee_block_serving_proven": False,
        "elapsed_s": round(float(elapsed_s), 6),
        "operator_next_steps": [
            "extend the verifier to ingest external phone draft token IDs or patch llama.cpp CLI for external draft tokens",
            "measure phone-backed draft-plus-verifier wall clock against verifier-only before speedup claims",
        ],
    }


def parse_token_id_list(value: str) -> list[int]:
    """Parse a JSON or comma-separated token-id list."""

    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("token-id JSON must be a list")
        return [int(token) for token in parsed]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_external_context_token_verifier_report(
    *,
    prompt: str,
    draft_text: str,
    phone_context_draft_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    generated_token_bytes: Sequence[bytes],
    elapsed_s: float,
    model_sha256: str,
    llama_cpp_python_version: str,
    phone_token_id_source: str,
    verifier_method: str = "target_token_sequence_comparison",
    logit_checks: Sequence[dict[str, Any]] | None = None,
    model_id: str = "ggml-org/tiny-llamas/stories15M.gguf",
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build a verifier report for externally supplied context draft token IDs."""

    phone_ids = [int(token) for token in phone_context_draft_token_ids]
    generated_ids = [int(token) for token in generated_token_ids]
    accepted_ids: list[int] = []
    accepted_bytes: list[bytes] = []
    mismatch: dict[str, Any] | None = None

    for index, phone_token in enumerate(phone_ids):
        if index >= len(generated_ids):
            mismatch = {
                "token_index": index,
                "phone_token_id": phone_token,
                "target_token_id": None,
                "reason": "target_generation_ended_before_phone_token",
            }
            break
        target_token = generated_ids[index]
        target_piece = generated_token_bytes[index]
        if phone_token == target_token:
            accepted_ids.append(phone_token)
            accepted_bytes.append(target_piece)
            continue
        mismatch = {
            "token_index": index,
            "phone_token_id": phone_token,
            "target_token_id": target_token,
            "target_token_text": target_piece.decode("utf-8", errors="replace"),
            "reason": "phone_token_id_did_not_match_verifier_greedy_token",
        }
        break

    accepted_text = _text_from_bytes(accepted_bytes)
    accepted = len(accepted_ids)
    proposed = len(phone_ids)
    proven = proposed > 0 and accepted == proposed and mismatch is None
    return {
        "claim_boundary": EXTERNAL_TOKEN_CLAIM_BOUNDARY,
        "prompt": prompt,
        "rendered_prompt": render_llama_cpp_chat_prompt(prompt),
        "prompt_template": PROMPT_TEMPLATE,
        "model_id": model_id,
        "model_path": model_path,
        "model_sha256": model_sha256,
        "llama_cpp_python_version": llama_cpp_python_version,
        "verifier_method": verifier_method,
        "draft_text": draft_text,
        "phone_external_token_id_source": phone_token_id_source,
        "phone_context_draft_token_ids": phone_ids,
        "generated_context_token_ids": generated_ids,
        "generated_context_token_texts": [
            token.decode("utf-8", errors="replace") for token in generated_token_bytes
        ],
        "accepted_context_token_ids": accepted_ids,
        "accepted_text": accepted_text,
        "accepted_external_token_count": accepted,
        "proposed_external_token_count": proposed,
        "rejected_external_token_count": proposed - accepted,
        "external_context_token_id_acceptance_proven": proven,
        "phone_external_token_ids_ingested": True,
        "phone_integrated_verifier_proven": proven,
        "logit_checks": list(logit_checks or []),
        "mismatch": mismatch,
        "speedup_proven": False,
        "bloombee_block_serving_proven": False,
        "elapsed_s": round(float(elapsed_s), 6),
        "operator_next_steps": [
            "replace the same-device verifier loop with a network bridge that receives phone token IDs live",
            "measure phone-backed draft-plus-verifier wall clock against verifier-only before speedup claims",
        ],
    }


def verify_external_context_tokens_with_llama_cpp(
    *,
    model_path: Path,
    prompt: str,
    draft_text: str,
    phone_context_draft_token_ids: Sequence[int],
    phone_token_id_source: str = "termux_llama_tokenize_context_suffix",
    n_ctx: int = 64,
    n_threads: int = 4,
    n_gpu_layers: int = 0,
) -> dict[str, Any]:
    """Consume externally supplied context draft token IDs and compare target logits.

    Forced-batch verification evaluates ``prompt_tokens + external_ids`` with
    ``logits_all=True`` and checks each supplied token against the target model's
    greedy argmax at the corresponding previous-token score row. Verification
    stops at the first mismatch because later rows are conditioned on rejected
    external tokens and must not count toward acceptance.
    """

    from llama_cpp import Llama
    import llama_cpp
    import numpy as np

    model_path = Path(model_path)
    start = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_batch=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )
    rendered = render_llama_cpp_chat_prompt(prompt)
    prompt_tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=True)
    phone_ids = [int(token) for token in phone_context_draft_token_ids]
    if len(prompt_tokens) + len(phone_ids) > n_ctx:
        raise ValueError(
            "prompt plus phone context draft tokens exceeds n_ctx: "
            f"{len(prompt_tokens)} + {len(phone_ids)} > {n_ctx}"
        )
    llm.eval(prompt_tokens + phone_ids)

    generated_ids: list[int] = []
    generated_bytes: list[bytes] = []
    logit_checks: list[dict[str, Any]] = []
    accepted_ids: list[int] = []
    for index, phone_token in enumerate(phone_ids):
        row = len(prompt_tokens) - 1 + index
        logits = llm.scores[row]
        target_token = int(np.argmax(logits))
        target_piece = bytes(
            llm.detokenize([target_token], prev_tokens=prompt_tokens + accepted_ids)
        )
        generated_ids.append(target_token)
        generated_bytes.append(target_piece)
        accepted = phone_token == target_token
        logit_checks.append(
            {
                "token_index": index,
                "score_row": row,
                "phone_token_id": phone_token,
                "target_argmax_token_id": target_token,
                "target_token_text": target_piece.decode("utf-8", errors="replace"),
                "phone_token_logit": float(logits[phone_token]),
                "target_argmax_logit": float(logits[target_token]),
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        accepted_ids.append(phone_token)

    return build_external_context_token_verifier_report(
        prompt=prompt,
        draft_text=draft_text,
        phone_context_draft_token_ids=phone_context_draft_token_ids,
        generated_token_ids=generated_ids,
        generated_token_bytes=generated_bytes,
        elapsed_s=time.perf_counter() - start,
        model_sha256=_sha256(model_path),
        llama_cpp_python_version=getattr(llama_cpp, "__version__", "unknown"),
        phone_token_id_source=phone_token_id_source,
        verifier_method="forced_batch_logits_all_argmax",
        logit_checks=logit_checks,
        model_path=str(model_path),
    )


def verify_draft_text_with_llama_cpp(
    *,
    model_path: Path,
    prompt: str,
    draft_text: str,
    max_tokens: int | None = None,
    n_ctx: int = 64,
    n_threads: int = 4,
    n_gpu_layers: int = 0,
) -> dict[str, Any]:
    """Run the local llama-cpp-python target verifier against phone draft text."""

    from llama_cpp import Llama
    import llama_cpp

    model_path = Path(model_path)
    max_tokens = max_tokens or max(1, len(draft_text.encode("utf-8")))
    start = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    rendered = render_llama_cpp_chat_prompt(prompt)
    prompt_tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=True)
    full_prompt_with_draft_tokens = llm.tokenize(
        (rendered + draft_text).encode("utf-8"), add_bos=True
    )
    context_retokenized_token_ids = full_prompt_with_draft_tokens[len(prompt_tokens) :]
    standalone_draft_token_ids = llm.tokenize(draft_text.encode("utf-8"), add_bos=False)
    llm.eval(prompt_tokens)

    proposed = draft_text.encode("utf-8")
    accepted = bytearray()
    generated_ids: list[int] = []
    generated_bytes: list[bytes] = []

    for _ in range(max_tokens):
        token = int(llm.sample(temp=0.0, top_k=0, top_p=1.0, min_p=0.0))
        piece = bytes(llm.detokenize([token]))
        generated_ids.append(token)
        generated_bytes.append(piece)
        candidate = bytes(accepted) + piece
        if proposed.startswith(candidate):
            accepted[:] = candidate
            llm.eval([token])
            if len(accepted) >= len(proposed):
                break
            continue
        # Evaluate no further after first non-prefix token; the draft is rejected here.
        break

    return build_text_prefix_verifier_report(
        prompt=prompt,
        draft_text=draft_text,
        generated_token_ids=generated_ids,
        generated_token_bytes=generated_bytes,
        context_retokenized_token_ids=context_retokenized_token_ids,
        standalone_draft_token_ids=standalone_draft_token_ids,
        elapsed_s=time.perf_counter() - start,
        model_sha256=_sha256(model_path),
        llama_cpp_python_version=getattr(llama_cpp, "__version__", "unknown"),
        model_path=str(model_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--draft-text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--phone-context-token-ids",
        default=None,
        help="JSON or comma-separated context-token IDs emitted by the phone for the draft suffix.",
    )
    parser.add_argument("--phone-token-id-source", default="termux_llama_tokenize_context_suffix")
    args = parser.parse_args()

    if args.phone_context_token_ids:
        report = verify_external_context_tokens_with_llama_cpp(
            model_path=Path(args.model),
            prompt=args.prompt,
            draft_text=args.draft_text,
            phone_context_draft_token_ids=parse_token_id_list(args.phone_context_token_ids),
            phone_token_id_source=args.phone_token_id_source,
        )
    else:
        report = verify_draft_text_with_llama_cpp(
            model_path=Path(args.model),
            prompt=args.prompt,
            draft_text=args.draft_text,
            max_tokens=args.max_tokens,
        )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": report.get("accepted_utf8_byte_count", report.get("accepted_external_token_count")),
                "proposed": report.get("proposed_utf8_byte_count", report.get("proposed_external_token_count")),
                "tokens": report.get("accepted_generated_token_count", report.get("accepted_external_token_count")),
                "proven": report.get("text_prefix_acceptance_proven", report.get("external_context_token_id_acceptance_proven")),
                "phone_external_token_ids_ingested": report["phone_external_token_ids_ingested"],
                "speedup_proven": report["speedup_proven"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
