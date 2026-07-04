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
    args = parser.parse_args()

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
                "accepted": report["accepted_utf8_byte_count"],
                "proposed": report["proposed_utf8_byte_count"],
                "tokens": report["accepted_generated_token_count"],
                "proven": report["text_prefix_acceptance_proven"],
                "speedup_proven": report["speedup_proven"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
