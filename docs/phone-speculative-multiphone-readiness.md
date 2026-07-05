# Phone speculative decoding: 3-4 phone readiness gate

**Status:** preparation gate only. This document and `mvp_capabilities/multi_phone_speculative_readiness.py` make tomorrow's 3-4 phones test reproducible and fail-closed.

## Claim boundary

The multi-phone readiness manifest proves only that each phone produced the prerequisite draft-token artifacts needed to attempt an integrated speculative-decoding wall-clock test.

It does **not** prove:

- a true `speedup_proven` flag;
- BloomBee block serving on phones;
- live network streaming token transport;
- GLM/MiniMax/DeepSeek output quality.

Until an integrated verifier consumes phone tokens without rerunning verifier-only decode and measures faster wall clock, every aggregate report must keep `speedup_proven=false` and `can_update_speculative_speedup_status=false`.

## Per-phone required artifact shape

Each physical phone should get one wrapper JSON with:

```json
{
  "phone_id": "pixel-a",
  "phone_model": "Pixel 8 Pro",
  "runtime": "termux",
  "termux_context_token_artifact": "mvp_capabilities/distributed_evidence/phone/pixel-a-termux-context-token-ids.json",
  "context_token_verifier": {
    "claim_boundary": "phone_context_token_id_llama_cpp_binding_verifier_no_speedup_claim",
    "model_id": "ggml-org/tiny-llamas/stories15M.gguf",
    "model_sha256": "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a",
    "phone_external_token_ids_ingested": true,
    "phone_integrated_verifier_proven": true,
    "external_context_token_id_acceptance_proven": true,
    "accepted_external_token_count": 8,
    "proposed_external_token_count": 8,
    "speedup_proven": false,
    "bloombee_block_serving_proven": false
  },
  "wallclock_gate": {
    "claim_boundary": "phone_speculative_wallclock_gate_fail_closed",
    "verifier_acceptance_proven": true,
    "tokenizer_id_match_proven": true,
    "speedup_proven": false,
    "wallclock_speedup_proven": false,
    "can_update_speculative_speedup_status": false
  }
}
```

## Command

After collecting 3-4 phones, run:

```bash
python mvp_capabilities/multi_phone_speculative_readiness.py \
  --min-phone-count 3 \
  --max-phone-count 4 \
  --phone-artifact .local/phone/pixel-a-ready.json \
  --phone-artifact .local/phone/pixel-b-ready.json \
  --phone-artifact .local/phone/pixel-c-ready.json \
  --out .local/phone/multi-phone-speculative-readiness.json
```

Add a fourth `--phone-artifact` when present.

The command exits `0` only when:

- phone count is within 3-4 phones;
- phone IDs are distinct;
- every phone uses the same GGUF `model_sha256`;
- every phone has context-token ingestion accepted;
- every phone has a wall-clock gate with verifier acceptance + tokenizer match;
- no input artifact claims speculative speedup yet.

## Tomorrow execution order

1. Run Termux context-token emission on each phone: tokenize the rendered prompt and rendered prompt + draft, then slice prompt IDs off the full token list.
2. Pull/copy each phone context-token JSON.
3. Run `phone_llama_cpp_binding_verifier.py` for each phone against the exact same GGUF hash.
4. Run `phone_speculative_wallclock_gate.py` for each phone.
5. Wrap those two reports into one per-phone artifact.
6. Run `multi_phone_speculative_readiness.py --phone-artifact ...` for all devices.
7. Only after the manifest passes, run the integrated non-sequential verifier path and compare verifier-only vs phone-draft-plus-verifier wall clock.

## Failure interpretation

- `duplicate_phone_id:*` — two artifacts describe the same phone; fix labels before testing.
- `phone_count_below_min:*` — not enough physical devices for tomorrow's planned 3-4 phone test.
- `model_sha256_mismatch` — phones are not using the same GGUF; do not compare timings.
- `context_token_ingestion_not_proven` — rerun Termux token emission and binding verifier.
- `wallclock_correctness_not_proven` — rerun `phone_speculative_wallclock_gate.py`.
- `unexpected_speedup_claim` — reject the artifact; readiness prep must not claim speedup.
