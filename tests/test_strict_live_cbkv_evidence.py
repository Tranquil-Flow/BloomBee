from __future__ import annotations


def _raw_fixture() -> dict:
    return {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "baseline": [
            {
                "row": 0,
                "output_ids": [101, 10, 11],
                "report": {
                    "tick_batches": [
                        {
                            "tick": 0,
                            "output_token_ids": [10],
                            "output_logits_sha256": ["base-a0"],
                            "output_logits_values": [[0.0, 2.0, 1.0]],
                        },
                        {
                            "tick": 1,
                            "output_token_ids": [11],
                            "output_logits_sha256": ["base-a1"],
                            "output_logits_values": [[1.0, 0.0, 2.0]],
                        },
                    ]
                },
            },
            {
                "row": 1,
                "output_ids": [201, 20, 21],
                "report": {
                    "tick_batches": [
                        {
                            "tick": 0,
                            "output_token_ids": [20],
                            "output_logits_sha256": ["base-b0"],
                            "output_logits_values": [[2.0, 0.0, 1.0]],
                        },
                        {
                            "tick": 1,
                            "output_token_ids": [21],
                            "output_logits_sha256": ["base-b1"],
                            "output_logits_values": [[0.0, 1.0, 2.0]],
                        },
                    ]
                },
            },
        ],
        "live_continuous_report": {
            "live_server_proven": True,
            "server_observed_live_continuous_batches": True,
            "tick_batches": [
                {
                    "tick": 0,
                    "request_ids": ["generate-0", "generate-1"],
                    "active_mask": [True, False],
                    "output_token_ids": [10, 20],
                    "output_logits_sha256": ["live-a0", "live-b0-inactive"],
                    "output_logits_values": [[0.0, 2.001, 1.0], [9.0, 9.0, 9.0]],
                },
                {
                    "tick": 1,
                    "request_ids": ["generate-0", "generate-1"],
                    "active_mask": [True, True],
                    "output_token_ids": [11, 20],
                    "output_logits_sha256": ["live-a1", "live-b0"],
                    "output_logits_values": [[1.0, 0.0, 2.001], [2.001, 0.0, 1.0]],
                },
                {
                    "tick": 2,
                    "request_ids": ["generate-0", "generate-1"],
                    "active_mask": [False, True],
                    "output_token_ids": [11, 21],
                    "output_logits_sha256": ["live-a-inactive", "live-b1"],
                    "output_logits_values": [[9.0, 9.0, 9.0], [0.0, 1.0, 2.001]],
                },
            ],
        },
    }


def test_build_continuous_verifier_payload_computes_numeric_logit_drift_from_full_values():
    from mvp_capabilities.strict_live_cbkv_evidence import build_continuous_verifier_payload

    payload = build_continuous_verifier_payload(_raw_fixture())

    assert payload["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert payload["opt_in_enabled"] is True
    assert payload["server_observed_live_continuous_batches"] is True
    assert payload["requests"][0]["arrival_tick"] == 0
    assert payload["requests"][1]["arrival_tick"] == 1
    assert payload["requests"][0]["baseline"]["generated_token_ids"] == [10, 11]
    assert payload["requests"][0]["continuous"]["generated_token_ids"] == [10, 11]
    assert payload["requests"][1]["continuous"]["generated_token_ids"] == [20, 21]
    assert payload["requests"][0]["logits_numeric_comparison"] == {
        "max_abs_diff": 0.001,
        "mean_abs_diff": 0.000333,
        "argmax_token_id_match": True,
        "top1_token_id_match": True,
    }


def test_build_continuous_verifier_payload_omits_numeric_comparison_without_full_values():
    from mvp_capabilities.strict_live_cbkv_evidence import build_continuous_verifier_payload

    raw = _raw_fixture()
    for batch in raw["live_continuous_report"]["tick_batches"]:
        batch.pop("output_logits_values", None)
    for item in raw["baseline"]:
        for batch in item["report"]["tick_batches"]:
            batch.pop("output_logits_values", None)

    payload = build_continuous_verifier_payload(raw)

    assert "logits_numeric_comparison" not in payload["requests"][0]
