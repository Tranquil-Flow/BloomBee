from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = PROJECT_ROOT / "mvp_capabilities/distributed_evidence/post_mvp/kv-prefix-reuse-planner-20260704.json"


def test_kv_prefix_reuse_planner_reuses_longest_prior_prefix_and_reconstructs_requests():
    from mvp_capabilities.kv_prefix_reuse import PrefixRequest, plan_kv_prefix_reuse

    report = plan_kv_prefix_reuse(
        requests=[
            PrefixRequest(request_id="doc-a", token_ids=(101, 102, 103, 104)),
            PrefixRequest(request_id="doc-b", token_ids=(101, 102, 103, 205)),
            PrefixRequest(request_id="other", token_ids=(7, 8)),
        ],
        min_reuse_tokens=2,
    )

    assert report["claim_boundary"] == "kv_prefix_reuse_planner_simulation_no_live_cache_proof"
    assert report["request_count"] == 3
    assert report["total_original_prefill_tokens"] == 10
    assert report["total_planned_prefill_tokens"] == 7
    assert report["saved_prefill_tokens"] == 3
    assert report["reuse_event_count"] == 1
    assert report["all_reconstructions_match"] is True
    assert report["plan"] == [
        {
            "request_id": "doc-a",
            "matched_cache_id": None,
            "reused_token_count": 0,
            "reused_token_ids": [],
            "prefill_token_ids": [101, 102, 103, 104],
            "reconstructed_token_ids": [101, 102, 103, 104],
            "reconstruction_matches": True,
        },
        {
            "request_id": "doc-b",
            "matched_cache_id": "doc-a:full",
            "reused_token_count": 3,
            "reused_token_ids": [101, 102, 103],
            "prefill_token_ids": [205],
            "reconstructed_token_ids": [101, 102, 103, 205],
            "reconstruction_matches": True,
        },
        {
            "request_id": "other",
            "matched_cache_id": None,
            "reused_token_count": 0,
            "reused_token_ids": [],
            "prefill_token_ids": [7, 8],
            "reconstructed_token_ids": [7, 8],
            "reconstruction_matches": True,
        },
    ]
    assert report["live_kv_cache_reuse_proven"] is False
    assert report["live_server_proven"] is False
    assert report["speedup_proven"] is False
    assert report["can_update_demo_status"] is False


def test_kv_prefix_reuse_rejects_non_prefix_overlap_even_when_tokens_match_later():
    from mvp_capabilities.kv_prefix_reuse import PrefixCacheEntry, PrefixRequest, plan_kv_prefix_reuse

    report = plan_kv_prefix_reuse(
        requests=[PrefixRequest(request_id="candidate", token_ids=(1, 2, 3, 4))],
        initial_cache_entries=[PrefixCacheEntry(cache_id="contains-but-not-prefix", token_ids=(9, 1, 2, 3))],
        min_reuse_tokens=2,
    )

    assert report["saved_prefill_tokens"] == 0
    assert report["plan"][0]["matched_cache_id"] is None
    assert report["plan"][0]["prefill_token_ids"] == [1, 2, 3, 4]
    assert report["plan"][0]["reconstruction_matches"] is True


def test_kv_prefix_reuse_cli_and_tracked_evidence_are_claim_bounded():
    proc = subprocess.run(
        [sys.executable, "mvp_capabilities/kv_prefix_reuse.py", "--example", "shared-doc-prefix"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    tracked = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["plan"] == tracked["plan"]
    assert tracked["verification_status"] == "passed"
    assert tracked["claim_boundary"] == "kv_prefix_reuse_planner_simulation_no_live_cache_proof"
    assert tracked["saved_prefill_tokens"] == 3
    assert tracked["total_planned_prefill_tokens"] == 7
    assert tracked["all_reconstructions_match"] is True
    assert tracked["live_kv_cache_reuse_proven"] is False
    assert tracked["live_server_proven"] is False
    assert tracked["speedup_proven"] is False
    assert tracked["operator_next_steps"] == [
        "wire prefix lookup into real prefill/session cache metadata behind an opt-in flag",
        "prove hidden-state/token parity for reused-prefix and full-prefill paths",
        "measure memory and wall-clock impact before any demo or speedup promotion",
    ]
