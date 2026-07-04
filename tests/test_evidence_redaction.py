from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evidence_secret_scanner_flags_raw_join_material(tmp_path: Path):
    from mvp_capabilities.evidence_redaction import scan_evidence_tree

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "token": "moon-token",
                "join_url": "bloombee://join?coordinator=http%3A%2F%2Fm4pro.local%3A8787&token=moon-token",
                "raw_join_url": "bloombee://join?token=moon-token",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    allowed = tmp_path / "allowed.json"
    allowed.write_text(
        json.dumps(
            {
                "token_sha256": "a" * 64,
                "join_url_sha256": "b" * 64,
                "raw_join_url_recorded_in_scratch_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    findings = scan_evidence_tree(tmp_path)

    pattern_ids = {item["pattern_id"] for item in findings}
    assert "raw_token_key" in pattern_ids
    assert "raw_join_url_key" in pattern_ids
    assert "raw_join_url_value" in pattern_ids
    assert "raw_token_query" in pattern_ids
    assert all(item["path"].endswith("bad.json") for item in findings)


def test_committed_distributed_evidence_contains_no_raw_join_material():
    from mvp_capabilities.evidence_redaction import scan_evidence_tree

    findings = scan_evidence_tree(PROJECT_ROOT / "mvp_capabilities/distributed_evidence")
    assert findings == []
