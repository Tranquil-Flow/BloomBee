"""Repo-backed remaining-work checklist for post-MVP tasks.

This module intentionally derives its list from ``mvp_status.build_status_report``
so operator handoffs do not drift from the status dashboard. It creates no new
proof and does not promote any model/route/demo status.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvp_capabilities.mvp_status import build_status_report

CLAIM_BOUNDARY = "status_derived_remaining_work_no_new_proof"
SOURCE = "mvp_capabilities.mvp_status.build_status_report"


def _remaining_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for task in report.get("post_mvp_tasks", []):
        status = task.get("status")
        if status == "complete":
            continue
        item = {
            "id": task.get("id"),
            "label": task.get("label"),
            "status": status,
            "done": False,
            "blocked": status == "blocked",
            "evidence": task.get("evidence"),
            "next_step": task.get("next_step"),
        }
        items.append(item)
    return items


def build_checklist() -> dict[str, Any]:
    report = build_status_report()
    items = _remaining_items(report)
    by_status = dict(Counter(item["status"] for item in items))
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "source": SOURCE,
        "core_mvp_complete": report.get("core_tasks_complete") is True,
        "mvp_bar": report.get("overall_bar"),
        "task_summary": report.get("task_summary"),
        "post_mvp_task_summary": report.get("post_mvp_task_summary"),
        "remaining_count": len(items),
        "by_status": by_status,
        "items": items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Remaining work checklist",
        "",
        f"**Claim boundary:** `{payload['claim_boundary']}`",
        f"**Source:** `{payload['source']}`",
        f"**MVP core:** `{payload['mvp_bar']}`; core complete = `{payload['core_mvp_complete']}`",
        f"**Remaining post-MVP items:** `{payload['remaining_count']}` ({payload['by_status']})",
        "",
        "No new proof is created by this checklist; it only renders current status-derived blockers and next steps.",
        "",
    ]
    for item in payload["items"]:
        status = item["status"]
        lines.append(f"- [ ] `{item['id']}` — **{status}** — {item['label']}")
        if item.get("next_step"):
            lines.append(f"  - Next: {item['next_step']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args(argv)

    payload = build_checklist()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_markdown(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
