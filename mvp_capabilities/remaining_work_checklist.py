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


def _blocker_classification(task: dict[str, Any]) -> tuple[str, list[str], bool]:
    """Return claim-bounded blocker metadata for a remaining status item.

    This is intentionally conservative and status-derived: it does not probe
    hardware, phones, or live servers. It only turns already-recorded next gates
    into explicit operator/hardware blocker categories so a handoff can tell
    whether autonomous code work remains.
    """
    task_id = task.get("id")
    text = " ".join(
        str(value or "")
        for value in (task.get("evidence"), task.get("next_step"), task.get("status"))
    ).lower()

    if task_id == "qwen35b_candidate" or "at least 80gb free memory" in text:
        return "hardware_memory", ["requires_at_least_80gb_free_memory"], True

    if task_id == "minimax_m3_candidate":
        reasons = ["requires_suitable_memory_for_real_weight_oneblock"]
        if "full mtp module proof" in text or "base_decoder_only mtp guard" in text:
            reasons.append("requires_real_weight_or_full_mtp_module_proof")
        return "hardware_memory_or_real_model_proof", reasons, True

    if task_id in {"speculative_decode", "phone_worker"}:
        reasons = []
        if "ios" in text:
            reasons.append("requires_ios_artifact")
        if "3-4 phone" in text or "phone_count_below_min" in text or "three" in text:
            reasons.append("requires_three_or_more_ready_phones")
        if "wall-clock" in text or "wallclock" in text or "speedup" in text:
            reasons.append("requires_integrated_non_sequential_wallclock_speedup")
        return "human_operator_devices", sorted(set(reasons)), True

    if task.get("status") == "blocked":
        return "status_blocked", ["status_marked_blocked"], True

    return "autonomous_followup_needed", [], False


def _remaining_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for task in report.get("post_mvp_tasks", []):
        status = task.get("status")
        if status == "complete":
            continue
        blocker_category, blocker_reasons, requires_human_or_hardware = _blocker_classification(task)
        item = {
            "id": task.get("id"),
            "label": task.get("label"),
            "status": status,
            "done": False,
            "blocked": status == "blocked",
            "requires_human_or_hardware": requires_human_or_hardware,
            "blocker_category": blocker_category,
            "blocker_reasons": blocker_reasons,
            "evidence": task.get("evidence"),
            "next_step": task.get("next_step"),
        }
        items.append(item)
    return items


def build_checklist() -> dict[str, Any]:
    report = build_status_report()
    items = _remaining_items(report)
    by_status = dict(Counter(item["status"] for item in items))
    by_blocker_category = dict(Counter(item["blocker_category"] for item in items))
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "source": SOURCE,
        "core_mvp_complete": report.get("core_tasks_complete") is True,
        "mvp_bar": report.get("overall_bar"),
        "task_summary": report.get("task_summary"),
        "post_mvp_task_summary": report.get("post_mvp_task_summary"),
        "remaining_count": len(items),
        "by_status": by_status,
        "all_remaining_require_human_or_hardware": all(
            item["requires_human_or_hardware"] for item in items
        ),
        "by_blocker_category": by_blocker_category,
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
        f"**Human/hardware blocked:** `{payload['all_remaining_require_human_or_hardware']}` ({payload['by_blocker_category']})",
        "",
        "No new proof is created by this checklist; it only renders current status-derived blockers and next steps.",
        "All remaining items require human/operator hardware or suitable-memory proof gates before further promotion.",
        "",
    ]
    for item in payload["items"]:
        status = item["status"]
        lines.append(f"- [ ] `{item['id']}` — **{status}** — {item['label']}")
        lines.append(
            f"  - Blocker: `{item['blocker_category']}`; reasons = `{item['blocker_reasons']}`"
        )
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
