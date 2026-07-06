#!/usr/bin/env python3
"""Generate a static BloomBee distributed-inference demo dashboard.

The dashboard is intentionally dependency-light: it reads the same JSON artifacts
used by the MVP CLI tools and writes one self-contained HTML file. For live demos,
run this script in a loop or use the built-in meta refresh while another process
updates capability/evidence/log files.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from mvp_capabilities.mvp_status import build_status_report
    from mvp_capabilities.request_telemetry import build_request_telemetry
    from mvp_capabilities.route_picker import (
        DEFAULT_REGISTRY,
        load_registry,
        route_report,
        synthetic_m4_laptops as make_synthetic_m4_laptops,
    )
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, roster_document
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/demo_dashboard.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.mvp_status import build_status_report  # type: ignore[no-redef]
    from mvp_capabilities.request_telemetry import build_request_telemetry  # type: ignore[no-redef]
    from mvp_capabilities.route_picker import (  # type: ignore[no-redef]
        DEFAULT_REGISTRY,
        load_registry,
        route_report,
        synthetic_m4_laptops as make_synthetic_m4_laptops,
    )
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, roster_document  # type: ignore[no-redef]


DEFAULT_EVIDENCE_DIR = Path(__file__).with_name("distributed_evidence")
DEFAULT_OUT = Path(".local/demo-dashboard.html")
_TELEMETRY_MARKERS = ("[RECOVERY_EVENT]", "[S2S_PUSH_EVENT]")
TOKEN_STREAM_CLAIM_BOUNDARY = "token_stream_observability_only_no_generation_proof"
_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")


def _read_json(path: str | Path | None, default: Any) -> Any:
    if not path:
        return default
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return default
    try:
        return json.loads(expanded.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_measured_rate(value: Any) -> str:
    try:
        rate = float(value or 0.0)
    except (TypeError, ValueError):
        return "unmeasured"
    if rate <= 0.0:
        return "unmeasured"
    return f"{rate:.2f}"


def _fmt_latency_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "unmeasured"
    if seconds <= 0.0:
        return "unmeasured"
    return f"{seconds:.2f}s"


def _bool_badge(value: Any) -> str:
    if value is True:
        return '<span class="badge ok">yes</span>'
    if value is False:
        return '<span class="badge fail">no</span>'
    return '<span class="badge muted">—</span>'


def load_evidence(evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR) -> list[dict[str, Any]]:
    """Load compact user-facing inference proof rows from evidence JSON files."""
    root = Path(evidence_dir).expanduser()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "file": path.name,
                "ok": payload.get("ok"),
                "mode": payload.get("mode"),
                "model": payload.get("model"),
                "server_to_server": payload.get("server_to_server"),
                "generated_ids_match": payload.get("generated_ids_match"),
                "generated_text_match": payload.get("generated_text_match"),
                "next_token_match": payload.get("next_token_match"),
                "distributed_seconds": payload.get("distributed_seconds"),
                "reference_seconds": payload.get("reference_seconds"),
                "server_count": len(payload.get("server_maddrs") or []),
                "server_placements": payload.get("server_placements") or payload.get("layer_placements") or [],
            }
        )
    return rows


def collect_layer_placements(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect real server/layer ownership metadata from proof evidence."""
    placements: list[dict[str, Any]] = []
    for row in evidence_rows:
        for raw in row.get("server_placements") or []:
            if not isinstance(raw, dict):
                continue
            layers = raw.get("layers")
            if not (
                isinstance(layers, list)
                and len(layers) == 2
                and all(isinstance(item, int) for item in layers)
                and layers[1] > layers[0]
            ):
                continue
            placements.append(
                {
                    "host": raw.get("host") or raw.get("hostname") or "unknown",
                    "layers": layers,
                    "model": row.get("model"),
                    "evidence_file": row.get("file"),
                    "server_maddr": raw.get("server_maddr") or raw.get("maddr"),
                }
            )
    return sorted(placements, key=lambda item: (item.get("model") or "", item["layers"][0], item.get("host") or ""))


def _parse_keyvals(line: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in _TOKEN_RE.finditer(line)}


def parse_telemetry_logs(paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Count structured recovery/S2S events from optional live log files."""
    event_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    examples: list[str] = []
    scanned: list[str] = []
    for raw in paths or []:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        scanned.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not any(marker in line for marker in _TELEMETRY_MARKERS):
                continue
            fields = _parse_keyvals(line)
            event_counts[fields.get("type", "unknown")] += 1
            if fields.get("reason"):
                reason_counts[fields["reason"]] += 1
            if len(examples) < 8:
                examples.append(line.strip())
    return {
        "markers": list(_TELEMETRY_MARKERS),
        "scanned_logs": scanned,
        "event_counts": dict(sorted(event_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "examples": examples,
    }


def _parse_token_stream_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("[TOKEN_STREAM]"):
        stripped = stripped[len("[TOKEN_STREAM]") :].strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        return _parse_keyvals(stripped)
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        event = payload.get("event") or payload.get("type")
        if event in {"generation_start", "token", "generation_end", "token_stream"}:
            return payload
    return None


def _elapsed_seconds(payload: dict[str, Any]) -> float | None:
    if payload.get("elapsed_seconds") is not None:
        return _as_float(payload.get("elapsed_seconds"), 0.0)
    if payload.get("elapsed_s") is not None:
        return _as_float(payload.get("elapsed_s"), 0.0)
    if payload.get("elapsed_ms") is not None:
        return _as_float(payload.get("elapsed_ms"), 0.0) / 1000.0
    return None


def parse_token_stream_logs(paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Parse optional live token JSONL/marker logs for dashboard display.

    This is observability only. Token rows help the operator watch generation
    progress while the proof harness runs, but do not prove generation parity or
    update any proof-status gates.
    """
    scanned: list[str] = []
    by_request: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    for raw in paths or []:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        scanned.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            errors.append({"log": str(path), "message": str(exc)})
            continue
        for line in lines:
            payload = _parse_token_stream_line(line)
            if not payload:
                continue
            event = payload.get("event") or payload.get("type") or "token"
            request_id = str(payload.get("request_id") or payload.get("request") or "default")
            row = by_request.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "model": payload.get("model"),
                    "prompt": payload.get("prompt"),
                    "tokens": [],
                    "generated_text": "",
                    "hosts": [],
                    "layer_ranges": [],
                    "started_at": payload.get("timestamp"),
                    "updated_at": payload.get("timestamp"),
                },
            )
            if payload.get("model") and not row.get("model"):
                row["model"] = payload.get("model")
            if payload.get("prompt") and not row.get("prompt"):
                row["prompt"] = payload.get("prompt")
            if payload.get("timestamp"):
                row["updated_at"] = payload.get("timestamp")
            if event == "generation_start":
                continue
            if event not in {"token", "token_stream"}:
                continue
            token_text = str(payload.get("token_text") if payload.get("token_text") is not None else payload.get("text") or "")
            token = {
                "step": payload.get("step"),
                "token_id": payload.get("token_id"),
                "token_text": token_text,
                "elapsed_seconds": _elapsed_seconds(payload),
                "host": payload.get("host") or payload.get("hostname"),
                "layers": payload.get("layers") or payload.get("block_range"),
            }
            row["tokens"].append(token)
            row["generated_text"] += token_text
            raw_hosts = payload.get("hosts")
            raw_layer_ranges = payload.get("layer_ranges")
            hosts_payload = raw_hosts if isinstance(raw_hosts, list) else []
            layer_ranges_payload = raw_layer_ranges if isinstance(raw_layer_ranges, list) else []
            for host in hosts_payload:
                if host not in row["hosts"]:
                    row["hosts"].append(host)
            for layer_range in layer_ranges_payload:
                if layer_range not in row["layer_ranges"]:
                    row["layer_ranges"].append(layer_range)
            if token.get("host") and token["host"] not in row["hosts"]:
                row["hosts"].append(token["host"])
            if token.get("layers") and token["layers"] not in row["layer_ranges"]:
                row["layer_ranges"].append(token["layers"])

    requests: list[dict[str, Any]] = []
    for row in by_request.values():
        tokens = row.get("tokens") or []
        elapsed = [float(token["elapsed_seconds"]) for token in tokens if token.get("elapsed_seconds")]
        last = tokens[-1] if tokens else {}
        duration = max(elapsed) if elapsed else None
        tokens_per_second = round(len(tokens) / duration, 3) if duration and duration > 0 else None
        requests.append(
            {
                **row,
                "token_count": len(tokens),
                "latest_token_id": last.get("token_id"),
                "latest_token_text": last.get("token_text"),
                "elapsed_seconds": duration,
                "tokens_per_second": tokens_per_second,
            }
        )
    requests.sort(key=lambda item: str(item.get("request_id") or ""))
    token_count = sum(int(item.get("token_count") or 0) for item in requests)
    return {
        "claim_boundary": TOKEN_STREAM_CLAIM_BOUNDARY,
        "scanned_logs": scanned,
        "live_tokens_seen": token_count > 0,
        "request_count": len(requests),
        "token_count": token_count,
        "requests": requests,
        "errors": errors,
        "next_step": "Pass --token-stream-log from a live generation harness and run dashboard watch mode for near-real-time token display.",
    }


def _parse_block_range(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
        return (start, end) if end > start else None
    if isinstance(value, str) and ":" in value:
        left, right = value.split(":", 1)
        try:
            start, end = int(left), int(right)
        except ValueError:
            return None
        return (start, end) if end > start else None
    return None


def build_layers_map(
    *,
    joined_layer_plan: dict[str, Any] | None,
    layer_placements: list[dict[str, Any]],
    multi_block_diagnostics: dict[str, Any] | None,
    chain_schedule: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build visual layer-map groups from plan, proof, and diagnostics artifacts."""
    groups: list[dict[str, Any]] = []
    if joined_layer_plan:
        placement = joined_layer_plan.get("placement") or {}
        health = (chain_schedule or {}).get("peer_health") or {}
        segments: list[dict[str, Any]] = []
        for item in placement.get("assignments") or []:
            parsed = _parse_block_range(item.get("block_range")) or _parse_block_range([item.get("start_layer"), item.get("end_layer")])
            if not parsed:
                continue
            start, end = parsed
            hostname = item.get("hostname") or "unknown"
            peer_health = health.get(hostname) or {}
            segments.append(
                {
                    "hostname": hostname,
                    "start_layer": start,
                    "end_layer": end,
                    "block_range": f"{start}:{end}",
                    "layer_count": end - start,
                    "status": peer_health.get("health_status") or "planned",
                    "utilization_fraction": peer_health.get("utilization_fraction"),
                    "source": "joined_layer_plan",
                }
            )
        if segments:
            groups.append(
                {
                    "source": "joined_layer_plan",
                    "title": "Joined-peer planned layers",
                    "model": joined_layer_plan.get("model_id"),
                    "claim_boundary": joined_layer_plan.get("claim_boundary"),
                    "total_layers": placement.get("num_layers") or max(seg["end_layer"] for seg in segments),
                    "segments": segments,
                }
            )

    if layer_placements:
        segments = []
        for item in layer_placements:
            parsed = _parse_block_range(item.get("layers"))
            if not parsed:
                continue
            start, end = parsed
            segments.append(
                {
                    "hostname": item.get("host") or "unknown",
                    "start_layer": start,
                    "end_layer": end,
                    "block_range": f"{start}:{end}",
                    "layer_count": end - start,
                    "status": "proof_evidence",
                    "evidence_file": item.get("evidence_file"),
                    "source": "proof_evidence",
                }
            )
        if segments:
            groups.append(
                {
                    "source": "proof_evidence",
                    "title": "Proof evidence layers",
                    "model": layer_placements[0].get("model"),
                    "claim_boundary": "layer_map_from_committed_proof_metadata",
                    "total_layers": max(seg["end_layer"] for seg in segments),
                    "segments": segments,
                }
            )

    if multi_block_diagnostics:
        coverage = multi_block_diagnostics.get("coverage") or {}
        segments = []
        for item in multi_block_diagnostics.get("servers") or []:
            parsed = _parse_block_range(item.get("block_range"))
            if not parsed:
                continue
            start, end = parsed
            segments.append(
                {
                    "hostname": f"server-{item.get('server_index')}",
                    "start_layer": start,
                    "end_layer": end,
                    "block_range": f"{start}:{end}",
                    "layer_count": end - start,
                    "status": item.get("health") or "diagnostic",
                    "started": item.get("started"),
                    "has_rpc_evidence": item.get("has_rpc_evidence"),
                    "source": "multi_block_diagnostics",
                }
            )
        if segments:
            groups.append(
                {
                    "source": "multi_block_diagnostics",
                    "title": "Live/diagnostic server layers",
                    "model": multi_block_diagnostics.get("model_id"),
                    "claim_boundary": multi_block_diagnostics.get("claim_boundary"),
                    "total_layers": coverage.get("total_layers") or max(seg["end_layer"] for seg in segments),
                    "segments": segments,
                }
            )
    return {"groups": groups, "group_count": len(groups)}


def _detect_layer_gaps(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Find uncovered layer ranges across all group segments."""
    ranges: list[tuple[int, int, str]] = []  # (start, end, status)
    for group in groups:
        for seg in group.get("segments") or []:
            start = int(seg.get("start_layer") or 0)
            end = int(seg.get("end_layer") or 0)
            status = str(seg.get("status") or "unknown")
            if end > start:
                ranges.append((start, end, status))
    if not ranges:
        return {"gap_count": 0, "gaps": [], "coverage_percent": 0.0}
    ranges.sort(key=lambda x: x[0])
    total = ranges[-1][1]
    if total == 0:
        return {"gap_count": 0, "gaps": [], "coverage_percent": 0.0}
    gaps: list[dict[str, Any]] = []
    gap_start = 0
    proven_count = 0
    for start, end, status in ranges:
        if start > gap_start:
            gaps.append({"start_layer": gap_start, "end_layer": start, "layer_count": start - gap_start, "status": "uncovered"})
        gap_start = max(gap_start, end)
        if status in ("proof_evidence", "healthy", "ready"):
            proven_count += end - start
    if gap_start < total:
        gaps.append({"start_layer": gap_start, "end_layer": total, "layer_count": total - gap_start, "status": "uncovered"})
    coverage = round(proven_count / total * 100, 1) if total > 0 else 0.0
    return {
        "gap_count": len(gaps),
        "gaps": gaps,
        "total_layers": total,
        "proven_layer_count": proven_count,
        "coverage_percent": coverage,
        "all_covered": len(gaps) == 0,
        "status": "fully_covered" if len(gaps) == 0 else "gaps_present",
    }


def _demo_readiness_panel(document: dict[str, Any]) -> str:
    """Emit a compact checklist of what the operator needs for a live demo."""
    items: list[dict[str, str]] = []
    roster = document.get("roster") or {}
    peers = roster.get("peers") or []
    evidence = document.get("evidence") or []
    token_stream = document.get("token_stream") or {}
    layers_map = document.get("layers_map") or {}
    route = document.get("real_route") or {}
    picked = route.get("picked") or route.get("serving") or {}
    gaps = layers_map.get("gaps") or {}

    # Peer count check
    peer_count = len(peers)
    items.append({
        "check": "Connected peers ≥ 2",
        "status": "ready" if peer_count >= 2 else "needed",
        "detail": f"{peer_count} peer{'s' if peer_count != 1 else ''} found",
    })
    # Free memory
    summary = roster.get("summary") or {}
    free_gb = summary.get("free_memory_gb") or 0.0
    items.append({
        "check": f"Swarm free memory ≥ 10 GB",
        "status": "ready" if free_gb >= 10 else "needed",
        "detail": f"{free_gb:.1f} GB free",
    })
    # Route picked
    items.append({
        "check": "Route selected a model",
        "status": "ready" if picked.get("model_id") else "needed",
        "detail": picked.get("model_id") or "no model fits",
    })
    # Layer coverage
    coverage_pct = gaps.get("coverage_percent") or 0.0
    items.append({
        "check": f"Layer map coverage ≥ 50%",
        "status": "ready" if coverage_pct >= 50 else ("partial" if coverage_pct > 0 else "needed"),
        "detail": f"{coverage_pct:.0f}% covered, {gaps.get('gap_count', 0)} gap{'s' if gaps.get('gap_count', 0) != 1 else ''}",
    })
    # Evidence
    evidence_ok = sum(1 for e in evidence if e.get("ok"))
    items.append({
        "check": "Proof evidence ≥ 1 passing",
        "status": "ready" if evidence_ok >= 1 else "needed",
        "detail": f"{evidence_ok}/{len(evidence)} evidence rows ok",
    })
    # Token stream
    items.append({
        "check": "Live token stream active",
        "status": "ready" if token_stream.get("live_tokens_seen") else "needed",
        "detail": f"{token_stream.get('token_count', 0)} tokens seen" if token_stream.get("live_tokens_seen") else "no token stream data",
    })

    rows = []
    for item in items:
        icon = {"ready": "✓", "needed": "✗", "partial": "~"}.get(item["status"], "?")
        color_class = {"ready": "ready", "needed": "needed", "partial": "partial"}.get(item["status"], "")
        rows.append(
            f"<tr class=\"{color_class}\">"
            f"<td><span class=\"icon\">{icon}</span></td>"
            f"<td>{_esc(item['check'])}</td>"
            f"<td class=\"muted\">{_esc(item['detail'])}</td>"
            "</tr>"
        )
    ready_count = sum(1 for item in items if item["status"] == "ready")
    return f"""
      <section class=\"card wide\">
        <h2>Demo readiness checklist</h2>
        <p class=\"muted\">{ready_count}/{len(items)} checks ready — this is planning visibility, not a live proof gate.</p>
        <table>
          <thead><tr><th></th><th>Check</th><th>Status</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def build_model_fit_matrix(route: dict[str, Any], *, limit: int = 24) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in route.get("candidates") or []:
        rows.append(
            {
                "model_id": candidate.get("model_id"),
                "can_run_now": bool(candidate.get("memory_fit") and candidate.get("architecture_supported")),
                "memory_fit": bool(candidate.get("memory_fit")),
                "architecture_supported": bool(candidate.get("architecture_supported")),
                "runtime_supported": bool(candidate.get("runtime_supported")),
                "selector_allowed": bool(candidate.get("selector_allowed")),
                "claim_level": candidate.get("claim_level"),
                "placement": candidate.get("placement"),
                "required_free_gb": candidate.get("required_free_gb"),
                "swarm_free_gb": candidate.get("swarm_free_gb"),
                "solo_hosts": candidate.get("solo_hosts") or [],
                "measured_decode_tok_per_s": candidate.get("measured_decode_tok_per_s"),
                "reason": candidate.get("selector_blocked_reason") or candidate.get("reason"),
            }
        )
    rows.sort(
        key=lambda item: (
            not item["can_run_now"],
            str(item.get("claim_level") != "demo_safe"),
            -_as_float(item.get("measured_decode_tok_per_s"), 0.0),
            _as_float(item.get("required_free_gb"), 0.0),
            str(item.get("model_id") or ""),
        )
    )
    return {
        "claim_boundary": "model_fit_matrix_planning_only_no_serving_proof",
        "candidate_count": len(rows),
        "rows": rows[:limit],
        "truncated": len(rows) > limit,
    }


def build_dashboard_document(
    *,
    cap_dirs: Iterable[str | Path] | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY,
    bench_matrix_path: str | Path | None = None,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    proof_state_path: str | Path | None = None,
    joined_layer_plan_path: str | Path | None = None,
    chain_schedule_path: str | Path | None = None,
    handoff_bundle_path: str | Path | None = None,
    proof_orchestration_path: str | Path | None = None,
    physical_showcase_path: str | Path | None = None,
    speculative_plan_path: str | Path | None = None,
    draft_report_path: str | Path | None = None,
    multi_block_diagnostics_path: str | Path | None = None,
    request_logs: Iterable[str | Path] | None = None,
    telemetry_logs: Iterable[str | Path] | None = None,
    token_stream_logs: Iterable[str | Path] | None = None,
    synthetic_m4_laptops: int = 0,
    synthetic_total_gb: float = 24.0,
    synthetic_free_gb: float = 20.0,
) -> dict[str, Any]:
    """Collect every dashboard panel from existing MVP artifacts."""
    real_peers = load_roster(list(cap_dirs or [DEFAULT_CAP_DIR]))
    registry = load_registry(registry_path)
    bench_matrix = _read_json(bench_matrix_path, {})
    real_route = route_report(real_peers, registry, bench_matrix=bench_matrix)
    synthetic_route = None
    if synthetic_m4_laptops > 0:
        synthetic_route = route_report(
            make_synthetic_m4_laptops(
                count=synthetic_m4_laptops,
                total_gb=synthetic_total_gb,
                free_gb=synthetic_free_gb,
            ),
            registry,
            scenario="mvp-10-laptop",
            bench_matrix=bench_matrix,
        )
    evidence = load_evidence(evidence_dir)
    proof_state = _read_json(proof_state_path, None)
    joined_layer_plan = _read_json(joined_layer_plan_path, None)
    chain_schedule = _read_json(chain_schedule_path, None)
    handoff_bundle = _read_json(handoff_bundle_path, None)
    proof_orchestration = _read_json(proof_orchestration_path, None)
    if proof_orchestration is None and isinstance(handoff_bundle, dict):
        proof_orchestration = handoff_bundle.get("proof_orchestration")
    physical_showcase = _read_json(physical_showcase_path, None)
    speculative_plan = _read_json(speculative_plan_path, None)
    if speculative_plan is None and isinstance(handoff_bundle, dict):
        speculative_plan = handoff_bundle.get("speculative_plan")
    draft_report = _read_json(draft_report_path, None)
    multi_block_diagnostics = _read_json(multi_block_diagnostics_path, None)
    layer_placements = collect_layer_placements(evidence)
    token_stream = parse_token_stream_logs(token_stream_logs)
    layers_map = build_layers_map(
        joined_layer_plan=joined_layer_plan,
        layer_placements=layer_placements,
        multi_block_diagnostics=multi_block_diagnostics,
        chain_schedule=chain_schedule,
    )
    layers_map["gaps"] = _detect_layer_gaps(layers_map.get("groups") or [])
    model_fit_matrix = build_model_fit_matrix(real_route)
    passed_evidence = sum(1 for row in evidence if row.get("ok") is True)
    claim_boundaries = [
        "TinyLlama distributed generation parity is proven by committed evidence.",
        "Qwen3-30B-A3B is proven for one live MoE block shard, not full distributed generation yet.",
        "Physical 10-laptop Qwen3 inference is not proven until real peers connect and run the proof harness.",
        "Phones are capability-discovery peers until throughput and block-serving proof exist.",
    ]
    if synthetic_route:
        claim_boundaries.insert(2, "Synthetic 10-laptop routing is planning only; it is hidden in real-demo mode unless explicitly requested.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roster": roster_document(real_peers),
        "real_route": real_route,
        "synthetic_10_laptop_route": synthetic_route,
        "benchmarks": bench_matrix,
        "evidence": evidence,
        "proof_state": proof_state,
        "joined_layer_plan": joined_layer_plan,
        "chain_schedule": chain_schedule,
        "handoff_bundle": handoff_bundle,
        "proof_orchestration": proof_orchestration,
        "physical_showcase": physical_showcase,
        "speculative_plan": speculative_plan,
        "draft_report": draft_report,
        "multi_block_diagnostics": multi_block_diagnostics,
        "request_telemetry": build_request_telemetry(request_logs),
        "token_stream": token_stream,
        "layers_map": layers_map,
        "model_fit_matrix": model_fit_matrix,
        "layer_placements": layer_placements,
        "evidence_summary": {"passed": passed_evidence, "total": len(evidence)},
        "telemetry": parse_telemetry_logs(telemetry_logs),
        "mvp_status": build_status_report(),
        "claim_boundaries": claim_boundaries,
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _route_card(title: str, route: dict[str, Any]) -> str:
    picked = route.get("serving") or route.get("picked") or {}
    best_available = route.get("best_available") or picked
    requested = route.get("requested_model") or "auto / none"
    quant_type = picked.get("quant_type") or "fp16 / none"
    if route.get("override_refused"):
        override = f"refused: {route.get('override_reason') or 'requested model refused'}"
    elif route.get("override_active"):
        override = f"active: {route.get('override_reason') or 'serving requested model'}"
    else:
        override = str(route.get("override_reason") or "auto / none")
    return f"""
      <section class="card route">
        <h2>{_esc(title)}</h2>
        <div class="hero-model">{_esc(picked.get('model_id') or 'No model selected')}</div>
        <div class="grid two">
          <div><span class="label">Serving</span><strong>{_esc(picked.get('model_id') or '—')}</strong></div>
          <div><span class="label">Quantization</span><strong>{_esc(quant_type)}</strong></div>
          <div><span class="label">Placement</span><strong>{_esc(picked.get('placement'))}</strong></div>
          <div><span class="label">Supported</span><strong>{_bool_badge(picked.get('supported'))}</strong></div>
          <div><span class="label">Swarm free GB</span><strong>{_fmt_num(picked.get('swarm_free_gb'), 1)}</strong></div>
          <div><span class="label">Measured decode tok/s</span><strong>{_fmt_measured_rate(picked.get('measured_decode_tok_per_s'))}</strong></div>
          <div><span class="label">Best available</span><strong>{_esc(best_available.get('model_id') or '—')}</strong></div>
          <div><span class="label">Route override</span><strong>{_esc(override)}</strong></div>
          <div><span class="label">Requested model</span><strong>{_esc(requested)}</strong></div>
          <div><span class="label">Selector mode</span><strong>{_esc(route.get('selector_mode') or picked.get('selector_mode') or 'planning')}</strong></div>
        </div>
        <p class="reason">{_esc(picked.get('reason'))}</p>
      </section>
    """


def _bench_summary_for_host(benchmarks: dict[str, Any], host: str | None) -> dict[str, Any]:
    models = ((benchmarks or {}).get(host or "") or {}).get("models") or {}
    best_decode = 0.0
    best_prefill = 0.0
    best_model = None
    for model, record in models.items():
        decode = _as_float(record.get("decode_tok_per_s"), 0.0) if isinstance(record, dict) else 0.0
        prefill = _as_float(record.get("prefill_tok_per_s"), 0.0) if isinstance(record, dict) else 0.0
        if decode > best_decode:
            best_decode = decode
            best_model = model
        best_prefill = max(best_prefill, prefill)
    return {
        "model_count": len(models),
        "best_decode_tok_per_s": best_decode,
        "best_prefill_tok_per_s": best_prefill,
        "best_model": best_model,
    }


def _devices_table(roster: dict[str, Any], benchmarks: dict[str, Any] | None = None) -> str:
    peers = roster.get("peers") or []
    rows = []
    for peer in peers:
        mem = peer.get("memory") or {}
        accel = peer.get("accelerator") or {}
        mobile = peer.get("mobile") or {}
        bench = _bench_summary_for_host(benchmarks or {}, peer.get("hostname"))
        bench_count = f"{bench['model_count']} bench models" if bench["model_count"] else "no bench"
        best_decode = f"{_fmt_num(bench.get('best_decode_tok_per_s'), 2)} tok/s" if bench["model_count"] else "—"
        best_prefill = f"{_fmt_num(bench.get('best_prefill_tok_per_s'), 1)} tok/s" if bench["model_count"] else "—"
        rows.append(
            "<tr>"
            f"<td>{_esc(peer.get('hostname'))}</td>"
            f"<td>{_esc(accel.get('device') or 'cpu')}</td>"
            f"<td>{_fmt_num(mem.get('free_gb'), 1)} / {_fmt_num(mem.get('total_gb'), 1)}</td>"
            f"<td>{_esc((peer.get('network') or {}).get('tailscale_ip') or '—')}</td>"
            f"<td>{_bool_badge(mobile.get('is_mobile'))}</td>"
            f"<td>{_esc(bench_count)}</td>"
            f"<td>{_esc(best_decode)}</td>"
            f"<td>{_esc(best_prefill)}</td>"
            "</tr>"
        )
    return """
      <section class="card wide">
        <h2>Connected devices</h2>
        <table>
          <thead><tr><th>Host</th><th>Device</th><th>Free / total GB</th><th>Tailscale</th><th>Mobile</th><th>Bench coverage</th><th>Best decode</th><th>Best prefill</th></tr></thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </section>
    """.format(rows="\n".join(rows) or '<tr><td colspan="8">No peers found</td></tr>')


def _bench_table(benchmarks: dict[str, Any]) -> str:
    rows: list[str] = []
    for host, payload in sorted((benchmarks or {}).items()):
        for model, record in sorted((payload.get("models") or {}).items()):
            rows.append(
                "<tr>"
                f"<td>{_esc(host)}</td>"
                f"<td>{_esc(model)}</td>"
                f"<td>{_fmt_num(record.get('prefill_tok_per_s'), 1)}</td>"
                f"<td>{_fmt_num(record.get('decode_tok_per_s'), 2)}</td>"
                f"<td>{_esc(record.get('device') or '—')} / {_esc(record.get('dtype') or '—')}</td>"
                "</tr>"
            )
    return """
      <section class="card wide">
        <h2>Measured throughput</h2>
        <table>
          <thead><tr><th>Host</th><th>Model</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>Device/dtype</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """.format(rows="\n".join(rows) or '<tr><td colspan="5">No benchmark matrix supplied yet</td></tr>')


def _evidence_table(evidence: list[dict[str, Any]]) -> str:
    rows = []
    for row in evidence:
        rows.append(
            "<tr>"
            f"<td>{_esc(row.get('file'))}</td>"
            f"<td>{_bool_badge(row.get('ok'))}</td>"
            f"<td>{_esc(row.get('mode') or '—')}</td>"
            f"<td>{_esc(row.get('model') or '—')}</td>"
            f"<td>{_bool_badge(row.get('server_to_server'))}</td>"
            f"<td>{_bool_badge(row.get('generated_text_match'))}</td>"
            f"<td>{_fmt_num(row.get('distributed_seconds'), 3)}</td>"
            "</tr>"
        )
    return """
      <section class="card wide">
        <h2>Inference proof bundle</h2>
        <table>
          <thead><tr><th>Evidence file</th><th>OK</th><th>Mode</th><th>Model</th><th>S2S</th><th>Text match</th><th>Distributed seconds</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """.format(rows="\n".join(rows) or '<tr><td colspan="7">No evidence JSON found</td></tr>')


def _layer_placement_table(placements: list[dict[str, Any]]) -> str:
    rows = []
    for item in placements:
        start, end = item.get("layers", [None, None])
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('host'))}</td>"
            f"<td><strong>layers {start}:{end}</strong></td>"
            f"<td>{_esc(item.get('model') or '—')}</td>"
            f"<td>{_esc(item.get('evidence_file') or '—')}</td>"
            f"<td><code>{_esc(item.get('server_maddr') or '—')}</code></td>"
            "</tr>"
        )
    return """
      <section class="card wide">
        <h2>Layer placement</h2>
        <p class="muted">Real server/layer ownership from proof metadata; no synthetic peers are shown unless explicitly requested.</p>
        <table>
          <thead><tr><th>Device / server</th><th>Transformer layers</th><th>Model</th><th>Evidence</th><th>Multiaddr</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """.format(rows="\n".join(rows) or '<tr><td colspan="5">No layer placement metadata supplied yet. Re-run the parity harness with --server-placement host=start:end.</td></tr>')


def _request_telemetry_panel(report: dict[str, Any]) -> str:
    counts = report.get("request_counts") or {}
    latency = report.get("latency_seconds") or {}
    forward = latency.get("forward") or {}
    backward = latency.get("backward") or {}
    models = ", ".join(f"{key} ({value})" for key, value in (report.get("models") or {}).items()) or "—"
    block_ranges = ", ".join(f"{key} ({value})" for key, value in (report.get("block_ranges") or {}).items()) or "—"
    error_rows = "".join(
        "<tr>"
        f"<td>{_esc(Path(item.get('log') or '—').name)}</td>"
        f"<td>{_esc(item.get('message') or '—')}</td>"
        "</tr>"
        for item in report.get("errors") or []
    )
    claim_copy = "load proof claimed" if report.get("load_proof_claimed") else "no load proof claimed"
    return f"""
      <section class="card wide request-telemetry">
        <h2>Live request telemetry</h2>
        <div class="grid two">
          <div><span class="label">Requests</span><strong>succeeded {_esc(counts.get('succeeded') or 0)} / failed {_esc(counts.get('failed') or 0)}</strong></div>
          <div><span class="label">Total observed</span><strong>{_esc(counts.get('total') or 0)}</strong></div>
          <div><span class="label">Forward latency</span><strong>forward avg {_fmt_latency_seconds(forward.get('avg'))} · p95 {_fmt_latency_seconds(forward.get('p95'))}</strong></div>
          <div><span class="label">Backward latency</span><strong>backward avg {_fmt_latency_seconds(backward.get('avg'))} · p95 {_fmt_latency_seconds(backward.get('p95'))}</strong></div>
          <div><span class="label">Models</span><strong>{_esc(models)}</strong></div>
          <div><span class="label">Block ranges</span><strong>{_esc(block_ranges)}</strong></div>
          <div><span class="label">Unmeasured latency</span><strong>forward {_esc(forward.get('unmeasured_count') or 0)} / backward {_esc(backward.get('unmeasured_count') or 0)}</strong></div>
          <div><span class="label">Traffic claim</span><strong class="warn">{_esc(claim_copy)}</strong></div>
          <div><span class="label">Claim boundary</span><code>{_esc(report.get('claim_boundary'))}</code></div>
        </div>
        <table>
          <thead><tr><th>Log</th><th>Error / blocker</th></tr></thead>
          <tbody>{error_rows or '<tr><td colspan="2">No request errors observed</td></tr>'}</tbody>
        </table>
        <p class="muted">{_esc(report.get('next_step') or 'Request telemetry is observability only.')}</p>
      </section>
    """


def _token_stream_panel(report: dict[str, Any]) -> str:
    requests = report.get("requests") or []
    rows = []
    for item in requests:
        host_path = " → ".join(str(host) for host in item.get("hosts") or []) or "—"
        layer_path = " → ".join(str(layer) for layer in item.get("layer_ranges") or []) or "—"
        generated = str(item.get("generated_text") or "")[-240:]
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('request_id'))}</td>"
            f"<td>{_esc(item.get('model') or '—')}</td>"
            f"<td>{_esc(item.get('token_count') or 0)}</td>"
            f"<td>{_esc(item.get('latest_token_text') or '—')}</td>"
            f"<td>{_fmt_num(item.get('tokens_per_second'), 3)}</td>"
            f"<td>{_esc(host_path)}</td>"
            f"<td>{_esc(layer_path)}</td>"
            f"<td><code>{_esc(generated or '—')}</code></td>"
            "</tr>"
        )
    live_copy = "live tokens seen" if report.get("live_tokens_seen") else "waiting for token stream"
    return f"""
      <section class="card wide token-stream">
        <h2>Live token stream</h2>
        <div class="grid two">
          <div><span class="label">Status</span><strong>{_esc(live_copy)}</strong></div>
          <div><span class="label">Requests / tokens</span><strong>{_esc(report.get('request_count') or 0)} / {_esc(report.get('token_count') or 0)}</strong></div>
          <div><span class="label">Token log files</span><strong>{_esc(len(report.get('scanned_logs') or []))}</strong></div>
          <div><span class="label">Claim boundary</span><code>{_esc(report.get('claim_boundary'))}</code></div>
        </div>
        <table>
          <thead><tr><th>Request</th><th>Model</th><th>Tokens</th><th>Latest token</th><th>Tok/s</th><th>Device path</th><th>Layer path</th><th>Generated text tail</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="8">No token stream rows yet. Pass --token-stream-log and run with --watch-seconds during inference.</td></tr>'}</tbody>
        </table>
        <p class="muted">{_esc(report.get('next_step') or 'Token stream is observability only and does not prove generation correctness.')}</p>
      </section>
    """


def _layers_map_panel(layers_map: dict[str, Any]) -> str:
    groups = layers_map.get("groups") or []
    gaps_data = layers_map.get("gaps") or {}
    if not groups:
        return """
      <section class="card wide layers-map">
        <h2>Layers map</h2>
        <p class="muted">No layer placement, joined plan, or multi-block diagnostics supplied yet.</p>
      </section>
    """
    rendered_groups: list[str] = []
    for group in groups:
        total_layers = max(1, int(group.get("total_layers") or 1))
        segments_html = []
        legend_rows = []
        for segment in group.get("segments") or []:
            start = int(segment.get("start_layer") or 0)
            end = int(segment.get("end_layer") or start)
            width = max(2.0, (end - start) / total_layers * 100.0)
            status = str(segment.get("status") or "unknown").lower().replace(" ", "-")
            label = f"{segment.get('hostname')} {segment.get('block_range')}"
            segments_html.append(
                f'<div class="layer-segment status-{_esc(status)}" style="width:{width:.3f}%" title="{_esc(label)}">'
                f'<strong>{_esc(segment.get("hostname"))}</strong><span>{_esc(segment.get("block_range"))}</span></div>'
            )
            util = segment.get("utilization_fraction")
            util_copy = f"{float(util) * 100:.0f}%" if isinstance(util, (float, int)) else "—"
            legend_rows.append(
                "<tr>"
                f"<td>{_esc(segment.get('hostname'))}</td>"
                f"<td>{_esc(segment.get('block_range'))}</td>"
                f"<td>{_esc(segment.get('layer_count'))}</td>"
                f"<td>{_esc(segment.get('status'))}</td>"
                f"<td>{_esc(util_copy)}</td>"
                "</tr>"
            )
        rendered_groups.append(
            f"""
            <div class="layer-map-group">
              <h3>{_esc(group.get('title') or group.get('source'))}</h3>
              <p class="muted">Model {_esc(group.get('model') or '—')} · total layers {_esc(total_layers)} · boundary <code>{_esc(group.get('claim_boundary'))}</code></p>
              <div class="layer-map-track">{''.join(segments_html)}</div>
              <table>
                <thead><tr><th>Device</th><th>Layers</th><th>Count</th><th>Status</th><th>Utilization</th></tr></thead>
                <tbody>{''.join(legend_rows)}</tbody>
              </table>
            </div>
            """
        )
    gap_summary = ""
    gap_count = int(gaps_data.get("gap_count") or 0)
    if gap_count > 0:
        coverage = gaps_data.get("coverage_percent") or 0.0
        gap_items = []
        for gap in gaps_data.get("gaps") or []:
            gap_items.append(
                f'<li>layers {gap.get("start_layer")}:{gap.get("end_layer")} — {gap.get("layer_count")} layers uncovered</li>'
            )
        gap_summary = f"""
        <div class="layer-map-group layer-gap-warning">
          <h3>⚠ Coverage gaps</h3>
          <p class="muted">{coverage:.0f}% proven/healthy coverage · {gap_count} uncovered range{'s' if gap_count != 1 else ''}</p>
          <ul>{''.join(gap_items)}</ul>
        </div>
        """
    return f"""
      <section class="card wide layers-map">
        <h2>Layers map</h2>
        <p class="muted">Visual route map from joined peers, committed proof placements, and multi-block diagnostics. This map observes routing; proof gates remain fail-closed.</p>
        {''.join(rendered_groups)}
        {gap_summary}
      </section>
    """


def _model_fit_matrix_panel(matrix: dict[str, Any]) -> str:
    rows = []
    for item in matrix.get("rows") or []:
        hosts = ", ".join(str(host) for host in item.get("solo_hosts") or []) or "—"
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('model_id'))}</td>"
            f"<td>{_bool_badge(item.get('can_run_now'))}</td>"
            f"<td>{_esc(item.get('claim_level') or '—')}</td>"
            f"<td>{_esc(item.get('placement') or '—')}</td>"
            f"<td>{_fmt_num(item.get('required_free_gb'), 1)}</td>"
            f"<td>{_fmt_num(item.get('swarm_free_gb'), 1)}</td>"
            f"<td>{_esc(hosts)}</td>"
            f"<td>{_fmt_measured_rate(item.get('measured_decode_tok_per_s'))}</td>"
            f"<td>{_esc(item.get('reason') or '—')}</td>"
            "</tr>"
        )
    truncated = "; truncated to top rows" if matrix.get("truncated") else ""
    return f"""
      <section class="card wide model-fit-matrix">
        <h2>Model capability matrix</h2>
        <p class="muted">Which models could fit the currently connected devices by memory/architecture. Planning only; demo-safe still requires proof gates. Candidates: {_esc(matrix.get('candidate_count') or 0)}{_esc(truncated)}.</p>
        <table>
          <thead><tr><th>Model</th><th>Can run now</th><th>Claim level</th><th>Placement</th><th>Need GB</th><th>Swarm free GB</th><th>Solo hosts</th><th>Measured decode</th><th>Reason / blocker</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="9">No route candidates loaded</td></tr>'}</tbody>
        </table>
        <p class="muted"><code>{_esc(matrix.get('claim_boundary'))}</code></p>
      </section>
    """


def _telemetry_panel(telemetry: dict[str, Any]) -> str:
    event_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in (telemetry.get("event_counts") or {}).items()
    )
    reason_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in (telemetry.get("reason_counts") or {}).items()
    )
    examples = "\n".join(_esc(line) for line in telemetry.get("examples") or [])
    markers = " · ".join(_esc(marker) for marker in telemetry.get("markers") or _TELEMETRY_MARKERS)
    return f"""
      <section class="card wide">
        <h2>Live telemetry counters</h2>
        <p class="muted">Structured markers watched: <code>{markers}</code></p>
        <div class="grid two">
          <table><thead><tr><th>Event</th><th>Count</th></tr></thead><tbody>{event_rows or '<tr><td colspan="2">No telemetry logs supplied</td></tr>'}</tbody></table>
          <table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{reason_rows or '<tr><td colspan="2">No classified reasons yet</td></tr>'}</tbody></table>
        </div>
        <pre>{examples or 'Run with --telemetry-log /path/to/server.log to show live recovery/S2S events.'}</pre>
      </section>
    """


def _status_panel(status: dict[str, Any]) -> str:
    milestones = status.get("milestones") or []
    rows = []
    for item in milestones:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('label'))}</td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('percent'))}%</td>"
            f"<td>{_esc(item.get('next_step') or item.get('evidence') or '—')}</td>"
            "</tr>"
        )
    planned_tasks = status.get("planned_tasks") or []
    core_tasks = status.get("core_tasks") or []
    post_mvp_tasks = status.get("post_mvp_tasks") or []
    post_mvp_rows = []
    for item in status.get("post_mvp_milestones") or []:
        post_mvp_rows.append(
            "<tr>"
            f"<td>{_esc(item.get('label'))}</td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('percent'))}%</td>"
            f"<td>{_esc(item.get('next_step') or item.get('evidence') or '—')}</td>"
            "</tr>"
        )
    def _task_rows(items: list[dict[str, Any]]) -> list[str]:
        rows: list[str] = []
        for item in items:
            done = "yes" if item.get("done") else "no"
            rows.append(
                "<tr>"
                f"<td>{_esc(item.get('label'))}</td>"
                f"<td>{_esc(item.get('status'))}</td>"
                f"<td>{_esc(done)}</td>"
                f"<td>{_esc(item.get('next_step') or item.get('evidence') or '—')}</td>"
                "</tr>"
            )
        return rows

    task_rows = _task_rows(planned_tasks)
    core_task_rows = _task_rows(core_tasks)
    post_mvp_task_rows = _task_rows(post_mvp_tasks)
    summary = status.get("task_summary") or {}
    core_summary = status.get("core_task_summary") or {}
    post_mvp_task_summary = status.get("post_mvp_task_summary") or {}
    def _summary_copy(payload: dict[str, Any]) -> str:
        return ", ".join(
            f"{payload.get(key, 0)} {key}" for key in ("complete", "partial", "pending", "blocked")
        )
    summary_copy = _summary_copy(summary)
    core_summary_copy = _summary_copy(core_summary)
    post_mvp_task_summary_copy = _summary_copy(post_mvp_task_summary)
    return f"""
      <section class="card wide status">
        <h2>MVP build status</h2>
        <div class="progress-wrap" aria-label="Distributed Inference MVP build progress">
          <div class="progress-bar" style="width:{_esc(status.get('overall_percent', 0))}%"></div>
        </div>
        <div class="grid two">
          <div><span class="label">Built from plan</span><strong class="status-bar-text">{_esc(status.get('overall_bar'))}</strong></div>
          <div><span class="label">Next gate</span><strong>{_esc(status.get('next_gate'))}</strong></div>
          <div><span class="label">Remaining</span><strong>{_esc(status.get('remaining_percent'))}%</strong></div>
          <div><span class="label">Claim boundary</span><code>{_esc(status.get('claim_boundary'))}</code></div>
        </div>
        <p class="muted">{_esc(status.get('interpretation'))}</p>
        <table>
          <thead><tr><th>Milestone</th><th>Status</th><th>Built</th><th>Evidence / next</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="4">No status milestones loaded</td></tr>'}</tbody>
        </table>
        <h3>Post-MVP / stretch milestones</h3>
        <p class="muted">Visible for planning, not part of MVP-core 100%.</p>
        <table>
          <thead><tr><th>Milestone</th><th>Status</th><th>Built</th><th>Evidence / next</th></tr></thead>
          <tbody>{''.join(post_mvp_rows) or '<tr><td colspan="4">No post-MVP milestones loaded</td></tr>'}</tbody>
        </table>
        <h3>Planned tasks</h3>
        <p class="muted">All-task summary: {_esc(summary_copy)}. This includes post-MVP backlog and should not be read as an MVP-core blocker.</p>
        <h4>MVP-core tasks</h4>
        <p class="muted">MVP-core task summary: {_esc(core_summary_copy)}</p>
        <table>
          <thead><tr><th>Task</th><th>Status</th><th>Done?</th><th>Evidence / next</th></tr></thead>
          <tbody>{''.join(core_task_rows) or '<tr><td colspan="4">No MVP-core tasks loaded</td></tr>'}</tbody>
        </table>
        <h4>Post-MVP backlog tasks</h4>
        <p class="muted">Post-MVP backlog task summary: {_esc(post_mvp_task_summary_copy)}</p>
        <table>
          <thead><tr><th>Task</th><th>Status</th><th>Done?</th><th>Evidence / next</th></tr></thead>
          <tbody>{''.join(post_mvp_task_rows) or '<tr><td colspan="4">No post-MVP backlog tasks loaded</td></tr>'}</tbody>
        </table>
        <details>
          <summary>All planned tasks, including post-MVP backlog</summary>
          <table>
            <thead><tr><th>Task</th><th>Status</th><th>Done?</th><th>Evidence / next</th></tr></thead>
            <tbody>{''.join(task_rows) or '<tr><td colspan="4">No planned tasks loaded</td></tr>'}</tbody>
          </table>
        </details>
      </section>
    """


def _proof_state_panel(proof_state: dict[str, Any] | None) -> str:
    if not proof_state:
        return """
      <section class="card wide proof-state">
        <h2>Live proof-prep state</h2>
        <p class="muted">No proof-state JSON supplied yet. Generate one with <code>python mvp_capabilities/proof_state.py</code>.</p>
      </section>
    """
    progress = proof_state.get("fetch_progress") or {}
    cache = proof_state.get("cache") or {}
    status = proof_state.get("download_status") or "unknown"
    inference_copy = "inference proven" if proof_state.get("inference_proven") else "inference not proven"
    progress_copy = (
        f"{progress.get('percent')}% ({progress.get('completed_files')}/{progress.get('total_files')} files)"
        if progress
        else "—"
    )
    cache_copy = cache.get("human") or _fmt_num(cache.get("bytes"), 0)
    eta = proof_state.get("eta_seconds")
    eta_copy = "complete" if eta == 0 else (f"~{_fmt_num(eta, 0)}s" if eta is not None else proof_state.get("eta_reason") or "—")
    snapshot_copy = "complete" if cache.get("snapshot_complete") else "not complete"
    return f"""
      <section class="card wide proof-state">
        <h2>Live proof-prep state</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(proof_state.get('model'))}</strong></div>
          <div><span class="label">Gate</span><strong>{_esc(proof_state.get('gate'))}</strong></div>
          <div><span class="label">Download status</span><strong>{_esc(status)}</strong></div>
          <div><span class="label">Fetch progress</span><strong>{_esc(progress_copy)}</strong></div>
          <div><span class="label">Host</span><strong>{_esc(proof_state.get('host') or '—')}</strong></div>
          <div><span class="label">Cache</span><strong>{_esc(cache_copy)} · {_esc(cache.get('weight_files'))} weight files</strong></div>
          <div><span class="label">ETA</span><strong>{_esc(eta_copy)}</strong></div>
          <div><span class="label">Snapshot</span><strong>{_esc(snapshot_copy)} · stale partials {_esc(cache.get('stale_incomplete_files') or 0)}</strong></div>
          <div><span class="label">Inference claim</span><strong class="warn">{_esc(inference_copy)}</strong></div>
          <div><span class="label">Claim boundary</span><code>{_esc(proof_state.get('claim_boundary'))}</code></div>
        </div>
        <p class="muted">{_esc(proof_state.get('next_step') or 'Only a dedicated proof verifier may promote this gate.')}</p>
      </section>
    """


def _joined_layer_plan_panel(plan: dict[str, Any] | None) -> str:
    if not plan:
        return """
      <section class="card wide joined-layer-plan">
        <h2>Joined-peer layer plan</h2>
        <p class="muted">No joined-layer plan supplied yet. Generate one with <code>python mvp_capabilities/join_layer_plan.py --coordinator-url ... --include-launch-commands</code>.</p>
      </section>
    """
    placement = plan.get("placement") or {}
    rows: list[str] = []
    for item in placement.get("assignments") or []:
        start = item.get("start_layer")
        end = item.get("end_layer")
        block_range = item.get("block_range") or (f"{start}:{end}" if start is not None and end is not None else "—")
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('hostname') or 'unknown')}</td>"
            f"<td><strong>layers {_esc(block_range)}</strong></td>"
            f"<td>{_esc(item.get('layer_count') or '—')}</td>"
            f"<td>{_esc(item.get('port') or '—')}</td>"
            f"<td><code>{_esc(item.get('launch_command') or '—')}</code></td>"
            "</tr>"
        )
    inference_copy = "inference proven" if plan.get("inference_proven") else "inference not proven"
    return f"""
      <section class="card wide joined-layer-plan">
        <h2>Joined-peer layer plan</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(plan.get('model_id') or '—')}</strong></div>
          <div><span class="label">Source</span><strong>{_esc(plan.get('source') or '—')}</strong></div>
          <div><span class="label">Active peers</span><strong>{_esc(plan.get('active_peer_count') or 0)}</strong></div>
          <div><span class="label">Supported</span><strong>{_bool_badge(placement.get('supported'))}</strong></div>
          <div><span class="label">Assigned layers</span><strong>{_esc(placement.get('assigned_layers') or 0)}/{_esc(placement.get('num_layers') or '—')}</strong></div>
          <div><span class="label">Inference claim</span><strong class="warn">{_esc(inference_copy)}</strong></div>
          <div><span class="label">Plan boundary</span><code>{_esc(plan.get('claim_boundary'))}</code></div>
          <div><span class="label">Launch boundary</span><code>{_esc(placement.get('launch_commands_claim_boundary'))}</code></div>
        </div>
        <p class="reason">{_esc(placement.get('reason') or plan.get('next_step') or 'Launch commands are a runbook only.')}</p>
        <table>
          <thead><tr><th>Joined peer</th><th>Transformer layers</th><th>Count</th><th>Port</th><th>Launch command</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="5">No joined peer assignments yet</td></tr>'}</tbody>
        </table>
      </section>
    """


def _chain_schedule_panel(schedule: dict[str, Any] | None) -> str:
    if not schedule:
        return """
      <section class="card wide chain-schedule">
        <h2>Chain scheduler rehearsal</h2>
        <p class="muted">No chain-schedule JSON supplied yet. Generate one with <code>python mvp_capabilities/chain_scheduler.py --joined-layer-plan ...</code>.</p>
      </section>
    """
    waves = schedule.get("waves") or []
    wave_rows: list[str] = []
    for wave in waves:
        request_ids = ", ".join(str(item) for item in wave.get("request_ids") or [])
        wave_rows.append(
            "<tr>"
            f"<td>{_esc(wave.get('wave_index'))}</td>"
            f"<td>{_esc(request_ids or '—')}</td>"
            f"<td>{_esc(wave.get('parallel_request_count') or 0)}</td>"
            "</tr>"
        )
    peer_rows: list[str] = []
    for hostname, item in sorted((schedule.get("peer_health") or {}).items()):
        peer_rows.append(
            "<tr>"
            f"<td>{_esc(item.get('hostname') or hostname)}</td>"
            f"<td>{_esc(item.get('block_range') or '—')}</td>"
            f"<td>{_esc(item.get('scheduled_requests') or 0)}</td>"
            f"<td>{_esc(item.get('scheduled_tokens') or 0)}</td>"
            f"<td>{_fmt_num(item.get('utilization_fraction'), 2)}</td>"
            f"<td>{_esc(item.get('health_status') or '—')}</td>"
            "</tr>"
        )
    inference_copy = "inference proven" if schedule.get("inference_proven") else "inference not proven"
    live_copy = "live requests sent" if schedule.get("live_requests_sent") else "no live requests sent"
    token_budget = schedule.get("token_budget") or {}
    return f"""
      <section class="card wide chain-schedule">
        <h2>Chain scheduler rehearsal</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(schedule.get('model_id') or '—')}</strong></div>
          <div><span class="label">Scheduler status</span><strong>{_esc(schedule.get('scheduler_status') or '—')}</strong></div>
          <div><span class="label">Requests / waves</span><strong>{_esc(schedule.get('request_count') or 0)} / {_esc(schedule.get('wave_count') or 0)}</strong></div>
          <div><span class="label">Stages</span><strong>{_esc(schedule.get('stage_count') or 0)}</strong></div>
          <div><span class="label">Tokens/request</span><strong>{_esc(token_budget.get('tokens_per_request') or 0)}</strong></div>
          <div><span class="label">Scheduled tokens</span><strong>{_esc(token_budget.get('scheduled_tokens') or 0)}</strong></div>
          <div><span class="label">Traffic claim</span><strong class="warn">{_esc(live_copy)} · {_esc(inference_copy)}</strong></div>
          <div><span class="label">Claim boundary</span><code>{_esc(schedule.get('claim_boundary'))}</code></div>
        </div>
        <div class="grid two">
          <table>
            <thead><tr><th>Wave</th><th>Request IDs</th><th>Parallel</th></tr></thead>
            <tbody>{''.join(wave_rows) or '<tr><td colspan="3">No request waves planned</td></tr>'}</tbody>
          </table>
          <table>
            <thead><tr><th>Peer</th><th>Layers</th><th>Requests</th><th>Tokens</th><th>Utilization</th><th>Health</th></tr></thead>
            <tbody>{''.join(peer_rows) or '<tr><td colspan="6">No peer health rows planned</td></tr>'}</tbody>
          </table>
        </div>
        <p class="muted">{_esc(schedule.get('next_step') or 'Scheduler rehearsal only; live request telemetry is still required for a load-proof claim.')}</p>
      </section>
    """


def _handoff_bundle_panel(bundle: dict[str, Any] | None) -> str:
    if not bundle:
        return """
      <section class="card wide handoff-bundle">
        <h2>Operator handoff bundle</h2>
        <p class="muted">No coordinator handoff JSON supplied yet. Generate one from <code>/handoff?token=...&amp;model=auto</code>.</p>
      </section>
    """
    route = bundle.get("route_decision") or {}
    picked = route.get("picked") or {}
    plan = bundle.get("plan") or {}
    readiness = plan.get("launch_readiness") or {}
    bootstrap = bundle.get("bootstrap_runbook") or {}
    heartbeat_loop = bootstrap.get("heartbeat_loop") or {}
    bootstrap_script = bootstrap.get("shell_script") or ""
    bootstrap_panel = ""
    if isinstance(bootstrap, dict) and bootstrap:
        bootstrap_panel = f"""
        <div class="handoff-bootstrap">
          <h3>Fresh-device bootstrap</h3>
          <div class="grid two">
            <div><span class="label">Heartbeat loop</span><strong>heartbeat count {_esc(heartbeat_loop.get('count') or '—')} · every {_esc(heartbeat_loop.get('interval_seconds') or '—')}s</strong></div>
            <div><span class="label">Bootstrap boundary</span><code>{_esc(bootstrap.get('claim_boundary') or '—')}</code></div>
          </div>
          <pre>{_esc(bootstrap_script or 'No bootstrap shell script supplied')}</pre>
        </div>
        """
    runbooks = bundle.get("proof_runbooks") or {}
    runbook_rows = "".join(
        "<tr>"
        f"<td>{_esc(name)}</td>"
        f"<td>{_esc((item or {}).get('proof_gate') or name)}</td>"
        f"<td><code>{_esc((item or {}).get('claim_boundary') or (item or {}).get('status') or '—')}</code></td>"
        f"<td>{_esc((item or {}).get('request_count') or (item or {}).get('max_new_tokens') or '—')}</td>"
        "</tr>"
        for name, item in sorted(runbooks.items())
        if isinstance(item, dict)
    )
    inference_copy = "inference proven" if bundle.get("inference_proven") else "inference not proven"
    ready_copy = "ready to start" if readiness.get("ready_to_start") else "placeholders/blockers remain"
    return f"""
      <section class="card wide handoff-bundle">
        <h2>Operator handoff bundle</h2>
        <div class="grid two">
          <div><span class="label">Source</span><strong>{_esc(bundle.get('source') or '—')}</strong></div>
          <div><span class="label">Selected model</span><strong>{_esc(plan.get('model_id') or picked.get('model_id') or '—')}</strong></div>
          <div><span class="label">Launch readiness</span><strong class="warn">{_esc(ready_copy)}</strong></div>
          <div><span class="label">Inference claim</span><strong class="warn">{_esc(inference_copy)}</strong></div>
          <div><span class="label">Bundle boundary</span><code>{_esc(bundle.get('claim_boundary'))}</code></div>
          <div><span class="label">Readiness boundary</span><code>{_esc(readiness.get('claim_boundary') or '—')}</code></div>
        </div>
        <table>
          <thead><tr><th>Runbook</th><th>Proof gate</th><th>Claim boundary / status</th><th>Size</th></tr></thead>
          <tbody>{runbook_rows or '<tr><td colspan="4">No proof runbooks in handoff bundle</td></tr>'}</tbody>
        </table>
        {bootstrap_panel}
        <p class="muted">Handoff bundles are operator checklists only: they do not start servers, send traffic, or update proof status.</p>
      </section>
    """


def _proof_orchestration_panel(plan: dict[str, Any] | None) -> str:
    if not plan:
        return """
      <section class="card wide proof-orchestration">
        <h2>Proof orchestration</h2>
        <p class="muted">No proof orchestration JSON supplied yet. Generate one with <code>python mvp_capabilities/proof_orchestrator.py --handoff-bundle .local/handoff-bundle.json</code>.</p>
      </section>
    """
    summary = plan.get("summary") or {}
    phase_order = " → ".join(str(item) for item in plan.get("phase_order") or []) or "—"
    placeholders = ", ".join(str(item) for item in summary.get("unresolved_placeholders") or []) or "none"
    available_gates = ", ".join(str(item) for item in summary.get("available_proof_gates") or []) or "—"
    ready_servers = "yes" if summary.get("ready_to_start_servers") else "no"
    ready_clients = "yes" if summary.get("ready_for_proof_clients") else "no"
    launch_rows = "".join(
        "<tr>"
        f"<td>{_esc(item.get('hostname') or '—')}</td>"
        f"<td>{_esc(item.get('role') or '—')}</td>"
        f"<td>{_esc(item.get('block_range') or '—')}</td>"
        f"<td>{_bool_badge(item.get('ready'))}</td>"
        "</tr>"
        for item in plan.get("launch_steps") or []
        if isinstance(item, dict)
    )
    proof_rows = "".join(
        "<tr>"
        f"<td>{_esc(item.get('proof_gate') or '—')}</td>"
        f"<td>{_bool_badge(item.get('ready'))}</td>"
        f"<td>{_esc(item.get('command_count') or 0)}</td>"
        f"<td>{_esc(', '.join(item.get('blocked_by') or []) or '—')}</td>"
        "</tr>"
        for item in plan.get("proof_steps") or []
        if isinstance(item, dict)
    )
    physical = plan.get("physical_showcase") or {}
    physical_blocked = ", ".join(str(item) for item in physical.get("blocked_by") or []) or "—"
    physical_panel = ""
    if physical:
        physical_panel = f"""
        <div class="physical-showcase-proof">
          <h3>Physical showcase evidence gate</h3>
          <div class="grid two">
            <div><span class="label">Proof gate</span><strong>{_esc(physical.get('proof_gate') or '—')}</strong></div>
            <div><span class="label">Ready</span><strong>{_esc(physical.get('ready'))}</strong></div>
            <div><span class="label">Evidence path</span><code>{_esc(physical.get('evidence_path') or '—')}</code></div>
            <div><span class="label">Blocked by</span><code>{_esc(physical_blocked)}</code></div>
            <div><span class="label">Boundary</span><code>{_esc(physical.get('claim_boundary') or '—')}</code></div>
            <div><span class="label">Verify command</span><code>{_esc(physical.get('verify_command') or '—')}</code></div>
          </div>
        </div>
        """
    return f"""
      <section class="card wide proof-orchestration">
        <h2>Proof orchestration</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(plan.get('model_id') or '—')}</strong></div>
          <div><span class="label">Status</span><strong>{_esc(summary.get('orchestration_status') or '—')}</strong></div>
          <div><span class="label">Phase order</span><strong>{_esc(phase_order)}</strong></div>
          <div><span class="label">Available gates</span><strong>{_esc(available_gates)}</strong></div>
          <div><span class="label">Server readiness</span><strong>ready to start servers: {_esc(ready_servers)}</strong></div>
          <div><span class="label">Client proof readiness</span><strong>ready for proof clients: {_esc(ready_clients)}</strong></div>
          <div><span class="label">Unresolved placeholders</span><code>{_esc(placeholders)}</code></div>
          <div><span class="label">Boundary</span><code>{_esc(plan.get('claim_boundary'))}</code></div>
        </div>
        <div class="grid two">
          <table>
            <thead><tr><th>Server</th><th>Role</th><th>Layers</th><th>Ready</th></tr></thead>
            <tbody>{launch_rows or '<tr><td colspan="4">No launch steps found</td></tr>'}</tbody>
          </table>
          <table>
            <thead><tr><th>Proof gate</th><th>Ready</th><th>Commands</th><th>Blocked by</th></tr></thead>
            <tbody>{proof_rows or '<tr><td colspan="4">No proof steps found</td></tr>'}</tbody>
          </table>
        </div>
        {physical_panel}
        <p class="muted">This is an operator checklist only: live commands executed = {_esc(plan.get('live_commands_executed'))}; proof status updates applied = {_esc(plan.get('proof_status_updates_applied'))}.</p>
      </section>
    """


def _physical_showcase_panel(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    failed_checks = ", ".join(str(item) for item in report.get("failed_checks") or []) or "none"
    fresh_peers = ", ".join(str(item) for item in report.get("fresh_peer_ids") or []) or "none"
    generation = report.get("generation") or {}
    load = report.get("load") or {}
    return f"""
      <section class="card wide physical-showcase-evidence">
        <h2>Physical showcase evidence</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(report.get('selected_model') or report.get('model_id') or '—')}</strong></div>
          <div><span class="label">Status</span><strong>{_esc(report.get('status') or '—')}</strong></div>
          <div><span class="label">Showcase proven</span><strong>{_esc(report.get('physical_showcase_proven'))}</strong></div>
          <div><span class="label">Fresh joined peers</span><strong>{_esc(report.get('fresh_joined_peer_count') or 0)} · {_esc(fresh_peers)}</strong></div>
          <div><span class="label">Generation proof</span><strong>{_esc(generation.get('proof_gate') or '—')} / {_esc(generation.get('status') or '—')}</strong></div>
          <div><span class="label">Joined placement match</span><strong>{_esc(generation.get('server_placements_match_joined_plan'))}</strong></div>
          <div><span class="label">Load proof</span><strong>{_esc(load.get('proof_gate') or '—')} / {_esc(load.get('status') or '—')} · requests {_esc(load.get('request_count') or 0)}</strong></div>
          <div><span class="label">Boundary</span><code>{_esc(report.get('claim_boundary') or '—')}</code></div>
        </div>
        <p class="reason">Failed checks: {_esc(failed_checks)}</p>
      </section>
    """



def _speculative_plan_panel(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ""
    verifier = plan.get("verifier") or {}
    draft = plan.get("draft") or {}
    contract = plan.get("correctness_contract") or {}
    phone_policy = plan.get("phone_policy") or {}
    phones = draft.get("phone_candidates") or []
    phone_rows = "".join(
        "<tr>"
        f"<td>{_esc((phone or {}).get('hostname'))}</td>"
        f"<td>{_esc((phone or {}).get('runtime') or '—')}</td>"
        f"<td>{_esc((phone or {}).get('role') or 'async_draft_provider_only')}</td>"
        "</tr>"
        for phone in phones
        if isinstance(phone, dict)
    )
    policy = "phones as draft providers only" if phone_policy.get("phones_as_draft_providers_only") else "no phone policy supplied"
    verifier_copy = "Verifier authoritative" if verifier.get("authoritative") else "Verifier not marked authoritative"
    return f"""
      <section class="card wide speculative-plan">
        <h2>Speculative decode plan</h2>
        <div class="grid two">
          <div><span class="label">Verifier</span><strong>{_esc(verifier.get('model_id') or '—')}</strong></div>
          <div><span class="label">Verifier authority</span><strong>{_esc(verifier_copy)}</strong></div>
          <div><span class="label">Draft model</span><strong>{_esc(draft.get('model_id') or '—')}</strong></div>
          <div><span class="label">Draft window</span><strong>{_esc(draft.get('max_draft_tokens') or '—')} tokens · acceptance {_esc(draft.get('acceptance_window') or '—')}</strong></div>
          <div><span class="label">Boundary</span><code>{_esc(plan.get('claim_boundary'))}</code></div>
          <div><span class="label">Phone policy</span><strong>{_esc(policy)}</strong></div>
        </div>
        <p class="reason">Accepted tokens require verifier match: {_esc(contract.get('accepted_tokens_require_verifier_match'))}. Drafting is speed planning only; generation is not proven.</p>
        <table>
          <thead><tr><th>Phone/draft peer</th><th>Runtime</th><th>Role</th></tr></thead>
          <tbody>{phone_rows or '<tr><td colspan="3">No phone draft candidates discovered</td></tr>'}</tbody>
        </table>
      </section>
    """


def _draft_report_panel(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    provider = report.get("provider") or {}
    proposal = report.get("proposal") or {}
    verdict = report.get("verdict") or {}
    counters = report.get("dashboard_counters") or {}
    accepted = ", ".join(str(token) for token in verdict.get("accepted_tokens") or []) or "—"
    rejected = ", ".join(str(token) for token in verdict.get("rejected_tokens") or []) or "—"
    committed = ", ".join(str(token) for token in verdict.get("committed_tokens") or []) or "—"
    return f"""
      <section class="card wide draft-report">
        <h2>Draft-provider contract smoke</h2>
        <div class="grid two">
          <div><span class="label">Provider</span><strong>{_esc(provider.get('provider_id') or proposal.get('provider_id') or '—')}</strong></div>
          <div><span class="label">Kind</span><strong>{_esc(provider.get('provider_kind') or proposal.get('provider_kind') or '—')}</strong></div>
          <div><span class="label">Proposed / accepted / rejected</span><strong>{_esc(counters.get('proposed'))} / {_esc(counters.get('accepted'))} / {_esc(counters.get('rejected'))}</strong></div>
          <div><span class="label">Acceptance rate</span><strong>{_esc(counters.get('acceptance_rate'))}</strong></div>
          <div><span class="label">Verifier fallback</span><strong>{_esc(verdict.get('verifier_fallback_token') if verdict.get('verifier_fallback_token') is not None else '—')}</strong></div>
          <div><span class="label">Boundary</span><code>{_esc(report.get('claim_boundary'))}</code></div>
        </div>
        <table>
          <thead><tr><th>Accepted prefix</th><th>Rejected draft suffix</th><th>Committed tokens</th></tr></thead>
          <tbody><tr><td>{_esc(accepted)}</td><td>{_esc(rejected)}</td><td>{_esc(committed)}</td></tr></tbody>
        </table>
        <p class="muted">Verifier remains authoritative. This is a provider-contract smoke only: generation speedup and phone inference are not proven.</p>
      </section>
    """


def _multi_block_diagnostics_panel(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    summary = report.get("summary") or {}
    coverage = report.get("coverage") or {}
    covered = coverage.get("covered_layers")
    total = coverage.get("total_layers")
    missing = coverage.get("missing_layers")
    coverage_copy = (
        f"coverage {covered}/{total} layers · missing {missing}"
        if covered is not None and total is not None and missing is not None
        else "coverage unavailable"
    )
    server_rows = "".join(
        "<tr>"
        f"<td>{_esc(item.get('server_index'))}</td>"
        f"<td>{_esc(item.get('block_range') or '—')}</td>"
        f"<td>{_esc(item.get('health') or '—')}</td>"
        f"<td>{_bool_badge(item.get('started'))}</td>"
        f"<td>{_bool_badge(item.get('announced_block_range'))}</td>"
        f"<td>{_bool_badge(item.get('has_rpc_evidence'))}</td>"
        f"<td>{_esc(', '.join(item.get('errors') or []) or '—')}</td>"
        "</tr>"
        for item in report.get("servers") or []
        if isinstance(item, dict)
    )
    action_rows = "".join(f"<li>{_esc(item)}</li>" for item in report.get("operator_actions") or [])
    inference_copy = "inference proven" if report.get("inference_proven") else "inference not proven"
    return f"""
      <section class="card wide multi-block-diagnostics">
        <h2>Multi-block diagnostics</h2>
        <div class="grid two">
          <div><span class="label">Model</span><strong>{_esc(report.get('model_id') or '—')}</strong></div>
          <div><span class="label">Status</span><strong>{_esc(summary.get('status') or '—')}</strong></div>
          <div><span class="label">Combined range</span><strong>{_esc(report.get('combined_block_range') or '—')}</strong></div>
          <div><span class="label">Layer coverage</span><strong>{_esc(coverage_copy)}</strong></div>
          <div><span class="label">Healthy / unhealthy servers</span><strong>{_esc(summary.get('healthy_servers'))} / {_esc(summary.get('unhealthy_servers'))}</strong></div>
          <div><span class="label">Inference claim</span><strong class="warn">{_esc(inference_copy)}</strong></div>
          <div><span class="label">Boundary</span><code>{_esc(report.get('claim_boundary'))}</code></div>
          <div><span class="label">Proof promotion</span><strong>{_bool_badge(report.get('can_update_proof_status'))}</strong></div>
        </div>
        <table>
          <thead><tr><th>#</th><th>Layers</th><th>Health</th><th>Started</th><th>Announced</th><th>RPC evidence</th><th>Errors</th></tr></thead>
          <tbody>{server_rows or '<tr><td colspan="7">No server diagnostics supplied</td></tr>'}</tbody>
        </table>
        <h3>Operator actions</h3>
        <ul>{action_rows or '<li>No operator actions supplied; use multi_block_proof.py verify for proof promotion.</li>'}</ul>
        <p class="muted">Diagnostics are observability only. They never start servers, send requests, or update proof status.</p>
      </section>
    """


def render_dashboard_html(document: dict[str, Any], *, refresh_seconds: int | None = 20) -> str:
    """Render a self-contained dashboard HTML document."""
    roster_summary = document.get("roster", {}).get("summary", {})
    evidence_summary = document.get("evidence_summary", {})
    refresh_meta = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds and refresh_seconds > 0
        else ""
    )
    refresh_copy = (
        f"auto-refreshes every {int(refresh_seconds)} seconds"
        if refresh_seconds and refresh_seconds > 0
        else "static snapshot"
    )
    boundaries = "".join(f"<li>{_esc(item)}</li>" for item in document.get("claim_boundaries") or [])
    synthetic_panel = ""
    if document.get("synthetic_10_laptop_route"):
        synthetic_panel = _route_card('Synthetic 10-laptop target route', document.get('synthetic_10_laptop_route') or {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>BloomBee Distributed Inference Demo Dashboard</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0d1b2f; --panel2:#122744; --line:#2a4b73; --text:#e9f3ff; --muted:#95acc8; --ok:#58d68d; --warn:#f7c948; --fail:#ff6b6b; --moon:#b9ccff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at 20% -10%, #233e73 0, transparent 34%), var(--bg); color:var(--text); }}
    header {{ padding:32px 36px 18px; border-bottom:1px solid var(--line); background:linear-gradient(135deg, rgba(185,204,255,.18), rgba(12,27,47,.6)); }}
    h1 {{ margin:0 0 8px; font-size: clamp(28px, 5vw, 48px); letter-spacing:-0.04em; }}
    h2 {{ margin:0 0 16px; color:var(--moon); }}
    p {{ line-height:1.5; }}
    code, pre {{ background:#07111f; color:#d7e5ff; border:1px solid var(--line); border-radius:10px; }}
    code {{ padding:.1rem .35rem; }}
    pre {{ padding:12px; overflow:auto; min-height:54px; }}
    main {{ padding:24px 36px 42px; display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
    .card {{ background:linear-gradient(180deg, rgba(18,39,68,.96), rgba(13,27,47,.96)); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 14px 40px rgba(0,0,0,.28); }}
    .wide {{ grid-column:1 / -1; }}
    .stats {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:14px; }}
    .stat {{ background:rgba(7,17,31,.68); border:1px solid var(--line); border-radius:14px; padding:12px 16px; min-width:150px; }}
    .stat strong {{ display:block; font-size:28px; }}
    .label, .muted {{ color:var(--muted); }}
    .label {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }}
    .hero-model {{ font-size:24px; font-weight:750; margin-bottom:16px; overflow-wrap:anywhere; }}
    .grid {{ display:grid; gap:14px; }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .reason {{ color:#c9d9ef; border-left:3px solid var(--moon); padding-left:12px; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; border-radius:12px; }}
    th, td {{ text-align:left; border-bottom:1px solid rgba(42,75,115,.7); padding:10px 9px; vertical-align:top; }}
    th {{ color:#bad0f3; background:rgba(7,17,31,.55); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .badge {{ display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:12px; }}
    .progress-wrap {{ height:18px; background:#07111f; border:1px solid var(--line); border-radius:999px; overflow:hidden; margin:0 0 16px; }}
    .progress-bar {{ height:100%; background:linear-gradient(90deg, #7dd3fc, #c4b5fd); border-radius:999px; }}
    .status-bar-text {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .ok {{ background:rgba(88,214,141,.18); color:var(--ok); }}
    .fail {{ background:rgba(255,107,107,.16); color:var(--fail); }}
    .warn {{ color:var(--warn); }}
    .muted.badge {{ background:rgba(149,172,200,.14); color:var(--muted); }}
    .layer-map-group {{ margin-top:18px; }}
    .layer-map-track {{ display:flex; min-height:74px; border:1px solid var(--line); border-radius:16px; overflow:hidden; background:#07111f; box-shadow: inset 0 0 24px rgba(185,204,255,.08); }}
    .layer-segment {{ display:flex; flex-direction:column; justify-content:center; gap:4px; min-width:90px; padding:10px; border-right:1px solid rgba(7,17,31,.78); color:#06111f; overflow:hidden; }}
    .layer-segment strong, .layer-segment span {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .layer-segment strong {{ font-size:13px; }}
    .layer-segment span {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
    .status-planned, .status-ready, .status-healthy {{ background:linear-gradient(135deg, #72f3b2, #87d8ff); }}
    .status-proof_evidence {{ background:linear-gradient(135deg, #c4b5fd, #93c5fd); }}
    .status-unhealthy, .status-failed {{ background:linear-gradient(135deg, #ff9f9f, #f7c948); }}
    ul {{ margin:0; padding-left:20px; }}
    .demo-readiness tr.needed {{ background:rgba(255,159,159,.12); }}
    .demo-readiness tr.partial {{ background:rgba(247,201,72,.10); }}
    .demo-readiness tr.ready {{ background:rgba(114,243,178,.08); }}
    .demo-readiness .icon {{ font-size:18px; font-weight:700; display:inline-block; width:24px; text-align:center; }}
    .demo-readiness .needed .icon {{ color:#ff9f9f; }}
    .demo-readiness .partial .icon {{ color:#f7c948; }}
    .demo-readiness .ready .icon {{ color:#72f3b2; }}
    .layer-gap-warning {{ border:1px solid rgba(247,201,72,.35); border-radius:12px; padding:12px 18px; background:rgba(247,201,72,.06); }}
    .layer-gap-warning h3 {{ color:#f7c948; margin:0 0 8px; }}
    footer {{ padding:0 36px 36px; color:var(--muted); }}
    @media (max-width: 900px) {{ main, .two {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
  </style>
</head>
<body>
  <header>
    <h1>BloomBee Distributed Inference Demo Dashboard</h1>
    <p class="muted">Generated {_esc(document.get('generated_at'))}; {refresh_copy}. Shows connected devices, route choices, measured throughput, inference proof status, and recovery/S2S telemetry.</p>
    <div class="stats">
      <div class="stat"><span class="label">Peers</span><strong>{_esc(roster_summary.get('peer_count', 0))}</strong></div>
      <div class="stat"><span class="label">Free memory</span><strong>{_fmt_num(roster_summary.get('free_memory_gb'), 1)} GB</strong></div>
      <div class="stat"><span class="label">Evidence OK</span><strong>{_esc(evidence_summary.get('passed', 0))}/{_esc(evidence_summary.get('total', 0))}</strong></div>
    </div>
  </header>
  <main>
    {_status_panel(document.get('mvp_status') or {})}
    {_demo_readiness_panel(document)}
    {_proof_state_panel(document.get('proof_state'))}
    {_joined_layer_plan_panel(document.get('joined_layer_plan'))}
    {_chain_schedule_panel(document.get('chain_schedule'))}
    {_handoff_bundle_panel(document.get('handoff_bundle'))}
    {_proof_orchestration_panel(document.get('proof_orchestration'))}
    {_physical_showcase_panel(document.get('physical_showcase'))}
    {_speculative_plan_panel(document.get('speculative_plan'))}
    {_draft_report_panel(document.get('draft_report'))}
    {_multi_block_diagnostics_panel(document.get('multi_block_diagnostics'))}
    {_layers_map_panel(document.get('layers_map') or {})}
    {_token_stream_panel(document.get('token_stream') or {})}
    {_model_fit_matrix_panel(document.get('model_fit_matrix') or {})}
    {_request_telemetry_panel(document.get('request_telemetry') or {})}
    {_route_card('Current real-swarm route', document.get('real_route') or {})}
    {synthetic_panel}
    {_devices_table(document.get('roster') or {}, document.get('benchmarks') or {})}
    {_layer_placement_table(document.get('layer_placements') or [])}
    {_bench_table(document.get('benchmarks') or {})}
    {_evidence_table(document.get('evidence') or [])}
    {_telemetry_panel(document.get('telemetry') or {})}
    <section class="card wide"><h2>Honest claim boundaries</h2><ul>{boundaries}</ul></section>
  </main>
  <footer>Moonlit dashboard artifact for distributed-inference-mvp. Re-run <code>python mvp_capabilities/demo_dashboard.py</code> during the demo to update the snapshot.</footer>
</body>
</html>
"""


def write_dashboard(
    document: dict[str, Any],
    out_path: str | Path = DEFAULT_OUT,
    *,
    refresh_seconds: int | None = 20,
) -> Path:
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard_html(document, refresh_seconds=refresh_seconds), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-dir", action="append", default=None, help="Capability directory; may be repeated")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--bench-matrix", default=None, help="JSON from bench_matrix.py")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--proof-state", default=None, help="Optional JSON from proof_state.py")
    parser.add_argument("--joined-layer-plan", default=None, help="Optional JSON from join_layer_plan.py")
    parser.add_argument("--chain-schedule", default=None, help="Optional JSON from chain_scheduler.py")
    parser.add_argument("--handoff-bundle", default=None, help="Optional JSON from join_http_server.py /handoff")
    parser.add_argument("--proof-orchestration", default=None, help="Optional JSON from proof_orchestrator.py or join_http_server.py /proof-orchestration")
    parser.add_argument("--physical-showcase", default=None, help="Optional JSON from physical_showcase_proof.py")
    parser.add_argument("--speculative-plan", default=None, help="Optional JSON from speculative_decode_plan.py or join_http_server.py /speculative")
    parser.add_argument("--draft-report", default=None, help="Optional JSON from draft_provider.py")
    parser.add_argument("--multi-block-diagnostics", default=None, help="Optional JSON from multi_block_diagnostics.py")
    parser.add_argument("--request-log", action="append", default=None, help="Direct-client request log with [direct] RESULT lines; may be repeated")
    parser.add_argument("--token-stream-log", action="append", default=None, help="Live generation JSONL or [TOKEN_STREAM] log; may be repeated")
    parser.add_argument("--telemetry-log", action="append", default=None, help="Server/client log file with [RECOVERY_EVENT]/[S2S_PUSH_EVENT] lines; may be repeated")
    parser.add_argument("--synthetic-m4-laptops", type=int, default=0, help="Opt-in planning view: append N synthetic M4 laptop peers. Default 0 for real-demo dashboards.")
    parser.add_argument("--synthetic-total-gb", type=float, default=24.0)
    parser.add_argument("--synthetic-free-gb", type=float, default=20.0)
    parser.add_argument("--refresh-seconds", type=int, default=20, help="HTML meta-refresh interval; use 0 to disable")
    parser.add_argument("--watch-seconds", type=float, default=0.0, help="Regenerate the dashboard every N seconds until interrupted; 0 writes once")
    parser.add_argument("--watch-ticks", type=int, default=None, help="Bound watch mode to N writes; mainly useful for tests")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    def build_write_once() -> tuple[Path, dict[str, Any]]:
        doc = build_dashboard_document(
            cap_dirs=args.cap_dir or [DEFAULT_CAP_DIR],
            registry_path=args.registry,
            bench_matrix_path=args.bench_matrix,
            evidence_dir=args.evidence_dir,
            proof_state_path=args.proof_state,
            joined_layer_plan_path=args.joined_layer_plan,
            chain_schedule_path=args.chain_schedule,
            handoff_bundle_path=args.handoff_bundle,
            proof_orchestration_path=args.proof_orchestration,
            physical_showcase_path=args.physical_showcase,
            speculative_plan_path=args.speculative_plan,
            draft_report_path=args.draft_report,
            multi_block_diagnostics_path=args.multi_block_diagnostics,
            request_logs=args.request_log,
            telemetry_logs=args.telemetry_log,
            token_stream_logs=args.token_stream_log,
            synthetic_m4_laptops=args.synthetic_m4_laptops,
            synthetic_total_gb=args.synthetic_total_gb,
            synthetic_free_gb=args.synthetic_free_gb,
        )
        path = write_dashboard(
            doc,
            args.out,
            refresh_seconds=args.refresh_seconds if args.refresh_seconds > 0 else None,
        )
        return path, doc

    tick = 0
    while True:
        tick += 1
        path, doc = build_write_once()
        print(
            json.dumps(
                {
                    "ok": True,
                    "tick": tick,
                    "out": str(path),
                    "peers": doc["roster"]["summary"].get("peer_count", 0),
                    "evidence": doc["evidence_summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not args.watch_seconds or (args.watch_ticks is not None and tick >= args.watch_ticks):
            break
        time.sleep(max(0.0, float(args.watch_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
