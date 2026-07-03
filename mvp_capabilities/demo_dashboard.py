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
    from mvp_capabilities.route_picker import (
        DEFAULT_REGISTRY,
        explain_route,
        load_registry,
        synthetic_m4_laptops as make_synthetic_m4_laptops,
    )
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, roster_document
except ModuleNotFoundError:  # direct script execution: python mvp_capabilities/demo_dashboard.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.mvp_status import build_status_report  # type: ignore[no-redef]
    from mvp_capabilities.route_picker import (  # type: ignore[no-redef]
        DEFAULT_REGISTRY,
        explain_route,
        load_registry,
        synthetic_m4_laptops as make_synthetic_m4_laptops,
    )
    from mvp_capabilities.swarm_roster import DEFAULT_CAP_DIR, load_roster, roster_document  # type: ignore[no-redef]


DEFAULT_EVIDENCE_DIR = Path(__file__).with_name("distributed_evidence")
DEFAULT_OUT = Path(".local/demo-dashboard.html")
_TELEMETRY_MARKERS = ("[RECOVERY_EVENT]", "[S2S_PUSH_EVENT]")
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


def build_dashboard_document(
    *,
    cap_dirs: Iterable[str | Path] | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY,
    bench_matrix_path: str | Path | None = None,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    telemetry_logs: Iterable[str | Path] | None = None,
    synthetic_m4_laptops: int = 0,
    synthetic_total_gb: float = 24.0,
    synthetic_free_gb: float = 20.0,
) -> dict[str, Any]:
    """Collect every dashboard panel from existing MVP artifacts."""
    real_peers = load_roster(list(cap_dirs or [DEFAULT_CAP_DIR]))
    registry = load_registry(registry_path)
    bench_matrix = _read_json(bench_matrix_path, {})
    real_route = explain_route(real_peers, registry, bench_matrix=bench_matrix)
    synthetic_route = None
    if synthetic_m4_laptops > 0:
        synthetic_route = explain_route(
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
    layer_placements = collect_layer_placements(evidence)
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
        "layer_placements": layer_placements,
        "evidence_summary": {"passed": passed_evidence, "total": len(evidence)},
        "telemetry": parse_telemetry_logs(telemetry_logs),
        "mvp_status": build_status_report(),
        "claim_boundaries": claim_boundaries,
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _route_card(title: str, route: dict[str, Any]) -> str:
    picked = route.get("picked") or {}
    return f"""
      <section class="card route">
        <h2>{_esc(title)}</h2>
        <div class="hero-model">{_esc(picked.get('model_id') or 'No model selected')}</div>
        <div class="grid two">
          <div><span class="label">Placement</span><strong>{_esc(picked.get('placement'))}</strong></div>
          <div><span class="label">Supported</span><strong>{_bool_badge(picked.get('supported'))}</strong></div>
          <div><span class="label">Swarm free GB</span><strong>{_fmt_num(picked.get('swarm_free_gb'), 1)}</strong></div>
          <div><span class="label">Measured decode tok/s</span><strong>{_fmt_measured_rate(picked.get('measured_decode_tok_per_s'))}</strong></div>
        </div>
        <p class="reason">{_esc(picked.get('reason'))}</p>
      </section>
    """


def _devices_table(roster: dict[str, Any]) -> str:
    peers = roster.get("peers") or []
    rows = []
    for peer in peers:
        mem = peer.get("memory") or {}
        accel = peer.get("accelerator") or {}
        mobile = peer.get("mobile") or {}
        rows.append(
            "<tr>"
            f"<td>{_esc(peer.get('hostname'))}</td>"
            f"<td>{_esc(accel.get('device') or 'cpu')}</td>"
            f"<td>{_fmt_num(mem.get('free_gb'), 1)} / {_fmt_num(mem.get('total_gb'), 1)}</td>"
            f"<td>{_esc((peer.get('network') or {}).get('tailscale_ip') or '—')}</td>"
            f"<td>{_bool_badge(mobile.get('is_mobile'))}</td>"
            "</tr>"
        )
    return """
      <section class="card wide">
        <h2>Connected devices</h2>
        <table>
          <thead><tr><th>Host</th><th>Device</th><th>Free / total GB</th><th>Tailscale</th><th>Mobile</th></tr></thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </section>
    """.format(rows="\n".join(rows) or '<tr><td colspan="5">No peers found</td></tr>')


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
    ul {{ margin:0; padding-left:20px; }}
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
    {_route_card('Current real-swarm route', document.get('real_route') or {})}
    {synthetic_panel}
    {_devices_table(document.get('roster') or {})}
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
            telemetry_logs=args.telemetry_log,
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
