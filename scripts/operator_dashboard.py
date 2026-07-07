#!/usr/bin/env python3
"""Generate a self-contained operator dashboard for distributed inference testing.

One HTML file — open in any browser. No server needed.

Features:
  - Configurable coordinator URL (type it in, no hardcoding)
  - Live QR code with share link (fetches join offer from coordinator)
  - Offline fallback: type a join URL manually to generate QR
  - Clear "what happens when I scan" explanation for users
  - Live swarm roster, layer plan, route panels
  - Computer-agnostic — works with any coordinator, not just m4pro
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path


CSS = r"""
:root {
  color-scheme: dark;
  --bg: #07111f; --panel: #0d1b2f; --panel2: #122744;
  --line: #2a4b73; --text: #e9f3ff; --muted: #95acc8;
  --ok: #58d68d; --warn: #f7c948; --fail: #ff6b6b;
  --moon: #b9ccff; --accent: #7dd3fc;
}
* { box-sizing: border-box; margin: 0; }
body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: radial-gradient(circle at 20% -10%, #233e73 0, transparent 34%), var(--bg);
  color: var(--text); min-height: 100vh;
}
header {
  padding: 20px 28px 14px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(185,204,255,.18), rgba(12,27,47,.6));
}
h1 { font-size: 28px; letter-spacing: -0.04em; margin-bottom:4px; }
h2 { color: var(--moon); font-size: 14px; font-weight: 400; }
#coord-bar {
  display: flex; gap: 8px; align-items: center; margin-top: 10px;
  padding: 8px 12px; background: rgba(7,17,31,.6); border: 1px solid var(--line);
  border-radius: 10px; font-size: 12px;
}
#coord-bar label { color: var(--muted); white-space: nowrap; }
#coord-bar input {
  flex: 1; background: #050e1a; border: 1px solid var(--line);
  color: var(--accent); padding: 6px 10px; border-radius: 6px;
  font-size: 12px; font-family: ui-monospace, monospace; min-width: 200px;
}
#coord-bar .status-dot { margin: 0 4px 0 0; }
nav { display: flex; gap: 6px; margin-top: 10px; }
nav button {
  background: rgba(185,204,255,.1); border: 1px solid var(--line);
  color: var(--muted); padding: 6px 14px; border-radius: 8px;
  cursor: pointer; font-size: 12px; font-weight: 600; transition: all .15s;
}
nav button.active, nav button:hover {
  background: rgba(125,211,252,.18); color: var(--accent); border-color: var(--accent);
}
main {
  padding: 20px 28px 40px;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px;
}
.card {
  background: linear-gradient(180deg, rgba(18,39,68,.96), rgba(13,27,47,.96));
  border: 1px solid var(--line); border-radius: 16px;
  padding: 18px; box-shadow: 0 12px 36px rgba(0,0,0,.26);
}
.card.wide { grid-column: 1 / -1; }
.card h3 { color: var(--moon); margin-bottom: 12px; font-size: 14px; letter-spacing: .04em; text-transform: uppercase; }
.step {
  background: rgba(7,17,31,.6); border: 1px solid var(--line);
  border-radius: 12px; padding: 14px; margin-bottom: 10px;
}
.step h4 { color: var(--accent); margin-bottom: 6px; font-size: 13px; }
.step p, .step li { color: var(--muted); line-height: 1.5; font-size: 12px; }
pre {
  background: #050e1a; color: #d7e5ff; border: 1px solid var(--line);
  border-radius: 8px; padding: 10px; overflow-x: auto; font-size: 11px;
  line-height: 1.5; margin: 6px 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.copy-btn {
  float: right; background: rgba(125,211,252,.15); border: 1px solid var(--accent);
  color: var(--accent); padding: 3px 8px; border-radius: 6px;
  cursor: pointer; font-size: 10px; margin-bottom: 2px;
}
.copy-btn:hover { background: rgba(125,211,252,.3); }
.copy-btn.copied { background: rgba(88,214,141,.2); border-color: var(--ok); color: var(--ok); }
.qr-section { text-align: center; padding: 12px; }
.qr-section canvas, .qr-section img { border-radius: 10px; background: #fff; padding: 10px; }
.stats { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.stat {
  background: rgba(7,17,31,.68); border: 1px solid var(--line);
  border-radius: 12px; padding: 10px 14px; min-width: 100px; text-align: center;
}
.stat strong { display: block; font-size: 24px; color: var(--accent); }
.stat .label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
table { width: 100%; border-collapse: collapse; border-radius: 10px; overflow: hidden; font-size: 12px; }
th, td { text-align: left; border-bottom: 1px solid rgba(42,75,115,.7); padding: 8px 7px; }
th { color: #bad0f3; background: rgba(7,17,31,.55); font-size: 10px; text-transform: uppercase; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 7px; font-weight: 700; font-size: 10px; }
.badge.ok { background: rgba(88,214,141,.18); color: var(--ok); }
.badge.warn { background: rgba(247,201,72,.12); color: var(--warn); }
.badge.fail { background: rgba(255,107,107,.16); color: var(--fail); }
.loading { color: var(--muted); font-style: italic; padding: 16px; }
.error { color: var(--fail); padding: 10px; background: rgba(255,107,107,.08); border-radius: 8px; font-size: 12px; }
.info-box {
  background: rgba(125,211,252,.08); border: 1px solid rgba(125,211,252,.25);
  border-radius: 10px; padding: 12px; margin-top: 10px; font-size: 12px; line-height: 1.6;
}
.info-box strong { color: var(--accent); }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.status-dot.online { background: var(--ok); box-shadow: 0 0 6px var(--ok); }
.status-dot.offline { background: var(--fail); }
#manual-qr { display: none; }
.manual-toggle { color: var(--muted); font-size: 11px; cursor: pointer; text-decoration: underline; margin-top: 8px; }
.manual-toggle:hover { color: var(--accent); }
@media (max-width: 800px) {
  main { grid-template-columns: 1fr; padding: 14px; }
  header { padding: 14px; }
}
/* Pipeline visualization */
.pipeline-row {
  display: flex; align-items: center; gap: 12px; margin: 16px 0;
  flex-wrap: wrap; font-size: 11px;
}
.pipeline-peer {
  background: var(--panel2); border: 2px solid var(--line);
  border-radius: 12px; padding: 12px 16px; min-width: 160px;
  text-align: center; transition: border-color .3s;
}
.pipeline-peer.serving { border-color: var(--ok); }
.pipeline-peer.idle { border-color: var(--warn); opacity: .7; }
.pipeline-peer .peer-name { font-weight: 700; color: var(--accent); font-size: 12px; }
.pipeline-peer .peer-layers { color: var(--muted); margin: 4px 0; }
.pipeline-peer .peer-mem { color: var(--moon); font-size: 10px; }
.pipeline-arrow { color: var(--accent); font-size: 24px; font-weight: 700; }
.pipeline-mem-bar {
  height: 6px; background: var(--line); border-radius: 3px;
  margin-top: 6px; overflow: hidden;
}
.pipeline-mem-fill {
  height: 100%; border-radius: 3px; transition: width .5s;
  background: var(--accent);
}
.pipeline-mem-fill.high { background: var(--warn); }
.pipeline-mem-fill.critical { background: var(--fail); }
.pipeline-token {
  background: var(--panel2); border: 1px solid var(--ok);
  border-radius: 8px; padding: 8px 14px; text-align: center;
  min-width: 100px;
}
.pipeline-token .tok-label { color: var(--muted); font-size: 10px; }
.pipeline-token .tok-value { color: var(--ok); font-size: 14px; font-weight: 700; }
"""

JS = r"""
let COORDINATOR = "__COORDINATOR__";

function setCoordinator() {
  const input = document.getElementById('coord-input');
  const val = input.value.trim();
  if (val) {
    COORDINATOR = val.replace(/\/+$/, '');
    checkCoordinator();
    updateOperatorCommands();
  }
}

function updateOperatorCommands() {
  const cmdCoord = document.getElementById('cmd-coordinator');
  const cmdDash = document.getElementById('cmd-dashboard');
  if (cmdCoord) {
    cmdCoord.textContent = `cd ~/Projects/distributed-inference-mvp
python3 mvp_capabilities/join_http_server.py \\
  --host 0.0.0.0 --port 8787 \\
  --coordinator "${COORDINATOR}"`;
  }
  if (cmdDash) {
    cmdDash.textContent = `cd ~/Projects/distributed-inference-mvp
python3 scripts/operator_dashboard.py \\
  --coordinator "${COORDINATOR}" \\
  --out .local/operator-dashboard.html && open .local/operator-dashboard.html`;
  }
}

async function checkCoordinator() {
  const dot = document.getElementById('coord-status');
  const label = document.getElementById('coord-label');
  try {
    const resp = await fetch(COORDINATOR + '/healthz');
    if (resp.ok) {
      dot.className = 'status-dot online';
      label.textContent = 'connected';
      label.style.color = 'var(--ok)';
    } else {
      throw new Error('bad status');
    }
  } catch(e) {
    dot.className = 'status-dot offline';
    label.textContent = 'unreachable';
    label.style.color = 'var(--fail)';
  }
  generateQR();
  refreshAll();
}

async function fetchJSON(url, fallback) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}`);
    return resp.json();
  } catch(e) {
    return fallback || null;
  }
}

async function loadRoster() {
  const el = document.getElementById('roster-content');
  const data = await fetchJSON(COORDINATOR + '/active?token=*&max_age_seconds=600');
  if (!data || !data.active_peers || !data.active_peers.length) {
    el.innerHTML = data === null
      ? '<div class="error">Cannot reach coordinator. Check the URL above.</div>'
      : '<div class="loading">No peers connected yet. Share the join QR below to onboard devices.</div>';
    return;
  }
  const peers = data.active_peers;
  let rows = '';
  for (const p of peers) {
    const pid = esc(p.peer_id || 'unknown');
    const hostname = esc((p.capabilities && p.capabilities.hostname) || '-');
    const cpu = esc((p.capabilities && p.capabilities.cpu && p.capabilities.cpu.model) || '-');
    const ram = (p.capabilities && p.capabilities.memory && p.capabilities.memory.total_gb) || '-';
    const ts = p.timestamp ? new Date(p.timestamp * 1000).toLocaleTimeString() : '-';
    const ok = p.ok !== false;
    rows += `<tr>
      <td><span class="status-dot ${ok ? 'online' : 'offline'}"></span>${pid}</td>
      <td>${hostname}</td><td>${cpu}</td><td>${ram} GB</td><td>${ts}</td>
      <td><span class="badge ${ok ? 'ok' : 'fail'}">${ok ? 'alive' : 'stale'}</span></td>
    </tr>`;
  }
  const okCount = peers.filter(p => p.ok !== false).length;
  document.getElementById('stat-peers').textContent = peers.length;
  document.getElementById('stat-online').textContent = okCount;
  el.innerHTML = `<table>
    <thead><tr><th>Peer</th><th>Hostname</th><th>CPU</th><th>RAM</th><th>Last seen</th><th>Status</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

async function refreshAll() {
  loadRoster();
  loadPlan();
  loadRoute();
  loadPipeline();
  loadIosPeers();
}

async function loadIosPeers() {
  const card = document.getElementById('ios-peers-card');
  const el = document.getElementById('ios-peers-content');
  const data = await fetchJSON(COORDINATOR + '/active?token=*&max_age_seconds=600');
  if (!data || !data.active_peers) {
    card.style.display = 'none';
    return;
  }
  const ios = data.active_peers.filter(p =>
    (p.capabilities && p.capabilities.platform === 'ios') ||
    (p.capabilities && p.capabilities.role === 'draft_peer')
  );
  if (!ios.length) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';
  let rows = '';
  for (const p of ios) {
    const c = p.capabilities || {};
    const hostname = esc(c.hostname || p.peer_id || '?');
    const model = esc(c.mlx_model_id || c.mlx_model || 'mlx model unknown');
    const mem = (c.memory && c.memory.total_gb) || '?';
    const tok = c.tokens_per_sec_est || c.tok_per_sec || '~40';
    const ok = p.ok !== false;
    const age = p.timestamp ? Math.round((Date.now()/1000 - p.timestamp)) : null;
    const ageStr = age !== null ? `${age}s ago` : '?';
    rows += `<tr>
      <td><span class="status-dot ${ok ? 'online' : 'offline'}"></span>${esc(p.peer_id || '?')}</td>
      <td>${hostname}</td>
      <td style="font-family:monospace;font-size:10px;">${model}</td>
      <td>${mem} GB</td>
      <td>~${tok} tok/s</td>
      <td>${ageStr}</td>
      <td><span class="badge ${ok ? 'ok' : 'fail'}">${ok ? 'connected' : 'stale'}</span></td>
    </tr>`;
  }
  el.innerHTML = `<table>
    <thead><tr><th>Peer ID</th><th>Device</th><th>MLX Model</th><th>RAM</th><th>Speed</th><th>Last seen</th><th>Status</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}
async function loadPipeline() {
  const card = document.getElementById('pipeline-card');
  const viz = document.getElementById('pipeline-viz');
  const data = await fetchJSON(COORDINATOR + '/pipeline');
  if (!data || !data.peers || !data.peers.length) {
    card.style.display = 'none'; return;
  }
  card.style.display = 'block';

  let html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">' +
    '<span style="color:var(--accent);">Model: ' + esc(data.model_id || 'none') + '</span>' +
    '<span style="color:var(--muted);">· ' + data.serving_count + '/' + data.peer_count + ' serving</span>' +
    '</div>';

  html += '<div class="pipeline-row">';
  // Input box
  html += '<div class="pipeline-token"><div class="tok-label">Prompt</div><div class="tok-value">📝</div></div>';
  html += '<div class="pipeline-arrow">→</div>';

  for (let i = 0; i < data.peers.length; i++) {
    const p = data.peers[i];
    const cls = p.status === 'serving' ? 'serving' : 'idle';
    const memClass = p.memory_pct > 80 ? 'critical' : (p.memory_pct > 60 ? 'high' : '');
    html += '<div class="pipeline-peer ' + cls + '">' +
      '<div class="peer-name">' + esc(p.hostname) +
        (p.role === 'draft_peer' ? ' <span style="font-size:9px;background:var(--moon);color:var(--bg);padding:1px 5px;border-radius:4px;">📱</span>' : '') +
      '</div>' +
      '<div class="peer-layers">layers ' + esc(p.block_range || '?') + '</div>' +
      '<div class="peer-mem">' + (p.memory_pct >= 0 ? p.memory_pct + '% mem' : 'mem N/A') + ' · ~' + (p.latency_ms_est || 0) + 'ms</div>' +
      (p.memory_pct >= 0
        ? '<div class="pipeline-mem-bar"><div class="pipeline-mem-fill ' + memClass + '" style="width:' + Math.min(100, Math.max(0, p.memory_pct)) + '%;"></div></div>'
        : '<div style="color:var(--muted);font-size:9px;">memory unknown</div>') +
      '</div>';
    if (i < data.peers.length - 1) {
      html += '<div class="pipeline-arrow">→</div>';
    }
  }

  // Output box
  html += '<div class="pipeline-arrow">→</div>';
  html += '<div class="pipeline-token"><div class="tok-label">Output</div><div class="tok-value">✨</div></div>';
  html += '</div>';

  viz.innerHTML = html;
}
async function loadPlan() {
  const el = document.getElementById('plan-content');
  const data = await fetchJSON(COORDINATOR + '/plan');
  if (!data || !data.plan) {
    el.innerHTML = '<div class="loading">No layer plan yet. Onboard peers first.</div>'; return;
  }
  const allocs = data.plan.allocations || data.plan.layers || [];
  let rows = '';
  for (const a of allocs) {
    rows += `<tr><td>${esc(a.peer_id||a.node_id||'?')}</td><td>${esc(a.layer_range||a.layers||'?')}</td><td>${esc(a.role||'shard')}</td></tr>`;
  }
  el.innerHTML = `<table><thead><tr><th>Peer</th><th>Layers</th><th>Role</th></tr></thead><tbody>${rows}</tbody></table>`;
}
async function loadRoute() {
  const el = document.getElementById('route-content');
  const data = await fetchJSON(COORDINATOR + '/route');
  if (!data) { el.innerHTML = '<div class="loading">Route unavailable.</div>'; return; }
  const model = data.picked_model || data.model || '?';
  const placements = data.placements || data.allocation || [];
  let rows = '';
  for (const p of placements) {
    rows += `<tr><td>${esc(p.peer_id||p.node_id||'?')}</td><td>${esc(p.layers||'-')}</td></tr>`;
  }
  el.innerHTML = `<div style="font-size:16px;margin-bottom:10px;color:var(--accent);">${esc(model)}</div>
    <table><thead><tr><th>Peer</th><th>Layers</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function generateQR(url) {
  const el = document.getElementById('qr-container');
  const linkBox = document.getElementById('share-link-box');
  const input = document.getElementById('share-link-text');
  const landingUrl = COORDINATOR + '/';

  // QR points to coordinator landing page (normal HTTP, any browser opens it)
  if (url) {
    renderQR(url, 'manual');
    return;
  }

  // QR = coordinator landing page. Share link = same.
  renderQR(landingUrl, 'live');
  if (input) input.value = landingUrl;
  if (linkBox) linkBox.style.display = 'block';
  document.getElementById('offer-url').textContent = landingUrl;
  document.getElementById('offer-expires').textContent = 'no expiry';

  // Update the info box URL
  const landingEl = document.getElementById('coord-landing-url');
  if (landingEl) landingEl.textContent = landingUrl;
}

function renderQR(joinUrl, source) {
  const el = document.getElementById('qr-container');
  const offerEl = document.getElementById('offer-url');
  const linkBox = document.getElementById('share-link-box');
  const input = document.getElementById('share-link-text');

  if (offerEl) offerEl.textContent = joinUrl;
  if (input) input.value = joinUrl;
  if (linkBox) linkBox.style.display = 'block';

  if (typeof QRCode !== 'undefined') {
    el.innerHTML = '';
    try {
      new QRCode(el, { text: joinUrl, width: 200, height: 200, colorDark: '#07111f', colorLight: '#ffffff' });
    } catch(err) {
      el.innerHTML = '<div class="error">QR generation failed: ' + esc(err.message) + '</div>';
    }
  } else {
    el.innerHTML = '<div class="muted" style="padding:20px;">QR library not loaded (offline?).<br>Use the share link below instead.</div>';
  }
}

function showManualQR() {
  const div = document.getElementById('manual-qr');
  div.style.display = 'block';
}

function generateManualQR() {
  const input = document.getElementById('manual-join-url');
  const url = input.value.trim();
  if (url) {
    document.getElementById('manual-qr').style.display = 'none';
    generateQR(url);
  }
}

function copyText(text) {
  // navigator.clipboard only works on HTTPS/localhost — fallback for HTTP LAN
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); } catch(e) {}
  document.body.removeChild(ta);
}

function copyShareLink() {
  const input = document.getElementById('share-link-text');
  copyText(input.value);
  const btn = document.getElementById('copy-link-btn');
  if (btn) { btn.textContent = '✓ Copied!'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = '📋 Copy'; btn.classList.remove('copied'); }, 2000); }
}

function copyCode(btn) {
  // Each Copy button is directly followed by its <pre> sibling.
  let pre = btn.nextElementSibling;
  // If the button isn't right before the pre (edge case), search its parent.
  if (!pre || pre.tagName !== 'PRE') {
    pre = btn.parentElement.querySelector('pre');
  }
  if (pre) {
    // Strip # comment lines and leading/trailing blank lines so the
    // pasted command runs cleanly in a shell (macOS zsh has
    // INTERACTIVE_COMMENTS off by default).
    const NL = String.fromCharCode(10);
    const text = pre.textContent
      .split(NL)
      .filter(line => !line.match(/^[ \t]*#/))
      .join(NL)
      .trim();
    copyText(text);
    btn.textContent = 'Copied!'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  }
}

function showTab(name) {
  document.querySelectorAll('.tab-page').forEach(p => p.style.display = 'none');
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  const tab = document.getElementById('tab-' + name);
  if (tab) tab.style.display = 'block';
  const btn = document.querySelector(`nav button[data-tab="${name}"]`);
  if (btn) btn.classList.add('active');
  if (name === 'live') refreshAll();
  if (name === 'models') loadCompatible();
  if (name === 'infer') loadInferenceModels();
  if (name === 'deploy') loadDeployModels();
}

async function loadInferenceModels() {
  const sel = document.getElementById('infer-model');
  const data = await fetchJSON(COORDINATOR + '/compatible?token=*');
  if (!data || !data.compatible_models) {
    sel.innerHTML = '<option value="">Coordinator unreachable</option>';
    return;
  }
  const runnable = data.compatible_models.filter(m => m.status === 'compatible' || m.status === 'single_peer');
  if (!runnable.length) {
    sel.innerHTML = '<option value="">No compatible models — connect more peers</option>';
    return;
  }
  sel.innerHTML = '<option value="">-- Select a model --</option>' +
    runnable.map(m => '<option value="' + esc(m.model_id) + '">' +
      esc(m.model_id.split('/').pop()) + ' (' + m.params_b + 'B' + (m.supports_moe ? ', MoE' : '') + ', ' + m.required_gb + 'GB)</option>'
    ).join('');
  sel.onchange = onModelSelect;
}

function onModelSelect() {
  const sel = document.getElementById('infer-model');
  const infoEl = document.getElementById('infer-model-info');
  const planEl = document.getElementById('infer-plan-content');
  const val = sel.value;
  if (!val) { infoEl.innerHTML = ''; planEl.innerHTML = '<div class="loading">Select a model above to see the plan.</div>'; return; }
  infoEl.innerHTML = 'Model: <strong style="color:var(--accent);">' + esc(val) + '</strong>';

  // Fetch layer plan for selected model
  fetchJSON(COORDINATOR + '/plan?token=*&model=' + encodeURIComponent(val)).then(data => {
    if (!data || !data.plan) {
      planEl.innerHTML = '<div class="loading">No plan available. Need peers connected.</div>';
      return;
    }
    const allocs = data.plan.allocations || data.plan.layers || [];
    if (!allocs.length) {
      planEl.innerHTML = '<div class="error">Plan returned no allocations. Try a smaller model or connect more peers.</div>';
      return;
    }
    let rows = '';
    for (const a of allocs) {
      rows += '<tr><td>' + esc(a.peer_id || a.node_id || '?') + '</td><td>' + esc(a.layer_range || a.layers || '?') + '</td><td>' + esc(a.role || 'shard') + '</td></tr>';
    }
    planEl.innerHTML = '<table><thead><tr><th>Peer</th><th>Layers</th><th>Role</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }).catch(() => {
    planEl.innerHTML = '<div class="error">Could not load plan from coordinator.</div>';
  });
}

async function runInference() {
  const sel = document.getElementById('infer-model');
  const prompt = document.getElementById('infer-prompt').value.trim();
  const btn = document.getElementById('infer-btn');
  const status = document.getElementById('infer-status');
  const outSection = document.getElementById('infer-output-section');
  const outText = document.getElementById('infer-output');
  const outStats = document.getElementById('infer-stats');

  if (!sel.value || !prompt) {
    status.textContent = 'Select a model and enter a prompt.';
    status.style.color = 'var(--warn)';
    return;
  }

  btn.disabled = true; btn.textContent = '⏳ Generating...';
  status.textContent = 'Sending request to swarm...'; status.style.color = 'var(--accent)';
  outSection.style.display = 'block';
  outText.textContent = 'Generating...'; outStats.textContent = '';

  try {
    const t0 = performance.now();
    const resp = await fetch(COORDINATOR + '/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: sel.value,
        prompt: prompt,
        max_tokens: 128,
      }),
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);

    if (resp.ok) {
      const result = await resp.json();
      if (result.ready) {
        outText.textContent = '✓ Swarm ready for inference!\n\n' +
          result.serving_count + '/' + result.peer_count + ' peers serving\n\n' +
          'Run command:\n' + (result.inference_command || '(no command)');
        outStats.innerHTML = '<span class="badge ok">ready</span> · ' +
          result.serving_count + ' peers serving · model: ' + esc(sel.value.split('/').pop());
      } else {
        outText.textContent = '⚠️ Swarm not ready for inference.\n\n' +
          result.serving_count + '/' + result.peer_count + ' peers serving.\n\n' +
          'Next step: ' + (result.next_step || 'Start servers first.') + '\n\n' +
          'Server commands:\n' + (result.server_commands || []).join('\n\n');
        outStats.innerHTML = '<span class="badge warn">not ready</span> · ' +
          result.serving_count + '/' + result.peer_count + ' serving';
      }
      const tokCount = result.tokens_generated || result.output_tokens || '?';
    } else if (resp.status === 409 || resp.status === 400) {
      const err = await resp.json();
      outText.textContent = '⚠️ ' + (err.error || 'Cannot infer');
      if (err.next_step) outText.textContent += '\n\n' + err.next_step;
      if (err.server_commands) outText.textContent += '\n\nServer commands:\n' + err.server_commands.join('\n\n');
      outStats.innerHTML = '<span class="badge warn">' + (err.error || 'not ready') + '</span>';
    } else {
      const err = await resp.text();
      throw new Error(err || resp.statusText);
    }
  } catch(e) {
    outText.textContent = '❌ Inference failed: ' + e.message;
    outStats.innerHTML = '<span class="badge fail">error</span>';
    status.textContent = 'Failed. Check coordinator.';
    status.style.color = 'var(--fail)';
  } finally {
    btn.disabled = false; btn.textContent = '⚡ Generate';
    if (status.style.color !== 'var(--fail)') {
      status.textContent = 'Done.';
      status.style.color = 'var(--ok)';
    }
  }
}

async function loadCompatible() {
  const bestEl = document.getElementById('best-model-content');
  const modelsEl = document.getElementById('models-content');
  const data = await fetchJSON(COORDINATOR + '/compatible?token=*');
  if (!data || !data.compatible_models || !data.compatible_models.length) {
    const msg = data === null
      ? '<div class="error">Cannot reach coordinator. Check the URL above.</div>'
      : '<div class="loading">No peers connected. Onboard devices first to see compatible models.</div>';
    bestEl.innerHTML = msg;
    modelsEl.innerHTML = '';
    return;
  }

  // Best model card
  const best = data.best_model;
  if (best) {
    const moeBadge = best.supports_moe ? '<span class="badge ok" style="font-size:10px;margin-left:4px;">MoE</span>' : '';
    bestEl.innerHTML = '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">' +
      '<div style="flex:1;min-width:200px;">' +
        '<div style="font-size:22px;font-weight:700;color:var(--accent);margin-bottom:4px;">' + esc(best.model_id.split('/').pop()) + moeBadge + '</div>' +
        '<div style="color:var(--muted);font-size:12px;margin-bottom:8px;">' + esc(best.model_id) + '</div>' +
        '<div class="stat" style="display:inline-block;margin-right:8px;"><strong>' + (best.params_b || '?') + 'B</strong><span class="label">Params</span></div>' +
        '<div class="stat" style="display:inline-block;margin-right:8px;"><strong>' + (best.num_layers || '?') + '</strong><span class="label">Layers</span></div>' +
        '<div class="stat" style="display:inline-block;"><strong>' + (best.hidden_size || '?') + '</strong><span class="label">Hidden</span></div>' +
      '</div>' +
      '<div style="text-align:center;">' +
        '<div style="font-size:28px;font-weight:700;color:var(--ok);">' + (best.required_gb || '?') + ' GB</div>' +
        '<div style="color:var(--muted);font-size:10px;">required · ' + data.total_free_gb + ' GB free in swarm</div>' +
        '<div style="margin-top:6px;"><span class="badge ok" style="font-size:11px;padding:4px 10px;">✓ Swarm can host</span></div>' +
      '</div>' +
    '</div>';
  } else {
    bestEl.innerHTML = '<div class="loading">No model fits the current swarm. Connect more peers.</div>';
  }

  // All models table
  let rows = '';
  const statusOrder = { compatible: 0, single_peer: 1, insufficient: 2 };
  const sorted = [...data.compatible_models].sort((a, b) => {
    const sa = statusOrder[a.status] || 3, sb = statusOrder[b.status] || 3;
    if (sa !== sb) return sa - sb;
    return (b.params_b || 0) - (a.params_b || 0);
  });
  for (const m of sorted) {
    const statusBadge = m.status === 'compatible'
      ? '<span class="badge ok">compatible</span>'
      : m.status === 'single_peer'
        ? '<span class="badge warn">single peer</span>'
        : '<span class="badge fail">insufficient</span>';
    const barPct = m.required_gb > 0 ? Math.min(100, (data.total_free_gb / m.required_gb * 100).toFixed(0)) : 0;
    const barColor = barPct >= 100 ? 'var(--ok)' : barPct >= 40 ? 'var(--warn)' : 'var(--fail)';
    const needsBar = m.required_gb > 0
      ? '<div style="height:3px;background:rgba(42,75,115,.4);border-radius:2px;margin-top:3px;overflow:hidden;">' +
          '<div style="height:100%;width:' + barPct + '%;background:' + barColor + ';border-radius:2px;min-width:2px;"></div>' +
        '</div>'
      : '';
    rows += '<tr>' +
      '<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(m.model_id.split('/').pop()) + '<br><span style="color:var(--muted);font-size:10px;">' + esc(m.model_id) + '</span></td>' +
      '<td>' + (m.params_b || '?') + 'B' + (m.active_params_b && m.active_params_b !== m.params_b ? '<br><span style="color:var(--muted);font-size:10px;">' + m.active_params_b + 'B active</span>' : '') + '</td>' +
      '<td>' + (m.num_layers || '?') + '</td>' +
      '<td>' + (m.hidden_size || '?') + '</td>' +
      '<td>' + (m.required_gb || '?') + ' GB' + needsBar + '</td>' +
      '<td>' + (m.supports_moe ? 'MoE' : 'Dense') + '</td>' +
      '<td>' + statusBadge + '</td>' +
    '</tr>';
  }
  modelsEl.innerHTML = '<table>' +
    '<thead><tr><th>Model</th><th>Params</th><th>Layers</th><th>Hidden</th><th>Required</th><th>Arch</th><th>Status</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table>';
}

document.addEventListener('DOMContentLoaded', () => {
  generateQR();
  updateOperatorCommands();
  checkCoordinator();
  setInterval(() => {
    const liveTab = document.getElementById('tab-live');
    if (liveTab && liveTab.style.display !== 'none') refreshAll();
  }, 15000);
});

// ── Deploy tab ──
async function loadDeployModels() {
  const sel = document.getElementById('deploy-model');
  const data = await fetchJSON(COORDINATOR + '/compatible?token=*');
  if (!data || !data.compatible_models) {
    sel.innerHTML = '<option value="">Coordinator unreachable</option>';
    return;
  }
  const runnable = data.compatible_models.filter(m => m.status === 'compatible' || m.status === 'single_peer');
  if (!runnable.length) {
    sel.innerHTML = '<option value="">No compatible models</option>';
    return;
  }
  sel.innerHTML = '<option value="">-- Select a model --</option>' +
    runnable.map(m => '<option value="' + esc(m.model_id) + '">' +
      esc(m.model_id.split('/').pop()) + ' (' + m.params_b + 'B, ' + m.required_gb + 'GB)</option>'
    ).join('');
}

async function deployModel() {
  const sel = document.getElementById('deploy-model');
  const resultEl = document.getElementById('deploy-result');
  const jobsCard = document.getElementById('deploy-jobs-card');
  const jobsEl = document.getElementById('deploy-jobs');
  const modelId = sel.value;
  if (!modelId) { resultEl.innerHTML = '<span style="color:var(--warn);">Select a model first.</span>'; return; }
  resultEl.innerHTML = '<div class="loading">Deploying ' + esc(modelId.split('/').pop()) + '...</div>';
  try {
    const resp = await fetch(COORDINATOR + '/deploy?model_id=' + encodeURIComponent(modelId) + '&token=*', {method:'POST'});
    const data = await resp.json();
    if (data.error) {
      resultEl.innerHTML = '<span style="color:var(--fail);">Deploy failed: ' + esc(data.error) + '</span><br><span style="font-size:11px;color:var(--muted);">' + esc(data.plan_summary || '') + '</span>';
      return;
    }
    const jobCount = data.peer_count || 0;
    resultEl.innerHTML = '<span style="color:var(--ok);">Deployed ' + esc(data.model_id) + ' to ' + jobCount + ' peer(s)</span>';
    if (data.jobs && Object.keys(data.jobs).length) {
      jobsCard.style.display = 'block';
      let rows = '';
      for (const [hostname, job] of Object.entries(data.jobs)) {
        rows += '<tr><td style="color:var(--accent);font-family:monospace;">' + esc(hostname) + '</td>' +
          '<td>' + esc(job.block_indices || '-') + '</td>' +
          '<td style="font-family:monospace;font-size:10px;">' + esc(job.command || '(no command)') + '</td>' +
          '<td>' + esc(job.status || 'queued') + '</td></tr>';
      }
      jobsEl.innerHTML = '<table><thead><tr><th>Hostname</th><th>Blocks</th><th>Command</th><th>Status</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }
  } catch (e) {
    resultEl.innerHTML = '<span style="color:var(--fail);">Error: ' + esc(e.message) + '</span>';
  }
}

async function pollPeerJob() {
  const inp = document.getElementById('job-peer-input');
  const resultEl = document.getElementById('job-peer-result');
  const peerId = inp.value.trim();
  if (!peerId) { resultEl.textContent = 'Enter a peer ID first.'; return; }
  resultEl.textContent = 'Polling...';
  try {
    const resp = await fetch(COORDINATOR + '/job?peer_id=' + encodeURIComponent(peerId));
    const data = await resp.json();
    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    resultEl.textContent = 'Error: ' + e.message;
  }
}

let coordTimer;
document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('coord-input');
  if (inp) {
    inp.addEventListener('input', () => {
      clearTimeout(coordTimer);
      coordTimer = setTimeout(setCoordinator, 800);
    });
  }
});
"""


def _build_html(coordinator: str) -> str:
    coord_esc = html.escape(coordinator, quote=True)
    js = JS.replace("__COORDINATOR__", coord_esc)

    # f-strings treat `\<newline>` as a line continuation and silently eat
    # the backslash, which corrupts the multi-line shell commands shown
    # in the dashboard (Step 1, Step 2, Step 4a). Use a placeholder that
    # survives f-string interpolation, then swap it back to a literal `\<newline>`.
    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>BloomBee Operator Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/qrcodejs@1.0.0/qrcode.min.js"></script>
  <style>{CSS}</style>
</head>
<body>

<header>
  <h1>🌙 BloomBee Distributed Inference</h1>
  <h2>Operator Dashboard</h2>
  <div id="coord-bar">
    <span class="status-dot offline" id="coord-status"></span>
    <label for="coord-input">Coordinator:</label>
    <input id="coord-input" type="text" value="{coord_esc}" placeholder="http://192.168.1.100:8787" onchange="setCoordinator()">
    <span id="coord-label" style="font-size:11px;color:var(--muted);">checking...</span>
  </div>
  <nav>
    <button data-tab="runbook" class="active" onclick="showTab('runbook')">📖 Onboarding</button>
    <button data-tab="live" onclick="showTab('live')">📡 Live Swarm</button>
    <button data-tab="models" onclick="showTab('models')">🧠 Models</button>
    <button data-tab="infer" onclick="showTab('infer')">🚀 Generate</button>
    <button data-tab="deploy" onclick="showTab('deploy')">⚡ Deploy</button>
  </nav>
</header>

<!-- ====== ONBOARDING TAB ====== -->
<div id="tab-runbook" class="tab-page">

<main>
<div class="card">
  <h3>🚀 Operator Quick Start</h3>
  <p style="color:var(--muted);font-size:12px;margin-bottom:14px;line-height:1.5;">
    You're the <strong>swarm operator</strong>. Two commands to get everything running.
    Keep both terminals open. Bookmark this page — you'll come back to it.
  </p>

  <div class="step">
  <h4>Step 1 — Start the coordinator server</h4>
  <button class="copy-btn" onclick="copyCode(this)">Copy</button>
  <pre id="cmd-coordinator">cd ~/Projects/distributed-inference-mvp
python3 mvp_capabilities/join_http_server.py __BSLASH__
  --host 0.0.0.0 --port 8787 __BSLASH__
  --coordinator "{coord_esc}"</pre>
  <p>Pure stdlib, no venv needed. <strong>Leave this terminal open.</strong><br>
  <span style="color:var(--muted);font-size:11px;">
  If you re-run this while a coordinator is already alive, it'll print
  <strong>"ALREADY RUNNING"</strong> and exit cleanly instead of crashing with
  <code>Address already in use</code>. To restart: Ctrl+C the running one first.
  </span></p>
</div>

  <div class="step">
    <h4>Step 2 — Open this dashboard</h4>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre id="cmd-dashboard">cd ~/Projects/distributed-inference-mvp
python3 scripts/operator_dashboard.py __BSLASH__
  --coordinator "{coord_esc}" __BSLASH__
  --out .local/operator-dashboard.html &amp;&amp; open .local/operator-dashboard.html</pre>
    <p>Generates this HTML file and opens it. <strong>Bookmark it.</strong><br>
    <span style="color:var(--muted);font-size:10px;">Already generated? Just run: <code>open .local/operator-dashboard.html</code></span></p>
  </div>

  <div class="step" style="border-color:var(--accent);">
    <h4>Step 3 — Share the QR code below</h4>
    <p>Anyone on the same WiFi who scans the QR (or opens the share link) sees a one-command join page.
    They paste it into terminal and appear in <strong>Live Swarm</strong> instantly. That's it.</p>
  </div>

  <div class="step" style="border-color:var(--moon);">
    <h4>Step 4 — <span style="color:var(--moon);">Optional:</span> Stand up the iOS orchestrator</h4>
    <p style="color:var(--muted);font-size:11px;margin-bottom:10px;line-height:1.5;">
      <strong>Skip this if you're only onboarding laptops + Android.</strong>
      iOS needs three extra pieces (gateway + anisette + IPA) because Apple blocks the
      native Python stack that Android gets via Termux. Run each in a separate terminal.
    </p>

    <p style="color:var(--moon);font-size:11px;margin-bottom:4px;"><strong>4a. iOS gateway</strong> — bridges iPhone HTTP/JSON drafts to the coordinator</p>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre>cd ~/Projects/bloombee-ios-gateway
PYTHONPATH=. /opt/homebrew/bin/python3 gateway/server.py __BSLASH__
  --host 0.0.0.0 --port 8432</pre>
    <p style="color:var(--muted);font-size:10px;margin-top:4px;">
      ℹ️ Gateway is idempotent — re-running while it's already up prints
      <code>ALREADY RUNNING</code> and exits 0. <code>PYTHONPATH=.</code> required so
      <code>gateway.bloombee_adapter</code> imports cleanly.
    </p>

    <p style="color:var(--moon);font-size:11px;margin-bottom:4px;margin-top:10px;"><strong>4b. Anisette signing</strong> — lets iPhones install the IPA without Apple's $99/yr fee.
      Self-hosted on this Mac on the same LAN — <strong>no VPS needed</strong> unless donors are on a different network.</p>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre>cd ~/Projects/bloombee-ios-gateway/anisette
docker build -t bloombee-anisette .
docker rm -f anisette 2>/dev/null || true
docker run -d --name anisette --restart unless-stopped __BSLASH__
  -p 6969:6969 bloombee-anisette
# Tell donors to enter http://YOUR_LAN_IP:6969 in SideStore settings</pre>
    <p style="color:var(--muted);font-size:10px;margin-top:4px;">⚠️ The included <code>Dockerfile</code> is a placeholder — pin the real upstream image from SideStore/SideStore before donor rollout.</p>

    <p style="color:var(--moon);font-size:11px;margin-bottom:4px;margin-top:10px;"><strong>4c. Build + pin the IPA</strong> — one-time per release on a Mac with Xcode 16+</p>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre>cd ~/Projects/bloombee-ios-gateway
./scripts/build_ipa.sh
# → build/BloomBee.ipa, host on any static file server or pinned link</pre>

    <p style="color:var(--muted);font-size:11px;margin-top:10px;">
      Once all three are running, donors can install the IPA and iPhones will appear in the
      <strong>Live Swarm</strong> tab with a 📱 badge.
    </p>
  </div>
</div>

<div class="card wide">
  <h3>🔗 Join the Swarm</h3>
  <div class="qr-section">
    <div id="qr-container"><div class="loading">Loading QR from coordinator...</div></div>
    <div id="share-link-box" style="margin-top:12px;padding:10px;background:rgba(7,17,31,.6);border:1px solid var(--line);border-radius:10px;">
      <p style="color:var(--muted);font-size:11px;margin-bottom:6px;">📋 <strong>Or share this link</strong> — AirDrop, paste, or type:</p>
      <div style="display:flex;gap:6px;align-items:center;">
        <input id="share-link-text" type="text" readonly
          style="flex:1;background:#050e1a;border:1px solid var(--line);color:var(--text);padding:6px 8px;border-radius:6px;font-size:11px;font-family:ui-monospace,monospace;"
          value="Loading...">
        <button id="copy-link-btn" class="copy-btn" onclick="copyShareLink()" style="float:none;white-space:nowrap;">📋 Copy</button>
      </div>
      <p style="color:var(--muted);font-size:10px;margin-top:4px;">Expires: <span id="offer-expires">?</span></p>
    </div>
    <p class="manual-toggle" onclick="showManualQR()">Coordinator offline? Enter a join URL manually →</p>
    <div id="manual-qr" style="margin-top:8px;">
      <input id="manual-join-url" type="text" placeholder="bloombee://join?coordinator=..." style="width:100%;background:#050e1a;border:1px solid var(--line);color:var(--text);padding:6px;border-radius:6px;font-size:11px;font-family:ui-monospace,monospace;">
      <button onclick="generateManualQR()" style="margin-top:6px;background:var(--accent);color:var(--bg);border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:600;">Generate QR</button>
    </div>
  </div>

  <div class="info-box" style="margin-top:14px;">
    <strong>📱 What happens when someone scans the QR?</strong><br>
    The QR opens <code id="coord-landing-url">COORDINATOR_URL</code> in the browser — a landing page
    that shows a <strong>single copy-paste command</strong>. Run that command in any terminal
    and the device is connected. No repo clone, no manual setup.<br><br>
    <strong>Laptop users:</strong> Python 3.8+ (pre-installed on macOS/Linux). Nothing else needed.<br>
    <strong>Android phones:</strong> Install <a href="https://f-droid.org/packages/com.termux/" style="color:var(--accent);">Termux from F-Droid</a> (<code>pkg install python</code>), then paste the command.<br>
    <strong>iPhone users:</strong> Different flow — see <strong>Step 3b below</strong>. iPhones install a free app via SideStore and contribute as MLX draft peers (speculative decoding).
  </div>
</div>

<div class="card">
  <h3>2. Onboard a Laptop</h3>
  <div class="step">
    <h4>The device scans the QR (or opens the share link) → copies one command → done</h4>
    <p style="color:var(--muted);">The landing page shows a single <code>curl ... | python3</code> command.
    The user pastes it into any terminal. <strong>No repo clone, no pip install, no setup.</strong>
    The bootstrap script scans hardware and starts heartbeating automatically.</p>
  </div>
</div>

<div class="card">
  <h3>📱 Onboard a Phone (3a Android · 3b iOS)</h3>
  <div class="info-box" style="margin-top:4px;">
    <strong>Why phones are different from laptops:</strong> Phones can't run the same <code>bloombee.cli.run_server</code>
    stack that laptops do (Android blocks the native libp2p/Hivemind deps, iOS lacks Python). So both phone types
    run a <strong>lightweight local model</strong> and send draft tokens to a gateway. Laptops running the main model
    <strong>verify</strong> them — this makes inference 2-4× faster for everyone.
    <br><br>
    <strong>After install:</strong> the phone auto-registers with the coordinator (one-time), then sends a heartbeat
    every ~60s. It shows up in the <strong>Live Swarm</strong> tab with a
    <span style="background:var(--moon);color:var(--bg);padding:1px 6px;border-radius:4px;font-size:10px;">📱</span>
    badge and a <code>draft_peer</code> role. You don't need to "add" it anywhere — just deploy a model
    on the <strong>Deploy</strong> tab and drafts start flowing automatically.
    <br><br>
    Pick the right card for your device: <strong>3a</strong> for Android (Termux), <strong>3b</strong> for iOS (SideStore+IPA).
  </div>
</div>

<div class="card">
  <h3>3a. Onboard a Phone (Android)</h3>
  <div class="step">
    <h4>Pre-requisites (one-time)</h4>
    <p>Install <strong>Termux</strong> from <a href="https://f-droid.org/packages/com.termux/" style="color:var(--accent);">F-Droid</a> (NOT Play Store). Then in Termux:</p>
    <pre>pkg update &amp;&amp; pkg install python</pre>
  </div>
  <div class="step">
    <h4>Scan QR → tap link → copy command → paste in Termux</h4>
    <p style="color:var(--muted);">The landing page auto-detects phones and shows the Termux-specific instructions.
    One pasteable command. Phone heartbeats every 60s.</p>
  </div>
</div>

<div class="card">
  <h3>3b. Onboard a Phone (iOS)</h3>
  <div class="step">
    <h4>Pre-requisites (one-time, on the iPhone)</h4>
    <p>iOS 17+ on iPhone 12 or newer. Free — no Apple Developer account needed.</p>
    <ol style="color:var(--muted);line-height:1.6;font-size:12px;margin-left:20px;">
      <li>Install <strong>SideStore</strong> from <a href="https://sidestore.io" style="color:var(--accent);">sidestore.io</a></li>
      <li>Enter our anisette server URL in SideStore settings (operator provides this — see Operator Quick Start above)</li>
      <li>Tap our pinned <strong>BloomBee.ipa</strong> install link → "Install"</li>
      <li>Open the BloomBee app → enter the coordinator URL shown on the QR/share link</li>
    </ol>
    <p style="color:var(--muted);font-size:11px;margin-top:6px;">
      ⚠️ <strong>Note:</strong> iOS onboarding requires the operator to stand up the iOS gateway + anisette server first
      (see Operator Quick Start, Step 4). Until then, iPhones can't connect — but Android phones work fine with just Steps 1-3.
    </p>
  </div>
  <div class="step">
    <h4>What the iPhone sees</h4>
    <p style="color:var(--muted);">Once installed: BloomBee app shows the coordinator URL, sends an install_token on first launch,
    and starts generating drafts on each request. <strong>Weekly refresh</strong> via SideStore (5-second tap on WiFi).</p>
  </div>
</div>

<div class="card">
  <h3>4. Verify & Bootstrap <span style="font-size:11px;color:var(--muted);font-weight:400;">— optional CLI escape hatch</span></h3>
  <p style="color:var(--muted);font-size:12px;line-height:1.5;margin-bottom:12px;">
    <strong>You don't need to run these commands.</strong> Every piece of state they reveal
    is already shown live in this dashboard's tabs:
    <br>• <strong>Live Swarm</strong> tab = <code>/active</code> (connected peers, hardware, model serving status)
    <br>• <strong>Models</strong> tab = <code>/compatible</code> + best-fit planner
    <br>• <strong>Deploy</strong> tab = <code>POST /deploy</code> + per-peer <code>GET /job</code>
    <br>• <strong>Generate</strong> tab = <code>POST /infer</code> + streaming tokens
    <br>The commands below are for SSH sessions, scripting, or debugging — copy them into any terminal.
  </p>
  <div class="step">
    <h4>See who's connected (raw JSON)</h4>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre>curl -s '{coord_esc}/active?token=*&max_age_seconds=60' | python3 -m json.tool</pre>
  </div>
  <div class="step">
    <h4>Generate a server bootstrap script (advanced — Deploy tab does this)</h4>
    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
    <pre>curl -s "{coord_esc}/bootstrap.sh?model=TinyLlama/TinyLlama-1.1B-Chat-v1.0&peer=YOUR_PEER_ID" | bash</pre>
    <p style="color:var(--muted);font-size:11px;">For most users, just click <strong>Deploy → Deploy Model</strong> in the Deploy tab — peers auto-poll and self-start.</p>
  </div>
</div>

<div class="card wide">
  <h3>⚠️ What Each Step Proves (and Doesn't)</h3>
  <table>
    <thead><tr><th>Step</th><th>Proves</th><th>Does NOT prove</th></tr></thead>
    <tbody>
      <tr><td>QR code scanned</td><td>Device got the join URL</td><td>No software running, no inference</td></tr>
      <tr><td>Peer appears in /active</td><td>Device is alive, reachable, sent heartbeat</td><td>No model loaded, no layers served</td></tr>
      <tr><td>Peer scan completed</td><td>Hardware capability measured</td><td>No model loaded</td></tr>
      <tr><td>Layer plan generated</td><td>Theoretical allocation fits in memory</td><td>No actual serving, no inference</td></tr>
      <tr><td>BloomBee servers running</td><td>Model loaded, blocks served</td><td>No end-to-end generation verified</td></tr>
      <tr><td>Generation output matches</td><td>Distributed inference works</td><td>— (this is the final proof)</td></tr>
    </tbody>
  </table>
</div>
</main>

</div>

<!-- ====== LIVE SWARM TAB ====== -->
<div id="tab-live" class="tab-page" style="display:none;">

<main>
<div class="card wide">
  <h3>📡 Swarm Status</h3>
  <div class="stats">
    <div class="stat"><strong id="stat-peers">0</strong><span class="label">Peers</span></div>
    <div class="stat"><strong id="stat-online">0</strong><span class="label">Online</span></div>
  </div>
</div>

<div class="card wide">
  <h3>🖥️ Active Peers</h3>
  <div id="roster-content"><div class="loading">Connecting to coordinator...</div></div>
</div>

<div class="card">
  <h3>🗺️ Layer Plan</h3>
  <div id="plan-content"><div class="loading">...</div></div>
</div>

<div class="card wide">
  <h3>🔀 Best Route</h3>
  <div id="route-content"><div class="loading">...</div></div>
</div>

<div class="card wide" id="pipeline-card" style="display:none;">
  <h3>🔗 Inference Pipeline</h3>
  <div id="pipeline-viz"></div>
</div>

<div class="card" id="ios-peers-card" style="display:none;">
  <h3>📱 iOS Draft Peers</h3>
  <div id="ios-peers-content"><div class="loading">Checking for iOS peers...</div></div>
  <div class="info-box" style="margin-top:10px;font-size:11px;">
    <strong>How iPhones contribute:</strong> iPhones can't run BloomBee directly, so they run MLX-Swift-LM locally and send draft tokens here for verification. <strong>Main model on laptops verifies drafts</strong> → 2-4× faster inference. Each peer shows its MLX model + token throughput.
  </div>
</div>
</main>

</div>

<!-- ====== MODELS TAB ====== -->
<div id="tab-models" class="tab-page" style="display:none;">

<main>
<div class="card wide" style="border-color:var(--accent);">
  <h3>⭐ Best Model for Current Swarm</h3>
  <div id="best-model-content"><div class="loading">Loading swarm compatibility...</div></div>
</div>

<div class="card wide">
  <h3>🧠 All Compatible Models</h3>
  <div style="margin-bottom:12px;">
    <span class="badge ok" style="margin-right:6px;">● compatible</span>
    <span class="badge warn" style="margin-right:6px;">● single peer</span>
    <span class="badge fail">● insufficient</span>
  </div>
  <div id="models-content"><div class="loading">Loading...</div></div>
</div>
</main>

</div>

<!-- ====== INFERENCE TAB ====== -->
<div id="tab-infer" class="tab-page" style="display:none;">

<main>
<div class="card wide">
  <h3>🚀 Request Inference</h3>
  <div class="step" style="margin-bottom:14px;">
    <h4>1. Select model</h4>
    <select id="infer-model" style="width:100%;background:#050e1a;border:1px solid var(--line);color:var(--accent);padding:10px;border-radius:8px;font-size:13px;font-family:ui-monospace,monospace;">
      <option value="">Loading models from swarm...</option>
    </select>
    <div id="infer-model-info" style="margin-top:6px;color:var(--muted);font-size:11px;"></div>
  </div>

  <div class="step" style="margin-bottom:14px;">
    <h4>2. Enter prompt</h4>
    <textarea id="infer-prompt" rows="4" placeholder="What would you like the swarm to generate?"
      style="width:100%;background:#050e1a;border:1px solid var(--line);color:var(--text);padding:10px;border-radius:8px;font-size:13px;font-family:ui-monospace,monospace;resize:vertical;"></textarea>
  </div>

  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;">
    <button id="infer-btn" onclick="runInference()" style="background:var(--accent);color:var(--bg);border:none;padding:10px 24px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:700;">⚡ Generate</button>
    <span id="infer-status" style="color:var(--muted);font-size:12px;"></span>
  </div>

  <div class="step" id="infer-output-section" style="display:none;">
    <h4>3. Output</h4>
    <pre id="infer-output" style="min-height:60px;white-space:pre-wrap;">Waiting for generation...</pre>
    <div id="infer-stats" style="color:var(--muted);font-size:11px;margin-top:8px;"></div>
  </div>
</div>

<div class="card">
  <h3>📊 Layer Assignment</h3>
  <p style="color:var(--muted);font-size:12px;margin-bottom:10px;">How layers would be distributed across the swarm for the selected model.</p>
  <div id="infer-plan-content"><div class="loading">Select a model above to see the plan.</div></div>
</div>
</main>

</div>

<!-- ====== DEPLOY TAB ====== -->
<div id="tab-deploy" class="tab-page" style="display:none;">

<main>
<div class="card wide">
  <h3>⚡ Zero-Touch Deploy</h3>
  <p style="color:var(--muted);font-size:12px;margin-bottom:14px;">
    Select a model and deploy it across all connected peers. Peers will auto-download and start serving.
  </p>

  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <select id="deploy-model" style="flex:1;min-width:200px;background:#050e1a;border:1px solid var(--line);color:var(--text);padding:8px 12px;border-radius:8px;font-size:13px;">
      <option value="">-- Select model --</option>
    </select>
    <button id="btn-deploy" onclick="deployModel()" style="background:var(--accent);color:#050e1a;border:none;padding:8px 24px;border-radius:8px;font-weight:700;cursor:pointer;font-size:13px;">
      Deploy
    </button>
  </div>

  <div id="deploy-result" style="margin-top:16px;"></div>
</div>

<div class="card wide" id="deploy-jobs-card" style="display:none;">
  <h3>📋 Job Assignments</h3>
  <div id="deploy-jobs"></div>
</div>

<div class="card wide">
  <h3>🔍 Check Peer Job</h3>
  <div style="display:flex;gap:8px;margin-bottom:8px;">
    <input id="job-peer-input" type="text" placeholder="Peer ID (e.g. evilaptop-abc123)" style="flex:1;background:#050e1a;border:1px solid var(--line);color:var(--accent);padding:6px 10px;border-radius:6px;font-size:12px;font-family:monospace;">
    <button onclick="pollPeerJob()" style="background:var(--panel2);border:1px solid var(--line);color:var(--text);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;">Poll Job</button>
  </div>
  <pre id="job-peer-result" style="background:#050e1a;padding:10px;border-radius:8px;font-size:11px;max-height:200px;overflow:auto;"></pre>
</div>
</main>

</div>

<script>{js}</script>
</body>
</html>"""
    # Restore literal backslash-newline in shell commands (see comment above).
    # The placeholder spans the line break so we don't end up with a blank
    # line between the `\` and the continuation argument (e.g. `\` + blank
    # line + `--host` makes zsh treat `--host` as a separate command).
    return html_body.replace("__BSLASH__\n", "\\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator dashboard HTML")
    parser.add_argument(
        "--coordinator", default="http://localhost:8787",
        help="Default coordinator URL (can be changed in the dashboard)",
    )
    parser.add_argument(
        "--out", default=".local/operator-dashboard.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    html_content = _build_html(args.coordinator)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"✅ Operator dashboard written: {out_path}")
    print(f"   Open:  open {out_path}")
    print(f"   Default coordinator: {args.coordinator} (change it in the top bar)")


if __name__ == "__main__":
    main()
