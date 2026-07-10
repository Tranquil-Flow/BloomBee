#!/usr/bin/env python3
"""Generate the HTML landing page for the coordinator root endpoint.

This is the page that opens when someone scans the QR code or taps the
share link. It shows a simple one-command bootstrap flow.

Usage:
  python scripts/coordinator_landing.py --coordinator http://192.168.1.100:8787 --out landing.html

The coordinator (join_http_server.py) should serve this at GET /.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Join BloomBee Swarm</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f; --panel: #0d1b2f;
      --line: #2a4b73; --text: #e9f3ff; --muted: #95acc8;
      --ok: #58d68d; --fail: #ff6b6b; --moon: #b9ccff; --accent: #7dd3fc;
    }
    * { box-sizing: border-box; margin: 0; }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 20% -10%, #233e73 0, transparent 34%), var(--bg);
      color: var(--text); min-height: 100vh;
      display: flex; flex-direction: column; align-items: center;
      padding: 40px 20px;
    }
    .card {
      background: linear-gradient(180deg, rgba(18,39,68,.96), rgba(13,27,47,.96));
      border: 1px solid var(--line); border-radius: 18px;
      padding: 32px; max-width: 580px; width: 100%;
      box-shadow: 0 14px 40px rgba(0,0,0,.28);
      text-align: center;
    }
    h1 { font-size: 28px; margin-bottom: 6px; letter-spacing: -0.03em; }
    h2 { color: var(--moon); font-size: 15px; font-weight: 400; margin-bottom: 24px; }
    .step {
      background: rgba(7,17,31,.6); border: 1px solid var(--line);
      border-radius: 12px; padding: 16px; margin: 16px 0;
      text-align: left;
    }
    .step h3 { color: var(--accent); margin-bottom: 8px; font-size: 13px; }
    .step p { color: var(--muted); font-size: 12px; line-height: 1.5; }
    pre {
      background: #050e1a; color: #d7e5ff; border: 1px solid var(--line);
      border-radius: 10px; padding: 14px; overflow-x: auto;
      font-size: 13px; line-height: 1.6; margin: 8px 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      text-align: left; word-break: break-all;
    }
    .copy-btn {
      display: inline-block; background: var(--accent); color: var(--bg);
      border: none; padding: 10px 24px; border-radius: 10px;
      cursor: pointer; font-size: 14px; font-weight: 700;
      margin-top: 8px; transition: all .15s;
    }
    .copy-btn:hover { opacity: .85; transform: scale(1.03); }
    .copy-btn.copied { background: var(--ok); }
    .platform-tabs { display: flex; gap: 6px; margin: 12px 0; justify-content: center; }
    .platform-tabs button {
      background: rgba(185,204,255,.1); border: 1px solid var(--line);
      color: var(--muted); padding: 6px 14px; border-radius: 8px;
      cursor: pointer; font-size: 12px; font-weight: 600;
    }
    .platform-tabs button.active {
      background: rgba(125,211,252,.18); color: var(--accent); border-color: var(--accent);
    }
    .platform-content { display: none; }
    .platform-content.active { display: block; }
    .note {
      color: var(--muted); font-size: 11px; margin-top: 12px;
      padding: 8px 12px; background: rgba(247,201,72,.06);
      border: 1px solid rgba(247,201,72,.15); border-radius: 8px;
    }
    .status { font-size: 28px; margin: 12px 0; }
    .connected { color: var(--ok); }
    .error { color: var(--fail); }
  </style>
</head>
<body>

<div class="card">
  <h1>🌙 Join the BloomBee Swarm</h1>
  <h2 id="coordinator-display">Coordinator: __COORDINATOR__</h2>

  <div class="step">
    <h3>📋 One command to join</h3>
    <p>Open a terminal and run:</p>
    <pre id="join-command">Loading join offer from coordinator...</pre>
    <button class="copy-btn" onclick="copyCommand()">📋 Copy to clipboard</button>
  </div>

  <div class="platform-tabs">
    <button class="active" onclick="showPlatform('any')">💻 Any Device</button>
    <button onclick="showPlatform('phone')">📱 Android Phone</button>
  </div>

  <div class="platform-content active" id="platform-any">
    <p style="color:var(--muted);font-size:12px;line-height:1.5;">
      The command above downloads a self-contained Python script and runs it.
      <strong>Requirements:</strong> Python 3.8+ (pre-installed on macOS and most Linux).
      No pip installs, no repo clones, no setup needed.
    </p>
  </div>

  <div class="platform-content" id="platform-phone">
    <div class="step">
      <h3>1. Install Termux</h3>
      <p>Get <strong>Termux</strong> from <a href="https://f-droid.org/packages/com.termux/" style="color:var(--accent);">F-Droid</a> (NOT Play Store). Open it and run:</p>
      <pre>pkg update && pkg install python</pre>
    </div>
    <div class="step">
      <h3>2. Run the join command</h3>
      <p>Copy this command and paste it into Termux:</p>
      <pre id="join-command-phone">Loading...</pre>
    </div>
  </div>

  <div class="note">
    ⚠️ <strong>What this does:</strong> Scans your device's hardware, registers it
    with the swarm coordinator, and keeps a heartbeat alive. It does NOT run any
    models or access your files. Press Ctrl+C to disconnect at any time.
  </div>

  <div id="join-status"></div>
</div>

<script>
const COORDINATOR = "__COORDINATOR__";
const BOOTSTRAP_URL = COORDINATOR + "/bootstrap.py";

let currentJoinUrl = "";

// Fetch live join offer and build the command
async function loadJoinOffer() {
  try {
    const resp = await fetch(COORDINATOR + "/offer");
    const data = await resp.json();
    currentJoinUrl = data.join_url;
    const cmd = `curl -s ${BOOTSTRAP_URL} | python3 - --join-url "${currentJoinUrl}" --loop --interval 15 --auto-serve`;
    const phoneCmd = `curl -s ${BOOTSTRAP_URL} | python3 - --join-url "${currentJoinUrl}" --loop --interval 60 --auto-serve`;
    document.getElementById('join-command').textContent = cmd;
    document.getElementById('join-command-phone').textContent = phoneCmd;
    document.getElementById('coordinator-display').textContent = 'Coordinator: ' + COORDINATOR;
  } catch(e) {
    document.getElementById('join-command').textContent = 'Error: cannot reach coordinator at ' + COORDINATOR;
    document.getElementById('coordinator-display').textContent = '⚠️ Coordinator unreachable';
  }
}

function copyCommand() {
  const cmd = document.getElementById('join-command').textContent;
  if (cmd.startsWith('curl')) {
    navigator.clipboard.writeText(cmd).then(() => {
      const btn = document.querySelector('.copy-btn');
      btn.textContent = '✅ Copied! Paste in terminal now.';
      btn.classList.add('copied');
    });
  }
}

function showPlatform(name) {
  document.querySelectorAll('.platform-tabs button').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.platform-content').forEach(c => c.classList.remove('active'));
  const idx = name === 'any' ? 0 : 1;
  document.querySelectorAll('.platform-tabs button')[idx].classList.add('active');
  document.getElementById('platform-' + name).classList.add('active');
}

// Auto-detect phone
if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) {
  showPlatform('phone');
}

document.addEventListener('DOMContentLoaded', loadJoinOffer);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coordinator landing page")
    parser.add_argument("--coordinator", required=True, help="Coordinator URL (e.g. http://192.168.1.100:8787)")
    parser.add_argument("--join-url", required=True, help="Join URL with token (e.g. bloombee://join?...)")
    parser.add_argument("--out", required=True, help="Output HTML file path")
    args = parser.parse_args()

    coord_esc = html.escape(args.coordinator, quote=True)
    join_esc = html.escape(args.join_url, quote=True)

    page = HTML.replace("__COORDINATOR__", coord_esc)
    page = page.replace("__JOIN_URL__", join_esc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"✅ Landing page: {out}")
    print(f"   Coordinator: {args.coordinator}")
    print(f"   Bootstrap URL: {args.coordinator}/bootstrap.py")


if __name__ == "__main__":
    main()
