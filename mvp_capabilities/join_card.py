#!/usr/bin/env python3
"""Render a dependency-free visual join card for BloomBee join URLs.

This intentionally does **not** claim QR scanner interoperability. The SVG
contains the exact join URL as text/data metadata plus a deterministic visual
code grid derived from the URL. A future slice can swap the grid for a true QR
encoder once a QR dependency is added and scanner compatibility is tested.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Iterable

CLAIM_BOUNDARY = "join_card_visual_only_no_inference_proof"
SCANNER_STATUS = "scanner_interop_unproven"


def _bit_stream(seed: bytes) -> Iterable[int]:
    counter = 0
    while True:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for byte in digest:
            for bit in range(7, -1, -1):
                yield (byte >> bit) & 1
        counter += 1


def _finder_pattern(x: int, y: int, size: int = 7) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for yy in range(size):
        for xx in range(size):
            border = xx in {0, size - 1} or yy in {0, size - 1}
            center = 2 <= xx <= 4 and 2 <= yy <= 4
            if border or center:
                cells.add((x + xx, y + yy))
    return cells


def visual_code_cells(join_url: str, *, modules: int = 29) -> set[tuple[int, int]]:
    """Return deterministic dark cells for the URL visual code grid.

    This is intentionally not a QR implementation. Finder-like corners make the
    card visually familiar, while SVG metadata/text carry the actual URL.
    """
    if modules < 21:
        raise ValueError("modules must be at least 21")
    cells: set[tuple[int, int]] = set()
    cells |= _finder_pattern(0, 0)
    cells |= _finder_pattern(modules - 7, 0)
    cells |= _finder_pattern(0, modules - 7)
    reserved = set(cells)
    reserved |= {(x, y) for x in range(8) for y in range(8)}
    reserved |= {(x, y) for x in range(modules - 8, modules) for y in range(8)}
    reserved |= {(x, y) for x in range(8) for y in range(modules - 8, modules)}

    bits = _bit_stream(join_url.encode("utf-8"))
    for y in range(modules):
        for x in range(modules):
            if (x, y) in reserved:
                continue
            if next(bits):
                cells.add((x, y))
    return cells


def _svg_rects(cells: set[tuple[int, int]], *, offset_x: int, offset_y: int, module_px: int) -> str:
    rects = []
    for x, y in sorted(cells, key=lambda item: (item[1], item[0])):
        rects.append(
            f'<rect x="{offset_x + x * module_px}" y="{offset_y + y * module_px}" '
            f'width="{module_px}" height="{module_px}" rx="1" />'
        )
    return "\n    ".join(rects)


def render_join_card_svg(join_url: str, *, title: str = "BloomBee join", expires_at: int | None = None) -> str:
    escaped_url = html.escape(join_url, quote=True)
    escaped_title = html.escape(title, quote=True)
    modules = 29
    module_px = 10
    grid_px = modules * module_px
    offset_x = 34
    offset_y = 96
    cells = visual_code_cells(join_url, modules=modules)
    expires_text = "never" if expires_at is None else str(int(expires_at))
    rects = _svg_rects(cells, offset_x=offset_x, offset_y=offset_y, module_px=module_px)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="460" viewBox="0 0 720 460" role="img" aria-label="{escaped_title}" data-join-url="{escaped_url}" data-claim-boundary="{CLAIM_BOUNDARY}" data-scanner-status="{SCANNER_STATUS}">
  <title>{escaped_title}</title>
  <desc>Visual BloomBee join card. Exact join URL is embedded as text and data metadata. Scanner interoperability is unproven.</desc>
  <rect x="0" y="0" width="720" height="460" fill="#07111f" />
  <rect x="18" y="18" width="684" height="424" rx="24" fill="#0d1b2e" stroke="#7dd3fc" stroke-width="2" />
  <text x="36" y="58" fill="#e0f2fe" font-size="28" font-family="ui-sans-serif, system-ui, -apple-system">{escaped_title}</text>
  <text x="36" y="82" fill="#bae6fd" font-size="13" font-family="ui-monospace, SFMono-Regular, Menlo">{CLAIM_BOUNDARY} · {SCANNER_STATUS}</text>
  <rect x="{offset_x - 10}" y="{offset_y - 10}" width="{grid_px + 20}" height="{grid_px + 20}" rx="12" fill="#e0f2fe" />
  <g fill="#07111f">
    {rects}
  </g>
  <text x="360" y="118" fill="#e0f2fe" font-size="18" font-family="ui-sans-serif, system-ui, -apple-system">Join URL</text>
  <foreignObject x="360" y="134" width="320" height="150">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font: 13px ui-monospace, SFMono-Regular, Menlo; color: #bae6fd; overflow-wrap: anywhere; line-height: 1.35;">{escaped_url}</div>
  </foreignObject>
  <text x="360" y="312" fill="#c4b5fd" font-size="14" font-family="ui-monospace, SFMono-Regular, Menlo">expires_at={html.escape(expires_text)}</text>
  <text x="360" y="338" fill="#fef3c7" font-size="14" font-family="ui-sans-serif, system-ui, -apple-system">Open with join_client.py or copy URL text.</text>
  <text x="360" y="364" fill="#fca5a5" font-size="13" font-family="ui-sans-serif, system-ui, -apple-system">Visual grid is not yet proven QR-scannable.</text>
</svg>
'''


def write_join_card(path: str | Path, join_url: str, *, title: str = "BloomBee join", expires_at: int | None = None) -> dict[str, str | int | None]:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_join_card_svg(join_url, title=title, expires_at=expires_at), encoding="utf-8")
    return {
        "out": str(out),
        "join_url": join_url,
        "expires_at": expires_at,
        "claim_boundary": CLAIM_BOUNDARY,
        "scanner_status": SCANNER_STATUS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-url", required=True)
    parser.add_argument("--title", default="BloomBee join")
    parser.add_argument("--expires-at", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = write_join_card(args.out, args.join_url, title=args.title, expires_at=args.expires_at)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
