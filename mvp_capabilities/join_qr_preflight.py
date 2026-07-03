#!/usr/bin/env python3
"""Fail-closed dependency preflight for true QR join-card scanner proof.

The current join card is a dependency-free visual SVG with embedded URL metadata.
This preflight does not generate or decode a QR image. It only checks whether the
environment has enough encoder+decoder support to run a future scanner-interop
proof. Until an actual generated artifact is decoded back to the exact join URL,
scanner interoperability remains unproven.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Mapping

CLAIM_BOUNDARY = "qr_scanner_preflight_only_no_scanner_proof"
BLOCKED_STATUS = "scanner_interop_blocked_missing_dependencies"
READY_STATUS = "scanner_interop_preflight_ready_no_scan_yet"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _current_availability() -> dict[str, bool]:
    return {name: _module_available(name) for name in ("qrcode", "PIL", "cv2", "pyzbar", "segno")}


def _has_qrcode_pil(availability: Mapping[str, bool]) -> bool:
    return bool(availability.get("qrcode") and availability.get("PIL"))


def _has_segno(availability: Mapping[str, bool]) -> bool:
    return bool(availability.get("segno"))


def check_qr_scanner_readiness(availability: Mapping[str, bool] | None = None) -> dict[str, object]:
    """Return a conservative QR scanner-proof readiness report.

    Encoder is satisfied by either qrcode+PIL or segno. Decoder is satisfied by
    either OpenCV (`cv2`) or pyzbar. Even when both sides exist, the report still
    refuses to mark scanner interop proven; a future proof must generate an
    artifact, decode it, and exact-match the URL.
    """
    availability = dict(_current_availability() if availability is None else availability)
    encoder_ready = _has_qrcode_pil(availability) or _has_segno(availability)
    decoder_ready = bool(availability.get("cv2") or availability.get("pyzbar"))
    missing_encoder_options: list[str] = []
    missing_decoder_options: list[str] = []
    if not _has_qrcode_pil(availability):
        missing_encoder_options.append("qrcode+PIL")
    if not _has_segno(availability):
        missing_encoder_options.append("segno")
    if not availability.get("cv2"):
        missing_decoder_options.append("cv2")
    if not availability.get("pyzbar"):
        missing_decoder_options.append("pyzbar")

    ready = encoder_ready and decoder_ready
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scanner_status": READY_STATUS if ready else BLOCKED_STATUS,
        "ready_for_scanner_proof": ready,
        "can_replace_visual_grid": False,
        "available_modules": availability,
        "encoder_ready": encoder_ready,
        "decoder_ready": decoder_ready,
        "missing_encoder_options": [] if encoder_ready else missing_encoder_options,
        "missing_decoder_options": [] if decoder_ready else missing_decoder_options,
        "inference_proven": False,
        "scanner_interop_proven": False,
        "next_step": "generate a true QR artifact, decode it with an installed scanner library, and compare the decoded URL exactly"
        if ready
        else "install one QR encoder option and one decoder option, then run a generated-artifact decode proof",
    }


def render_markdown(report: dict[str, object]) -> str:
    modules = report.get("available_modules") or {}
    module_lines = [f"- {name}: {available}" for name, available in sorted(dict(modules).items())]
    return "\n".join(
        [
            "# BloomBee join-card QR scanner preflight",
            "",
            f"**Scanner status:** `{report['scanner_status']}`",
            f"**Ready for scanner proof:** `{report['ready_for_scanner_proof']}`",
            f"**Claim boundary:** `{report['claim_boundary']}`",
            "",
            "Available modules:",
            *module_lines,
            "",
            f"Next step: {report['next_step']}",
            "",
            "This does not prove QR scanner interoperability and does not replace the visual-grid join card.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown")
    args = parser.parse_args(argv)
    report = check_qr_scanner_readiness()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
