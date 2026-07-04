#!/usr/bin/env python3
"""Generate and locally decode a true QR join artifact.

This is a scanner-library proof for the join-flow QR slice: it generates a real
QR PNG, decodes that artifact with an installed QR decoder, and checks that the
decoded value exactly matches the original join URL. It still does not prove a
physical phone camera can scan the card and it never proves inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from mvp_capabilities.join_qr_preflight import BLOCKED_STATUS, check_qr_scanner_readiness
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mvp_capabilities.join_qr_preflight import BLOCKED_STATUS, check_qr_scanner_readiness

CLAIM_BOUNDARY = "qr_artifact_exact_decode_proof_no_physical_scanner_no_inference"
PROVEN_STATUS = "local_qr_exact_decode_proven"
FAILED_STATUS = "local_qr_exact_decode_failed"

Encoder = Callable[[str, Path], None]
Decoder = Callable[[Path], str]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_join_url(url: str) -> str:
    """Redact token-like query parameters while preserving operator-readable shape."""
    parts = urlsplit(url)
    redacted_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"token", "access_token", "refresh_token", "authorization", "auth"}:
            redacted_pairs.append((key, "***"))
        else:
            redacted_pairs.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted_pairs), parts.fragment))


def _encode_with_qrcode(join_url: str, out: Path) -> None:
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(join_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    image.save(out)


def _encode_with_segno(join_url: str, out: Path) -> None:
    import segno

    qr = segno.make(join_url, error="m")
    qr.save(str(out), scale=10, border=4)


def _decode_with_cv2(out: Path) -> str:
    import cv2

    image = cv2.imread(str(out))
    if image is None:
        raise RuntimeError(f"cv2 could not read QR artifact: {out}")
    decoded, _points, _straight_qrcode = cv2.QRCodeDetector().detectAndDecode(image)
    return decoded or ""


def _decode_with_pyzbar(out: Path) -> str:
    from PIL import Image
    from pyzbar.pyzbar import decode

    results = decode(Image.open(out))
    if not results:
        return ""
    return results[0].data.decode("utf-8")


def _select_encoder(availability: Mapping[str, bool]) -> tuple[str, Encoder] | None:
    if availability.get("qrcode") and availability.get("PIL"):
        return "qrcode+PIL", _encode_with_qrcode
    if availability.get("segno"):
        return "segno", _encode_with_segno
    return None


def _select_decoder(availability: Mapping[str, bool]) -> tuple[str, Decoder] | None:
    if availability.get("cv2"):
        return "cv2", _decode_with_cv2
    if availability.get("pyzbar") and availability.get("PIL"):
        return "pyzbar+PIL", _decode_with_pyzbar
    return None


def _base_report(
    *,
    join_url: str,
    out: Path,
    status: str,
    preflight: dict[str, object],
    encoder_name: str | None,
    decoder_name: str | None,
) -> dict[str, object]:
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "scanner_status": status,
        "generated_artifact": str(out),
        "expected_url_redacted": redact_join_url(join_url),
        "expected_url_sha256": _sha256_text(join_url),
        "decoded_url_redacted": None,
        "decoded_url_sha256": None,
        "exact_match": False,
        "local_exact_decode_proven": False,
        "physical_scanner_interop_proven": False,
        "scanner_interop_proven": False,
        "inference_proven": False,
        "can_update_proof_status": False,
        "can_replace_visual_grid": False,
        "encoder": encoder_name,
        "decoder": decoder_name,
        "available_modules": preflight.get("available_modules", {}),
        "preflight": preflight,
    }


def run_qr_artifact_proof(
    join_url: str,
    out_path: str | Path,
    *,
    encoder: Encoder | None = None,
    decoder: Decoder | None = None,
    availability: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Generate a QR artifact, decode it, and report whether the exact URL round-tripped.

    Optional encoder/decoder injection keeps unit tests dependency-free. Production
    CLI usage selects installed libraries from ``join_qr_preflight.py``.
    """
    out = Path(out_path).expanduser()
    preflight = check_qr_scanner_readiness(availability=availability)
    available_modules = dict(preflight.get("available_modules") or {})

    encoder_name = "injected" if encoder else None
    decoder_name = "injected" if decoder else None
    if encoder is None:
        selected_encoder = _select_encoder(available_modules)
        if selected_encoder:
            encoder_name, encoder = selected_encoder
    if decoder is None:
        selected_decoder = _select_decoder(available_modules)
        if selected_decoder:
            decoder_name, decoder = selected_decoder

    if encoder is None or decoder is None:
        report = _base_report(
            join_url=join_url,
            out=out,
            status=BLOCKED_STATUS,
            preflight=preflight,
            encoder_name=encoder_name,
            decoder_name=decoder_name,
        )
        report.update(
            {
                "ready_for_scanner_proof": False,
                "error": "missing QR encoder or decoder dependency",
                "next_step": preflight.get("next_step"),
            }
        )
        return report

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoder(join_url, out)
        decoded = decoder(out)
    except Exception as exc:  # pragma: no cover - exercised via CLI/tooling, not deterministic in CI
        report = _base_report(
            join_url=join_url,
            out=out,
            status=FAILED_STATUS,
            preflight=preflight,
            encoder_name=encoder_name,
            decoder_name=decoder_name,
        )
        report.update(
            {
                "ready_for_scanner_proof": bool(preflight.get("ready_for_scanner_proof")),
                "error": f"{type(exc).__name__}: {exc}",
                "next_step": "fix QR encode/decode dependency/runtime error, then rerun this exact-artifact proof",
            }
        )
        return report

    exact_match = decoded == join_url
    report = _base_report(
        join_url=join_url,
        out=out,
        status=PROVEN_STATUS if exact_match else FAILED_STATUS,
        preflight=preflight,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
    )
    report.update(
        {
            "ready_for_scanner_proof": bool(preflight.get("ready_for_scanner_proof")) or exact_match,
            "decoded_url_redacted": redact_join_url(decoded),
            "decoded_url_sha256": _sha256_text(decoded),
            "exact_match": exact_match,
            "local_exact_decode_proven": exact_match,
            "can_replace_visual_grid": False,
            "next_step": "scan the generated artifact with physical devices, then run a repeated-heartbeat fresh-laptop join loop"
            if exact_match
            else "decoder returned a non-matching value; inspect artifact/error correction settings before sharing it",
        }
    )
    return report


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# BloomBee join-card QR artifact proof",
        "",
        f"**Scanner status:** `{report['scanner_status']}`",
        f"**Exact local decode:** `{report['local_exact_decode_proven']}`",
        f"**Physical scanner interop proven:** `{report['physical_scanner_interop_proven']}`",
        f"**Claim boundary:** `{report['claim_boundary']}`",
        "",
        f"Artifact: `{report['generated_artifact']}`",
        f"Encoder: `{report.get('encoder')}`",
        f"Decoder: `{report.get('decoder')}`",
        f"Expected URL: `{report['expected_url_redacted']}`",
        f"Decoded URL: `{report.get('decoded_url_redacted')}`",
        f"Exact match: `{report['exact_match']}`",
    ]
    if report.get("error"):
        lines.append(f"Error: `{report['error']}`")
    lines.extend(
        [
            "",
            f"Next step: {report['next_step']}",
            "",
            "This proves only a local encoder/decoder round-trip. It does not prove physical phone scanning or inference.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-url", required=True)
    parser.add_argument("--out", required=True, help="PNG artifact path to write")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown")
    args = parser.parse_args(argv)

    report = run_qr_artifact_proof(args.join_url, args.out)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0 if report.get("local_exact_decode_proven") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
