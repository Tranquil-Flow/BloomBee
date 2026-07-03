#!/usr/bin/env python3
"""Fetch or redact coordinator /handoff bundles for operator dashboards.

This CLI materializes the no-execution /handoff response from join_http_server.py
into a local JSON artifact suitable for demo_dashboard.py --handoff-bundle. It
redacts tokens before writing or printing. It does not start servers, send proof
traffic, or update proof status.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

FETCH_CLAIM_BOUNDARY = "join_handoff_fetch_only_no_server_started"
FETCH_SOURCE = "join_handoff_cli"
REDACTED = "***"
_TOKEN_KEYS = {"token", "access_token", "refresh_token", "authorization", "auth"}


def _bool_param(value: bool) -> str:
    return "1" if value else "0"


def _append_query(url: str, params: list[tuple[str, str]]) -> str:
    split = urlsplit(url)
    existing = parse_qsl(split.query, keep_blank_values=True)
    query = urlencode(existing + [(key, value) for key, value in params if value is not None])
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


def build_handoff_url(
    coordinator_url: str,
    *,
    token: str,
    model: str = "auto",
    selector_mode: str = "planning",
    max_age_seconds: int = 30,
    include_launch_commands: bool = True,
    include_launch_readiness: bool = True,
    request_count: int = 3,
    now: int | None = None,
    base_port: int | None = None,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Build a coordinator /handoff URL without performing network I/O."""
    base = coordinator_url.rstrip("/") + "/handoff"
    params: list[tuple[str, str]] = [
        ("token", token),
        ("model", model),
        ("selector_mode", selector_mode),
        ("max_age_seconds", str(int(max_age_seconds))),
        ("include_launch_commands", _bool_param(include_launch_commands)),
        ("include_launch_readiness", _bool_param(include_launch_readiness)),
        ("request_count", str(int(request_count))),
    ]
    if now is not None:
        params.append(("now", str(int(now))))
    if base_port is not None:
        params.append(("base_port", str(int(base_port))))
    if prompt is not None:
        params.append(("prompt", prompt))
    if max_new_tokens is not None:
        params.append(("max_new_tokens", str(int(max_new_tokens))))
    return _append_query(base, params)


def _redact_url(value: str) -> str:
    try:
        split = urlsplit(value)
    except ValueError:
        return value
    if not split.query:
        return value
    changed = False
    redacted_query: list[tuple[str, str]] = []
    for key, raw in parse_qsl(split.query, keep_blank_values=True):
        if key.lower() in _TOKEN_KEYS or "token" in key.lower():
            redacted_query.append((key, REDACTED))
            changed = True
        else:
            redacted_query.append((key, raw))
    if not changed:
        return value
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(redacted_query), split.fragment))


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    if key and (key.lower() in _TOKEN_KEYS or "token" in key.lower()):
        return REDACTED
    if isinstance(value, dict):
        return {str(item_key): _redact_value(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_url(value)
    return value


def redact_handoff_bundle(bundle: dict[str, Any], *, fetched_url: str | None = None) -> dict[str, Any]:
    """Return a dashboard-ready handoff bundle with all token fields redacted."""
    redacted = _redact_value(deepcopy(bundle))
    if not isinstance(redacted, dict):
        raise ValueError("handoff bundle must be a JSON object")
    redacted["handoff_fetch_claim_boundary"] = FETCH_CLAIM_BOUNDARY
    redacted["handoff_fetch_source"] = FETCH_SOURCE
    redacted["handoff_fetch_redacted"] = True
    redacted["inference_proven"] = False
    redacted["can_update_proof_status"] = False
    if fetched_url:
        redacted["handoff_fetch_url"] = _redact_url(fetched_url)
    return redacted


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("handoff JSON must be an object")
    return payload


def _fetch_json(url: str, *, timeout: float, urlopen_fn: Callable[..., Any] = urlopen) -> dict[str, Any]:
    with urlopen_fn(url, timeout=timeout) as response:  # noqa: S310 - operator-provided coordinator URL
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("coordinator /handoff response must be a JSON object")
    return payload


def fetch_handoff_bundle(
    coordinator_url: str,
    *,
    token: str,
    model: str = "auto",
    selector_mode: str = "planning",
    max_age_seconds: int = 30,
    include_launch_commands: bool = True,
    include_launch_readiness: bool = True,
    request_count: int = 3,
    now: int | None = None,
    base_port: int | None = None,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    timeout: float = 10.0,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    url = build_handoff_url(
        coordinator_url,
        token=token,
        model=model,
        selector_mode=selector_mode,
        max_age_seconds=max_age_seconds,
        include_launch_commands=include_launch_commands,
        include_launch_readiness=include_launch_readiness,
        request_count=request_count,
        now=now,
        base_port=base_port,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return redact_handoff_bundle(_fetch_json(url, timeout=timeout, urlopen_fn=urlopen_fn), fetched_url=url)


def write_handoff_artifact(bundle: dict[str, Any], out: str | Path) -> Path:
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", default=None, help="Redact an existing raw /handoff JSON artifact instead of fetching")
    source.add_argument("--coordinator-url", default=None, help="Coordinator base URL, e.g. http://127.0.0.1:8787")
    parser.add_argument("--token", default=None, help="Join token for live coordinator fetch; never written unredacted")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--selector-mode", default="planning")
    parser.add_argument("--max-age-seconds", type=int, default=30)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--now", type=int, default=None)
    parser.add_argument("--base-port", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-launch-commands", action="store_true", help="Set include_launch_commands=0 on live fetch")
    parser.add_argument("--no-launch-readiness", action="store_true", help="Set include_launch_readiness=0 on live fetch")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--fetched-url", default=None, help="Original URL to record/redact when using --input-json")
    parser.add_argument("--out", default=None, help="Write redacted dashboard-ready handoff JSON to this path")
    args = parser.parse_args(argv)

    if args.input_json:
        bundle = redact_handoff_bundle(_read_json(args.input_json), fetched_url=args.fetched_url)
    else:
        if not args.token:
            parser.error("--token is required with --coordinator-url")
        bundle = fetch_handoff_bundle(
            args.coordinator_url,
            token=args.token,
            model=args.model,
            selector_mode=args.selector_mode,
            max_age_seconds=args.max_age_seconds,
            include_launch_commands=not args.no_launch_commands,
            include_launch_readiness=not args.no_launch_readiness,
            request_count=args.request_count,
            now=args.now,
            base_port=args.base_port,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )

    if args.out:
        write_handoff_artifact(bundle, args.out)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
