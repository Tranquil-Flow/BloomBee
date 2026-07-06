from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

OPT_IN_FLAG = "BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING"
PROOF_GATE = "continuous_batching"


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _outputs_from_batches(batches: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    out: list[Any] = []
    for batch in batches:
        values = batch.get(key)
        if isinstance(values, list):
            out.extend(values)
    return out


def _active_outputs_for(report: Mapping[str, Any], row_idx: int, key: str) -> list[Any]:
    out: list[Any] = []
    for batch in _list_or_empty(report.get("tick_batches")):
        if not isinstance(batch, Mapping):
            continue
        request_ids = batch.get("request_ids")
        values = batch.get(key)
        if not isinstance(request_ids, list) or not isinstance(values, list) or row_idx >= len(values):
            continue
        active_mask = batch.get("active_mask")
        if isinstance(active_mask, list) and row_idx < len(active_mask) and not bool(active_mask[row_idx]):
            continue
        out.append(values[row_idx])
    return out


def _first_active_tick(report: Mapping[str, Any], row_idx: int) -> int:
    for batch in _list_or_empty(report.get("tick_batches")):
        if not isinstance(batch, Mapping):
            continue
        tick = batch.get("tick")
        request_ids = batch.get("request_ids")
        if not isinstance(tick, int) or isinstance(tick, bool) or not isinstance(request_ids, list):
            continue
        if row_idx >= len(request_ids):
            continue
        active_mask = batch.get("active_mask")
        if isinstance(active_mask, list) and row_idx < len(active_mask) and not bool(active_mask[row_idx]):
            continue
        return int(tick)
    return 0


def _numeric_comparison(baseline_values: Sequence[Any], continuous_values: Sequence[Any]) -> dict[str, Any] | None:
    if len(baseline_values) != len(continuous_values) or not baseline_values:
        return None
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    argmax_match = True
    for base_row, cont_row in zip(baseline_values, continuous_values):
        if not isinstance(base_row, list) or not isinstance(cont_row, list) or len(base_row) != len(cont_row) or not base_row:
            return None
        base_f = [float(value) for value in base_row]
        cont_f = [float(value) for value in cont_row]
        if max(range(len(base_f)), key=base_f.__getitem__) != max(range(len(cont_f)), key=cont_f.__getitem__):
            argmax_match = False
        for base, cont in zip(base_f, cont_f):
            delta = abs(base - cont)
            max_abs = max(max_abs, delta)
            total_abs += delta
            count += 1
    if count == 0:
        return None
    return {
        "max_abs_diff": round(max_abs, 6),
        "mean_abs_diff": round(total_abs / count, 6),
        "argmax_token_id_match": argmax_match,
        "top1_token_id_match": argmax_match,
    }


def build_continuous_verifier_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build continuous-batching verifier input from a strict CBKV raw artifact."""
    raw_live_report = raw.get("live_continuous_report")
    live_report: Mapping[str, Any] = raw_live_report if isinstance(raw_live_report, Mapping) else {}
    requests: list[dict[str, Any]] = []
    for index, item in enumerate(_list_or_empty(raw.get("baseline"))):
        if not isinstance(item, Mapping):
            continue
        row_idx = int(item.get("row", index))
        raw_report = item.get("report")
        report: Mapping[str, Any] = raw_report if isinstance(raw_report, Mapping) else {}
        baseline_batches = [cast(Mapping[str, Any], batch) for batch in _list_or_empty(report.get("tick_batches")) if isinstance(batch, Mapping)]
        baseline_tokens = [int(token) for token in _outputs_from_batches(baseline_batches, "output_token_ids")]
        continuous_tokens = [int(token) for token in _active_outputs_for(live_report, row_idx, "output_token_ids")]
        baseline_hashes = [str(value) for value in _outputs_from_batches(baseline_batches, "output_logits_sha256")]
        continuous_hashes = [str(value) for value in _active_outputs_for(live_report, row_idx, "output_logits_sha256")]
        request = {
            "request_id": f"generate-{row_idx}",
            "arrival_tick": _first_active_tick(live_report, row_idx),
            "baseline": {
                "generated_token_ids": baseline_tokens,
                "logits_sha256": "|".join(baseline_hashes),
            },
            "continuous": {
                "generated_token_ids": continuous_tokens,
                "logits_sha256": "|".join(continuous_hashes),
            },
        }
        numeric = _numeric_comparison(
            _outputs_from_batches(baseline_batches, "output_logits_values"),
            _active_outputs_for(live_report, row_idx, "output_logits_values"),
        )
        if numeric is not None:
            request["logits_numeric_comparison"] = numeric
        requests.append(request)
    return {
        "model_id": raw.get("model_id") or raw.get("model"),
        "proof_gate": PROOF_GATE,
        "opt_in_flag": OPT_IN_FLAG,
        "opt_in_enabled": True,
        "server_observed_live_continuous_batches": live_report.get("server_observed_live_continuous_batches") is True,
        "live_server_proven": live_report.get("live_server_proven") is True,
        "speedup_proven": False,
        "wallclock_speedup_proven": False,
        "requests": requests,
        "live_continuous_report": live_report,
        "source": str(raw.get("source") or "strict_live_cbkv_evidence.py"),
    }
