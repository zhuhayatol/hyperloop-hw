"""Compute comparable metrics and validate raw runtime observations."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping

from .evidence import EvidenceProblem

METRICS = ("throughput", "ttft_ms", "tpot_ms")
METRIC_DEFINITIONS = {
    "throughput": "output tokens divided by elapsed end-to-end seconds",
    "ttft_ms": "nearest-rank P95 across requests in each repetition",
    "tpot_ms": "nearest-rank P95 across requests in each repetition",
    "repetition_summary": "median",
}


def numeric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class EvaluationPolicy:
    repeats: int = 3
    warmup: int = 0
    tolerance: float = 0.001
    min_gain_pct: float = 2
    noise_floor_pct: float = 0
    max_ttft_regression_pct: float = 5
    max_tpot_regression_pct: float = 5

    @classmethod
    def load(cls, policy: Mapping[str, Any]) -> EvaluationPolicy:
        known = {name: policy[name] for name in cls.__dataclass_fields__ if name in policy}
        result = cls(**known)
        if type(result.repeats) is not int or result.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(result.warmup) is not int or result.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        for name in cls.__dataclass_fields__.keys() - {"repeats", "warmup"}:
            value = getattr(result, name)
            if not numeric(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        return result


def workload_requests(workload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(workload.get("id"), str) or not workload["id"]:
        raise ValueError("workload requires a nonempty id")
    concurrency = workload.get("concurrency", 1)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("workload concurrency must be a positive integer")
    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("workload requires a nonempty requests list")
    keyed = {}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("workload requests must be JSON objects")
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id or request_id in keyed:
            raise ValueError("request ids must be nonempty and unique")
        tokens = request.get("input_tokens")
        counts_valid = type(tokens) is int and tokens >= 0
        ids_valid = isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens)
        if not (counts_valid or ids_valid):
            raise ValueError("input_tokens must be a token ID list or a nonnegative token count")
        if type(request.get("output_tokens")) is not int or request["output_tokens"] < 1:
            raise ValueError("output_tokens must be a positive integer")
        keyed[request_id] = request
    return keyed


def keyed_rows(rows: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise EvidenceProblem("request_records_invalid")
    keyed = {}
    for row in rows:
        identifier = row.get(key) if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in keyed:
            raise EvidenceProblem("request_records_invalid")
        keyed[identifier] = row
    return keyed


def correctness(
    observation: Mapping[str, Any],
    requests: Mapping[str, Any],
    reference: Mapping[str, list[float]],
    tolerance: float,
    dtype: str,
) -> dict[str, Any]:
    cases = keyed_rows(observation.get("cases"), "request_id")
    if cases.keys() != requests.keys():
        raise EvidenceProblem("correctness_cases_missing", "missing", "partial")
    results = []
    for request_id, request in requests.items():
        case = cases[request_id]
        if case.get("input_tokens") != request["input_tokens"] or case.get("dtype") != dtype:
            raise EvidenceProblem("correctness_inputs_mismatch", "incomparable")
        output = case.get("output")
        expected = reference[request_id]
        valid = (
            isinstance(output, list)
            and len(output) == len(expected)
            and case.get("shape") == [len(expected)]
            and all(numeric(value) for value in output)
        )
        error = max(abs(actual - target) for actual, target in zip(output, expected)) if valid else None
        results.append({
            "request_id": request_id,
            "dtype": dtype,
            "shape": case.get("shape"),
            "max_absolute_error": error,
            "pass": error is not None and error <= tolerance,
        })
    return {"pass": all(record["pass"] for record in results), "tolerance": tolerance, "cases": results}


def benchmark_sample(
    raw: Mapping[str, Any], requests: Mapping[str, Any], repeat: int, warmup: int
) -> dict[str, float]:
    if raw.get("boundary") != "service_end_to_end" or raw.get("profile_enabled") is not False:
        raise EvidenceProblem("timing_boundary_mismatch", "incomparable")
    if raw.get("repeat") != repeat or raw.get("warmup") != warmup:
        raise EvidenceProblem("timing_schedule_mismatch", "incomparable")
    elapsed = raw.get("elapsed_seconds")
    if not numeric(elapsed) or elapsed <= 0:
        raise EvidenceProblem("invalid_elapsed_seconds")
    actual = keyed_rows(raw.get("requests"), "request_id")
    if actual.keys() != requests.keys():
        raise EvidenceProblem("timing_requests_missing", "missing", "partial")
    ttft: list[float] = []
    tpot: list[float] = []
    total_tokens = 0
    for request_id, request in requests.items():
        row = actual[request_id]
        if row.get("status") != "ok":
            raise EvidenceProblem("request_failed", "missing", "partial")
        if row.get("input_tokens") != request["input_tokens"] or row.get("output_tokens") != request["output_tokens"]:
            raise EvidenceProblem("timing_tokens_mismatch", "incomparable")
        for metric, values in (("ttft_ms", ttft), ("tpot_ms", tpot)):
            value = row.get(metric)
            if not numeric(value) or value <= 0:
                raise EvidenceProblem("invalid_request_latency")
            values.append(value)
        total_tokens += row["output_tokens"]
    percentile_index = math.ceil(len(actual) * 0.95) - 1
    return {"throughput": total_tokens / elapsed,
            "ttft_ms": sorted(ttft)[percentile_index], "tpot_ms": sorted(tpot)[percentile_index]}


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    return {metric: median(sample[metric] for sample in samples) for metric in METRICS}


def change_ratios(measured: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, float]:
    for key in ("mode", "environment_fingerprint", "workload_fingerprint", "protocol_fingerprint", "boundary"):
        if key not in anchor or measured[key] != anchor[key]:
            raise EvidenceProblem(f"comparison_{key}_mismatch", "incomparable")
    left = measured["summary"]
    right = anchor.get("summary", {})
    if any(not numeric(right.get(key)) or right[key] <= 0 for key in METRICS):
        raise EvidenceProblem("comparison_samples_missing", "missing", "partial")
    keys = ("throughput_pct", "ttft_pct", "tpot_pct")
    return {label: 100 * (left[key] - right[key]) / right[key] for label, key in zip(keys, METRICS)}


def rank_hotspots(rank_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = rank_record.get("events")
    if not isinstance(events, list):
        raise EvidenceProblem("profile_events_invalid")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            raise EvidenceProblem("profile_event_invalid")
        name = event.get("name")
        callsite = event.get("callsite")
        duration = event.get("duration_us")
        if not isinstance(name, str) or not isinstance(callsite, str) or not numeric(duration) or duration < 0:
            raise EvidenceProblem("profile_event_invalid")
        grouped[name, callsite].append(duration)
    return sorted(
        [{"rank": rank_record["rank"], "name": name, "callsite": callsite, "events": len(durations),
          "duration_us": sum(durations)} for (name, callsite), durations in grouped.items()],
        key=lambda item: (-item["duration_us"], item["name"], item["callsite"]),
    )
