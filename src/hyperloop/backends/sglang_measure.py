"""Read real token output and streaming timing from the owned SGLang service."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException, HTTPResponse
from time import perf_counter_ns
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from .runtime import BackendError, Record


_MAX_FRAME_BYTES = 16 * 1024 * 1024


@contextmanager
def _request(
    base_url: str, path: str, timeout: float, payload: Record | None = None
) -> Iterator[HTTPResponse]:
    address = urlsplit(base_url)
    if (address.scheme != "http" or address.hostname not in {"127.0.0.1", "::1"}
            or address.username is not None or address.password is not None
            or address.path not in {"", "/"} or address.query or address.fragment):
        raise BackendError("SGLang measurements require an owned loopback HTTP service")
    connection = HTTPConnection(address.hostname, address.port, timeout=timeout)
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            connection.request("POST", path, data, {"Content-Type": "application/json"})
        with connection.getresponse() as response:
            if response.status != 200:
                detail = response.read(512).decode("utf-8", "replace")
                raise BackendError(f"SGLang {path} returned HTTP {response.status}: {detail}")
            yield response
    except (OSError, HTTPException) as error:
        raise BackendError(f"SGLang {path} transport failed: {error}") from error
    finally:
        connection.close()


def _payload(request: Mapping[str, Any], workload: Mapping[str, Any], *, stream: bool) -> Record:
    tokens = request["input_tokens"]
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise BackendError("SGLang measurement requires explicit nonempty input token ID lists")
    planned = request["output_tokens"]
    if type(planned) is not int or planned < (2 if stream else 1):
        raise BackendError("SGLang streaming timing requires at least two output tokens per request")
    sampling = dict(workload["sampling"])
    if (type(sampling.get("temperature")) not in (int, float) or sampling["temperature"] != 0
            or sampling.get("ignore_eos") is not True or sampling.get("n", 1) != 1):
        raise BackendError("SGLang verification requires one greedy output and ignore_eos=true")
    if "max_new_tokens" in sampling or "stream_interval" in sampling:
        raise BackendError("SGLang output length and stream interval are set by the measurement protocol")
    if "seed" in sampling:
        if "sampling_seed" in sampling:
            raise BackendError("SGLang sampling must provide at most one seed field")
        sampling["sampling_seed"] = sampling.pop("seed")
    sampling["max_new_tokens"] = planned
    if stream:
        sampling["stream_interval"] = 1
    return {
        "rid": uuid4().hex,
        "input_ids": tokens,
        "sampling_params": sampling,
        "stream": stream,
        "return_prompt_token_ids": not stream,
    }


def _response_object(data: bytes) -> Record:
    if len(data) > _MAX_FRAME_BYTES:
        raise BackendError("SGLang response exceeds the measurement frame limit")
    try:
        result = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackendError("SGLang response was not valid UTF-8 JSON") from error
    if not isinstance(result, dict) or "error" in result:
        raise BackendError(f"SGLang generation failed: {str(result)[:400]}")
    return result


def _metadata(response: Record, payload: Record) -> Record:
    meta = response.get("meta_info")
    if not isinstance(meta, dict) or meta.get("id") != payload["rid"]:
        raise BackendError("SGLang response request ID did not match the sent request")
    count = meta.get("prompt_tokens")
    if type(count) is not int or count != len(payload["input_ids"]):
        raise BackendError(f"SGLang reported {count!r} prompt tokens, expected {len(payload['input_ids'])}")
    return meta


def _check_completion(meta: Record, payload: Record) -> None:
    reason = meta.get("finish_reason")
    if not isinstance(reason, dict) or reason.get("type") not in {"length", "stop"}:
        raise BackendError(f"SGLang generation did not finish normally: {reason!r}")
    count = meta.get("completion_tokens")
    expected = payload["sampling_params"]["max_new_tokens"]
    if type(count) is not int or count != expected:
        raise BackendError(f"SGLang reported {count!r} output tokens, expected {expected}")


def _complete(base_url: str, payload: Record, timeout: float) -> Record:
    with _request(base_url, "/generate", timeout, payload) as response:
        reply = _response_object(response.read(_MAX_FRAME_BYTES + 1))
    metadata = _metadata(reply, payload)
    _check_completion(metadata, payload)
    prompt_ids = reply.get("prompt_token_ids")
    if prompt_ids != payload["input_ids"]:
        raise BackendError("SGLang did not echo the exact frozen prompt token IDs")
    output = reply.get("output_ids")
    if (not isinstance(output, list) or len(output) != metadata["completion_tokens"]
            or any(type(token) is not int or token < 0 for token in output)):
        raise BackendError("SGLang did not return a complete generated token ID sequence")
    return reply


def observe_outputs(base_url: str, workload: Mapping[str, Any], *, timeout: float = 120) -> Record:
    """Return actual prompt and generated token IDs for independent exact comparison."""
    cases = []
    for request in workload["requests"]:
        payload = _payload(request, workload, stream=False)
        reply = _complete(base_url, payload, timeout)
        output = reply["output_ids"]
        cases.append({
            "request_id": request["id"],
            "input_tokens": reply["prompt_token_ids"],
            "dtype": workload["dtype"],
            "shape": [len(output)],
            "output": output,
            "server_meta_info": reply["meta_info"],
        })
    return {"scope": "service", "output_encoding": "sglang_native_token_ids", "cases": cases}


def _stream_data(response: HTTPResponse) -> Iterator[bytes]:
    while line := response.readline(_MAX_FRAME_BYTES + 1):
        if len(line) > _MAX_FRAME_BYTES:
            raise BackendError("SGLang SSE frame exceeds the measurement frame limit")
        line = line.strip()
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            raise BackendError("SGLang sent unexpected event framing on /generate")
        yield line[5:].strip()


def _measure_request(
    base_url: str, request: Mapping[str, Any], payload: Record, timeout: float, sample_start: int
) -> Record:
    events: list[dict[str, int]] = []
    row: Record = {"request_id": request["id"], "sent_request_id": payload["rid"],
                   "input_tokens": payload["input_ids"], "token_events": events}
    started = perf_counter_ns()
    row["start_offset_ns"] = started - sample_start
    last_count = 0
    metadata: Record = {}
    try:
        with _request(base_url, "/generate", timeout, payload) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise BackendError("SGLang did not respond with a token stream")
            for data in _stream_data(response):
                arrived = perf_counter_ns()
                if arrived - started > timeout * 1_000_000_000:
                    raise BackendError("SGLang stream exceeded the per-request time limit")
                if data == b"[DONE]":
                    row["done_offset_ns"] = arrived - sample_start
                    break
                metadata = _metadata(_response_object(data), payload)
                count = metadata.get("completion_tokens")
                limit = payload["sampling_params"]["max_new_tokens"]
                if type(count) is not int or not last_count <= count <= limit:
                    raise BackendError("SGLang stream returned an invalid cumulative output count")
                if count > last_count:
                    events.append({"offset_ns": arrived - sample_start, "completion_tokens": count})
                    last_count = count
            else:
                raise BackendError("SGLang stream ended without its completion marker")
        _check_completion(metadata, payload)
        if len(events) < 2 or events[0]["completion_tokens"] != 1:
            raise BackendError("SGLang stream did not expose the first and later tokens separately")
        first_ns = events[0]["offset_ns"]
        last_ns = events[-1]["offset_ns"]
        row.update(
            status="ok", output_tokens=last_count, server_meta_info=metadata,
            ttft_ms=(first_ns - row["start_offset_ns"]) / 1_000_000,
            tpot_ms=(last_ns - first_ns) / ((last_count - 1) * 1_000_000),
        )
    except BackendError as error:
        row.update(status="failed", error=str(error), server_meta_info=metadata)
    return row


def benchmark(
    base_url: str, workload: Mapping[str, Any], *, repeat: int, warmup: int, timeout: float = 120
) -> Record:
    """Time real native HTTP generation with a bounded number of concurrent clients."""
    requests = workload["requests"]
    for index in range(warmup):
        _complete(base_url, _payload(requests[index % len(requests)], workload, stream=False), timeout)
    with _request(base_url, "/flush_cache", timeout) as response:
        response.read(512)
    payloads = [_payload(request, workload, stream=True) for request in requests]
    workers = min(workload["concurrency"], len(requests))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        start = perf_counter_ns()
        tasks = [pool.submit(_measure_request, base_url, request, payload, timeout, start)
                 for request, payload in zip(requests, payloads)]
        rows = [task.result() for task in tasks]
        elapsed_ns = max(row.get("done_offset_ns", 0) for row in rows)
        if any(row["status"] != "ok" for row in rows):
            elapsed_ns = perf_counter_ns() - start
    return {
        "boundary": "service_end_to_end", "profile_enabled": False,
        "repeat": repeat, "warmup": warmup, "elapsed_seconds": elapsed_ns / 1_000_000_000,
        "client_protocol": "sglang_native_sse", "client_concurrency": workers,
        "cache_policy": "flush_after_warmup_before_timing", "requests": rows,
    }
