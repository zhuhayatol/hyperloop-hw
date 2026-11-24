"""Collect isolated SGLang Ascend profiles and normalize their Chrome traces."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic, sleep
from typing import Any

from .runtime import BackendError, Record
from .sglang_measure import _complete, _payload, _request


_PHASES = ("prefill", "decode")
_ACTIVITIES = ("CPU", "GPU")  # SGLang 0.5.16 maps its GPU activity to NPU on Ascend.
_KERNEL_TASKS = {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV"}
_MAX_TRACE_BYTES = 256 * 1024 * 1024
_SGLANG_FRAME = re.compile(r"(?:^|[/\\])(?P<file>sglang[/\\](?:srt|lang)[/\\][^;\r\n()]+\.py)"
                           r"\((?P<line>\d+)\)")


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _point(event: Mapping[str, Any]) -> tuple[Any, Any, Decimal] | None:
    timestamp = _number(event.get("ts"))
    process, thread = event.get("pid"), event.get("tid")
    if timestamp is None or type(process) not in (int, str) or type(thread) not in (int, str):
        return None
    return process, thread, timestamp


def _arguments(event: Mapping[str, Any]) -> Mapping[str, Any]:
    args = event.get("args")
    return args if isinstance(args, Mapping) else {}


def _host_context(event: Mapping[str, Any]) -> Record:
    args = _arguments(event)
    stack = args.get("Call stack")
    frame = _SGLANG_FRAME.search(stack) if isinstance(stack, str) else None
    file = frame["file"].replace("\\", "/") if frame else ""
    dims = args.get("Input Dims", args.get("Input Shapes", []))
    shape = dims[0] if isinstance(dims, list) and dims and isinstance(dims[0], list) else dims
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 1 for dim in shape):
        shape = []
    types = args.get("Input type", args.get("Input Types", ""))
    dtype = types[0] if isinstance(types, list) and types else types
    name = event.get("name", "")
    return {
        "operator": name if isinstance(name, str) else "",
        "callsite": f"{file}:{frame['line']}" if frame else "",
        "input_shape": shape,
        "dtype": dtype if isinstance(dtype, str) else "",
        "source_correlation": "torch_to_npu",
    }


def _trace_events(document: Any, phase: str) -> tuple[list[Record], list[float]]:
    entries = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(entries, list):
        raise BackendError("Ascend Chrome trace has no event list")
    entries = [event for event in entries if isinstance(event, dict)]
    hardware_pids = {
        event["pid"] for event in entries
        if event.get("ph") == "M" and event.get("name") == "process_name"
        and type(event.get("pid")) in (int, str)
        and str(_arguments(event).get("name", "")).casefold().startswith("ascend hardware")
    }
    host_ops: dict[tuple[Any, Any, Decimal], list[Record]] = {}
    starts: dict[int | str, set[tuple[Any, Any, Decimal]]] = {}
    ends: dict[tuple[Any, Any, Decimal], set[int | str]] = {}
    hardware = []
    for event in entries:
        point = _point(event)
        if point is None:
            continue
        if event.get("ph") == "X":
            if point[0] in hardware_pids:
                hardware.append((point, event))
            elif event.get("cat") == "cpu_op":
                host_ops.setdefault(point, []).append(event)
        elif event.get("name") == "torch_to_npu" and type(event.get("id")) in (int, str):
            identifier = event["id"]
            if event.get("ph") == "s":
                starts.setdefault(identifier, set()).add(point)
            elif event.get("ph") == "f" and point[0] in hardware_pids:
                ends.setdefault(point, set()).add(identifier)

    rows = []
    window: list[Decimal] = []
    for point, event in hardware:
        duration = _number(event.get("dur"))
        name = event.get("name")
        if duration is None or duration <= 0 or not isinstance(name, str) or not name:
            continue
        source_points = {start for key in ends.get(point, ()) for start in starts.get(key, ())}
        sources = [operator for start in source_points for operator in host_ops.get(start, ())]
        host = _host_context(sources[0]) if len(sources) == 1 else {
            "operator": "", "callsite": "", "input_shape": [], "dtype": "", "source_correlation": "unresolved",
        }
        timestamp = point[2]
        task_type = _arguments(event).get("Task Type", "")
        task_type = task_type if isinstance(task_type, str) else ""
        rows.append({"name": name, "duration_us": float(duration), "timestamp_us": str(timestamp),
                     "phase": phase, "device_stream": point[1], "device_task_type": task_type,
                     "event_kind": "kernel" if task_type in _KERNEL_TASKS else "device_runtime", **host})
        end = timestamp + duration
        window = [min(window[0], timestamp), max(window[1], end)] if window else [timestamp, end]
    if not rows:
        raise BackendError("Ascend Chrome trace has no timed hardware events")
    return rows, [float(value) for value in window]


def _read_trace(path: Path, phase: str, evidence_root: Path) -> tuple[list[Record], Record]:
    with path.open("rb") as stream:
        data = stream.read(_MAX_TRACE_BYTES + 1)
    if len(data) > _MAX_TRACE_BYTES:
        raise BackendError(f"Ascend trace exceeds the {_MAX_TRACE_BYTES // 1024**2} MiB parse limit: {path}")
    rows, window = _trace_events(json.loads(data, parse_float=Decimal), phase)
    location = path.resolve().relative_to(evidence_root).as_posix()
    return rows, {"path": location, "sha256": sha256(data).hexdigest(), "bytes": len(data), "window_us": window,
                  "device_events": len(rows), "kernel_events": sum(row["event_kind"] == "kernel" for row in rows),
                  "resolved_callsites": sum(bool(row["callsite"]) for row in rows)}


def _await_trace(directory: Path, phase: str, evidence_root: Path, timeout: float) -> tuple[list[Record], Record]:
    deadline = monotonic() + timeout
    last_error = "trace_view.json has not been exported"
    while True:
        traces = sorted(path for path in directory.rglob("trace_view.json") if path.is_file())
        if len(traces) > 1:
            raise BackendError(f"Expected one Ascend Chrome trace for the {phase} window, found {len(traces)}")
        if traces:
            path = traces[0]
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise BackendError("Ascend trace escaped its owned capture directory")
            try:
                return _read_trace(path, phase, evidence_root)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                # An asynchronous export can make the file visible before writing its final JSON frame.
                last_error = str(error)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise BackendError(f"Ascend {phase} profile export unavailable in {directory}: {last_error}")
        sleep(min(0.25, remaining))


def _profile_api(base_url: str, endpoint: str, timeout: float, body: Record | None = None) -> str:
    with _request(base_url, endpoint, timeout, body) as response:
        return response.read(512).decode("utf-8", "replace").strip()


def _profile_phase(
    base_url: str, workload: Mapping[str, Any], phase: str, requests: list[Record], directory: Path,
    *, evidence_root: Path, request_timeout: float, export_timeout: float,
) -> tuple[list[Record], Record]:
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise BackendError(f"Could not create the owned {phase} profile directory: {error}") from error
    # Warm the service once for this shape and remove any radix-cache prefix before capture.
    _complete(base_url, _payload(requests[0], workload, stream=False), request_timeout)
    cache_ack = _profile_api(base_url, "/flush_cache", request_timeout)
    if not cache_ack.casefold().startswith("cache flushed."):
        raise BackendError(f"SGLang did not confirm cache flush before the {phase} profile: {cache_ack}")
    start_ack = _profile_api(base_url, "/start_profile", request_timeout, {
        "output_dir": str(directory), "activities": list(_ACTIVITIES), "with_stack": True, "record_shapes": True,
    })
    workers = min(workload["concurrency"], len(requests))
    try:
        payloads = [_payload(request, workload, stream=False) for request in requests]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_complete, base_url, payload, request_timeout) for payload in payloads]
            observations = [future.result()["meta_info"] for future in futures]
    except Exception as error:
        try:
            _profile_api(base_url, "/stop_profile", export_timeout, {})
        except BackendError as stop_error:
            raise BackendError(f"{error}; could not stop the Ascend profile: {stop_error}") from error
        raise
    stop_ack = _profile_api(base_url, "/stop_profile", export_timeout, {})
    rows, trace = _await_trace(directory, phase, evidence_root, export_timeout)
    cases = [{"request_id": request["id"], "server_request_id": meta["id"],
              "input_tokens": meta["prompt_tokens"], "output_tokens": meta["completion_tokens"]}
             for request, meta in zip(requests, observations)]
    return rows, {"phase": phase, "trace": trace, "requests": cases, "concurrency": workers,
                  "start_ack": start_ack, "stop_ack": stop_ack, "cache_ack": cache_ack}


def collect_profile(
    base_url: str, workload: Mapping[str, Any], *, revision: str, tool_version: str, directory: Path,
    evidence_root: Path, request_timeout: float, export_timeout: float,
) -> Record:
    """Export one distinct rank-zero trace per frozen request group, outside performance timing."""
    grouped: dict[str, list[Record]] = {phase: [] for phase in _PHASES}
    for case in workload["requests"]:
        phase = case.get("phase")
        if not isinstance(phase, str) or phase not in grouped:
            raise BackendError("Ascend profiling requires each request to declare prefill or decode")
        grouped[phase].append(case)
    phases = []
    events = []
    for phase, cases in grouped.items():
        if not cases:
            continue
        rows, capture = _profile_phase(base_url, workload, phase, cases, directory / phase,
                                       evidence_root=evidence_root, request_timeout=request_timeout,
                                       export_timeout=export_timeout)
        phases.append(capture)
        events.extend(rows)
    if not phases:
        raise BackendError("Ascend profiling requires at least one frozen request group")
    windows = [phase["trace"]["window_us"] for phase in phases]
    return {
        "boundary": "profile_capture",
        "capture": {
            "tool": "SGLang Ascend Torch NPU profiler", "tool_version": tool_version,
            "window_us": [min(window[0] for window in windows), max(window[1] for window in windows)],
            "activities": list(_ACTIVITIES), "phase_grouping": "frozen_request_group", "phases": phases,
        },
        "ranks": [{"rank": 0, "revision": revision, "events": events,
                   "trace_files": [phase["trace"] for phase in phases]}],
    }
