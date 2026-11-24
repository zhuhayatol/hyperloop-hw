"""Measure the actual eager SGLang Qwen2 forward boundary on an Ascend stream."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

from .runtime import BackendError, Record
from .sglang_measure import _complete, _payload, _request


FORWARD_SOURCE = "python/sglang/srt/models/qwen2.py"
FORWARD_CONTROL_ENV = "HYPERLOOP_MODEL_FORWARD_CONTROL"
FORWARD_SYMBOL = "sglang.srt.models.qwen2.Qwen2ForCausalLM.forward"
FORWARD_MARKER = "HYPERLOOP_MODEL_FORWARD="
_MAX_WINDOW_BYTES = 8 * 1024**2
_PHASES = {"EXTEND": "prefill", "DECODE": "decode", "MIXED": "mixed"}
_ANCHOR = "EntryClass = Qwen2ForCausalLM\n"
_WRAPPER = '''
# Hyperloop's separate, eager model-forward measurement scope.
import json as _hl_forward_json
import os as _hl_forward_os
from functools import wraps as _hl_forward_wraps

_hl_forward_original = Qwen2ForCausalLM.forward
_hl_forward_control = _hl_forward_os.environ["HYPERLOOP_MODEL_FORWARD_CONTROL"]


@_hl_forward_wraps(_hl_forward_original)
def _hl_forward_timed(self, input_ids, positions, forward_batch, *args, **kwargs):
    try:
        with open(_hl_forward_control, encoding="utf-8") as source:
            capture = source.read(80).strip()
    except FileNotFoundError:
        return _hl_forward_original(self, input_ids, positions, forward_batch, *args, **kwargs)
    if input_ids.device.type != "npu":
        raise RuntimeError("Hyperloop's model forward must run on an Ascend NPU")
    stream = torch.npu.current_stream()
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    start.record(stream)
    result = _hl_forward_original(self, input_ids, positions, forward_batch, *args, **kwargs)
    stop.record(stream)
    stop.synchronize()
    seq_lengths = forward_batch.seq_lens_cpu
    print("HYPERLOOP_MODEL_FORWARD=" + _hl_forward_json.dumps({
        "window": capture, "pid": _hl_forward_os.getpid(), "forward_mode": forward_batch.forward_mode.name,
        "batch_size": int(forward_batch.batch_size), "input_shape": list(input_ids.shape),
        "position_shape": list(positions.shape), "input_dtype": str(input_ids.dtype),
        "model_dtype": str(self.get_input_embeddings().weight.dtype), "device": str(input_ids.device),
        "sequence_lengths": seq_lengths.tolist() if seq_lengths is not None else [],
        "extend_lengths": forward_batch.extend_seq_lens_cpu or [],
        "prefix_lengths": forward_batch.extend_prefix_lens_cpu or [],
        "elapsed_ms": float(start.elapsed_time(stop)),
    }, allow_nan=False), flush=True)
    return result


Qwen2ForCausalLM.forward = _hl_forward_timed
'''


def forward_repeats(config: Any) -> int | None:
    if (not isinstance(config, Mapping) or config.keys() - {"enabled", "repeats"}
            or type(config.get("enabled", False)) is not bool):
        raise BackendError("target.sglang.model_forward requires enabled and optional repeats")
    repeats = config.get("repeats", 3)
    if "repeats" in config and not config.get("enabled"):
        raise BackendError("target.sglang.model_forward.repeats requires enabled")
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise BackendError("target.sglang.model_forward.repeats must be an integer between 1 and 10")
    return repeats if config.get("enabled") else None


def forward_source_edit() -> Record:
    return {"candidate_id": "hyperloop-model-forward-timer", "patch_path": FORWARD_SOURCE,
            "source_edit": {"before": _ANCHOR, "after": _ANCHOR + _WRAPPER}}


def _valid_dims(dims: Any, *, allow_zero: bool = False) -> bool:
    return (isinstance(dims, list) and bool(dims)
            and all(type(value) is int and value >= (0 if allow_zero else 1) for value in dims))


def forward_samples(log: Path, offset: int, window: str, allowed_workers: Collection[int]) -> list[Record]:
    try:
        with log.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(_MAX_WINDOW_BYTES + 1)
        if len(data) > _MAX_WINDOW_BYTES:
            raise BackendError("SGLang model-forward log exceeded the capture window limit")
        samples = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            if not line.startswith(FORWARD_MARKER):
                continue
            row = json.loads(line.removeprefix(FORWARD_MARKER))
            mode, elapsed = row["forward_mode"], row["elapsed_ms"]
            if (row["window"] != window or type(row["pid"]) is not int or row["pid"] not in allowed_workers
                    or mode not in _PHASES or row["device"] != "npu:0" or row["model_dtype"] != "torch.bfloat16"
                    or row["input_dtype"] not in {"torch.int32", "torch.int64"}
                    or type(row["batch_size"]) is not int or row["batch_size"] < 1
                    or not _valid_dims(row["input_shape"]) or not _valid_dims(row["position_shape"])
                    or type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0):
                raise BackendError("SGLang model-forward sample conflicts with its owned capture")
            for key in ("sequence_lengths", "extend_lengths", "prefix_lengths"):
                if row[key] != [] and (not _valid_dims(row[key], allow_zero=key != "sequence_lengths")
                                        or len(row[key]) > row["batch_size"]):
                    raise BackendError("SGLang model-forward sample has invalid per-sequence shapes")
            samples.append({**row, "phase": _PHASES[mode]})
        if not samples:
            raise BackendError("SGLang produced no owned model-forward samples after correctness verification")
        return samples
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise BackendError(f"could not read the owned SGLang model-forward window: {error}") from error


def measure_forward(
    base_url: str, workload: Mapping[str, Any], *, repeats: int, control: Path, log: Path,
    workers: Collection[int], timeout: float,
) -> Record:
    requests = workload["requests"]
    concurrency = min(workload["concurrency"], len(requests))
    captured = []
    phases: dict[str, list[float]] = {}
    requested_phases = {request.get("phase") for request in requests} & {"prefill", "decode"}
    for repeat in range(repeats):
        with _request(base_url, "/flush_cache", timeout) as response:
            response.read(512)
        payloads = [_payload(request, workload, stream=False) for request in requests]
        offset = log.stat().st_size
        accession = uuid4().hex
        with control.open("x", encoding="utf-8") as stream:
            stream.write(accession + "\n")
        try:
            control.chmod(0o644)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending = [pool.submit(_complete, base_url, payload, timeout) for payload in payloads]
                replies = [job.result() for job in pending]
        finally:
            control.unlink()
        samples = forward_samples(log, offset, accession, workers)
        missing = requested_phases - {row["phase"] for row in samples}
        if missing:
            raise BackendError(f"SGLang model-forward repeat {repeat} missed required phases: {sorted(missing)}")
        for row in samples:
            phases.setdefault(row["phase"], []).append(row["elapsed_ms"])
        capture_requests = []
        for request, payload, reply in zip(requests, payloads, replies):
            capture_requests.append({"request_id": request["id"], "server_request_id": payload["rid"],
                                     "prompt_tokens": len(payload["input_ids"]), "output": reply["output_ids"]})
        captured.append({"repeat": repeat, "window": accession, "requests": capture_requests, "samples": samples})
    return {
        "boundary": "model_forward", "method": "torch.npu.Event on the model's current stream",
        "model_symbol": FORWARD_SYMBOL, "profile_enabled": False, "graph_replay": False,
        "http_role": "drive frozen requests outside the event boundary", "concurrency": concurrency,
        "cache_policy": "flush the radix cache before each capture window",
        "warmup_policy": "complete the untimed frozen-output verification before opening a capture window",
        "repeats": captured,
        "samples_ms": [row["elapsed_ms"] for repeat in captured for row in repeat["samples"]],
        "phases": {phase: {"count": len(values), "median_ms": median(values)} for phase, values in phases.items()},
    }
