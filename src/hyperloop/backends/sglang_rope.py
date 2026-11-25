"""Observe real SGLang Qwen2 RoPE layouts and evaluate the copy tradeoff separately."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

from ..kernel.ascend_forge import _WATCH_COORDINATOR, _regular_bytes, _run_worker
from .runtime import BackendError, Record, fingerprint
from .sglang_measure import _complete, _payload, _request
from .sglang_overlay import dedicated_launch, protected_paths


ROPE_CONTROL_ENV = "HYPERLOOP_ROPE_LAYOUT_CONTROL"
ROPE_MODEL_SOURCE = "python/sglang/srt/models/qwen2.py"
ROPE_SOURCE = "python/sglang/srt/layers/rotary_embedding/base.py"
_MARKER = "HYPERLOOP_ROPE_LAYOUT="
_ARMS = ("split_rope", "contiguous_rope", "contiguous_copy_only", "contiguous_copy_then_rope")
_PHASES = {"EXTEND": "prefill", "DECODE": "decode"}
_WORKER = Path(__file__).with_name("_sglang_rope_worker.py")
_MAX_WINDOW_BYTES = 8 * 1024**2
_MAX_TOKENS = 8192
_ANCHOR = ('        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)\n'
           '        q, k = self.rotary_emb(positions, q, k)\n')
_OBSERVER = '''\
        import os as _hl_rope_os

        if _hl_rope_control := _hl_rope_os.environ.get("HYPERLOOP_ROPE_LAYOUT_CONTROL"):
            try:
                with open(_hl_rope_control, encoding="utf-8") as _hl_rope_file:
                    _hl_rope_window = _hl_rope_file.read(80).strip()
            except FileNotFoundError:
                _hl_rope_window = ""
            if _hl_rope_window:
                import json as _hl_rope_json

                _hl_rope_mode = forward_batch.forward_mode.name
                _hl_rope_layouts = {name: {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                                          "storage_offset": tensor.storage_offset(),
                                          "is_contiguous": tensor.is_contiguous(), "dtype": str(tensor.dtype)}
                                    for name, tensor in (("packed_qkv", qkv), ("split_q", q),
                                                         ("split_k", k), ("positions", positions))}
                _hl_rope_key = (_hl_rope_mode, _hl_rope_json.dumps(_hl_rope_layouts, sort_keys=True))
                if globals().get("_hl_rope_window_id") != _hl_rope_window:
                    globals()["_hl_rope_window_id"] = _hl_rope_window
                    globals()["_hl_rope_observed"] = set()
                _hl_rope_seen = globals()["_hl_rope_observed"]
                if _hl_rope_key not in _hl_rope_seen and len(_hl_rope_seen) < 128:
                    _hl_rope_seen.add(_hl_rope_key)
                    print("HYPERLOOP_ROPE_LAYOUT=" + _hl_rope_json.dumps({
                        "window": _hl_rope_window, "pid": _hl_rope_os.getpid(), "forward_mode": _hl_rope_mode,
                        "tokens": int(q.shape[0]), "device": str(q.device), "layouts": _hl_rope_layouts,
                        "position_values": positions.detach().cpu().tolist(),
                        "cache_dtype": str(self.rotary_emb.cos_sin_cache.dtype),
                    }, allow_nan=False), flush=True)
'''


def rope_layout_source_edit() -> Record:
    split, call = _ANCHOR.splitlines(keepends=True)
    return {"candidate_id": "hyperloop-qwen2-rope-layout-observer", "patch_path": ROPE_MODEL_SOURCE,
            "source_edit": {"before": _ANCHOR, "after": split + _OBSERVER + call}}


def _valid_layout(item: Any, *, tokens: int, dim: int, dtypes: set[str]) -> bool:
    if not isinstance(item, Mapping):
        return False
    shape, stride = item.get("shape"), item.get("stride")
    return (isinstance(shape, list) and len(shape) == dim and shape[0] == tokens
            and all(type(v) is int and v > 0 for v in shape)
            and isinstance(stride, list) and len(stride) == dim and all(type(v) is int and v >= 0 for v in stride)
            and type(item.get("storage_offset")) is int and item["storage_offset"] >= 0
            and type(item.get("is_contiguous")) is bool and item.get("dtype") in dtypes)


def _checked_observation(row: Any) -> bool:
    if not isinstance(row, Mapping) or row.get("phase") not in _PHASES.values() or row.get("device") != "npu:0":
        return False
    tokens, layouts, positions = row.get("tokens"), row.get("layouts"), row.get("position_values")
    return (type(tokens) is int and 1 <= tokens <= _MAX_TOKENS and isinstance(layouts, Mapping)
            and all(_valid_layout(layouts.get(name), tokens=tokens, dim=2, dtypes={"torch.bfloat16"})
                    for name in ("packed_qkv", "split_q", "split_k"))
            and _valid_layout(layouts.get("positions"), tokens=tokens, dim=1, dtypes={"torch.int32", "torch.int64"})
            and isinstance(positions, list) and len(positions) == tokens
            and all(type(value) is int and value >= 0 for value in positions)
            and row.get("cache_dtype") == "torch.bfloat16")


def rope_layout_samples(log: Path, offset: int, window: str, allowed_workers: Collection[int]) -> list[Record]:
    try:
        with log.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(_MAX_WINDOW_BYTES + 1)
        if len(data) > _MAX_WINDOW_BYTES:
            raise BackendError("SGLang RoPE observation exceeded its log capture window")
        rows = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            if not line.startswith(_MARKER):
                continue
            observed = json.loads(line.removeprefix(_MARKER))
            if observed["window"] != window or type(observed["pid"]) is not int or observed["pid"] not in allowed_workers:
                raise BackendError("RoPE layout observation came from outside the owned service window")
            if (phase := _PHASES.get(observed["forward_mode"])) is None:
                continue
            row = {**observed, "phase": phase}
            if not _checked_observation(row):
                raise BackendError("RoPE layout observation differs from the locked Qwen2 path")
            rows.append(row)
        return rows
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise BackendError(f"could not read SGLang RoPE layout observations: {error}") from error


def observe_rope_layouts(
    base_url: str, workload: Mapping[str, Any], *, control: Path, log: Path,
    workers: Collection[int], timeout: float,
) -> Record:
    """Drive the fixed requests outside timing and retain owned service input layout receipts."""
    result: Record = {"layout_source": "service_observed", "layouts": [], "requests": []}
    for phase in _PHASES.values():
        requests = [request for request in workload["requests"] if request.get("phase") == phase]
        if not requests:
            continue
        with _request(base_url, "/flush_cache", timeout) as response:
            if not response.read(512).decode("utf-8", "replace").casefold().startswith("cache flushed."):
                raise BackendError("SGLang did not flush the cache before RoPE layout observation")
        payloads = [_payload(request, workload, stream=False) for request in requests]
        offset = log.stat().st_size
        window = uuid4().hex
        with control.open("x", encoding="utf-8") as stream:
            stream.write(window + "\n")
        try:
            control.chmod(0o644)
            with ThreadPoolExecutor(max_workers=min(workload["concurrency"], len(requests))) as pool:
                pending = [pool.submit(_complete, base_url, payload, timeout) for payload in payloads]
                replies = [job.result() for job in pending]
        finally:
            control.unlink()
        rows = [row for row in rope_layout_samples(log, offset, window, workers) if row["phase"] == phase]
        if not rows:
            raise BackendError(f"SGLang emitted no owned {phase} RoPE input layouts")
        result["layouts"].extend(rows)
        result["requests"].extend({
            "request_id": request["id"], "server_request_id": payload["rid"], "phase": phase,
            "prompt_tokens": len(payload["input_ids"]), "output_ids": reply["output_ids"], "window": window,
        } for request, payload, reply in zip(requests, payloads, replies, strict=True))
    return result


def _model_spec(model: Any) -> Record:
    if not isinstance(model, Mapping) or model.get("model_type") != "qwen2" or any(
        model.get(key) is not None for key in ("rope_scaling", "rope_parameters", "dual_chunk_attention_config")
    ):
        raise BackendError("RoPE layout probe requires the stock unscaled Qwen2 path")
    heads, kv_heads, size = (model.get(key) for key in ("num_attention_heads", "num_key_value_heads", "hidden_size"))
    if any(type(value) is not int or not 1 <= value <= 16384 for value in (heads, kv_heads, size)):
        raise BackendError("Qwen2 config contains unsupported attention dimensions")
    divisor = model.get("original_num_attention_heads", heads)
    if type(divisor) is not int or not 1 <= divisor <= size or size % divisor:
        raise BackendError("Qwen2 config contains unsupported head sizing")
    dim = model.get("head_dim") or size // divisor
    theta, max_position = model.get("rope_theta", 1000000), model.get("max_position_embeddings", 32768)
    if (type(dim) is not int or not 2 <= dim <= 256 or dim % 2 or not 1 <= kv_heads <= heads
            or type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0
            or type(max_position) is not int or not 1 <= max_position <= 131072):
        raise BackendError("Qwen2 config contains unsupported stock rotary dimensions")
    return {"head_dim": dim, "q_width": heads * dim, "kv_width": kv_heads * dim,
            "rope_theta": theta, "max_position_embeddings": max_position, "tp_size": 1}


def _cases(workload: Mapping[str, Any], observed: list[Record] | None, model: Record) -> list[Record]:
    result: list[Record] = []
    for phase in _PHASES.values():
        requests = [r for r in workload["requests"] if r.get("phase") == phase]
        if not requests:
            continue
        measured = [r for r in (observed or []) if r.get("phase") == phase]
        if measured:
            if any(not _checked_observation(row) or max(row["position_values"]) >= model["max_position_embeddings"]
                   for row in measured):
                raise BackendError("Cannot derive a RoPE probe case from invalid observed positions/layouts")
            sizes = sorted({row["tokens"] for row in measured})
            for tokens in dict.fromkeys((sizes[0], sizes[-1])):
                first = next(row for row in measured if row["tokens"] == tokens)
                result.append({"phase": phase, "tokens": tokens, "layout_source": "service_observed",
                               "observation_window": first["window"], "observation_pid": first["pid"],
                               "positions": first["position_values"], "positions_dtype": first["layouts"]["positions"]["dtype"].removeprefix("torch."),
                               "observed_layouts": first["layouts"]})
            continue
        lengths = [len(request["input_tokens"]) for request in requests]
        if phase == "prefill":
            positions = list(range(max(lengths)))
        else:
            positions = lengths[:workload["concurrency"]]
        if not positions or len(positions) > _MAX_TOKENS or max(positions) >= model["max_position_embeddings"]:
            raise BackendError("Frozen request sizes exceed the bounded RoPE layout probe")
        result.append({"phase": phase, "tokens": len(positions), "positions": positions, "positions_dtype": "int64",
                       "layout_source": "derived_model_config_and_frozen_request",
                       "representative_request_ids": [request["id"] for request in requests]})
    if not result:
        raise BackendError("RoPE layout probe requires a frozen prefill or decode request group")
    return result


def _result(artifact: bytes, inputs: bytes, worker: bytes, cases: list[Record]) -> Record:
    report = json.loads(artifact)
    if (report["scope"] != "qwen2_ascend_rope_layout_probe" or report["mode"] != "real"
            or report["input_sha256"] != sha256(inputs).hexdigest() or report["worker_sha256"] != sha256(worker).hexdigest()):
        raise BackendError("RoPE layout probe worker identity did not match the sealed job")
    if report["status"] != "ok":
        return report
    if len(report["cases"]) != len(cases) or report["warmup"] != 10 or report["repeats"] != 25:
        raise BackendError("RoPE layout probe did not return all frozen cases, warmups and samples")
    for measured, expected in zip(report["cases"], cases, strict=True):
        if (any(measured[key] != expected[key] for key in ("phase", "tokens", "layout_source"))
                or measured["outputs_match"] is not True or set(measured["samples_us"]) != set(_ARMS)):
            raise BackendError("RoPE timing was not paired after an output-compatible frozen case")
        for arm, samples in measured["samples_us"].items():
            if (len(samples) != report["repeats"] or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in samples)
                    or measured["median_us"][arm] != median(samples)):
                raise BackendError("RoPE worker returned inconsistent raw NPU timings")
    return report


def run_rope_layout_probe(
    *, python: Path, source_root: Path, model_config: Path, workload: Mapping[str, Any], profile: Mapping[str, Any],
    revision: str, workspace: Path, environment: Mapping[str, str], uid: int, gid: int, timeout: int = 180,
    observed_layouts: list[Record] | None = None,
) -> Record:
    """Evaluate original q/k views and explicit copies with an isolated pinned Ascend worker."""
    outcome: Record = {"mode": "real", "status": "blocked", "revision": revision, "report_ref": profile.get("report_ref")}
    try:
        if profile.get("status") != "ok" or profile.get("revision") != revision or not profile.get("report_ref"):
            return {**outcome, "reason": "no_matching_profile_for_rope"}
        source = source_root / ROPE_SOURCE
        source_bytes = source.read_bytes()
        callsites = [f"{ROPE_SOURCE.removeprefix('python/')}:{number}"
                     for number, code in enumerate(source_bytes.splitlines(), 1) if b"torch_npu.npu_mrope(" in code]
        if len(callsites) != 1:
            raise BackendError("Pinned SGLang source does not contain exactly one matching NPU RoPE callsite")
        hits = [hit for hit in profile.get("hotspots", []) if hit.get("callsite") == callsites[0]
                and ("ropewithsincoscache" in hit.get("name", "").casefold()
                     or "ropewithsincoscache" in hit.get("operator", "").casefold())]
        if not hits:
            return {**outcome, "reason": "no_source_correlated_rope_hotspot"}
        model_raw = model_config.read_bytes()
        model = _model_spec(json.loads(model_raw))
        cases = _cases(workload, observed_layouts, model)
        worker = _WORKER.read_bytes()
        workspace.mkdir(mode=0o755, exist_ok=False)
        worker_path, input_path = workspace / "rope-worker.py", workspace / "rope-input.json"
        worker_path.write_bytes(worker)
        inputs = json.dumps({"revision": revision, "rope_source_path": str(source),
                             "rope_source_sha256": sha256(source_bytes).hexdigest(), "model_spec": model,
                             "model_config_sha256": sha256(model_raw).hexdigest(), "cases": cases,
                             "workload_fingerprint": fingerprint(workload), "profile_ref": profile["report_ref"]},
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
        input_path.write_bytes(inputs)
        protected_paths([python, model_config, source, worker_path, input_path], uid, gid)
        output = workspace / "runtime" / "rope-result.json"
        command, env, runtime = dedicated_launch([
            str(python), "-c", _WATCH_COORDINATOR, str(os.getpid()), str(worker_path),
            "--input", str(input_path), "--output", str(output),
        ], {**environment, "PYTHONPATH": str(source_root / "python")}, workspace, None, uid, gid)
        supervisor = _run_worker(command, env, runtime, workspace, uid, timeout,
                                 log_name="rope-worker.log", lease_kind="rope_layout_probe")
        outcome.update(profile_hotspots=hits, worker_process=supervisor,
                       paths={"worker": "rope-worker.py", "input": "rope-input.json", "result": "runtime/rope-result.json",
                              "log": "rope-worker.log", "lease": "sglang-lease.json"},
                       workload_fingerprint=fingerprint(workload), model_config_sha256=sha256(model_raw).hexdigest())
        if not supervisor["process_group_clear"]:
            return {**outcome, "status": "failed", "reason": "cleanup_unconfirmed"}
        if supervisor["timed_out"] or not output.is_file():
            return {**outcome, "status": "failed",
                    "reason": "rope_layout_probe_timeout" if supervisor["timed_out"] else "rope_worker_missing_result"}
        report = _result(_regular_bytes(output), inputs, worker, cases)
        if supervisor["exit_code"] != 0 and report["status"] == "ok":
            raise BackendError("RoPE worker status conflicts with its exit code")
        return {**outcome, **report}
    except (BackendError, OSError, ValueError, KeyError, TypeError) as error:
        return {**outcome, "status": "failed", "reason": "rope_layout_driver_error", "error": str(error)}
