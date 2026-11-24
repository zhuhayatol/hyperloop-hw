"""Resolve a real SwiGLU hotspot and run the packaged Ascend Forge strategy."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
from statistics import median
import subprocess
from textwrap import indent
import time
from typing import Any

from ..backends.runtime import BackendError, Record
from ..backends.sglang_overlay import dedicated_launch, protected_paths


SWIGLU_PATH = "python/sglang/srt/layers/activation.py"
_PROFILE_PATH = SWIGLU_PATH.removeprefix("python/")
_WORKER = Path(__file__).with_name("_ascend_forge_worker.py")
_MARKER = "HYPERLOOP_FORGE_SWIGLU_INVOKED"
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

_WATCH_COORDINATOR = """\
import ctypes
import os
import runpy
import signal
import sys

def stop_group(*_args):
    os.killpg(os.getpgrp(), signal.SIGKILL)

parent = int(sys.argv[1])
signal.signal(signal.SIGTERM, stop_group)
if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0):
    raise OSError(ctypes.get_errno(), 'could not watch the Forge coordinator')
if os.getppid() != parent:
    stop_group()
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _kernel_source(worker: str) -> str:
    lines = worker.splitlines(keepends=True)
    functions = [node for node in ast.parse(worker).body if isinstance(node, ast.FunctionDef) and node.name in {
        "_hyperloop_forge_swiglu", "_hyperloop_forge_swiglu_launch",
    }]
    if len(functions) != 2:
        raise BackendError("could not locate the two packaged SwiGLU functions")
    start = functions[0].decorator_list[0].lineno - 1
    return "".join(lines[start:functions[-1].end_lineno])


def _operator_edit(source: bytes, worker: str, width: int, hotspots: list[Record]) -> Record | None:
    if _MARKER.encode() in source:
        raise BackendError("the accepted activation source already contains the Ascend Forge implementation")
    text = source.decode("utf-8")
    lines = text.splitlines(keepends=True)
    classes = [node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == "SiluAndMul"]
    if len(classes) != 1:
        raise BackendError("the pinned activation source must define one SiluAndMul class")
    cls = classes[0]
    forwards = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_npu"]
    if len(forwards) != 1:
        raise BackendError("the pinned activation source must define one SiluAndMul.forward_npu")
    forward = forwards[0]
    if len(forward.body) != 2 or [ast.unparse(node) for node in forward.body] != [
        "out = torch_npu.npu_swiglu(x)", "return out",
    ]:
        raise BackendError("the pinned activation source does not match the supported stock SwiGLU implementation")
    callsite = f"{_PROFILE_PATH}:{forward.body[0].lineno}"
    events = [hit for hit in hotspots if hit.get("callsite") == callsite and (
        str(hit.get("name", "")).casefold() == "swiglu" or str(hit.get("operator", "")).casefold() == "aclnnswiglu"
    )]
    if not events:
        return None

    kernel = _kernel_source(worker)
    digest = _sha(kernel.encode("utf-8"))
    marker = f"{_MARKER}_{digest[:12]}"
    helper = "if _is_npu:\n    import os as _hyperloop_forge_os\n    import triton\n    import triton.language as tl\n\n"
    helper += indent(kernel, "    ") + "\n\n_HYPERLOOP_FORGE_SWIGLU_RECORDED = False\n\n\n"
    call = f'''        global _HYPERLOOP_FORGE_SWIGLU_RECORDED
        if (x.ndim == 2 and 1 <= x.shape[0] <= 512 and x.shape[1] == {2 * width}
                and x.dtype == torch.bfloat16 and x.device.type == "npu" and x.is_contiguous()):
            candidate, compiled = _hyperloop_forge_swiglu_launch(x)
            if not _HYPERLOOP_FORGE_SWIGLU_RECORDED:
                print(f"{marker} pid={{_hyperloop_forge_os.getpid()}} tokens={{x.shape[0]}} width={width} "
                      f"kernel={{compiled.name}} hash={{compiled.hash}}", flush=True)
                _HYPERLOOP_FORGE_SWIGLU_RECORDED = True
            return candidate
'''
    first_line = cls.lineno - 1
    last_line = cls.end_lineno
    original_call = forward.body[0].lineno - 1
    before = "".join(lines[first_line:last_line])
    after = helper + "".join(lines[first_line:original_call]) + call + "".join(lines[original_call:last_line])
    if text.count(before) != 1:
        raise BackendError("the activation source patch anchor is ambiguous")
    compile(text.replace(before, after, 1), SWIGLU_PATH, "exec")
    return {
        "id": f"ascend-forge-swiglu-{digest[:12]}", "operator": "SwiGLU", "backend": "forge",
        "patch_path": SWIGLU_PATH, "source_edit": {"before": before, "after": after},
        "invocation_marker": marker, "pinned_source_sha256": _sha(source), "operator_source_sha256": digest,
        "hotspots": events, "supported_input": {"dtype": "bfloat16", "width": 2 * width, "tokens": [1, 512]},
        "reason": "real Ascend profile identified the SiluAndMul.forward_npu SwiGLU hotspot",
        "expected_metric": "single-operation NPU time and end-to-end service throughput",
        "risk": "compiled kernel and stock numerical results can differ; service results can regress",
        "validation": "independent CPU reference, interleaved device timings, and isolated SGLang service A/B",
    }


def _regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise BackendError(f"Forge artifact is not a regular file: {path.name}")
        content = stream.read(_MAX_ARTIFACT_BYTES + 1)
    if len(content) > _MAX_ARTIFACT_BYTES:
        raise BackendError(f"Forge artifact is too large: {path.name}")
    return content


def _check_worker(output: Path, *, worker_digest: str, model_digest: str, width: int) -> Record:
    if output.is_symlink():
        raise BackendError("Forge worker output directory must not be a symlink")
    result = json.loads(_regular_bytes(output / "result.json"))
    if (not isinstance(result, dict) or result.get("source_sha256") != worker_digest
            or result.get("scope") != "ascend_forge_swiglu"):
        raise BackendError("Forge worker identity did not match the staged worker")
    if result.get("status") != "complete":
        return result
    if (result.get("model_config_sha256") != model_digest or result.get("width") != width
            or result.get("warmup") != 10 or result.get("repeats") != 25 or len(result.get("cases", [])) != 2):
        raise BackendError("Forge did not measure both locked model-width cases")
    for index, (tokens, row) in enumerate(zip((2, 512), result["cases"], strict=True)):
        if row["input_shape"] != [tokens, 2 * width] or row["dtype"] != "bfloat16":
            raise BackendError("Forge measured an unexpected input shape or dtype")
        if any(row["correctness"][arm]["pass"] is not True or row["correctness"][arm]["failed_elements"] != 0
               for arm in ("stock_vs_cpu", "candidate_vs_cpu")):
            raise BackendError("Forge evidence contains a failed correctness check")
        for arm in ("torch_npu_stock", "triton_candidate"):
            samples = row["stream_event_us"][arm]
            if (len(samples) != result["repeats"]
                    or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in samples)
                    or row["median_us"][arm] != median(samples)):
                raise BackendError("Forge reported inconsistent raw NPU timing samples")
        compiled = row["compiled"]
        artifacts = compiled["artifacts"]
        if row["timing_status"] != "complete" or not any(item["kind"] == "npubin" for item in artifacts):
            raise BackendError("Forge result omitted required compiled or timing evidence")
        for entry in artifacts:
            kind = entry["kind"]
            name = f"shape-{index}.{kind}"
            if kind not in {"ttir", "ttadapter", "npubin"} or entry["file"] != name:
                raise BackendError("Forge worker reported an unrecognized compile artifact")
            content = _regular_bytes(output / name)
            if len(content) != entry["bytes"] or _sha(content) != entry["sha256"]:
                raise BackendError(f"Forge compile artifact differs from its reported identity: {name}")
    return result


def _group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


def _run_worker(command: list[str], environment: Mapping[str, str], runtime: Path, log: Path, timeout: int) -> Record:
    started = time.monotonic()
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=runtime, env=environment, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if _group_exists(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    pass
                for _ in range(20):
                    if not _group_exists(process.pid):
                        break
                    time.sleep(0.05)
        return {"exit_code": process.poll(), "timed_out": timed_out,
                "process_group_clear": not _group_exists(process.pid),
                "elapsed_seconds": time.monotonic() - started}


def run_forge(
    *, python: Path, model_config: Path, source_root: Path, hotspots: list[Record], workspace: Path,
    environment: Mapping[str, str], uid: int, gid: int, timeout: int = 180,
) -> Record:
    """Build and test one source-correlated candidate before offering its exact source edit."""
    receipt: Record = {"strategy": "packaged_triton_ascend_swiglu", "worker_log": "forge-worker.log",
                       "worker_evidence_root": "runtime/evidence"}
    result: Record = {"status": "blocked", "evidence": receipt}
    try:
        model = model_config.read_bytes()
        model_spec = json.loads(model)
        width = model_spec.get("intermediate_size")
        if model_spec.get("model_type") != "qwen2" or type(width) is not int or not 4 <= width <= 65536:
            return {**result, "reason": "unsupported_model_for_forge_swiglu"}
        source = (source_root / SWIGLU_PATH).read_bytes()
        worker = _WORKER.read_bytes()
        candidate = _operator_edit(source, worker.decode("utf-8"), width, hotspots)
        if candidate is None:
            return {**result, "reason": "no_source_correlated_forge_swiglu_hotspot"}
        receipt.update(profile_hotspots=candidate["hotspots"], pinned_source_sha256=candidate["pinned_source_sha256"],
                       worker_sha256=_sha(worker), model_config_sha256=_sha(model))
        workspace.mkdir(mode=0o755, exist_ok=False)
        worker_path = workspace / "forge-worker.py"
        worker_path.write_bytes(worker)
        protected_paths([python, model_config, source_root / SWIGLU_PATH, worker_path], uid, gid)
        output = workspace / "runtime" / "evidence"
        command, env, runtime = dedicated_launch([
            str(python), "-c", _WATCH_COORDINATOR, str(os.getpid()), str(worker_path),
            "--output", str(output), "--model-config", str(model_config),
        ], {key: value for key, value in environment.items() if key != "PYTHONPATH"}, workspace, None, uid, gid)
        process = _run_worker(command, env, runtime, workspace / "forge-worker.log", timeout)
        receipt["worker_process"] = process
        if not process["process_group_clear"]:
            return {**result, "status": "failed", "reason": "forge_worker_cleanup_unconfirmed"}
        if process["timed_out"]:
            return {**result, "status": "failed", "reason": "forge_worker_timeout"}
        if not (output / "result.json").is_file():
            return {**result, "status": "failed", "reason": "forge_worker_missing_result"}
        measured = _check_worker(output, worker_digest=_sha(worker), model_digest=_sha(model), width=width)
        receipt["operator"] = measured
        if process["exit_code"] != 0 or measured.get("status") != "complete":
            return {**result, "status": "failed", "reason": "forge_operator_verification_failed"}
        receipt["all_cases_faster"] = all(
            row["median_us"]["triton_candidate"] < row["median_us"]["torch_npu_stock"] for row in measured["cases"]
        )
        return {**result, "status": "ok", "reason": "forge_operator_built_and_verified", "candidate": candidate}
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, BackendError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
        return {**result, "status": "failed", "reason": "forge_driver_error"}
