"""Benchmark the exact SwiGLU implementation installed by the Ascend Forge route."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
from statistics import median
from time import monotonic

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _hyperloop_forge_swiglu(packed, output, width: tl.constexpr, block: tl.constexpr):
    blocks_per_row: tl.constexpr = (width + block - 1) // block
    program = tl.program_id(0)
    row = program // blocks_per_row
    column = (program % blocks_per_row) * block + tl.arange(0, block)
    row_start = row * (2 * width)
    valid = column < width
    gate = tl.load(packed + row_start + column, mask=valid, other=0).to(tl.float32)
    up = tl.load(packed + row_start + column + width, mask=valid, other=0).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    tl.store(output + row * width + column, activated * up, mask=valid)


def _hyperloop_forge_swiglu_launch(packed):
    tokens, packed_width = packed.shape
    width = packed_width // 2
    output = torch.empty((tokens, width), device=packed.device, dtype=packed.dtype)
    block = 256 if tokens <= 2 else 1024
    compiled = _hyperloop_forge_swiglu[(tokens * triton.cdiv(width, block),)](packed, output, width, block)
    return output, compiled


_WARMUP = 10
_REPEATS = 25
_ABS_TOLERANCE = 0.01
_REL_TOLERANCE = 0.015


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check(observed, expected):
    correct_identity = observed.dtype == torch.bfloat16 and observed.shape == expected.shape
    if not correct_identity:
        return {"pass": False, "observed_dtype": str(observed.dtype), "observed_shape": list(observed.shape)}
    values = observed.cpu().float()
    difference = (values - expected).abs()
    bad = ~torch.isclose(values, expected, atol=_ABS_TOLERANCE, rtol=_REL_TOLERANCE)
    maximum = float(difference.max())
    return {
        "pass": not bool(bad.any()),
        "max_absolute_error": maximum if math.isfinite(maximum) else None,
        "failed_elements": int(bad.sum()),
        "elements": values.numel(),
        "abs_tolerance": _ABS_TOLERANCE,
        "relative_tolerance": _REL_TOLERANCE,
    }


def _time(launch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    output = launch()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end)) * 1000
    del output
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("NPU stream event returned an invalid elapsed time")
    return elapsed


def _compiled_evidence(compiled, output: Path, index: int):
    artifacts = []
    for kind, value in compiled.asm.items():
        if kind not in {"ttir", "ttadapter", "npubin"} or not isinstance(value, (bytes, str)):
            continue
        content = value.encode() if isinstance(value, str) else value
        path = output / f"shape-{index}.{kind}"
        path.write_bytes(content)
        artifacts.append({"kind": kind, "file": path.name, "bytes": len(content), "sha256": _sha(content)})
    if not any(artifact["kind"] == "npubin" for artifact in artifacts):
        raise RuntimeError("Triton-Ascend did not provide native NPU binary evidence")
    return {"name": compiled.name, "hash": compiled.hash, "artifacts": artifacts}


def _case(tokens, width, output, index, evidence):
    packed_cpu = torch.randn((tokens, 2 * width), dtype=torch.float32).bfloat16().contiguous()
    packed_cpu[0, :4] = torch.tensor([0.0, -8.0, 8.0, -0.125], dtype=torch.bfloat16)
    left, right = packed_cpu.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(left) * right).bfloat16().float()
    packed = packed_cpu.to("npu:0")

    def candidate():
        return _hyperloop_forge_swiglu_launch(packed)

    def stock():
        return torch_npu.npu_swiglu(packed)

    row = {"input_shape": list(packed.shape), "dtype": "bfloat16",
           "phase_hint": "decode" if tokens <= 2 else "prefill"}
    evidence.append(row)
    started = monotonic()
    first, compiled = candidate()
    torch.npu.synchronize()
    row["first_compile_and_launch_seconds"] = monotonic() - started
    row["compiled"] = _compiled_evidence(compiled, output, index)
    row["correctness"] = {"stock_vs_cpu": _check(stock(), expected), "candidate_vs_cpu": _check(first, expected)}
    if not all(outcome["pass"] for outcome in row["correctness"].values()):
        row["timing_status"] = "skipped_wrong_output"
        return
    for _ in range(_WARMUP):
        stock()
        candidate()
    torch.npu.synchronize()
    timings = {"torch_npu_stock": [], "triton_candidate": []}
    routines = [("torch_npu_stock", stock), ("triton_candidate", candidate)]
    for repeat in range(_REPEATS):
        for name, operation in (routines if repeat % 2 == 0 else reversed(routines)):
            timings[name].append(_time(operation))
    row.update(timing_status="complete", stream_event_us=timings,
               median_us={name: median(values) for name, values in timings.items()})


def _run(args, result):
    if not torch.npu.is_available() or torch.npu.device_count() != 1:
        raise RuntimeError("Forge requires exactly one visible Ascend device")
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    target = triton.runtime.driver.active.get_current_target()
    if not ("ascend" in str(target).casefold() or getattr(target, "backend", None) == "npu"):
        raise RuntimeError(f"Triton active target is not Ascend: {target}")
    model_raw = args.model_config.read_bytes()
    config = json.loads(model_raw)
    width = config.get("intermediate_size")
    if config.get("model_type") != "qwen2" or type(width) is not int or not 4 <= width <= 65536:
        raise RuntimeError("Forge SwiGLU strategy requires a supported Qwen2 intermediate width")
    result.update(device=torch.npu.get_device_name(0), target=str(target), framework=metadata.version("sglang"),
                  triton=triton.__version__, triton_ascend=metadata.version("triton-ascend"),
                  torch=torch.__version__, torch_npu=torch_npu.__version__, model_config_sha256=_sha(model_raw),
                  width=width, cases=[])
    torch.manual_seed(911)
    for index, tokens in enumerate((2, 512)):
        _case(tokens, width, args.output, index, result["cases"])
    result["status"] = ("complete" if all(row.get("timing_status") == "complete" for row in result["cases"])
                        else "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "model-config"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, exist_ok=False)
    result = {"mode": "real", "scope": "ascend_forge_swiglu", "status": "started",
              "source_sha256": _sha(Path(__file__).read_bytes()),
              "workload_origin": "locked_model_width_and_representative_prefill_decode_tokens",
              "timing_boundary": "single_operation_npu_stream_event", "warmup": _WARMUP, "repeats": _REPEATS}
    try:
        _run(args, result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
