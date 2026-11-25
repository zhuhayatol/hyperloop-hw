"""Measure stock Qwen2 RoPE with controlled input layouts on one Ascend stream."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
from statistics import median

import torch
import torch_npu


_ARMS = ("split_rope", "contiguous_rope", "contiguous_copy_only", "contiguous_copy_then_rope")
_WARMUP = 10
_REPEATS = 25


def _layout(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
            "is_contiguous": tensor.is_contiguous(), "dtype": str(tensor.dtype)}


def _outputs(reference, candidate):
    cases = []
    for left, right in zip(reference, candidate, strict=True):
        left, right = left.cpu(), right.cpu()
        same_type = left.dtype == right.dtype and left.shape == right.shape
        difference = (left.float() - right.float()).abs() if same_type else None
        maximum = float(difference.max()) if difference is not None else None
        cases.append({"match": same_type and torch.equal(left, right), "shape": list(right.shape),
                      "dtype": str(right.dtype), "mismatched_elements": int((left != right).sum()) if same_type else None,
                      "max_abs_difference": maximum if maximum is not None and math.isfinite(maximum) else None})
    return cases


def _elapsed(launch, stream):
    start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record(stream)
    result = launch()
    stop.record(stream)
    stop.synchronize()
    elapsed = float(start.elapsed_time(stop)) * 1000
    del result
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("RoPE probe got an invalid NPU stream timing")
    return elapsed


def _measure_case(spec, rope, widths):
    tokens, q_width, kv_width = spec["tokens"], widths[0], widths[1]
    packed_width = q_width + 2 * kv_width
    packed = torch.randn((tokens, packed_width), dtype=torch.float32).bfloat16().to("npu:0").contiguous()
    q, k, _ = packed.split([q_width, kv_width, kv_width], dim=-1)
    q_contiguous, k_contiguous = q.contiguous(), k.contiguous()
    positions = torch.tensor(spec["positions"], dtype=getattr(torch, spec["positions_dtype"]), device="npu:0")
    layouts = {"packed_qkv": _layout(packed), "split_q": _layout(q), "split_k": _layout(k),
               "contiguous_q": _layout(q_contiguous), "contiguous_k": _layout(k_contiguous),
               "positions": _layout(positions)}
    observed = spec.get("observed_layouts")
    if observed is not None and any(layouts[name] != layout for name, layout in observed.items()):
        raise RuntimeError("The synthetic RoPE inputs do not reproduce the observed service layout")
    row = {key: value for key, value in spec.items() if key not in {"positions", "observed_layouts"}}
    row.update(layouts=layouts, copy_materialized={"q": q.data_ptr() != q_contiguous.data_ptr(),
                                                  "k": k.data_ptr() != k_contiguous.data_ptr()},
               qkv_values_source="seeded_synthetic_bfloat16", position_values=spec["positions"])

    def split_rope():
        return rope.forward_npu(positions, q, k)

    def contiguous_rope():
        return rope.forward_npu(positions, q_contiguous, k_contiguous)

    def contiguous_copy_only():
        return q.contiguous(), k.contiguous()

    def contiguous_copy_then_rope():
        return rope.forward_npu(positions, q.contiguous(), k.contiguous())

    routines = dict(zip(_ARMS, (split_rope, contiguous_rope, contiguous_copy_only, contiguous_copy_then_rope)))
    reference = split_rope()
    checks = {name: _outputs(reference, routines[name]()) for name in ("contiguous_rope", "contiguous_copy_then_rope")}
    row["correctness"] = checks
    row["outputs_match"] = all(item["match"] for check in checks.values() for item in check)
    if not row["outputs_match"]:
        return row
    for _ in range(_WARMUP):
        for operation in routines.values():
            operation()
    stream = torch.npu.current_stream()
    stream.synchronize()
    samples = {name: [] for name in _ARMS}
    for pair in range(_REPEATS):
        offset = pair % len(_ARMS)
        for name in (*_ARMS[offset:], *_ARMS[:offset]):
            samples[name].append(_elapsed(routines[name], stream))
    row.update(samples_us=samples, median_us={key: median(values) for key, values in samples.items()},
               paired_median_us={
                   "rope_only_contiguous_minus_split": median(b - a for a, b in zip(samples["split_rope"], samples["contiguous_rope"], strict=True)),
                   "copy_and_rope_minus_split": median(b - a for a, b in zip(samples["split_rope"], samples["contiguous_copy_then_rope"], strict=True)),
               })
    return row


def _initialize_runtime():
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    settings = {"model_path": "dummy", "device": "npu", "dtype": "bfloat16", "tp_size": 1,
                "attention_backend": "ascend", "rl_on_policy_target": None}
    # The pinned ServerArgs dummy mode publishes typed defaults without starting a server.
    # RotaryEmbedding reads rl_on_policy_target while constructing the ordinary cache.
    get_context().set_server_args(ServerArgs(**settings))
    return {"scope": "standalone_operator", "publisher": "RuntimeContext.set_server_args", **settings}


def _run(inputs, output):
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope

    recorded_source = Path(inputs["rope_source_path"]).resolve(strict=True)
    loaded_source = Path(inspect.getsourcefile(RotaryEmbedding.forward_npu)).resolve(strict=True)
    if loaded_source != recorded_source or sha256(loaded_source.read_bytes()).hexdigest() != inputs["rope_source_sha256"]:
        raise RuntimeError("RoPE worker did not import its pinned SGLang implementation")
    if not torch.npu.is_available() or torch.npu.device_count() != 1:
        raise RuntimeError("RoPE probe requires one visible Ascend device")
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(9103)
    output["runtime_context"] = _initialize_runtime()
    spec = inputs["model_spec"]
    with torch.inference_mode():
        # The pinned SGLang model loader initializes the module on its target device.
        with torch.device("npu:0"):
            rope = get_rope(spec["head_dim"], rotary_dim=spec["head_dim"], max_position=spec["max_position_embeddings"],
                            base=spec["rope_theta"], dtype=torch.bfloat16)
        if (type(rope) is not RotaryEmbedding or rope._forward_method != rope.forward_npu
                or rope.cos_sin_cache.dtype != torch.bfloat16 or str(rope.cos_sin_cache.device) != "npu:0"):
            raise RuntimeError("RoPE probe expected the Qwen2 stock two-dimensional NPU path and BF16 cache")
        output.update(device=torch.npu.get_device_name(0), software={"sglang": version("sglang"),
                      "torch": torch.__version__, "torch_npu": torch_npu.__version__},
                      loaded_source={"path": str(loaded_source), "sha256": inputs["rope_source_sha256"]},
                      implementation="sglang.srt.layers.rotary_embedding.base.RotaryEmbedding.forward_npu",
                      primitive="torch_npu.npu_mrope", cache_dtype=str(rope.cos_sin_cache.dtype), cases=[])
        for case in inputs["cases"]:
            measured = _measure_case(case, rope, (spec["q_width"], spec["kv_width"]))
            output["cases"].append(measured)
            if not measured["outputs_match"]:
                output.update(status="failed", reason="rope_output_mismatch")
                return
    output.update(status="ok", reason="rope_layout_probe_complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("input", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = {"mode": "real", "scope": "qwen2_ascend_rope_layout_probe", "status": "started", "input_sha256": sha256(raw).hexdigest(),
              "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "warmup": _WARMUP, "repeats": _REPEATS,
              "timing_boundary": "single_rope_and_copy_npu_events",
              "timing_protocol": "torch.npu.Event on current stream; one specified arm per bracket",
              "pair_order": "four arms per pair; rotate first arm each pair"}
    try:
        _run(json.loads(raw), result)
    except Exception as error:
        result.update(status="failed", reason="rope_layout_probe_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
