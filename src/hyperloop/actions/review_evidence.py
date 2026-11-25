"""Collect and independently summarize evidence for a pending RoPE layout edit."""

from __future__ import annotations

from math import isfinite
from statistics import median, quantiles
from typing import Any

from ..backends import BackendError, RuntimeBackend
from ..backends.runtime import fingerprint
from ..contracts import read_json
from ..session import SessionStore
from .candidates import stack_identity


ROPE_ACTION = "rope_layout_probe"
ROPE_SOURCE = "python/sglang/srt/models/qwen2.py"
ROPE_BOUNDARY = "single_rope_and_copy_npu_events"
_CALL = "        q, k = self.rotary_emb(positions, q, k)\n"
_COPY_VARIANTS = (
    "        q = q.contiguous()\n        k = k.contiguous()\n" + _CALL,
    "        q, k = q.contiguous(), k.contiguous()\n" + _CALL,
    "        q, k = self.rotary_emb(positions, q.contiguous(), k.contiguous())\n",
)
_ARMS = ("split_rope", "contiguous_rope", "contiguous_copy_only", "contiguous_copy_then_rope")
_LAYOUT_SOURCES = {"service_observed", "derived_model_config_and_frozen_request", "simulated"}


def is_rope_layout_edit(proposal: dict) -> bool:
    candidate = proposal.get("params", {}).get("candidate", {})
    if proposal.get("action") != "integrate_patch" or candidate.get("patch_path") != ROPE_SOURCE:
        return False
    edit = candidate.get("source_edit", {})
    before, after = edit.get("before"), edit.get("after")
    return (
        isinstance(before, str) and before.count(_CALL) == 1
        and any(before.replace(_CALL, variant, 1) == after for variant in _COPY_VARIANTS)
    )


def _summarize_case(case: dict) -> dict:
    raw = case["samples_us"]
    if set(raw) != set(_ARMS):
        raise ValueError("unexpected RoPE timing arms")
    samples = [raw[arm] for arm in _ARMS]
    if any(not isinstance(arm, list) for arm in samples) or len({len(arm) for arm in samples}) != 1:
        raise ValueError("RoPE timing arms must have the same paired sample count")
    if len(samples[0]) < 4 or any(
        type(value) not in (int, float) or not isfinite(value) or value < 0
        for arm in samples for value in arm
    ):
        raise ValueError("RoPE timings must contain finite nonnegative samples")
    if (case["phase"] not in ("prefill", "decode") or type(case["tokens"]) is not int
            or case["tokens"] < 1 or case["layout_source"] not in _LAYOUT_SOURCES
            or type(case["outputs_match"]) is not bool):
        raise ValueError("invalid RoPE case identity or correctness record")
    strides = [case["layouts"][key]["stride"] for key in ("split_q", "split_k")]
    if any(not isinstance(stride, list) or not stride or any(type(dim) is not int or dim < 0 for dim in stride)
           for stride in strides):
        raise ValueError("invalid RoPE input stride")

    original, contiguous, copy_only, copy_then_rope = samples
    rope_gain = [a - b for a, b in zip(original, contiguous, strict=True)]
    net_gain = [a - b for a, b in zip(original, copy_then_rope, strict=True)]
    p25, _, p75 = quantiles(net_gain, n=4, method="inclusive")
    return {
        "phase": case["phase"], "tokens": case["tokens"], "layout_source": case["layout_source"],
        "sample_count": len(original), "outputs_match": case["outputs_match"],
        "q_stride": strides[0], "k_stride": strides[1],
        "split_rope_us": median(original), "contiguous_rope_us": median(contiguous),
        "copy_us": median(copy_only), "copy_plus_rope_us": median(copy_then_rope),
        "rope_gain_us": median(rope_gain), "net_gain_us": median(net_gain),
        "net_p25_us": p25, "net_p75_us": p75,
    }


def collect_rope_evidence(store: SessionStore, backend: RuntimeBackend, task: dict, state: dict) -> dict:
    params = task["params"]
    revision = state["current_revision"]
    profile_ref = params["profile_ref"]
    if (revision["id"] != params["revision"] or revision["framework"] or revision["kernels"]
            or not store.verify_evidence(profile_ref)):
        return {"status": "failed", "reason": "rope_probe_parent_or_profile_changed"}
    profile = {**read_json(store.artifact_path(profile_ref["path"])), "status": "ok",
               "revision": revision["id"], "report_ref": profile_ref}
    probe = getattr(backend, "run_rope_layout_probe", None)
    if probe is None:
        return {"status": "failed", "reason": "rope_layout_probe_unavailable"}
    try:
        backend.bind_revision(revision, expected_stack=stack_identity(revision))
        raw: dict[str, Any] = probe(
            workload=store.manifest["task"]["workload"], profile=profile,
            revision=revision, attempt=task["task_id"],
        )
    except BackendError as error:
        raw = {"status": "failed", "reason": "rope_layout_probe_failed", "detail": str(error)}

    mode = store.manifest["mode"]
    status = "failed"
    reason = raw.get("reason", "rope_layout_probe_failed")
    summary = None
    if raw.get("status") == "ok":
        try:
            if raw.get("mode") != mode or raw.get("timing_boundary") != ROPE_BOUNDARY:
                raise ValueError("RoPE evidence mode or timing boundary mismatch")
            cases = [_summarize_case(case) for case in raw["cases"]]
            if not any(case["phase"] == "prefill" for case in cases):
                raise ValueError("no Prefill RoPE observations")
            if any((case["layout_source"] == "simulated") != (mode == "simulated") for case in cases):
                raise ValueError("RoPE layout source differs from Session mode")
            if mode == "real":
                observed = raw["observation"]
                identity = observed["identity"]
                outputs = {row["request_id"]: row["output_ids"] for row in observed["requests"]}
                if (identity["revision"] != revision["id"] or identity["mode"] != mode
                        or identity["environment_fingerprint"] != store.manifest["runtime"]["environment_fingerprint"]
                        or identity["workload_fingerprint"] != fingerprint(store.manifest["task"]["workload"])
                        or outputs != store.manifest["reference"] or len(observed["requests"]) != len(outputs)
                        or any(case["layout_source"] != "service_observed" for case in cases)):
                    raise ValueError("RoPE layout observation must match the owned revision and frozen reference")
            summary = {"kind": "rope_layout", "mode": mode, "boundary": ROPE_BOUNDARY, "cases": cases}
            status, reason = "ok", "rope_layout_evidence_ready"
        except (KeyError, TypeError, ValueError) as error:
            reason = "invalid_rope_layout_evidence"
            raw = {**raw, "verification_error": str(error)}

    evidence = store.put_evidence(f"evidence/review/{task['task_id']}-rope-layout.json", {
        "mode": mode, "status": status, "reason": reason, "proposal_id": params["proposal_id"],
        "revision": revision["id"], "profile_ref": profile_ref, "summary": summary, "raw": raw,
    }, producer=ROPE_ACTION)
    return {"status": status, "reason": reason, "evidence": {"rope_layout": evidence}, "summary": summary}
