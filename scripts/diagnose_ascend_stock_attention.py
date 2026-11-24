#!/usr/bin/env python3
"""Compare SGLang's two prebuilt Ascend Attention modes at one frozen service workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from hyperloop.__main__ import _reference_identity
from hyperloop.actions.candidates import candidate_revision, original_revision, stack_identity
from hyperloop.backends.factory import checked_probe
from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang import SGLangBackend
from hyperloop.contracts import ContractError, read_json, validate_reference, validate_target, validate_task
from hyperloop.evaluation import Evaluator
from hyperloop.session import atomic_write


def _prefill_attention(profile: dict) -> dict[str, list[dict]]:
    matches: dict[str, list[dict]] = {"default_flash_attention": [], "fused_infer_attention": []}
    for row in profile.get("hotspots", []):
        if row.get("phase") != "prefill":
            continue
        name = re.sub(r"[^a-z]", "", (row.get("name", "") + " " + row.get("operator", "")).casefold())
        if "fusedinferattention" in name:
            matches["fused_infer_attention"].append(row)
        elif "flashattention" in name:
            matches["default_flash_attention"].append(row)
    return matches


def _prefill_workload(workload: dict) -> dict:
    # An output length of one completes on Prefill's first token. Mixed captures
    # could otherwise include the same prebuilt operator from graph Decode.
    requests = [{**case, "output_tokens": 1} for case in workload["requests"] if case["phase"] == "prefill"]
    if not requests:
        raise ContractError("the frozen task has no Prefill request for operator identification")
    return {**workload, "id": f"{workload['id']}-operator-prefill", "requests": requests}


def _run(args: argparse.Namespace) -> dict:
    task = validate_task(read_json(args.task))
    target = validate_target(read_json(args.target))
    reference = validate_reference(read_json(args.reference), task)
    receipt = read_json(args.reference_capture)
    identity = _reference_identity(task, target, reference)
    if receipt.get("mode") != "real" or any(receipt.get(key) != value for key, value in identity.items()):
        raise ContractError("the diagnostic requires a reference captured on the frozen target and workload")
    if target.get("mode") != "real" or target.get("sglang", {}).get("ascend_use_fia") is not False:
        raise ContractError("the diagnostic requires a real SGLang target with an explicit false initial ASCEND_USE_FIA")

    runtime = SGLangBackend(target)
    probe = checked_probe(runtime, target, mode="real")
    if probe["environment_fingerprint"] != receipt.get("environment_fingerprint"):
        raise ContractError("the measured environment differs from the frozen reference")
    if "profile" not in probe["capabilities"]:
        raise ContractError("the diagnostic requires the real SGLang Profiler to identify the selected operators")

    original = original_revision(task["initial_revision"])
    parent = candidate_revision(read_json(args.common_patch), original, "patch") if args.common_patch else original
    spec = {
        "id": "stock-fused-attention", "parameters": {"ascend_use_fia": True},
        "reason": "Test SGLang's existing Ascend fused-inference Attention route on the same workload.",
        "expected_metric": "Service throughput and latency", "risk": "The alternate implementation may change outputs",
        "validation": "Compare frozen output references, repeat service timing, then collect separate prefill profiles.",
    }
    candidate = candidate_revision(spec, parent, "parameters")
    for revision in (original, parent, candidate):
        runtime.bind_revision(revision, expected_stack=stack_identity(revision))
    workload = task["workload"]
    operator_workload = _prefill_workload(workload)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime.bind_session(output)
    evaluator = Evaluator(runtime, output)
    rules = task["rules"]
    result = {"mode": "real", "scope": "stock_attention_service_diagnostic", "status": "running",
              "probe": probe, "workload": workload, "operator_capture_workload": operator_workload,
              "policy": rules, "reference_identity": identity,
              "revisions": {"original": original, "parent": parent, "candidate": candidate}}

    def save() -> None:
        atomic_write(output / "stock_attention_diagnostic.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())

    save()
    baseline = evaluator.measure_revision(original["id"], workload, reference, rules, attempt_id="service-original",
                                          expected_stack=stack_identity(original))
    result["baseline"] = baseline
    save()
    if baseline["acceptance"] != "pass":
        result.update(status="blocked", reason="baseline_failed")
    else:
        comparison = evaluator.candidate(candidate["id"], parent["id"], workload, reference, rules,
                                         attempt_id="service-pair", baseline=baseline,
                                         expected_stack=stack_identity(candidate), expected_parent_stack=stack_identity(parent))
        result["service_comparison"] = comparison
        save()
        if comparison["status"] != "ok":
            result.update(status="blocked", reason="service_pair_incomplete")
        else:
            before = evaluator.profile(parent["id"], operator_workload, attempt_id="operators-default",
                                       expected_stack=stack_identity(parent))
            result["profile_default"] = before
            save()
            if before["status"] != "ok":
                result.update(status="blocked", reason="default_profile_incomplete")
            else:
                after = evaluator.profile(candidate["id"], operator_workload, attempt_id="operators-fused",
                                          expected_stack=stack_identity(candidate))
                result["profile_fused"] = after
                matches_before = _prefill_attention(before)
                matches_after = _prefill_attention(after)
                result["prefill_operator_matches"] = {"default": matches_before, "fused": matches_after}
                confirmed = bool(after["status"] == "ok" and matches_before["default_flash_attention"]
                                 and matches_after["fused_infer_attention"] and not matches_before["fused_infer_attention"])
                result.update(status="complete" if confirmed else "inconclusive",
                              reason="stock_operator_switch_observed" if confirmed else "prefill_operator_switch_unconfirmed")
    save()
    return {"status": result["status"], "scope": result["scope"], "reason": result["reason"],
            "diagnostic": str(output / "stock_attention_diagnostic.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "target", "reference", "reference-capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--common-patch", type=Path, help="An approved source candidate shared by both compared arms")
    try:
        result = _run(parser.parse_args())
    except (BackendError, ContractError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
