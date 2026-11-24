"""Reconstruct the measured Critic verdict from immutable Session evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from statistics import median
from typing import Any

from ..backends import fingerprint
from ..contracts import ContractError
from ..session import SessionStore
from .evidence import EvidenceProblem
from .measurements import (
    METRIC_DEFINITIONS,
    EvaluationPolicy,
    benchmark_sample,
    change_ratios,
    correctness,
    numeric,
    summarize,
    workload_requests,
)
from .workflow import Evaluator

Record = dict[str, Any]


def _unique_keys(pairs: list[tuple[str, Any]]) -> Record:
    record: Record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate key: {key}")
        record[key] = value
    return record


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


class _AttemptProof:
    """Bind each raw artifact to its attempt, operation and original bytes."""

    def __init__(self, store: SessionStore, result: Mapping[str, Any]):
        self.store = store
        self.attempt_id = result.get("attempt_id")
        self.references = result.get("evidence")
        self.verified: dict[str, str] = {}
        if not isinstance(self.attempt_id, str) or not isinstance(self.references, dict):
            raise EvidenceProblem("adjudication_attempt_missing", "missing", "partial")

    def read(self, kind: str, producer: str) -> Record:
        reference = self.references.get(kind)
        if reference is None:
            raise EvidenceProblem(f"adjudication_evidence_missing:{kind}", "missing", "partial")
        path = f"evaluation/{self.attempt_id}/{kind}.json"
        if (
            not isinstance(reference, dict)
            or reference.get("producer") != producer
            or reference.get("path") != path
            or reference.get("id") != f"{self.attempt_id}:{kind}"
        ):
            raise EvidenceProblem(f"adjudication_evidence_identity:{kind}")
        try:
            content = self.store.artifact_path(path).read_bytes()
        except FileNotFoundError as exc:
            raise EvidenceProblem(f"adjudication_evidence_missing:{kind}", "missing", "partial") from exc
        except (ContractError, OSError) as exc:
            raise EvidenceProblem(f"adjudication_evidence_unreadable:{kind}") from exc
        if hashlib.sha256(content).hexdigest() != reference.get("sha256"):
            raise EvidenceProblem(f"adjudication_evidence_digest:{kind}")
        try:
            record = json.loads(content, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
        except (UnicodeError, ValueError) as exc:
            raise EvidenceProblem(f"adjudication_evidence_invalid:{kind}") from exc
        if record is None:
            raise EvidenceProblem(f"adjudication_evidence_missing:{kind}", "missing", "partial")
        if not isinstance(record, dict):
            raise EvidenceProblem(f"adjudication_evidence_invalid:{kind}")
        self.verified[kind] = reference["id"]
        return record


def _context(
    proof: _AttemptProof, manifest: Record, rules: EvaluationPolicy, required: set[str], reference: Record
) -> Record:
    workload = manifest["task"]["workload"]
    if proof.read("workload", "evaluation") != workload:
        raise EvidenceProblem("adjudication_workload_mismatch", "incomparable")
    if proof.read("policy", "evaluation") != asdict(rules):
        raise EvidenceProblem("adjudication_policy_mismatch", "incomparable")
    if proof.read("reference", "evaluation") != reference:
        raise EvidenceProblem("adjudication_reference_mismatch")
    probe = proof.read("probe", "runtime_backend")
    Evaluator._verify_probe(probe, required)
    if probe["mode"] != manifest["mode"]:
        raise EvidenceProblem("adjudication_session_mode_mismatch", "incomparable")
    target = manifest.get("target", {})
    environment = probe["environment"]
    for declared, observed in (("id", "target_id"), ("devices", "devices"), ("tp_size", "tp_size")):
        if declared in target and target[declared] != environment.get(observed):
            raise EvidenceProblem(f"adjudication_target_{declared}_mismatch", "incomparable")
    if any(environment.get("software", {}).get(key) != value for key, value in target.get("software", {}).items()):
        raise EvidenceProblem("adjudication_target_software_mismatch", "incomparable")
    return {
        "mode": manifest["mode"],
        "probe": probe,
        "workload": workload,
        "workload_fingerprint": fingerprint(workload),
        "environment_fingerprint": probe["environment_fingerprint"],
        "requests": workload_requests(workload),
    }


def _revision_id(revision: Any, field: str) -> str:
    identifier = revision.get("id") if isinstance(revision, dict) else None
    if not isinstance(identifier, str) or not identifier:
        raise EvidenceProblem(f"adjudication_revision_missing:{field}", "missing", "partial")
    return identifier


def _instance(
    proof: _AttemptProof, context: Record, revision: Record, arm: str, kind: str
) -> Record:
    identifier = _revision_id(revision, arm)
    if not {"framework", "kernels", "parameters"} <= revision.keys():
        raise EvidenceProblem(f"adjudication_loaded_stack_missing:{arm}", "missing", "partial")
    expected_stack = {part: revision[part] for part in ("framework", "kernels", "parameters")}
    prepared = proof.read(f"{arm}_prepared", "runtime_backend")
    launch = proof.read(f"{arm}_launch", "runtime_backend")
    if (
        prepared.get("revision") != identifier
        or prepared.get("mode") != context["mode"]
        or prepared.get("kind") != kind
        or not prepared.get("workspace")
        or prepared["workspace"] != launch.get("workspace")
    ):
        raise EvidenceProblem(f"adjudication_workspace_mismatch:{arm}")
    Evaluator._assert_instance(launch, context, identifier, expected_stack)
    if not isinstance(launch["instance_id"], str):
        raise EvidenceProblem(f"adjudication_instance_identity_invalid:{arm}")
    cleanup = proof.read(f"{arm}_cleanup", "runtime_backend")
    if cleanup.get("released") is not True:
        raise EvidenceProblem("cleanup_unconfirmed", "missing", "blocked")
    return launch


def _verify(
    proof: _AttemptProof,
    context: Record,
    instance: Record,
    rules: EvaluationPolicy,
    reference: Record,
    arm: str,
    scope: str = "service",
) -> Record:
    raw = proof.read(f"{arm}_{scope}_outputs", "runtime_backend")
    Evaluator._assert_observation(raw, instance, context)
    if raw.get("scope") != scope:
        raise EvidenceProblem("adjudication_correctness_scope_mismatch", "incomparable")
    computed = correctness(raw, context["requests"], reference, rules.tolerance,
                           context["workload"].get("dtype", "float32"))
    recorded = proof.read(f"{arm}_{scope}_correctness", "verifier")
    if computed != recorded:
        raise EvidenceProblem(f"adjudication_correctness_mismatch:{arm}:{scope}")
    return computed


def native_operator_publication(store: SessionStore, parent: str, kernel: Record) -> Record:
    """Bind the isolated Forge build and test to the exact patch offered to the service."""
    reference = kernel.get("operator_evidence")
    if (not isinstance(reference, dict) or reference.get("producer") != "native_forge"
            or not store.verify_evidence(reference)):
        raise EvidenceProblem("native_forge_publication_unverified", "missing", "blocked")
    try:
        raw = store.artifact_path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("publication changed after verification")
        publication = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
        spec = publication["candidate"]
        operator = publication["evidence"]["operator"]
        supported = spec["supported_input"]
        if (publication["mode"] != "real" or publication["status"] != "ok"
                or publication["delivery"] != "operator_patch" or publication["target_revision"] != parent
                or spec["id"] != kernel["candidate_id"] or spec["parent"] != parent
                or not store.verify_evidence(publication["profile_ref"])
                or operator["status"] != "complete" or len(operator["cases"]) != 2
                or any(spec[key] != kernel[key] for key in (
                    "operator", "backend", "source_edit", "patch_path", "invocation_marker", "pinned_source_sha256"
                ))):
            raise ValueError("publication identity mismatch")
        names = set()
        for row in operator["cases"]:
            compiled = row["compiled"]
            timings = row["stream_event_us"]
            if (row["timing_status"] != "complete" or row["dtype"] != supported["dtype"]
                    or row["input_shape"][-1] != supported["width"]
                    or any(row["correctness"][key]["pass"] is not True
                           for key in ("stock_vs_cpu", "candidate_vs_cpu"))
                    or not any(item["kind"] == "npubin" for item in compiled["artifacts"])
                    or any(len(timings[key]) != operator["repeats"] or not timings[key]
                           or not all(numeric(sample) and sample > 0 for sample in timings[key])
                           or median(timings[key]) != row["median_us"][key]
                           for key in ("torch_npu_stock", "triton_candidate"))):
                raise ValueError("operator checks or samples incomplete")
            names.add(compiled["name"])
    except (OSError, ContractError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise EvidenceProblem("native_forge_publication_invalid", "incomparable", "blocked") from error
    return {"reference": reference, "compiled_names": names, "supported_input": supported}


def _verify_native_activation(proof: _AttemptProof, instance: Record, parent: str) -> Record:
    try:
        kernel = instance["loaded_stack"]["kernels"][-1]
        verified = native_operator_publication(proof.store, parent, kernel)
        observed = proof.read("candidate_service_outputs", "runtime_backend")["kernel_activation"]
        source = instance["source_overlay"]
        file = next(item for item in source["files"] if item["path"] == kernel["patch_path"])
        loaded = next(item for item in source["loaded_files"] if item["path"] == file["staged_path"])
        supported = verified["supported_input"]
        if (observed["candidate_id"] != kernel["candidate_id"] or observed["marker"] != kernel["invocation_marker"]
                or observed["kernel"] not in verified["compiled_names"]
                or not supported["tokens"][0] <= int(observed["tokens"]) <= supported["tokens"][1]
                or int(observed["width"]) * 2 != supported["width"]
                or file["pinned_sha256"] != kernel["pinned_source_sha256"]
                or observed["source_sha256"] != loaded["sha256"] or loaded["sha256"] != file["expected_sha256"]):
            raise ValueError("observed operator identity differs from the verified patch")
    except (TypeError, ValueError, IndexError, KeyError, StopIteration) as error:
        raise EvidenceProblem("native_forge_activation_invalid", "incomparable", "blocked") from error
    proof.verified["native_forge_publication"] = verified["reference"]["id"]
    return observed


def _measure(
    proof: _AttemptProof, context: Record, instance: Record, rules: EvaluationPolicy, arm: str
) -> Record:
    samples = []
    for repeat in range(rules.repeats):
        raw = proof.read(f"{arm}_repeat_{repeat}", "runner")
        Evaluator._assert_observation(raw, instance, context)
        samples.append(benchmark_sample(raw, context["requests"], repeat, rules.warmup))
    protocol = {"repeats": rules.repeats, "warmup": rules.warmup, "metrics": dict(METRIC_DEFINITIONS)}
    computed = {
        "mode": context["mode"],
        "revision": instance["loaded_revision"],
        "loaded_stack": instance["loaded_stack"],
        "environment_fingerprint": context["environment_fingerprint"],
        "workload_fingerprint": context["workload_fingerprint"],
        "boundary": "service_end_to_end",
        "protocol": protocol,
        "protocol_fingerprint": fingerprint(protocol),
        "samples": samples,
        "summary": summarize(samples),
    }
    if proof.read(f"{arm}_measurement", "evaluation") != computed:
        raise EvidenceProblem(f"adjudication_measurement_mismatch:{arm}")
    return computed


def _device(proof: _AttemptProof, context: Record, instance: Record) -> Record:
    raw = proof.read("device_timing_raw", "runtime_backend")
    Evaluator._assert_observation(raw, instance, context)
    if raw.get("boundary") != "kernel_device":
        raise EvidenceProblem("adjudication_device_boundary_mismatch", "incomparable")
    samples = raw.get("samples_us")
    if not isinstance(samples, list) or not samples or any(not numeric(value) or value <= 0 for value in samples):
        raise EvidenceProblem("adjudication_device_samples_missing", "missing", "partial")
    computed = {"boundary": "kernel_device", "samples_us": samples, "median_us": median(samples)}
    if proof.read("device_timing", "evaluation") != computed:
        raise EvidenceProblem("adjudication_device_measurement_mismatch")
    return computed


def _outcome(result: Record, reason: str | None, *, keep_allowed: bool = True) -> None:
    if reason is not None:
        result.update(status="ok", evidence_state="sufficient", acceptance="fail", version_action="REVERT",
                      reason=reason)
    elif keep_allowed:
        result.update(status="ok", evidence_state="sufficient", acceptance="pass", version_action="KEEP",
                      reason="gates_passed")
    else:
        result.update(status="ok", evidence_state="sufficient", acceptance="pass", version_action="HOLD",
                      reason="stronger_candidate_selected")


def _performance_failure(changes: Record, rules: EvaluationPolicy) -> str | None:
    if changes["ttft_pct"] > rules.max_ttft_regression_pct:
        return "ttft_regression"
    if changes["tpot_pct"] > rules.max_tpot_regression_pct:
        return "tpot_regression"
    if changes["throughput_pct"] <= rules.noise_floor_pct:
        return "gain_within_noise"
    if changes["throughput_pct"] < rules.min_gain_pct:
        return "gain_below_threshold"
    return None


def _review(
    result: Record, decision: Record, baseline: Record, proof: _AttemptProof, baseline_proof: _AttemptProof | None,
    manifest: Record, rules: EvaluationPolicy,
) -> None:
    reference = manifest["reference"]
    kernel_reference = manifest.get("scenario", {}).get("kernel_reference") or reference
    revision = decision.get("revision")
    parent_revision = decision.get("parent_revision")
    if _revision_id(revision, "candidate") != decision.get("candidate_id"):
        raise EvidenceProblem("adjudication_candidate_identity_mismatch")
    if _revision_id(parent_revision, "parent") != decision.get("parent") or revision.get("parent") != decision.get("parent"):
        raise EvidenceProblem("adjudication_parent_identity_mismatch")

    kernel = decision.get("kind") == "kernel"
    standalone_kernel = kernel and "kernel_launch" in proof.references
    standalone_service = kernel and "revision_launch" in proof.references
    required = {"verify"}
    if not standalone_kernel:
        required.add("benchmark")
    if kernel and not standalone_service:
        required.add("kernel")
    context = _context(proof, manifest, rules, required, kernel_reference if standalone_kernel else reference)
    arm = "kernel" if standalone_kernel else "revision" if standalone_service else "candidate"
    instance = _instance(proof, context, revision, arm, "kernel" if kernel and not standalone_service else "framework")
    result["runtime"] = instance
    if decision.get("kind") == "operator_patch":
        result["kernel_activation"] = _verify_native_activation(proof, instance, parent_revision["id"])
    if kernel and not standalone_service:
        if proof.read("kernel_reference", "evaluation") != kernel_reference:
            raise EvidenceProblem("adjudication_kernel_reference_mismatch")
        result["kernel_correctness"] = _verify(proof, context, instance, rules, kernel_reference, "operator", "kernel")
        if not result["kernel_correctness"]["pass"]:
            _outcome(result, "kernel_correctness_failed")
            return
        result["device_timing"] = _device(proof, context, instance)
    if standalone_kernel:
        raise EvidenceProblem("adjudication_integrated_service_missing", "missing", "partial")
    result["correctness"] = _verify(proof, context, instance, rules, reference, arm)
    if not result["correctness"]["pass"]:
        _outcome(result, "service_correctness_failed")
        return
    if standalone_service:
        raise EvidenceProblem("adjudication_integrated_service_missing", "missing", "partial")

    if baseline_proof is None:
        raise EvidenceProblem("adjudication_baseline_missing", "missing", "partial")
    base_context = _context(baseline_proof, manifest, rules, {"verify", "benchmark"}, reference)
    base_instance = _instance(baseline_proof, base_context, baseline.get("revision"), "revision", "framework")
    parent_instance = _instance(proof, context, parent_revision, "parent", "framework")
    geak = manifest.get("target", {}).get("sglang", {})
    if manifest["mode"] == "real" and decision.get("backend") == "geak" and geak.get("native_geak") is True:
        parent_fia = parent_revision["parameters"].get("ascend_use_fia", geak.get("ascend_use_fia"))
        candidate_fia = revision["parameters"].get("ascend_use_fia")
        if type(parent_fia) is not bool or type(candidate_fia) is not bool or candidate_fia == parent_fia or any(
            arm.get("stock_attention_launch_control") != {"ASCEND_USE_FIA": "1" if enabled else "0", "ASCEND_USE_FA": "0"}
            for arm, enabled in ((instance, candidate_fia), (parent_instance, parent_fia))
        ):
            raise EvidenceProblem("native_geak_attention_setting_mismatch", "incomparable", "blocked")
    if len({item["instance_id"] for item in (instance, parent_instance, base_instance)}) != 3:
        raise EvidenceProblem("adjudication_measurement_instances_overlap")
    if not _verify(proof, context, parent_instance, rules, reference, "parent")["pass"]:
        raise EvidenceProblem("parent_correctness_failed", "incomparable")
    if not _verify(baseline_proof, base_context, base_instance, rules, reference, "revision")["pass"]:
        raise EvidenceProblem("adjudication_baseline_correctness_failed", "incomparable")
    measured = _measure(proof, context, instance, rules, "candidate")
    parent = _measure(proof, context, parent_instance, rules, "parent")
    original = _measure(baseline_proof, base_context, base_instance, rules, "revision")
    comparison = {
        "parent_revision": parent["revision"],
        "baseline_revision": original["revision"],
        "over_parent": change_ratios(measured, parent),
        "over_original": change_ratios(measured, original),
    }
    if proof.read("paired_comparison", "evaluation") != comparison:
        raise EvidenceProblem("adjudication_comparison_mismatch")
    result.update(measurement=measured, parent_measurement=parent, comparison=comparison)
    _outcome(result, _performance_failure(comparison["over_parent"], rules),
             keep_allowed=decision.get("reason") != "stronger_candidate_selected")


def adjudicate(store: SessionStore, decision: dict, baseline: dict, rules: dict) -> dict:
    """Apply the measured Critic gate using the frozen manifest and raw receipts.

    Promotion remains a Coordinator operation. This reader neither opens the
    candidate workspace nor changes the accepted revision.
    """
    result = dict(decision)
    proof = None
    baseline_proof = None
    basis = "verified_records"
    try:
        if (
            decision.get("status") != "ok"
            or decision.get("evidence_state") != "sufficient"
            or decision.get("acceptance") == "undetermined"
        ):
            result.update(acceptance="undetermined", version_action="HOLD")
            basis = "existing_block"
        else:
            manifest = store.manifest
            if rules != manifest["task"]["rules"]:
                raise EvidenceProblem("adjudication_locked_rules_mismatch", "incomparable")
            proof = _AttemptProof(store, decision)
            if isinstance(baseline, dict) and isinstance(baseline.get("attempt_id"), str):
                baseline_proof = _AttemptProof(store, baseline)
            _review(result, decision, baseline, proof, baseline_proof, manifest, EvaluationPolicy.load(rules))
    except EvidenceProblem as error:
        result.update(status=error.status, evidence_state=error.state, acceptance="undetermined",
                      version_action="HOLD", reason=error.reason)
        basis = "evidence_rejected"
    result["adjudication"] = {
        "source": "independent_critic",
        "basis": basis,
        "attempt_evidence": {} if proof is None else proof.verified,
        "baseline_evidence": {} if baseline_proof is None else baseline_proof.verified,
    }
    return result
