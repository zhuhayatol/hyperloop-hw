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
