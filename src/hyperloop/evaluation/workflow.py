"""Independent evaluation over a replaceable runtime adapter."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterator, Mapping

from ..backends import BackendError, RuntimeBackend, fingerprint
from .evidence import AttemptEvidence, EvidenceProblem
from .measurements import (
    METRIC_DEFINITIONS,
    EvaluationPolicy,
    benchmark_sample,
    change_ratios,
    correctness,
    numeric,
    rank_hotspots,
    summarize,
    workload_requests,
)

Record = dict[str, Any]
Operation = Callable[[AttemptEvidence, Record, Record], None]


def _positive_outcome(result: Record, reason: str = "evaluation_complete") -> None:
    result.update(status="ok", evidence_state="sufficient", acceptance="pass", reason=reason)


def _failed_correctness(result: Record, reason: str) -> None:
    result.update(
        status="ok", evidence_state="sufficient", acceptance="fail", version_action="REVERT", reason=reason
    )


class Evaluator:
    """Own comparison policy and raw references separately from candidate execution."""

    def __init__(self, backend: RuntimeBackend, workspace_root: Path) -> None:
        self.backend = backend
        self.workspace_root = Path(workspace_root)

    def baseline(
        self,
        revision: str,
        workload: Mapping[str, Any],
        reference: Mapping[str, list[float]],
        policy: Mapping[str, Any],
        *,
        attempt_id: str = "baseline",
    ) -> Record:
        return self.measure_revision(revision, workload, reference, policy, attempt_id=attempt_id)

    def measure_revision(
        self,
        revision: str,
        workload: Mapping[str, Any],
        reference: Mapping[str, list[float]],
        policy: Mapping[str, Any],
        *,
        attempt_id: str,
        expected_stack: Mapping[str, Any] | None = None,
    ) -> Record:
        """Measure one independently validated revision for baseline, CLOSE, or SWEEP."""
        frozen = copy.deepcopy(dict(workload))
        trusted = self._freeze_reference(reference, frozen)
        rules = EvaluationPolicy.load(policy)

        def operation(attempt: AttemptEvidence, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "revision", "framework", expected_stack) as instance:
                result["runtime"] = instance
                verified = self._verify(attempt, context, instance, trusted, rules, "revision", "service")
                result["correctness"] = verified
                if not verified["pass"]:
                    _failed_correctness(result, "service_correctness_failed")
                    result["version_action"] = "HOLD"
                    return
                result["measurement"] = self._measure(attempt, context, instance, rules, "revision")
            _positive_outcome(result)

        return self._run(frozen, attempt_id, {"verify", "benchmark"}, operation, rules, trusted)

    def kernel(
        self,
        revision: str,
        workload: Mapping[str, Any],
        reference: Mapping[str, list[float]],
        policy: Mapping[str, Any],
        *,
        attempt_id: str,
        expected_stack: Mapping[str, Any] | None = None,
    ) -> Record:
        """Verify and time the operator at the device boundary only."""
        frozen = copy.deepcopy(dict(workload))
        trusted = self._freeze_reference(reference, frozen)
        rules = EvaluationPolicy.load(policy)

        def operation(attempt: AttemptEvidence, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "kernel", "kernel", expected_stack) as instance:
                result["runtime"] = instance
                if not self._check_kernel(attempt, context, instance, trusted, rules, result):
                    return
            _positive_outcome(result, "kernel_evaluation_complete")

        return self._run(frozen, attempt_id, {"verify", "kernel"}, operation, rules, trusted)

    def candidate(
        self,
        revision: str,
        parent: str,
        workload: Mapping[str, Any],
        reference: Mapping[str, list[float]],
        policy: Mapping[str, Any],
        *,
        attempt_id: str,
        baseline: Mapping[str, Any] | None = None,
        kind: str = "framework",
        kernel_reference: Mapping[str, list[float]] | None = None,
        expected_stack: Mapping[str, Any] | None = None,
        expected_parent_stack: Mapping[str, Any] | None = None,
    ) -> Record:
        """Validate candidate outputs, then make a fresh same-workload parent pair."""
        if revision == parent:
            raise ValueError("candidate revision must differ from its parent")
        frozen = copy.deepcopy(dict(workload))
        trusted = self._freeze_reference(reference, frozen)
        trusted_kernel = trusted if kernel_reference is None else self._freeze_reference(kernel_reference, frozen)
        rules = EvaluationPolicy.load(policy)
        required = {"verify", "benchmark"}
        if kind == "kernel":
            required.add("kernel")

        def operation(attempt: AttemptEvidence, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "candidate", kind, expected_stack) as instance:
                result["runtime"] = instance
                if kind == "kernel" and not self._check_kernel(attempt, context, instance, trusted_kernel, rules, result):
                    return
                verified = self._verify(attempt, context, instance, trusted, rules, "candidate", "service")
                result["correctness"] = verified
                if not verified["pass"]:
                    _failed_correctness(result, "service_correctness_failed")
                    return
                result["measurement"] = self._measure(attempt, context, instance, rules, "candidate")
            with self._service(attempt, context, parent, "parent", "framework", expected_parent_stack) as instance:
                result["parent_runtime"] = instance
                verified = self._verify(attempt, context, instance, trusted, rules, "parent", "service")
                result["parent_correctness"] = verified
                if not verified["pass"]:
                    raise EvidenceProblem("parent_correctness_failed", "incomparable")
                result["parent_measurement"] = self._measure(attempt, context, instance, rules, "parent")
            self._decision(attempt, result, baseline, rules)

        return self._run(frozen, attempt_id, required, operation, rules, trusted)

    def profile(
        self,
        revision: str,
        workload: Mapping[str, Any],
        *,
        attempt_id: str,
        expected_stack: Mapping[str, Any] | None = None,
    ) -> Record:
        """Capture and parse a separate raw trace for every expected rank."""
        frozen = copy.deepcopy(dict(workload))

        def operation(attempt: AttemptEvidence, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "profile", "framework", expected_stack) as instance:
                result["runtime"] = instance
                raw = self.backend.collect_profile(copy.deepcopy(instance), copy.deepcopy(frozen))
                attempt.save("profile_raw", raw, "profiler")
                self._assert_observation(raw, instance, context)
                self._parse_profile(attempt, context, revision, raw, result)
            _positive_outcome(result, "profile_complete")

        return self._run(frozen, attempt_id, {"profile"}, operation)

    @staticmethod
    def _freeze_reference(
        reference: Mapping[str, list[float]], workload: Mapping[str, Any]
    ) -> dict[str, list[float]]:
        if reference.keys() != workload_requests(workload).keys():
            raise ValueError("reference must cover exactly the frozen workload requests")
        for expected in reference.values():
            if not isinstance(expected, list) or not expected or not all(numeric(value) for value in expected):
                raise ValueError("reference outputs must be nonempty lists of finite numbers")
        return copy.deepcopy(dict(reference))

    def _run(
        self,
        workload: Record,
        attempt_id: str,
        required: set[str],
        operation: Operation,
        rules: EvaluationPolicy | None = None,
        reference: Mapping[str, Any] | None = None,
    ) -> Record:
        requests = workload_requests(workload)
        workload_fingerprint = fingerprint(workload)
        attempt = AttemptEvidence(self.workspace_root, attempt_id)
        attempt.save("workload", workload, "evaluation")
        if rules is not None:
            attempt.save("policy", asdict(rules), "evaluation")
        if reference is not None:
            attempt.save("reference", reference, "evaluation")
        result: Record = {
            "attempt_id": attempt_id, "status": "failed", "evidence_state": "missing",
            "acceptance": "undetermined", "version_action": "HOLD", "reason": "no_evaluation",
        }
        try:
            probe = self.backend.probe()
            attempt.save("probe", probe, "runtime_backend")
            self._verify_probe(probe, required)
            result["mode"] = probe["mode"]
            context = {
                "probe": probe, "mode": probe["mode"], "requests": requests, "workload": workload,
                "environment_fingerprint": probe["environment_fingerprint"],
                "workload_fingerprint": workload_fingerprint,
            }
            operation(attempt, context, result)
        except EvidenceProblem as exc:
            result.update(
                status=exc.status, evidence_state=exc.state, acceptance="undetermined",
                version_action="HOLD", reason=exc.reason,
            )
        except BackendError as exc:
            attempt.save("backend_failure", {"message": str(exc)}, "runtime_backend")
            result.update(status="failed", evidence_state="missing", reason="backend_failure")
        decision_fields = ("status", "evidence_state", "acceptance", "version_action", "reason", "comparison")
        attempt.save("decision", {name: result[name] for name in decision_fields if name in result}, "evaluation")
        result["evidence"] = copy.deepcopy(attempt.records)
        return result

    @staticmethod
    def _verify_probe(probe: Any, required: set[str]) -> None:
        if not isinstance(probe, dict) or probe.get("mode") not in {"simulated", "real"}:
            raise EvidenceProblem("invalid_runtime_mode")
        environment = probe.get("environment")
        if not isinstance(environment, dict) or probe.get("environment_fingerprint") != fingerprint(environment):
            raise EvidenceProblem("environment_probe_conflict")
        tp_size = environment.get("tp_size")
        if type(tp_size) is not int or tp_size < 1 or environment.get("mode") != probe["mode"]:
            raise EvidenceProblem("environment_probe_conflict")
        rank_map = environment.get("rank_map")
        devices = environment.get("devices")
        if not isinstance(rank_map, dict) or rank_map.keys() != {str(i) for i in range(tp_size)}:
            raise EvidenceProblem("environment_ranks_missing", "missing", "blocked")
        ranks = list(rank_map.values())
        if not isinstance(devices, list) or any(not isinstance(device, str) or device not in devices for device in ranks):
            raise EvidenceProblem("environment_probe_conflict")
        if len(set(ranks)) != len(ranks):
            raise EvidenceProblem("environment_probe_conflict")
        needed = required | ({"hccl"} if tp_size > 1 else set())
        capabilities = probe.get("capabilities")
        if not isinstance(capabilities, list) or any(not isinstance(value, str) for value in capabilities):
            raise EvidenceProblem("runtime_capabilities_invalid")
        missing = sorted(needed - set(capabilities))
        if missing:
            raise EvidenceProblem("capabilities_missing:" + ",".join(missing), "missing", "blocked")

    @contextmanager
    def _service(
        self,
        attempt: AttemptEvidence,
        context: Record,
        revision: str,
        label: str,
        kind: str,
        expected_stack: Mapping[str, Any] | None,
    ) -> Iterator[Record]:
        prepared = self.backend.prepare(revision, attempt.workspace(label), kind=kind)
        instance = None
        try:
            attempt.save(f"{label}_prepared", prepared, "runtime_backend")
            instance = self.backend.launch(copy.deepcopy(prepared), copy.deepcopy(context["workload"]))
            attempt.save(f"{label}_launch", instance, "runtime_backend")
            self._assert_instance(instance, context, revision, expected_stack)
            yield instance
        finally:
            try:
                cleanup = self.backend.cleanup(copy.deepcopy(prepared), copy.deepcopy(instance))
            except BackendError as exc:
                cleanup = {"released": False, "reason": str(exc)}
            attempt.save(f"{label}_cleanup", cleanup, "runtime_backend")
            if not isinstance(cleanup, dict) or cleanup.get("released") is not True:
                raise EvidenceProblem("cleanup_unconfirmed", "missing", "blocked")

    @staticmethod
    def _assert_instance(
        instance: Any,
        context: Record,
        revision: str,
        expected_stack: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(instance, dict) or instance.get("ready") is not True or not instance.get("instance_id"):
            raise EvidenceProblem("service_not_ready", "missing")
        if instance.get("revision") != revision or instance.get("loaded_revision") != revision:
            raise EvidenceProblem("loaded_revision_mismatch")
        if instance.get("candidate_hit") is not True or not isinstance(instance.get("loaded_stack"), dict):
            raise EvidenceProblem("candidate_load_unconfirmed", "missing")
        if expected_stack is not None and instance["loaded_stack"] != dict(expected_stack):
            raise EvidenceProblem("loaded_stack_mismatch")
        if instance.get("rank_map") != context["probe"]["environment"]["rank_map"]:
            raise EvidenceProblem("loaded_ranks_mismatch", "incomparable")
        for key in ("mode", "environment_fingerprint", "workload_fingerprint"):
            if instance.get(key) != context[key]:
                raise EvidenceProblem(f"runtime_{key}_mismatch", "incomparable")

    @staticmethod
    def _assert_observation(raw: Any, instance: Record, context: Record) -> None:
        if not isinstance(raw, dict):
            raise EvidenceProblem("observation_invalid")
        if raw.get("revision") != instance["loaded_revision"] or raw.get("instance_id") != instance["instance_id"]:
            raise EvidenceProblem("observation_identity_mismatch")
        for key in ("mode", "environment_fingerprint", "workload_fingerprint"):
            if raw.get(key) != context[key]:
                raise EvidenceProblem(f"observation_{key}_mismatch", "incomparable")

    def _verify(
        self,
        attempt: AttemptEvidence,
        context: Record,
        instance: Record,
        reference: Mapping[str, list[float]],
        rules: EvaluationPolicy,
        label: str,
        scope: str,
    ) -> Record:
        raw = self.backend.observe_outputs(copy.deepcopy(instance), copy.deepcopy(context["workload"]), scope=scope)
        attempt.save(f"{label}_{scope}_outputs", raw, "runtime_backend")
        self._assert_observation(raw, instance, context)
        if raw.get("scope") != scope:
            raise EvidenceProblem("correctness_scope_mismatch", "incomparable")
        verified = correctness(raw, context["requests"], reference, rules.tolerance,
                               context["workload"].get("dtype", "float32"))
        attempt.save(f"{label}_{scope}_correctness", verified, "verifier")
        return verified

    def _check_kernel(
        self,
        attempt: AttemptEvidence,
        context: Record,
        instance: Record,
        reference: Mapping[str, list[float]],
        rules: EvaluationPolicy,
        result: Record,
    ) -> bool:
        attempt.save("kernel_reference", reference, "evaluation")
        result["kernel_correctness"] = self._verify(attempt, context, instance, reference, rules, "operator", "kernel")
        if not result["kernel_correctness"]["pass"]:
            _failed_correctness(result, "kernel_correctness_failed")
            return False
        raw = self.backend.device_timer(copy.deepcopy(instance), copy.deepcopy(context["workload"]))
        attempt.save("device_timing_raw", raw, "runtime_backend")
        self._assert_observation(raw, instance, context)
        samples = raw.get("samples_us")
        if raw.get("boundary") != "kernel_device":
            raise EvidenceProblem("device_timing_boundary_mismatch", "incomparable")
        if not isinstance(samples, list) or not samples or any(not numeric(value) or value <= 0 for value in samples):
            raise EvidenceProblem("device_timing_missing", "missing", "partial")
        result["device_timing"] = {"boundary": "kernel_device", "samples_us": samples, "median_us": median(samples)}
        attempt.save("device_timing", result["device_timing"], "evaluation")
        return True

    def _measure(
        self, attempt: AttemptEvidence, context: Record, instance: Record, rules: EvaluationPolicy, label: str
    ) -> Record:
        samples = []
        for repeat in range(rules.repeats):
            raw = self.backend.benchmark(copy.deepcopy(instance), copy.deepcopy(context["workload"]), repeat=repeat, warmup=rules.warmup)
            attempt.save(f"{label}_repeat_{repeat}", raw, "runner")
            if raw is None:
                raise EvidenceProblem("timing_samples_missing", "missing", "partial")
            self._assert_observation(raw, instance, context)
            samples.append(benchmark_sample(raw, context["requests"], repeat, rules.warmup))
        protocol = {"repeats": rules.repeats, "warmup": rules.warmup, "metrics": dict(METRIC_DEFINITIONS)}
        measured = {
            "mode": context["mode"], "revision": instance["loaded_revision"], "loaded_stack": instance["loaded_stack"],
            "environment_fingerprint": context["environment_fingerprint"],
            "workload_fingerprint": context["workload_fingerprint"], "boundary": "service_end_to_end",
            "protocol": protocol, "protocol_fingerprint": fingerprint(protocol),
            "samples": samples, "summary": summarize(samples),
        }
        attempt.save(f"{label}_measurement", measured, "evaluation")
        return measured

    @staticmethod
    def _decision(
        attempt: AttemptEvidence, result: Record, baseline: Mapping[str, Any] | None, rules: EvaluationPolicy
    ) -> None:
        if baseline is None:
            raise EvidenceProblem("original_baseline_missing", "missing", "partial")
        original = baseline.get("measurement", baseline)
        if not isinstance(original, Mapping):
            raise EvidenceProblem("original_baseline_missing", "missing", "partial")
        result["comparison"] = {
            "parent_revision": result["parent_measurement"]["revision"],
            "baseline_revision": original.get("revision"),
            "over_parent": change_ratios(result["measurement"], result["parent_measurement"]),
            "over_original": change_ratios(result["measurement"], original),
        }
        attempt.save("paired_comparison", result["comparison"], "evaluation")
        changes = result["comparison"]["over_parent"]
        if changes["ttft_pct"] > rules.max_ttft_regression_pct:
            reason = "ttft_regression"
        elif changes["tpot_pct"] > rules.max_tpot_regression_pct:
            reason = "tpot_regression"
        elif changes["throughput_pct"] <= rules.noise_floor_pct:
            reason = "gain_within_noise"
        elif changes["throughput_pct"] < rules.min_gain_pct:
            reason = "gain_below_threshold"
        else:
            _positive_outcome(result, "gates_passed")
            result["version_action"] = "KEEP"
            return
        result.update(status="ok", evidence_state="sufficient", acceptance="fail", version_action="REVERT", reason=reason)

    @staticmethod
    def _parse_profile(attempt: AttemptEvidence, context: Record, revision: str, raw: Record, result: Record) -> None:
        rank_rows = raw.get("ranks")
        if raw.get("boundary") != "profile_capture" or not isinstance(rank_rows, list):
            raise EvidenceProblem("profile_boundary_mismatch", "incomparable")
        expected = {int(rank) for rank in context["probe"]["environment"]["rank_map"]}
        ranks: set[int] = set()
        hotspots: list[Record] = []
        for row in rank_rows:
            rank = row.get("rank") if isinstance(row, dict) else None
            if type(rank) is not int or rank not in expected or rank in ranks:
                raise EvidenceProblem("profile_rank_conflict")
            attempt.save(f"profile_rank_{rank}", row, "profiler")
            if row.get("revision") != revision:
                raise EvidenceProblem("profile_rank_revision_mismatch")
            ranks.add(rank)
            hotspots.extend(rank_hotspots(row))
        result.update(ranks=sorted(ranks), missing_ranks=sorted(expected - ranks), hotspots=hotspots,
                      revision=revision, workload_fingerprint=context["workload_fingerprint"],
                      environment_fingerprint=context["environment_fingerprint"])
        attempt.save("profile_analysis", {key: result[key] for key in ("ranks", "missing_ranks", "hotspots")}, "trace_analysis")
        if result["missing_ranks"]:
            raise EvidenceProblem("profile_ranks_missing", "missing", "partial")
