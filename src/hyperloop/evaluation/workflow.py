"""Independent evaluation over a replaceable runtime adapter."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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


@dataclass
class _TraceResponse:
    message: str
    state: str = "succeeded"


class _ValidationAttempt(AttemptEvidence):
    def __init__(self, root: Path, attempt_id: str) -> None:
        super().__init__(root, attempt_id)
        self.steps: list[Record] = []

    def step(
        self, sender: str, receiver: str, operation: str, scope: str, message: str, state: str,
        *evidence_keys: str,
    ) -> None:
        row = {"sender": sender, "receiver": receiver, "operation": operation, "scope": scope,
               "message": message, "state": state}
        keys = [key for key in dict.fromkeys(evidence_keys) if key in self.records]
        if keys:
            row["evidence_keys"] = keys
        self.steps.append(row)

    @contextmanager
    def exchange(
        self, sender: str, receiver: str, operation: str, scope: str, request: str, response: str,
        *evidence_keys: str,
    ) -> Iterator[_TraceResponse]:
        self.step(sender, receiver, operation, scope, request, "requested", *evidence_keys)
        before = len(self.records)
        receipt = _TraceResponse(response)
        try:
            yield receipt
        except (BackendError, EvidenceProblem) as exc:
            reason = exc.reason if isinstance(exc, EvidenceProblem) else "backend_failure"
            self.step(receiver, sender, operation, scope, f"执行未完成（{reason}）", "failed",
                      *evidence_keys, *list(self.records)[before:])
            raise
        self.step(receiver, sender, operation, scope, receipt.message, receipt.state,
                  *evidence_keys, *list(self.records)[before:])


Operation = Callable[[_ValidationAttempt, Record, Record], None]


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

        def operation(attempt: _ValidationAttempt, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "revision", "framework", expected_stack) as instance:
                result["runtime"] = instance
                verified = self._verify(attempt, context, instance, trusted, rules, "revision", "service")
                result["correctness"] = verified
                if not verified["pass"]:
                    attempt.step("Evaluator", "BenchmarkRunner", "benchmark_sample", "revision",
                                 "输出校验未通过，跳过端到端计时", "skipped", "revision_service_correctness")
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

        def operation(attempt: _ValidationAttempt, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "kernel", "kernel", expected_stack) as instance:
                result["runtime"] = instance
                if not self._check_kernel(attempt, context, instance, trusted, rules, result):
                    return
            _positive_outcome(result, "kernel_evaluation_complete")

        return self._run(frozen, attempt_id, {"verify", "kernel"}, operation, rules, trusted)

    def forward(
        self,
        revision: str,
        workload: Mapping[str, Any],
        reference: Mapping[str, list[float]],
        policy: Mapping[str, Any],
        *,
        attempt_id: str,
        expected_stack: Mapping[str, Any] | None = None,
    ) -> Record:
        """Record the optional model-forward timing at its own boundary."""
        frozen = copy.deepcopy(dict(workload))
        trusted = self._freeze_reference(reference, frozen)
        rules = EvaluationPolicy.load(policy)

        def operation(attempt: _ValidationAttempt, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "forward", "model_forward", expected_stack) as instance:
                verified = self._verify(attempt, context, instance, trusted, rules, "forward", "service")
                result["correctness"] = verified
                if not verified["pass"]:
                    attempt.step("Evaluator", "RuntimeBackend", "forward_timer", "forward",
                                 "输出校验未通过，跳过模型 forward 计时", "skipped", "forward_service_correctness")
                    _failed_correctness(result, "forward_correctness_failed")
                    result["version_action"] = "HOLD"
                    return
                with attempt.exchange("Evaluator", "RuntimeBackend", "forward_timer", "forward",
                                      "按模型 forward 边界请求计时", "返回模型 forward 原始样本",
                                      "forward_service_correctness"):
                    raw = self.backend.forward_timer(copy.deepcopy(instance), copy.deepcopy(frozen))
                    attempt.save("forward_raw", raw, "model_timer")
                with attempt.exchange("Evaluator", "BenchmarkRunner", "forward_timing_summary", "forward",
                                      "核对 forward 计时边界和来源", "返回模型 forward 计时汇总", "forward_raw"):
                    self._assert_observation(raw, instance, context)
                    samples = raw.get("samples_ms")
                    if (
                        raw.get("boundary") != "model_forward" or not isinstance(samples, list)
                        or not samples or not all(numeric(item) and item > 0 for item in samples)
                    ):
                        raise EvidenceProblem("invalid_forward_timing", "incomparable")
                    timing = {"boundary": "model_forward", "samples_ms": samples, "median_ms": median(samples)}
                    result["forward_timing"] = timing
                    attempt.save("forward_timing", timing, "evaluation")
            _positive_outcome(result, "forward_evaluation_complete")

        return self._run(frozen, attempt_id, {"verify", "forward"}, operation, rules, trusted)

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

        def operation(attempt: _ValidationAttempt, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "candidate", kind, expected_stack) as instance:
                result["runtime"] = instance
                if kind == "kernel" and not self._check_kernel(attempt, context, instance, trusted_kernel, rules, result):
                    attempt.step("Evaluator", "Verifier", "correctness", "candidate",
                                 "算子校验未通过，跳过候选服务校验", "skipped", "operator_kernel_correctness")
                    attempt.step("Evaluator", "BenchmarkRunner", "benchmark_sample", "candidate",
                                 "算子校验未通过，跳过端到端计时", "skipped", "operator_kernel_correctness")
                    attempt.step("Evaluator", "RuntimeBackend", "prepare", "parent",
                                 "候选未通过校验，跳过父版本配对", "skipped", "operator_kernel_correctness")
                    return
                verified = self._verify(attempt, context, instance, trusted, rules, "candidate", "service")
                result["correctness"] = verified
                if not verified["pass"]:
                    attempt.step("Evaluator", "BenchmarkRunner", "benchmark_sample", "candidate",
                                 "输出校验未通过，跳过端到端计时", "skipped", "candidate_service_correctness")
                    attempt.step("Evaluator", "RuntimeBackend", "prepare", "parent",
                                 "候选未通过校验，跳过父版本配对", "skipped", "candidate_service_correctness")
                    _failed_correctness(result, "service_correctness_failed")
                    return
                result["measurement"] = self._measure(attempt, context, instance, rules, "candidate")
            with self._service(attempt, context, parent, "parent", "framework", expected_parent_stack) as instance:
                result["parent_runtime"] = instance
                verified = self._verify(attempt, context, instance, trusted, rules, "parent", "service")
                result["parent_correctness"] = verified
                if not verified["pass"]:
                    attempt.step("Evaluator", "BenchmarkRunner", "benchmark_sample", "parent",
                                 "父版本输出校验未通过，跳过端到端计时", "skipped", "parent_service_correctness")
                    raise EvidenceProblem("parent_correctness_failed", "incomparable")
                result["parent_measurement"] = self._measure(attempt, context, instance, rules, "parent")
            with attempt.exchange("Evaluator", "DecisionGate", "paired_comparison", "evaluation",
                                  "按锁定门槛比较候选、父版本和原始基线", "返回版本比较和评测结果",
                                  "candidate_measurement", "parent_measurement", "policy"):
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

        def operation(attempt: _ValidationAttempt, context: Record, result: Record) -> None:
            with self._service(attempt, context, revision, "profile", "framework", expected_stack) as instance:
                result["runtime"] = instance
                with attempt.exchange("Profiler", "RuntimeBackend", "collect_profile", "profile",
                                      "采集已加载版本的各 rank trace", "返回原始 Profile 响应", "profile_launch"):
                    raw = self.backend.collect_profile(copy.deepcopy(instance), copy.deepcopy(frozen))
                    attempt.save("profile_raw", raw, "profiler")
                with attempt.exchange("Evaluator", "TraceAnalyzer", "parse_profile", "profile",
                                      "核对采集身份、窗口及各 rank trace", "返回各 rank 热点和定位依据", "profile_raw"):
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
        attempt = _ValidationAttempt(self.workspace_root, attempt_id)
        attempt.save("workload", workload, "evaluation")
        if rules is not None:
            attempt.save("policy", asdict(rules), "evaluation")
        if reference is not None:
            attempt.save("reference", reference, "evaluation")
        attempt.step("Evaluator", "EvidenceStore", "save_contract", "evaluation", "保存本次评测约定", "succeeded",
                     "workload", "policy", "reference")
        result: Record = {
            "attempt_id": attempt_id, "status": "failed", "evidence_state": "missing",
            "acceptance": "undetermined", "version_action": "HOLD", "reason": "no_evaluation",
        }
        mode = None
        try:
            with attempt.exchange("Evaluator", "RuntimeBackend", "probe", "evaluation",
                                  "探测目标设备、并行拓扑和可用能力", "返回运行环境探测记录"):
                probe = self.backend.probe()
                attempt.save("probe", probe, "runtime_backend")
                if isinstance(probe, dict) and probe.get("mode") in {"simulated", "real"}:
                    mode = probe["mode"]
            with attempt.exchange("Evaluator", "Verifier", "probe_identity", "evaluation",
                                  "核对环境指纹、rank 映射和本次所需能力", "环境预检通过", "probe", "workload"):
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
        action = result["version_action"]
        reason = result["reason"]
        attempt.step("Evaluator", "EvidenceStore", "decision", "evaluation", f"记录评测结论 {action}（{reason}）",
                     "succeeded", "decision", "paired_comparison", "backend_failure")
        attempt.save("validation_flow", {"schema_version": 1, "attempt_id": attempt_id, "mode": mode,
                                         "steps": attempt.steps}, "evaluation_trace")
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
        attempt: _ValidationAttempt,
        context: Record,
        revision: str,
        label: str,
        kind: str,
        expected_stack: Mapping[str, Any] | None,
    ) -> Iterator[Record]:
        prepared_key = f"{label}_prepared"
        launch_key = f"{label}_launch"
        attempt.step("Evaluator", "RuntimeBackend", "prepare", label, "在本次评测的隔离目录准备版本",
                     "requested", "probe")
        try:
            prepared = self.backend.prepare(revision, attempt.workspace(label), kind=kind)
        except BackendError:
            attempt.step("RuntimeBackend", "Evaluator", "prepare", label, "后端未能准备隔离版本", "failed", "probe")
            raise
        instance = None
        try:
            attempt.save(prepared_key, prepared, "runtime_backend")
            attempt.step("RuntimeBackend", "Evaluator", "prepare", label, "返回构建准备记录", "succeeded", prepared_key)
            with attempt.exchange("Evaluator", "RuntimeBackend", "launch", label,
                                  "按锁定负载启动已准备的版本", "返回服务启动记录", prepared_key, "workload"):
                instance = self.backend.launch(copy.deepcopy(prepared), copy.deepcopy(context["workload"]))
                attempt.save(launch_key, instance, "runtime_backend")
            with attempt.exchange("Evaluator", "Verifier", "instance_identity", label,
                                  "核对实际加载版本、执行栈和 rank 映射", "运行实例身份核对通过", launch_key, "probe"):
                self._assert_instance(instance, context, revision, expected_stack)
            yield instance
        finally:
            with attempt.exchange("Evaluator", "RuntimeBackend", "cleanup", label,
                                  "清理本次隔离实例和资源", "确认本次资源已释放", prepared_key, launch_key):
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
        attempt: _ValidationAttempt,
        context: Record,
        instance: Record,
        reference: Mapping[str, list[float]],
        rules: EvaluationPolicy,
        label: str,
        scope: str,
    ) -> Record:
        step_scope = "kernel" if scope == "kernel" else label
        boundary = "算子" if scope == "kernel" else "服务"
        output_key = f"{label}_{scope}_outputs"
        reference_key = "kernel_reference" if scope == "kernel" else "reference"
        with attempt.exchange("Evaluator", "RuntimeBackend", "observe_outputs", step_scope,
                              f"按锁定输入请求{boundary}输出", f"返回{boundary}原始输出", "workload"):
            raw = self.backend.observe_outputs(copy.deepcopy(instance), copy.deepcopy(context["workload"]), scope=scope)
            attempt.save(output_key, raw, "runtime_backend")
        attempt.step("Evaluator", "Verifier", "reference", step_scope, f"交付预先固化的{boundary}参考输出",
                     "succeeded", reference_key)
        with attempt.exchange("Evaluator", "Verifier", "correctness", step_scope,
                              f"核对{boundary}输出来源并按锁定容差独立比对", f"{boundary}正确性校验通过",
                              output_key, reference_key, "policy") as receipt:
            self._assert_observation(raw, instance, context)
            if raw.get("scope") != scope:
                raise EvidenceProblem("correctness_scope_mismatch", "incomparable")
            verified = correctness(raw, context["requests"], reference, rules.tolerance,
                                   context["workload"].get("dtype", "float32"))
            attempt.save(f"{label}_{scope}_correctness", verified, "verifier")
            if not verified["pass"]:
                receipt.state = "failed"
                receipt.message = f"{boundary}正确性校验未通过"
        return verified

    def _check_kernel(
        self,
        attempt: _ValidationAttempt,
        context: Record,
        instance: Record,
        reference: Mapping[str, list[float]],
        rules: EvaluationPolicy,
        result: Record,
    ) -> bool:
        attempt.save("kernel_reference", reference, "evaluation")
        result["kernel_correctness"] = self._verify(attempt, context, instance, reference, rules, "operator", "kernel")
        if not result["kernel_correctness"]["pass"]:
            attempt.step("Evaluator", "RuntimeBackend", "device_timer", "kernel",
                         "算子校验未通过，跳过设备计时", "skipped", "operator_kernel_correctness")
            _failed_correctness(result, "kernel_correctness_failed")
            return False
        with attempt.exchange("Evaluator", "RuntimeBackend", "device_timer", "kernel",
                              "请求算子设备计时", "返回算子设备计时原始样本", "operator_kernel_correctness"):
            raw = self.backend.device_timer(copy.deepcopy(instance), copy.deepcopy(context["workload"]))
            attempt.save("device_timing_raw", raw, "runtime_backend")
        with attempt.exchange("Evaluator", "BenchmarkRunner", "device_timing_summary", "kernel",
                              "独立核对设备计时边界和来源", "返回设备计时汇总", "device_timing_raw"):
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
        self, attempt: _ValidationAttempt, context: Record, instance: Record, rules: EvaluationPolicy, label: str
    ) -> Record:
        samples = []
        for repeat in range(rules.repeats):
            raw_key = f"{label}_repeat_{repeat}"
            with attempt.exchange("BenchmarkRunner", "RuntimeBackend", "benchmark_sample", label,
                                  f"请求第 {repeat + 1} 轮独立端到端计时",
                                  f"返回第 {repeat + 1} 轮端到端响应", "policy"):
                raw = self.backend.benchmark(copy.deepcopy(instance), copy.deepcopy(context["workload"]),
                                             repeat=repeat, warmup=rules.warmup)
                attempt.save(raw_key, raw, "runner")
            with attempt.exchange("Evaluator", "BenchmarkRunner", "validate_sample", label,
                                  f"核对第 {repeat + 1} 轮样本及计时边界", f"第 {repeat + 1} 轮样本通过核对", raw_key):
                if raw is None:
                    raise EvidenceProblem("timing_samples_missing", "missing", "partial")
                self._assert_observation(raw, instance, context)
                samples.append(benchmark_sample(raw, context["requests"], repeat, rules.warmup))
        with attempt.exchange("Evaluator", "BenchmarkRunner", "measurement_summary", label,
                              "按锁定协议汇总端到端样本", "返回端到端测量汇总",
                              *(f"{label}_repeat_{repeat}" for repeat in range(rules.repeats))):
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
        capture = raw.get("capture")
        window = capture.get("window_us") if isinstance(capture, dict) else None
        if (
            not isinstance(window, list) or len(window) != 2 or not all(numeric(value) for value in window)
            or window[0] >= window[1] or not capture.get("tool") or not capture.get("tool_version")
        ):
            raise EvidenceProblem("profile_capture_metadata_missing", "missing", "partial")
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
        result.update(ranks=sorted(ranks), missing_ranks=sorted(expected - ranks), hotspots=hotspots, capture=capture,
                      revision=revision, workload_fingerprint=context["workload_fingerprint"],
                      environment_fingerprint=context["environment_fingerprint"])
        attempt.save(
            "profile_analysis", {key: result[key] for key in ("ranks", "missing_ranks", "hotspots", "capture")},
            "trace_analysis",
        )
        if result["missing_ranks"]:
            raise EvidenceProblem("profile_ranks_missing", "missing", "partial")
