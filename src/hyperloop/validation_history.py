"""Reconstruct historical validation handoffs from saved artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ARM_LABELS = {
    "candidate": "候选",
    "parent": "父版",
    "revision": "被测版本",
    "kernel": "算子",
    "forward": "模型 forward",
    "profile": "采集目标",
}


def _indexed_keys(keys: set[str], prefix: str) -> list[tuple[int, str]]:
    matches = []
    for key in keys:
        if key.startswith(prefix):
            suffix = key.removeprefix(prefix)
            if suffix.isascii() and suffix.isdecimal():
                matches.append((int(suffix), key))
    return sorted(matches)


def _check_state(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("pass") is True:
            return "succeeded"
        if value.get("pass") is False:
            return "failed"
    return "recorded"


def reconstruct_validation_flow(records: dict[str, dict], *, result: dict) -> list[dict]:
    """Infer an ordered handoff view without inventing calls or timestamps.

    Each row must cite a persisted evidence key. Paired comparisons, final
    decisions and independent Critic events belong to the Session-level view.
    """
    keys = {name for name, record in records.items() if isinstance(name, str) and isinstance(record, Mapping)}
    rows: list[dict] = []

    def add(
        operation: str, scope: str, sender: str, receiver: str, message: str,
        *references: str, state: str = "recorded",
    ) -> None:
        present = [name for name in references if name in keys]
        if present:
            rows.append({
                "sender": sender, "receiver": receiver, "message": message, "operation": operation,
                "state": state, "scope": scope, "evidence_keys": present,
            })

    def service_correctness(scope: str, result_key: str) -> None:
        label = _ARM_LABELS[scope]
        add(
            "observe_outputs", scope, "RuntimeBackend", "Verifier",
            f"交回{label}的逐请求原始服务输出。", f"{scope}_service_outputs",
        )
        correctness = f"{scope}_service_correctness"
        add(
            "correctness", scope, "Verifier", "Evaluator",
            f"登记{label}与冻结 Reference 的正确性结果。", correctness,
            state=_check_state(result.get(result_key, records.get(correctness))),
        )

    def service_timing(scope: str) -> None:
        label = _ARM_LABELS[scope]
        for index, evidence in _indexed_keys(keys, f"{scope}_repeat_"):
            add(
                "benchmark_sample", scope, "BenchmarkRunner", "Evaluator",
                f"登记{label}第 {index + 1} 次端到端计时的原始记录。", evidence,
            )
        add(
            "measurement_summary", scope, "Evaluator", "EvidenceStore",
            f"保存{label}的端到端计时口径与指标汇总。", f"{scope}_measurement",
        )

    def operator() -> None:
        add(
            "reference", "kernel", "EvidenceStore", "Verifier",
            "提供冻结的算子 Reference。", "kernel_reference",
        )
        add(
            "observe_outputs", "kernel", "RuntimeBackend", "Verifier",
            "交回算子原始输出。", "operator_kernel_outputs",
        )
        add(
            "correctness", "kernel", "Verifier", "Evaluator",
            "登记算子正确性结论。", "operator_kernel_correctness",
            state=_check_state(result.get("kernel_correctness", records.get("operator_kernel_correctness"))),
        )
        add(
            "device_timer", "kernel", "RuntimeBackend", "Evaluator",
            "交回设备计时原始样本。", "device_timing_raw",
        )
        add(
            "device_timing_summary", "kernel", "Evaluator", "EvidenceStore",
            "保存独立于服务端到端的设备计时汇总。", "device_timing",
        )

    frozen_inputs = {"workload": "冻结负载", "policy": "评测规则", "reference": "Reference"}
    add(
        "reference", "evaluation", "EvidenceStore", "Evaluator",
        "登记本次保存的" + "、".join(label for key, label in frozen_inputs.items() if key in keys) + "。",
        *frozen_inputs,
    )
    add(
        "probe", "evaluation", "RuntimeBackend", "Evaluator",
        "登记实际环境、设备与能力探测。", "probe",
    )

    arm_scopes = ("revision", "kernel", "candidate", "parent", "forward", "profile")
    cleanup_scopes = [scope for scope in arm_scopes if f"{scope}_cleanup" in keys]
    failed_cleanup = cleanup_scopes[-1] if cleanup_scopes and result.get("reason") == "cleanup_unconfirmed" else None
    has_kernel_arm = bool({"kernel_prepared", "kernel_launch", "kernel_cleanup"} & keys)
    for scope in arm_scopes:
        label = _ARM_LABELS[scope]
        add(
            "prepare", scope, "RuntimeBackend", "Evaluator",
            f"交回{label}的隔离工作区与准备记录。", f"{scope}_prepared",
        )
        add(
            "launch", scope, "RuntimeBackend", "Evaluator",
            f"交回{label}的启动身份与实际加载配置。", f"{scope}_launch",
        )
        if scope == ("kernel" if has_kernel_arm else "candidate"):
            operator()
        if scope in {"revision", "candidate", "parent", "forward"}:
            service_correctness(scope, "parent_correctness" if scope == "parent" else "correctness")
        if scope in {"revision", "candidate", "parent"}:
            service_timing(scope)
        if scope == "forward":
            add(
                "forward_timer", scope, "RuntimeBackend", "Evaluator",
                "交回模型 forward 原始计时样本。", "forward_raw",
            )
            add(
                "forward_timing_summary", scope, "Evaluator", "EvidenceStore",
                "保存模型 forward 计时汇总。", "forward_timing",
            )
        if scope == "profile":
            add(
                "collect_profile", scope, "Profiler", "TraceAnalyzer",
                "交回原始采集时间线及采集窗口。", "profile_raw",
            )
            for rank, evidence in _indexed_keys(keys, "profile_rank_"):
                add(
                    "collect_profile", scope, "Profiler", "EvidenceStore",
                    f"保存 rank {rank} 的独立原始时间线。", evidence,
                )
            add(
                "parse_profile", scope, "TraceAnalyzer", "Evaluator",
                "登记热点、有效 rank 和缺失 rank。", "profile_analysis",
                state="failed" if result.get("missing_ranks") else "recorded",
            )
        cleanup_key = f"{scope}_cleanup"
        cleanup = records.get(cleanup_key) or {}
        released = cleanup.get("released") if isinstance(cleanup, Mapping) else None
        cleanup_state = "recorded"
        if released is False or scope == failed_cleanup:
            cleanup_state = "failed"
        elif released is True or result.get("status") == "ok":
            cleanup_state = "succeeded"
        add(
            "cleanup", scope, "RuntimeBackend", "Evaluator",
            f"交回{label}的资源清理记录。", cleanup_key, state=cleanup_state,
        )

    add(
        "tool_publication", "kernel", "RuntimeBackend", "EvidenceStore",
        "保存算子后端的候选交付凭据。", "publication", "tool_publication",
    )
    add(
        "backend_failure", "evaluation", "RuntimeBackend", "EvidenceStore",
        "保存运行端报出的失败。", "backend_failure", state="failed",
    )
    return rows
