"""Confirm a goal on a fresh pair without changing the accepted revision."""

from __future__ import annotations

from math import isclose
from typing import Any

from ..contracts import ContractError
from .adjudication import verify_service_retest
from .evidence import EvidenceProblem
from .measurements import change_ratios, numeric

GOAL_RETEST_COST = 2


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold or isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def goal_checks(deltas: dict, goals: dict, rules: dict) -> dict[str, bool]:
    return {
        "throughput": _at_least(deltas["throughput_pct"], goals["throughput_gain_pct"]),
        "ttft_p95": _at_least(-deltas["ttft_pct"], goals["ttft_reduction_pct"]),
        "tpot_p95": _at_least(rules["max_tpot_regression_pct"], deltas["tpot_pct"]),
    }


def warrants_retest(state: dict, goals: dict, rules: dict) -> bool:
    """Use accepted screening samples only to decide when to dispatch the fresh pair."""
    baseline = state["baseline"]["measurement"]["summary"]
    accepted = state["current_revision"]["summary"]
    metrics = {"throughput_pct": "throughput", "ttft_pct": "ttft_ms", "tpot_pct": "tpot_ms"}
    if any(not numeric(baseline.get(metric)) or baseline[metric] <= 0
           or not numeric(accepted.get(metric)) for metric in metrics.values()):
        return False
    deltas = {name: 100 * (accepted[metric] - baseline[metric]) / baseline[metric]
              for name, metric in metrics.items()}
    return all(goal_checks(deltas, goals, rules).values())


def _forge_origin(state: dict, revision: dict) -> str | None:
    kernels = revision["kernels"]
    if not kernels or kernels[-1].get("kind") != "operator_patch":
        return None
    return next((row["parent"] for row in state["decisions"]
                 if row["candidate_id"] == kernels[-1]["candidate_id"] and row["version_action"] == "KEEP"), None)


def paired_goal_retest(actions: Any, state: dict, task_id: str, goals: dict) -> dict:
    original = state["baseline"]["revision"]
    accepted = state["current_revision"]
    arms: dict[str, dict] = {}
    result: dict[str, Any] = {
        "status": "incomplete", "reason": "goal_retest_incomplete", "revision": accepted["id"],
        "baseline_revision": original["id"], "goals": goals, "checks": {}, "arms": arms,
    }
    verified: dict[str, dict] = {}
    try:
        for arm, revision, suffix in (("original", original, "o"), ("candidate", accepted, "c")):
            measured = actions.measure(revision, f"{task_id}-{suffix}")
            arms[arm] = {key: measured[key] for key in ("attempt_id", "status", "reason", "evidence")}
            if measured["status"] != "ok":
                result["reason"] = measured["reason"]
                break
            checked = verify_service_retest(actions.store, measured, revision, forge_parent=_forge_origin(state, revision))
            verified[arm] = checked
            arms[arm]["correctness"] = checked["correctness"]
            if not checked["correctness"]["pass"]:
                if arm == "candidate":
                    result.update(status="fail", reason="goal_candidate_correctness_failed", checks={"correctness": False})
                else:
                    result["reason"] = "goal_original_correctness_failed"
                break
            arms[arm]["summary"] = checked["measurement"]["summary"]
        else:
            if verified["original"]["instance_id"] == verified["candidate"]["instance_id"]:
                raise EvidenceProblem("goal_retest_instances_overlap", "incomparable")
            recorded_original = state["baseline"]["runtime"]["instance_id"]
            if any(arm["instance_id"] == recorded_original for arm in verified.values()):
                raise EvidenceProblem("goal_retest_reused_baseline_instance", "incomparable")
            deltas = change_ratios(verified["candidate"]["measurement"], verified["original"]["measurement"])
            checks = {"correctness": True, **goal_checks(deltas, goals, actions.rules)}
            reached = all(checks.values())
            result.update(
                status="pass" if reached else "fail", reason="goals_met" if reached else "goals_not_met",
                checks=checks, comparison={"baseline_revision": original["id"], "over_original": deltas},
            )
    except EvidenceProblem as error:
        result["reason"] = error.reason
    except ContractError as error:
        result.update(reason="goal_retest_registration_failed", detail=str(error))
    return result
