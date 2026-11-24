"""Decide phase transitions from persisted state and frozen policy."""

from __future__ import annotations

from typing import Any

SWEEP_PAIR_COST = 2


def after_prelude(settings: dict[str, Any]) -> str:
    if settings["framework_enabled"]:
        return "FRAMEWORK_AGENT"
    return after_framework(settings)


def after_framework(settings: dict[str, Any]) -> str:
    return "SWEEP" if settings["kernel_backend"] == "disabled" else "KERNEL_AGENT"


def sweep_skip_reason(
    settings: dict[str, Any],
    revision: str,
    previous_revision: str | None,
    budget: int,
) -> str | None:
    if not settings["sweep_enabled"]:
        return "sweep_disabled"
    if previous_revision == revision:
        return "no_validated_change_since_sweep"
    if budget < SWEEP_PAIR_COST:
        return "insufficient_sweep_budget"
    return None


def after_sweep(
    state: dict[str, Any],
    settings: dict[str, Any],
    budget: int,
) -> tuple[str, str]:
    sweep = state["last_conc_sweep"] or {}
    if sweep.get("status") in ("failed", "partial"):
        return "CLOSE", "sweep_failed_or_incomplete"
    baseline = state.get("baseline") or {}
    original = service_throughput(baseline)
    current = service_throughput(state.get("current_revision") or {})
    if original and current and current / original >= 1 + settings["target_gain_pct"] / 100:
        return "CLOSE", "target_reached"
    if not settings["framework_enabled"]:
        return "CLOSE", "framework_disabled"
    if budget < 1:
        return "CLOSE", "budget_exhausted"
    if state["cycle"] >= settings["max_cycles"]:
        return "CLOSE", "cycle_limit"
    if sweep.get("reason") == "no_validated_change_since_sweep":
        return "CLOSE", "converged"
    return "FRAMEWORK_AGENT", "continue_from_accepted_revision"


def service_throughput(result: dict[str, Any]) -> float | None:
    summary = (result.get("measurement") or {}).get("summary") or result.get("summary", {})
    value = summary.get("throughput")
    return float(value) if type(value) in (float, int) and value > 0 else None


def should_reprofile(revision: dict[str, Any], last: dict[str, Any] | None, ratio: float) -> bool:
    if last is None:
        return True
    if revision["id"] == last.get("revision"):
        return False
    anchor = last.get("anchor_throughput")
    now = service_throughput(revision)
    return bool(anchor and now and now / anchor >= ratio)
