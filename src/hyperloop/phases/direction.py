"""Identify repeated unsuccessful experiments from committed candidate outcomes."""

from collections.abc import Iterable, Mapping
from typing import Any


_DIRECTIONS = {"parameters": "parameters", "patch": "patch", "operator_patch": "kernel", "kernel": "kernel"}
_ACTIONS = {"explore": "parameters", "integrate_patch": "patch", "forge": "kernel", "geak": "kernel"}


def _approach(decision: Mapping[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    direction = _DIRECTIONS.get(decision.get("kind"))
    revision = decision.get("revision") or {}
    parent = decision.get("parent_revision") or {}
    if direction == "parameters":
        before, after = parent.get("parameters", {}), revision.get("parameters", {})
        changed = tuple(sorted(key for key, value in after.items() if key not in before or before[key] != value))
        return (direction, changed) if changed else None
    if direction is not None:
        key, field = ("framework", "patch_path") if direction == "patch" else ("kernels", "operator")
        changes = [row for row in revision.get(key, []) if row.get("candidate_id") == decision.get("candidate_id")]
        return direction, tuple(str(row[field]) for row in changes if row.get(field))
    return None


def direction_change(state: Mapping[str, Any], allowed_actions: Iterable[str]) -> dict[str, Any] | None:
    """Propose another investigation after two measured reverts of one approach."""
    decisions = [(row, _approach(row)) for row in state.get("decisions", []) if row.get("comparison")]
    measured = [(row, approach) for row, approach in decisions if approach is not None]
    framework = [(row, approach) for row, approach in measured if approach[0] != "kernel"]
    relevant = framework or measured
    if len(relevant) < 2:
        return None
    latest, approach = relevant[-1]
    previous, earlier_approach = relevant[-2]
    if approach != earlier_approach or any(row.get("version_action") != "REVERT" for row in (previous, latest)):
        return None

    direction, mechanism = approach
    enabled = {_ACTIONS.get(action, action) for action in allowed_actions}
    alternatives = [value for value in ("profile", "patch", "kernel", "parameters")
                    if value in enabled and value != direction]
    if not alternatives:
        return None

    profile = next((row for row in reversed(state.get("profiles", [])) if row.get("status") == "ok"), {})
    return {
        "reason": "repeated_approach_without_accepted_result",
        "tested_direction": direction,
        "mechanism": list(mechanism),
        "tested_candidates": [previous["candidate_id"], latest["candidate_id"]],
        "alternatives": alternatives,
        "profile_ref": profile.get("report_ref"),
        "cycle": state.get("cycle"),
    }
