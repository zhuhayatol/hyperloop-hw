"""Build the two simulated kernel handoffs with distinct ownership."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..contracts import ContractError


def plan_kernel_handoff(
    backend: str,
    parent: dict[str, Any],
    workload: dict[str, Any],
    profile: dict[str, Any] | None,
    fixture: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if backend not in ("geak", "forge"):
        raise ContractError(f"unknown kernel route: {backend}")
    handoff = {
        "backend": backend,
        "target_revision": parent["id"],
        "workload": workload,
        "loaded_parent_stack": {
            "framework": parent["framework"], "kernels": parent["kernels"],
            "parameters": parent["parameters"],
        },
        "profile_ref": (profile or {}).get("report_ref"),
        "delivery": "service_candidate" if backend == "geak" else "operator_patch",
    }
    if profile is None or profile.get("status") != "ok" or not handoff["profile_ref"]:
        return {**handoff, "reason": "no_usable_profile"}, []

    hotspots = profile.get("hotspots", [])
    candidates = []
    for spec in fixture:
        operator = spec.get("operator", "")
        matches = [
            hotspot for hotspot in hotspots
            if operator.casefold() in str(hotspot.get("name", hotspot.get("operator", ""))).casefold()
        ] if isinstance(operator, str) and operator else []
        if matches:
            candidates.append({
                **deepcopy(spec),
                "backend": backend,
                "profile_ref": handoff["profile_ref"],
                "hotspot": matches[0],
            })
    return {**handoff, "selected": [item["id"] for item in candidates]}, candidates
