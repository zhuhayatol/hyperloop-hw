"""Search the supported Ascend Attention switch against a running SGLang service."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ..actions.candidates import candidate_revision, stack_identity
from ..contracts import ContractError
from ..evaluation.evidence import EvidenceProblem
from ..evaluation.measurements import change_ratios, numeric

Record = dict[str, Any]


def _attention_hotspot(profile: Record) -> Record | None:
    matches = []
    for item in profile.get("hotspots", []):
        if not isinstance(item, dict) or item.get("event_kind", "kernel") != "kernel":
            continue
        name = "".join(character for character in str(item.get("name", "")).casefold() if character.isalnum())
        duration = item.get("duration_us")
        if ("flashattention" in name or "fusedinferattentionscore" in name) and numeric(duration) and duration > 0:
            matches.append(item)
    return max(matches, key=lambda item: (bool(item.get("callsite")), item["duration_us"]), default=None)


def _measure_trial(
    measure: Callable[[Record, str], Record], revision: Record, attempt: str, use_fia: bool,
) -> tuple[Record, Record | None]:
    try:
        run = measure(revision, attempt)
    except ContractError as error:
        return {"attempt_id": attempt, "revision": revision["id"], "status": "blocked", "reason": str(error)}, None
    runtime = run.get("runtime") or {}
    trial = {
        "attempt_id": attempt, "revision": revision["id"], "status": run["status"], "reason": run["reason"],
        "instance_id": runtime.get("instance_id"), "launch_control": runtime.get("stock_attention_launch_control"),
        "correctness_passed": (run.get("correctness") or {}).get("pass") is True,
        "evidence": deepcopy(run.get("evidence", {})),
    }
    if "activation" in run:
        trial["activation"] = deepcopy(run["activation"])
    if trial["status"] != "ok":
        return trial, None

    expected_control = {"ASCEND_USE_FIA": "1" if use_fia else "0", "ASCEND_USE_FA": "0"}
    measurement = run.get("measurement")
    if (
        run.get("mode") != "real" or not trial["correctness_passed"] or not trial["instance_id"]
        or runtime.get("loaded_revision") != revision["id"] or trial["launch_control"] != expected_control
        or not isinstance(measurement, dict) or measurement.get("revision") != revision["id"]
        or measurement.get("boundary") != "service_end_to_end"
    ):
        trial.update(status="blocked", reason="geak_service_identity_unconfirmed")
        return trial, None
    trial["summary"] = deepcopy(measurement["summary"])
    return trial, measurement


def run_ascend_geak(
    *, task_id: str, parent: Record, profile: Record | None, fia_default: bool | None,
    measure: Callable[[Record, str], Record],
) -> tuple[Record, Record | None]:
    """Publish one independently reviewable service candidate from a bounded real service trial."""
    publication: Record = {
        "mode": "real", "backend": "geak", "delivery": "service_candidate", "target_revision": parent["id"],
        "search_policy": {"parameter": "ascend_use_fia", "candidate_limit": 1,
                          "selection": "compare the current setting with its opposite"},
        "status": "blocked",
    }
    if type(fia_default) is not bool:
        return {**publication, "reason": "ascend_fia_default_unconfigured"}, None
    if not profile or profile.get("status") != "ok" or profile.get("revision") != parent["id"] or not profile.get("report_ref"):
        return {**publication, "reason": "no_usable_profile"}, None
    publication.update(profile_ref=deepcopy(profile["report_ref"]), workload_point=profile.get("point"))
    hotspot = _attention_hotspot(profile)
    if hotspot is None:
        return {**publication, "reason": "no_stock_attention_hotspot"}, None
    publication["hotspot"] = deepcopy(hotspot)

    current = parent["parameters"].get("ascend_use_fia", fia_default)
    if type(current) is not bool:
        return {**publication, "reason": "ascend_fia_parent_setting_invalid"}, None
    candidate = {
        "id": f"{task_id}-geak-fia", "parent": parent["id"], "backend": "geak",
        "parameters": {"ascend_use_fia": not current}, "profile_ref": deepcopy(profile["report_ref"]),
        "hotspot": deepcopy(hotspot),
        "reason": "Compare the stock Attention implementations at the observed hotspot.",
        "expected_metric": "service throughput", "risk": "Attention output or service latency may regress.",
        "validation": "Repeat model-output and end-to-end checks against the same parent and frozen workload.",
    }
    revision = candidate_revision(candidate, parent, "parameters")
    parent_trial, parent_measurement = _measure_trial(measure, parent, f"{task_id}-geak-parent", current)
    publication["parent_trial"] = parent_trial
    if parent_measurement is None:
        return {**publication, "status": parent_trial["status"], "reason": parent_trial["reason"]}, None
    candidate_trial, candidate_measurement = _measure_trial(measure, revision, f"{task_id}-geak-candidate", not current)
    publication["candidate_trial"] = candidate_trial
    if candidate_measurement is None:
        return {**publication, "status": candidate_trial["status"], "reason": candidate_trial["reason"]}, None
    if parent_trial["instance_id"] == candidate_trial["instance_id"]:
        return {**publication, "reason": "geak_service_instances_overlap"}, None
    try:
        changes = change_ratios(candidate_measurement, parent_measurement)
    except EvidenceProblem as error:
        return {**publication, "reason": error.reason}, None

    # Trial timings explain the proposal. Only the independent remeasurement can accept it.
    publication.update(status="ok", reason="service_candidate_ready_for_review", comparison=changes,
                       service_candidate={"spec": deepcopy(candidate), "configuration": deepcopy(revision["parameters"]),
                                          "loadable_stack": stack_identity(revision)})
    return publication, candidate
