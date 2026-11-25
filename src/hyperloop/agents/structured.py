"""Bound the inputs and intents of the live Agent provider."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ..contracts import ContractError, safe_id

_RATIONALE = ("reason", "expected_metric", "risk", "validation")
_METRICS = ("throughput", "ttft_ms", "tpot_ms")
_DELTAS = ("throughput_pct", "ttft_pct", "tpot_pct")
_ROPE_LAYOUT_METRICS = (
    "tokens", "sample_count", "split_rope_us", "contiguous_rope_us", "copy_us",
    "copy_plus_rope_us", "rope_gain_us", "net_gain_us", "net_p25_us", "net_p75_us",
)
_CODE_LIMIT = 2400
_FOCUS_LIMIT = 128
_SOURCE_PATH = re.compile(r"(?:python/)?sglang/[A-Za-z0-9_./-]+\.py")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+ -]{0,127}")


class InvalidAgentResponse(ValueError):
    """The provider's structured content does not fit the role contract."""


def source_excerpts(raw: Any, target: Mapping[str, Any]) -> list[dict[str, str]]:
    """Freeze only excerpts explicitly supplied for the task and allowed target paths."""
    if not isinstance(raw, list) or len(raw) > 4:
        raise ContractError("agent_source_excerpts must contain up to four reviewed source excerpts")
    approved = target.get("sglang", {}).get("allowed_source_paths", [])
    rows = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, dict) or row.keys() != {"id", "path", "before"}:
            raise ContractError("each Agent source excerpt must have id, path and before fields")
        identity = safe_id(row["id"], "agent_source_excerpt.id")
        path, before = row["path"], row["before"]
        if (
            identity in seen or not isinstance(path, str) or path not in approved
            or not _SOURCE_PATH.fullmatch(path) or ".." in PurePosixPath(path).parts
            or not isinstance(before, str) or not before.strip() or len(before) > _CODE_LIMIT
            or "\x00" in before or "\r" in before
        ):
            raise ContractError("Agent source excerpt must name an approved Python path and a bounded LF source block")
        seen.add(identity)
        rows.append(dict(row))
    return rows


def _number(value: Any) -> bool:
    return type(value) in (float, int) and abs(value) < 1e15 and math.isfinite(value)


def _selected_numbers(row: Any, keys: tuple[str, ...]) -> dict[str, float]:
    return {key: row[key] for key in keys if _number(row.get(key))} if isinstance(row, Mapping) else {}


def _safe_labels(row: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    return {
        key: row[key] for key in keys
        if isinstance(row.get(key), str) and _LABEL.fullmatch(row[key])
    }


def _parameters(raw: Any, allowed: set[str]) -> dict[str, int | float | bool]:
    return {
        key: value for key, value in raw.items() if key in allowed and (type(value) is bool or _number(value))
    } if isinstance(raw, Mapping) else {}


def _server_settings(target: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    args = target.get("sglang", {}).get("server_args", [])
    settings = {}
    for index, flag in enumerate(args):
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        name, separator, value = flag[2:].partition("=")
        name = name.replace("-", "_")
        if name not in allowed:
            continue
        if not separator:
            value = args[index + 1] if index + 1 < len(args) else ""
        if isinstance(value, str) and re.fullmatch(r"-?\d{1,12}(?:\.\d{1,8})?", value):
            settings[name] = float(value) if "." in value else int(value)
    return settings


def _workload(task: Mapping[str, Any]) -> dict[str, Any]:
    workload = task.get("workload", {})
    rows = workload.get("requests", [])
    ranges = {}
    for phase in ("prefill", "decode"):
        phase_rows = [row for row in rows if row.get("phase") == phase]
        if phase_rows:
            lengths = [len(value) if isinstance(value, list) else value
                       for value in (row["input_tokens"] for row in phase_rows)]
            outputs = [row["output_tokens"] for row in phase_rows]
            ranges[phase] = {"requests": len(phase_rows), "input_tokens_range": [min(lengths), max(lengths)],
                             "output_tokens_range": [min(outputs), max(outputs)]}
    return {**_selected_numbers(workload, ("concurrency",)), **_safe_labels(workload, ("dtype",)), **ranges}


def _profile(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    hotspots = []
    ordered = sorted(raw.get("hotspots", []),
                     key=lambda item: (item.get("event_kind") != "device_runtime", item.get("duration_us", 0)),
                     reverse=True)
    for row in ordered[:8]:
        fields = {**_safe_labels(row, ("name", "dtype", "phase", "event_kind", "operator")),
                  **_selected_numbers(row, ("rank", "duration_us", "events"))}
        callsite = row.get("callsite", "")
        if isinstance(callsite, str):
            # Absolute prefixes belong to the local runtime. Task-relative package location is enough.
            match = re.search(r"(?:python/)?sglang/[A-Za-z0-9_./-]+\.py(?::\d+)?", callsite.replace("\\", "/"))
            if match and ".." not in PurePosixPath(match.group()).parts:
                fields["callsite"] = match.group()
            elif _LABEL.fullmatch(callsite):
                fields["callsite"] = callsite
        shape = row.get("input_shape")
        if isinstance(shape, list) and len(shape) <= 8 and all(type(dim) is int and 0 < dim < 1e12 for dim in shape):
            fields["input_shape"] = shape
        if "name" in fields:
            hotspots.append(fields)
    return {**_safe_labels(raw, ("revision",)), "hotspots": hotspots}


def _reviews(raw: Mapping[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    reviews = []
    for identity, proposal in list(raw.items())[:8]:
        fields = _safe_labels({"proposal_id": identity, **proposal}, ("proposal_id", "action", "source"))
        params = proposal.get("params", {})
        for key in ("parent", "mode"):
            fields.update(_safe_labels(params, (key,)))
        specs = params.get("candidates")
        if not isinstance(specs, list):
            specs = [params.get("candidate")]
        candidates = []
        for spec in specs[:4]:
            if not isinstance(spec, dict):
                continue
            row = _safe_labels(spec, ("id",))
            row["parameters"] = _parameters(spec.get("parameters"), allowed)
            for key in _RATIONALE:
                if isinstance(spec.get(key), str):
                    row[key] = spec[key][:400]
            path = spec.get("patch_path")
            if isinstance(path, str) and _SOURCE_PATH.fullmatch(path):
                row["patch_path"] = path
                edit = spec.get("source_edit")
                if isinstance(edit, dict):
                    row["source_edit"] = {key: edit[key][:_CODE_LIMIT] for key in ("before", "after")
                                          if isinstance(edit.get(key), str)}
            candidates.append(row)
        fields["candidates"] = candidates
        evidence = _review_evidence(proposal.get("review_evidence"))
        if evidence is not None:
            fields["review_evidence"] = evidence
        reviews.append(fields)
    return reviews


def _review_evidence(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    summary = raw.get("summary")
    evidence_ref = raw.get("evidence_ref")
    prior = raw.get("prior_review")
    if (
        not isinstance(summary, Mapping) or summary.get("kind") != "rope_layout"
        or summary.get("mode") not in ("real", "simulated") or not isinstance(summary.get("cases"), list)
        or not isinstance(evidence_ref, Mapping) or not isinstance(prior, Mapping)
    ):
        return None
    boundary = _safe_labels(summary, ("boundary",)).get("boundary")
    evidence_id = _safe_labels(evidence_ref, ("id",)).get("id")
    if boundary is None or evidence_id is None:
        return None
    cases = []
    for case in summary["cases"][:8]:
        if not isinstance(case, Mapping):
            continue
        numbers = _selected_numbers(case, _ROPE_LAYOUT_METRICS)
        if len(numbers) != len(_ROPE_LAYOUT_METRICS) or type(case.get("outputs_match")) is not bool:
            continue
        projected = {**_safe_labels(case, ("phase", "layout_source")), **numbers,
                     "outputs_match": case["outputs_match"]}
        for key in ("q_stride", "k_stride"):
            stride = case.get(key)
            if isinstance(stride, list) and 0 < len(stride) <= 8 and all(
                type(value) is int and abs(value) < 1e12 for value in stride
            ):
                projected[key] = stride
        cases.append(projected)
    if not cases:
        return None
    prior_review = _safe_labels(prior, ("reviewer", "verdict"))
    if _short_text(prior.get("reason")):
        prior_review["reason"] = prior["reason"]
    return {
        "evidence_id": evidence_id,
        "summary": {"kind": "rope_layout", "mode": summary["mode"], "boundary": boundary, "cases": cases},
        "prior_review": prior_review,
    }


def project_context(
    role: str, context: Mapping[str, Any], task: Mapping[str, Any], target: Mapping[str, Any],
    excerpts: list[dict[str, str]],
) -> dict[str, Any]:
    """Project local context into an explicit outbound schema for one role."""
    allowed = {key for key in task.get("allowed_parameters", []) if isinstance(key, str) and _LABEL.fullmatch(key)}
    if role == "specialist" and isinstance(context.get("startup_failure"), Mapping):
        failure = context["startup_failure"]
        return {
            "role": role, "phase": "ENABLEMENT",
            "startup_failure": {
                **_safe_labels(failure, ("code", "assertion")),
                **_selected_numbers(failure, ("chunked_prefill_size", "page_size")),
            },
            "allowed_parameters": sorted(allowed & {"chunked_prefill_size"}),
        }
    current = context.get("current_revision") or {}
    baseline = context.get("baseline") or {}
    updates = _parameters(current.get("parameters"), allowed)
    history = []
    for row in context.get("decisions", [])[-8:]:
        comparison = row.get("comparison") or {}
        history.append({**_safe_labels(row, ("candidate_id", "parent", "kind", "version_action", "reason")),
                        "over_parent": _selected_numbers(comparison.get("over_parent"), _DELTAS)})
    data = {
        "role": role,
        **_safe_labels(context, ("phase",)),
        **_selected_numbers(context, ("cycle", "budget", "goal")),
        "target": {**_safe_labels(target, ("mode", "device_name")), **_selected_numbers(target, ("tp_size",)),
                   **_safe_labels(target.get("software", {}), ("sglang", "cann"))},
        "model": _safe_labels(task.get("model", {}), ("id", "dtype")),
        "workload": _workload(task),
        "rules": _selected_numbers(task.get("rules", {}), ("min_gain_pct", "noise_floor_pct",
                                                            "max_ttft_regression_pct", "max_tpot_regression_pct")),
        "allowed_parameters": sorted(allowed),
        "current": {**_safe_labels(current, ("id",)), "summary": _selected_numbers(current.get("summary"), _METRICS),
                    "parameters": {**_server_settings(target, allowed), **updates}},
        "baseline": _selected_numbers(baseline.get("summary"), _METRICS),
        "profile": _profile(context.get("profile")),
        "recent_decisions": history,
    }
    if "agent_objective" in task:
        data["agent_objective"] = task["agent_objective"]
    if isinstance(task.get("goals"), Mapping):
        data["goals"] = _selected_numbers(task["goals"], ("throughput_gain_pct", "ttft_reduction_pct"))
    direction = context.get("direction_change")
    if role == "orchestration" and isinstance(direction, Mapping):
        data["direction_change"] = {
            **_safe_labels(direction, ("reason", "tested_direction")),
            **{key: [value for value in direction.get(key, [])[:8]
                     if isinstance(value, str) and _LABEL.fullmatch(value)]
               for key in ("mechanism", "tested_candidates", "alternatives")},
            "instruction": "Investigate a different supported cause or obtain evidence that distinguishes it before repeating the same approach.",
        }
    if role == "critic":
        data["pending_proposals"] = _reviews(context.get("pending_proposals", {}), allowed)
    elif role == "specialist":
        assignment = context.get("assignment", {})
        data["assignment"] = _safe_labels(assignment, ("mode",))
        focus = assignment.get("focus")
        if _short_text(focus, _FOCUS_LIMIT):
            data["assignment"]["focus"] = focus
        data["source_excerpts"] = excerpts
    else:
        data["source_excerpt_paths"] = sorted({row["path"] for row in excerpts})
        feedback = []
        for row in context.get("review_feedback", [])[-8:]:
            if not isinstance(row, Mapping) or row.get("status") not in ("needs_review", "reject"):
                continue
            entry = _safe_labels(row, (
                "proposal_id", "source", "action", "status", "parent", "evidence_status", "evidence_reason",
            ))
            if _short_text(row.get("reason")):
                entry["reason"] = row["reason"]
            entry["candidate_ids"] = [
                value for value in row.get("candidate_ids", [])[:4]
                if isinstance(value, str) and _LABEL.fullmatch(value)
            ]
            evidence = _review_evidence(row.get("review_evidence"))
            if evidence is not None:
                entry["review_evidence"] = evidence
            if "proposal_id" in entry:
                feedback.append(entry)
        if feedback:
            data["review_feedback"] = feedback
    recipe = context.get("recipe") or {}
    if recipe.get("status") == "applicable":
        data["recipe_suggestions"] = _parameters(recipe.get("parameters"), allowed)
    return data


def _keys(row: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> bool:
    return isinstance(row, dict) and set(required) <= row.keys() <= set(required) | set(optional)


def _short_text(value: Any, limit: int = 400) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip()) and value.isprintable()


def _identity(value: Any) -> None:
    try:
        safe_id(value, "Agent intent identifier")
    except ContractError:
        raise InvalidAgentResponse from None


def _candidate(spec: Any, *, patch: bool, excerpts: list[dict[str, str]]) -> dict[str, Any]:
    fields = ("source_excerpt_id", "replacement") if patch else ("parameters",)
    if not _keys(spec, ("id", *_RATIONALE, *fields)) or any(not _short_text(spec[key]) for key in _RATIONALE):
        raise InvalidAgentResponse
    _identity(spec["id"])
    if not patch:
        params = spec["parameters"]
        if not isinstance(params, dict) or not 1 <= len(params) <= 8 or any(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key) is None or not (type(value) is bool or _number(value))
            for key, value in params.items()
        ):
            raise InvalidAgentResponse
        return spec
    excerpt = next((row for row in excerpts if row["id"] == spec["source_excerpt_id"]), None)
    replacement = spec["replacement"]
    if (excerpt is None or not isinstance(replacement, str) or not replacement.strip()
        or len(replacement) > _CODE_LIMIT or any(char in replacement for char in ("\x00", "\r"))
        or replacement == excerpt["before"]):
        raise InvalidAgentResponse
    return {key: spec[key] for key in ("id", *_RATIONALE)} | {
        "patch_path": excerpt["path"], "source_edit": {"before": excerpt["before"], "after": replacement},
    }


def parse_intents(
    raw: Any, role: str, excerpts: list[dict[str, str]], *, startup_repair: bool = False,
) -> list[dict[str, Any]]:
    """Parse one JSON object into strict role intents for the local Router."""
    if not _keys(raw, ("intents",)) or not isinstance(raw["intents"], list) or len(raw["intents"]) > 8:
        raise InvalidAgentResponse
    if startup_repair:
        if role != "specialist" or len(raw["intents"]) > 1:
            raise InvalidAgentResponse
        for row in raw["intents"]:
            if (
                not _keys(row, ("kind", "parameters", "reason")) or row["kind"] != "repair_startup"
                or not _keys(row["parameters"], ("chunked_prefill_size",))
                or type(row["parameters"]["chunked_prefill_size"]) is not int
                or not 0 < row["parameters"]["chunked_prefill_size"] < 10**9
                or not _short_text(row["reason"])
            ):
                raise InvalidAgentResponse
        return raw["intents"]
    intents = []
    for row in raw["intents"]:
        if not isinstance(row, dict):
            raise InvalidAgentResponse
        kind, action = row.get("kind"), row.get("action")
        if role == "critic":
            if (
                kind != "review_verdict" or not _keys(row, ("kind", "proposal_id", "verdict", "reason"))
                or row["verdict"] not in ("approve", "advise", "reject", "needs_review")
                or not _short_text(row["reason"])
            ):
                raise InvalidAgentResponse
            _identity(row["proposal_id"])
        elif kind == "escalate_strategy_change" and role == "orchestration":
            if not _keys(row, ("kind", "next_action_hint")) or row["next_action_hint"] not in ("skip_to_kernel", "skip_to_sweep"):
                raise InvalidAgentResponse
        elif (role, kind, action) in (
            ("orchestration", "delegate", "explore"), ("orchestration", "delegate", "specialist"),
            ("specialist", "propose_action", "integrate_patch"), ("specialist", "propose_action", "explore"),
        ):
            if not _keys(row, ("kind", "action", "params")):
                raise InvalidAgentResponse
            params = row["params"]
            if action == "explore":
                if not _keys(params, ("candidates",)) or not isinstance(params["candidates"], list) or not 1 <= len(params["candidates"]) <= 3:
                    raise InvalidAgentResponse
                params = {"candidates": [_candidate(spec, patch=False, excerpts=[]) for spec in params["candidates"]]}
            elif action == "integrate_patch":
                if not _keys(params, ("candidate",)):
                    raise InvalidAgentResponse
                params = {"candidate": _candidate(params["candidate"], patch=True, excerpts=excerpts)}
            elif (
                not _keys(params, ("mode", "focus")) or params["mode"] != "author"
                or not _short_text(params["focus"], _FOCUS_LIMIT)
            ):
                raise InvalidAgentResponse
            row = {**row, "params": params}
        else:
            raise InvalidAgentResponse
        intents.append(row)
    return intents
