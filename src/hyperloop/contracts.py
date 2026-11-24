"""Validate and freeze the data shared by the CLI, Coordinator and adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any


class ContractError(ValueError):
    """A configuration or recorded object violates a data contract."""


def safe_id(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) is None
        or value.endswith(".")
        or PureWindowsPath(value).is_reserved()
    ):
        raise ContractError(f"{field} must be a safe, portable identifier")
    return value


def json_bytes(data: Any) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def digest(data: Any) -> str:
    return hashlib.sha256(json_bytes(data)).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)
        json_bytes(result)
    except (UnicodeError, ValueError) as error:
        raise ContractError(f"{path}: {error}") from error
    return _mapping(result, str(path))


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, minimum: float = 0) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum:
        raise ContractError(f"{field} must be a finite number >= {minimum}")
    return float(value)


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a nonempty array")
    return value


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    safe_id(task.get("id"), "task.id")
    if "agent_objective" in task:
        objective = task["agent_objective"]
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 500 or not objective.isprintable():
            raise ContractError("task.agent_objective must be a single line of at most 500 characters")
    model = _mapping(task.get("model"), "model")
    for key in ("id", "revision", "tokenizer", "tokenizer_revision", "dtype"):
        _text(model.get(key), f"model.{key}")
    if "allowed_parameters" in task:
        allowed = task["allowed_parameters"]
        if not isinstance(allowed, list) or any(not isinstance(item, str) or not item for item in allowed):
            raise ContractError("allowed_parameters must be a list of parameter names")

    workload = _mapping(task.get("workload"), "workload")
    safe_id(workload.get("id"), "workload.id")
    if workload.get("dtype") != model["dtype"]:
        raise ContractError("workload dtype must match the frozen model dtype")
    _integer(workload.get("concurrency"), "workload.concurrency", minimum=1)
    requests = _list(workload.get("requests"), "workload.requests")
    request_ids: set[str] = set()
    for request in requests:
        request = _mapping(request, "request")
        request_id = safe_id(request.get("id"), "request.id")
        if request_id in request_ids:
            raise ContractError(f"duplicate request: {request_id}")
        request_ids.add(request_id)
        input_tokens = request.get("input_tokens")
        if isinstance(input_tokens, list):
            for token in _list(input_tokens, "request.input_tokens"):
                _integer(token, "input token ID", minimum=0)
        else:
            _integer(input_tokens, "request.input_tokens", minimum=1)
        _integer(request.get("output_tokens"), "request.output_tokens", minimum=1)
    _mapping(workload.get("sampling"), "workload.sampling")

    rules = _mapping(task.get("rules"), "rules")
    _integer(rules.get("repeats"), "rules.repeats", minimum=2)
    _integer(rules.get("warmup"), "rules.warmup", minimum=0)
    for key in ("min_gain_pct", "max_ttft_regression_pct", "max_tpot_regression_pct", "tolerance"):
        _number(rules.get(key), f"rules.{key}")
    if "noise_floor_pct" in rules:
        _number(rules["noise_floor_pct"], "rules.noise_floor_pct")
    if rules.get("time_boundary") != "service_end_to_end":
        raise ContractError("service performance rules must use the service_end_to_end boundary")

    settings = _mapping(task.get("settings"), "settings")
    if settings.get("kernel_backend") not in ("geak", "forge", "disabled"):
        raise ContractError("settings.kernel_backend must be geak, forge or disabled")
    for key in ("framework_enabled", "enablement", "profile_enabled", "sweep_enabled"):
        if type(settings.get(key)) is not bool:
            raise ContractError(f"settings.{key} must be boolean")
    if "record_model_forward" in settings and type(settings["record_model_forward"]) is not bool:
        raise ContractError("settings.record_model_forward must be boolean")
    for key in ("max_cycles", "budget", "max_turns_per_cycle"):
        _integer(settings.get(key), f"settings.{key}", minimum=1)
    _number(settings.get("reprofile_ratio"), "settings.reprofile_ratio", minimum=1)
    _number(settings.get("target_gain_pct"), "settings.target_gain_pct")
    concurrency = [_integer(item, "sweep concurrency", minimum=1) for item in _list(
        settings.get("sweep_concurrency"), "settings.sweep_concurrency"
    )]
    if concurrency != sorted(set(concurrency)):
        raise ContractError("sweep concurrency must be sorted and unique")

    safe_id(task.get("initial_revision"), "initial_revision")
    return task


def validate_reference(reference: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    wanted = {item["id"] for item in task["workload"]["requests"]}
    if set(reference) != wanted:
        raise ContractError("reference IDs must match the frozen request set")
    for request_id, values in reference.items():
        for value in _list(values, f"reference.{request_id}"):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ContractError(f"reference.{request_id} must contain finite numbers")
    return reference


def validate_target(target: dict[str, Any]) -> dict[str, Any]:
    safe_id(target.get("id"), "target.id")
    if target.get("mode") not in ("simulated", "real"):
        raise ContractError("target.mode must be simulated or real")
    devices = [_text(item, "device") for item in _list(target.get("devices"), "target.devices")]
    if len(set(devices)) != len(devices):
        raise ContractError("target.devices must be unique")
    tp_size = _integer(target.get("tp_size"), "target.tp_size", minimum=1)
    if tp_size > len(devices):
        raise ContractError("target.tp_size exceeds visible devices")
    for field in ("software", "model_sha256", "tokenizer_sha256"):
        if field in target:
            _mapping(target[field], f"target.{field}")
    return target
