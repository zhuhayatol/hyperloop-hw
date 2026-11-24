"""Validate reusable settings from an earlier closed Session."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..contracts import ContractError, digest, json_bytes, read_json, safe_id
from ..session import SessionStore

RECIPE_PATH = "evidence/recipes/final.json"
_KEYS = frozenset({
    "schema_version", "mode", "target_id", "environment_fingerprint", "model_fingerprint",
    "workload_fingerprint", "source_session_id", "revision", "parameters", "requires_revalidation",
})


def _scope(task: dict, target: dict, runtime: dict) -> dict[str, Any]:
    return {
        "mode": target["mode"],
        "target_id": target["id"],
        "environment_fingerprint": runtime["environment_fingerprint"],
        "model_fingerprint": digest(task["model"]),
        "workload_fingerprint": digest(task["workload"]),
    }


def build_recipe(
    task: dict,
    target: dict,
    runtime: dict,
    *,
    source_session_id: str,
    revision: str,
    parameters: dict,
) -> dict[str, Any]:
    """Freeze only portable parameter settings and their measured scope."""
    recipe = {
        "schema_version": 1,
        **_scope(task, target, runtime),
        "source_session_id": source_session_id,
        "revision": revision,
        "parameters": deepcopy(parameters),
        "requires_revalidation": True,
    }
    _validate_payload(recipe)
    return recipe


def _validate_payload(recipe: Any) -> None:
    if not isinstance(recipe, dict) or recipe.keys() != _KEYS:
        raise ContractError("recipe fields do not match the supported schema")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ContractError("unsupported recipe schema version")
    if recipe["requires_revalidation"] is not True:
        raise ContractError("recipe must require independent revalidation")
    if recipe["mode"] not in ("simulated", "real"):
        raise ContractError("unsupported recipe mode")
    for key in ("target_id", "source_session_id", "revision"):
        safe_id(recipe[key], f"recipe.{key}")
    for key in ("environment_fingerprint", "model_fingerprint", "workload_fingerprint"):
        value = recipe[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ContractError(f"recipe.{key} must be a SHA-256 digest")
    parameters = recipe["parameters"]
    if not isinstance(parameters, dict) or any(not isinstance(key, str) or not key for key in parameters):
        raise ContractError("recipe parameters must map nonempty names to finite JSON values")
    try:
        json_bytes(parameters)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("recipe parameters must contain finite JSON values") from error


def validate_recipe(recipe: Any, task: dict, target: dict, runtime: dict) -> dict[str, Any]:
    """Classify compatibility while keeping settings advisory until remeasured."""
    try:
        _validate_payload(recipe)
    except ContractError as error:
        return {"status": "invalid", "reason": "invalid_recipe", "detail": str(error)}

    origin = {
        "source_session_id": recipe["source_session_id"],
        "revision": recipe["revision"],
        "requires_revalidation": True,
    }
    scope = _scope(task, target, runtime)
    for key, label in (
        ("mode", "mode"), ("target_id", "target"), ("model_fingerprint", "model"),
        ("workload_fingerprint", "workload"), ("environment_fingerprint", "environment"),
    ):
        if recipe[key] != scope[key]:
            return {**origin, "status": "incompatible", "reason": f"{label}_mismatch"}
    allowed = task.get("allowed_parameters")
    if allowed is not None and set(recipe["parameters"]) - set(allowed):
        return {**origin, "status": "incompatible", "reason": "parameter_not_allowed"}
    return {
        **origin,
        "status": "applicable",
        "reason": "matched_frozen_scope",
        "parameters": deepcopy(recipe["parameters"]),
    }


def load_verified_recipe(path: str | Path) -> dict[str, Any]:
    """Read a completed Session's immutable, indexed final recipe."""
    location = Path(path).resolve(strict=True)
    if location.parts[-3:] != ("evidence", "recipes", "final.json") or not location.is_file():
        raise ContractError(f"recipe must be a completed Session's {RECIPE_PATH}")
    store = SessionStore(location.parents[2])
    state = store.state()
    if state["phase"] != "DONE" or not state["stop_reason"]:
        raise ContractError("recipe source Session is not closed")
    records = [
        item for item in state["recipes"]
        if item.get("reference", {}).get("path") == RECIPE_PATH
    ]
    if len(records) != 1:
        raise ContractError("recipe source must record one final recipe")
    item = records[0]
    reference = item["reference"]
    if reference.get("producer") != "close" or not store.verify_evidence(reference):
        raise ContractError("recipe evidence digest does not match the Session")

    recipe = read_json(location)
    _validate_payload(recipe)
    recorded = {key: value for key, value in item.items() if key != "reference"}
    current = state.get("current_revision") or {}
    if (
        digest(recipe) != reference["sha256"]
        or recipe != recorded
        or recipe["source_session_id"] != store.manifest["session_id"]
        or recipe["revision"] != current.get("id")
        or recipe["parameters"] != current.get("parameters", {})
    ):
        raise ContractError("recipe does not describe the Session's final accepted revision")

    runtime = state.get("runtime")
    if not runtime or validate_recipe(recipe, store.manifest["task"], store.manifest["target"], runtime)["status"] != "applicable":
        raise ContractError("recipe source scope is inconsistent with the Session records")
    return recipe
