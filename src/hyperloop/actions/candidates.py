"""Create isolated candidate identities from an already accepted revision."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..contracts import ContractError, safe_id


def original_revision(revision_id: str) -> dict[str, Any]:
    return {"id": safe_id(revision_id, "revision"), "parameters": {}, "framework": [], "kernels": []}


def candidate_revision(
    candidate: dict[str, Any], parent: dict[str, Any], kind: str
) -> dict[str, Any]:
    if candidate.get("parent", parent["id"]) != parent["id"]:
        raise ContractError("candidate parent is no longer the accepted revision")
    revision_id = safe_id(candidate.get("id"), "candidate.id")
    if revision_id == parent["id"]:
        raise ContractError("candidate must have a new revision identity")
    for field in ("reason", "expected_metric", "risk", "validation"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ContractError(f"candidate must record {field}")

    revision = deepcopy(parent)
    revision.pop("summary", None)
    revision.pop("validated", None)
    revision["id"] = revision_id
    revision["parent"] = parent["id"]
    entry = {"candidate_id": revision_id, "kind": kind}
    if kind == "parameters":
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict) or not parameters or all(
            parent["parameters"].get(key) == value for key, value in parameters.items()
        ):
            raise ContractError("parameter candidate does not change the accepted configuration")
        revision["parameters"].update(parameters)
        revision["framework"].append({**entry, "parameters": deepcopy(parameters)})
    elif kind == "patch":
        if not isinstance(candidate.get("patch_path"), str) or not candidate["patch_path"]:
            raise ContractError("framework patch must name its changed path")
        entry["patch_path"] = candidate["patch_path"]
        if "source_edit" in candidate:
            entry["source_edit"] = deepcopy(candidate["source_edit"])
        revision["framework"].append(entry)
    elif kind == "kernel":
        if not candidate.get("operator") or not candidate.get("profile_ref"):
            raise ContractError("kernel candidate must name a hotspot and its evidence")
        revision["kernels"].append({
            **entry, "operator": candidate["operator"], "backend": candidate["backend"],
        })
    else:
        raise ContractError(f"unsupported candidate kind: {kind}")
    return revision


def stack_identity(revision: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": deepcopy(revision["framework"]),
        "kernels": deepcopy(revision["kernels"]),
        "parameters": deepcopy(revision["parameters"]),
    }
