"""Turn a verified SGLang startup assertion into one bounded repair opportunity."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from ..contracts import ContractError, read_json, safe_id
from ..session import SessionStore


_ASSERTION = "chunked_prefill_size must be divisible by page_size"
_MAX_LOG_BYTES = 65536


def _server_integer(args: list, flag: str) -> int | None:
    values = []
    for index, arg in enumerate(args):
        if arg == flag:
            values.append(args[index + 1] if index + 1 < len(args) else "")
        elif isinstance(arg, str) and arg.startswith(f"{flag}="):
            values.append(arg.partition("=")[2])
    if len(values) != 1 or not isinstance(values[0], str) or not re.fullmatch(r"[1-9][0-9]{0,8}", values[0]):
        return None
    return int(values[0])


def observed_startup_failure(store: SessionStore, baseline: dict) -> dict[str, Any] | None:
    """Read only the startup assertion from the failed task's verified launch workspace."""
    target = store.manifest["target"]
    if target.get("mode") != "real" or target.get("provider") != "sglang":
        return None
    if "chunked_prefill_size" not in store.manifest["task"].get("allowed_parameters", ()):
        return None
    args = target.get("sglang", {}).get("server_args", [])
    chunk = _server_integer(args, "--chunked-prefill-size")
    page = _server_integer(args, "--page-size")
    if not chunk or not page or chunk % page == 0:
        return None
    result = baseline.get("result") or {}
    if result.get("status") != "failed" or result.get("reason") != "backend_failure":
        return None
    refs = result.get("evidence", {})
    try:
        task_id = safe_id(baseline["task_id"], "baseline.task_id")
        records = {}
        for key in ("revision_prepared", "backend_failure", "revision_cleanup", "validation_flow"):
            ref = refs[key]
            if ref.get("path") != f"evaluation/{task_id}/{key}.json" or not store.verify_evidence(ref):
                return None
            records[key] = read_json(store.artifact_path(ref["path"]))
        workspace = store.artifact_path(f"workspaces/{task_id}/revision")
        log = workspace / "server.log"
        cleanup = records["revision_cleanup"]
        if (
            records["revision_prepared"].get("workspace") != str(workspace)
            or records["backend_failure"].get("message") != f"SGLang stopped before readiness. See {log}"
            or cleanup.get("released") is not True or cleanup.get("workspace") != str(workspace)
            or not any(
                row.get("operation") == "launch" and row.get("state") == "failed"
                for row in records["validation_flow"].get("steps", [])
            )
        ):
            return None
        if log.is_symlink() or not log.is_file() or log.stat().st_size > _MAX_LOG_BYTES:
            return None
        contents = log.read_bytes()
        if len(contents) > _MAX_LOG_BYTES or f"AssertionError: {_ASSERTION}" not in contents.decode("utf-8", "replace").splitlines():
            return None
    except (ContractError, KeyError, OSError, TypeError, ValueError):
        return None
    return {
        "code": "chunked_prefill_page_alignment", "assertion": _ASSERTION,
        "chunked_prefill_size": chunk, "page_size": page, "failed_baseline": task_id,
        "evidence": refs["backend_failure"],
        "startup_log": {"path": log.relative_to(store.path).as_posix(), "sha256": sha256(contents).hexdigest()},
    }


def accept_startup_repair(intents: list[dict], failure: dict, task_id: str) -> dict | None:
    if len(intents) != 1:
        return None
    proposal = intents[0]
    parameters = proposal.get("parameters")
    if proposal.get("kind") != "repair_startup" or not isinstance(parameters, dict) or parameters.keys() != {
        "chunked_prefill_size"
    }:
        return None
    size = parameters["chunked_prefill_size"]
    chunk, page = failure["chunked_prefill_size"], failure["page_size"]
    if type(size) is not int or size <= 0 or size % page or size == chunk:
        return None
    upper = (chunk // page + 1) * page
    lower = (chunk // page) * page
    if abs(size - chunk) != min(chunk - lower if lower else upper - chunk, upper - chunk):
        return None
    reason = proposal.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 400 or not reason.isprintable():
        return None
    return {
        "revision": safe_id(f"{task_id}-startup-repair", "enablement.revision"), "parameters": parameters,
        "reason": reason, "source": "live_specialist", "startup_failure": failure,
    }
