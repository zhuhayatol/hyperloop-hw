"""Materialize portable Session indexes from committed state and evidence references."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import csv
from io import StringIO
from pathlib import PurePosixPath
from typing import Any

from .contracts import ContractError, json_bytes, safe_id
from .session import SessionStore, atomic_write


_REFERENCE_FIELDS = ("id", "path", "sha256", "producer")
_MEASUREMENT_FIELDS = (
    "mode", "revision", "loaded_stack", "boundary", "environment_fingerprint", "workload_fingerprint", "summary"
)


def _evidence(value: Any) -> Iterator[dict[str, str]]:
    if isinstance(value, Mapping):
        if all(isinstance(value.get(key), str) for key in _REFERENCE_FIELDS):
            yield {key: value[key] for key in _REFERENCE_FIELDS}
        for child in value.values():
            yield from _evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _evidence(child)


def _write(store: SessionStore, reference: str, content: bytes) -> None:
    destination = store.artifact_path(reference)
    expected = store.path.joinpath(*PurePosixPath(reference).parts)
    if destination != expected or expected.is_symlink():
        raise ContractError(f"cannot publish an index through a redirected path: {reference}")
    atomic_write(destination, content)


def _write_json(store: SessionStore, reference: str, data: dict) -> None:
    _write(store, reference, json_bytes(data))


def _revision_id(value: Any) -> Any:
    return value.get("id") if isinstance(value, dict) else value


def _measurement(value: dict | None) -> dict:
    if not value:
        return {}
    return {key: value[key] for key in _MEASUREMENT_FIELDS if key in value}


def _run_summary(run: dict | None) -> dict | None:
    if run is None:
        return None
    return {
        "revision": _revision_id(run.get("revision")),
        "attempt_id": run.get("attempt_id"),
        "status": run.get("status"),
        "reason": run.get("reason"),
        "evidence_state": run.get("evidence_state"),
        "acceptance": run.get("acceptance"),
        "measurement": _measurement(run.get("measurement")),
        "evidence": run.get("evidence", {}),
    }


def _round_bundle(
    scope: dict, round_result: dict, baseline: dict | None, point: str, rules: dict
) -> dict:
    evidence = round_result.get("evidence", {})
    return {
        **scope,
        "candidate_id": round_result["candidate_id"],
        "task_id": round_result.get("task_id"),
        "attempt_id": round_result.get("attempt_id"),
        "proposal_id": round_result.get("proposal_id"),
        "point": point,
        "kind": round_result.get("kind"),
        "backend": round_result.get("backend"),
        "parent": {
            "revision": round_result.get("parent"),
            "measurement": _measurement(round_result.get("parent_measurement")),
            "measurement_ref": evidence.get("parent_measurement"),
        },
        "original_baseline": _run_summary(baseline),
        "candidate": _run_summary(round_result),
        "rules": rules,
        "comparison": round_result.get("comparison"),
        "reason": round_result.get("reason"),
        "version_action": round_result.get("version_action"),
        "correctness_reference": evidence.get("reference"),
        "profile_ref": round_result.get("profile_ref"),
        "receipt": round_result.get("receipt"),
        "tool_evidence": round_result.get("tool_evidence", {}),
    }


def _sweep_pairs(rows: Iterable[dict]) -> list[dict]:
    return [
        {
            "concurrency": row["concurrency"],
            "original": _run_summary(row.get("original")),
            "candidate": _run_summary(row.get("candidate")),
            "comparison": row.get("comparison"),
        }
        for row in rows
    ]


def _sweep_csv(pairs: list[dict]) -> bytes:
    target = StringIO()
    writer = csv.writer(target, lineterminator="\n")
    metrics = ("throughput", "ttft_ms", "tpot_ms")
    writer.writerow(("concurrency", *(
        f"{arm}_{metric}" for arm in ("original", "candidate") for metric in metrics
    )))
    for pair in pairs:
        if pair["comparison"] is None:
            continue
        writer.writerow((pair["concurrency"], *(
            pair[arm]["measurement"].get("summary", {}).get(metric)
            for arm in ("original", "candidate") for metric in metrics
        )))
    return target.getvalue().encode("utf-8")


def publish_indexes(store: SessionStore, state: dict, events: list[dict]) -> None:
    """Publish derived index files without overwriting any original evaluation record."""
    manifest = store.manifest
    task = manifest["task"]
    point = safe_id(task["workload"]["id"], "workload.id")
    scope = {
        "schema_version": 1,
        "session_id": manifest["session_id"],
        "mode": manifest["mode"],
        "event_seq": state["last_seq"],
        "manifest_path": "manifest.json",
        "manifest_fingerprint": manifest["fingerprint"],
    }
    baseline = state["baseline"]
    if baseline is not None:
        _write_json(store, f"baseline/{point}/summary.json", {
            **scope, "point": point, "baseline": _run_summary(baseline),
        })
    for profile in state["profiles"]:
        attempt = safe_id(profile.get("attempt_id") or profile["task_id"], "profile.attempt_id")
        evidence = profile.get("evidence", {})
        ranks = profile.get("ranks", [])
        _write_json(store, f"profile/{attempt}/summary.json", {
            **scope,
            "attempt_id": attempt,
            "point": point,
            "task_id": profile.get("task_id"),
            "revision": profile.get("revision"),
            "status": profile.get("status"),
            "reason": profile.get("reason"),
            "ranks": ranks,
            "missing_ranks": profile.get("missing_ranks", []),
            "rank_evidence": {
                str(rank): evidence[f"profile_rank_{rank}"]
                for rank in ranks if f"profile_rank_{rank}" in evidence
            },
            "hotspots": profile.get("hotspots", []),
            "capture": profile.get("capture"),
            "evidence": evidence,
        })
    for round_result in state["decisions"]:
        candidate = safe_id(round_result["candidate_id"], "candidate.id")
        _write_json(store, f"rounds/{candidate}/evidence_bundle.json",
                    _round_bundle(scope, round_result, baseline, point, task["rules"]))
    for sweep in state["sweeps"]:
        cycle = sweep["cycle"]
        if type(cycle) is not int or cycle < 1:
            raise ContractError("sweep.cycle must be a positive integer")
        pairs = _sweep_pairs(sweep.get("pairs", []))
        _write_json(store, f"sweep/cycle-{cycle}.json", {
            **scope,
            "cycle": cycle,
            "status": sweep.get("status"),
            "reason": sweep.get("reason"),
            "revision": sweep.get("revision"),
            "pairs": pairs,
            "evidence": sweep.get("evidence", {}),
        })
        if any(pair["comparison"] is not None for pair in pairs):
            _write(store, f"sweep/cycle-{cycle}.csv", _sweep_csv(pairs))
    unique = {
        tuple(ref[key] for key in _REFERENCE_FIELDS): ref for ref in _evidence([state, events])
    }
    _write(store, "evidence/index.jsonl", b"".join(json_bytes(unique[key]) for key in sorted(unique)))
    _write(store, "recipes.jsonl", b"".join(json_bytes(recipe) for recipe in state["recipes"]))
