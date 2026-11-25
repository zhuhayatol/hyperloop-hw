"""Load the most recent compatible, independently measured original baseline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

from .actions.candidates import original_revision, stack_identity
from .contracts import ContractError, digest, read_json, validate_reference
from .evaluation.adjudication import verify_service_retest
from .evaluation.evidence import EvidenceProblem
from .session import SessionStore, atomic_write


@dataclass
class HistoricalBaseline:
    source: SessionStore
    reference: dict
    baseline: dict
    profile: dict | None
    provenance: dict
    files: dict[str, bytes]
    traces: dict[str, dict]
    last_task_number: int

    def copy_evidence(self, store: SessionStore) -> None:
        for path, content in self.files.items():
            atomic_write(store.artifact_path(path), content)
        for path, trace in self.traces.items():
            destination = store.artifact_path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(self.source.artifact_path(path), destination)
            _check_trace(destination, trace)


def _evidence_bytes(store: SessionStore, *records: dict) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for record in records:
        for link in record.values():
            path = link["path"]
            if not isinstance(path, str) or not path.startswith(("evaluation/", "evidence/")):
                raise ContractError("historical evidence has an invalid path")
            content = store.artifact_path(path).read_bytes()
            if hashlib.sha256(content).hexdigest() != link["sha256"]:
                raise ContractError(f"historical evidence hash mismatch: {path}")
            files[path] = content
    return files


def _check_trace(path: Path, trace: dict) -> None:
    with path.open("rb") as stream:
        if path.stat().st_size != trace["bytes"] or hashlib.file_digest(stream, "sha256").hexdigest() != trace["sha256"]:
            raise ContractError(f"historical Profile trace does not match its receipt: {path}")


def _matching_profile(
    store: SessionStore, events: list[dict], task: dict, runtime: dict,
) -> tuple[dict, dict[str, bytes], dict[str, dict]] | None:
    if not task["settings"]["profile_enabled"] or "profile" not in runtime["capabilities"]:
        return None
    revision = original_revision(task["initial_revision"])
    ranks = list(range(store.manifest["target"]["tp_size"]))
    for event in reversed(events):
        if event["type"] != "profile_recorded":
            continue
        profile = event["payload"]
        if (
            profile.get("reused_from") or profile.get("status") != "ok"
            or profile.get("revision") != revision["id"] or profile.get("point") != task["workload"]["id"]
            or profile.get("environment_fingerprint") != runtime["environment_fingerprint"]
            or profile.get("mode") != store.manifest["mode"]
            or profile.get("ranks") != ranks or profile.get("missing_ranks") != []
            or profile.get("runtime", {}).get("loaded_stack") != stack_identity(revision)
        ):
            continue
        try:
            links = profile["evidence"]
            analysis = links["profile_analysis"]
            files = _evidence_bytes(store, links, {"receipt": profile["receipt"]})
            report = read_json(store.artifact_path(analysis["path"]))
            if (
                profile["report_ref"] != analysis
                or any(report[key] != profile[key] for key in ("ranks", "missing_ranks", "hotspots", "capture"))
                or read_json(store.artifact_path(links["workload"]["path"])) != task["workload"]
                or read_json(store.artifact_path(links["profile_cleanup"]["path"])).get("released") is not True
            ):
                continue
            raw = read_json(store.artifact_path(links["profile_raw"]["path"]))
            traces = {trace["path"]: trace for row in raw["ranks"] for trace in row.get("trace_files", [])}
            for path, trace in traces.items():
                if not path.startswith(f"workspaces/{profile['attempt_id']}/profile/"):
                    raise ContractError("historical Profile trace must belong to its capture workspace")
                _check_trace(store.artifact_path(path), trace)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            continue
        return event, files, traces
    return None


def latest_baseline(history: Path, task: dict, target: dict, runtime: dict, scenario: dict) -> HistoricalBaseline:
    history = history.resolve()
    mode = target["mode"]
    revision = original_revision(task["initial_revision"])
    search = (*history.glob(f"{mode}/*/manifest.json"), *history.glob(f"*/{mode}/*/manifest.json"))
    candidates = []
    for path in set(search):
        try:
            store = SessionStore(path.parent)
            manifest = store.manifest
            historical_task = manifest["task"]
            if (
                not store.path.is_relative_to(history) or manifest["target"] != target
                or manifest.get("baseline_reuse") or manifest.get("targeted_origin")
                or manifest["runtime"]["environment_fingerprint"] != runtime["environment_fingerprint"]
                or any(historical_task[key] != task[key] for key in ("initial_revision", "model", "workload", "rules"))
                or bool(historical_task["settings"].get("record_model_forward")) != bool(task["settings"].get("record_model_forward"))
            ):
                continue
            if mode == "simulated" and (
                manifest["scenario"]["revisions"][revision["id"]] != scenario["revisions"][revision["id"]]
            ):
                continue
            events = store.events()
            records = [event for event in events if event["type"] == "baseline_recorded"]
            if len(records) != 1:
                continue
            record = records[0]
            result = record["payload"]["result"]
            if (
                not record["payload"].get("reused_from") and result["revision"] == revision
                and result["status"] == "ok" and result["acceptance"] == "pass"
                and result["point"] == task["workload"]["id"]
            ):
                candidates.append((record["at"], store, record, events))
        except (KeyError, OSError, TypeError, ValueError):
            continue

    for _, store, event, events in sorted(candidates, key=lambda row: (row[0], str(row[1].path)), reverse=True):
        result = event["payload"]["result"]
        try:
            reference = validate_reference(store.manifest["reference"], task)
            proven = verify_service_retest(store, result, revision)
            if (
                not proven["correctness"]["pass"] or proven["correctness"] != result["correctness"]
                or proven["measurement"] != result["measurement"]
                or proven["instance_id"] != result["runtime"]["instance_id"]
            ):
                continue
            files = _evidence_bytes(store, result["evidence"], result.get("model_forward", {}).get("evidence", {}))
            guard = {"session_id": store.manifest["session_id"], "session": str(store.path),
                     "manifest_sha256": store.manifest["fingerprint"], "baseline_event": event["seq"],
                     "measured_at": event["at"], "measurement": result["evidence"]["revision_measurement"]}
            capture = store.manifest["scenario"].get("reference_capture")
            if capture is not None:
                if (not isinstance(capture, dict) or capture.get("status") != "ok"
                        or capture.get("reference_sha256") != digest(reference)
                        or capture.get("environment_fingerprint") != runtime["environment_fingerprint"]):
                    continue
                guard["reference_capture_sha256"] = digest(capture)
            historical_profile = _matching_profile(store, events, task, runtime)
            profile = None
            traces = {}
            if historical_profile:
                profile_event, profile_files, traces = historical_profile
                guard["profile_event"] = profile_event["seq"]
                guard["profile_captured_at"] = profile_event["at"]
                profile = profile_event["payload"]
                files.update(profile_files)
            reserved = max((int(number) for path in files
                            for number in re.findall(r"(?:^|/)task-(\d+)(?=/|\.json$)", path)), default=0)
            return HistoricalBaseline(store, reference, result, profile, guard, files, traces, reserved)
        except (KeyError, OSError, TypeError, ValueError, EvidenceProblem):
            continue
    raise ContractError(f"no compatible verified baseline in {history}; set baseline_mode: measure to create one")
