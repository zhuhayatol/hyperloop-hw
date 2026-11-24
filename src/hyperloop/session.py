"""Durable Session events, state projection and evidence references."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .contracts import ContractError, digest, json_bytes, read_json, safe_id


class SessionBusy(RuntimeError):
    """Another process holds this Session's lease."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lease(path: Path) -> Iterator[None]:
    with path.open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as error:
            raise SessionBusy(f"Session lease in use: {path.name}") from error
        except OSError as error:
            if os.name == "nt" and error.errno in (13, 36):
                raise SessionBusy(f"Session lease in use: {path.name}") from error
            raise
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _initial_state() -> dict[str, Any]:
    return {
        "last_seq": 0,
        "phase": "PRELUDE",
        "cycle": 1,
        "current_revision": None,
        "baseline": None,
        "runtime": None,
        "tasks": {},
        "proposals": {},
        "inbox": {},
        "control": {},
        "decisions": [],
        "profiles": [],
        "sweeps": [],
        "last_conc_sweep": None,
        "stack_validations": [],
        "recipes": [],
        "retrieved_recipe": None,
        "round_attempts": [],
        "stop_reason": None,
    }


def project(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild state entirely from committed events. The snapshot is a cache."""
    state = _initial_state()
    for event in events:
        kind = event["type"]
        data = event["payload"]
        if kind == "runtime_probed":
            state["runtime"] = data
        elif kind == "phase_changed":
            state["phase"] = data["phase"]
            state["cycle"] = data["cycle"]
        elif kind == "baseline_recorded":
            if state["baseline"] is not None:
                raise ContractError("the original baseline is immutable")
            state["baseline"] = data["result"]
            state["current_revision"] = data["revision"]
        elif kind == "control_checkpoint":
            state["control"] = data
            router = data.get("router", {})
            state["tasks"] = data.get("tasks", data.get("dispatcher", {}).get("tasks", {}))
            state["proposals"] = data.get("proposals", router.get("proposals", {}))
            state["inbox"] = data.get("inbox", router.get("inboxes", {}))
        elif kind == "candidate_decision":
            decision = data["decision"]
            if any(item["candidate_id"] == decision["candidate_id"] for item in state["decisions"]):
                raise ContractError(f"candidate already settled: {decision['candidate_id']}")
            if "promote" in data:
                current = state["current_revision"] or {}
                if decision["version_action"] != "KEEP" or current.get("id") != decision["parent"]:
                    raise ContractError("promotion must refer to the current accepted parent")
                state["current_revision"] = data["promote"]
            state["decisions"].append(decision)
        elif kind == "profile_recorded":
            state["profiles"].append(data)
        elif kind == "sweep_recorded":
            state["sweeps"].append(data)
            state["last_conc_sweep"] = data
        elif kind == "stack_validated":
            state["stack_validations"].append(data)
        elif kind == "recipe_recorded":
            state["recipes"].append(data)
        elif kind == "recipe_retrieved":
            state["retrieved_recipe"] = data
        elif kind == "candidate_attempt":
            state["round_attempts"].append(data)
        elif kind == "stopped":
            state["stop_reason"] = data["reason"]
        state["last_seq"] = event["seq"]
    return state


class SessionStore:
    """Owns the filesystem layout and the single writer for a Session."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        manifest = read_json(self.path / "manifest.json")
        fingerprint = manifest.pop("fingerprint", None)
        if fingerprint != digest(manifest):
            raise ContractError("Session manifest fingerprint mismatch")
        self.manifest = {**manifest, "fingerprint": fingerprint}
        if self.manifest["session_id"] != self.path.name or self.path.parent.name != self.manifest["mode"]:
            raise ContractError("Session mode or identity does not match its directory")

    @classmethod
    def create(cls, root: Path, mode: str, session_id: str, fields: dict[str, Any]) -> SessionStore:
        safe_id(session_id, "session_id")
        if mode not in ("simulated", "real"):
            raise ContractError("unsupported Session mode")
        manifest = {
            **fields,
            "schema_version": 1,
            "session_id": session_id,
            "mode": mode,
            "created_at": utc_now(),
        }
        manifest.pop("fingerprint", None)
        manifest["fingerprint"] = digest(manifest)
        root = root.resolve()
        path = root / mode / session_id
        if not path.resolve().is_relative_to(root):
            raise ContractError("Session directory escapes its root")
        path.mkdir(parents=True)
        atomic_write(path / "manifest.json", json_bytes(manifest))
        atomic_write(path / "events.jsonl", b"")
        atomic_write(path / "state.json", json_bytes(_initial_state()))
        return cls(path)

    @contextmanager
    def execution_lease(self) -> Iterator[None]:
        # A Coordinator must hold this lease while calling an external adapter.
        with file_lease(self.path / ".run.lock"):
            yield

    def _events(self) -> tuple[list[dict[str, Any]], int, bytes]:
        raw = (self.path / "events.jsonl").read_bytes()
        committed_length = raw.rfind(b"\n") + 1
        events: list[dict[str, Any]] = []
        for frame in raw[:committed_length].splitlines():
            try:
                record = json.loads(frame)
                rows = record["events"]
            except (ValueError, KeyError, TypeError) as error:
                raise ContractError("invalid committed event frame") from error
            if not isinstance(rows, list) or not rows:
                raise ContractError("empty committed event frame")
            for event in rows:
                if (
                    not isinstance(event, dict)
                    or event.get("seq") != len(events) + 1
                    or not isinstance(event.get("type"), str)
                    or not isinstance(event.get("payload"), dict)
                    or not isinstance(event.get("at"), str)
                ):
                    raise ContractError("invalid event or non-contiguous event sequence")
                events.append(event)
        return events, committed_length, raw[committed_length:]

    def events(self) -> list[dict[str, Any]]:
        return self._events()[0]

    def state(self) -> dict[str, Any]:
        return project(self.events())

    def commit(self, changes: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if not changes:
            return self.state()
        with file_lease(self.path / ".events.lock"):
            events, committed_length, tail = self._events()
            timestamp = utc_now()
            rows = [
                {"seq": len(events) + index, "at": timestamp, "type": kind, "payload": data}
                for index, (kind, data) in enumerate(changes, start=1)
            ]
            state = project(events + rows)
            frame = json_bytes({"events": rows})
            with (self.path / "events.jsonl").open("r+b") as handle:
                if tail:
                    # Keep the interrupted bytes for audit. Only complete frames replay.
                    label = hashlib.sha256(tail).hexdigest()
                    atomic_write(self.path / "recovery" / f"torn-{label}.bin", tail)
                    handle.truncate(committed_length)
                handle.seek(committed_length)
                handle.write(frame)
                handle.flush()
                os.fsync(handle.fileno())
            atomic_write(self.path / "state.json", json_bytes(state))
            return state

    def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.commit([(kind, payload)])

    def artifact_path(self, reference: str) -> Path:
        if not isinstance(reference, str) or "\\" in reference or ":" in reference:
            raise ContractError("evidence paths must be relative POSIX paths")
        relative = PurePosixPath(reference)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ContractError("evidence path escapes the Session")
        resolved = (self.path / relative).resolve()
        if not resolved.is_relative_to(self.path):
            raise ContractError("evidence path escapes the Session")
        return resolved

    def put_evidence(self, reference: str, data: Any, *, producer: str) -> dict[str, str]:
        location = self.artifact_path(reference)
        parts = PurePosixPath(reference).parts
        if len(parts) < 2 or parts[0] != "evidence":
            raise ContractError("Session-owned evidence must be stored under evidence/")
        content = json_bytes(data)
        with file_lease(self.path / ".evidence.lock"):
            if location.exists():
                if location.read_bytes() != content:
                    raise ContractError(f"evidence is immutable: {reference}")
            else:
                atomic_write(location, content)
        return {
            "id": hashlib.sha256(content).hexdigest()[:16],
            "sha256": hashlib.sha256(content).hexdigest(),
            "path": reference,
            "producer": producer,
        }

    def verify_evidence(self, item: dict[str, str]) -> bool:
        try:
            content = self.artifact_path(item["path"]).read_bytes()
            return hashlib.sha256(content).hexdigest() == item["sha256"]
        except (ContractError, KeyError, OSError, TypeError):
            return False
