"""Write raw evidence outside a candidate's workspace."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class EvidenceProblem(RuntimeError):
    """A populated observation was incomplete or unusable for acceptance."""

    def __init__(self, reason: str, state: str = "conflict", status: str = "failed") -> None:
        super().__init__(reason)
        self.reason = reason
        self.state = state
        self.status = status


class AttemptEvidence:
    """An append-only evaluation attempt anchored under the session directory."""

    def __init__(self, root: Path, attempt_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt_id):
            raise ValueError("attempt_id must be a single path-safe identifier")
        self.root = root.resolve()
        self.attempt_id = attempt_id
        self.path = self.root / "evaluation" / attempt_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.records: dict[str, dict[str, str]] = {}

    def save(self, kind: str, data: Any, producer: str) -> None:
        path = self.path / f"{kind}.json"
        encoded = (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with path.open("xb") as evidence_file:
            evidence_file.write(encoded)
        self.records[kind] = {
            "id": f"{self.attempt_id}:{kind}",
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "producer": producer,
        }

    def workspace(self, label: str) -> Path:
        return self.root / "workspaces" / self.attempt_id / label
