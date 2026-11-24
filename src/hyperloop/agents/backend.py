"""Agent backends emit structured intents and never dispatch work themselves."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

AGENT_ROLES = frozenset({"orchestration", "critic", "specialist"})


class AgentBackend(Protocol):
    def run(self, role: str, context: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Emit intents for one role turn."""
        ...


class ReplayAgentBackend:
    """Replay role turns deterministically without using a model service."""

    def __init__(self, responses: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]]) -> None:
        if not isinstance(responses, Mapping):
            raise ValueError("agent fixture must map roles to turn arrays")
        self._responses: dict[str, list[list[dict[str, Any]]]] = {}
        for role, turns in responses.items():
            if role not in AGENT_ROLES or not isinstance(turns, (list, tuple)):
                raise ValueError(f"invalid agent fixture role or turn list: {role!r}")
            self._responses[role] = [self._parse_turn(role, turn) for turn in turns]
        self._cursors = {role: 0 for role in AGENT_ROLES}

    @staticmethod
    def _parse_turn(role: str, turn: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(turn, (list, tuple)):
            raise ValueError(f"{role} fixture turn must be an intent array")
        if any(
            not isinstance(intent, Mapping)
            or not isinstance(intent.get("kind", intent.get("type")), str)
            or not intent.get("kind", intent.get("type"))
            for intent in turn
        ):
            raise ValueError(f"{role} fixture intents must be objects with a kind")
        return json.loads(json.dumps([dict(intent) for intent in turn], allow_nan=False))

    @classmethod
    def from_path(cls, path: str | Path) -> ReplayAgentBackend:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def run(self, role: str, context: Mapping[str, Any]) -> list[dict[str, Any]]:
        if role not in self._cursors:
            raise ValueError(f"unknown agent role: {role!r}")
        if role == "critic" and not context.get("pending_proposals"):
            return []
        index = self._cursors[role]
        turns = self._responses.get(role, [])
        if index >= len(turns):
            return []
        self._cursors[role] += 1
        return deepcopy(turns[index])

    def has_next(self, role: str) -> bool:
        """An explicit empty turn still occupies a saved response slot."""
        return self._cursors[role] < len(self._responses.get(role, ()))

    def snapshot(self) -> dict[str, Any]:
        return {"cursors": dict(self._cursors)}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        cursors = snapshot.get("cursors")
        if not isinstance(cursors, Mapping) or cursors.keys() != self._cursors.keys():
            raise ValueError("agent checkpoint must contain every role cursor")
        for role, index in cursors.items():
            if type(index) is not int or not 0 <= index <= len(self._responses.get(role, [])):
                raise ValueError(f"agent checkpoint has an invalid {role} cursor")
        self._cursors = dict(cursors)
