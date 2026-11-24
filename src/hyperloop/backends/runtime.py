"""Operations and evidence records exchanged with a runtime adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

Record = dict[str, Any]


def fingerprint(value: Any) -> str:
    """Hash a JSON value using one stable encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BackendError(RuntimeError):
    """An adapter operation could not complete."""


class RuntimeBackend(Protocol):
    """Translate execution requests into raw records for independent evaluation.

    A prepared workspace owns the attempted instance, including any resources
    allocated before launch returns. Cleanup must account for them if launch
    raises. Samples carry observed identity alongside measurements.
    """

    def probe(self) -> Record: ...

    def bind_revision(
        self, revision: Mapping[str, Any], *, expected_stack: Mapping[str, Any]
    ) -> None: ...

    def prepare(self, revision: str, workspace: Path, *, kind: str) -> Record: ...

    def launch(self, prepared: Mapping[str, Any], workload: Mapping[str, Any]) -> Record: ...

    def observe_outputs(
        self, instance: Mapping[str, Any], workload: Mapping[str, Any], *, scope: str
    ) -> Record: ...

    def benchmark(
        self,
        instance: Mapping[str, Any],
        workload: Mapping[str, Any],
        *,
        repeat: int,
        warmup: int,
    ) -> Record | None: ...

    def collect_profile(
        self, instance: Mapping[str, Any], workload: Mapping[str, Any]
    ) -> Record: ...

    def device_timer(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record: ...

    def forward_timer(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record: ...

    def cleanup(
        self, prepared: Mapping[str, Any], instance: Mapping[str, Any] | None
    ) -> Record: ...

    def recover_cleanup(self) -> Record: ...
