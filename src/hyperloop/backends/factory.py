"""Create a provider and validate the hardware identity frozen by a Session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from ..contracts import ContractError
from .runtime import Record, RuntimeBackend, fingerprint
from .simulated import SimulatedBackend

RuntimeBackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], RuntimeBackend]


def build_runtime(
    target: Mapping[str, Any],
    scenario: Mapping[str, Any],
    initial_revision: str,
    factory: RuntimeBackendFactory | None,
) -> RuntimeBackend:
    """Give providers copies so their inputs remain frozen by the Coordinator."""
    if factory is not None:
        return factory(deepcopy(target), deepcopy(scenario))
    if target.get("mode") != "simulated":
        raise ContractError("a real target requires an explicitly supplied runtime provider")
    revisions = scenario.get("revisions")
    if not isinstance(revisions, dict) or initial_revision not in revisions:
        raise ContractError("the scenario must define an observation for the original revision")
    if any(not isinstance(spec, dict) for spec in revisions.values()):
        raise ContractError("simulated revision observations must be JSON objects")
    return SimulatedBackend(target, revisions)


def checked_probe(
    backend: RuntimeBackend,
    target: Mapping[str, Any],
    *,
    mode: str,
    locked: Mapping[str, Any] | None = None,
) -> Record:
    """Check declared target constraints and lock the observed environment and capabilities."""
    if target.get("mode") != mode:
        raise ContractError("runtime target mode does not match the Session")
    probe = backend.probe()
    if not isinstance(probe, dict) or probe.get("mode") != mode:
        raise ContractError("runtime probe mode does not match the Session")
    environment = probe.get("environment")
    if not isinstance(environment, dict) or environment.get("mode") != mode:
        raise ContractError("runtime environment mode does not match the Session")
    if type(environment.get("tp_size")) is not int:
        raise ContractError("runtime probe tp_size must be an integer")
    for declared, observed in (("id", "target_id"), ("devices", "devices"), ("tp_size", "tp_size")):
        if target[declared] != environment.get(observed):
            raise ContractError(f"runtime probe does not match target.{declared}")
    ranks = environment.get("rank_map")
    expected_ranks = {str(rank) for rank in range(target["tp_size"])}
    if (
        not isinstance(ranks, dict)
        or ranks.keys() != expected_ranks
        or any(not isinstance(device, str) or device not in target["devices"] for device in ranks.values())
        or len(set(ranks.values())) != len(ranks)
    ):
        raise ContractError("runtime probe has an invalid rank-to-device mapping")
    software = environment.get("software", {})
    if not isinstance(software, dict) or any(software.get(key) != value for key, value in target.get("software", {}).items()):
        raise ContractError("runtime probe does not match target.software")
    capabilities = probe.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(name, str) or not name for name in capabilities)
        or len(set(capabilities)) != len(capabilities)
    ):
        raise ContractError("runtime probe capabilities must be unique names")
    try:
        observed_fingerprint = fingerprint(environment)
        fingerprint(probe)
    except (TypeError, ValueError) as error:
        raise ContractError("runtime probe must be a finite JSON record") from error
    if probe.get("environment_fingerprint") != observed_fingerprint:
        raise ContractError("runtime probe environment fingerprint is invalid")
    if locked is not None:
        locked_capabilities = locked.get("capabilities")
        if (
            locked.get("mode") != mode
            or locked.get("environment_fingerprint") != observed_fingerprint
            or not isinstance(locked_capabilities, list)
            or any(not isinstance(name, str) for name in locked_capabilities)
            or sorted(locked_capabilities) != sorted(capabilities)
        ):
            raise ContractError("target identity changed after Session creation")
    return deepcopy(probe)
