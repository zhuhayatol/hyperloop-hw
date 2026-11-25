"""Task registry, resource leases and completion bookkeeping."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from typing import Any

EventSink = Callable[[str, dict[str, Any]], None]

_DEFAULT_CAPACITIES = {"benchmark": 1, "research": 1, "workspace": 1}
_ACTION_RESOURCES = {
    "baseline": ("benchmark",),
    "profile": ("benchmark",),
    "roofline": ("benchmark",),
    "explore": ("benchmark",),
    "conc_sweep": ("benchmark",),
    "stack_revalidate": ("benchmark",),
    "replay_warm_recipe": ("benchmark", "workspace"),
    "specialist": ("research",),
    "targeted_build": ("workspace",),
    "enablement": ("benchmark", "workspace"),
    "integrate_patch": ("benchmark", "workspace"),
    "rope_layout_probe": ("benchmark", "workspace"),
    "integrate": ("benchmark", "workspace"),
    "kernel_agent": ("benchmark", "workspace"),
}
_HOLDING_STATUSES = frozenset({"running", "cleanup_pending"})
_FAILED_RESULTS = frozenset({"failure", "failed", "error", "interrupted", "cancelled"})


class Dispatcher:
    """Account for tasks and lease resources without invoking an execution backend."""

    def __init__(self, capacities: Mapping[str, int] | None = None, emit: EventSink | None = None,
                 *, reserved_task_numbers: int = 0) -> None:
        self._capacities = {**_DEFAULT_CAPACITIES, **(capacities or {})}
        if any(type(count) is not int or count < 0 for count in self._capacities.values()):
            raise ValueError("resource capacities must be nonnegative integers")
        self._emit = emit
        self._tasks: dict[str, dict[str, Any]] = {}
        self._task_keys: dict[str, str] = {}
        if type(reserved_task_numbers) is not int or reserved_task_numbers < 0:
            raise ValueError("reserved task numbers must be a nonnegative integer")
        self._reserved_task_numbers = reserved_task_numbers
        self._task_counter = reserved_task_numbers

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._tasks)

    @property
    def has_inflight(self) -> bool:
        return any(task["status"] in _HOLDING_STATUSES for task in self._tasks.values())

    def get(self, task_id: str) -> dict[str, Any]:
        return deepcopy(self._tasks[task_id])

    def enqueue(
        self,
        action: str,
        params: Mapping[str, Any],
        *,
        source: str,
        proposal_id: str | None = None,
        idempotency_key: str | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        resources = dict.fromkeys(_ACTION_RESOURCES.get(action, ()), 1)
        if action == "specialist" and (params.get("needs_gpu") is True or params.get("mode") == "bench"):
            resources["benchmark"] = 1
        estimated_seconds = params.get("estimated_seconds", 1)
        self._validate_budget(estimated_seconds)
        fields = {
            "action": action,
            "params": deepcopy(dict(params)),
            "source": source,
            "proposal_id": proposal_id,
            "idempotency_key": idempotency_key,
            "phase": phase,
            "resources": resources,
            "estimated_seconds": max(1, estimated_seconds),
        }
        if idempotency_key is not None and idempotency_key in self._task_keys:
            existing = self._tasks[self._task_keys[idempotency_key]]
            if any(existing[key] != value for key, value in fields.items()):
                raise ValueError("idempotency key has already identified a different task")
            return deepcopy(existing)
        self._task_counter += 1
        task_id = f"task-{self._task_counter:04d}"
        task = {**fields, "task_id": task_id, "status": "queued"}
        self._tasks[task_id] = task
        if idempotency_key is not None:
            self._task_keys[idempotency_key] = task_id
        self._record("task_queued", task)
        return deepcopy(task)

    def start_ready(
        self, *, phase: str | None = None, remaining_budget: float | None = None
    ) -> list[dict[str, Any]]:
        if remaining_budget is not None:
            self._validate_budget(remaining_budget)
        occupied = dict.fromkeys(self._capacities, 0)
        for task in self._tasks.values():
            if task["status"] in _HOLDING_STATUSES:
                self._account_resources(occupied, task)
        started = []
        for task in self._tasks.values():
            if task["status"] != "queued" or (phase is not None and task["phase"] not in (None, phase)):
                continue
            if remaining_budget is not None and task["estimated_seconds"] > remaining_budget:
                continue
            if any(occupied[lane] + count > self._capacities[lane] for lane, count in task["resources"].items()):
                continue
            self._account_resources(occupied, task)
            if remaining_budget is not None:
                remaining_budget -= task["estimated_seconds"]
            task["status"] = "running"
            self._record("task_started", task)
            started.append(deepcopy(task))
        return started

    @staticmethod
    def _account_resources(occupied: dict[str, int], task: Mapping[str, Any]) -> None:
        for lane, count in task["resources"].items():
            occupied[lane] += count

    @staticmethod
    def _validate_budget(budget: Any) -> None:
        if type(budget) not in {int, float} or budget < 0 or (type(budget) is float and not math.isfinite(budget)):
            raise ValueError("budget must be finite and nonnegative")

    def settle(
        self, task_id: str, result: Mapping[str, Any], cleanup: Mapping[str, Any] | bool
    ) -> dict[str, Any]:
        task = self._tasks[task_id]
        cleaned = {"confirmed": cleanup} if isinstance(cleanup, bool) else dict(cleanup)
        result = deepcopy(dict(result))
        if task.get("result") == result and task.get("cleanup") == cleaned:
            return deepcopy(task)
        if task["status"] not in _HOLDING_STATUSES:
            raise ValueError("only a running task can settle")
        task.update(result=result, cleanup=deepcopy(cleaned))
        if cleaned.get("confirmed") is not True:
            task["status"] = "cleanup_pending"
            self._record("task_cleanup_pending", task)
            return deepcopy(task)
        result_status = result.get("status", "succeeded")
        if result_status in {"interrupted", "cancelled"}:
            task["status"] = result_status
        else:
            task["status"] = "failed" if result_status in _FAILED_RESULTS else "succeeded"
        self._record("task_settled", task)
        return deepcopy(task)

    def cancel_queued(self, allowed_actions: Collection[str], reason: str, *, new_phase: str | None = None) -> list[str]:
        cancelled = []
        for task in self._tasks.values():
            if task["status"] != "queued":
                continue
            if task["action"] in allowed_actions:
                if new_phase is not None:
                    task["phase"] = new_phase
                continue
            task.update(status="cancelled", reason=reason)
            self._record("task_cancelled", task)
            cancelled.append(task["task_id"])
        return cancelled

    def snapshot(self) -> dict[str, Any]:
        return {"tasks": self.tasks, "task_counter": self._task_counter}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        tasks = deepcopy(dict(snapshot["tasks"]))
        counter = snapshot["task_counter"]
        start = self._reserved_task_numbers + 1
        expected_ids = [f"task-{index:04d}" for index in range(start, counter + 1)] if type(counter) is int else []
        if (
            type(counter) is not int
            or counter - self._reserved_task_numbers != len(tasks)
            or tasks.keys() != set(expected_ids)
            or any(task["task_id"] != task_id for task_id, task in tasks.items())
        ):
            raise ValueError("invalid dispatcher checkpoint")
        self._task_keys = {
            task["idempotency_key"]: task_id for task_id, task in tasks.items() if task["idempotency_key"] is not None
        }
        self._tasks = {task_id: tasks[task_id] for task_id in expected_ids}
        self._task_counter = counter

    def _record(self, kind: str, task: Mapping[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, deepcopy(dict(task)))
