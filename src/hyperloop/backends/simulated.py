"""Deterministic local adapter with no hardware or subprocess dependencies."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .runtime import BackendError, Record, fingerprint


class SimulatedBackend:
    """Return observations from target and revision fixtures.

    Every operation is local and marked simulated. Revisions may override
    throughput, TTFT, TPOT, numeric outputs, loaded stack, or a fault. Numeric
    metrics support a scalar or a per-repeat list.
    """

    def __init__(
        self,
        target: Mapping[str, Any],
        revisions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._target = copy.deepcopy(dict(target))
        self._revisions = {name: copy.deepcopy(dict(spec)) for name, spec in (revisions or {}).items()}
        self._active: str | None = None
        self.calls: list[tuple[str, str | None]] = []

    def register_revision(self, revision: str, spec: Mapping[str, Any]) -> None:
        """Register a distinct simulated revision for later evaluation."""
        self._revisions[revision] = copy.deepcopy(dict(spec))

    def _spec(self, revision: str) -> Record:
        return self._revisions.get(revision, {})

    def _faults(self, revision: str) -> set[str]:
        spec = self._spec(revision)
        faults = spec.get("faults", spec.get("fault", ()))
        return {faults} if isinstance(faults, str) else set(faults)

    def _environment(self) -> Record:
        devices = self._target["devices"]
        tp_size = self._target["tp_size"]
        return {
            "mode": "simulated",
            "target_id": self._target["id"],
            "devices": list(devices),
            "tp_size": tp_size,
            "rank_map": {str(rank): device for rank, device in enumerate(devices[:tp_size])},
            "software": copy.deepcopy(self._target.get("software", {})),
        }

    def probe(self) -> Record:
        self.calls.append(("probe", None))
        environment = self._environment()
        default_capabilities = ["verify", "benchmark", "profile", "kernel"]
        if environment["tp_size"] > 1:
            default_capabilities.append("hccl")
        capabilities = self._target.get("capabilities", default_capabilities)
        if isinstance(capabilities, dict):
            capabilities = [name for name, supported in capabilities.items() if supported]
        return {
            "mode": "simulated",
            "environment": environment,
            "environment_fingerprint": fingerprint(environment),
            "capabilities": sorted(capabilities),
        }

    def prepare(self, revision: str, workspace: Path, *, kind: str) -> Record:
        self.calls.append(("prepare", revision))
        if self._active is not None:
            raise BackendError("simulated resource is still held by an earlier attempt")
        if "prepare" in self._faults(revision):
            raise BackendError("simulated preparation failure")
        workspace.mkdir(parents=True, exist_ok=False)
        identity = {"mode": "simulated", "revision": revision, "kind": kind}
        (workspace / "prepared.json").write_text(json.dumps(identity) + "\n", encoding="utf-8")
        return {**identity, "workspace": str(workspace)}

    def launch(self, prepared: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        revision = prepared["revision"]
        self.calls.append(("launch", revision))
        if self._active is not None:
            raise BackendError("simulated resource is already held")
        self._active = prepared["workspace"]
        faults = self._faults(revision)
        if "launch" in faults:
            raise BackendError("simulated launch failure after resource acquisition")
        loaded_revision = f"old-{revision}" if "old_implementation" in faults else revision
        spec = self._spec(revision)
        loaded_stack = copy.deepcopy(spec.get("loaded_stack", {
            "framework": spec.get("framework", loaded_revision),
            "kernels": spec.get("kernels", []),
        }))
        environment = self._environment()
        environment_fingerprint = fingerprint(environment)
        if "wrong_environment" in faults:
            environment_fingerprint = fingerprint({"mode": "simulated", "different": True})
        return {
            "mode": "simulated",
            "instance_id": fingerprint({"workspace": self._active})[:16],
            "revision": revision,
            "loaded_revision": loaded_revision,
            "loaded_stack": loaded_stack,
            "rank_map": environment["rank_map"],
            "environment_fingerprint": environment_fingerprint,
            "workload_fingerprint": fingerprint(workload),
            "workspace": self._active,
            "ready": True,
            "candidate_hit": "not_hit" not in faults and loaded_revision == revision,
        }

    def _observation(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        if self._active is None or instance["workspace"] != self._active:
            raise BackendError("simulated instance is no longer active")
        return {
            "mode": "simulated",
            "instance_id": instance["instance_id"],
            "revision": instance["loaded_revision"],
            "environment_fingerprint": instance["environment_fingerprint"],
            "workload_fingerprint": fingerprint(workload),
        }

    @staticmethod
    def _default_output(input_tokens: int | list[int]) -> list[float]:
        return [float(input_tokens if isinstance(input_tokens, int) else sum(input_tokens))]

    def observe_outputs(
        self, instance: Mapping[str, Any], workload: Mapping[str, Any], *, scope: str
    ) -> Record:
        revision = instance["revision"]
        self.calls.append((f"verify_{scope}", revision))
        spec = self._spec(revision)
        key = "kernel_outputs" if scope == "kernel" else "outputs"
        configured = spec.get(key, spec.get("outputs", {}))
        corrupt = "wrong_output" in self._faults(revision) or f"wrong_{scope}_output" in self._faults(revision)
        cases = []
        for request in workload["requests"]:
            output = list(configured.get(request["id"], self._default_output(request["input_tokens"])))
            if corrupt and not cases:
                output = [output[0] + 1000, *output[1:]] if output else [1000.0]
            cases.append({
                "request_id": request["id"],
                "input_tokens": copy.deepcopy(request["input_tokens"]),
                "dtype": workload.get("dtype", "float32"),
                "shape": [len(output)],
                "output": output,
            })
        return {**self._observation(instance, workload), "scope": scope, "cases": cases}

    @staticmethod
    def _metric(spec: Mapping[str, Any], field: str, default: float, repeat: int) -> float:
        value = spec.get(field, default)
        return float(value[repeat] if isinstance(value, (list, tuple)) else value)

    def benchmark(
        self,
        instance: Mapping[str, Any],
        workload: Mapping[str, Any],
        *,
        repeat: int,
        warmup: int,
    ) -> Record | None:
        revision = instance["revision"]
        self.calls.append(("benchmark", revision))
        spec = self._spec(revision)
        faults = self._faults(revision)
        if repeat and "partial_samples" in faults:
            return None
        try:
            throughput = self._metric(spec, "throughput", 100.0, repeat)
            ttft_ms = self._metric(spec, "ttft_ms", 125.0, repeat)
            tpot_ms = self._metric(spec, "tpot_ms", 28.0, repeat)
        except IndexError:
            return None
        requests = []
        for request in workload["requests"]:
            output_tokens = request["output_tokens"]
            if not requests and "short_output" in faults:
                output_tokens = max(0, output_tokens - 1)
            requests.append({
                "request_id": request["id"],
                "input_tokens": copy.deepcopy(request["input_tokens"]),
                "output_tokens": output_tokens,
                "status": "ok",
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
            })
        total_tokens = sum(request["output_tokens"] for request in requests)
        return {
            **self._observation(instance, workload),
            "boundary": "service_end_to_end",
            "profile_enabled": "profile_mixed" in faults,
            "repeat": repeat,
            "warmup": warmup,
            "elapsed_seconds": total_tokens / throughput if throughput > 0 else 0,
            "requests": requests,
        }

    def collect_profile(
        self, instance: Mapping[str, Any], workload: Mapping[str, Any]
    ) -> Record:
        revision = instance["revision"]
        self.calls.append(("profile", revision))
        faults = self._faults(revision)
        ranks = range(self._target["tp_size"] - (1 if "profile_missing_rank" in faults else 0))
        return {
            **self._observation(instance, workload),
            "boundary": "profile_capture",
            "clock_domain": "per_rank",
            "ranks": [
                {
                    "rank": rank,
                    "revision": f"old-{revision}" if "profile_wrong_revision" in faults else revision,
                    "events": [{
                        "name": "simulated_attention",
                        "callsite": "simulated://attention",
                        "duration_us": 90.0 + rank,
                    }],
                }
                for rank in ranks
            ],
        }

    def device_timer(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        revision = instance["revision"]
        self.calls.append(("device_timer", revision))
        samples = self._spec(revision).get("device_us", [90.0, 91.0, 89.0])
        return {
            **self._observation(instance, workload),
            "boundary": "kernel_device",
            "samples_us": list(samples) if isinstance(samples, (list, tuple)) else [float(samples)],
        }

    def cleanup(
        self, prepared: Mapping[str, Any], instance: Mapping[str, Any] | None
    ) -> Record:
        revision = prepared["revision"]
        self.calls.append(("cleanup", revision))
        holds_resource = self._active == prepared["workspace"]
        if holds_resource and "cleanup" in self._faults(revision):
            return {"mode": "simulated", "released": False, "reason": "simulated process still active"}
        if holds_resource:
            self._active = None
        return {"mode": "simulated", "released": True}

    def recover_cleanup(self) -> Record:
        """Check adapter ownership before resuming a task after interruption."""
        # This adapter owns in-memory instances only. A new adapter cannot
        # inherit an actual process or device allocation.
        return {"mode": "simulated", "released": self._active is None, "scope": "in_memory"}
