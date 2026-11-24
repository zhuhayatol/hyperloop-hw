"""Execute a dispatched action through the target RuntimeBackend and evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..backends import BackendError, RuntimeBackend
from ..contracts import ContractError
from ..evaluation import Evaluator
from ..evaluation.adjudication import adjudicate
from ..evaluation.measurements import change_ratios
from ..kernel.routes import plan_kernel_handoff
from ..session import SessionStore
from .candidates import candidate_revision, original_revision, stack_identity


class RuntimeActions:
    def __init__(self, store: SessionStore, backend: RuntimeBackend):
        self.store = store
        self.backend = backend
        self.evaluator = Evaluator(backend, store.path)
        self.task = store.manifest["task"]
        self.reference = store.manifest["reference"]
        self.scenario = store.manifest["scenario"]
        self.workload = self.task["workload"]
        self.rules = self.task["rules"]

    def register(self, revision: dict[str, Any]) -> None:
        try:
            self.backend.bind_revision(deepcopy(revision), expected_stack=stack_identity(revision))
        except BackendError as error:
            raise ContractError(str(error)) from error

    def measure(self, revision: dict[str, Any], attempt: str, workload: dict[str, Any] | None = None) -> dict:
        self.register(revision)
        return self.evaluator.measure_revision(
            revision["id"], workload or self.workload, self.reference, self.rules,
            attempt_id=attempt, expected_stack=stack_identity(revision),
        )

    def baseline(self, task: dict, _: dict) -> dict:
        revision = original_revision(task["params"]["revision"])
        try:
            result = self.measure(revision, task["task_id"])
        except ContractError as error:
            return {"status": "failed", "reason": str(error), "revision": revision}
        self._with_forward(result, revision, task["task_id"])
        return {**result, "revision": revision, "point": self.workload["id"]}

    def enablement(self, task: dict, _: dict) -> dict:
        fix = task["params"].get("fix")
        if not fix:
            return {"status": "failed", "reason": "no_enablement_candidate"}
        revision = original_revision(fix["revision"])
        revision["parameters"].update(fix.get("parameters", {}))
        try:
            result = self.measure(revision, task["task_id"])
        except ContractError as error:
            return {"status": "failed", "reason": str(error), "revision": revision, "enablement": fix}
        self._with_forward(result, revision, task["task_id"])
        evidence = self.store.put_evidence(
            f"evidence/enablement/{task['task_id']}.json", fix, producer="enablement_specialist"
        )
        result["evidence"]["enablement"] = evidence
        return {**result, "revision": revision, "point": self.workload["id"], "enablement": fix}

    def profile(self, task: dict, state: dict) -> dict:
        revision = state["current_revision"]
        self.register(revision)
        captured = self.evaluator.profile(
            revision["id"], self.workload, attempt_id=task["task_id"], expected_stack=stack_identity(revision)
        )
        return {
            **captured,
            "revision": revision["id"],
            "point": self.workload["id"],
            "anchor_throughput": revision.get("summary", {}).get("throughput"),
            "report_ref": captured["evidence"].get("profile_analysis"),
        }

    @staticmethod
    def _blocked_candidate(spec: dict, parent: dict, reason: str, *, kind: str) -> dict:
        return {
            "candidate_id": spec.get("id"),
            "parent": parent["id"],
            "parent_revision": parent,
            "kind": kind,
            "status": "blocked",
            "version_action": "HOLD",
            "acceptance": "undetermined",
            "evidence_state": "conflict",
            "reason": reason,
            "evidence": {},
        }

    def evaluate_candidate(
        self, spec: dict, parent: dict, baseline: dict, attempt: str, kind: str
    ) -> dict:
        try:
            revision = candidate_revision(spec, parent, kind)
            self.register(revision)
            self.register(parent)
        except ContractError as error:
            return self._blocked_candidate(spec, parent, str(error), kind=kind)
        evaluated = self.evaluator.candidate(
            revision["id"], parent["id"], self.workload, self.reference, self.rules,
            attempt_id=attempt,
            baseline=baseline,
            kind="kernel" if kind == "kernel" else "framework",
            kernel_reference=self.scenario.get("kernel_reference"),
            expected_stack=stack_identity(revision),
            expected_parent_stack=stack_identity(parent),
        )
        proposal = self.store.put_evidence(
            f"evidence/candidates/{attempt}.json", spec, producer="candidate_submission"
        )
        evaluated["evidence"]["proposal"] = proposal
        self._with_forward(evaluated, revision, attempt)
        return {
            **evaluated,
            "candidate_id": revision["id"],
            "kind": kind,
            "backend": spec.get("backend"),
            "parent": parent["id"],
            "parent_revision": parent,
            "revision": revision,
            "profile_ref": spec.get("profile_ref"),
        }

    def _with_forward(self, result: dict, revision: dict, attempt: str) -> None:
        if result.get("status") == "ok" and result.get("measurement") and self.task["settings"].get("record_model_forward"):
            forward = self.evaluator.forward(
                revision["id"], self.workload, self.reference, self.rules,
                attempt_id=f"{attempt}-forward", expected_stack=stack_identity(revision),
            )
            result["model_forward"] = forward
            if forward.get("reason") == "cleanup_unconfirmed":
                result.update(
                    status="blocked", version_action="HOLD", acceptance="undetermined",
                    evidence_state="missing", reason="cleanup_unconfirmed",
                )

    def explore(self, task: dict, state: dict) -> dict:
        seen: set[str] = set()
        decisions = []
        for index, spec in enumerate(task["params"].get("candidates", [])):
            if not isinstance(spec, dict) or not isinstance(spec.get("id"), str) or spec["id"] in seen:
                continue
            seen.add(spec["id"])
            decisions.append(self.evaluate_candidate(
                self._submitted(spec, task), state["current_revision"], state["baseline"],
                f"{task['task_id']}-p{index}", "parameters",
            ))
            if decisions[-1]["reason"] == "cleanup_unconfirmed":
                break
        qualified = [item for item in decisions if item["version_action"] == "KEEP"]
        if qualified:
            first = max(qualified, key=lambda item: item["measurement"]["summary"]["throughput"])
            for decision in qualified:
                if decision is not first:
                    decision.update(version_action="HOLD", reason="stronger_candidate_selected")
        return {"status": "ok", "decisions": decisions}

    def integrate(self, task: dict, state: dict) -> dict:
        spec = task["params"].get("candidate")
        if not isinstance(spec, dict):
            return {"status": "failed", "reason": "missing_patch_candidate"}
        decision = self.evaluate_candidate(
            self._submitted(spec, task), state["current_revision"], state["baseline"], task["task_id"], "patch"
        )
        return {"status": "ok", "decisions": [decision]}

    @staticmethod
    def _submitted(spec: dict, task: dict) -> dict:
        params = task["params"]
        return {
            **({"parent": params["parent"]} if "parent" in params else {}),
            **({"profile_ref": params["profile_ref"]} if "profile_ref" in params else {}),
            **spec,
        }

    def kernel(self, task: dict, state: dict) -> dict:
        parent = state["current_revision"]
        route = task["params"]["backend"]
        real = self.store.manifest["mode"] == "real"
        last_profile = next((item for item in reversed(state["profiles"]) if item["status"] == "ok" and (
            not real or item.get("revision") == parent["id"]
        )), None)
        if last_profile is not None and not self.store.verify_evidence(last_profile.get("report_ref") or {}):
            last_profile = None
        plans = [] if real else self.scenario.get("kernel_plans", {}).get(route, [])
        handoff, specs = plan_kernel_handoff(
            route, parent, self.workload, last_profile, plans
        )
        if real:
            handoff["reason"] = handoff.get("reason") or "native_kernel_driver_unconfigured"
            handoff["dispatch"] = "blocked_before_native_dispatch"
            handoff["source_correlated_hotspots"] = [hit for hit in (last_profile or {}).get("hotspots", [])
                                                     if hit.get("callsite")]
        if task["params"].get("cycle", 1) > 1:
            settled = {item["candidate_id"] for item in state["decisions"]}
            specs = [spec for spec in specs if spec["id"] not in settled]
            handoff["selected"] = [item["id"] for item in specs]
        handoff_ref = self.store.put_evidence(
            f"evidence/kernel/{task['task_id']}-handoff.json", handoff, producer="kernel_handoff"
        )
        if real:
            return {"status": "blocked", "backend": route, "handoff": handoff, "decisions": [],
                    "reason": handoff["reason"], "evidence": {"handoff": handoff_ref}}
        results = []
        for index, spec in enumerate(specs):
            attempt = f"{task['task_id']}-k{index}"
            evaluated = self._kernel_candidate(spec, parent, state["baseline"], attempt)
            results.append(evaluated)
            if evaluated["version_action"] == "KEEP":
                checked = adjudicate(self.store, evaluated, state["baseline"], self.rules)
                if checked["version_action"] == "KEEP":
                    parent = accepted_revision(checked)
            if evaluated["reason"] == "cleanup_unconfirmed":
                break
        tool_ref = self.store.put_evidence(
            f"evidence/kernel/{task['task_id']}-delivery.json",
            {
                "backend": route, "delivery": handoff["delivery"],
                "candidates": [item["candidate_id"] for item in results],
                "tool_evidence": [item.get("tool_evidence", {}) for item in results],
                "reason": handoff.get("reason") or ("no_candidates" if not results else None),
            },
            producer=f"simulated_{route}",
        )
        return {
            "status": "ok", "backend": route, "handoff": handoff, "decisions": results,
            "reason": handoff.get("reason") or ("no_candidates" if not results else "candidates_settled"),
            "evidence": {"handoff": handoff_ref, "delivery": tool_ref},
        }

    def _kernel_candidate(self, spec: dict, parent: dict, baseline: dict, attempt: str) -> dict:
        try:
            revision = candidate_revision(spec, parent, "kernel")
            self.register(revision)
        except ContractError as error:
            return self._blocked_candidate(spec, parent, str(error), kind="kernel")
        kernel_reference = self.scenario.get("kernel_reference") or self.reference
        tool = self.evaluator.kernel(
            revision["id"], self.workload, kernel_reference, self.rules,
            attempt_id=f"{attempt}-tool", expected_stack=stack_identity(revision),
        )
        tool_evidence = {"operator": tool["evidence"]}
        publication = {
            "mode": self.store.manifest["mode"], "candidate_id": spec["id"], "operator": spec["operator"],
            "parent": parent["id"], "status": tool["status"], "reason": tool["reason"],
        }
        if tool["acceptance"] == "pass" and spec["backend"] == "geak":
            tool = self.measure(revision, f"{attempt}-tool-service")
            tool_evidence["service"] = tool["evidence"]
            publication.update(
                delivery="service_candidate", configuration=revision["parameters"],
                loadable_stack=stack_identity(revision), service=(tool.get("measurement") or {}).get("summary"),
            )
        elif spec["backend"] == "forge":
            publication.update(
                delivery="operator_patch", source_location=spec["hotspot"].get("callsite"),
                candidate_stack=revision["kernels"], device_timing=tool.get("device_timing"),
            )
        publication.update(status=tool["status"], reason=tool["reason"])
        publication["upstream_evidence"] = tool_evidence
        tool_evidence["publication"] = self.store.put_evidence(
            f"evidence/kernel/{attempt}-{spec['backend']}-publication.json", publication, producer=f"simulated_{spec['backend']}",
        )
        if tool["acceptance"] == "pass":
            result = self.evaluate_candidate(spec, parent, baseline, attempt, "kernel")
        else:
            result = {
                **tool, "candidate_id": revision["id"], "revision": revision,
                "parent": parent["id"], "parent_revision": parent, "kind": "kernel",
                "backend": spec["backend"], "profile_ref": spec["profile_ref"],
            }
        return {**result, "tool_evidence": tool_evidence}

    def stack(self, task: dict, state: dict) -> dict:
        revision = state["current_revision"]
        result = self.measure(revision, task["task_id"])
        if result["acceptance"] == "pass":
            result["comparison"] = change_ratios(result["measurement"], state["baseline"]["measurement"])
        return {**result, "revision": revision["id"], "reason_context": task["params"].get("reason")}

    def sweep(self, task: dict, state: dict) -> dict:
        original = state["baseline"]["revision"]
        current = state["current_revision"]
        pairs = []
        status = "ok"
        reason = "all_pairs_complete"
        requested = self.task["settings"]["sweep_concurrency"]
        levels = task["params"].get("concurrency_levels", requested)
        for level in levels:
            point = {**self.workload, "id": f"{self.workload['id']}-c{level}", "concurrency": level}
            base = self.measure(original, f"{task['task_id']}-c{level}-base", point)
            pair = {"concurrency": level, "original": base, "candidate": None}
            pairs.append(pair)
            if base["acceptance"] != "pass":
                status, reason = "failed", base["reason"]
                break
            candidate = self.measure(current, f"{task['task_id']}-c{level}-accepted", point)
            pair["candidate"] = candidate
            if candidate["acceptance"] != "pass":
                status, reason = "failed", candidate["reason"]
                break
            pair["comparison"] = change_ratios(candidate["measurement"], base["measurement"])
        report = {
            "status": status, "reason": reason, "revision": current["id"], "pairs": pairs,
            "configured_levels": requested, "scheduled_levels": levels,
            "budget_omissions": [level for level in requested if level not in levels],
        }
        ref = self.store.put_evidence(f"evidence/sweep/{task['task_id']}.json", report, producer="sweep")
        return {**report, "evidence": {"paired_sweep": ref}}


def accepted_revision(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **deepcopy(decision["revision"]),
        "summary": deepcopy(decision["measurement"]["summary"]),
        "validation_ref": deepcopy(decision["evidence"]["candidate_measurement"]),
    }
