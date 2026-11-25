"""Execute a dispatched action through the target RuntimeBackend and evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..backends import BackendError, RuntimeBackend
from ..contracts import ContractError
from ..evaluation import Evaluator
from ..evaluation.adjudication import adjudicate, native_operator_publication
from ..evaluation.evidence import EvidenceProblem
from ..evaluation.goal import goal_checks
from ..evaluation.measurements import change_ratios
from ..kernel.ascend_geak import run_ascend_geak
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
            if kind == "operator_patch":
                native_operator_publication(self.store, parent["id"], revision["kernels"][-1])
            self.register(revision)
            self.register(parent)
        except ContractError as error:
            return self._blocked_candidate(spec, parent, str(error), kind=kind)
        except EvidenceProblem as error:
            return self._blocked_candidate(spec, parent, error.reason, kind=kind)
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
        if len(qualified) > 1:
            goals = self.task.get("goals")
            scores: dict[str, tuple[bool, float]] = {}
            for decision in qualified:
                checked = adjudicate(self.store, decision, state["baseline"], self.rules) if goals else decision
                if checked["version_action"] != "KEEP":
                    continue
                meets_goals = bool(goals) and all(
                    goal_checks(checked["comparison"]["over_original"], goals, self.rules).values()
                )
                scores[decision["candidate_id"]] = (meets_goals, checked["measurement"]["summary"]["throughput"])
            if scores:
                winner = max(scores, key=scores.__getitem__)
                for decision in qualified:
                    candidate_id = decision["candidate_id"]
                    if candidate_id != winner and candidate_id in scores:
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
            config = self.store.manifest.get("target", {}).get("sglang", {})
            enabled = config.get(f"native_{route}") is True
            if not enabled:
                handoff["reason"] = handoff.get("reason") or "native_kernel_driver_unconfigured"
            if route == "forge" and parent["kernels"]:
                handoff["reason"] = handoff.get("reason") or "native_forge_operator_already_loaded"
            handoff["dispatch"] = "blocked_before_native_dispatch" if handoff.get("reason") else "native_dispatch"
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
            if handoff.get("reason"):
                return {"status": "blocked", "backend": route, "handoff": handoff, "decisions": [],
                        "reason": handoff["reason"], "evidence": {"handoff": handoff_ref}}
            return self._native_kernel(task["task_id"], state, last_profile, handoff, handoff_ref, config)
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

    def _native_kernel(
        self, task_id: str, state: dict, profile: dict, handoff: dict, handoff_ref: dict, config: dict,
    ) -> dict:
        parent = state["current_revision"]
        route = handoff["backend"]
        publication_path = f"evidence/kernel/{task_id}-delivery.json"
        if route == "geak":
            publication, spec = run_ascend_geak(
                task_id=task_id, parent=parent, profile=profile, fia_default=config.get("ascend_use_fia"),
                measure=self._measure_geak_trial,
            )
            kind = "parameters"
            delivery_ref = self.store.put_evidence(publication_path, publication, producer="native_geak")
        else:
            kind = "operator_patch"
            run_forge = getattr(self.backend, "run_native_forge", None)
            try:
                if run_forge is None:
                    raise BackendError("the pinned runtime does not expose native Forge")
                publication = run_forge(profile=profile, attempt=f"{task_id}-forge")
            except BackendError as error:
                publication = {"status": "blocked", "reason": "native_forge_unavailable", "detail": str(error)}
            publication.update(mode="real", backend="forge", delivery="operator_patch",
                               target_revision=parent["id"], profile_ref=profile["report_ref"])
            spec = publication.get("candidate") if publication["status"] == "ok" else None
            if spec is not None:
                spec = {
                    **spec, "id": f"{task_id}-forge-swiglu", "parent": parent["id"], "backend": "forge",
                    "profile_ref": profile["report_ref"],
                }
                publication["candidate"] = spec
            delivery_ref = self.store.put_evidence(publication_path, publication, producer="native_forge")
            if spec is not None:
                spec = {**spec, "operator_evidence": delivery_ref}
        result = {"status": publication["status"], "reason": publication["reason"], "backend": route,
                  "handoff": handoff, "decisions": [], "evidence": {"handoff": handoff_ref, "delivery": delivery_ref}}
        if result["reason"] == "forge_worker_cleanup_unconfirmed":
            result["reason"] = "cleanup_unconfirmed"
        if spec is not None:
            decision = self.evaluate_candidate(spec, parent, state["baseline"], f"{task_id}-independent", kind)
            decision["tool_evidence"] = {"publication": delivery_ref}
            result["decisions"].append(decision)
            if decision["status"] != "ok":
                result.update(status=decision["status"], reason=decision["reason"])
        return result

    def _measure_geak_trial(self, revision: dict, attempt: str) -> dict:
        measured = self.measure(revision, attempt)
        if measured["status"] != "ok" or not attempt.endswith("-geak-candidate"):
            return measured
        profile = self.profile({"task_id": f"{attempt}-activation"}, {"current_revision": revision})
        use_fia = revision["parameters"]["ascend_use_fia"]
        operator = "fusedinferattentionscore" if use_fia else "flashattention"
        matches = [
            hit for hit in profile.get("hotspots", [])
            if hit.get("phase") == "prefill" and hit.get("event_kind") == "kernel"
            and operator in hit.get("name", "").casefold()
            and hit.get("callsite", "").startswith("sglang/srt/hardware_backend/npu/attention/ascend_backend.py:")
        ]
        measured["activation"] = {
            "status": profile["status"], "revision": profile["revision"], "expected_operator": operator,
            "matches": matches, "evidence": profile["evidence"],
        }
        if not matches or profile["status"] != "ok":
            measured.update(status="blocked", reason=profile["reason"] if profile["status"] != "ok"
                            else "geak_attention_activation_unconfirmed")
        return measured

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
