"""The single writer for optimization phases, task results and version promotion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from .actions.knowledge import RECIPE_PATH, build_recipe, validate_recipe
from .actions.runtime import RuntimeActions, accepted_revision
from .agents import ReplayAgentBackend
from .artifacts import publish_indexes
from .backends import RuntimeBackend
from .backends.factory import RuntimeBackendFactory, build_runtime, checked_probe
from .contracts import ContractError, digest, read_json, safe_id, validate_reference, validate_target, validate_task
from .dispatcher import Dispatcher
from .evaluation.adjudication import adjudicate
from .phases.machine import SWEEP_PAIR_COST, after_framework, after_prelude, after_sweep, should_reprofile, sweep_skip_reason
from .policy import PolicyGate
from .report import write_report
from .router import IntentRouter
from .session import SessionStore, project


class Coordinator:
    @classmethod
    def create(
        cls, root: Path, session_id: str, task: dict, target: dict, reference: dict, scenario: dict,
        *, backend_factory: RuntimeBackendFactory | None = None,
    ) -> Coordinator:
        validate_task(task)
        validate_target(target)
        validate_reference(reference, task)
        ReplayAgentBackend(scenario.get("agents", {}))
        backend = build_runtime(target, scenario, task["initial_revision"], backend_factory)
        runtime = checked_probe(backend, target, mode=target["mode"])
        capture = scenario.get("reference_capture")
        if capture is not None and (
            not isinstance(capture, dict)
            or capture.get("environment_fingerprint") != runtime["environment_fingerprint"]
        ):
            raise ContractError("reference capture environment does not match the runtime probe")
        store = SessionStore.create(root, target["mode"], session_id, {
            "task": task, "target": target, "reference": reference, "scenario": scenario,
            "runtime": runtime,
            "contributions": {
                "design": "按项目规划与本人指定的上游交接",
                "implementation": "Agent 编写并在本地验证",
                "measurements": ("模拟执行器生成，独立评测程序复算"
                                 if target["mode"] == "simulated" else "所选运行端采集，独立评测程序复算"),
                "third_party": "Hyperloom 1.1.5 行为参考",
            },
        })
        return cls(store, _backend=backend)

    @classmethod
    def open(cls, path: Path, *, backend_factory: RuntimeBackendFactory | None = None) -> Coordinator:
        coordinator = cls(SessionStore(path), backend_factory=backend_factory)
        coordinator._runtime_probe()
        return coordinator

    def __init__(
        self, store: SessionStore, *, backend_factory: RuntimeBackendFactory | None = None,
        _backend: RuntimeBackend | None = None,
    ):
        self.store = store
        self.task = store.manifest["task"]
        self.settings = self.task["settings"]
        self.scenario = store.manifest["scenario"]
        self.backend = _backend if _backend is not None else build_runtime(
            store.manifest["target"], self.scenario, self.task["initial_revision"], backend_factory,
        )
        bind_session = getattr(self.backend, "bind_session", None)
        if bind_session is not None:
            bind_session(store.path)
        self.actions = RuntimeActions(store, self.backend)
        self.agents = ReplayAgentBackend(self.scenario.get("agents", {}))
        self.dispatcher = Dispatcher(emit=self._emit)
        self.router = IntentRouter(
            PolicyGate(store.path / "workspaces", self.task.get("allowed_parameters")),
            self.dispatcher,
            emit=self._emit,
        )
        self.state: dict[str, Any] = {}
        self._events: list[tuple[str, dict]] = []
        self._control: dict[str, Any] = {}

    def _runtime_probe(self) -> dict:
        manifest = self.store.manifest
        return checked_probe(self.backend, manifest["target"], mode=manifest["mode"], locked=manifest["runtime"])

    def _restore(self) -> None:
        self.state = self.store.state()
        checkpoint = self.state["control"]
        self._control = deepcopy(checkpoint.get("coordinator", {
            "spent": 0, "turns": {}, "hints": {}, "review_attempts": {}, "next_turn": 1,
        }))
        if isinstance(self._control["review_attempts"], list):
            self._control["review_attempts"] = {
                digest(bundle): 1 for bundle in self._control["review_attempts"]
            }
        if checkpoint:
            self.dispatcher.restore(checkpoint["dispatcher"])
            self.router.restore(checkpoint["router"])
            self.agents.restore(checkpoint["agents"])
        self._events.clear()

    def _emit(self, event: str, payload: dict) -> None:
        self._events.append((event, deepcopy(payload)))

    def _checkpoint(self) -> None:
        self._emit("control_checkpoint", {
            "router": self.router.snapshot(), "dispatcher": self.dispatcher.snapshot(),
            "agents": self.agents.snapshot(), "coordinator": self._control,
        })
        self.state = self.store.commit(self._events)
        self._events.clear()

    def run(self, max_steps: int | None = None) -> dict:
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ContractError("max_steps must be a positive integer")
        with self.store.execution_lease():
            self._runtime_probe()
            self._restore()
            steps = 0
            while self.state["phase"] != "DONE" and (max_steps is None or steps < max_steps):
                if not self._tick():
                    break
                steps += 1
            if self.state["phase"] == "DONE":
                self.report()
            return deepcopy(self.state)

    def report(self) -> dict:
        events = self.store.events()
        state = project(events)
        publish_indexes(self.store, state, events)
        return write_report(self.store.path, self.store.manifest, state, events)

    def reproduce(self, candidate_id: str) -> dict:
        """Re-measure a recorded proposal through the frozen independent path."""
        with self.store.execution_lease():
            self._runtime_probe()
            state = self.store.state()
            decision = next((row for row in state["decisions"] if row["candidate_id"] == candidate_id), None)
            proposal = (decision or {}).get("evidence", {}).get("proposal", {})
            if decision is None or not self.store.verify_evidence(proposal):
                raise ContractError(f"candidate has no verified proposal record: {candidate_id}")
            spec = read_json(self.store.artifact_path(proposal["path"]))
            attempt_id = f"reproduce-{uuid4().hex[:12]}"
            rerun = self.actions.evaluate_candidate(
                spec, decision["parent_revision"], state["baseline"], attempt_id, decision["kind"]
            )
            rerun = adjudicate(self.store, rerun, state["baseline"], self.task["rules"])
            result = {**rerun, "attempt_id": attempt_id, "recorded_action": decision["version_action"],
                      "mode": self.store.manifest["mode"]}
            reference = self.store.put_evidence(
                f"evidence/reproductions/{attempt_id}.json",
                {"session_fingerprint": self.store.manifest["fingerprint"], "result": result},
                producer="independent_reproduction",
            )
            self.report()
            return {**result, "reproduction_receipt": reference}

    @property
    def phase(self) -> str:
        return self.state["phase"]

    @property
    def cycle(self) -> int:
        return self.state["cycle"]

    @property
    def budget(self) -> float:
        return max(0, self.settings["budget"] - self._control["spent"])

    def _tick(self) -> bool:
        if self.state["runtime"] is None:
            actual = self._runtime_probe()
            actual["evidence"] = self.store.put_evidence(
                "evidence/runtime/probe.json", actual, producer="target_probe"
            )
            self._emit("runtime_probed", actual)
            if "retrieved_recipe" in self.scenario:
                disposition = validate_recipe(
                    self.scenario["retrieved_recipe"], self.task, self.store.manifest["target"], actual,
                )
                self._emit("recipe_retrieved", disposition)
            self._checkpoint()
        elif self.dispatcher.has_inflight:
            return self._recover_tasks()
        elif ready := self.dispatcher.start_ready(phase=self.phase, remaining_budget=self.budget):
            self._control["spent"] += sum(task["estimated_seconds"] for task in ready)
            self._checkpoint()  # Record task identity and lease before running work.
            for task in ready:
                result = self._execute_task(task)
                receipt = self.store.put_evidence(
                    self._receipt(task), {
                        "task_id": task["task_id"], "action": task["action"], "phase": task["phase"], "result": result,
                    },
                    producer="action_executor",
                )
                self._finish_task(task, result, receipt)
                if self.dispatcher.get(task["task_id"])["status"] == "cleanup_pending":
                    break
        else:
            self._advance_phase()
            self._checkpoint()
        return True

    @staticmethod
    def _receipt(task: dict) -> str:
        return f"evidence/actions/{task['task_id']}.json"

    def _recover_tasks(self) -> bool:
        for task in self.dispatcher.tasks.values():
            if task["status"] not in ("running", "cleanup_pending"):
                continue
            path = self.store.artifact_path(self._receipt(task))
            if path.exists():
                data = read_json(path)
                if any(data.get(key) != task[key] for key in ("task_id", "action", "phase")):
                    raise ContractError("action receipt identity conflict")
                receipt = self.store.put_evidence(self._receipt(task), data, producer="action_executor")
                cleanup = self.backend.recover_cleanup()
                if not cleanup["released"]:
                    return False
                self._emit("task_recovered", {"task_id": task["task_id"], "receipt": receipt, "cleanup": cleanup})
                self._finish_task(task, data["result"], receipt, recovered_cleanup=True)
                continue

            cleanup = self.backend.recover_cleanup()
            if not cleanup["released"]:
                return False
            result = {"status": "interrupted", "reason": "interrupted_before_action_receipt"}
            settled = self.dispatcher.settle(task["task_id"], result, cleanup=True)
            self.router.record_task_result(settled)
            if not task["params"].get("retry_of") and self.budget >= task["estimated_seconds"]:
                self.dispatcher.enqueue(
                    task["action"], {**task["params"], "retry_of": task["task_id"]},
                    source=task["source"], proposal_id=task["proposal_id"], phase=self.phase,
                    idempotency_key=f"retry:{task['task_id']}",
                )
            self._emit("task_recovered", {"task_id": task["task_id"], "cleanup": cleanup, **result})
            self._checkpoint()
        return True

    def _execute_task(self, task: dict) -> dict:
        action = task["action"]
        if action == "specialist":
            if task["params"].get("mode") == "enablement":
                fix = self.scenario.get("enablement")
                return {"status": "ok", "fix": fix, "reason": "suggested_fix" if fix else "no_enablement_candidate"}
            context = self._context("specialist")
            context["assignment"] = task["params"]
            responses = self.agents.run("specialist", context)
            return {"status": "ok", "role_context": context, "responses": responses, "agents": self.agents.snapshot()}
        run = {
            "baseline": self.actions.baseline, "enablement": self.actions.enablement,
            "profile": self.actions.profile,
            "explore": self.actions.explore, "integrate_patch": self.actions.integrate,
            "kernel_agent": self.actions.kernel, "conc_sweep": self.actions.sweep,
            "stack_revalidate": self.actions.stack,
        }.get(action)
        if run is None:
            return {"status": "failed", "reason": f"unimplemented_action:{action}"}
        self._runtime_probe()
        return run(task, self.state)

    def _finish_task(self, task: dict, result: dict, receipt: dict, *, recovered_cleanup: bool = False) -> None:
        cleanup_pending = result.get("reason") == "cleanup_unconfirmed" or any(
            row.get("reason") == "cleanup_unconfirmed" for row in result.get("decisions", [])
        )
        finished = self.dispatcher.settle(
            task["task_id"], self._result_summary(result, receipt), cleanup=recovered_cleanup or not cleanup_pending
        )
        if finished["status"] == "cleanup_pending":
            self._checkpoint()
            return

        self.router.record_task_result(finished)
        action = task["action"]
        common = {"task_id": task["task_id"], "receipt": receipt, "cycle": self.cycle, "phase": self.phase}
        if action == "specialist" and task["params"].get("mode") == "enablement":
            self._emit("enablement_suggestion", {**result, **common})
        elif action == "specialist":
            self.agents.restore(result["agents"])
            self._route_turn("specialist", result["role_context"], result["responses"], task_id=task["task_id"])
        elif action in ("baseline", "enablement"):
            if result.get("acceptance") == "pass":
                current = {
                    **result["revision"], "summary": result["measurement"]["summary"],
                    "validation_ref": result["evidence"]["revision_measurement"],
                }
                self._emit("baseline_recorded", {"result": result, "revision": current})
            else:
                self._emit("baseline_failure", {**result, **common})
        elif action == "profile":
            self._emit("profile_recorded", {**result, **common})
        elif action == "conc_sweep":
            self._emit("sweep_recorded", {**result, **common})
        elif action == "stack_revalidate":
            self._emit("stack_validated", {**result, **common})
        elif action in ("explore", "integrate_patch", "kernel_agent"):
            self._candidate_results(task, result, common)
        self._checkpoint()

    @staticmethod
    def _decision_summary(row: dict) -> dict:
        fields = ("candidate_id", "parent", "kind", "backend", "status", "evidence_state",
                  "acceptance", "version_action", "reason", "comparison", "receipt")
        return {key: row[key] for key in fields if key in row}

    @classmethod
    def _result_summary(cls, result: dict, receipt: dict) -> dict:
        # Full adapter output stays in one immutable receipt. Task snapshots
        # and role inboxes carry its reference instead of copying every sample.
        fields = ("status", "reason", "point", "backend", "fix", "ranks", "missing_ranks", "revision",
                  "version_action", "acceptance", "evidence_state", "evidence")
        summary = {key: result[key] for key in fields if key in result}
        if isinstance(summary.get("revision"), dict):
            summary["revision"] = summary["revision"]["id"]
        if "handoff" in result:
            handoff = result["handoff"]
            summary["handoff"] = {
                key: handoff[key] for key in ("backend", "delivery", "target_revision", "selected", "reason", "profile_ref")
                if key in handoff
            }
        if "decisions" in result:
            summary["decisions"] = [cls._decision_summary(row) for row in result["decisions"]]
        return {**summary, "receipt": receipt}

    def _candidate_results(self, task: dict, result: dict, common: dict) -> None:
        current = self.state["current_revision"]
        settled = {row["candidate_id"] for row in self.state["decisions"]}
        for row in result.get("decisions", []):
            decision = {**row, **common, "proposal_id": task["proposal_id"]}
            candidate_id = decision["candidate_id"]
            try:
                safe_id(candidate_id, "candidate.id")
            except ContractError:
                self._emit("candidate_attempt", {**decision, "reason": "invalid_candidate_id"})
                continue
            if candidate_id in settled:
                self._emit("candidate_attempt", {**decision, "reason": "missing_or_settled_candidate_id"})
                continue
            settled.add(candidate_id)
            self._emit("candidate_attempt", decision)
            decision = adjudicate(self.store, decision, self.state["baseline"], self.task["rules"])
            assessment = {
                "candidate_id": candidate_id, "task_id": task["task_id"], "role": "critic",
                "evidence_state": decision["evidence_state"], "acceptance": decision["acceptance"],
                "version_action": decision["version_action"], "reason": decision["reason"],
                "comparison": decision.get("comparison"), "review": decision["adjudication"],
            }
            assessment["evidence"] = self.store.put_evidence(
                f"evidence/decisions/{task['task_id']}-{candidate_id}.json", assessment,
                producer="independent_critic",
            )
            decision["evidence"] = {**decision["evidence"], "critic_assessment": assessment["evidence"]}
            self._emit("critic_measured_verdict", assessment)
            if decision["version_action"] == "KEEP":
                required = {"candidate_measurement", "parent_measurement", "paired_comparison", "proposal"}
                if decision["kind"] == "kernel":
                    required.add("device_timing")
                proofs = decision["evidence"]
                if decision["parent"] != current["id"] or not all(
                    key in proofs and self.store.verify_evidence(proofs[key]) for key in required
                ):
                    decision.update(version_action="HOLD", acceptance="undetermined",
                                    evidence_state="conflict", reason="writeback_identity_or_evidence_mismatch")
                else:
                    current = accepted_revision(decision)
                    self._emit("candidate_decision", {"decision": decision, "promote": current})
                    continue
            self._emit("candidate_decision", {"decision": decision})

    def _context(self, role: str) -> dict:
        baseline = self.state["baseline"]
        baseline_summary = None if baseline is None else {
            "revision": baseline["revision"]["id"], "point": baseline["point"],
            "summary": baseline["measurement"]["summary"],
            "measurement_ref": baseline["evidence"]["revision_measurement"],
        }
        profile = self._last_profile()
        profile_summary = None if profile is None else {
            key: profile[key] for key in ("revision", "point", "rank", "ranks", "capture", "hotspots", "report_ref")
            if key in profile
        }
        sweep = self.state["last_conc_sweep"]
        sweep_summary = None if sweep is None else {
            key: sweep[key] for key in ("status", "reason", "revision", "cycle", "receipt", "evidence") if key in sweep
        }
        return {
            "phase": self.phase, "cycle": self.cycle, "budget": self.budget,
            "goal": self.settings["target_gain_pct"], "current_revision": self.state["current_revision"],
            "baseline": baseline_summary, "profile": profile_summary,
            "decisions": [self._decision_summary(row) for row in self.state["decisions"]],
            "inbox": self.router.inbox_for(role), "pending_proposals": self.router.pending_proposals,
            "last_conc_sweep": sweep_summary,
            "recipe": self.state["retrieved_recipe"],
        }

    def _route_turn(self, role: str, context: dict, intents: list[dict], task_id: str | None = None) -> None:
        turn_id = f"turn-{self._control['next_turn']:04d}"
        self._control["next_turn"] += 1
        self._emit("agent_turn_started", {"turn_id": turn_id, "role": role, "task_id": task_id, "context": context})
        for intent in intents:
            fields = deepcopy(intent)
            payload = fields.get("payload", fields)
            params = payload.get("params") if isinstance(payload, dict) else None
            kind = fields.get("kind", fields.get("type"))
            if kind in ("delegate", "propose_action") and isinstance(params, dict) and context["current_revision"]:
                params.setdefault("parent", context["current_revision"]["id"])
                profile = context.get("profile") or {}
                if profile.get("report_ref"):
                    params.setdefault("profile_ref", profile["report_ref"])
            self._emit("agent_intent", {"turn_id": turn_id, "role": role, "intent": fields})
            routed = self.router.route(role, self.phase, fields)
            if routed.get("next_action_hint"):
                self._control["hints"][str(self.cycle)] = routed["next_action_hint"]
            self._emit("intent_result", {"turn_id": turn_id, "role": role, "result": routed})
        self._emit("agent_turn_completed", {"turn_id": turn_id, "role": role, "response_count": len(intents)})

    def _call_role(self, role: str) -> list[dict]:
        context = self._context(role)
        replies = self.agents.run(role, context)
        self._route_turn(role, context, replies)
        return replies

    def _phase_tasks(self, action: str) -> list[dict]:
        return [
            task for task in self.dispatcher.tasks.values()
            if task["action"] == action and task["phase"] == self.phase
            and task["params"].get("cycle", 1) == self.cycle
        ]

    def _register(self, action: str, *, key: str | None = None, **params: Any) -> None:
        self.router.register_internal(
            self.phase, action, {"cycle": self.cycle, **params},
            idempotency_key=f"{self.phase}:{self.cycle}:{key or action}",
        )

    def _change_phase(self, phase: str, reason: str, *, next_cycle: bool = False) -> None:
        self.dispatcher.cancel_queued(set(), reason)
        self.router.expire_pending(self.phase, reason)
        self._emit("phase_changed", {
            "previous": self.phase, "phase": phase, "cycle": self.cycle + int(next_cycle), "reason": reason,
        })

    def _advance_phase(self) -> None:
        handler = {
            "PRELUDE": self._prelude, "ENABLEMENT": self._enablement, "FRAMEWORK_AGENT": self._framework,
            "KERNEL_AGENT": self._kernel, "SWEEP": self._sweep, "CLOSE": self._close,
        }[self.phase]
        handler()

    def _prelude(self) -> None:
        if self.state["baseline"] is None:
            if not self._phase_tasks("baseline") and self.budget >= 1:
                self._register("baseline", revision=self.task["initial_revision"])
            elif self.settings["enablement"] and self.budget >= 1:
                self._change_phase("ENABLEMENT", "original_baseline_failed")
            else:
                self._change_phase("CLOSE", "baseline_unavailable")
        elif self._profile_due() and not self._phase_tasks("profile"):
            self._register("profile", reason="initial")
        else:
            if not self._phase_tasks("profile"):
                self._emit("profile_skipped", {"reason": "disabled_capability_or_budget", "phase": self.phase})
            self._change_phase(after_prelude(self.settings), "prelude_complete")

    def _enablement(self) -> None:
        scouts = self._phase_tasks("specialist")
        if self.state["baseline"] is not None:
            if self._profile_due() and not self._phase_tasks("profile"):
                self._register("profile", reason="enablement_recovered")
            else:
                self._change_phase(after_prelude(self.settings), "enablement_complete")
        elif self.budget >= 1 and not scouts:
            self._register("specialist", mode="enablement", reason="original_baseline_failed")
        elif self.budget >= 1 and not self._phase_tasks("enablement") and scouts:
            fix = (scouts[-1].get("result") or {}).get("fix")
            if not self._enablement_fix_allowed(fix):
                self._change_phase("CLOSE", "enablement_candidate_unavailable_or_denied")
                return
            self._register("enablement", reason="approved_minimal_startup_fix", fix=fix)
        else:
            self._change_phase("CLOSE", "enablement_failed")

    def _enablement_fix_allowed(self, fix: dict | None) -> bool:
        if not isinstance(fix, dict) or not fix.get("reason"):
            return False
        try:
            revision = safe_id(fix.get("revision"), "enablement.revision")
        except ContractError:
            return False
        return revision != self.task["initial_revision"] and isinstance(fix.get("parameters", {}), dict)

    def _last_profile(self) -> dict | None:
        return next((item for item in reversed(self.state["profiles"]) if item["status"] == "ok"), None)

    def _profile_due(self) -> bool:
        revision = self.state["current_revision"]
        if (
            not revision or not self.settings["profile_enabled"] or self.budget < 1
            or "profile" not in self.state["runtime"]["capabilities"]
        ):
            return False
        if any(item.get("revision") == revision["id"] for item in self.state["profiles"]):
            return False
        return should_reprofile(revision, self._last_profile(), self.settings["reprofile_ratio"])

    def _framework(self) -> None:
        if self._profile_due():
            revision = self.state["current_revision"]["id"]
            self._register("profile", key=f"profile:{revision}", reason="accepted_gain_or_missing_profile")
            return
        pending = sorted(self.router.pending_proposals)
        bundle = digest(pending)
        attempts = self._control["review_attempts"]
        if (
            pending and attempts.get(bundle, 0) < self.settings["max_turns_per_cycle"]
            and self.budget >= 1 and self.agents.has_next("critic")
        ):
            attempts[bundle] = attempts.get(bundle, 0) + 1
            self._call_role("critic")
            return
        cycle = str(self.cycle)
        count = self._control["turns"].get(cycle, 0)
        hint = self._control["hints"].get(cycle)
        if (
            not hint and self.budget >= 1 and count < self.settings["max_turns_per_cycle"]
            and self.agents.has_next("orchestration")
        ):
            self._control["turns"][cycle] = count + 1
            self._call_role("orchestration")
            return
        if pending:
            self._emit("critic_bundle_unresolved", {"proposal_ids": pending, "cycle": self.cycle})
        else:
            self._emit("critic_skipped", {"reason": "empty_review_bundle", "cycle": self.cycle})
        self._change_phase(after_framework(self.settings), hint or "framework_round_complete")

    def _kernel(self) -> None:
        route = self.settings["kernel_backend"]
        if route == "forge" and self._profile_due():
            revision = self.state["current_revision"]["id"]
            self._register("profile", key=f"profile:{revision}", reason="forge_entry")
        elif self.budget >= 1 and not self._phase_tasks("kernel_agent"):
            self._register("kernel_agent", backend=route)
        else:
            if not self._phase_tasks("kernel_agent"):
                self._emit("kernel_skipped", {"reason": "budget_exhausted", "backend": route, "cycle": self.cycle})
            self._change_phase("SWEEP", "kernel_tasks_settled")

    def _stack_needs_measurement(self) -> bool:
        revision = self.state["current_revision"]
        if revision is None:
            return False
        if self.store.verify_evidence(revision.get("validation_ref") or {}):
            return False
        return not any(
            row["revision"] == revision["id"] and row.get("acceptance") == "pass"
            and self.store.verify_evidence(row["evidence"].get("revision_measurement") or {})
            for row in self.state["stack_validations"]
        )

    def _sweep(self) -> None:
        completed = self.state["sweeps"]
        if completed and completed[-1]["cycle"] == self.cycle:
            next_phase, reason = after_sweep(self.state, self.settings, self.budget)
            self._change_phase(next_phase, reason, next_cycle=next_phase == "FRAMEWORK_AGENT")
            return
        revision = self.state["current_revision"]["id"]
        if self._stack_needs_measurement() and self.budget >= 1 and not self._phase_tasks("stack_revalidate"):
            self._register("stack_revalidate", reason="before_sweep")
            return
        previous = next((row["revision"] for row in reversed(completed) if row["status"] == "ok"), None)
        skip = sweep_skip_reason(self.settings, revision, previous, self.budget)
        if self._stack_needs_measurement():
            skip = "stack_unverified"
        if skip:
            self._emit("sweep_recorded", {
                "status": "skipped", "reason": skip, "revision": revision, "cycle": self.cycle, "pairs": [],
            })
        else:
            levels = self.settings["sweep_concurrency"][: int(self.budget // SWEEP_PAIR_COST)]
            self._register(
                "conc_sweep", revision=revision, concurrency_levels=levels,
                estimated_seconds=len(levels) * SWEEP_PAIR_COST,
            )

    def _close(self) -> None:
        if self._stack_needs_measurement() and self.budget >= 1 and not self._phase_tasks("stack_revalidate"):
            self._register("stack_revalidate", reason="close_stack_unverified")
            return
        if self._profile_due() and not self._phase_tasks("profile"):
            self._register("profile", reason="terminal")
            return
        close_event = next(
            event["payload"] for event in reversed(self.store.events())
            if event["type"] == "phase_changed" and event["payload"]["phase"] == "CLOSE"
        )
        current = self.state["current_revision"]
        if current and self.state["decisions"] and not self._stack_needs_measurement():
            recipe = build_recipe(
                self.task, self.store.manifest["target"], self.state["runtime"],
                source_session_id=self.store.manifest["session_id"], revision=current["id"],
                parameters=current.get("parameters", {}),
            )
            recipe["reference"] = self.store.put_evidence(RECIPE_PATH, recipe, producer="close")
            self._emit("recipe_recorded", recipe)
        for decision in self.state["decisions"]:
            if decision["version_action"] == "KEEP":
                continue
            candidate_id = decision["candidate_id"]
            lesson = {
                "kind": "failure", "candidate_id": candidate_id, "parent": decision["parent"],
                "reason": decision["reason"], "mode": self.store.manifest["mode"],
                "target_id": self.store.manifest["target"]["id"], "workload": self.task["workload"]["id"],
                "evidence": decision["evidence"].get("critic_assessment"),
            }
            lesson["reference"] = self.store.put_evidence(
                f"evidence/recipes/failure-{candidate_id}.json", lesson, producer="close"
            )
            self._emit("recipe_recorded", lesson)
        if self._stack_needs_measurement():
            self._emit("stack_validation_unavailable", {"reason": "required_evidence_missing", "budget": self.budget})
        self._emit("stopped", {"reason": close_event["reason"], "revision": (current or {}).get("id")})
        self._change_phase("DONE", "close_complete")
