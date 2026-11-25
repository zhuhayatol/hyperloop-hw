"""The single writer for optimization phases, task results and version promotion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from .actions.enablement import accept_startup_repair, observed_startup_failure
from .actions.knowledge import RECIPE_PATH, build_recipe, validate_recipe
from .actions.review_evidence import ROPE_ACTION, collect_rope_evidence, is_rope_layout_edit
from .actions.runtime import RuntimeActions, accepted_revision
from .agents import AgentInvocationError, build_agent_backend
from .artifacts import publish_indexes
from .backends import RuntimeBackend
from .backends.factory import RuntimeBackendFactory, build_runtime, checked_probe
from .baseline import latest_baseline
from .contracts import ContractError, digest, read_json, safe_id, validate_reference, validate_target, validate_task
from .dispatcher import Dispatcher
from .evaluation.adjudication import adjudicate
from .evaluation.goal import GOAL_RETEST_COST, paired_goal_retest, warrants_retest
from .phases.direction import direction_change
from .phases.machine import SWEEP_PAIR_COST, after_framework, after_prelude, after_sweep, should_reprofile, sweep_skip_reason
from .policy import PolicyGate
from .report import write_report
from .router import IntentRouter
from .session import SessionStore, project


_INITIAL_CONTROL = {"spent": 0, "turns": {}, "hints": {}, "review_attempts": {}, "next_turn": 1}


class Coordinator:
    @classmethod
    def create(
        cls, root: Path, session_id: str, task: dict, target: dict, reference: dict | None, scenario: dict,
        *, backend_factory: RuntimeBackendFactory | None = None, baseline_history: Path | None = None,
    ) -> Coordinator:
        validate_task(task)
        validate_target(target)
        if baseline_history is None:
            validate_reference(reference, task)
        agents = build_agent_backend(scenario, task, target)
        if scenario.get("agent_provider", {}).get("provider") == "deepseek":
            scenario = {**scenario, "agent_provider": agents.snapshot()}
        backend = build_runtime(target, scenario, task["initial_revision"], backend_factory)
        runtime = checked_probe(backend, target, mode=target["mode"])
        historical = None
        if baseline_history is not None:
            historical = latest_baseline(baseline_history, task, target, runtime, scenario)
            reference = historical.reference
        capture = scenario.get("reference_capture")
        if capture is not None and (
            not isinstance(capture, dict)
            or capture.get("environment_fingerprint") != runtime["environment_fingerprint"]
        ):
            raise ContractError("reference capture environment does not match the runtime probe")
        fields = {
            "task": task, "target": target, "reference": reference, "scenario": scenario,
            "runtime": runtime,
            "contributions": {
                "design": "按项目规划与本人指定的上游交接",
                "implementation": "Agent 编写并在本地验证",
                "measurements": ("模拟执行器生成，独立评测程序复算"
                                 if target["mode"] == "simulated" else "所选运行端采集，独立评测程序复算"),
                "third_party": "Hyperloom 1.1.5 行为参考",
            },
        }
        if historical is not None:
            fields["baseline_reuse"] = {**historical.provenance,
                                        "reserved_task_numbers": historical.last_task_number}
        store = SessionStore.create(root, target["mode"], session_id, fields)
        coordinator = cls(store, _backend=backend)
        if historical is not None:
            with store.execution_lease():
                historical.copy_evidence(store)
                coordinator._record_baseline(historical.baseline, historical.provenance)
                if historical.profile is not None:
                    coordinator._emit("profile_recorded", {**historical.profile, "reused_from": historical.provenance})
                coordinator._checkpoint()
        return coordinator

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
        self.goals = self.task.get("goals")
        self.scenario = store.manifest["scenario"]
        self.backend = _backend if _backend is not None else build_runtime(
            store.manifest["target"], self.scenario, self.task["initial_revision"], backend_factory,
        )
        bind_session = getattr(self.backend, "bind_session", None)
        if bind_session is not None:
            bind_session(store.path)
        self.actions = RuntimeActions(store, self.backend)
        self.agents = build_agent_backend(self.scenario, self.task, store.manifest["target"])
        self.dispatcher = Dispatcher(emit=self._emit, reserved_task_numbers=store.manifest.get(
            "baseline_reuse", {}).get("reserved_task_numbers", 0))
        self.router = IntentRouter(
            PolicyGate(store.path / "workspaces", self.task.get("allowed_parameters")),
            self.dispatcher,
            emit=self._emit,
        )
        self.state: dict[str, Any] = {}
        self._events: list[tuple[str, dict]] = []
        self._control: dict[str, Any] = deepcopy(_INITIAL_CONTROL)
        self.observer = None

    def _runtime_probe(self) -> dict:
        manifest = self.store.manifest
        return checked_probe(self.backend, manifest["target"], mode=manifest["mode"], locked=manifest["runtime"])

    def _restore(self) -> None:
        self.state = self.store.state()
        if self.store.manifest.get("baseline_reuse") and self.state["baseline"] is None:
            raise ContractError("historical baseline import did not finish; start a new Session")
        checkpoint = self.state["control"]
        self._control = deepcopy(checkpoint.get("coordinator", _INITIAL_CONTROL))
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
        if self.observer is not None:
            self.observer.checkpoint(self.store, self.state)

    def run(self, max_steps: int | None = None) -> dict:
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ContractError("max_steps must be a positive integer")
        with self.store.execution_lease():
            self._runtime_probe()
            self._restore()
            steps = 0
            while self.state["phase"] != "DONE" and (max_steps is None or steps < max_steps):
                try:
                    if not self._tick():
                        break
                except AgentInvocationError as error:
                    if self.goals is None:
                        raise
                    self._emit("run_blocked", {
                        "phase": self.phase, "cycle": self.cycle, "reason": error.receipt["reason"],
                        "call_id": error.receipt["call_id"], "role": error.receipt["role"],
                    })
                    self._checkpoint()
                    break
                steps += 1
            if self.state["phase"] == "DONE" or self.observer is not None or self.goals is not None:
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
                failure = result.get("agent_failure")
                retryable = result.get("reason") == "forge_worker_timeout" or (
                    failure is not None and self._retryable_api(failure)
                )
                if self.goals is not None and retryable and self._queue_retry(task, result["reason"]):
                    self._checkpoint()
                elif failure:
                    raise AgentInvocationError(failure)
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
                    return self._wait_for_cleanup(task, cleanup)
                self._emit("task_recovered", {"task_id": task["task_id"], "receipt": receipt, "cleanup": cleanup})
                self._finish_task(task, data["result"], receipt, recovered_cleanup=True)
                continue

            cleanup = self.backend.recover_cleanup()
            if not cleanup["released"]:
                return self._wait_for_cleanup(task, cleanup)
            result = {"status": "interrupted", "reason": "interrupted_before_action_receipt"}
            settled = self.dispatcher.settle(task["task_id"], result, cleanup=True)
            self.router.record_task_result(settled)
            self._queue_retry(task, result["reason"])
            self._emit("task_recovered", {"task_id": task["task_id"], "cleanup": cleanup, **result})
            self._checkpoint()
        return True

    def _queue_retry(self, task: dict, reason: str) -> bool:
        if task["params"].get("retry_of") or self.budget < task["estimated_seconds"]:
            return False
        retry = self.dispatcher.enqueue(
            task["action"], {**task["params"], "retry_of": task["task_id"]},
            source=task["source"], proposal_id=task["proposal_id"], phase=task["phase"],
            idempotency_key=f"retry:{task['task_id']}",
        )
        if self.goals is not None or self.observer is not None:
            self._emit("task_retry_scheduled", {
                "task_id": task["task_id"], "action": task["action"], "retry_task_id": retry["task_id"],
                "reason": reason,
            })
        return True

    def _wait_for_cleanup(self, task: dict, cleanup: dict) -> bool:
        if self.goals is not None or self.observer is not None:
            self._emit("task_cleanup_wait", {"task_id": task["task_id"], "action": task["action"], "cleanup": cleanup})
            self._checkpoint()
        return False

    def _execute_task(self, task: dict) -> dict:
        action = task["action"]
        if action == "specialist":
            if task["params"].get("mode") == "enablement":
                if (
                    self.store.manifest["mode"] == "real"
                    and self.scenario.get("agent_provider", {}).get("provider") == "deepseek"
                ):
                    return self._diagnose_startup(task)
                fix = self.scenario.get("enablement")
                return {"status": "ok", "fix": fix, "reason": "suggested_fix" if fix else "no_enablement_candidate"}
            context = self._context("specialist")
            context["assignment"] = task["params"]
            self._agent_provider_started("specialist", task_id=task["task_id"])
            try:
                responses = self.agents.run("specialist", context)
            except AgentInvocationError as error:
                return {"status": "failed", "reason": error.receipt["reason"], "agent_failure": error.receipt}
            result = {"status": "ok", "role_context": context, "responses": responses, "agents": self.agents.snapshot()}
            if self.agents.last_receipt is not None:
                result["agent_call"] = self.agents.last_receipt
                result["role_context"] = {
                    "current_revision": {"id": context["current_revision"]["id"]},
                    "profile": {"report_ref": (context["profile"] or {}).get("report_ref")},
                }
            return result
        if action == "stack_revalidate" and task["params"].get("reason") == "goal_revalidate":
            self._runtime_probe()
            assessment = paired_goal_retest(self.actions, self.state, task["task_id"], self.goals or {})
            return {
                "status": "failed" if assessment["status"] == "incomplete" else "ok",
                "reason": assessment["reason"], "revision": assessment["revision"], "goal_assessment": assessment,
            }
        if action == ROPE_ACTION:
            self._runtime_probe()
            return collect_rope_evidence(self.store, self.backend, task, self.state)
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

    def _diagnose_startup(self, task: dict) -> dict:
        baseline = next((
            row for row in reversed(tuple(self.dispatcher.tasks.values())) if row["action"] == "baseline"
            and row["phase"] == "PRELUDE" and (row.get("result") or {}).get("status") == "failed"
        ), None)
        failure = observed_startup_failure(self.store, baseline) if baseline else None
        if failure is None:
            return {"status": "ok", "fix": None, "reason": "no_supported_startup_failure"}
        context = {"startup_failure": failure}
        self._agent_provider_started("specialist", task_id=task["task_id"])
        try:
            responses = self.agents.run("specialist", context)
        except AgentInvocationError as error:
            return {"status": "failed", "reason": error.receipt["reason"], "agent_failure": error.receipt}
        fix = accept_startup_repair(responses, failure, task["task_id"])
        return {
            "status": "ok", "reason": "suggested_fix" if fix else "no_verified_startup_repair",
            "fix": fix, "startup_failure": failure, "responses": responses, "agent_call": self.agents.last_receipt,
        }

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
            call = result.get("agent_call") or result.get("agent_failure")
            if call:
                responses = result.get("responses", [])
                turn_id = self._start_turn("specialist", {}, task["task_id"], call, responses)
                if call["status"] == "ok":
                    for intent in responses:
                        self._emit("agent_intent", {"turn_id": turn_id, "role": "specialist", "intent": intent})
                    self._emit("agent_turn_completed", {"turn_id": turn_id, "role": "specialist",
                                                        "response_count": len(responses)})
            self._emit("enablement_suggestion", {
                **{key: value for key, value in result.items() if key not in {"agent_call", "agent_failure", "responses"}},
                **common,
            })
        elif action == "specialist":
            if result.get("agent_failure"):
                self._start_turn("specialist", {}, task["task_id"], result["agent_failure"], [])
            else:
                self.agents.restore(result["agents"])
                self._route_turn("specialist", result["role_context"], result["responses"], task_id=task["task_id"],
                                 call=result.get("agent_call"))
        elif action in ("baseline", "enablement"):
            if result.get("acceptance") == "pass":
                self._record_baseline(result)
            else:
                self._emit("baseline_failure", {**result, **common})
        elif action == "profile":
            self._emit("profile_recorded", {**result, **common})
        elif action == ROPE_ACTION:
            self._record_review_evidence(task, result, common)
        elif action == "conc_sweep":
            self._emit("sweep_recorded", {**result, **common})
        elif action == "stack_revalidate":
            if "goal_assessment" in result:
                self._record_goal(result["goal_assessment"], task["task_id"], common)
            else:
                self._emit("stack_validated", {**result, **common})
        elif action in ("explore", "integrate_patch", "kernel_agent"):
            self._candidate_results(task, result, common)
        self._checkpoint()

    def _record_baseline(self, result: dict, reused_from: dict | None = None) -> None:
        current = {
            **result["revision"], "summary": result["measurement"]["summary"],
            "validation_ref": result["evidence"]["revision_measurement"],
        }
        provenance = {"reused_from": reused_from} if reused_from is not None else {}
        self._emit("baseline_recorded", {"result": result, "revision": current, **provenance})

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
            "goal": self.goals["throughput_gain_pct"] if self.goals else self.settings["target_gain_pct"],
            "current_revision": self.state["current_revision"],
            "baseline": baseline_summary, "profile": profile_summary,
            "decisions": [self._decision_summary(row) for row in self.state["decisions"]],
            "inbox": self.router.inbox_for(role), "pending_proposals": self.router.pending_proposals,
            **({"review_feedback": self._review_feedback()} if role == "orchestration"
               and self.goals is not None else {}),
            "last_conc_sweep": sweep_summary,
            "recipe": self.state["retrieved_recipe"],
            **({"direction_change": self.state["direction_changes"][-1]} if self.state["direction_changes"] else {}),
        }

    def _start_turn(self, role: str, context: dict, task_id: str | None, call: dict | None,
                    intents: list[dict]) -> str:
        turn_id = f"turn-{self._control['next_turn']:04d}"
        self._control["next_turn"] += 1
        identity = {"turn_id": turn_id, "role": role, "task_id": task_id}
        self._emit("agent_turn_started", {**identity, "context": context} if call is None else identity)
        if call is not None:
            metadata = {**identity, "phase": self.phase, "cycle": self.cycle}
            evidence = self.store.put_evidence(
                f"evidence/agents/{safe_id(call['call_id'], 'agent.call_id')}.json",
                {**call, **metadata, "normalized_intents": intents}, producer="agent_provider",
            )
            self._emit("agent_provider_call", {
                **{key: value for key, value in call.items() if key not in {"request_context", "response_content"}},
                **metadata, "evidence": evidence,
            })
            if call["status"] == "failed":
                self._emit("agent_turn_failed", {**identity, "reason": call["reason"]})
        return turn_id

    def _route_turn(self, role: str, context: dict, intents: list[dict], task_id: str | None = None,
                    *, call: dict | None = None) -> None:
        turn_id = self._start_turn(role, context, task_id, call, intents)
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

    def _agent_provider_started(self, role: str, *, task_id: str | None = None,
                                retry_of: str | None = None) -> None:
        if self.observer is None and self.goals is None:
            return
        self._emit("agent_provider_started", {
            "role": role, "phase": self.phase, "cycle": self.cycle, "task_id": task_id,
            "request_key": f"turn-{self._control['next_turn']:04d}", "retry_of": retry_of,
            "provider": self.scenario.get("agent_provider", {}).get("provider", "replay"),
        })
        self._checkpoint()

    @staticmethod
    def _retryable_api(failure: dict) -> bool:
        status = failure.get("http_status")
        return failure.get("reason") in {
            "request_timeout", "transport_error", "incomplete_response", "invalid_structured_response",
        } or (type(status) is int and (status in {408, 429} or 500 <= status < 600))

    def _call_role(self, role: str) -> list[dict]:
        context = self._context(role)
        retry_of = None
        while True:
            self._agent_provider_started(role, retry_of=retry_of)
            try:
                replies = self.agents.run(role, context)
            except AgentInvocationError as error:
                identity = self._start_turn(role, context, None, error.receipt, [])
                if (self.goals is None or retry_of is not None or self.budget < 1
                        or not self._retryable_api(error.receipt)):
                    self._checkpoint()
                    raise
                retry_of = identity
                self._emit("task_retry_scheduled", {
                    "role": role, "retry_of": retry_of, "reason": error.receipt["reason"],
                })
                self._checkpoint()
                continue
            self._route_turn(role, context, replies, call=self.agents.last_receipt)
            return replies

    def _phase_tasks(self, action: str) -> list[dict]:
        return [
            task for task in self.dispatcher.tasks.values()
            if task["action"] == action and task["phase"] == self.phase
            and task["params"].get("cycle", 1) == self.cycle
        ]

    def _register(self, action: str, *, key: str | None = None, **params: Any) -> dict:
        return self.router.register_internal(
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
            if not self._phase_tasks("profile") and self._last_profile() is None:
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

    def _profile_due(self, *, require_current: bool = False) -> bool:
        revision = self.state["current_revision"]
        if (
            not revision or not self.settings["profile_enabled"] or self.budget < 1
            or "profile" not in self.state["runtime"]["capabilities"]
        ):
            return False
        if any(item.get("revision") == revision["id"] for item in self.state["profiles"]):
            return False
        return require_current or should_reprofile(revision, self._last_profile(), self.settings["reprofile_ratio"])

    def _framework(self) -> None:
        if self._profile_due():
            revision = self.state["current_revision"]["id"]
            self._register("profile", key=f"profile:{revision}", reason="accepted_gain_or_missing_profile")
            return
        proposals = self.router.pending_proposals
        pending = sorted(proposals)
        evidence_ids = {key: row["review_evidence"]["evidence_ref"]["id"]
                        for key, row in proposals.items() if "review_evidence" in row}
        bundle = digest({"proposals": pending, "evidence": evidence_ids} if evidence_ids else pending)
        attempts = self._control["review_attempts"]
        if (
            pending and attempts.get(bundle, 0) < self.settings["max_turns_per_cycle"]
            and self.budget >= 1 and self.agents.has_next("critic")
        ):
            self._call_role("critic")
            attempts[bundle] = attempts.get(bundle, 0) + 1
            return
        if self._schedule_review_evidence():
            return
        cycle = str(self.cycle)
        count = self._control["turns"].get(cycle, 0)
        hint = self._control["hints"].get(cycle)
        replan_proposals = self._unresolved_reviews()
        replans = self._control.setdefault("review_replans", {}) if replan_proposals else {}
        extra_turn = bool(replan_proposals) and not replans.get(cycle)
        if (
            not hint and self.budget >= 1 and (count < self.settings["max_turns_per_cycle"] or extra_turn)
            and self.agents.has_next("orchestration")
        ):
            if pivot := self._new_direction():
                self._emit("direction_changed", pivot)
                self._checkpoint()
            if replan_proposals:
                self._emit("review_replan", {"phase": self.phase, "cycle": self.cycle,
                                             "proposal_ids": replan_proposals, "reason": "framework_review_unresolved"})
            self._call_role("orchestration")
            self._control["turns"][cycle] = count + 1
            if replan_proposals:
                self._control.setdefault("review_feedback_handled", []).extend(replan_proposals)
                if count >= self.settings["max_turns_per_cycle"]:
                    replans[cycle] = True
            return
        if pending:
            self._emit("critic_bundle_unresolved", {"proposal_ids": pending, "cycle": self.cycle})
        else:
            self._emit("critic_skipped", {"reason": "empty_review_bundle", "cycle": self.cycle})
        self._change_phase(after_framework(self.settings), hint or "framework_round_complete")

    def _schedule_review_evidence(self) -> bool:
        if self.goals is None or self.budget < 3 or ROPE_ACTION not in self.state["runtime"]["capabilities"]:
            return False
        revision = self.state["current_revision"]
        profile = self._last_profile()
        if (not profile or profile.get("revision") != revision["id"] or revision["framework"] or revision["kernels"]
                or not self.store.verify_evidence(profile.get("report_ref") or {})):
            return False
        profile_ref = profile["report_ref"]
        if any(task["action"] == ROPE_ACTION and task["params"].get("profile_ref") == profile_ref
               for task in self.dispatcher.tasks.values()):
            return False
        for proposal_id, proposal in self.router.review_evidence_candidates.items():
            params = proposal["params"]
            if (proposal["phase"] != self.phase or params.get("parent") != revision["id"]
                    or params.get("profile_ref") != profile_ref or not is_rope_layout_edit(proposal)):
                continue
            task = self._register(ROPE_ACTION, key=f"review:{proposal_id}", proposal_id=proposal_id,
                                  revision=revision["id"], profile_ref=profile_ref, estimated_seconds=2)
            self._emit("review_evidence_requested", {
                "phase": self.phase, "cycle": self.cycle, "proposal_id": proposal_id,
                "task_id": task["task_id"], "kind": "rope_layout", "reason": proposal["review"]["reason"],
            })
            return True
        return False

    def _record_review_evidence(self, task: dict, result: dict, common: dict) -> None:
        params = task["params"]
        proposal_id = params["proposal_id"]
        ref = result.get("evidence", {}).get("rope_layout")
        status, reason = result["status"], result["reason"]
        summary = result.get("summary")
        valid_ref = ref is not None and self.store.verify_evidence(ref)
        if status == "ok":
            raw = read_json(self.store.artifact_path(ref["path"])) if valid_ref else {}
            if (raw.get("proposal_id") != proposal_id or raw.get("revision") != params["revision"]
                    or raw.get("profile_ref") != params["profile_ref"] or not summary or raw.get("summary") != summary):
                status, reason = "failed", "review_evidence_identity_mismatch"
        self._emit("review_evidence_recorded", {
            **common, "proposal_id": proposal_id, "status": status, "reason": reason,
            "summary": summary if status == "ok" else None,
            "evidence": {"rope_layout": ref} if valid_ref else {},
        })
        if status == "ok":
            self.router.reopen_for_review(self.phase, proposal_id, ref, summary)

    def _unresolved_reviews(self) -> list[str]:
        if self.goals is None:
            return []
        handled = set(self._control.get("review_feedback_handled", []))
        revision = self.state["current_revision"]["id"]
        return [row["proposal_id"] for row in self.router.review_feedback
                if row["action"] == "integrate_patch" and row.get("parent") == revision
                and row["proposal_id"] not in handled]

    def _review_feedback(self) -> list[dict]:
        collection_results = {
            task["params"]["proposal_id"]: task["result"]
            for task in self.dispatcher.tasks.values()
            if task["action"] == ROPE_ACTION and task.get("result")
        }
        feedback = self.router.review_feedback
        for row in feedback:
            if result := collection_results.get(row["proposal_id"]):
                row.update(evidence_status=result["status"], evidence_reason=result["reason"])
        return feedback

    def _kernel(self) -> None:
        route = self.settings["kernel_backend"]
        native = f"native_{route}" in self.state["runtime"]["capabilities"]
        if (native or route == "forge") and self._profile_due(require_current=native):
            revision = self.state["current_revision"]["id"]
            self._register("profile", key=f"profile:{revision}", reason=f"{route}_entry")
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

    def _new_direction(self) -> dict | None:
        if self.goals is None:
            return None
        supported = self.state["runtime"]["capabilities"]
        allowed = []
        if self.task.get("allowed_parameters"):
            allowed.append("parameters")
        if self.store.manifest["mode"] == "simulated" or self.scenario.get("agent_source_excerpts"):
            allowed.append("patch")
        if self.settings["profile_enabled"] and "profile" in supported:
            allowed.append("profile")
        if self.settings["kernel_backend"] != "disabled":
            allowed.append("kernel")
        pivot = direction_change(self.state, allowed)
        if pivot is not None and all(prior["tested_candidates"] != pivot["tested_candidates"]
                                    for prior in self.state["direction_changes"]):
            return pivot
        return None

    def _current_goal(self) -> dict | None:
        current = (self.state["current_revision"] or {}).get("id")
        for goal in reversed(self.state["goal_assessments"]):
            if goal.get("revision") == current:
                if not self.store.verify_evidence(goal["evidence"]["goal_assessment"]):
                    return None
                if goal["status"] == "pass" and any(
                    not self.store.verify_evidence(reference)
                    for arm in goal["arms"].values() for reference in arm["evidence"].values()
                ):
                    return None
                return goal
        return None

    def _record_goal(self, assessment: dict, attempt: str, common: dict | None = None) -> None:
        evidence = self.store.put_evidence(
            f"evidence/goals/{attempt}.json", assessment, producer="independent_goal_retest",
        )
        self._emit("goal_assessed", {
            **assessment, **(common or {"cycle": self.cycle, "phase": self.phase}),
            "evidence": {"goal_assessment": evidence},
        })
        if assessment["reason"] == "goal_candidate_correctness_failed":
            self._rollback_invalid_stack(assessment["revision"], evidence)

    def _rollback_invalid_stack(self, revision: str, evidence: dict) -> None:
        accepted = next((row for row in self.state["decisions"]
                         if row["candidate_id"] == revision and row["version_action"] == "KEEP"), None)
        previous = (accepted or {}).get("parent_revision") or {}
        if not self.store.verify_evidence(previous.get("validation_ref") or {}):
            baseline = self.state["baseline"]
            previous = {**baseline["revision"], "summary": baseline["measurement"]["summary"],
                        "validation_ref": baseline["evidence"]["revision_measurement"]}
        self._emit("accepted_stack_rolled_back", {
            "rejected_revision": revision, "revision": previous,
            "reason": "goal_candidate_correctness_failed", "evidence": evidence,
        })

    def _changed_from_baseline(self) -> bool:
        baseline = self.state["baseline"]
        current = self.state["current_revision"]
        return bool(baseline and current and baseline["revision"]["id"] != current["id"])

    def _schedule_goal_retest(self) -> None:
        self._register("stack_revalidate", key="goal", reason="goal_revalidate", estimated_seconds=GOAL_RETEST_COST)

    def _sweep(self) -> None:
        completed = self.state["sweeps"]
        if completed and completed[-1]["cycle"] == self.cycle:
            goal = self._current_goal() if self.goals else None
            if (self.goals and not goal and self._changed_from_baseline()
                    and self.budget >= GOAL_RETEST_COST and not self._phase_tasks("stack_revalidate")
                    and warrants_retest(self.state, self.goals, self.task["rules"])):
                self._schedule_goal_retest()
                return
            pivot = self._new_direction()
            next_phase, reason = after_sweep(
                self.state, self.settings, self.budget, goal_driven=self.goals is not None,
                goal_passed=bool(goal and goal["status"] == "pass"), new_direction=pivot is not None,
            )
            if pivot and next_phase == "FRAMEWORK_AGENT":
                self._emit("direction_changed", pivot)
            self._change_phase(next_phase, reason, next_cycle=next_phase == "FRAMEWORK_AGENT")
            return
        revision = self.state["current_revision"]["id"]
        if self._stack_needs_measurement() and self.budget >= 1 and not self._phase_tasks("stack_revalidate"):
            self._register("stack_revalidate", reason="before_sweep")
            return
        previous = next((row["revision"] for row in reversed(completed) if row["status"] == "ok"), None)
        available = self.budget
        if self.goals and self._changed_from_baseline() and self._current_goal() is None:
            available = max(0, available - GOAL_RETEST_COST)
        skip = sweep_skip_reason(self.settings, revision, previous, available)
        if self._stack_needs_measurement():
            skip = "stack_unverified"
        if skip:
            self._emit("sweep_recorded", {
                "status": "skipped", "reason": skip, "revision": revision, "cycle": self.cycle, "pairs": [],
            })
        else:
            levels = self.settings["sweep_concurrency"][: int(available // SWEEP_PAIR_COST)]
            self._register(
                "conc_sweep", revision=revision, concurrency_levels=levels,
                estimated_seconds=len(levels) * SWEEP_PAIR_COST,
            )

    def _close(self) -> None:
        if self.goals and self._current_goal() is None:
            if (self._changed_from_baseline() and self.budget >= GOAL_RETEST_COST
                    and not self._phase_tasks("stack_revalidate")):
                self._schedule_goal_retest()
                return
            current = (self.state["current_revision"] or {}).get("id")
            original = ((self.state["baseline"] or {}).get("revision") or {}).get("id")
            status, reason = "incomplete", "goal_retest_unavailable"
            if not original:
                reason = "goal_original_unavailable"
            elif original == current:
                status, reason = "fail", "goal_no_accepted_candidate"
            self._record_goal({"status": status, "reason": reason, "goals": self.goals,
                               "revision": current, "baseline_revision": original, "checks": {}}, "close")
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
