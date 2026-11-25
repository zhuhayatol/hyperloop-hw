"""Verify campaign-level acceptance through real dispatcher and evidence paths."""

import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hyperloop.backends import SimulatedBackend
from hyperloop.coordinator import Coordinator

from test_deepseek_agents import KEY, completion, explore
from test_pipeline import load_fixture


class GoalAcceptanceTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "sessions"

    def create(self, name: str, ttft: float, *, cycles: int = 2, live_agent: bool = False,
               scenario: dict | None = None) -> Coordinator:
        task = load_fixture("configs/task.simulated.json")
        task["goals"] = {"throughput_gain_pct": 15, "ttft_reduction_pct": 15}
        task["rules"]["tolerance"] = 0
        task["settings"].update(target_gain_pct=10, record_model_forward=False, sweep_enabled=False, max_cycles=cycles)
        if scenario is None:
            scenario = load_fixture("fixtures/scenario.simulated.json")
        scenario["revisions"]["K-FORGE"]["ttft_ms"] = ttft
        if live_agent:
            task["settings"].update(kernel_backend="disabled", max_turns_per_cycle=1)
            scenario["agent_provider"] = {"provider": "deepseek"}
        return Coordinator.create(
            self.root, name, task, load_fixture("configs/target.910b3.simulated.json"),
            load_fixture("fixtures/reference.simulated.json"), scenario,
        )

    def test_accepted_stack_passes_a_fresh_pair_and_can_stop(self):
        campaign = self.create("pass-pair", 100)
        state = campaign.run()
        goal = state["goal_assessments"][-1]
        self.assertEqual((state["phase"], state["stop_reason"], state["cycle"]), ("DONE", "target_reached", 1))
        self.assertEqual((goal["status"], goal["revision"], goal["baseline_revision"]), ("pass", "K-FORGE", "base"))
        self.assertEqual(goal["checks"], {"throughput": True, "ttft_p95": True, "tpot_p95": True, "correctness": True})
        self.assertAlmostEqual(goal["comparison"]["over_original"]["throughput_pct"], 22)
        for arm in goal["arms"].values():
            self.assertNotEqual(arm["attempt_id"], state["baseline"]["attempt_id"])
            self.assertTrue(campaign.store.verify_evidence(arm["evidence"]["revision_measurement"]))
        self.assertTrue(campaign.store.verify_evidence(goal["evidence"]["goal_assessment"]))

    def test_throughput_alone_cannot_stop_and_terminal_accepted_stack_is_tested(self):
        state = self.create("one-metric", 123).run()
        self.assertEqual((state["phase"], state["cycle"]), ("DONE", 2))
        self.assertNotEqual(state["stop_reason"], "target_reached")
        goal = state["goal_assessments"][-1]
        self.assertEqual(goal["status"], "fail")
        self.assertTrue(goal["checks"]["throughput"])
        self.assertFalse(goal["checks"]["ttft_p95"])
        self.assertEqual(goal["phase"], "CLOSE")

    def test_explore_prefers_both_goals_against_the_original_baseline(self):
        target = load_fixture("configs/target.910b3.simulated.json")
        reference = load_fixture("fixtures/reference.simulated.json")
        for uses_goals, winner, other in ((True, "Both", "Fast"), (False, "Fast", "Both")):
            with self.subTest(uses_goals=uses_goals):
                task = load_fixture("configs/task.simulated.json")
                task["rules"]["tolerance"] = 0
                task["settings"].update(
                    kernel_backend="disabled", record_model_forward=False, sweep_enabled=False,
                    max_cycles=1, max_turns_per_cycle=2,
                )
                if uses_goals:
                    task["goals"] = {"throughput_gain_pct": 15, "ttft_reduction_pct": 15}
                scenario = load_fixture("fixtures/scenario.simulated.json")
                orchestration = scenario["agents"]["orchestration"]
                first = orchestration[0][0]
                spec = first["params"]["candidates"][0]
                orchestration[1] = [{
                    **first, "idempotency_key": "explore-second-batch",
                    "params": {"candidates": [
                        {**spec, "id": name, "parameters": {"chunked_prefill_size": chunk}}
                        for name, chunk in (("Fast", 2048), ("Both", 1024), ("SmallStep", 8192))
                    ]},
                }]
                scenario["revisions"].update({
                    "R01": {"throughput": 114.8, "ttft_ms": 112.5, "tpot_ms": 27.8},
                    "Fast": {"throughput": 122, "ttft_ms": 115, "tpot_ms": 27},
                    "Both": {"throughput": 118, "ttft_ms": 100, "tpot_ms": 27},
                    "SmallStep": {"throughput": 115.8, "ttft_ms": 90, "tpot_ms": 27},
                })
                campaign = Coordinator.create(self.root, f"explore-{winner}", task, target, reference, scenario)
                state = campaign.run()
                decisions = {row["candidate_id"]: row for row in state["decisions"]}
                self.assertEqual(state["current_revision"]["id"], winner)
                self.assertEqual(decisions[winner]["version_action"], "KEEP")
                self.assertEqual(decisions[other]["reason"], "stronger_candidate_selected")
                self.assertEqual(decisions["SmallStep"]["reason"], "gain_below_threshold")
                self.assertGreater(decisions["SmallStep"]["comparison"]["over_original"]["throughput_pct"], 15)
                both = decisions["Both"]
                self.assertEqual(both["parent"], "R01")
                self.assertGreater(both["comparison"]["over_parent"]["ttft_pct"], -15)
                self.assertAlmostEqual(both["comparison"]["over_original"]["ttft_pct"], -20)
                if uses_goals:
                    goal = state["goal_assessments"][-1]
                    self.assertEqual((goal["revision"], goal["status"]), ("Both", "pass"))
                    self.assertNotEqual(goal["arms"]["candidate"]["attempt_id"], both["attempt_id"])
                else:
                    self.assertEqual(state["goal_assessments"], [])

    def test_screening_pass_is_not_a_substitute_for_final_independent_samples(self):
        campaign = self.create("retest-regression", 100)
        benchmark = SimulatedBackend.benchmark

        def final_ttft_regression(adapter, instance, workload, *, repeat, warmup):
            observed = benchmark(adapter, instance, workload, repeat=repeat, warmup=warmup)
            if Path(instance["workspace"]).parent.name.endswith("-c"):
                for row in observed["requests"]:
                    row["ttft_ms"] = 120
            return observed

        with patch.object(SimulatedBackend, "benchmark", final_ttft_regression):
            state = campaign.run()
        goal = state["goal_assessments"][-1]
        self.assertTrue(goal["checks"]["throughput"])
        self.assertFalse(goal["checks"]["ttft_p95"])
        self.assertNotEqual(state["stop_reason"], "target_reached")

    def test_known_transient_forge_action_retries_once_with_both_receipts(self):
        campaign = self.create("retry-forge", 100)
        original = campaign.actions.kernel
        calls = []

        def timeout_once(task, state):
            calls.append(task["task_id"])
            return {"status": "failed", "reason": "forge_worker_timeout"} if len(calls) == 1 else original(task, state)

        with patch.object(campaign.actions, "kernel", timeout_once):
            state = campaign.run()
        retries = [event["payload"] for event in campaign.store.events() if event["type"] == "task_retry_scheduled"]
        self.assertEqual(retries, [{"action": "kernel_agent", "reason": "forge_worker_timeout",
                                    "task_id": calls[0], "retry_task_id": calls[1]}])
        self.assertEqual(state["tasks"][calls[0]]["result"]["reason"], "forge_worker_timeout")
        self.assertEqual(state["tasks"][calls[1]]["params"]["retry_of"], calls[0])
        self.assertEqual(state["goal_assessments"][-1]["status"], "pass")

    def test_output_mismatch_in_final_pair_rolls_back_the_accepted_stack(self):
        campaign = self.create("retest-correctness", 100, cycles=1)
        observe = SimulatedBackend.observe_outputs

        def damaged_candidate(adapter, instance, workload, *, scope):
            observed = observe(adapter, instance, workload, scope=scope)
            if instance["revision"] == "K-FORGE" and Path(instance["workspace"]).parent.name.endswith("-c"):
                observed["cases"][0]["output"][0] += 1
            return observed

        with patch.object(SimulatedBackend, "observe_outputs", damaged_candidate):
            state = campaign.run()
        self.assertEqual(state["current_revision"]["id"], "R02")
        rollback = next(event["payload"] for event in campaign.store.events() if event["type"] == "accepted_stack_rolled_back")
        self.assertEqual(rollback["rejected_revision"], "K-FORGE")
        rejected = next(goal for goal in state["goal_assessments"] if goal["revision"] == "K-FORGE")
        self.assertEqual((rejected["status"], rejected["reason"]), ("fail", "goal_candidate_correctness_failed"))

    def test_live_agent_timeout_recovers_without_a_second_coordinator(self):
        campaign = self.create("retry-agent", 100, cycles=1, live_agent=True)
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}),
            patch("hyperloop.agents.deepseek._HTTP.open", side_effect=[TimeoutError(), completion([explore()])]),
        ):
            state = campaign.run()
        self.assertEqual(state["phase"], "DONE")
        self.assertIsNone(state["blocker"])
        events = campaign.store.events()
        provider = [event["payload"] for event in events if event["type"] == "agent_provider_call"]
        self.assertEqual([row["status"] for row in provider], ["failed", "ok"])
        self.assertTrue(all(campaign.store.verify_evidence(row["evidence"]) for row in provider))
        retried, = [event["payload"] for event in events if event["type"] == "task_retry_scheduled"]
        self.assertEqual((retried["role"], retried["reason"]), ("orchestration", "request_timeout"))

    def test_two_reverted_trials_of_same_parameter_advise_the_next_role(self):
        scenario = load_fixture("fixtures/scenario.simulated.json")
        candidates = scenario["agents"]["orchestration"][0][0]["params"]["candidates"]
        new_trial = {**candidates[0], "id": "R01-alt", "parameters": {"chunked_prefill_size": 3072}}
        candidates.append(new_trial)
        scenario["revisions"]["R01"]["throughput"] = 100.4
        scenario["revisions"]["R01-alt"] = {**deepcopy(scenario["revisions"]["R01"]), "throughput": 101}
        campaign = self.create("scope-pivot", 100, cycles=1, scenario=scenario)
        state = campaign.run()
        pivot, = state["direction_changes"]
        self.assertEqual(pivot["tested_candidates"], ["R01", "R01-alt"])
        self.assertEqual(pivot["mechanism"], ["chunked_prefill_size"])
        self.assertIn("patch", pivot["alternatives"])
        turns = [event["payload"] for event in campaign.store.events() if event["type"] == "agent_turn_started"]
        self.assertTrue(any(row["role"] == "orchestration" and row["context"].get("direction_change") == pivot for row in turns))
