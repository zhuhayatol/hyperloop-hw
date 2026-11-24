"""Provider injection reaches the same upper flow and preserves the Session target lock."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.backends import SimulatedBackend, fingerprint
from hyperloop.contracts import ContractError
from hyperloop.coordinator import Coordinator

PROJECT = Path(__file__).resolve().parents[1]


def fixture(path: str) -> dict:
    return json.loads((PROJECT / path).read_text(encoding="utf-8"))


class AlternateRuntime:
    """A composition adapter that accepts declared revisions through the public seam."""

    def __init__(self, target: dict, revisions: dict, *, drifted: bool = False):
        self._runtime = SimulatedBackend(target, revisions)
        self.bound: dict[str, dict] = {}
        self.probe_count = 0
        self.drifted = drifted

    def probe(self) -> dict:
        result = self._runtime.probe()
        self.probe_count += 1
        result["diagnostic_probe_number"] = self.probe_count
        if self.drifted:
            result["environment"]["software"]["changed_dependency"] = "new"
            result["environment_fingerprint"] = fingerprint(result["environment"])
        return result

    def bind_revision(self, revision: dict, *, expected_stack: dict) -> None:
        self.bound[revision["id"]] = deepcopy(expected_stack)
        self._runtime.bind_revision(revision, expected_stack=expected_stack)

    def __getattr__(self, attribute: str):
        if attribute == "register_revision":
            raise AttributeError(attribute)
        return getattr(self._runtime, attribute)


class BackendFactoryTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = fixture("configs/task.simulated.json")
        self.task["settings"].update(max_cycles=1, kernel_backend="disabled", sweep_enabled=False)
        self.reference = fixture("fixtures/reference.simulated.json")
        self.scenario = fixture("fixtures/scenario.simulated.json")
        self.target = fixture("configs/target.910b2-tp2.simulated.json")

    @staticmethod
    def signature(state: dict) -> tuple:
        return state["phase"], [(row["candidate_id"], row["version_action"]) for row in state["decisions"]]

    def test_distinct_provider_reuses_upper_flow_and_gets_locked_stack(self) -> None:
        local = Coordinator.create(
            self.root, "local", self.task, fixture("configs/target.910b3.simulated.json"),
            self.reference, self.scenario,
        )
        custom_scenario = deepcopy(self.scenario)
        observations = custom_scenario.pop("revisions")
        adapters: list[AlternateRuntime] = []

        def factory(target, scenario):
            self.assertNotIn("revisions", scenario)
            adapter = AlternateRuntime(target, observations)
            adapters.append(adapter)
            return adapter

        alternate = Coordinator.create(
            self.root, "alternate", self.task, self.target, self.reference, custom_scenario, backend_factory=factory,
        )
        states = local.run(), alternate.run()
        self.assertEqual(self.signature(states[0]), self.signature(states[1]))
        self.assertEqual(states[1]["phase"], "DONE")
        profiles = [profile for profile in states[1]["profiles"] if profile["status"] == "ok"]
        self.assertTrue(profiles)
        self.assertEqual(profiles[0]["ranks"], [0, 1])
        accepted = states[1]["current_revision"]
        self.assertNotEqual(accepted["id"], self.task["initial_revision"])
        self.assertEqual(adapters[0].bound[accepted["id"]]["framework"], accepted["framework"])
        self.assertEqual(adapters[0].bound[accepted["id"]]["parameters"], accepted["parameters"])
        original_phases = [item["payload"]["phase"] for item in local.store.events() if item["type"] == "phase_changed"]
        alternate_phases = [item["payload"]["phase"] for item in alternate.store.events() if item["type"] == "phase_changed"]
        self.assertEqual(original_phases, alternate_phases)

    def test_drift_checks_each_new_run_and_open(self) -> None:
        adapter = AlternateRuntime(self.target, self.scenario["revisions"])
        session = Coordinator.create(
            self.root, "drift", self.task, self.target, self.reference, self.scenario,
            backend_factory=lambda target, scenario: adapter,
        )
        session.run(max_steps=1)
        adapter.drifted = True
        before = session.store.events()
        with self.assertRaisesRegex(ContractError, "target identity changed"):
            session.run(max_steps=1)
        self.assertEqual(session.store.events(), before)
        with self.assertRaisesRegex(ContractError, "target identity changed"):
            Coordinator.open(session.store.path, backend_factory=lambda target, scenario: adapter)
        adapter.drifted = False
        reopened = Coordinator.open(session.store.path, backend_factory=lambda target, scenario: adapter)
        self.assertEqual(reopened.run(max_steps=1)["phase"], "PRELUDE")

    def test_initial_provider_mode_and_target_are_verified(self) -> None:
        class WrongTarget(AlternateRuntime):
            def probe(self):
                result = super().probe()
                result["environment"]["tp_size"] = 1
                result["environment_fingerprint"] = fingerprint(result["environment"])
                return result

        with self.assertRaisesRegex(ContractError, "target.tp_size"):
            Coordinator.create(
                self.root, "invalid", self.task, self.target, self.reference, self.scenario,
                backend_factory=lambda target, scenario: WrongTarget(target, scenario["revisions"]),
            )
        real = {**self.target, "mode": "real"}
        with self.assertRaisesRegex(ContractError, "requires an explicitly supplied"):
            Coordinator.create(self.root, "default-real", self.task, real, self.reference, self.scenario)
        with self.assertRaisesRegex(ContractError, "probe mode"):
            Coordinator.create(
                self.root, "invalid-real", self.task, real, self.reference, self.scenario,
                backend_factory=lambda target, scenario: AlternateRuntime(target, scenario["revisions"]),
            )
