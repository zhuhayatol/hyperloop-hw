"""Verify that historical settings are advisory and bound to Session evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.actions.knowledge import RECIPE_PATH, build_recipe, load_verified_recipe, validate_recipe
from hyperloop.backends import SimulatedBackend
from hyperloop.contracts import ContractError, read_json
from hyperloop.session import SessionStore

PROJECT = Path(__file__).resolve().parents[1]


class RecipeTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = read_json(PROJECT / "configs/task.simulated.json")
        self.target = read_json(PROJECT / "configs/target.910b3.simulated.json")
        self.runtime = SimulatedBackend(self.target).probe()
        self.parameters = {"chunked_prefill_size": 4096}
        self.recipe = build_recipe(
            self.task, self.target, self.runtime,
            source_session_id="source", revision="accepted", parameters=self.parameters,
        )

    def make_source(self, *, close: bool = True, recorded_revision: str = "accepted") -> SessionStore:
        store = SessionStore.create(
            self.root, "simulated", "source", {"task": self.task, "target": self.target}
        )
        receipt = store.put_evidence(RECIPE_PATH, self.recipe, producer="close")
        changes = [
            ("runtime_probed", self.runtime),
            ("baseline_recorded", {"result": {}, "revision": {"id": "base", "parameters": {}}}),
            ("candidate_decision", {
                "decision": {"candidate_id": recorded_revision, "parent": "base", "version_action": "KEEP"},
                "promote": {"id": recorded_revision, "parameters": self.parameters},
            }),
            ("recipe_recorded", {**self.recipe, "reference": receipt}),
        ]
        if close:
            changes.extend([("stopped", {"reason": "finished"}), ("phase_changed", {"phase": "DONE", "cycle": 1})])
        store.commit(changes)
        return store

    def test_matching_recipe_has_only_copied_settings_and_requires_revalidation(self) -> None:
        result = validate_recipe(self.recipe, self.task, self.target, self.runtime)
        self.assertEqual((result["status"], result["reason"]), ("applicable", "matched_frozen_scope"))
        self.assertTrue(result["requires_revalidation"])
        self.assertEqual(result["parameters"], self.parameters)
        self.assertNotIn("version_action", result)
        result["parameters"]["chunked_prefill_size"] = 1
        self.assertEqual(self.recipe["parameters"]["chunked_prefill_size"], 4096)

    def test_model_tokenizer_workload_and_target_environment_are_matched(self) -> None:
        cases = [
            ("model_mismatch", "task", "model", "revision", "other-model"),
            ("model_mismatch", "task", "model", "tokenizer_revision", "other-tokenizer"),
            ("workload_mismatch", "task", "workload", "concurrency", 8),
            ("target_mismatch", "target", None, "id", "another-target"),
            ("mode_mismatch", "target", None, "mode", "real"),
            ("environment_mismatch", "runtime", None, "environment_fingerprint", "1" * 64),
        ]
        for reason, context, group, field, value in cases:
            with self.subTest(reason=reason, field=field):
                inputs = deepcopy({"task": self.task, "target": self.target, "runtime": self.runtime})
                changed = inputs[context][group] if group else inputs[context]
                changed[field] = value
                result = validate_recipe(self.recipe, **inputs)
                self.assertEqual((result["status"], result["reason"]), ("incompatible", reason))
                self.assertNotIn("parameters", result)

    def test_new_parameter_policy_and_schema_apply(self) -> None:
        restricted = {**self.task, "allowed_parameters": ["other"]}
        result = validate_recipe(self.recipe, restricted, self.target, self.runtime)
        self.assertEqual(result["reason"], "parameter_not_allowed")
        for recipe in (
            {**self.recipe, "version_action": "KEEP"},
            {**self.recipe, "requires_revalidation": False},
            {**self.recipe, "parameters": {"chunked_prefill_size": float("nan")}},
        ):
            self.assertEqual(validate_recipe(recipe, self.task, self.target, self.runtime)["status"], "invalid")

    def test_load_verified_recipe_and_refuse_modified_evidence(self) -> None:
        source = self.make_source()
        path = source.artifact_path(RECIPE_PATH)
        self.assertEqual(load_verified_recipe(path), self.recipe)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ContractError, "digest"):
            load_verified_recipe(path)

    def test_recipe_source_must_be_closed(self) -> None:
        source = self.make_source(close=False)
        with self.assertRaisesRegex(ContractError, "not closed"):
            load_verified_recipe(source.artifact_path(RECIPE_PATH))

    def test_recipe_must_describe_final_accepted_revision(self) -> None:
        source = self.make_source(recorded_revision="other")
        with self.assertRaisesRegex(ContractError, "final accepted revision"):
            load_verified_recipe(source.artifact_path(RECIPE_PATH))
