"""Exercise the operator's two-file entry with real schemas and simulated work."""

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hyperloop.__main__ import main
from hyperloop.actions.runtime import RuntimeActions
from hyperloop.contracts import ContractError, read_json
from hyperloop.session import SessionStore, file_lease


_TASK = """# 910B3 推理优化

原始分块 512。最终相对原始基线，吞吐提高至少 15%，
首 token 延迟（TTFT P95）降低至少 15%。
输出逐 token 一致，后续 token 延迟（TPOT P95）最多增加 10%。
"""


class StartTests(unittest.TestCase):
    def _inputs(self, job: Path, config: str) -> list[str]:
        task, launch = job / "任务.md", job / "启动配置.yaml"
        task.write_text(_TASK, encoding="utf-8")
        launch.write_text(config, encoding="utf-8")
        return ["start", "--task", str(task), "--config", str(launch)]

    def _run_baseline_case(
        self, job: Path, session_id: str, *, options: str = "", profile: bool = True, framework: bool = False,
    ) -> tuple[int, str, Path]:
        config = f"""target: 910b3-tp1-simulated
session_id: {session_id}
session_root: results
allowed_parameters: [chunked_prefill_size]
settings:
  max_cycles: 1
  max_turns_per_cycle: 1
  sweep_enabled: false
  kernel_backend: disabled
  record_model_forward: false
  profile_enabled: {str(profile).lower()}
  framework_enabled: {str(framework).lower()}
{options}"""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(self._inputs(job, config))
        return status, stderr.getvalue(), job / "results" / session_id

    def test_one_command_preserves_inputs_and_runs_a_frozen_session(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            config = """target: 910b3-tp1-simulated
session_id: owner
session_root: results
allowed_parameters: [chunked_prefill_size]
settings:
  max_cycles: 1
  sweep_enabled: false
  kernel_backend: disabled
"""
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(self._inputs(job, config))
            self.assertEqual(status, 0, stderr.getvalue())
            result = json.loads(stdout.getvalue())
            root = job / "results" / "owner"
            store = SessionStore(root / "simulated" / "owner")
            self.assertEqual(result["phase"], "DONE")
            self.assertEqual(store.manifest["task"]["goals"], {"throughput_gain_pct": 15.0, "ttft_reduction_pct": 15.0})
            self.assertEqual(store.manifest["task"]["allowed_parameters"], ["chunked_prefill_size"])
            self.assertEqual((root / "inputs" / "任务.md").read_text(encoding="utf-8"), _TASK)
            self.assertEqual((root / "inputs" / "启动配置.yaml").read_text(encoding="utf-8"), config)
            startup = read_json(root / "startup.json")
            self.assertEqual((startup["status"], startup["baseline_mode"]), ("done", "measure"))
            self.assertNotIn("baseline_reuse", store.manifest)
            self.assertTrue(any(task["action"] == "baseline" for task in store.state()["tasks"].values()))
            self.assertTrue((store.path / "report.md").is_file())
            self.assertIn("已锁定模拟参考值", (root / "console.log").read_text(encoding="utf-8"))

    def test_reuse_latest_selects_newest_compatible_original_and_its_profile(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            for identity, options, framework in (
                ("older", "", False),
                ("match", "", True),
                ("mismatch", "workload:\n  concurrency: 2\n", False),
            ):
                status, stderr, _ = self._run_baseline_case(job, identity, options=options, framework=framework)
                self.assertEqual(status, 0, stderr)
            source = SessionStore(job / "results" / "match" / "simulated" / "match")
            self.assertEqual(source.state()["current_revision"]["id"], "R01")
            scenario = read_json(Path(__file__).resolve().parents[1] / "fixtures" / "scenario.simulated.json")
            scenario["revisions"]["R01"] = {"throughput": 118, "ttft_ms": 100, "tpot_ms": 27}
            (job / "candidate.json").write_text(json.dumps(scenario), encoding="utf-8")
            status, stderr, root = self._run_baseline_case(
                job, "reuse", options="baseline_mode: reuse_latest\nscenario: candidate.json\n", framework=True,
            )
            self.assertEqual(status, 0, stderr)
            startup = read_json(root / "startup.json")
            self.assertEqual((startup["status"], startup["baseline_mode"]), ("done", "reuse_latest"))
            store = SessionStore(root / "simulated" / "reuse")
            self.assertEqual(store.manifest["baseline_reuse"]["session_id"], "match")
            events = store.events()
            baseline, = [row["payload"] for row in events if row["type"] == "baseline_recorded"]
            profile, = [row["payload"] for row in events if row["type"] == "profile_recorded"
                        and row["payload"]["revision"] == "base"]
            self.assertEqual(baseline["reused_from"]["session_id"], "match")
            self.assertEqual(profile["reused_from"]["session_id"], "match")
            self.assertEqual(baseline["result"]["revision"]["id"], "base")
            for result in (baseline["result"], profile):
                for reference in result["evidence"].values():
                    self.assertTrue(store.verify_evidence(reference), reference)
                    self.assertEqual(store.artifact_path(reference["path"]).read_bytes(),
                                     source.artifact_path(reference["path"]).read_bytes())
            state = store.state()
            actions = [row["action"] for row in state["tasks"].values()]
            self.assertNotIn("baseline", actions)
            self.assertIn("explore", actions)
            self.assertFalse(any(row["action"] == "profile" and row["result"].get("revision") == "base"
                                 for row in state["tasks"].values()))
            goal, = state["goal_assessments"]
            self.assertEqual((goal["status"], goal["baseline_revision"], goal["revision"]), ("pass", "base", "R01"))
            self.assertNotEqual(goal["arms"]["original"]["attempt_id"], baseline["result"]["attempt_id"])

    def test_reuse_runs_fresh_profile_when_latest_matching_source_has_none(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            status, stderr, _ = self._run_baseline_case(job, "previous-with-profile")
            self.assertEqual(status, 0, stderr)
            status, stderr, root = self._run_baseline_case(job, "newest-without-profile", profile=False)
            self.assertEqual(status, 0, stderr)
            source = SessionStore(root / "simulated" / "newest-without-profile")
            self.assertEqual(source.state()["profiles"], [])
            status, stderr, root = self._run_baseline_case(job, "fresh-profile", options="baseline_mode: reuse_latest\n")
            self.assertEqual(status, 0, stderr)
            store = SessionStore(root / "simulated" / "fresh-profile")
            self.assertEqual(store.manifest["baseline_reuse"]["session_id"], "newest-without-profile")
            state = store.state()
            self.assertNotIn("baseline", [row["action"] for row in state["tasks"].values()])
            profile, = state["profiles"]
            self.assertEqual((profile["revision"], profile["status"]), ("base", "ok"))
            self.assertNotIn("reused_from", profile)
            self.assertEqual(state["tasks"][profile["task_id"]]["action"], "profile")
            self.assertTrue(store.verify_evidence(profile["report_ref"]))

    def test_reuse_without_compatible_evidence_explains_how_to_measure(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            status, stderr, _ = self._run_baseline_case(job, "original", profile=False)
            self.assertEqual(status, 0, stderr)
            scenario = read_json(Path(__file__).resolve().parents[1] / "fixtures" / "scenario.simulated.json")
            scenario["revisions"]["base"]["ttft_ms"] = 90
            (job / "different.json").write_text(json.dumps(scenario), encoding="utf-8")
            for identity, options in (
                ("different-protocol", "rules:\n  repeats: 4\n"),
                ("different-original", "scenario: different.json\n"),
            ):
                with self.subTest(identity=identity):
                    status, stderr, root = self._run_baseline_case(
                        job, identity, options="baseline_mode: reuse_latest\n" + options,
                    )
                    self.assertEqual(status, 1)
                    self.assertIn("set baseline_mode: measure", stderr)
                    startup = read_json(root / "startup.json")
                    self.assertEqual((startup["status"], startup["baseline_mode"]), ("failed", "reuse_latest"))
                    self.assertFalse((root / "simulated" / identity / "manifest.json").exists())

    def test_conflicting_metrics_or_duplicate_keys_exit_before_launch(self) -> None:
        for tail, detail in (
            ("goals:\n  throughput_gain_pct: 30\n", "disagree on throughput_gain_pct"),
            ("target: 910b3-tp1\n", "duplicate YAML field"),
        ):
            with self.subTest(detail=detail), TemporaryDirectory() as temporary:
                job = Path(temporary)
                config = "target: 910b3-tp1-simulated\nsession_root: results\n" + tail
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(self._inputs(job, config)), 1)
                self.assertIn(detail, stderr.getvalue())
                self.assertFalse((job / "results").exists())

    def test_single_command_recovers_a_failed_action_and_delivers_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            scenario = read_json(Path(__file__).resolve().parents[1] / "fixtures" / "scenario.simulated.json")
            scenario["revisions"]["K-FORGE"]["ttft_ms"] = 100.0
            (job / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
            config = """target: 910b3-tp1-simulated
session_id: retry
session_root: results
scenario: scenario.json
settings:
  max_cycles: 1
  sweep_enabled: false
  record_model_forward: false
"""
            kernel = RuntimeActions.kernel
            calls = []

            def timeout_once(actions, task, state):
                calls.append(task["task_id"])
                if len(calls) == 1:
                    return {"status": "failed", "reason": "forge_worker_timeout"}
                return kernel(actions, task, state)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (patch.object(RuntimeActions, "kernel", timeout_once),
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                status = main(self._inputs(job, config))
            self.assertEqual(status, 0, stderr.getvalue())
            result = json.loads(stdout.getvalue())
            root = job / "results" / "retry"
            store = SessionStore(root / "simulated" / "retry")
            self.assertEqual((result["phase"], result["goal_assessment"]["status"]), ("DONE", "pass"))
            retry, = [row["payload"] for row in store.events() if row["type"] == "task_retry_scheduled"]
            self.assertEqual((retry["task_id"], retry["retry_task_id"]), tuple(calls))
            self.assertIn("重试", (root / "console.log").read_text(encoding="utf-8"))
            self.assertTrue(store.verify_evidence(result["goal_assessment"]["evidence"]["goal_assessment"]))

    def test_reference_failure_is_retained_without_creating_a_session(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            config = "target: 910b3-tp1\nsession_id: failed\nsession_root: results\n"
            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-placeholder"}),
                patch("hyperloop.start._check_idle_device"),
                patch("hyperloop.__main__._capture_reference", side_effect=ContractError("fixture startup error")),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(self._inputs(job, config)), 1)
            output = job / "results" / "failed"
            status = read_json(output / "startup.json")
            self.assertEqual((status["status"], status["stage"], status["reason"]),
                             ("failed", "reference", "fixture startup error"))
            self.assertIn("fixture startup error", (output / "console.log").read_text(encoding="utf-8"))
            self.assertFalse((output / "real" / "failed").exists())

    def test_real_reference_is_not_launched_while_output_root_has_an_npu_owner(self) -> None:
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            output = job / "results"
            output.mkdir()
            config = "target: 910b3-tp1\nsession_id: busy\nsession_root: results\n"
            with (
                file_lease(output / ".hyperloop-npu.lock"),
                patch("hyperloop.__main__._capture_reference") as capture,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(self._inputs(job, config)), 1)
                capture.assert_not_called()
            state = read_json(output / "busy" / "startup.json")
            self.assertEqual(state["status"], "blocked")
            self.assertIn(".hyperloop-npu.lock", state["reason"])
