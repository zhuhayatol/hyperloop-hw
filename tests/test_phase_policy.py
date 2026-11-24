import unittest

from hyperloop.phases.machine import (
    after_framework, after_prelude, after_sweep, should_reprofile, sweep_skip_reason,
)


class PhasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "framework_enabled": True,
            "kernel_backend": "forge",
            "sweep_enabled": True,
            "target_gain_pct": 30,
            "max_cycles": 3,
        }
        self.state = {
            "last_conc_sweep": {"status": "ok"},
            "baseline": {"measurement": {"summary": {"throughput": 100}}},
            "current_revision": {"id": "r1", "summary": {"throughput": 115}},
            "cycle": 1,
        }

    def test_phase_switches(self):
        self.assertEqual(after_prelude(self.settings), "FRAMEWORK_AGENT")
        self.assertEqual(after_prelude({**self.settings, "framework_enabled": False}), "KERNEL_AGENT")
        self.assertEqual(after_framework({**self.settings, "kernel_backend": "disabled"}), "SWEEP")

    def test_scan_admission_distinguishes_skips(self):
        self.assertIsNone(sweep_skip_reason(self.settings, "r1", "base", 2))
        self.assertEqual(sweep_skip_reason(self.settings, "r1", "r1", 1), "no_validated_change_since_sweep")
        self.assertEqual(sweep_skip_reason(self.settings, "r1", "base", 1), "insufficient_sweep_budget")

    def test_scan_terminal_under_failure_or_cycle_limit(self):
        self.assertEqual(after_sweep(self.state, self.settings, 5)[0], "FRAMEWORK_AGENT")
        self.assertEqual(after_sweep({**self.state, "cycle": 3}, self.settings, 5), ("CLOSE", "cycle_limit"))
        self.state["last_conc_sweep"] = {"status": "failed"}
        self.assertEqual(after_sweep(self.state, self.settings, 5), ("CLOSE", "sweep_failed_or_incomplete"))

    def test_reprofile_uses_verified_anchor(self):
        revision = {"id": "r1", "summary": {"throughput": 110}}
        self.assertTrue(should_reprofile(revision, None, 1.1))
        self.assertTrue(should_reprofile(revision, {"revision": "base", "anchor_throughput": 100}, 1.1))
        self.assertFalse(should_reprofile(revision, {"revision": "base", "anchor_throughput": 101}, 1.1))
        self.assertFalse(should_reprofile(revision, {"revision": "r1", "anchor_throughput": 90}, 1.1))
