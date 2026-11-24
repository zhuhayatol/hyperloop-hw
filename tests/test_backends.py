"""Runtime adapter contract tests using deterministic local targets."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.backends import BackendError, SimulatedBackend, fingerprint


class SimulatedBackendTests(TestCase):
    def test_target_probe_distinguishes_single_and_dual_rank(self):
        single = SimulatedBackend({"id": "910b3", "devices": ["card0"], "tp_size": 1}).probe()
        dual = SimulatedBackend({"id": "dual", "devices": ["card2", "card3"], "tp_size": 2}).probe()
        self.assertEqual(single["environment"]["rank_map"], {"0": "card0"})
        self.assertEqual(dual["environment"]["rank_map"], {"0": "card2", "1": "card3"})
        self.assertIn("hccl", dual["capabilities"])
        self.assertEqual(dual["mode"], "simulated")
        self.assertEqual(dual["environment_fingerprint"], fingerprint(dual["environment"]))
        self.assertNotEqual(single["environment_fingerprint"], dual["environment_fingerprint"])

    def test_adapter_accounts_for_a_failed_launch(self):
        backend = SimulatedBackend({"id": "target", "devices": ["card0"], "tp_size": 1},
                                   {"broken": {"faults": ["launch"]}})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = backend.prepare("broken", root / "a", kind="framework")
            with self.assertRaises(BackendError):
                backend.launch(prepared, {"id": "point"})
            self.assertFalse(backend.recover_cleanup()["released"])
            with self.assertRaises(BackendError):
                backend.prepare("original", root / "b", kind="framework")
            self.assertTrue(backend.cleanup(prepared, None)["released"])
            self.assertTrue(backend.recover_cleanup()["released"])
            self.assertEqual(backend.prepare("original", root / "b", kind="framework")["revision"], "original")
