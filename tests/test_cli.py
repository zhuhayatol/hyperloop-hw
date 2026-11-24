"""Check the CLI validation and offline Session commands."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hyperloop.__main__ import main
from hyperloop.contracts import read_json
from hyperloop.session import SessionStore


class CliTests(unittest.TestCase):
    def test_max_steps_rejects_zero_before_opening_session(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["resume", "unopened-session", "--max-steps", "0"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("max steps must be a positive integer", stderr.getvalue())

    def test_status_and_report_read_the_same_committed_events(self) -> None:
        with TemporaryDirectory() as temporary:
            configs = Path(__file__).resolve().parents[1] / "configs"
            store = SessionStore.create(Path(temporary), "simulated", "cli-demo", {
                "task": read_json(configs / "task.simulated.json"),
                "target": read_json(configs / "target.910b3.simulated.json"),
            })
            store.emit("phase_changed", {"phase": "FRAMEWORK_AGENT", "cycle": 2})

            responses = {}
            for command in ("status", "report"):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main([command, str(store.path)]), 0)
                responses[command] = json.loads(stdout.getvalue())

            for response in responses.values():
                self.assertEqual((response["phase"], response["cycle"], response["last_seq"]), ("FRAMEWORK_AGENT", 2, 1))
            for path in responses["report"]["artifacts"].values():
                self.assertTrue(Path(path).is_file())
            self.assertTrue((store.path / "evidence/index.jsonl").is_file())
            self.assertTrue((store.path / "recipes.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
