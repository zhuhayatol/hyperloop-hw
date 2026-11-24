from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperloop.contracts import ContractError, digest, read_json, safe_id, validate_target
from hyperloop.session import SessionBusy, SessionStore, atomic_write


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = SessionStore.create(
            self.root, "simulated", "case", {"task": {"id": "prefill"}, "target": {"id": "local"}}
        )

    def test_manifest_fingerprint_and_session_identity(self):
        record = read_json(self.session.path / "manifest.json")
        self.assertEqual(record.pop("fingerprint"), digest(record))
        with self.assertRaises(FileExistsError):
            SessionStore.create(self.root, "simulated", "case", {})
        (self.session.path / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "fingerprint mismatch"):
            SessionStore(self.session.path)

    def test_torn_frame_preserves_ordinals_and_original_bytes(self):
        self.session.emit("first", {})
        log = self.session.path / "events.jsonl"
        with log.open("ab") as handle:
            handle.write(b'{"events":[')
        self.assertEqual([e["type"] for e in self.session.events()], ["first"])
        self.session.commit([("second", {}), ("third", {})])
        self.assertEqual([e["seq"] for e in self.session.events()], [1, 2, 3])
        saved = list((self.session.path / "recovery").glob("*.bin"))
        self.assertEqual([item.read_bytes() for item in saved], [b'{"events":['])

    def test_committed_corruption_is_not_hidden_as_crash_tail(self):
        log = self.session.path / "events.jsonl"
        log.write_bytes(b'invalid\n')
        with self.assertRaisesRegex(ContractError, "invalid committed"):
            self.session.state()

    def test_replay_works_if_snapshot_replacement_fails_after_commit(self):
        def fail_snapshot(path, data):
            if path.name == "state.json":
                raise OSError("fault after event flush")
            atomic_write(path, data)

        with patch("hyperloop.session.atomic_write", side_effect=fail_snapshot):
            with self.assertRaisesRegex(OSError, "fault after event flush"):
                self.session.emit("phase_changed", {"phase": "FRAMEWORK", "cycle": 2})
        self.assertEqual(read_json(self.session.path / "state.json")["phase"], "PRELUDE")
        self.assertEqual(SessionStore(self.session.path).state()["phase"], "FRAMEWORK")

    def test_original_baseline_and_parent_identity_guard_promotion(self):
        baseline = {"result": {"points": ["decode"]}, "revision": {"id": "base"}}
        self.session.emit("baseline_recorded", baseline)
        with self.assertRaisesRegex(ContractError, "immutable"):
            self.session.emit("baseline_recorded", baseline)
        with self.assertRaisesRegex(ContractError, "current accepted parent"):
            self.session.emit("candidate_decision", {
                "decision": {"candidate_id": "r1", "parent": "stale", "version_action": "KEEP"},
                "promote": {"id": "r1"},
            })
        self.session.emit("candidate_decision", {
            "decision": {"candidate_id": "r1", "parent": "base", "version_action": "KEEP"},
            "promote": {"id": "r1"},
        })
        self.assertEqual(self.session.state()["current_revision"], {"id": "r1"})
        self.assertEqual(self.session.state()["baseline"], baseline["result"])
        self.assertEqual(len(self.session.events()), 2)

    def test_evidence_is_contained_immutable_and_verifiable(self):
        ref = self.session.put_evidence("evidence/example.json", {"data": 1}, producer="test")
        self.assertTrue(self.session.verify_evidence(ref))
        self.assertEqual(ref, self.session.put_evidence(ref["path"], {"data": 1}, producer="test"))
        with self.assertRaisesRegex(ContractError, "immutable"):
            self.session.put_evidence(ref["path"], {"data": 2}, producer="test")
        for name in ("../other.json", "evidence/../../other.json", "C:/escape", "/escape", "evidence\\escape"):
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.session.artifact_path(name)
        self.session.artifact_path(ref["path"]).write_bytes(b"corrupt")
        self.assertFalse(self.session.verify_evidence(ref))

    def test_execution_lease_excludes_second_coordinator(self):
        another = SessionStore(self.session.path)
        with self.session.execution_lease():
            with self.assertRaises(SessionBusy):
                with another.execution_lease():
                    self.fail("a second Coordinator acquired the same Session")
        with another.execution_lease():
            self.assertEqual(another.state()["phase"], "PRELUDE")

    def test_configuration_rejects_ambiguous_input(self):
        config = self.root / "target.json"
        config.write_text('{"mode":"simulated","mode":"real"}', encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            read_json(config)
        with self.assertRaisesRegex(ContractError, "tp_size exceeds"):
            validate_target({"id": "test", "mode": "simulated", "devices": ["card"], "tp_size": 2})
        for item in ("../run", "x/y", "NUL", "a."):
            with self.subTest(item=item), self.assertRaises(ContractError):
                safe_id(item, "candidate")


if __name__ == "__main__":
    unittest.main()
