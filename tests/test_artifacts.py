"""The Session views index existing evidence and can be regenerated."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.artifacts import publish_indexes
from hyperloop.contracts import ContractError
from hyperloop.session import SessionStore


class ArtifactTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = SessionStore.create(
            Path(temporary.name),
            "simulated",
            "indexed",
            {"task": {"workload": {"id": "prefill-decode"}, "rules": {"min_gain_pct": 2}}},
        )

    def read(self, relative: str) -> dict:
        return json.loads((self.store.path / relative).read_text(encoding="utf-8"))

    def publish(self) -> None:
        publish_indexes(self.store, self.store.state(), self.store.events())

    def measurement(self, name: str, throughput: int) -> dict:
        samples = [{"request_id": "prefill", "output_tokens": throughput}]
        raw = self.store.put_evidence(f"evidence/raw/{name}.json", samples, producer="runner")
        return {
            "revision": {"id": name},
            "attempt_id": f"attempt-{name}",
            "status": "ok",
            "acceptance": "pass",
            "evidence": {"revision_measurement": raw},
            "measurement": {
                "boundary": "service_end_to_end",
                "revision": name,
                "summary": {"throughput": throughput, "ttft_ms": 25, "tpot_ms": 7},
                "samples": samples,
            },
        }

    def test_index_views_reference_originals_and_rebuild_idempotently(self) -> None:
        base = self.measurement("base", 100)
        trial = self.measurement("candidate", 110)
        reference = self.store.put_evidence("evidence/raw/reference.json", [0.0], producer="verifier")
        trace = self.store.put_evidence("evidence/raw/rank0.json", {"rank": 0}, producer="profiler")
        recipe = {"revision": "candidate", "requires_revalidation": True}
        decision = {
            **trial,
            "candidate_id": "candidate",
            "task_id": "task-01",
            "proposal_id": "P01",
            "parent": "base",
            "version_action": "KEEP",
            "parent_measurement": base["measurement"],
            "comparison": {"over_parent": {"throughput_pct": 10}},
            "evidence": {
                "candidate_measurement": trial["evidence"]["revision_measurement"],
                "parent_measurement": base["evidence"]["revision_measurement"],
                "reference": reference,
            },
        }
        self.store.commit([
            ("baseline_recorded", {"result": base, "revision": {"id": "base"}}),
            ("profile_recorded", {"attempt_id": "capture-01", "revision": "base", "status": "partial",
                                  "ranks": [0], "missing_ranks": [1], "evidence": {"profile_rank_0": trace}}),
            ("candidate_decision", {"decision": decision, "promote": {"id": "candidate"}}),
            ("sweep_recorded", {
                "cycle": 1, "status": "partial", "revision": "candidate",
                "pairs": [
                    {"concurrency": 1, "original": base, "candidate": trial, "comparison": {"throughput_pct": 10}},
                    {"concurrency": 4, "original": base, "candidate": None},
                ],
            }),
            ("sweep_recorded", {"cycle": 2, "status": "skipped", "reason": "no_validated_change_since_sweep"}),
            ("recipe_recorded", recipe),
        ])
        original_bytes = {
            path: path.read_bytes() for path in self.store.path.rglob("*.json") if path.is_file()
        }

        self.publish()
        summary = self.read("baseline/prefill-decode/summary.json")
        self.assertEqual(summary["baseline"]["revision"], "base")
        self.assertNotIn("samples", summary["baseline"]["measurement"])
        self.assertEqual(summary["mode"], "simulated")
        self.assertEqual(summary["event_seq"], self.store.state()["last_seq"])
        self.assertEqual(summary["manifest_fingerprint"], self.store.manifest["fingerprint"])

        profile = self.read("profile/capture-01/summary.json")
        self.assertEqual(profile["rank_evidence"], {"0": trace})
        self.assertEqual(profile["missing_ranks"], [1])
        bundle = self.read("rounds/candidate/evidence_bundle.json")
        self.assertEqual(bundle["parent"]["measurement_ref"], base["evidence"]["revision_measurement"])
        self.assertEqual(bundle["candidate"]["measurement"]["summary"]["throughput"], 110)
        self.assertEqual(bundle["correctness_reference"], reference)
        self.assertEqual(bundle["rules"]["min_gain_pct"], 2)
        self.assertEqual(bundle["version_action"], "KEEP")
        self.assertEqual(bundle["proposal_id"], "P01")

        sweep = self.read("sweep/cycle-1.json")
        self.assertIsNone(sweep["pairs"][-1]["candidate"])
        self.assertEqual(self.read("sweep/cycle-2.json")["status"], "skipped")
        with (self.store.path / "sweep/cycle-1.csv").open(encoding="utf-8", newline="") as sample:
            rows = list(csv.DictReader(sample))
        self.assertEqual([(row["concurrency"], row["candidate_throughput"]) for row in rows], [("1", "110")])
        self.assertEqual(json.loads((self.store.path / "recipes.jsonl").read_text(encoding="utf-8")), recipe)

        index = (self.store.path / "evidence/index.jsonl").read_bytes()
        evidence = [json.loads(line) for line in index.splitlines()]
        self.assertEqual(len(evidence), 4)
        self.assertTrue(all(self.store.verify_evidence(item) for item in evidence))
        self.publish()
        self.assertEqual((self.store.path / "evidence/index.jsonl").read_bytes(), index)
        self.assertTrue(all(path.read_bytes() == contents for path, contents in original_bytes.items()))

    def test_unsafe_derived_candidate_identifier_is_rejected(self) -> None:
        self.store.emit(
            "candidate_decision", {"decision": {"candidate_id": "../escape", "version_action": "HOLD"}}
        )
        with self.assertRaises(ContractError):
            self.publish()
        self.assertFalse((self.store.path / "escape").exists())
