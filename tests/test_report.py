"""Session projections keep review decisions and raw evidence legible offline."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyperloop.report import write_report
from hyperloop.session import SessionStore


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = SessionStore.create(
            Path(temporary.name),
            "simulated",
            "report-test",
            {
                "task": {
                    "model": {"id": "model-a"},
                    "workload": {"id": "prefill-and-decode"},
                    "rules": {"min_gain_pct": 2},
                },
                "target": {"id": "simulated-tp2", "tp_size": 2},
                "contributions": [{"kind": "simulated", "owner": "team"}],
            },
        )
        self.root = self.session.path

    def build(self) -> dict:
        return write_report(self.root, self.session.manifest, self.session.state(), self.session.events())

    def test_replayed_report_links_measurements_and_keeps_axes_separate(self) -> None:
        raw = self.session.put_evidence("evidence/timing/rep 1# samples.json", {"tokens": 16}, producer="runner")
        trace = self.session.put_evidence("evidence/profile/rank0.json", {"rank": 0}, producer="profiler")
        original = {"summary": {"throughput": 100, "ttft_ms": 12, "tpot_ms": 3}}
        parent = {"summary": {"throughput": 110, "ttft_ms": 11, "tpot_ms": 2.9}}
        tested = {"summary": {"throughput": 112, "ttft_ms": 11, "tpot_ms": 2.8}}
        first = {
            "candidate_id": "first",
            "parent": "original",
            "evidence_state": "sufficient",
            "acceptance": "pass",
            "version_action": "KEEP",
        }
        decision = {
            "candidate_id": "next",
            "parent": "first",
            "evidence_state": "sufficient",
            "acceptance": "fail",
            "version_action": "REVERT",
            "reason": "收益低于预设门槛",
            "correctness": {"pass": True, "cases": [{"request_id": "P1", "pass": True}]},
            "measurement": tested,
            "parent_measurement": parent,
            "comparison": {
                "over_parent": {"throughput_pct": 1.81818, "ttft_pct": 0, "tpot_pct": -3.448},
                "over_original": {"throughput_pct": 12, "ttft_pct": -8.333, "tpot_pct": -6.667},
            },
            "evidence": {"timing": raw},
        }
        self.session.commit(
            [
                ("runtime_probed", {"rank_devices": {"0": "left", "1": "right"}}),
                (
                    "baseline_recorded",
                    {
                        "result": {"revision": "original", "measurement": original},
                        "revision": {"id": "original"},
                    },
                ),
                ("agent_turn", {"role": "Orchestration", "intent": "<script>alert(1)</script>"}),
                ("candidate_decision", {"decision": first, "promote": {"id": "first", "framework": ["first"]}}),
                ("candidate_decision", {"decision": decision}),
                ("profile_recorded", {"revision": "first", "ranks": [0], "missing_ranks": [1], "evidence": trace}),
                ("phase_changed", {"phase": "CLOSE", "cycle": 1}),
                ("stopped", {"reason": "budget_exhausted"}),
            ]
        )
        # The report's state comes from the event log even when the fast snapshot is stale.
        (self.root / "state.json").write_text("{}", encoding="utf-8")

        summary = self.build()
        html = (self.root / "report.html").read_text(encoding="utf-8")
        markdown = (self.root / "report.md").read_text(encoding="utf-8")
        self.assertEqual(summary, json.loads((self.root / "summary.json").read_text(encoding="utf-8")))
        self.assertEqual(summary["mode"], "simulated")
        self.assertEqual(summary["stop_reason"], "budget_exhausted")
        self.assertEqual(summary["current_revision"]["id"], "first")
        self.assertEqual(summary["decisions"][-1], decision)
        self.assertEqual(summary["scope"]["runtime"]["rank_devices"]["1"], "right")
        self.assertEqual(summary["state_event_seq"], len(summary["timeline"]))
        self.assertIn("evidence/timing/rep%201%23%20samples.json", html)
        self.assertIn("evidence/timing/rep%201%23%20samples.json", markdown)
        self.assertIn("evidence/profile/rank0.json", html)
        self.assertIn("缺少 rank 1", html)
        self.assertIn("服务正确性 · 通过", markdown)
        self.assertIn("充分 | 未达标 | REVERT", markdown)
        self.assertIn("吞吐 +1.82%，TTFT +0.00%", markdown)
        self.assertIn("吞吐 +12.00%", markdown)
        self.assertIn("候选 | 112 | 11 | 2.8", markdown)
        self.assertIn("原始基线 | 100 | 12 | 3", markdown)
        self.assertIn("Orchestration", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("https://", html)
        self.assertEqual(self.build(), summary)
        self.assertEqual((self.root / "report.html").read_text(encoding="utf-8"), html)

    def test_unavailable_and_conflicting_evidence_does_not_offer_unsafe_links(self) -> None:
        changed = self.session.put_evidence("evidence/timing.json", {"value": 1}, producer="runner")
        (self.root / "evidence" / "timing.json").write_text('{"value":2}', encoding="utf-8")
        self.session.emit(
            "candidate_decision",
            {
                "decision": {
                    "candidate_id": "hold",
                    "evidence_state": "missing",
                    "acceptance": "undetermined",
                    "version_action": "HOLD",
                    "evidence": {
                        "timing": changed,
                        "absent": {"path": "evidence/absent.json"},
                        "outside": {"path": "../manifest.json"},
                        "external": {"path": "https://outside.example/trace"},
                    },
                },
            },
        )
        summary = self.build()
        html = (self.root / "report.html").read_text(encoding="utf-8")
        markdown = (self.root / "report.md").read_text(encoding="utf-8")
        entries = {entry["path"]: entry for entry in summary["evidence"]}
        self.assertEqual(entries["evidence/timing.json"]["link_state"], "hash_mismatch")
        self.assertEqual(entries["evidence/absent.json"]["link_state"], "missing")
        self.assertEqual(entries["../manifest.json"]["link_state"], "unsafe_path")
        self.assertIsNone(entries["https://outside.example/trace"]["href"])
        self.assertNotIn('href="../manifest.json"', html)
        self.assertNotIn('href="https://outside.example/trace"', html)
        self.assertIn("hash_mismatch", html)
        self.assertIn("缺失 | 无法判定 | HOLD", markdown)
        self.assertEqual(summary["decisions"][0]["version_action"], "HOLD")
        self.assertIsNone(summary["current_revision"])
