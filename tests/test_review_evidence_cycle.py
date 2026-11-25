"""Use simulated timing to test who receives a measured re-review and rejection."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hyperloop.actions.review_evidence import ROPE_ACTION, ROPE_BOUNDARY
from hyperloop.backends import SimulatedBackend
from hyperloop.contracts import read_json
from hyperloop.coordinator import Coordinator

from test_deepseek_agents import KEY, completion
from test_pipeline import load_fixture


class _RoPEBackend(SimulatedBackend):
    def probe(self):
        result = super().probe()
        result["capabilities"].append(ROPE_ACTION)
        return result

    def run_rope_layout_probe(self, **kwargs):
        return {
            "status": "ok", "mode": "simulated", "timing_boundary": ROPE_BOUNDARY,
            "cases": [{"phase": "prefill", "tokens": 512, "layout_source": "simulated", "outputs_match": True,
                       "layouts": {"split_q": {"stride": [1152, 1]}, "split_k": {"stride": [1152, 1]}},
                       "samples_us": {"split_rope": [30, 31, 29, 30], "contiguous_rope": [18, 19, 17, 18],
                                      "contiguous_copy_only": [6, 7, 6, 7], "contiguous_copy_then_rope": [24, 25, 23, 25]}}],
        }


class ReviewEvidenceCycleTests(TestCase):
    def test_same_proposal_is_re_reviewed_then_integrated_or_replanned(self):
        root = Path(__file__).resolve().parent
        fixture = read_json(root / "fixtures/h18_specialist_responses.json")
        old_reply = fixture["contiguous"]
        offset = old_reply["offset"]
        original = old_reply["response_content"]
        proposal, = json.loads(original[:offset] + "}" + original[offset:])["intents"]
        candidate_id = proposal["params"]["candidate"]["id"]
        excerpt = fixture["reviewed_excerpt"]
        for re_review in ("approve", "needs_review"):
            with self.subTest(re_review=re_review), TemporaryDirectory() as temporary:
                task = load_fixture("configs/task.simulated.json")
                task["allowed_parameters"] = []
                task["goals"] = {"throughput_gain_pct": 15, "ttft_reduction_pct": 15}
                task["rules"]["tolerance"] = 0
                task["settings"].update(max_cycles=1, max_turns_per_cycle=1, kernel_backend="disabled",
                                        sweep_enabled=False, record_model_forward=False)
                scenario = load_fixture("fixtures/scenario.simulated.json")
                scenario["agent_provider"] = {"provider": "deepseek"}
                scenario["agent_source_excerpts"] = [excerpt]
                scenario["revisions"][candidate_id] = deepcopy(scenario["revisions"]["R02"])
                target = load_fixture("configs/target.910b3.simulated.json")
                target["sglang"] = {"allowed_source_paths": [excerpt["path"]]}
                campaign = Coordinator.create(Path(temporary), "review-evidence", task, target,
                    load_fixture("fixtures/reference.simulated.json"), scenario,
                    backend_factory=lambda actual, config: _RoPEBackend(actual, config["revisions"]))
                reviewed = []
                replanned = []

                def provider(request, *, timeout):
                    sent = json.loads(json.loads(request.data)["messages"][1]["content"])
                    if sent["role"] == "orchestration":
                        if sent.get("review_feedback"):
                            replanned.extend(sent["review_feedback"])
                            return completion([])
                        return completion([{"kind": "delegate", "action": "specialist",
                                            "params": {"mode": "author", "focus": "Investigate Prefill RoPE layout"}}])
                    if sent["role"] == "specialist":
                        return completion([proposal])
                    row, = sent["pending_proposals"]
                    reviewed.append(row)
                    evidence = row.get("review_evidence")
                    if evidence:
                        self.assertEqual(evidence["summary"]["cases"][0]["copy_us"], 6.5)
                        self.assertEqual(evidence["summary"]["cases"][0]["net_gain_us"], 6)
                    return completion([{"kind": "review_verdict", "proposal_id": row["proposal_id"],
                                        "verdict": re_review if evidence else "needs_review",
                                        "reason": "Check separately what the layout and copy cost."}])

                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", provider):
                    state = campaign.run()
                self.assertEqual(state["phase"], "DONE")
                first, second = reviewed
                self.assertEqual(first["proposal_id"], second["proposal_id"])
                self.assertEqual(first["candidates"], second["candidates"])
                task_actions = [item["action"] for item in state["tasks"].values()]
                self.assertEqual(task_actions.count(ROPE_ACTION), 1)
                events = campaign.store.events()
                evidence_recorded, = [e["payload"] for e in events if e["type"] == "review_evidence_recorded"]
                self.assertTrue(campaign.store.verify_evidence(evidence_recorded["evidence"]["rope_layout"]))
                if re_review == "approve":
                    self.assertEqual(task_actions.count("integrate_patch"), 1)
                    decision, = state["decisions"]
                    self.assertTrue(decision["comparison"]["over_parent"])
                    self.assertFalse(replanned)
                else:
                    self.assertNotIn("integrate_patch", task_actions)
                    self.assertEqual(len(replanned), 1)
                    self.assertEqual(replanned[0]["review_evidence"], second["review_evidence"])
                    self.assertEqual(sum(e["type"] == "review_replan" for e in events), 1)
