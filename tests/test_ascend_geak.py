from copy import deepcopy
import unittest
from unittest.mock import Mock

from hyperloop.actions.candidates import original_revision
from hyperloop.kernel.ascend_geak import run_ascend_geak


class AscendGeakTests(unittest.TestCase):
    def setUp(self):
        self.parent = original_revision("accepted")
        self.parent["parameters"]["chunked_prefill_size"] = 512
        self.profile = {
            "status": "ok", "revision": "accepted", "point": "fixed",
            "report_ref": {"path": "evaluation/profile/profile_analysis.json", "sha256": "example"},
            "hotspots": [
                {"name": "Memcpy", "callsite": "elsewhere", "duration_us": 400},
                {"name": "FusedInferAttentionScore", "duration_us": 210, "callsite": "", "event_kind": "kernel"},
                {"name": "UnpadFlashAttention", "callsite": "sglang/srt/hardware_backend/npu/attention/ascend_backend.py:1479",
                 "duration_us": 99, "event_kind": "kernel", "phase": "prefill"},
            ],
        }

    @staticmethod
    def measure(revision, attempt):
        fia = revision["parameters"].get("ascend_use_fia", False)
        return {
            "mode": "real", "status": "ok", "reason": "evaluation_complete", "correctness": {"pass": True},
            "runtime": {"instance_id": attempt, "loaded_revision": revision["id"],
                        "stock_attention_launch_control": {"ASCEND_USE_FIA": str(int(fia)), "ASCEND_USE_FA": "0"}},
            "measurement": {"mode": "real", "revision": revision["id"], "boundary": "service_end_to_end",
                            "environment_fingerprint": "env", "workload_fingerprint": "load", "protocol_fingerprint": "rule",
                            "summary": {"throughput": 38 if fia else 40, "ttft_ms": 2, "tpot_ms": 1}},
            "evidence": {"revision_launch": {"path": f"evaluation/{attempt}/revision_launch.json"}},
        }

    def search(self, measure, fia_default=False):
        return run_ascend_geak(task_id="task-123", parent=self.parent, profile=self.profile,
                              fia_default=fia_default, measure=measure)

    def test_real_trial_proposes_a_slower_but_runnable_switch_for_independent_review(self):
        measure = Mock(side_effect=self.measure)
        before = deepcopy(self.parent)
        tool, proposal = self.search(measure)
        self.assertEqual(self.parent, before)
        self.assertEqual(tool["status"], "ok")
        self.assertEqual(tool["hotspot"]["name"], "UnpadFlashAttention")
        self.assertEqual(tool["comparison"]["throughput_pct"], -5)
        self.assertEqual(proposal["parameters"], {"ascend_use_fia": True})
        self.assertEqual(tool["service_candidate"]["configuration"], {"ascend_use_fia": True, "chunked_prefill_size": 512})
        self.assertEqual(tool["parent_trial"]["evidence"]["revision_launch"]["path"],
                         "evaluation/task-123-geak-parent/revision_launch.json")
        self.assertEqual(tool["candidate_trial"]["launch_control"]["ASCEND_USE_FIA"], "1")
        self.assertEqual([call.args[1] for call in measure.call_args_list], ["task-123-geak-parent", "task-123-geak-candidate"])

    def test_opposite_setting_uses_current_revision_instead_of_target_default(self):
        self.parent["parameters"]["ascend_use_fia"] = True
        tool, proposal = self.search(self.measure)
        self.assertEqual(tool["comparison"]["throughput_pct"], 100 * 2 / 38)
        self.assertEqual(proposal["parameters"], {"ascend_use_fia": False})

    def test_missing_usable_profile_or_switch_control_keeps_services_unstarted(self):
        measure = Mock()
        self.assertEqual(self.search(measure, fia_default=None)[0]["reason"], "ascend_fia_default_unconfigured")
        self.profile["revision"] = "older"
        self.assertEqual(self.search(measure)[0]["reason"], "no_usable_profile")
        self.profile["revision"] = "accepted"
        self.profile["hotspots"] = self.profile["hotspots"][:1]
        self.assertEqual(self.search(measure)[0]["reason"], "no_stock_attention_hotspot")
        measure.assert_not_called()

    def test_failed_cleanup_stops_at_parent_trial(self):
        measure = Mock(return_value={"status": "blocked", "reason": "cleanup_unconfirmed", "evidence": {}})
        tool, proposal = self.search(measure)
        self.assertIsNone(proposal)
        self.assertEqual(tool["reason"], "cleanup_unconfirmed")
        measure.assert_called_once()

    def test_unconfirmed_candidate_launch_is_blocked(self):
        def wrong_control(revision, attempt):
            run = self.measure(revision, attempt)
            run["runtime"]["stock_attention_launch_control"]["ASCEND_USE_FIA"] = "0"
            return run

        tool, proposal = self.search(wrong_control)
        self.assertIsNone(proposal)
        self.assertEqual(tool["reason"], "geak_service_identity_unconfirmed")
