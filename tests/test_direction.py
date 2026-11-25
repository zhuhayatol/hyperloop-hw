"""Direction changes require actual evidence from repeated attempts."""

import unittest

from hyperloop.agents.structured import project_context
from hyperloop.phases.direction import direction_change


def trial(candidate_id, chunk, action="REVERT"):
    return {
        "candidate_id": candidate_id,
        "kind": "parameters",
        "version_action": action,
        "comparison": {"over_parent": {"throughput_pct": 2}},
        "parent_revision": {"parameters": {"chunked_prefill_size": 512}},
        "revision": {"parameters": {"chunked_prefill_size": chunk}},
    }


class DirectionTests(unittest.TestCase):
    def test_distinct_values_on_the_same_parameter_do_not_reset_direction(self):
        state = {"cycle": 2, "decisions": [trial("chunk768", 768), trial("chunk640", 640)]}
        change = direction_change(state, ("explore", "integrate_patch", "forge"))
        self.assertEqual(change["mechanism"], ["chunked_prefill_size"])
        self.assertEqual(change["tested_candidates"], ["chunk768", "chunk640"])
        self.assertEqual(change["alternatives"], ["patch", "kernel"])
        context = project_context("orchestration", {"direction_change": change}, {}, {}, [])
        self.assertEqual(context["direction_change"]["alternatives"], ["patch", "kernel"])

    def test_accepted_or_unmeasured_attempts_do_not_count_as_repeated_failures(self):
        rejected = trial("chunk768", 768)
        accepted = trial("chunk640", 640, "KEEP")
        self.assertIsNone(direction_change({"decisions": [rejected, accepted]}, ("patch",)))
        rejected_unmeasured = {**trial("chunk896", 896), "comparison": None}
        self.assertIsNone(direction_change({"decisions": [rejected, rejected_unmeasured]}, ("patch",)))

    def test_only_supported_alternatives_are_suggested(self):
        state = {"decisions": [trial("chunk768", 768), trial("chunk640", 640)]}
        self.assertIsNone(direction_change(state, ("explore",)))
        self.assertEqual(direction_change(state, ("profile", "explore"))["alternatives"], ["profile"])


if __name__ == "__main__":
    unittest.main()
