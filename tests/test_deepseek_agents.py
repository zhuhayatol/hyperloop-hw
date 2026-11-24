"""Check live Agent boundaries with an in-process HTTP transport."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError

from hyperloop.__main__ import main
from hyperloop.agents import AgentInvocationError, build_agent_backend
from hyperloop.contracts import ContractError, read_json
from hyperloop.coordinator import Coordinator

PROJECT = Path(__file__).resolve().parents[1]
KEY = "sk-unit-test-do-not-record"
REASONING = "private reasoning not for evidence"
SOURCE = "python/sglang/srt/attention.py"
FRAGMENT = "    return attention(query, key, value)\n"


class Response(io.BytesIO):
    status = 200


def completion(intents, response_id="chatcmpl-unit-test", finish_reason="stop"):
    return Response(json.dumps({
        "id": response_id,
        "choices": [{"finish_reason": finish_reason, "message": {
            "role": "assistant", "content": json.dumps({"intents": intents}), "reasoning_content": REASONING,
        }}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 13, "total_tokens": 34,
                  "provider_debug": REASONING},
    }).encode())


def explore(parameter="chunked_prefill_size"):
    return {
        "kind": "delegate", "action": "explore", "params": {"candidates": [{
            "id": "R01", "parameters": {parameter: 4096}, "reason": "Test <agent-preview> chunk sizing.",
            "expected_metric": "throughput", "risk": "TTFT regression", "validation": "Check outputs and repeat timing",
        }]},
    }


class DeepSeekAgentTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = read_json(PROJECT / "configs/task.simulated.json")
        self.task["settings"].update(max_cycles=1, kernel_backend="disabled", sweep_enabled=False)
        self.target = read_json(PROJECT / "configs/target.910b3.simulated.json")
        self.target["sglang"] = {
            "allowed_source_paths": [SOURCE], "model_path": "/private/model/directory",
            "server_args": ["--chunked-prefill-size", "2048", "--unrelated", "/private/argv/path"],
        }
        self.scenario = read_json(PROJECT / "fixtures/scenario.simulated.json")
        self.scenario["agent_provider"] = {"provider": "deepseek"}
        self.scenario["agent_source_excerpts"] = [{"id": "attention-1", "path": SOURCE, "before": FRAGMENT}]
        self.reference = read_json(PROJECT / "fixtures/reference.simulated.json")

    def coordinator(self):
        return Coordinator.create(self.root, "deepseek-test", self.task, self.target, self.reference, self.scenario)

    def calls(self, coordinator):
        return [event["payload"] for event in coordinator.store.events() if event["type"] == "agent_provider_call"]

    def evidence(self, coordinator, call):
        self.assertTrue(coordinator.store.verify_evidence(call["evidence"]))
        return read_json(coordinator.store.path / call["evidence"]["path"])

    def test_three_roles_keep_safe_inputs_outputs_and_local_outcomes(self):
        chosen = {"kind": "delegate", "action": "specialist", "params": {"mode": "author", "focus": "attention"}}
        specialist = {"kind": "propose_action", "action": "integrate_patch", "params": {"candidate": {
            "id": "R02", "source_excerpt_id": "attention-1", "replacement": "    return fused(query, key, value)\n",
            "reason": "Test a narrower attention path.", "expected_metric": "throughput", "risk": "wrong outputs",
            "validation": "Run reference equality before timing.",
        }}}
        critic = {"kind": "review_verdict", "proposal_id": "proposal-0001", "verdict": "approve",
                  "reason": "The edit is bounded and has independent checks."}
        orchestration = iter([[explore()], [chosen], [{"kind": "escalate_strategy_change", "next_action_hint": "skip_to_sweep"}]])
        observed = []

        def transport(request, *, timeout):
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
            self.assertGreater(timeout, 0)
            body = json.loads(request.data)
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertNotIn(KEY, request.data.decode())
            self.assertNotIn("/private/", request.data.decode())
            data = json.loads(body["messages"][1]["content"])
            observed.append(data)
            self.assertNotIn("input_tokens", data["workload"]["prefill"])
            role = data["role"]
            if role == "orchestration":
                self.assertNotIn("source_excerpts", data)
                intents = next(orchestration)
            elif role == "specialist":
                self.assertEqual(data["source_excerpts"][0]["before"], FRAGMENT)
                self.assertEqual(data["assignment"], {"mode": "author", "focus": "attention"})
                intents = [specialist]
            else:
                self.assertEqual(data["pending_proposals"][0]["proposal_id"], "proposal-0001")
                self.assertEqual(data["pending_proposals"][0]["candidates"][0]["patch_path"], SOURCE)
                intents = [critic]
            return completion(intents, f"chatcmpl-{role}-{len(observed)}")

        coordinator = self.coordinator()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", transport):
            final = coordinator.run()
        calls = self.calls(coordinator)
        self.assertEqual([call["role"] for call in calls], ["orchestration", "orchestration", "specialist", "critic", "orchestration"])
        self.assertEqual(final["phase"], "DONE")
        self.assertEqual([(row["candidate_id"], row["version_action"]) for row in final["decisions"]], [("R01", "KEEP"), ("R02", "KEEP")])
        self.assertEqual(coordinator.store.manifest["scenario"]["agent_provider"], {"provider": "deepseek", "model": "deepseek-flash"})
        self.assertEqual(observed[0]["current"]["parameters"], {"chunked_prefill_size": 2048})
        patch_output = self.evidence(coordinator, calls[2])["normalized_intents"][0]["params"]["candidate"]
        self.assertEqual(patch_output["source_edit"], {"before": FRAGMENT, "after": specialist["params"]["candidate"]["replacement"]})
        for call, sent in zip(calls, observed):
            self.assertNotIn("request_context", call)
            self.assertTrue(call["response_id"].startswith("chatcmpl-"))
            self.assertEqual(call["usage"], {"prompt_tokens": 21, "completion_tokens": 13, "total_tokens": 34})
            self.assertEqual(self.evidence(coordinator, call)["request_context"], sent)
        summary = read_json(coordinator.store.path / "summary.json")
        self.assertEqual(summary["agent_calls"][3]["outcomes"][0]["candidate_id"], "R02")
        report = (coordinator.store.path / "report.html").read_text(encoding="utf-8")
        self.assertIn('id="agents"', report)
        self.assertIn(calls[2]["evidence"]["path"], report)
        self.assertIn("#candidate-outcome-2", report)
        self.assertIn("&lt;agent-preview&gt;", report)
        self.assertNotIn("<agent-preview>", report)
        self.assertNotIn(REASONING, report)
        self.assertNotIn(KEY, report)
        self.assertNotIn(KEY, json.dumps(coordinator.store.manifest))

    def test_http_failure_is_preserved_and_resume_retries_live_turn(self):
        self.task["settings"]["max_turns_per_cycle"] = 1
        coordinator = self.coordinator()
        failure = HTTPError("https://api.deepseek.com/chat/completions", 429, KEY, None, io.BytesIO(KEY.encode()))
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", side_effect=failure):
            with self.assertRaisesRegex(AgentInvocationError, "http_error"):
                coordinator.run()
        fail, = self.calls(coordinator)
        receipt = self.evidence(coordinator, fail)
        self.assertEqual((receipt["status"], receipt["http_status"], receipt["normalized_intents"]), ("failed", 429, []))
        self.assertNotIn(KEY, json.dumps(receipt))
        self.assertFalse(coordinator.store.state()["decisions"])
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", return_value=completion([])):
            final = Coordinator.open(coordinator.store.path).run()
        self.assertEqual(final["phase"], "DONE")
        self.assertEqual([call["status"] for call in self.calls(coordinator)], ["failed", "ok"])
        self.assertFalse(final["decisions"])

    def test_role_schema_and_router_both_gate_agent_output(self):
        self.task["settings"]["max_turns_per_cycle"] = 1
        coordinator = self.coordinator()
        forbidden = {"kind": "review_verdict", "proposal_id": "proposal-1", "verdict": "approve", "reason": "other role"}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", return_value=completion([forbidden])):
            with self.assertRaisesRegex(AgentInvocationError, "invalid_structured_response"):
                coordinator.run()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open", return_value=completion([explore("private_option")])):
            self.assertFalse(Coordinator.open(coordinator.store.path).run()["decisions"])
        denied = [event["payload"] for event in coordinator.store.events() if event["type"] == "policy_denied"]
        self.assertEqual(denied[0]["code"], "unsupported_parameter")

    def test_credential_and_source_scope_guards(self):
        backend = build_agent_backend(self.scenario, self.task, self.target)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), patch("hyperloop.agents.deepseek._HTTP.open") as transport:
            self.assertEqual(backend.run("critic", {"pending_proposals": {}}), [])
            with self.assertRaisesRegex(AgentInvocationError, "missing_or_invalid_api_key"):
                backend.run("orchestration", {})
            transport.assert_not_called()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": KEY}), patch("hyperloop.agents.deepseek._HTTP.open") as transport:
            self.scenario["agent_source_excerpts"][0]["before"] = f"x = '{KEY}'\n"
            with self.assertRaisesRegex(AgentInvocationError, "credential_detected_in_context") as error:
                build_agent_backend(self.scenario, self.task, self.target).run("specialist", {})
            self.assertNotIn("request_context", error.exception.receipt)
            transport.assert_not_called()
        self.scenario["agent_source_excerpts"][0]["path"] = "sglang/other.py"
        with self.assertRaisesRegex(ContractError, "approved Python path"):
            build_agent_backend(self.scenario, self.task, self.target)

    def test_cli_explicit_provider_is_frozen_without_early_network_call(self):
        with patch("hyperloop.agents.deepseek._HTTP.open") as transport:
            args = ["run", "--task", str(PROJECT / "configs/task.simulated.json"), "--target",
                    str(PROJECT / "configs/target.910b3.simulated.json"), "--reference",
                    str(PROJECT / "fixtures/reference.simulated.json"), "--scenario",
                    str(PROJECT / "fixtures/scenario.simulated.json"), "--session-root", str(self.root),
                    "--session-id", "provider-cli", "--agent-provider", "deepseek", "--max-steps", "1"]
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(args), 0)
                self.assertEqual(json.loads(stdout.getvalue())["agent_provider"]["model"], "deepseek-flash")
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(["status", str(self.root / "simulated/provider-cli")]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["agent_provider"]["provider"], "deepseek")
            transport.assert_not_called()
