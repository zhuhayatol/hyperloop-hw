"""Check the evidence and repair acceptance gates for real startup diagnosis."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.actions.enablement import accept_startup_repair, observed_startup_failure
from hyperloop.contracts import json_bytes
from hyperloop.session import SessionStore


class EnablementTests(TestCase):
    def test_verified_failed_launch_exposes_only_its_supported_assertion(self):
        with TemporaryDirectory() as directory:
            store = SessionStore.create(Path(directory), "real", "enablement-test", {
                "task": {"allowed_parameters": ["chunked_prefill_size"]},
                "target": {"mode": "real", "provider": "sglang", "sglang": {
                    "server_args": ["--chunked-prefill-size", "513", "--page-size=128"],
                }},
            })
            workspace = store.path / "workspaces" / "task-0001" / "revision"
            workspace.mkdir(parents=True)
            log = workspace / "server.log"
            assertion = "AssertionError: chunked_prefill_size must be divisible by page_size"
            log.write_text(f"irrelevant private path\n{assertion}\n", encoding="utf-8")
            evidence = {}
            for name, row in {
                "revision_prepared": {"workspace": str(workspace)},
                "backend_failure": {"message": f"SGLang stopped before readiness. See {log}"},
                "revision_cleanup": {"workspace": str(workspace), "released": True},
                "validation_flow": {"steps": [{"operation": "launch", "state": "failed"}]},
            }.items():
                path = f"evaluation/task-0001/{name}.json"
                content = json_bytes(row)
                artifact = store.path / path
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(content)
                evidence[name] = {"path": path, "sha256": sha256(content).hexdigest()}
            baseline = {"task_id": "task-0001", "result": {
                "status": "failed", "reason": "backend_failure", "evidence": evidence,
            }}
            observed = observed_startup_failure(store, baseline)
            self.assertIsNotNone(observed)
            self.assertEqual(observed["chunked_prefill_size"], 513)
            self.assertEqual(observed["page_size"], 128)
            self.assertNotIn("irrelevant private path", str(observed))
            self.assertEqual(observed["startup_log"]["sha256"], sha256(log.read_bytes()).hexdigest())
            (store.path / evidence["revision_cleanup"]["path"]).write_bytes(b"tampered")
            self.assertIsNone(observed_startup_failure(store, baseline))

    def test_model_repair_must_be_the_minimal_supported_delta(self):
        failure = {"chunked_prefill_size": 513, "page_size": 128}

        def proposal(**parameters):
            return {"kind": "repair_startup", "parameters": parameters, "reason": "Align to the page size."}

        accepted = accept_startup_repair([proposal(chunked_prefill_size=512)], failure, "task-0002")
        self.assertEqual(accepted["parameters"], {"chunked_prefill_size": 512})
        self.assertEqual(accepted["source"], "live_specialist")
        for values in ({"chunked_prefill_size": 640}, {"chunked_prefill_size": 513},
                       {"chunked_prefill_size": 512, "page_size": 128}, {"chunked_prefill_size": True}):
            self.assertIsNone(accept_startup_repair([proposal(**values)], failure, "task-0002"))
