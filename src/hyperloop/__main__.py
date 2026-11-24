"""Run and inspect local Hyperloop Sessions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .actions.knowledge import load_verified_recipe
from .agents import provider_config
from .artifacts import publish_indexes
from .backends import BackendError
from .contracts import ContractError, digest, read_json, safe_id, validate_reference, validate_target, validate_task
from .report import write_report
from .session import SessionBusy, SessionStore, atomic_write, project


def positive_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max steps must be a positive integer") from error
    if steps < 1:
        raise argparse.ArgumentTypeError("max steps must be a positive integer")
    return steps


def _status(store: SessionStore, state: dict[str, Any]) -> dict[str, Any]:
    current = state.get("current_revision") or {}
    task_counts = Counter(task.get("status", "unknown") for task in state.get("tasks", {}).values())
    decisions = [
        {
            "candidate_id": decision.get("candidate_id"),
            "parent": decision.get("parent"),
            "version_action": decision.get("version_action"),
            "reason": decision.get("reason"),
        }
        for decision in state.get("decisions", [])
    ]
    return {
        "session": str(store.path),
        "mode": store.manifest["mode"],
        "phase": state["phase"],
        "cycle": state["cycle"],
        "last_seq": state["last_seq"],
        "accepted_revision": current.get("id"),
        "agent_provider": store.manifest.get("scenario", {}).get("agent_provider", {"provider": "replay"}),
        "stop_reason": state.get("stop_reason"),
        "task_counts": dict(task_counts),
        "decisions": decisions,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from .coordinator import Coordinator

    task = read_json(args.task)
    if args.kernel_backend is not None:
        settings = task.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("task.settings must be an object")
        settings["kernel_backend"] = args.kernel_backend
    target = read_json(args.target)
    reference = read_json(args.reference)
    scenario = read_json(args.scenario)
    if any(key in scenario for key in ("agent_provider", "agent_source_excerpts")):
        raise ContractError("Agent provider and source excerpts must be selected with run options")
    if args.agent_provider != "replay" or args.agent_model is not None:
        scenario["agent_provider"] = provider_config(args.agent_provider, args.agent_model)
    if args.agent_source_excerpts is not None:
        if args.agent_provider != "deepseek":
            raise ContractError("--agent-source-excerpts requires --agent-provider deepseek")
        source = read_json(args.agent_source_excerpts)
        if source.keys() != {"excerpts"}:
            raise ContractError("Agent source excerpt file must contain exactly one excerpts field")
        scenario["agent_source_excerpts"] = source["excerpts"]
    if "retrieved_recipe" in scenario:
        raise ContractError("retrieved_recipe must be supplied using --recipe")
    if args.recipe is not None:
        scenario["retrieved_recipe"] = load_verified_recipe(args.recipe)
    if "reference_capture" in scenario:
        raise ContractError("reference_capture must be supplied using --reference-receipt")
    if args.reference_receipt is not None:
        capture = read_json(args.reference_receipt)
        validate_task(task)
        identity = _reference_identity(task, target, reference)
        if capture.get("status") != "ok" or any(capture.get(key) != value for key, value in identity.items()):
            raise ContractError("reference capture does not match the task, target and reference")
        scenario["reference_capture"] = capture
    coordinator = Coordinator.create(
        args.session_root,
        args.session_id,
        task,
        target,
        reference,
        scenario,
    )
    return _status(coordinator.store, coordinator.run(max_steps=args.max_steps))


def _reference_identity(task: dict, target: dict, reference: dict) -> dict[str, str]:
    return {
        "initial_revision": task["initial_revision"],
        "model_sha256": digest(task["model"]),
        "workload_sha256": digest(task["workload"]),
        "target_sha256": digest(target),
        "reference_sha256": digest(reference),
    }


def _capture_reference(args: argparse.Namespace) -> dict[str, Any]:
    from .actions.candidates import original_revision, stack_identity
    from .backends.factory import build_runtime, checked_probe
    from .evaluation import Evaluator

    task = validate_task(read_json(args.task))
    target = validate_target(read_json(args.target))
    if target["mode"] != "real":
        raise ContractError("reference capture requires a real target")
    revision = original_revision(task["initial_revision"])
    backend = build_runtime(target, {}, revision["id"], None)
    probe = checked_probe(backend, target, mode="real")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend.bind_revision(revision, expected_stack=stack_identity(revision))
    capture = Evaluator(backend, output).capture_reference(
        revision["id"], task["workload"], expected_stack=stack_identity(revision),
    )
    receipt = {key: capture[key] for key in ("status", "reason", "attempt_id", "evidence")}
    reference_file = output / "reference.json"
    receipt_file = output / "capture.json"
    if capture["status"] == "ok" and (
        capture["runtime"]["environment_fingerprint"] == probe["environment_fingerprint"]
    ):
        reference = validate_reference(capture["reference"], task)
        receipt.update(_reference_identity(task, target, reference))
        receipt.update(environment_fingerprint=probe["environment_fingerprint"], mode="real")
        atomic_write(reference_file, (json.dumps(reference, indent=2, allow_nan=False) + "\n").encode())
    elif capture["status"] == "ok":
        receipt.update(status="failed", reason="reference_environment_changed")
    atomic_write(receipt_file, (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    if receipt["status"] != "ok":
        raise ContractError(f"reference capture failed ({receipt['reason']}); see {receipt_file}")
    return {"mode": "real", "reference": str(reference_file), "receipt": str(receipt_file)}


def _resume(args: argparse.Namespace) -> dict[str, Any]:
    from .coordinator import Coordinator

    coordinator = Coordinator.open(args.session)
    return _status(coordinator.store, coordinator.run(max_steps=args.max_steps))


def _read_status(args: argparse.Namespace) -> dict[str, Any]:
    store = SessionStore(args.session)
    return _status(store, store.state())


def _report(args: argparse.Namespace) -> dict[str, Any]:
    store = SessionStore(args.session)
    with store.execution_lease():
        events = store.events()
        state = project(events)
        publish_indexes(store, state, events)
        write_report(store.path, store.manifest, state, events)
    return {
        **_status(store, state),
        "artifacts": {name: str(store.path / name) for name in ("summary.json", "report.md", "report.html")},
    }


def _reproduce(args: argparse.Namespace) -> dict[str, Any]:
    from .coordinator import Coordinator

    candidate_id = safe_id(args.candidate_id, "candidate_id")
    coordinator = Coordinator.open(args.session)
    result = coordinator.reproduce(candidate_id)
    fields = (
        "mode", "attempt_id", "candidate_id", "parent", "status", "evidence_state",
        "acceptance", "recorded_action", "version_action", "reason", "comparison",
    )
    evidence = result.get("evidence", {})
    return {
        "session": str(coordinator.store.path),
        "validation_report": str(coordinator.store.path / "report.html") + "#validation",
        **{key: result[key] for key in fields if key in result},
        "evidence": {
            name: evidence[name]
            for name in ("decision", "paired_comparison", "candidate_measurement", "parent_measurement", "validation_flow")
            if name in evidence
        },
        "reproduction_receipt": result.get("reproduction_receipt"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperloop", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="freeze JSON inputs and start a Session")
    for name in ("task", "target", "reference", "scenario"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--session-root", type=Path, default=Path("sessions"))
    run.add_argument("--session-id", required=True)
    run.add_argument("--kernel-backend", choices=("forge", "geak", "disabled"))
    run.add_argument("--agent-provider", choices=("replay", "deepseek"), default="replay",
                     help="choose fixture replay or the DeepSeek API")
    run.add_argument("--agent-model", help="DeepSeek model ID (default deepseek-flash)")
    run.add_argument("--agent-source-excerpts", type=Path,
                     help="reviewed JSON source blocks to share with DeepSeek for patch proposals")
    run.add_argument("--recipe", type=Path, help="use a completed Session's verified final recipe as guidance")
    run.add_argument("--reference-receipt", type=Path, help="check the separately captured reference provenance")
    run.add_argument("--max-steps", type=positive_steps)
    run.set_defaults(handler=_run)

    reference = commands.add_parser("capture-reference", help="capture outputs from the unmodified real service")
    for name in ("task", "target", "output"):
        reference.add_argument(f"--{name}", type=Path, required=True)
    reference.set_defaults(handler=_capture_reference)

    resume = commands.add_parser("resume", help="continue a Session using its frozen inputs")
    resume.add_argument("session", type=Path)
    resume.add_argument("--max-steps", type=positive_steps)
    resume.set_defaults(handler=_resume)

    status = commands.add_parser("status", help="read current state from committed events")
    status.add_argument("session", type=Path)
    status.set_defaults(handler=_read_status)

    report = commands.add_parser("report", help="regenerate the offline Session reports")
    report.add_argument("session", type=Path)
    report.set_defaults(handler=_report)

    reproduce = commands.add_parser("reproduce", help="rerun an evaluated candidate with frozen inputs")
    reproduce.add_argument("session", type=Path)
    reproduce.add_argument("candidate_id")
    reproduce.set_defaults(handler=_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ValueError, OSError, SessionBusy, BackendError) as error:
        print(f"hyperloop: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
