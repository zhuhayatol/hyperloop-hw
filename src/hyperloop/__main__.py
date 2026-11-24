"""Run and inspect local Hyperloop Sessions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .actions.knowledge import load_verified_recipe
from .artifacts import publish_indexes
from .contracts import ContractError, read_json, safe_id
from .report import write_report
from .session import SessionBusy, SessionStore, project


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
    scenario = read_json(args.scenario)
    if "retrieved_recipe" in scenario:
        raise ContractError("retrieved_recipe must be supplied using --recipe")
    if args.recipe is not None:
        scenario["retrieved_recipe"] = load_verified_recipe(args.recipe)
    coordinator = Coordinator.create(
        args.session_root,
        args.session_id,
        task,
        read_json(args.target),
        read_json(args.reference),
        scenario,
    )
    return _status(coordinator.store, coordinator.run(max_steps=args.max_steps))


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
        **{key: result[key] for key in fields if key in result},
        "evidence": {
            name: evidence[name]
            for name in ("decision", "paired_comparison", "candidate_measurement", "parent_measurement")
            if name in evidence
        },
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
    run.add_argument("--recipe", type=Path, help="use a completed Session's verified final recipe as guidance")
    run.add_argument("--max-steps", type=positive_steps)
    run.set_defaults(handler=_run)

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
    except (ValueError, OSError, SessionBusy) as error:
        print(f"hyperloop: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
