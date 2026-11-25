"""Authorize actions before the router creates proposals or tasks."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from .agents.backend import AGENT_ROLES

PHASE_ACTIONS = {
    "PRELUDE": frozenset({"target_analysis", "baseline", "profile", "roofline", "replay_warm_recipe"}),
    "ENABLEMENT": frozenset(
        {
            "target_analysis",
            "enablement",
            "baseline",
            "profile",
            "roofline",
            "specialist",
            "integrate_patch",
            "targeted_build",
        }
    ),
    "FRAMEWORK_AGENT": frozenset({
        "explore", "specialist", "integrate_patch", "profile", "roofline", "stack_revalidate", "rope_layout_probe",
    }),
    "KERNEL_AGENT": frozenset({"kernel_agent", "integrate", "profile", "roofline", "stack_revalidate"}),
    "SWEEP": frozenset({"conc_sweep", "stack_revalidate", "integrate"}),
    "CLOSE": frozenset({"report", "session_breakdown", "stack_revalidate", "profile", "roofline", "finalize_recipe"}),
}

INTERNAL_ACTIONS = frozenset(
    {
        "baseline",
        "replay_warm_recipe",
        "enablement",
        "targeted_build",
        "kernel_agent",
        "conc_sweep",
        "stack_revalidate",
        "report",
        "session_breakdown",
        "finalize_recipe",
        "rope_layout_probe",
    }
)
ENABLEMENT_INTERNAL = frozenset({"specialist", "integrate_patch"})
DIRECT_ACTIONS = frozenset({"explore", "specialist"})
REVIEW_VERDICTS = frozenset({"approve", "advise", "reject", "needs_review"})
_STRATEGY_HINTS = frozenset({"skip_to_kernel", "skip_to_sweep"})
_PATCH_KEYS = frozenset({"patch", "patch_path", "patches", "patch_paths", "target_path", "target_paths", "file_path", "files"})
_PARAMETER_KEYS = frozenset({"parameters", "settings", "config_changes"})
_PROTECTED_PATHS = frozenset({".git", "baseline", "baselines", "evidence", "fixtures", "reference", "rules", "sessions"})
_RESERVED_PARAMS = frozenset({"cwd", "shell", "command", "resource_overrides", "evidence_root", "workspace_root"})


class PolicyDenied(ValueError):
    """A refused intent with an observable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def intent_fields(intent: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Accept flat intents and the upstream type/payload envelope."""
    if not isinstance(intent, Mapping):
        raise PolicyDenied("invalid_intent", "intent must be an object")
    envelope = intent.get("payload", {})
    if not isinstance(envelope, Mapping):
        raise PolicyDenied("invalid_intent", "intent payload must be an object")
    kind = intent.get("kind", intent.get("type"))
    if not isinstance(kind, str) or not kind.strip():
        raise PolicyDenied("invalid_intent", "intent kind is required")
    fields = dict(envelope)
    fields.update({key: value for key, value in intent.items() if key not in {"kind", "type", "payload"}})
    return kind.strip().lower(), fields


def action_name(fields: Mapping[str, Any]) -> str:
    action = fields.get("action", fields.get("action_name"))
    if not isinstance(action, str) or not action.strip():
        raise PolicyDenied("invalid_action", "action name is required")
    return action.strip().lower()


class PolicyGate:
    def __init__(
        self,
        workspace: str | Path,
        allowed_parameters: Collection[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.allowed_parameters = None if allowed_parameters is None else frozenset(allowed_parameters)

    def authorize(self, role: str, phase: str, intent: Mapping[str, Any]) -> None:
        kind, fields = intent_fields(intent)
        phase = self._phase(phase)
        if role not in AGENT_ROLES and role != "coordinator":
            raise PolicyDenied("unknown_role", f"unknown role: {role!r}")
        if kind == "review_verdict":
            self._review(role, fields)
        elif kind == "escalate_strategy_change":
            self._escalate(role, phase, fields)
        elif kind in {"delegate", "propose_action"}:
            self._authorize_agent_action(role, phase, kind, fields)
        else:
            raise PolicyDenied("unknown_intent", f"unsupported intent: {kind}")

    def authorize_internal(self, phase: str, action: str, params: Mapping[str, Any]) -> None:
        self._check_phase(self._phase(phase), action)
        self._validate_params(params)

    @staticmethod
    def _phase(phase: str) -> str:
        if isinstance(phase, str) and phase.strip().upper() in PHASE_ACTIONS:
            return phase.strip().upper()
        raise PolicyDenied("unknown_phase", f"unknown phase: {phase!r}")

    @staticmethod
    def _check_phase(phase: str, action: str) -> None:
        if action not in PHASE_ACTIONS[phase]:
            raise PolicyDenied("phase_incompatible", f"{action} is unavailable in {phase}")

    @staticmethod
    def _review(role: str, fields: Mapping[str, Any]) -> None:
        if role != "critic":
            raise PolicyDenied("role_forbidden", "only Critic can review a proposal")
        proposal_id = fields.get("proposal_id", fields.get("target_proposal_msg_id"))
        if not isinstance(proposal_id, str) or not proposal_id.strip():
            raise PolicyDenied("invalid_review", "review must name its proposal")
        verdict = fields.get("verdict")
        if not isinstance(verdict, str) or verdict not in REVIEW_VERDICTS:
            raise PolicyDenied("invalid_review", "review must provide a supported verdict")

    @staticmethod
    def _escalate(role: str, phase: str, fields: Mapping[str, Any]) -> None:
        if role != "orchestration":
            raise PolicyDenied("role_forbidden", "only Orchestration can suggest a stage change")
        hint = fields.get("next_action_hint", fields.get("hint"))
        if not isinstance(hint, str) or hint not in _STRATEGY_HINTS:
            raise PolicyDenied("invalid_hint", "stage change must specify a supported hint")
        if phase != "FRAMEWORK_AGENT":
            raise PolicyDenied("phase_incompatible", f"stage hints are unavailable in {phase}")

    def _authorize_agent_action(self, role: str, phase: str, kind: str, fields: Mapping[str, Any]) -> None:
        action = action_name(fields)
        if action in INTERNAL_ACTIONS or (
            phase == "ENABLEMENT" and action in ENABLEMENT_INTERNAL and role != "coordinator"
        ):
            raise PolicyDenied("coordinator_owned", f"{action} is reserved for internal dispatch")
        self._check_phase(phase, action)
        if kind == "delegate" and (role != "orchestration" or action not in DIRECT_ACTIONS):
            raise PolicyDenied("review_required", f"{role} cannot directly delegate {action}")
        if kind == "propose_action" and (
            role == "critic" or (role == "specialist" and action not in {"explore", "integrate_patch"})
        ):
            raise PolicyDenied("role_forbidden", f"{role} cannot propose {action}")
        self._validate_params(fields.get("params", {}))

    def _validate_params(self, params: Mapping[str, Any]) -> None:
        if not isinstance(params, Mapping) or any(not isinstance(key, str) for key in params):
            raise PolicyDenied("invalid_parameters", "action params must be an object with string keys")
        try:
            json.dumps(dict(params), allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PolicyDenied("invalid_parameters", "action params must be finite JSON values") from exc
        self._validate_values(params)

    def _validate_values(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            if key in _RESERVED_PARAMS:
                raise PolicyDenied("reserved_parameter", f"action cannot override {key}")
            if key in _PATCH_KEYS:
                self._validate_paths(value)
            elif key in _PARAMETER_KEYS:
                self._validate_parameter_names(value)
            elif key == "estimated_seconds":
                self._validate_duration(value)
            elif isinstance(value, Mapping):
                self._validate_values(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        self._validate_values(item)

    def _validate_parameter_names(self, values: Any) -> None:
        if not isinstance(values, Mapping) or any(not isinstance(name, str) or not name for name in values):
            raise PolicyDenied("invalid_parameters", "parameter changes must map names to values")
        if self.allowed_parameters is not None:
            invalid = set(values) - self.allowed_parameters
            if invalid:
                raise PolicyDenied("unsupported_parameter", f"parameter is not allowed: {min(invalid)}")
        self._validate_values(values)

    @staticmethod
    def _validate_duration(value: Any) -> None:
        if type(value) not in {int, float} or value < 0 or (type(value) is float and not math.isfinite(value)):
            raise PolicyDenied("invalid_parameters", "estimated_seconds must be finite and nonnegative")

    def _validate_paths(self, value: Any) -> None:
        if isinstance(value, str):
            self._validate_path(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._validate_paths(item)
            return
        if isinstance(value, Mapping) and "path" in value:
            self._validate_path(value["path"])
            self._validate_values({key: item for key, item in value.items() if key != "path"})
            return
        raise PolicyDenied("patch_path_invalid", "patch paths must be relative workspace paths")

    def _validate_path(self, raw: Any) -> None:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or ":" in raw:
            raise PolicyDenied("patch_path_invalid", "patch path must be a relative workspace file")
        parts = raw.replace("\\", "/").split("/")
        if PureWindowsPath(raw).drive or any(
            not part or part.rstrip(" .") != part or part in {".", ".."} for part in parts
        ):
            raise PolicyDenied("patch_path_outside_workspace", f"patch path leaves the workspace: {raw}")
        try:
            resolved = self.workspace.joinpath(*parts).resolve().relative_to(self.workspace)
        except ValueError as exc:
            raise PolicyDenied("patch_path_outside_workspace", f"patch path leaves the workspace: {raw}") from exc
        except (OSError, RuntimeError) as exc:
            raise PolicyDenied("patch_path_invalid", f"patch path could not be resolved: {raw}") from exc
        if not resolved.parts or any(part.casefold() in _PROTECTED_PATHS for part in (*parts, *resolved.parts)):
            raise PolicyDenied("patch_path_protected", f"patch path is protected: {raw}")
