"""Route accepted intents, record proposal reviews and publish task feedback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .dispatcher import Dispatcher, EventSink
from .policy import PolicyDenied, PolicyGate, action_name, intent_fields


class IntentRouter:
    def __init__(
        self, policy: PolicyGate, dispatcher: Dispatcher, emit: EventSink | None = None
    ) -> None:
        self.policy = policy
        self.dispatcher = dispatcher
        self._emit = emit
        self._proposals: dict[str, dict[str, Any]] = {}
        self._inboxes: dict[str, list[dict[str, Any]]] = {}
        self._proposal_counter = 0

    @property
    def pending_proposals(self) -> dict[str, dict[str, Any]]:
        return deepcopy(
            {key: proposal for key, proposal in self._proposals.items() if proposal["status"] == "pending"}
        )

    def inbox_for(self, role: str) -> list[dict[str, Any]]:
        return deepcopy(self._inboxes.get(role, []))

    def route(self, role: str, phase: str, intent: Mapping[str, Any]) -> dict[str, Any]:
        try:
            self.policy.authorize(role, phase, intent)
            kind, fields = intent_fields(intent)
            phase = phase.strip().upper()
            if kind == "delegate":
                return self._delegate(role, phase, fields)
            if kind == "propose_action":
                return self._propose(role, phase, fields)
            if kind == "review_verdict":
                return self._review(role, phase, fields)
            hint = fields.get("next_action_hint", fields.get("hint"))
            result = {"status": "recorded", "source": role, "phase": phase, "next_action_hint": hint}
            self._record("strategy_change", result)
            return result
        except PolicyDenied as exc:
            return self._deny(role, phase, exc)

    def register_internal(
        self,
        phase: str,
        action: str,
        params: Mapping[str, Any],
        *,
        source: str = "coordinator",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.policy.authorize_internal(phase, action, params)
        return self.dispatcher.enqueue(
            action, params, source=source, phase=phase.strip().upper(), idempotency_key=idempotency_key
        )

    def record_task_result(self, task: Mapping[str, Any]) -> None:
        """Deliver a completed work item to its requester and next orchestration turn."""
        payload = {key: task.get(key) for key in ("task_id", "proposal_id", "action", "status", "result", "cleanup")}
        source = str(task.get("source") or "coordinator")
        self._publish(source, "delegated_result", payload)
        if source != "orchestration":
            self._inboxes.setdefault("orchestration", []).append(
                deepcopy({"kind": "delegated_result", **payload})
            )

    def _delegate(self, role: str, phase: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        action = action_name(fields)
        params = fields.get("params", {})
        key = fields.get("idempotency_key")
        if key is None:
            encoded = json.dumps(dict(params), sort_keys=True, separators=(",", ":")).encode("utf-8")
            key = hashlib.sha256(encoded).hexdigest()
        elif not isinstance(key, str) or not key.strip():
            raise PolicyDenied("invalid_idempotency_key", "delegation key must be a nonempty string")
        try:
            return self.dispatcher.enqueue(
                action, params, source=role, phase=phase, idempotency_key=f"delegate:{role}:{phase}:{action}:{key}"
            )
        except ValueError as exc:
            raise PolicyDenied("duplicate_task_conflict", str(exc)) from exc

    def _propose(self, role: str, phase: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        proposal_id = fields.get("proposal_id")
        if proposal_id is not None and (not isinstance(proposal_id, str) or not proposal_id.strip()):
            raise PolicyDenied("invalid_proposal_id", "proposal ID must be a nonempty string")
        metadata = deepcopy(
            {key: value for key, value in fields.items() if key not in {"action", "action_name", "params", "proposal_id"}}
        )
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PolicyDenied("invalid_proposal_metadata", "proposal metadata must be finite JSON values") from exc
        proposal = {
            "source": role,
            "phase": phase,
            "action": action_name(fields),
            "params": deepcopy(dict(fields.get("params", {}))),
            "metadata": metadata,
        }
        if proposal_id is not None and proposal_id in self._proposals:
            previous = self._proposals[proposal_id]
            if any(previous[key] != value for key, value in proposal.items()):
                raise PolicyDenied("duplicate_proposal_conflict", f"{proposal_id} already identifies another proposal")
            return deepcopy(previous)
        if proposal_id is None:
            self._proposal_counter += 1
            proposal_id = f"proposal-{self._proposal_counter:04d}"
            while proposal_id in self._proposals:
                self._proposal_counter += 1
                proposal_id = f"proposal-{self._proposal_counter:04d}"
        proposal.update(proposal_id=proposal_id, status="pending")
        self._proposals[proposal_id] = proposal
        self._publish("critic", "proposal_registered", proposal)
        return deepcopy(proposal)

    def _review(self, role: str, phase: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        proposal_id = fields.get("proposal_id", fields.get("target_proposal_msg_id"))
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal["status"] != "pending":
            raise PolicyDenied("proposal_not_pending", f"{proposal_id} is not awaiting review")
        verdict = fields["verdict"]
        reason = fields.get("reason", fields.get("reasoning", ""))
        if not isinstance(reason, str):
            raise PolicyDenied("invalid_review", "review rationale must be text")
        review = {
            "proposal_id": proposal_id,
            "action": proposal["action"],
            "reviewer": role,
            "status": verdict,
            "verdict": verdict,
            "reason": reason,
        }
        if verdict in {"approve", "advise"}:
            if proposal["phase"] != phase:
                proposal["status"] = "expired"
                self._publish(proposal["source"], "proposal_expired", proposal)
                raise PolicyDenied("stale_proposal", f"{proposal_id} belongs to an earlier phase")
            self.policy.authorize(
                proposal["source"],
                phase,
                {"kind": "propose_action", "action": proposal["action"], "params": proposal["params"]},
            )
        proposal.update(status=verdict, review=review)
        self._publish(proposal["source"], "review_verdict", review)
        if verdict in {"approve", "advise"}:
            task = self.dispatcher.enqueue(
                proposal["action"],
                proposal["params"],
                source=proposal["source"],
                phase=phase,
                proposal_id=proposal_id,
                idempotency_key=f"proposal:{proposal_id}",
            )
            review["task_id"] = task["task_id"]
        return deepcopy(review)

    def _deny(self, role: str, phase: str, error: PolicyDenied) -> dict[str, Any]:
        result = {"status": "denied", "source": role, "phase": phase, "reason": str(error), "code": error.code}
        self._publish(role, "policy_denied", result)
        return result

    def _publish(self, role: str, kind: str, payload: Mapping[str, Any]) -> None:
        self._inboxes.setdefault(role, []).append(deepcopy({"kind": kind, **payload}))
        self._record(kind, payload)

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, deepcopy(dict(payload)))

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(
            {"proposals": self._proposals, "inboxes": self._inboxes, "proposal_counter": self._proposal_counter}
        )

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        restored = deepcopy(dict(snapshot))
        self._proposals = restored["proposals"]
        self._inboxes = restored["inboxes"]
        self._proposal_counter = restored["proposal_counter"]
