"""Build and play back the Session's committed handoffs in event order."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from html import escape
import json
from typing import Any

from .observability import describe_event, status_label
from .validation_graph import read_native_delivery

Record = Mapping[str, Any]

_ROLES = {"orchestration": "Orchestration", "specialist": "Specialist", "critic": "Critic"}
_EXECUTORS = {
    "baseline": "Baseline Runner", "enablement": "Enablement", "profile": "Profiler",
    "explore": "Explorer", "specialist": "Specialist", "integrate_patch": "Integrate Patch",
    "kernel_agent": "Kernel Agent", "conc_sweep": "SWEEP", "stack_revalidate": "整栈复测",
    "rope_layout_probe": "RoPE 布局探测",
}
_RETURNS = {
    "baseline_recorded": ("Evaluator", "原始基线"), "baseline_failure": ("Evaluator", "基线失败"),
    "profile_recorded": ("Profiler", "Profile 回执"), "sweep_recorded": ("SWEEP", "扫描回执"),
    "stack_validated": ("整栈复测", "整栈复测回执"), "enablement_suggestion": ("Specialist", "修复建议"),
}
_SAVES = {
    "runtime_probed": "设备和运行环境", "phase_changed": "阶段更新", "recipe_retrieved": "历史配方",
    "candidate_decision": "候选裁决", "accepted_stack_rolled_back": "整栈回退",
    "run_blocked": "保存阻断位置", "stopped": "会话停止", "agent_turn_failed": "Agent 回合失败",
}


def _record(value: Any) -> Record:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, limit: int = 420) -> str:
    result = str(value) if isinstance(value, (str, int, float)) else ""
    return result if len(result) <= limit else result[:limit] + "…"


def _join(*parts: Any) -> str:
    return " · ".join(text for part in parts if (text := _text(part)))


def _actor(role: Any) -> str:
    name = _text(role, 75)
    return _ROLES.get(name, "Coordinator" if name == "coordinator" else name or "Agent")


def _provider(data: Record) -> str:
    name = _text(data.get("provider"), 75)
    return {"deepseek": "DeepSeek", "replay": "本地回放"}.get(name, name or "Agent 服务")


def _call_key(data: Record, turn_field: str) -> tuple[Any, ...]:
    return tuple(_text(data.get(key)) for key in (turn_field, "role", "provider", "task_id"))


def _refs(value: Any) -> Iterator[Record]:
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        else:
            for item in value.values():
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _json_preview(value: Any) -> str:
    if isinstance(value, Mapping) and value:
        return _text(json.dumps(value, ensure_ascii=False, indent=2), 1400)
    return _text(value, 1400)


def _request_preview(receipt: Record) -> tuple[str, str]:
    context = _record(receipt.get("request_context"))
    selected: dict[str, Any] = {}
    labels = []
    pending = context.get("pending_proposals")
    if isinstance(pending, list):
        review_evidence = [{key: item[key] for key in ("proposal_id", "review_evidence") if key in item}
                           for item in pending if isinstance(item, Mapping) and item.get("review_evidence")]
        if review_evidence:
            selected["pending_proposals"] = review_evidence
            simulated = any(_record(_record(item["review_evidence"]).get("summary")).get("mode") == "simulated"
                            for item in review_evidence)
            labels.append("复审请求附有模拟布局补证" if simulated else "复审请求附有布局补证")
    if context.get("review_feedback"):
        selected["review_feedback"] = context["review_feedback"]
        labels.append("请求附有前次审查反馈")
    if "direction_change" in context:
        labels.append("请求附有换向要求")
    selected.update({key: context[key] for key in (
        "direction_change", "goals", "agent_objective", "pending_proposals", "assignment",
        "current", "profile", "recent_decisions",
    ) if key in context and key not in selected})
    return _join(*labels) or "角色请求", _json_preview(selected or context)


def _kernel_delivery(
    task_id: str, result: Record, reader: Callable[[Record], Record | None],
    evidence_href: Callable[[Record], str | None],
) -> dict[str, Any] | None:
    ref = _record(_record(result.get("evidence")).get("delivery"))
    if ref.get("path") != f"evidence/kernel/{task_id}-delivery.json":
        return None
    href = evidence_href(ref)
    delivery = read_native_delivery(dict(ref), reader) if href else None
    if not delivery:
        return None
    route = delivery["backend"]
    status = delivery.get("status")
    if status == "ok" and route == "forge":
        candidate = _record(delivery.get("candidate"))
        message = _join("算子补丁及设备测量", candidate.get("operator"), candidate.get("id"), candidate.get("patch_path"))
    elif status == "ok" and route == "geak":
        service = _record(delivery.get("service_candidate"))
        spec = _record(service.get("spec"))
        setting = _record(service.get("configuration")).get("ascend_use_fia")
        control = f"ascend_use_fia={setting}" if type(setting) is bool else ""
        message = _join("服务配置候选及试跑测量", spec.get("id"), control)
    elif _record(delivery.get("evidence")).get("worker_process") or delivery.get("parent_trial"):
        message = _join("路线运行回执", status_label(status), delivery.get("reason"))
    else:
        return None
    return {"kind": "kernel_delivery", "sender": {"forge": "昇腾 Forge", "geak": "昇腾 GEAK"}[route],
            "receiver": "Kernel Agent", "message": message,
            "operation": _join(f"{task_id} 的结算事件附有这份工具回执", delivery.get("reason"),
                               "独立复测和裁决见本任务验证交接。"),
            "preview": "", "replay": False, "links": [[_text(ref["path"], 150), href]]}


def _handoff(event: Record, proposals: Record, tasks: Record) -> tuple[str, str, str, str] | None:
    """Return only transitions implied by the committed event and its producer."""
    kind = event["type"]
    data = _record(event.get("payload"))
    result = _record(data.get("result"))
    task = _text(data.get("task_id"))
    action = _text(data.get("action"))
    role = _actor(data.get("role"))
    desc = describe_event(event)
    if kind == "sweep_recorded" and not task and data.get("status") == "skipped":
        return "Coordinator", "Session", _join("跳过扫描", data.get("revision"), data.get("reason")), "保存跳过原因。"
    if kind in ("baseline_recorded", "profile_recorded") and data.get("reused_from"):
        label = "原始基线和参考输出" if kind == "baseline_recorded" else "各 rank Profile"
        source = _record(data["reused_from"])
        return "Session", "Coordinator", _join(f"读取历史{label}", source.get("session_id")), desc or ""
    if kind in _RETURNS:
        sender, message = _RETURNS[kind]
        return sender, "Coordinator", _join(message, task, data.get("reason"), desc if not task else ""), "登记回执。"
    if kind in _SAVES:
        return "Coordinator", "Session", desc or _SAVES[kind], f'保存{_SAVES[kind]}。'
    if kind == "agent_intent":
        intent = _record(data.get("intent"))
        recipient = "Coordinator" if intent.get("kind", intent.get("type")) == "repair_startup" else "Intent Router"
        return role, recipient, desc or "角色意图", "收到角色意图，后续事件记录处理结果。"
    if kind == "intent_result":
        return "Intent Router", "Coordinator", desc or "意图处理结果", "Coordinator 记录处理结果。"
    if kind == "proposal_registered":
        return "Intent Router", "Critic", _join(data.get("proposal_id"), action, "正式提案"), desc or "登记待审。"
    if kind == "review_evidence_requested":
        return "Coordinator", "Session", desc or "安排 RoPE 布局补证", "保存已登记的任务编号和补证理由。"
    if kind == "review_evidence_recorded":
        return "RoPE 布局探测", "Coordinator", desc or "RoPE 布局探测回执", "登记探测范围、状态和可用证据。"
    if kind == "proposal_reopened":
        return "Intent Router", "Critic", desc or "提案与补证重回待审", "等待 Critic 的下一次实际复审。"
    if kind == "review_replan":
        return "Coordinator", "Session", desc or "记录后续方向", "保存本轮审查反馈，供后续提案回合参考。"
    if kind in {"review_verdict", "proposal_expired"}:
        proposal = _text(data.get("proposal_id"))
        recipient = _actor(proposals[proposal]) if proposal in proposals else "Session"
        return "Intent Router", recipient, _join(proposal, data.get("verdict"), data.get("reason")), "保存提案处理结果。"
    if kind == "policy_denied":
        return "Intent Router", _actor(data.get("source")), _join("准入拒绝", data.get("code"), data.get("reason")), "登记拒绝原因。"
    if kind == "task_queued":
        sender = "Coordinator" if data.get("source") == "coordinator" else "Intent Router"
        params = _record(data.get("params"))
        retry = _join("重试", params["retry_of"]) if params.get("retry_of") else ""
        message = _join(task, action, retry)
        if action == "rope_layout_probe":
            message = _join(task, _EXECUTORS[action], params.get("proposal_id"), params.get("revision"), retry)
        return sender, "Dispatcher", message, desc or "登记任务。"
    if kind == "task_started":
        return "Dispatcher", _EXECUTORS.get(action, action or "任务执行端"), _join(task, action), desc or "任务开始。"
    if kind in {"task_settled", "task_cancelled", "task_cleanup_pending"}:
        backend = _record(result.get("handoff")).get("backend") or result.get("backend")
        return "Dispatcher", "Coordinator", _join(task, backend, status_label(data.get("status")), result.get("reason")), desc or "记录任务状态。"
    if kind == "delegated_result":
        source = tasks.get(task)
        recipient = _actor(source) if source else "Orchestration"
        operation = "写入请求方收件箱。" if recipient == "Orchestration" else "写入请求方及 Orchestration 的收件箱。"
        return "Intent Router", recipient, _join(task, status_label(data.get("status")), result.get("reason")), operation
    if kind in {"task_cleanup_wait", "task_recovered"}:
        cleanup = _record(data.get("cleanup"))
        released = cleanup.get("released", cleanup.get("confirmed"))
        label = "资源已释放" if released is True else "资源尚未释放" if released is False else "会话记录清理状态"
        return "执行端清理", "Coordinator", _join(task, label, data.get("reason")), desc or "记录清理回执。"
    if kind == "task_retry_scheduled":
        retry = data.get("retry_of") if data.get("role") else data.get("retry_task_id")
        recipient = "Session" if data.get("role") else "Dispatcher"
        return "Coordinator", recipient, _join("安排重试", task or role, retry, data.get("reason")), desc or "登记重试。"
    if kind == "direction_changed":
        return "Coordinator", "Session", desc or "记录换向", "保存换向依据，后续请求需有实际回执才能确认是否附上。"
    if kind == "critic_measured_verdict":
        return "规则 Critic", "Coordinator", _join(data.get("candidate_id"), data.get("version_action"), data.get("reason")), "交回按固定规则复算的裁决。"
    if kind == "goal_assessed":
        if not task:
            return "Coordinator", "Session", desc or _join("目标验收", data.get("status")), "保存当前目标结论。此事件没有对应的复测任务。"
        return "整栈复测", "Coordinator", desc or _join("目标验收", data.get("status")), "保存对原始基线的比较和验收结论。"
    return None


def build_signal_flow(
    events: list[Record], read_evidence: Callable[[Record], Record | None],
    evidence_href: Callable[[Record], str | None], task_links: Mapping[str, str],
) -> list[dict[str, Any]]:
    ordered = sorted((event for event in events if type(event.get("seq")) is int), key=lambda event: event["seq"])
    replies = {_call_key(data, "turn_id"): event for event in ordered if event.get("type") == "agent_provider_call"
               if (data := _record(event.get("payload"))).get("turn_id")}
    # Intent events carry the turn and role, including for Specialist tasks whose turn also records a task ID.
    replay_turns = {(_text(data.get("turn_id")), _text(data.get("role"))) for event in ordered
                    if event.get("type") == "agent_turn_started" and "context" in (data := _record(event.get("payload")))}
    checked_calls = {}
    proposals: dict[str, str] = {}
    tasks: dict[str, str] = {}
    steps = []
    for event in ordered:
        seq, kind = event["seq"], event["type"]
        data = _record(event.get("payload"))
        task_id = _text(data.get("task_id"))
        if kind == "proposal_registered" and data.get("proposal_id") and data.get("source"):
            proposals[_text(data["proposal_id"])] = _text(data["source"])
        if kind == "task_queued" and task_id and data.get("source"):
            tasks[task_id] = _text(data["source"])
        related: list[list[str]] = []
        preview = ""
        linked_refs = data
        if kind in {"agent_provider_started", "agent_provider_call"}:
            request = kind == "agent_provider_started"
            key = _call_key(data, "request_key" if request else "turn_id")
            turn, role, provider, _ = key
            counterpart = replies.get(key) if request and turn else event if not request else None
            if request and counterpart and counterpart["seq"] <= seq:
                counterpart = None
            receipt: Record = {}
            if counterpart:
                linked_refs = _record(counterpart.get("payload"))
                if counterpart["seq"] not in checked_calls:
                    verified = _record(read_evidence(_record(linked_refs.get("evidence"))))
                    checked_calls[counterpart["seq"]] = verified if _call_key(verified, "turn_id") == key else {}
                receipt = checked_calls[counterpart["seq"]]
            if request:
                desc = "Coordinator 已写入开始事件。服务是否返回由后续回执确认。"
                label, preview = _request_preview(receipt) if receipt else ("角色请求", "")
                if counterpart:
                    related.append([f'调用结果 #{counterpart["seq"]}', f'#event-{counterpart["seq"]}'])
                if not preview and provider != "replay":
                    desc += " 此处没有可核验的请求正文。"
                handoff = ("Coordinator", _provider(data), _join(turn, _actor(role), label, data.get("retry_of")), desc)
            else:
                usage = _record(data.get("usage"))
                http = f'HTTP {data["http_status"]}' if data.get("http_status") is not None else ""
                status = _join(status_label(data.get("status")), http, data.get("reason"))
                what = "回复" if receipt.get("response_content") or data.get("status") == "ok" else "调用结果"
                detail = _join(data.get("model"), data.get("response_id"),
                               f'{usage["total_tokens"]} tokens' if usage.get("total_tokens") is not None else "",
                               f'{data["intent_count"]} 条结构化意图' if data.get("intent_count") is not None else "")
                preview = _json_preview(receipt.get("response_content"))
                desc = _join(detail, "已记录模型返回内容。" if preview else "回执里无可展示的回复正文。")
                sender = _provider(data) if http or data.get("status") == "ok" else "Agent API 客户端"
                handoff = (sender, "Coordinator", _join(turn, _actor(role), what, status), desc)
        else:
            handoff = _handoff(event, proposals, tasks)
        if not handoff:
            continue
        sender, receiver, message, operation = handoff
        if not message:
            message = _text(kind, 80)
        if task_id in task_links:
            related.append([f"查看 {task_id} 的验证交接", task_links[task_id]])
        if (kind == "task_settled" and task_id and data.get("action") == "kernel_agent"
                and (delivery := _kernel_delivery(task_id, _record(data.get("result")), read_evidence, evidence_href))):
            delivery["links"].extend(related)
            steps.append({**delivery, "seq": seq, "at": _text(event.get("at"))})
        evidence_seen = set()
        for ref in _refs(linked_refs):
            href = evidence_href(ref)
            if href and href not in evidence_seen:
                evidence_seen.add(href)
                related.append([_text(ref["path"], 150), href])
            if len(evidence_seen) == 6:
                break
        replay = (data.get("provider") == "replay" or
                  _record(data.get("summary")).get("mode") in ("simulated", "replay") or
                  (_text(data.get("turn_id")), _text(data.get("role"))) in replay_turns)
        steps.append({"seq": seq, "at": _text(event.get("at")), "kind": kind, "sender": sender, "receiver": receiver,
                      "message": _text(message), "operation": _text(operation), "replay": replay,
                      "preview": preview, "links": related})
    return steps


SIGNAL_FLOW_STYLES = """
.signals-controls,.signals-proof {display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
.signals-controls button,.signals-controls select {font:inherit;padding:.3rem .55rem;max-width:100%}
.signals-board {position:relative;display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
  gap:2.6rem 1.25rem;padding:2.6rem .8rem .8rem;margin-block:.85rem;border:1px solid GrayText;border-radius:.5rem}
.signals-node {position:relative;min-width:0;padding:.45rem;text-align:center;overflow-wrap:anywhere;
  font-weight:600;border:1px solid GrayText;border-radius:.35rem;color:CanvasText;background:Canvas}
.signals-node[data-active] {border-color:LinkText;outline:2px solid LinkText}
.signals-node small {display:block;font-size:.68rem;min-height:1.55em;color:GrayText}
.signals-board svg {position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;color:LinkText}
.signals-arrow {fill:none;stroke:currentColor;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
.signals-arrow-label {position:absolute;z-index:1;transform:translate(-50%,-108%);box-sizing:border-box;
  max-width:min(24rem,92%);padding:.12rem .38rem;font-size:.8rem;font-weight:650;color:CanvasText;background:Canvas;
  border:1px solid LinkText;border-radius:.3rem;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}
.signals-board[data-playing] .signals-arrow {stroke-dasharray:8 6;animation:signals-flow .6s linear infinite}
@keyframes signals-flow {to {stroke-dashoffset:-14}}
@media(prefers-reduced-motion:reduce) {.signals-board[data-playing] .signals-arrow {animation:none}}
.signals-message {padding:.7rem .85rem;border:1px solid GrayText;border-radius:.4rem;overflow-wrap:anywhere}
.signals-route {display:grid;grid-template-columns:minmax(4rem,1fr) minmax(8rem,3fr) minmax(4rem,1fr);
  gap:.55rem;align-items:end;text-align:center;margin-block:.35rem .7rem}
.signals-route strong {padding:.3rem 0}
.signals-label {position:relative;padding:0 .3rem .35rem;border-bottom:2px solid LinkText}
.signals-label::after {content:'';position:absolute;right:0;bottom:-.31rem;width:.48rem;height:.48rem;
  border:solid LinkText;border-width:2px 2px 0 0;transform:rotate(45deg)}
.signals-message p {margin-block:.35rem .65rem}
.signals-message pre {max-height:15rem;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
"""

_SCRIPT = """
(() => {
  const panel = document.getElementById('signal-flow');
  const get = name => panel.querySelector('[data-signal="' + name + '"]');
  const events = JSON.parse(get('events').textContent);
  const board = get('board');
  const nodes = new Map([...board.querySelectorAll('[data-name]')].map(node => [node.dataset.name, node]));
  const menu = get('menu');
  const play = get('play');
  let current = 0;
  let timer = null;

  function draw() {
    const event = events[current];
    const box = board.getBoundingClientRect();
    const a = nodes.get(event.sender).getBoundingClientRect();
    const b = nodes.get(event.receiver).getBoundingClientRect();
    const path = get('arrow');
    const x1 = a.left + a.width / 2 - box.left;
    const x2 = b.left + b.width / 2 - box.left;
    const y1 = a.top - box.top;
    const y2 = b.top - box.top;
    const rail1 = y1 - 10;
    const rail2 = y2 - 10;
    let labelX, labelY;
    // Keep the active arrow and label in the gaps so nearby participant names remain readable.
    if (Math.abs(y1 - y2) < 1) {
      path.setAttribute('d', `M${x1},${y1} V${rail1} H${x2} V${y2 - 3}`);
      labelX = (x1 + x2) / 2;
      labelY = rail1;
    } else {
      const lane = a.right - box.left + 8;
      path.setAttribute('d', `M${x1},${y1} V${rail1} H${lane} V${rail2} H${x2} V${y2 - 3}`);
      const useFirst = Math.abs(lane - x1) > Math.abs(lane - x2);
      labelX = (lane + (useFirst ? x1 : x2)) / 2;
      labelY = useFirst ? rail1 : rail2;
    }
    const label = get('arrow-label');
    const margin = label.offsetWidth / 2 + 3;
    label.style.left = `${Math.min(box.width - margin, Math.max(margin, labelX))}px`;
    label.style.top = `${labelY}px`;
  }

  function show(index) {
    current = index;
    const event = events[index];
    for (const [name, node] of nodes) {
      const role = name === event.sender ? '发出' : name === event.receiver ? '接收' : '';
      node.toggleAttribute('data-active', !!role);
      node.querySelector('small').textContent = role;
    }
    const values = {
      position: `${index + 1} / ${events.length} · #${event.seq} · ${event.at}` +
        (event.kind === 'kernel_delivery' ? ' · 同一事件所附工具回执' : '') + (event.replay ? ' · 回放' : ''),
      sender: event.sender, receiver: event.receiver, message: event.message,
      operation: event.operation, preview: event.preview,
    };
    for (const [name, value] of Object.entries(values)) get(name).textContent = value;
    const label = get('arrow-label');
    label.textContent = event.message;
    label.title = event.message;
    label.setAttribute('aria-label', event.message);
    label.hidden = false;
    get('content').hidden = !event.preview;
    get('content').open = false;
    const links = [[`原事件 #${event.seq}`, `#event-${event.seq}`], ...event.links];
    get('links').replaceChildren(...links.map(([label, href]) => {
      const link = document.createElement('a');
      link.textContent = label;
      link.setAttribute('href', href);
      if (href.startsWith('#')) link.addEventListener('click', () => {
        const target = document.getElementById(href.slice(1));
        const detail = target && target.closest('details');
        if (detail) detail.open = true;
      });
      return link;
    }));
    get('prev').disabled = index === 0;
    get('next').disabled = index === events.length - 1;
    menu.value = index;
    draw();
  }

  function pause() {
    clearTimeout(timer);
    timer = null;
    play.textContent = '播放';
    board.removeAttribute('data-playing');
  }

  function advance() {
    if (current + 1 === events.length) return pause();
    show(current + 1);
    timer = setTimeout(advance, 1900);
  }

  events.forEach((event, index) => menu.add(new Option(
    `#${event.seq} ${event.sender} → ${event.receiver} · ${event.message.slice(0, 68)}`, index
  )));
  menu.addEventListener('change', () => { pause(); show(Number(menu.value)); });
  get('prev').addEventListener('click', () => { pause(); show(current - 1); });
  get('next').addEventListener('click', () => { pause(); show(current + 1); });
  play.addEventListener('click', () => {
    if (timer !== null) return pause();
    if (current + 1 === events.length) show(0);
    play.textContent = '暂停';
    board.setAttribute('data-playing', '');
    timer = setTimeout(advance, 1900);
  });
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(draw).observe(board);
  else window.addEventListener('resize', draw);
  show(0);
  get('controls').hidden = false;
})();
"""


def signal_flow_html(steps: list[dict[str, Any]], mode: str | None) -> str:
    if not steps:
        return ""
    event_json = json.dumps(steps, ensure_ascii=False, separators=(",", ":"))
    for unsafe, safe in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        event_json = event_json.replace(unsafe, safe)
    participants = dict.fromkeys(step[key] for step in steps for key in ("sender", "receiver"))
    nodes = "".join(f'<div class="signals-node" data-name="{escape(name)}">{escape(name)}<small></small></div>'
                    for name in participants)
    runtime = {"real": "真机事件", "simulated": "模拟会话"}.get(mode, "运行模式未记录")
    return f"""
<section id="signal-flow"><h2>全局信号流</h2>
<p><strong>{runtime}</strong> · 按会话实际写入的事件编号播放。箭头展示已记录的发送内容，步骤下方可打开原事件和可用证据。
附在结算事件中的工具回执沿用该事件编号。
单次验证内部的交接可在<a href="#validation">验证交接图</a>展开。</p>
<div data-signal="controls" hidden><div class="signals-controls">
<button type="button" data-signal="play">播放</button><button type="button" data-signal="prev">上一步</button>
<button type="button" data-signal="next">下一步</button><label for="signal-choice">跳到事件</label>
<select id="signal-choice" data-signal="menu"></select></div></div>
<div class="signals-board" data-signal="board" aria-label="本轮参与部件">{nodes}
<svg aria-hidden="true"><defs><marker id="signal-arrow-tip" viewBox="0 0 10 10" refX="9" refY="5"
  markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="currentColor"/></marker></defs>
<path class="signals-arrow" data-signal="arrow" marker-end="url(#signal-arrow-tip)"/></svg>
<span class="signals-arrow-label" data-signal="arrow-label" tabindex="0" hidden></span></div>
<div class="signals-message"><div data-signal="position"></div><div class="signals-route" aria-live="polite">
<strong data-signal="sender"></strong><span class="signals-label" data-signal="message"></span>
<strong data-signal="receiver"></strong></div><p data-signal="operation"></p>
<div class="signals-proof" data-signal="links"></div>
<details data-signal="content" hidden><summary>查看实际请求或回复摘录</summary><pre data-signal="preview"></pre></details></div>
<noscript><p>浏览器允许本页脚本后即可播放，也可直接查看<a href="#timeline">事件时间线</a>。</p></noscript>
<script type="application/json" data-signal="events">{event_json}</script>
<script data-signal="player">{_SCRIPT}</script></section>"""
