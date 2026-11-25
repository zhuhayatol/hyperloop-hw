"""Resolve editable inputs and launch a self-contained optimization run."""

from __future__ import annotations

import argparse
import math
import os
import re
import stat
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from uuid import uuid4

from .agents import provider_config
from .agents.structured import source_excerpts
from .backends import BackendError
from .contracts import ContractError, json_bytes, read_json, safe_id, validate_reference, validate_target, validate_task
from .session import SessionBusy, atomic_write, file_lease


_PROJECT = Path(__file__).resolve().parents[2]
_PRESETS = {
    "910b3-tp1": ("task.910b3.initial.json", "target.910b3.sglang.json"),
    "910b3-tp1-simulated": ("task.simulated.json", "target.910b3.simulated.json"),
    "910b2-tp2-simulated": ("task.simulated.json", "target.910b2-tp2.simulated.json"),
}
_KEYS = {
    "target", "target_config", "task_template", "session_id", "session_root", "baseline_mode",
    "workload", "rules", "settings", "goals", "allowed_parameters", "agent_provider", "agent_model",
    "reference", "scenario", "agent_source_excerpts", "agent_key_file", "recipe",
}
_PERCENT = r"\s*(\d+(?:\.\d+)?)\s*[%％]"
_GOAL_PATTERNS = {
    "throughput_gain_pct": re.compile(
        r"(?:吞吐(?:量|率)?|\bthroughput)\s*(?:至少\s*)?(?:提高|提升|增加|\+|increase(?:s|d)?(?:\s+by)?|gain(?:\s+of)?)"
        r"\s*(?:至少|at least)?" + _PERCENT, re.I,
    ),
    "ttft_reduction_pct": re.compile(
        r"\bTTFT\s*(?:P95)?\s*[)）]?\s*(?:至少\s*)?(?:降低|减少|缩短|[-−]|reduc(?:e|ed|tion)(?:\s+(?:by|of))?|lower(?:ed)?)"
        r"\s*(?:至少|at least)?" + _PERCENT, re.I,
    ),
}
_TPOT_LIMIT = re.compile(r"\bTPOT\s*(?:P95)?\s*[)）]?\s*(?:最多增加|最多增长|回归不超过|at most)" + _PERCENT, re.I)
_BASELINE_CHUNK = re.compile(r"(?:原始分块|基线分块|baseline\s+(?:prefill\s+)?chunk(?:\s+size)?)\s*[`=：: ]*(\d+)", re.I)


def _load_config(path: Path, raw: bytes) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise ContractError("the start command requires PyYAML; install the Hyperloop project dependencies") from error

    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ContractError(f"{path}: invalid or duplicate YAML field {key!r}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    try:
        config = yaml.load(raw.decode("utf-8-sig"), Loader=UniqueLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ContractError(f"{path}: {error}") from error
    if not isinstance(config, dict):
        raise ContractError(f"{path}: launch configuration must be a YAML mapping")
    if unknown := sorted(config.keys() - _KEYS):
        raise ContractError(f"{path}: unsupported fields {', '.join(unknown)}")
    return config


def _path(config: dict, name: str, directory: Path, default: Path | None = None) -> Path | None:
    if name not in config:
        return default
    value = config[name]
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty path")
    return (directory / Path(value).expanduser()).resolve()


def _matched_value(pattern: re.Pattern, text: str, label: str) -> float | None:
    values = {float(value) for value in pattern.findall(text)}
    if len(values) > 1:
        raise ContractError(f"the task specifies conflicting values for {label}")
    return next(iter(values), None)


def _goals(text: str, configured: object) -> dict:
    if not isinstance(configured, dict) or configured.keys() - _GOAL_PATTERNS.keys():
        raise ContractError("goals may contain only throughput_gain_pct and ttft_reduction_pct")
    goals = {}
    for key, pattern in _GOAL_PATTERNS.items():
        stated = _matched_value(pattern, text, key)
        value = configured.get(key, stated)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ContractError(f"state a positive {key} in the task, or set it in configuration goals")
        if stated is not None and stated != value:
            raise ContractError(f"task and launch configuration disagree on {key}")
        if key == "ttft_reduction_pct" and value >= 100:
            raise ContractError("ttft_reduction_pct must be less than 100")
        goals[key] = float(value)
    return goals


def _read_specs(config: dict, directory: Path, text: str) -> tuple[dict, dict]:
    target_id = safe_id(config.get("target"), "target")
    preset = _PRESETS.get(target_id)
    task_path = _path(config, "task_template", directory, _PROJECT / "configs" / preset[0] if preset else None)
    target_path = _path(config, "target_config", directory, _PROJECT / "configs" / preset[1] if preset else None)
    if task_path is None or target_path is None:
        raise ContractError(f"target {target_id} requires task_template and target_config paths")
    task = validate_task(read_json(task_path))
    target = validate_target(read_json(target_path))
    if target["id"] != target_id:
        raise ContractError("target name does not match target_config.id")

    for group in ("workload", "rules", "settings"):
        overrides = config.get(group, {})
        if not isinstance(overrides, dict):
            raise ContractError(f"{group} overrides must be a mapping")
        if unknown := sorted(overrides.keys() - task[group].keys()):
            raise ContractError(f"unsupported {group} overrides: {', '.join(unknown)}")
        task[group].update(overrides)
    if "allowed_parameters" in config:
        task["allowed_parameters"] = config["allowed_parameters"]

    objective = " ".join(text.split())
    if not objective:
        raise ContractError("the Markdown task is empty")
    task["agent_objective"] = objective
    task["goals"] = _goals(text, config.get("goals", {}))
    tpot_limit = _matched_value(_TPOT_LIMIT, text, "TPOT regression")
    if tpot_limit is not None:
        if "max_tpot_regression_pct" in config.get("rules", {}) and task["rules"]["max_tpot_regression_pct"] != tpot_limit:
            raise ContractError("task and launch configuration disagree on the TPOT limit")
        task["rules"]["max_tpot_regression_pct"] = tpot_limit
    if "tolerance" in config.get("rules", {}) and task["rules"]["tolerance"] != 0:
        raise ContractError("dual-target validation requires zero output tolerance")
    task["rules"]["tolerance"] = 0

    if target["mode"] == "real" and target.get("provider") == "sglang":
        sglang = target.get("sglang", {})
        if not isinstance(sglang, dict) or not isinstance(sglang.get("profile", {}), dict):
            raise ContractError("the SGLang target and its Profile settings must be mappings")
        chunk = _matched_value(_BASELINE_CHUNK, text, "baseline prefill chunk")
        if chunk is not None:
            target_args = sglang.get("server_args", [])
            if not isinstance(target_args, list) or any(not isinstance(arg, str) for arg in target_args):
                raise ContractError("the SGLang target's server_args must be string arguments")
            configured = [arg.split("=", 1)[1] for arg in target_args if arg.startswith("--chunked-prefill-size=")]
            configured.extend(value for flag, value in zip(target_args, target_args[1:]) if flag == "--chunked-prefill-size")
            if configured != [str(int(chunk))]:
                raise ContractError("the task's baseline prefill chunk disagrees with the SGLang target")
        profile = sglang.get("profile", {}).get("enabled", False)
        if task["settings"]["profile_enabled"] and not profile:
            raise ContractError("the SGLang target must enable its Profile capability")
        route = task["settings"]["kernel_backend"]
        if route != "disabled" and not sglang.get(f"native_{route}"):
            raise ContractError(f"the SGLang target must enable native {route}")
    return validate_task(task), target


@contextmanager
def _provider_key(provider: str, path: Path | None):
    key_name = "DEEPSEEK_API_KEY"
    if provider != "deepseek" or os.environ.get(key_name):
        yield
        return
    path = path or Path.home() / ".config" / "hyperloop" / "deepseek.key"
    if not path.is_file() or (os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ContractError(f"provide DEEPSEEK_API_KEY or a private key file at {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value or any(char.isspace() for char in value):
        raise ContractError("the DeepSeek key file must contain a single token")
    os.environ[key_name] = value
    try:
        yield
    finally:
        os.environ.pop(key_name, None)


def launch_lease(session_root: Path, target: dict):
    if target["mode"] == "real" and target.get("provider") == "sglang":
        return file_lease(session_root / ".hyperloop-npu.lock")
    return nullcontext()


def _check_idle_device(target: dict, root: Path) -> None:
    if target["mode"] != "real" or target.get("provider") != "sglang":
        return
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                                timeout=30, check=False)
    except subprocess.SubprocessError as error:
        raise ContractError(f"NPU availability check failed: {error}") from error
    atomic_write(root / "npu-before.txt", (result.stdout + result.stderr).encode("utf-8"))
    if result.returncode or "No running processes found" not in result.stdout:
        raise ContractError(f"the target NPU must be available and idle; see {root / 'npu-before.txt'}")


def start(args: argparse.Namespace, *, capture_reference, run_session) -> dict:
    from .observability import RunObserver

    config_path = args.config.resolve()
    raw_config = config_path.read_bytes()
    config = _load_config(config_path, raw_config)
    raw_task = args.task.read_bytes()
    task, target = _read_specs(config, config_path.parent, raw_task.decode("utf-8-sig"))
    mode = target["mode"]
    agent = provider_config(config.get("agent_provider", "deepseek" if mode == "real" else "replay"),
                            config.get("agent_model"))
    directory = config_path.parent
    session_id = safe_id(config.get("session_id", f"h{uuid4().hex[:8]}"), "session_id")
    session_root = _path(config, "session_root", directory, directory / "sessions")
    root = session_root / session_id
    baseline_mode = config.get("baseline_mode", "measure")
    if baseline_mode not in ("measure", "reuse_latest"):
        raise ContractError("baseline_mode must be measure or reuse_latest")
    reuse_baseline = baseline_mode == "reuse_latest"
    scenario_path = _path(config, "scenario", directory)
    explicit_reference = _path(config, "reference", directory)
    if reuse_baseline and explicit_reference is not None:
        raise ContractError("baseline_mode: reuse_latest uses the selected Session's reference")
    if mode == "real" and explicit_reference is not None:
        raise ContractError("real starts independently capture reference outputs")
    if mode == "simulated" and config["target"] in ("910b3-tp1-simulated", "910b2-tp2-simulated"):
        scenario_path = scenario_path or _PROJECT / "fixtures" / "scenario.simulated.json"
        explicit_reference = explicit_reference or _PROJECT / "fixtures" / "reference.simulated.json"
    if mode == "simulated" and explicit_reference is None and not reuse_baseline:
        raise ContractError("a custom simulated target requires a reference fixture")
    scenario = read_json(scenario_path) if scenario_path else {}
    reserved = {"startup", "agent_provider", "agent_source_excerpts", "retrieved_recipe", "reference_capture"}
    if scenario.keys() & reserved:
        raise ContractError("start reserves scenario fields for startup, Agent provider, recipe, and reference metadata")
    key_path = _path(config, "agent_key_file", directory)
    scenario["startup"] = {"schema_version": 1, "inputs": "inputs", "agent_key_file": str(key_path) if key_path else None}

    excerpt_path = _path(config, "agent_source_excerpts", directory)
    excerpt_file = read_json(excerpt_path) if excerpt_path else None
    if excerpt_file is not None:
        if agent["provider"] != "deepseek" or excerpt_file.keys() != {"excerpts"}:
            raise ContractError("agent_source_excerpts requires DeepSeek and an excerpts-only JSON file")
        source_excerpts(excerpt_file["excerpts"], target)
    recipe = _path(config, "recipe", directory)

    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    atomic_write(inputs / "任务.md", raw_task)
    atomic_write(inputs / "启动配置.yaml", raw_config)
    for name, record in (("task", task), ("target", target), ("scenario", scenario)):
        atomic_write(inputs / f"{name}.json", json_bytes(record))
    if excerpt_file is not None:
        atomic_write(inputs / "excerpts.json", json_bytes(excerpt_file))

    observer = RunObserver(root, latest_path=session_root / "最新说明.md")
    startup = {"status": "running", "stage": "reference", "mode": mode, "session_id": session_id,
               "target": target["id"], "goals": task["goals"], "session": f"{mode}/{session_id}",
               "baseline_mode": baseline_mode}
    atomic_write(root / "startup.json", json_bytes(startup))
    goals = task["goals"]
    observer.progress("启动", f"目标 {target['id']}，吞吐提高 {goals['throughput_gain_pct']:g}%，"
                      f"TTFT P95 降低 {goals['ttft_reduction_pct']:g}%，工作点和规则见 {inputs}")
    observer.progress("冻结配置", f"基线 {task['initial_revision']}，模型 {task['model']['id']}，"
                      f"设备 {','.join(target['devices'])}，TP={target['tp_size']}，"
                      f"并发 {task['workload']['concurrency']}，预热 {task['rules']['warmup']} 轮，"
                      f"正式测速 {task['rules']['repeats']} 轮")
    try:
        with launch_lease(session_root, target), _provider_key(agent["provider"], key_path):
            _check_idle_device(target, root)
            if reuse_baseline:
                reference_file, receipt = None, None
                observer.progress("基线复用", f"将在 {session_root} 中查找最近一次匹配且有原始证据的基线")
            elif mode == "real":
                observer.progress("参考采集", "启动独立原版服务，采集冻结请求的参考输出")
                with observer.activate():
                    captured = capture_reference(argparse.Namespace(task=inputs / "task.json", target=inputs / "target.json",
                                                                     output=root / "reference"))
                reference_file, receipt = Path(captured["reference"]), Path(captured["receipt"])
                observer.progress("参考采集", f"参考及采集回执已保存到 {reference_file.parent}")
            else:
                reference = validate_reference(read_json(explicit_reference), task)
                reference_file, receipt = root / "reference" / "reference.json", None
                atomic_write(reference_file, json_bytes(reference))
                observer.progress("参考采集", "已锁定模拟参考值")
            startup.update(stage="session")
            atomic_write(root / "startup.json", json_bytes(startup))
            observer.progress("会话", f"创建 {mode}/{session_id} 并开始优化")
            invocation = argparse.Namespace(
                task=inputs / "task.json", target=inputs / "target.json", reference=reference_file,
                reference_receipt=receipt, scenario=inputs / "scenario.json", session_root=root,
                session_id=session_id, kernel_backend=None, agent_provider=agent["provider"],
                agent_model=agent.get("model"), agent_source_excerpts=inputs / "excerpts.json" if excerpt_file else None,
                recipe=recipe, max_steps=None, baseline_history=session_root if reuse_baseline else None,
            )
            result = run_session(invocation, observer=observer)
    except (ValueError, OSError, BackendError, SessionBusy, KeyboardInterrupt) as error:
        status = "paused" if isinstance(error, KeyboardInterrupt) else "blocked" if isinstance(error, SessionBusy) else "failed"
        startup.update(status=status, reason=str(error) or "interrupted")
        atomic_write(root / "startup.json", json_bytes(startup))
        if not (root / mode / session_id / "manifest.json").is_file():
            observer.progress(startup["stage"], f"{startup['reason']}，诊断记录位于 {root}", status=status)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise ContractError(f"{startup['stage']} failed; see {root / 'startup.json'}: {error}") from error

    record_start_result(root, result)
    return {**result, "output": str(root), "startup": str(root / "startup.json")}


def record_start_result(root: Path, result: dict, *, paused: bool = False) -> None:
    path = root / "startup.json"
    if not path.is_file():
        return
    startup = read_json(path)
    blocker = result.get("blocker") or {}
    status = "done" if result["phase"] == "DONE" else "paused" if paused and not blocker else "blocked"
    startup.update(status=status, phase=result["phase"], reason=result["stop_reason"] or blocker.get("reason"))
    atomic_write(path, json_bytes(startup))
