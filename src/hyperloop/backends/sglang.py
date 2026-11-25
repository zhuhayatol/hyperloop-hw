"""Run an isolated, local SGLang service on one visible Ascend NPU."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from .runtime import BackendError, Record, fingerprint
from .sglang_forward import (
    FORWARD_CONTROL_ENV, FORWARD_SOURCE, FORWARD_SYMBOL, forward_repeats, forward_source_edit, measure_forward,
)
from .sglang_overlay import dedicated_launch, loaded_sources, protected_paths, sandbox_launch, source_edit, source_path, stage_overlay
from .sglang_profile_export import export_capture, exporter_owner


_HOST = "127.0.0.1"
_LEASE = "sglang-lease.json"
_LEASE_ENV = "HYPERLOOP_SGLANG_LEASE"
_LOOPBACK = build_opener(ProxyHandler({}))
_PARAMETER_FLAG = "--chunked-prefill-size"
_DISABLE_GRAPH = "--disable-cuda-graph"
_DECODE_GRAPH = "--cuda-graph-backend-decode"
_PREFILL_GRAPH = "--cuda-graph-backend-prefill"
_GRAPH_SWITCHES = {_DISABLE_GRAPH, _DECODE_GRAPH, _PREFILL_GRAPH,
                   "--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"}
_FORGE_SOURCE = "python/sglang/srt/layers/activation.py"
_FORGE_MARKER = re.compile(r"HYPERLOOP_FORGE_SWIGLU_INVOKED_[0-9a-f]{12}")
_FORGE_CALL = re.compile(
    r"pid=(?P<pid>[1-9][0-9]*) tokens=(?P<tokens>[1-9][0-9]*) width=(?P<width>[1-9][0-9]*) "
    r"kernel=(?P<kernel>_hyperloop_forge_swiglu\w*) hash=(?P<hash>[0-9a-f]{16,128})"
)
_RESERVED_FLAGS = {
    "--host", "--port", "--model-path", "--model", "--tokenizer-path", "--device", "--dtype",
    "--attention-backend", "--tp-size", "--tensor-parallel-size", "--tp", "--dp-size",
    "--data-parallel-size", "--dp", "--pp-size", "--pipeline-parallel-size", "--pp", "--nnodes",
    "--node-rank", "--dist-init-addr", "--base-gpu-id", "--gpu-id-step", "--served-model-name",
    "--api-key", "--admin-api-key", "--config", "--grpc-mode", "--encoder-only", "--language-only",
    "--ssl-certfile", "--ssl-keyfile", "--trust-remote-code",
}
_MODEL_FILES = {
    "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model",
}
_PROBE = """
import json
import platform
from importlib import metadata
import sys

import torch
import torch_npu
import sglang

count = torch.npu.device_count()
if not torch.npu.is_available() or count != 1:
    raise SystemExit(f'Expected exactly one visible NPU, found {count}')
value = torch.arange(1, 5, device='npu:0', dtype=torch.float32)
check = (value * value).cpu().tolist()
if check != [1.0, 4.0, 9.0, 16.0]:
    raise SystemExit(f'NPU arithmetic check failed: {check}')
packages = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'sglang': metadata.version('sglang')}
for package in ('sgl-kernel-npu', 'custom-ops', 'triton', 'triton-ascend'):
    try:
        packages[package.replace('-', '_')] = metadata.version(package)
    except metadata.PackageNotFoundError:
        pass
print('HYPERLOOP_NPU_PROBE=' + json.dumps({'visible_npus': count, 'device_name': torch.npu.get_device_name(0),
                  'smoke': check, 'packages': packages, 'module': sglang.__file__,
                  'python': sys.executable, 'python_version': platform.python_version(),
                  'machine': platform.machine()}))
"""


@dataclass(frozen=True)
class _Process:
    state: str
    group: int
    session: int
    start_ticks: int


def _process(pid: int) -> _Process | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return _Process(fields[0], int(fields[2]), int(fields[3]), int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def _write_lease(workspace: Path, lease: Record) -> None:
    temporary = workspace / f".{_LEASE}.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(lease, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(workspace / _LEASE)


def _lease_processes(lease: Mapping[str, Any]) -> dict[int, _Process]:
    """Use a unique inherited launch marker so recovery cannot stop another service."""
    owned: dict[int, _Process] = {}
    marker = f"{_LEASE_ENV}={lease['nonce']}".encode()
    uid = lease.get("run_uid", os.getuid())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            if marker not in (entry / "environ").read_bytes().split(b"\0"):
                continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (identity := _process(int(entry.name))) is not None and identity.state != "Z":
            owned[int(entry.name)] = identity
    return owned


def _live_group(lease: Mapping[str, Any]) -> list[int]:
    group = lease.get("pid")
    start = lease.get("start_ticks")
    if type(group) is not int or group < 1 or type(start) is not int:
        return []
    leader = _process(group)
    if leader is not None and leader.start_ticks != start:
        return []
    remaining = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            pid = int(entry.name)
            if (identity := _process(pid)) is not None and identity.state != "Z" and identity.group == identity.session == group:
                remaining.append(pid)
    return sorted(remaining)


def _signal_owned(processes: Mapping[int, _Process], sig: signal.Signals) -> list[int]:
    signalled = []
    for pid, identity in processes.items():
        current = _process(pid)
        if current is not None and current.state != "Z" and current.start_ticks == identity.start_ticks:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            signalled.append(pid)
    return sorted(signalled)


def _wait_lease(lease: Mapping[str, Any], timeout: float) -> list[int]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = sorted(set(_lease_processes(lease)) | set(_live_group(lease)))
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.25)


def _stop_lease(lease: Mapping[str, Any]) -> Record:
    started = time.monotonic()
    phases = []
    remaining = _wait_lease(lease, 0)
    leader = lease.get("pid")
    for phase, sig, timeout in (("launcher_grace", signal.SIGTERM, 30),
                                ("worker_grace", signal.SIGTERM, 8),
                                ("forced_wait", signal.SIGKILL, 25)):
        if not remaining:
            break
        owned = _lease_processes(lease)
        if "run_uid" in lease and type(leader) is int:
            owned = {pid: identity for pid, identity in owned.items() if identity.session == leader}
        if phase == "launcher_grace":
            identity = owned.get(leader)
            if (identity is None or identity.start_ticks != lease.get("start_ticks")
                    or identity.group != leader or identity.session != leader):
                continue
            # Let the verified launcher finish the SGLang shutdown and reap its own workers.
            owned = {leader: identity}
        sent = _signal_owned(owned, sig)
        if phase == "launcher_grace" and not sent:
            continue
        remaining = _wait_lease(lease, timeout)
        phases.append({"phase": phase, "signal": sig.name, "signalled_pids": sent,
                       "wait_limit_seconds": timeout, "remaining_pids": remaining})
    return {"released": not remaining, "remaining_pids": remaining,
            "cleanup_wait_seconds": round(time.monotonic() - started, 3), "cleanup_phases": phases}


def _local_path(config: Mapping[str, Any], key: str, *, default: Path | None = None) -> Path:
    value = config.get(key, str(default) if default else None)
    if not isinstance(value, str) or not value.strip():
        raise BackendError(f"target.sglang.{key} must be a local path")
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.exists():
        raise BackendError(f"target.sglang.{key} must be an existing absolute path: {path}")
    return path


def _time_limit(config: Mapping[str, Any], name: str, default: int) -> float:
    value = config.get(name, default)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise BackendError(f"target.sglang.{name} must be positive")
    return float(value)


def _chunk_args(args: list[str], chunk: int | None) -> tuple[list[str], int | None]:
    """Keep one exact chunk flag regardless of whether the target used '=' syntax."""
    result = []
    iterator = iter(args)
    original = None
    for arg in iterator:
        flag, separator, value = arg.partition("=")
        if flag != _PARAMETER_FLAG:
            result.append(arg)
            continue
        if original is not None:
            raise BackendError(f"target repeats {_PARAMETER_FLAG}")
        value = value if separator else next(iterator, "")
        try:
            original = int(value)
        except ValueError as error:
            raise BackendError(f"invalid {_PARAMETER_FLAG}: {value!r}") from error
    actual = original if chunk is None else chunk
    if actual is not None:
        result.extend((_PARAMETER_FLAG, str(actual)))
    return result, actual


def _decode_graph_args(args: list[str], enabled: bool) -> tuple[list[str], Record]:
    """Override decode only for the pinned graph-off target, leaving prefill off."""
    if type(enabled) is not bool:
        raise BackendError("decode_graph_enabled must be a boolean")
    switches = [arg for arg in args if arg.partition("=")[0] in _GRAPH_SWITCHES]
    if switches != [_DISABLE_GRAPH]:
        raise BackendError("a decode graph candidate needs one graph-off target flag and no phase graph overrides")
    if not enabled:
        return args, {"disable_cuda_graph": True}
    updated = [arg for arg in args if arg != _DISABLE_GRAPH]
    updated.extend((f"{_DECODE_GRAPH}=full", f"{_PREFILL_GRAPH}=disabled"))
    return updated, {"disable_cuda_graph": False, "cuda_graph_backend_decode": "full",
                     "cuda_graph_backend_prefill": "disabled"}


def _verify_decode_graph_capture(log: Path) -> None:
    """Require capture completion from the owned, freshly started server log."""
    began = False
    try:
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "Capture target decode NPU graph begin. backend=full" in line:
                    began = True
                elif began and "Capture target decode NPU graph end." in line:
                    return
    except OSError as error:
        raise BackendError(f"could not read decode graph capture proof: {log}") from error
    raise BackendError(f"SGLang did not record completion of full decode NPU graph capture: {log}")


def _provider_credential(name: str) -> bool:
    key = name.upper()
    return key.endswith(("_API_KEY", "_API_TOKEN")) or key in {"GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"}


class SGLangBackend:
    """Use the installed SGLang and model with an owned loopback subprocess."""

    def __init__(self, target: Mapping[str, Any]) -> None:
        if target.get("mode") != "real" or target.get("tp_size") != 1 or target.get("devices") != ["0"]:
            raise BackendError("SGLang real runtime currently requires tp_size=1 and devices=['0']")
        config = target.get("sglang")
        if not isinstance(config, Mapping):
            raise BackendError("real SGLang runtime requires target.sglang")
        self._target = deepcopy(dict(target))
        self._python = _local_path(config, "python")
        self._source = _local_path(config, "source_path").resolve()
        self._model = _local_path(config, "model_path").resolve()
        self._tokenizer = _local_path(config, "tokenizer_path", default=self._model).resolve()
        if not self._python.is_file() or not all(path.is_dir() for path in (self._source, self._model, self._tokenizer)):
            raise BackendError("SGLang runtime requires a Python executable and local source, model and tokenizer directories")
        allowed_paths = config.get("allowed_source_paths", [])
        if not isinstance(allowed_paths, list):
            raise BackendError("target.sglang.allowed_source_paths must be a list of exact SGLang files")
        self._allowed_source = frozenset(source_path(path) for path in allowed_paths)
        trial_user = config.get("source_trial_user")
        self._trial_user = None
        if trial_user is not None:
            if (not isinstance(trial_user, Mapping) or trial_user.keys() != {"uid", "gid"}
                    or any(type(value) is not int or value < 1 for value in trial_user.values())
                    or config.get("source_sandbox") is not None):
                raise BackendError("target.sglang.source_trial_user requires positive uid and gid without source_sandbox")
            if sys.platform != "linux" or os.geteuid() != 0 or not Path("/usr/bin/setpriv").is_file():
                raise BackendError("the dedicated SGLang account requires Linux, setpriv and a trusted root evaluator")
            self._trial_user = (trial_user["uid"], trial_user["gid"])
        self._source_sandbox = None
        if self._allowed_source and self._trial_user is None:
            sandbox = _local_path(config, "source_sandbox").resolve()
            if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
                raise BackendError("target.sglang.source_sandbox must name the executable bubblewrap program")
            self._source_sandbox = sandbox
        args = config.get("server_args", [])
        if not isinstance(args, list) or any(not isinstance(arg, str) or not arg for arg in args):
            raise BackendError("target.sglang.server_args must be a list of non-empty strings")
        forbidden = sorted({arg.partition("=")[0] for arg in args} & _RESERVED_FLAGS)
        if forbidden:
            raise BackendError(f"SGLang runtime owns the following server flags: {', '.join(forbidden)}")
        self._server_args = list(args)
        self._ascend_use_fia = config.get("ascend_use_fia")
        if self._ascend_use_fia is not None and type(self._ascend_use_fia) is not bool:
            raise BackendError("target.sglang.ascend_use_fia must be a boolean")
        self._startup_timeout = _time_limit(config, "startup_timeout", 240)
        self._request_timeout = _time_limit(config, "request_timeout", 120)
        profile = config.get("profile", {})
        if (not isinstance(profile, Mapping) or profile.keys() - {"enabled", "export_timeout", "offline_export", "steps"}
                or type(profile.get("enabled", False)) is not bool
                or type(profile.get("offline_export", False)) is not bool):
            raise BackendError("target.sglang.profile requires valid enabled, offline_export, steps and export_timeout")
        self._profile_enabled = profile.get("enabled", False)
        self._profile_export_timeout = _time_limit(profile, "export_timeout", 120)
        self._profile_steps = profile.get("steps")
        if self._profile_steps is not None and (type(self._profile_steps) is not int or self._profile_steps <= 0):
            raise BackendError("target.sglang.profile.steps must be a positive integer")
        if profile.get("offline_export") and (not self._profile_enabled or not self._trial_user):
            raise BackendError("offline Ascend export requires an enabled Profile and isolated capture user")
        self._native_geak = config.get("native_geak", False)
        self._native_forge = config.get("native_forge", False)
        if type(self._native_geak) is not bool or type(self._native_forge) is not bool:
            raise BackendError("target.sglang native_geak and native_forge must be booleans")
        if self._native_geak and (self._ascend_use_fia is None or not self._profile_enabled):
            raise BackendError("native GEAK requires a locked FIA default and enabled Profile")
        if self._native_forge and (not self._profile_enabled or not self._trial_user
                                   or _FORGE_SOURCE not in self._allowed_source):
            raise BackendError("native Forge requires Profile and an isolated, allow-listed SwiGLU source path")
        self._rope_layout_probe = config.get("rope_layout_probe", False)
        if type(self._rope_layout_probe) is not bool:
            raise BackendError("target.sglang.rope_layout_probe must be a boolean")
        if self._rope_layout_probe and (not self._profile_enabled or self._trial_user is None):
            raise BackendError("RoPE layout evidence requires Profile and a dedicated SGLang account")
        self._forward_repeats = forward_repeats(config.get("model_forward", {}))
        if self._forward_repeats is not None:
            if self._trial_user is None:
                raise BackendError("real model-forward measurement requires a dedicated, isolated SGLang account")
            try:
                model_config = json.loads((self._model / "config.json").read_bytes())
            except (OSError, ValueError) as error:
                raise BackendError(f"could not read the locked model-forward architecture: {error}") from error
            if (not isinstance(model_config, dict) or model_config.get("model_type") != "qwen2"
                    or model_config.get("architectures") != ["Qwen2ForCausalLM"]):
                raise BackendError("real model-forward measurement requires Qwen2ForCausalLM")
        self._session: Path | None = None
        self._stacks: dict[str, Record] = {}
        self._hashes: dict[Path, tuple[tuple[int, int, int], str]] = {}
        self._probe: Record | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._active: Record | None = None
        self._env = {key: value for key, value in os.environ.items() if not _provider_credential(key)}
        self._env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                          "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})
        self._profile_export_owner = exporter_owner(self._env, self._trial_user) if profile.get("offline_export") else None

    def bind_session(self, path: Path) -> None:
        self._session = path.resolve()
        if self._trial_user is not None:
            protected_paths([self._session], *self._trial_user)

    def _run(self, argv: list[str], timeout: float) -> str:
        try:
            run = subprocess.run(argv, env=self._env, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BackendError(f"SGLang preflight command {argv[0]} failed: {error}") from error
        if run.returncode:
            raise BackendError(f"SGLang preflight command {argv[0]} returned {run.returncode}: "
                               f"{(run.stderr or run.stdout).strip()[-2400:]}")
        return run.stdout

    def _file_hash(self, file: Path) -> str:
        status = file.stat()
        key = (status.st_ino, status.st_size, status.st_mtime_ns)
        previous = self._hashes.get(file)
        if previous is None or key != previous[0]:
            with file.open("rb") as stream:
                self._hashes[file] = (key, hashlib.file_digest(stream, "sha256").hexdigest())
        return self._hashes[file][1]

    def _source_revision(self) -> str:
        marker = self._source / ".hyperloom-source-commit"
        if (self._source / ".git").exists():
            revision = self._run(["git", "-C", str(self._source), "rev-parse", "HEAD"], 10).strip()
            modified = self._run(["git", "-C", str(self._source), "status", "--porcelain", "--", "python/sglang"], 10)
            if modified.strip():
                raise BackendError("SGLang source checkout has uncommitted changes under python/sglang")
        elif marker.is_file():
            revision = marker.read_text(encoding="utf-8").strip()
        else:
            raise BackendError("SGLang source needs a Git HEAD or a .hyperloom-source-commit marker")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise BackendError("SGLang source revision is not a full Git commit")
        return revision.lower()

    def _cann_version(self, machine: str) -> str:
        toolkit = Path(self._env.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
        candidates = [toolkit / f"{machine}-linux" / "ascend_toolkit_install.info", toolkit / "version.cfg"]
        for path in candidates:
            if path.is_file():
                data = path.read_text(encoding="utf-8", errors="replace")
                if match := re.search(r"^\s*(?:version|Version)\s*=\s*([^\r\n]+)", data, flags=re.M):
                    return match.group(1).strip().strip("'\"")
        raise BackendError(f"could not read the installed CANN version under {toolkit}")

    def probe(self) -> Record:
        if sys.platform != "linux":
            raise BackendError("the real SGLang runtime requires Linux")
        if self._trial_user is not None:
            sources = [self._source / path for path in self._allowed_source]
            if self._forward_repeats is not None:
                sources.append(self._source / FORWARD_SOURCE)
            protected_paths([self._python, self._source, self._model, self._tokenizer, Path(__file__), *sources],
                            *self._trial_user)
        try:
            prefix = "HYPERLOOP_NPU_PROBE="
            lines = self._run([str(self._python), "-c", _PROBE], 90).splitlines()
            payload = next((line.removeprefix(prefix) for line in reversed(lines) if line.startswith(prefix)), "")
            observed = json.loads(payload)
            module = Path(observed["module"]).resolve()
            installed = self._source / "python" / "sglang"
            if not module.is_relative_to(installed):
                raise BackendError(f"SGLang imports {module}, outside required source {installed}")
            software = {
                **observed["packages"], "sglang_commit": self._source_revision(),
                "cann": self._cann_version(observed["machine"]), "python": observed["python_version"],
            }
            inventory = self._run(["npu-smi", "info"], 20)
            device_name = observed["device_name"]
            expected = self._target.get("device_name")
            if expected and (not isinstance(expected, str) or expected.casefold() not in (device_name + inventory).casefold()):
                raise BackendError(f"NPU probe did not find required device {expected!r}")
            model_files = {
                file.name: self._file_hash(file) for file in sorted(self._model.iterdir())
                if file.is_file() and (file.name in _MODEL_FILES or file.suffix == ".safetensors")
            }
            tokenizer_files = {
                file.name: self._file_hash(file) for file in sorted(self._tokenizer.iterdir())
                if file.is_file() and file.name in _MODEL_FILES
            }
            if "config.json" not in model_files or not any(name.endswith(".safetensors") for name in model_files):
                raise BackendError("SGLang model directory must include config.json and safetensors weights")
            if not any(name in tokenizer_files for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
                raise BackendError("SGLang tokenizer directory has no recognized vocabulary file")
            if self._trial_user is not None:
                locked_files = [self._model / name for name in model_files] + [self._tokenizer / name for name in tokenizer_files]
                protected_paths(locked_files, *self._trial_user)
            environment = {
                "mode": "real", "target_id": self._target["id"], "devices": ["0"], "tp_size": 1,
                "rank_map": {"0": "0"}, "software": software, "device_name": device_name,
                "visible_npus": observed["visible_npus"], "device_visibility": self._env.get("ASCEND_RT_VISIBLE_DEVICES"),
                "architecture": observed["machine"], "python_path": observed["python"],
                "sglang_module": str(module), "model_path": str(self._model), "tokenizer_path": str(self._tokenizer),
                "model_sha256": model_files, "tokenizer_sha256": tokenizer_files,
                "fixed_server_args": self._server_args,
            }
            if self._source_sandbox is not None:
                environment["source_integration"] = {"allowed_paths": sorted(self._allowed_source),
                                                     "sandbox": str(self._source_sandbox),
                                                     "sandbox_sha256": self._file_hash(self._source_sandbox)}
            if self._trial_user is not None:
                uid, gid = self._trial_user
                environment["source_integration"] = {"allowed_paths": sorted(self._allowed_source),
                                                     "dedicated_run_uid": uid, "dedicated_run_gid": gid}
            if self._ascend_use_fia is not None:
                environment["stock_attention_control"] = {"ASCEND_USE_FIA": self._ascend_use_fia, "ASCEND_USE_FA": False}
            if self._profile_export_owner is not None:
                uid, gid, binary = self._profile_export_owner
                environment["offline_profile_export"] = {
                    "uid": uid, "gid": gid, "msprof": str(binary), "sha256": self._file_hash(binary),
                }
            if self._profile_steps is not None:
                environment["profile_scheduler_steps"] = self._profile_steps
            if self._rope_layout_probe:
                from .sglang_rope import ROPE_MODEL_SOURCE, ROPE_SOURCE

                environment["rope_layout_sources"] = {
                    path: self._file_hash(self._source / path) for path in (ROPE_MODEL_SOURCE, ROPE_SOURCE)
                }
            if self._forward_repeats is not None:
                environment["model_forward_measurement"] = {
                    "symbol": FORWARD_SYMBOL, "repeats": self._forward_repeats, "graph_replay": False,
                    "pinned_source_sha256": self._file_hash(self._source / FORWARD_SOURCE),
                }
        except (OSError, KeyError, IndexError, ValueError, TypeError) as error:
            raise BackendError(f"could not confirm the real SGLang environment: {error}") from error
        capabilities = ["benchmark", "verify"]
        if self._profile_enabled:
            capabilities.append("profile")
        if self._forward_repeats is not None:
            capabilities.append("forward")
        if self._rope_layout_probe:
            capabilities.append("rope_layout_probe")
        for route, enabled in (("native_geak", self._native_geak), ("native_forge", self._native_forge)):
            if enabled:
                capabilities.append(route)
        self._probe = {
            "mode": "real", "environment": environment, "environment_fingerprint": fingerprint(environment),
            "capabilities": capabilities,
            "npu_smoke": observed["smoke"], "npu_smi": inventory,
        }
        return deepcopy(self._probe)

    def bind_revision(self, revision: Mapping[str, Any], *, expected_stack: Mapping[str, Any]) -> None:
        identifier = revision["id"]
        stack = deepcopy(dict(expected_stack))
        parameters = stack["parameters"]
        if stack["kernels"]:
            self._validate_kernel_stack(stack["kernels"])
        for change in stack["framework"]:
            if change["kind"] == "patch":
                source_edit(change, self._allowed_source)
                if self._source_sandbox is None and self._trial_user is None:
                    raise BackendError("SGLang source candidates require an isolated runtime")
            elif change["kind"] != "parameters":
                raise BackendError(f"unsupported SGLang framework change: {change['kind']}")
        if parameters.keys() - {"chunked_prefill_size", "ascend_use_fia", "decode_graph_enabled"}:
            raise BackendError("unsupported real SGLang framework parameter")
        if ("ascend_use_fia" in parameters
                and (self._ascend_use_fia is None or type(parameters["ascend_use_fia"]) is not bool)):
            raise BackendError("Ascend operator selection requires a locked target default and a boolean candidate")
        if "chunked_prefill_size" in parameters:
            chunk = parameters["chunked_prefill_size"]
            if type(chunk) is not int or (chunk < 1 and chunk != -1):
                raise BackendError("chunked_prefill_size must be positive or -1")
        if "decode_graph_enabled" in parameters:
            decode_graph = parameters["decode_graph_enabled"]
            _decode_graph_args(self._server_args, decode_graph)
            if decode_graph and stack["kernels"]:
                raise BackendError("the native Forge invocation proof requires decode graph to stay disabled")
        if identifier in self._stacks and self._stacks[identifier] != stack:
            raise BackendError(f"revision {identifier} is already bound to a different stack")
        self._stacks[identifier] = stack

    def _validate_kernel_stack(self, kernels: list[Record]) -> None:
        if not self._native_forge or len(kernels) != 1:
            raise BackendError("this real runtime only supports one native Forge SwiGLU overlay")
        kernel = kernels[0]
        if (kernel.get("kind") != "operator_patch" or kernel.get("backend") != "forge"
                or kernel.get("operator") != "SwiGLU" or kernel.get("patch_path") != _FORGE_SOURCE
                or not isinstance(kernel.get("invocation_marker"), str)
                or not _FORGE_MARKER.fullmatch(kernel["invocation_marker"])):
            raise BackendError("unrecognized native Forge operator overlay")
        source_edit(kernel, self._allowed_source)
        if kernel.get("pinned_source_sha256") != self._file_hash(self._source / _FORGE_SOURCE):
            raise BackendError("native Forge overlay does not match the pinned SGLang source")

    def run_native_forge(self, *, profile: Mapping[str, Any], attempt: str) -> Record:
        from ..kernel.ascend_forge import run_forge

        if not self._native_forge or not self._trial_user or self._session is None or self._child is not None:
            raise BackendError("native Forge requires an idle, bound and protected Session")
        uid, gid = self._trial_user
        return run_forge(
            python=self._python, model_config=self._model / "config.json", source_root=self._source,
            hotspots=profile["hotspots"], workspace=self._session / "workspaces" / attempt,
            environment=self._env, uid=uid, gid=gid,
        )

    def run_rope_layout_probe(
        self, *, workload: Mapping[str, Any], profile: Mapping[str, Any],
        revision: Mapping[str, Any], attempt: str,
    ) -> Record:
        from .sglang_rope import ROPE_MODEL_SOURCE, observe_rope_layouts, run_rope_layout_probe

        if not self._rope_layout_probe or not self._trial_user or self._session is None or self._child is not None:
            raise BackendError("RoPE layout measurement requires an idle, bound and protected Session")
        if _DISABLE_GRAPH not in self._server_args or revision["parameters"].get("decode_graph_enabled") is True:
            raise BackendError("RoPE layout evidence requires the frozen graph-disabled configuration")
        workspace = self._session / "workspaces" / f"{attempt}-rope-observer"
        prepared = self.prepare(revision["id"], workspace, kind="rope_layout")
        instance = None
        observation = None
        failure = None
        try:
            instance = self.launch(prepared, workload)
            identity = self._observation(instance, workload)
            lease = json.loads((workspace / _LEASE).read_bytes())
            workers = set(_lease_processes(lease)) | set(_live_group(lease))
            source = instance["source_overlay"]
            staged = next(row["staged_path"] for row in source["files"] if row["path"] == ROPE_MODEL_SOURCE)
            loaded = next(row for row in source["loaded_files"] if row["path"] == staged)
            if loaded["pid"] not in workers:
                raise BackendError("RoPE layout observer did not load in the owned SGLang service")
            observation = observe_rope_layouts(
                instance["base_url"], workload, control=workspace / "rope-layout.window",
                log=Path(instance["server_log"]), workers=workers, timeout=self._request_timeout,
            )
            self._observation(instance, workload)
            observation.update(identity=identity, loaded_instrumentation=loaded, server_log=instance["server_log"])
        except (BackendError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failure = str(error)
        finally:
            try:
                cleanup = self.cleanup(prepared, instance)
            except BackendError as error:
                cleanup = {"released": False, "reason": str(error)}
        if not cleanup["released"]:
            return {"status": "failed", "reason": "cleanup_unconfirmed", "observer_cleanup": cleanup}
        if observation is None or failure is not None:
            return {"status": "failed", "reason": "rope_layout_observation_failed", "detail": failure,
                    "observer_cleanup": cleanup}

        uid, gid = self._trial_user
        raw = run_rope_layout_probe(
            python=self._python, source_root=self._source, model_config=self._model / "config.json",
            workload=workload, profile=profile, revision=revision["id"],
            workspace=self._session / "workspaces" / f"{attempt}-rope-worker",
            environment=self._env, uid=uid, gid=gid, observed_layouts=observation["layouts"],
        )
        return {**raw, "observation": observation, "observer_cleanup": cleanup}

    def prepare(self, revision: str, workspace: Path, *, kind: str) -> Record:
        if revision not in self._stacks or kind not in {"framework", "model_forward", "rope_layout"}:
            raise BackendError("real SGLang prepare requires a bound revision and supported measurement kind")
        if kind == "model_forward" and self._forward_repeats is None:
            raise BackendError("real SGLang model-forward measurement is not enabled")
        if kind == "rope_layout" and not self._rope_layout_probe:
            raise BackendError("real SGLang RoPE layout measurement is not enabled")
        if self._child is not None:
            raise BackendError("an earlier SGLang instance still needs cleanup")
        workspace = workspace.resolve()
        if self._session is not None and not workspace.is_relative_to(self._session / "workspaces"):
            raise BackendError("SGLang workspace is outside this Session")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise BackendError(f"could not create SGLang workspace: {error}") from error
        prepared = {"mode": "real", "revision": revision, "kind": kind, "workspace": str(workspace)}
        patches = [item for item in self._stacks[revision]["framework"] if item["kind"] == "patch"]
        patches.extend(self._stacks[revision]["kernels"])
        allowed_source = self._allowed_source
        if kind == "model_forward":
            patches.append(forward_source_edit())
            allowed_source |= {FORWARD_SOURCE}
        elif kind == "rope_layout":
            from .sglang_rope import ROPE_MODEL_SOURCE, rope_layout_source_edit

            patches.append(rope_layout_source_edit())
            allowed_source |= {ROPE_MODEL_SOURCE}
        if patches:
            commit = self._source_revision()
            if commit != self._target.get("software", {}).get("sglang_commit"):
                raise BackendError("a source candidate requires the pinned SGLang commit")
            prepared["source_overlay"] = stage_overlay(self._source, workspace, patches, allowed_source, commit=commit)
        return prepared

    def _wait_ready(self, opener_url: str, expected: Mapping[str, Any], log: Path) -> Record:
        deadline = time.monotonic() + self._startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._child is None or self._child.poll() is not None:
                raise BackendError(f"SGLang stopped before readiness. See {log}")
            try:
                with _LOOPBACK.open(f"{opener_url}/health", timeout=4) as response:
                    response.read()
                with _LOOPBACK.open(f"{opener_url}/server_info", timeout=8) as response:
                    info = json.load(response)
            except (URLError, OSError, ValueError) as error:
                last_error = error
                time.sleep(0.4)
                continue
            if not isinstance(info, dict) or info.get("served_model_name") != expected["served_model_name"]:
                raise BackendError("loopback endpoint does not belong to the current SGLang launch")
            mismatches = {key: {"expected": value, "observed": info.get(key)}
                          for key, value in expected.items() if info.get(key) != value}
            if mismatches:
                raise BackendError(f"SGLang serving identity mismatch: {mismatches}. See {log}")
            keys = (*expected, "context_length", "disable_radix_cache", "chunked_prefill_size", "mem_fraction_static")
            return {key: info[key] for key in keys if key in info}
        raise BackendError(f"SGLang did not become ready within {self._startup_timeout:g}s: {last_error}. See {log}")

    def launch(self, prepared: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        if self._child is not None:
            raise BackendError("an earlier SGLang instance still needs cleanup")
        revision = prepared["revision"]
        stack = self._stacks[revision]
        if workload.get("dtype") != "bfloat16":
            raise BackendError("the real Ascend SGLang target requires workload.dtype=bfloat16")
        probe = self._probe if self._probe is not None else self.probe()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
            port_socket.bind((_HOST, 0))
            port = port_socket.getsockname()[1]
        nonce = uuid4().hex
        model_name = f"hyperloop-{nonce}"
        base_url = f"http://{_HOST}:{port}"
        workspace = Path(prepared["workspace"])
        log = workspace / "server.log"
        args, chunk = _chunk_args(self._server_args, stack["parameters"].get("chunked_prefill_size"))
        forward_only = prepared["kind"] == "model_forward"
        graph_expected = {}
        if "decode_graph_enabled" in stack["parameters"]:
            decode_graph = stack["parameters"]["decode_graph_enabled"]
            if forward_only and decode_graph:
                raise BackendError("model-forward timing requires decode graph to stay disabled")
            args, graph_expected = _decode_graph_args(args, decode_graph)
        if forward_only and _DISABLE_GRAPH not in args:
            args.append(_DISABLE_GRAPH)
        command = [str(self._python), "-u", "-m", "sglang.launch_server", "--host", _HOST, "--port", str(port),
                   "--model-path", str(self._model), "--tokenizer-path", str(self._tokenizer),
                   "--served-model-name", model_name, "--device", "npu", "--attention-backend", "ascend",
                   "--tp-size", "1", "--dtype", "bfloat16", *args]
        env, cwd = self._env, workspace
        source = prepared.get("source_overlay")
        stock_control = None
        if self._ascend_use_fia is not None:
            fia = stack["parameters"].get("ascend_use_fia", self._ascend_use_fia)
            stock_control = {"ASCEND_USE_FIA": "1" if fia else "0", "ASCEND_USE_FA": "0"}
            env = {**env, **stock_control}
        if self._source_sandbox is not None:
            command, env, cwd = sandbox_launch(command, env, self._source_sandbox, workspace, source)
        if self._trial_user is not None:
            protected_paths([workspace], *self._trial_user)
            command, env, cwd = dedicated_launch(command, env, workspace, source, *self._trial_user)
        if stack["kernels"]:
            # The generating /health probe would consume the first, request-scoped invocation marker.
            env = {**env, "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"}
        if forward_only:
            env = {**env, FORWARD_CONTROL_ENV: str(workspace / "model-forward.window")}
        elif prepared["kind"] == "rope_layout":
            from .sglang_rope import ROPE_CONTROL_ENV

            env = {**env, ROPE_CONTROL_ENV: str(workspace / "rope-layout.window")}
        lease = {"nonce": nonce, "state": "launching", "pid": None, "port": port,
                 "revision": revision, "launch_command": command}
        if self._trial_user is not None:
            lease["run_uid"] = self._trial_user[0]
        try:
            _write_lease(workspace, lease)
            with log.open("xb") as stdout:
                self._child = subprocess.Popen(command, env={**env, _LEASE_ENV: nonce}, cwd=cwd,
                                               stdout=stdout, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True,
                                               umask=0o027 if self._trial_user is not None else -1)
            lease["pid"] = self._child.pid
            if (identity := _process(self._child.pid)) is not None:
                lease["start_ticks"] = identity.start_ticks
            _write_lease(workspace, lease)
        except OSError as error:
            raise BackendError(f"could not start isolated SGLang: {error}. See {log}") from error
        expected = {
            "served_model_name": model_name, "model_path": str(self._model), "tokenizer_path": str(self._tokenizer),
            "host": _HOST, "port": port, "device": "npu", "dtype": "bfloat16", "tp_size": 1,
            "attention_backend": "ascend", "version": probe["environment"]["software"]["sglang"],
            **graph_expected,
        }
        if chunk is not None:
            expected["chunked_prefill_size"] = chunk
        if forward_only or prepared["kind"] == "rope_layout":
            expected["disable_cuda_graph"] = True
        loaded = self._wait_ready(base_url, expected, log)
        if stack["parameters"].get("decode_graph_enabled") is True:
            _verify_decode_graph_capture(log)
        observed = loaded_sources(log, source) if source else []
        self._active = {
            "mode": "real", "instance_id": nonce, "revision": revision, "loaded_revision": revision,
            "loaded_stack": deepcopy(stack), "rank_map": probe["environment"]["rank_map"],
            "environment_fingerprint": probe["environment_fingerprint"], "workload_fingerprint": fingerprint(workload),
            "workspace": str(workspace), "ready": True, "candidate_hit": True, "base_url": base_url,
            "pid": self._child.pid, "launch_command": command, "server_log": str(log), "server_info": loaded,
        }
        if self._source_sandbox is not None or self._trial_user is not None:
            self._active["writable_runtime"] = str(cwd)
        if source:
            self._active["source_overlay"] = {**source, "loaded_files": observed}
        if forward_only:
            self._active["model_forward"] = {
                "symbol": FORWARD_SYMBOL, "source": FORWARD_SOURCE, "repeats": self._forward_repeats,
                "graph_replay": False,
            }
        elif prepared["kind"] == "rope_layout":
            self._active["rope_layout_observer"] = True
        if stack["kernels"]:
            try:
                self._active["kernel_log_offset"] = log.stat().st_size
            except OSError as error:
                raise BackendError(f"could not read the native Forge log boundary: {error}") from error
            self._active["health_readiness_scope"] = "model_loaded_without_generation"
        if stock_control is not None:
            self._active["stock_attention_launch_control"] = stock_control
        return deepcopy(self._active)

    def _observation(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        if (self._active is None or instance != self._active or self._child is None or self._child.poll() is not None
                or fingerprint(workload) != instance["workload_fingerprint"]):
            raise BackendError("real SGLang observation requires the current owned service and frozen workload")
        keys = ("mode", "instance_id", "revision", "environment_fingerprint", "workload_fingerprint")
        return {key: self._active[key] for key in keys}

    def observe_outputs(self, instance: Mapping[str, Any], workload: Mapping[str, Any], *, scope: str) -> Record:
        from .sglang_measure import observe_outputs

        if scope != "service":
            raise BackendError("real SGLang output adapter supports service outputs only")
        identity = self._observation(instance, workload)
        results = observe_outputs(instance["base_url"], workload, timeout=self._request_timeout)
        self._observation(instance, workload)
        if instance["loaded_stack"]["kernels"]:
            results["kernel_activation"] = self._kernel_activation(instance)
        return {**results, **identity}

    def _kernel_activation(self, instance: Mapping[str, Any]) -> Record:
        """Confirm the first correctness workload reached the staged native kernel."""
        workspace = Path(instance["workspace"])
        kernel = instance["loaded_stack"]["kernels"][0]
        marker = kernel["invocation_marker"]
        try:
            lease = json.loads((workspace / _LEASE).read_text(encoding="utf-8"))
            owned = _lease_processes(lease)
            source = instance["source_overlay"]
            staged_path = next(item["staged_path"] for item in source["files"] if item["path"] == kernel["patch_path"])
            loaded = next(item for item in source["loaded_files"] if item["path"] == staged_path)
            # The dedicated scheduler's /proc environment can be unreadable even to the evaluator.
            loader_joined_lease = loaded["pid"] in _live_group(lease)
            with Path(instance["server_log"]).open("rb") as stream:
                stream.seek(instance["kernel_log_offset"])
                for line in stream:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(f"{marker} "):
                        continue
                    match = _FORGE_CALL.fullmatch(text.removeprefix(f"{marker} "))
                    if match is None:
                        continue
                    pid = int(match["pid"])
                    inherited = pid in owned
                    if inherited or (loader_joined_lease and pid == loaded["pid"]):
                        return {**match.groupdict(), "candidate_id": kernel["candidate_id"], "marker": marker,
                                "source_sha256": loaded["sha256"],
                                "process_binding": "inherited_lease" if inherited else "same_session_and_loaded_file"}
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            raise BackendError(f"could not verify native Forge invocation: {error}") from error
        raise BackendError(f"SGLang did not invoke the staged Forge kernel during correctness requests: {marker}")

    def benchmark(self, instance: Mapping[str, Any], workload: Mapping[str, Any], *, repeat: int, warmup: int) -> Record:
        from .sglang_measure import benchmark

        identity = self._observation(instance, workload)
        if instance.get("model_forward"):
            raise BackendError("the isolated model-forward service does not provide end-to-end timing")
        if instance.get("rope_layout_observer"):
            raise BackendError("the isolated RoPE layout observer does not provide end-to-end timing")
        results = benchmark(instance["base_url"], workload, repeat=repeat, warmup=warmup, timeout=self._request_timeout)
        self._observation(instance, workload)
        return {**results, **identity}

    def collect_profile(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        from .sglang_profile import collect_profile

        if not self._profile_enabled:
            raise BackendError("real SGLang NPU profile is disabled for this target")
        identity = self._observation(instance, workload)
        workspace = Path(instance["workspace"]).resolve()
        writable = (workspace if self._trial_user is not None
                    else Path(instance.get("writable_runtime", workspace)).resolve())
        if not writable.is_relative_to(workspace):
            raise BackendError("SGLang profile output must stay inside the owned workspace")
        software = (self._probe or self.probe())["environment"]["software"]
        output = collect_profile(
            instance["base_url"], workload, revision=instance["revision"],
            tool_version=f"SGLang {software['sglang']} / Torch NPU {software['torch_npu']}",
            directory=writable / "profile", evidence_root=self._session or workspace,
            request_timeout=self._request_timeout, export_timeout=self._profile_export_timeout,
            output_owner=self._trial_user, steps=self._profile_steps,
            offline_export=self._export_profile if self._profile_export_owner is not None else None,
        )
        self._observation(instance, workload)
        return {**output, **identity}

    def _export_profile(self, phase_directory: Path, _: str) -> tuple[Path, Record]:
        return export_capture(
            phase_directory, python=self._python, environment=self._env, owner=self._profile_export_owner,
            timeout=self._profile_export_timeout,
        )

    def device_timer(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        raise BackendError("real SGLang kernel timing is not available in this provider")

    def forward_timer(self, instance: Mapping[str, Any], workload: Mapping[str, Any]) -> Record:
        identity = self._observation(instance, workload)
        if self._forward_repeats is None or not instance.get("model_forward"):
            raise BackendError("real model-forward timing requires a dedicated measurement service")
        workspace = Path(instance["workspace"])
        try:
            lease = json.loads((workspace / _LEASE).read_text(encoding="utf-8"))
            source = instance["source_overlay"]
            staged = next(row["staged_path"] for row in source["files"] if row["path"] == FORWARD_SOURCE)
            loaded = next(row for row in source["loaded_files"] if row["path"] == staged)
            workers = set(_lease_processes(lease))
            if loaded["pid"] in _live_group(lease):
                workers.add(loaded["pid"])
            if not workers:
                raise BackendError("could not verify the active SGLang model-forward worker")
            samples = measure_forward(instance["base_url"], workload, repeats=self._forward_repeats,
                                      control=workspace / "model-forward.window", log=Path(instance["server_log"]),
                                      workers=workers, timeout=self._request_timeout)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            raise BackendError(f"could not confirm the owned SGLang model-forward measurement: {error}") from error
        self._observation(instance, workload)
        return {**samples, "loaded_instrumentation": loaded, **identity}

    def _cleanup_workspace(self, workspace: Path) -> Record:
        path = workspace / _LEASE
        try:
            if not path.exists():
                return {"released": self._child is None, "workspace": str(workspace), "reason": "no_launch_lease"}
            lease = json.loads(path.read_text(encoding="utf-8"))
            nonce = lease["nonce"]
            if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
                raise ValueError("invalid lease marker")
            if lease.get("run_uid") != (self._trial_user[0] if self._trial_user is not None else None):
                raise ValueError("SGLang lease user differs from the locked trial user")
            result = _stop_lease(lease)
            if result["released"]:
                lease["state"] = "released"
                _write_lease(workspace, lease)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BackendError(f"could not confirm SGLang cleanup in {workspace}: {error}") from error
        return {**result, "workspace": str(workspace)}

    def cleanup(self, prepared: Mapping[str, Any], instance: Mapping[str, Any] | None) -> Record:
        try:
            result = self._cleanup_workspace(Path(prepared["workspace"]))
            if result["released"]:
                if self._child is not None:
                    result["exit_code"] = self._child.wait(timeout=2)
                self._child = None
                self._active = None
            return result
        except subprocess.TimeoutExpired as error:
            raise BackendError("SGLang child did not exit after its lease was released") from error

    def recover_cleanup(self) -> Record:
        if self._session is None:
            return {"released": False, "reason": "SGLang provider has no bound Session for recovery"}
        attempts = []
        workspaces = self._session / "workspaces"
        leases = [*workspaces.glob(f"*/*/{_LEASE}"), *workspaces.glob(f"*/{_LEASE}")]
        for path in sorted(leases):
            if not path.resolve().is_relative_to(workspaces) or path.is_symlink():
                return {"released": False, "reason": f"invalid lease path: {path}", "attempts": attempts}
            try:
                lease = json.loads(path.read_text(encoding="utf-8"))
                if lease.get("state") != "released":
                    attempts.append(self.cleanup({"workspace": str(path.parent)}, None))
            except (BackendError, OSError, ValueError, AttributeError) as error:
                attempts.append({"released": False, "workspace": str(path.parent), "reason": str(error)})
        return {"released": all(item["released"] for item in attempts), "attempts": attempts}
