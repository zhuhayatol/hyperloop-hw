"""Stage an allow-listed SGLang source change and verify which file was loaded."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from difflib import unified_diff
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any

from .runtime import BackendError, Record


_PACKAGE = Path("python", "sglang")
_MAX_EDIT_BYTES = 16_384
_EDIT_PATH = re.compile(r"python/sglang/srt/(?:[a-zA-Z_][a-zA-Z_0-9]*/)*[a-zA-Z_][a-zA-Z_0-9]*\.py")
_IMPORT_MARKER = "HYPERLOOP_SOURCE_LOAD="
_MODULE_GUARD = '''import hashlib
import json
import os
import sys


def _install():
    watched = json.loads(os.environ["HYPERLOOP_SGLANG_SOURCE_FILES"])
    seen = set()

    def record_import(event, args):
        if event != "exec" or not args:
            return
        filename = getattr(args[0], "co_filename", None)
        if filename not in watched or filename in seen:
            return
        with open(filename, "rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != watched[filename]:
            raise RuntimeError(f"SGLang staged source digest mismatch: {filename}")
        seen.add(filename)
        print("HYPERLOOP_SOURCE_LOAD=" + json.dumps({"path": filename, "sha256": digest, "pid": os.getpid()}),
              flush=True)

    sys.addaudithook(record_import)


_install()
'''


def source_path(raw: Any) -> str:
    if not isinstance(raw, str) or _EDIT_PATH.fullmatch(raw) is None or raw.endswith("/__init__.py"):
        raise BackendError("a SGLang source path must name a Python file under python/sglang/srt")
    return raw


def source_edit(entry: Mapping[str, Any], allowed_paths: Collection[str]) -> tuple[str, bytes, bytes]:
    path = source_path(entry.get("patch_path"))
    if path not in allowed_paths:
        raise BackendError(f"SGLang source change is outside the target allow-list: {path}")
    edit = entry.get("source_edit")
    if not isinstance(edit, Mapping) or edit.keys() != {"before", "after"}:
        raise BackendError("a source change requires one exact before/after block")
    before, after = edit["before"], edit["after"]
    if (not isinstance(before, str) or not isinstance(after, str) or not before.strip() or before == after
            or any("\0" in value or "\r" in value for value in (before, after))):
        raise BackendError("SGLang source edit must change a non-empty LF-delimited UTF-8 anchor")
    try:
        old, new = before.encode("utf-8"), after.encode("utf-8")
    except UnicodeError as error:
        raise BackendError("SGLang source edit is not valid UTF-8") from error
    if max(len(old), len(new)) > _MAX_EDIT_BYTES:
        raise BackendError("SGLang source edit exceeds the size limit")
    return path, old, new


def stage_overlay(
    source: Path, workspace: Path, entries: list[Record], allowed_paths: Collection[str], *, commit: str,
) -> Record:
    """Use a fresh package copy and apply the accepted stack in revision order."""
    package = source / _PACKAGE
    staged = workspace / "source"
    patch_file = workspace / "source.patch"
    staged_package = staged / _PACKAGE
    edits = []
    originals: dict[str, bytes] = {}
    try:
        for entry in entries:
            raw, before, after = source_edit(entry, allowed_paths)
            original = source.joinpath(*PurePosixPath(raw).parts)
            if original.resolve(strict=True) != original or not original.is_file():
                raise BackendError(f"SGLang source path must refer to a regular file: {raw}")
            edits.append((entry["candidate_id"], raw, before, after))
            if raw not in originals:
                originals[raw] = original.read_bytes()
        shutil.copytree(package, staged_package, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        files = dict(originals)
        changes = []
        for candidate_id, raw, before, after in edits:
            current = files[raw]
            if current.count(before) != 1:
                raise BackendError(f"SGLang edit anchor is missing or ambiguous in {raw} for {candidate_id}")
            changed = current.replace(before, after, 1)
            compile(changed, raw, "exec")
            files[raw] = changed
            changes.append({"candidate_id": candidate_id, "path": raw, "before_sha256": _sha(current),
                            "after_sha256": _sha(changed)})
        diff = []
        observed_files = []
        for raw, changed in files.items():
            path = staged.joinpath(*PurePosixPath(raw).parts)
            if path.read_bytes() != originals[raw]:
                raise BackendError(f"the SGLang source changed during preparation: {raw}")
            if changed == originals[raw]:
                raise BackendError(f"the final SGLang edit reverts the source to the pinned version: {raw}")
            path.write_bytes(changed)
            diff.extend(unified_diff(originals[raw].decode().splitlines(keepends=True),
                                     changed.decode().splitlines(keepends=True), f"a/{raw}", f"b/{raw}"))
            observed_files.append({"path": raw, "staged_path": str(path), "pinned_sha256": _sha(originals[raw]),
                                   "expected_sha256": _sha(changed)})
        patch_file.write_bytes("".join(diff).encode("utf-8"))
        (staged / "python" / "sitecustomize.py").write_text(_MODULE_GUARD, encoding="utf-8")
    except (OSError, UnicodeError, SyntaxError) as error:
        raise BackendError(f"could not stage the SGLang source overlay: {error}") from error
    return {"source_commit": commit, "package_parent": str(staged / "python"), "changes": changes,
            "files": observed_files, "patch_artifact": str(patch_file), "patch_sha256": _sha(patch_file.read_bytes()),
            "unified_diff": "".join(diff)}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _private_mounts(environment: Mapping[str, str], runtime: Path) -> list[str]:
    home = Path(environment.get("HOME") or Path.home())
    directories = {home / ".config" / "hyperloop"}
    if config := environment.get("XDG_CONFIG_HOME"):
        directories.add(Path(config) / "hyperloop")
    mask = []
    mounted = set()
    for directory in sorted(directories):
        if not directory.is_absolute() or not directory.is_dir():
            continue
        target = directory.resolve()
        if runtime.is_relative_to(target):
            raise BackendError("candidate runtime overlaps a private credential directory")
        if target not in mounted:
            mask.extend(("--tmpfs", str(target)))
            mounted.add(target)
        key = directory / "deepseek.key"
        if key.is_file():
            resolved_key = key.resolve()
            if not resolved_key.is_relative_to(target) and resolved_key not in mounted:
                mask.extend(("--ro-bind", "/dev/null", str(resolved_key)))
                mounted.add(resolved_key)
    return mask


def sandbox_launch(
    command: list[str], environment: Mapping[str, str], sandbox: Path, workspace: Path,
    source: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, str], Path]:
    """Permit temporary output writes while the source, model and evidence remain read-only."""
    runtime = workspace / "runtime"
    try:
        runtime.mkdir(exist_ok=False)
        (runtime / "cache").mkdir()
    except OSError as error:
        raise BackendError(f"could not create writable candidate runtime directory: {error}") from error

    allowed = {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LANG", "LC_ALL", "TZ", "HF_HUB_OFFLINE",
               "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"}
    prefixes = ("ASCEND_", "ATB_", "HCCL_", "OMP_", "MKL_", "LC_", "TRITON_", "TORCH_", "NPU_")
    child_env = {name: value for name, value in environment.items()
                 if name in allowed or name.startswith(prefixes) and not re.search(r"KEY|TOKEN|SECRET", name)}
    cache = runtime / "cache"
    child_env.update({"HOME": str(runtime), "TMPDIR": "/tmp", "XDG_CACHE_HOME": str(cache),
                      "TRITON_HOME": str(runtime), "TRITON_CACHE_DIR": str(cache / "triton"),
                      "TRITON_OVERRIDE_DIR": str(cache / "triton-override"), "TRITON_DUMP_DIR": str(cache / "triton-dump"),
                      "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "PYTHONNOUSERSITE": "1"})
    locations = [source["package_parent"]] if source is not None else []
    if original := environment.get("PYTHONPATH"):
        locations.append(original)
    if locations:
        child_env["PYTHONPATH"] = os.pathsep.join(locations)
    if source is not None:
        child_env["HYPERLOOP_SGLANG_SOURCE_FILES"] = json.dumps(
            {file["staged_path"]: file["expected_sha256"] for file in source["files"]})
    return [str(sandbox), "--unshare-user", "--unshare-pid", "--cap-drop", "ALL", "--die-with-parent", "--ro-bind", "/", "/",
            "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", *_private_mounts(environment, runtime),
            "--bind", str(runtime),
            str(runtime), "--chdir", str(runtime), "--", *command], child_env, runtime


def loaded_sources(log: Path, source: Mapping[str, Any]) -> list[Record]:
    """Require an observed Python execution record for every intended source file."""
    expected = {file["staged_path"]: file["expected_sha256"] for file in source["files"]}
    imported = {}
    try:
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if _IMPORT_MARKER not in line:
                    continue
                record = json.loads(line.split(_IMPORT_MARKER, 1)[1])
                path, digest, pid = record["path"], record["sha256"], record["pid"]
                if expected.get(path) != digest or type(pid) is not int or pid <= 0:
                    raise BackendError(f"SGLang loaded a source file outside the prepared overlay: {record}")
                imported[path] = record
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BackendError(f"could not verify the loaded SGLang source files: {error}") from error
    missing = expected.keys() - imported.keys()
    if missing:
        raise BackendError(f"no activation evidence for the changed SGLang files: {sorted(missing)}")
    return [imported[path] for path in expected]
