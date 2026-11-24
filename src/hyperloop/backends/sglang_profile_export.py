"""Re-export the same Ascend capture with the installed toolkit's owner."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess

from .runtime import BackendError, Record, fingerprint


_MAX_CAPTURE_BYTES = 2 * 1024**3
_EXPORT = (
    "import sys; from torch_npu.profiler.profiler import analyse; "
    "analyse(sys.argv[1], max_process_number=1)"
)


def exporter_owner(environment: Mapping[str, str], trial_user: tuple[int, int]) -> tuple[int, int, Path]:
    binary = shutil.which("msprof", path=environment.get("PATH", ""))
    if binary is None:
        raise BackendError("offline Ascend export requires msprof from the locked CANN environment")
    target = Path(binary).resolve(strict=True)
    permissions = target.stat()
    if not permissions.st_uid or permissions.st_uid == trial_user[0]:
        raise BackendError("offline Ascend export requires a separate nonroot toolkit owner")
    return permissions.st_uid, permissions.st_gid, target


def _stage_capture(capture: Path, output: Path, uid: int, gid: int) -> tuple[Path, Record]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(capture, flags)
    total = 0
    manifests = []
    try:
        captures = sorted(name for name in os.listdir(descriptor) if name.endswith("_ascend_pt"))
        if len(captures) != 1:
            raise BackendError(f"expected one raw Torch NPU capture, found {len(captures)}")
        raw = captures[0]
        for directory, folders, files, source_fd in os.fwalk(raw, dir_fd=descriptor, follow_symlinks=False):
            if directory == raw:
                folders[:] = [name for name in folders if name == "FRAMEWORK" or name.startswith("PROF_")]
            target = output / directory
            target.mkdir(mode=0o700)
            os.chown(target, uid, gid)
            for name in sorted(files):
                if directory == raw and not name.endswith(".json"):
                    continue
                path = target / name
                file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd)
                with os.fdopen(file_fd, "rb") as source:
                    state = os.fstat(source.fileno())
                    if not stat.S_ISREG(state.st_mode) or total + state.st_size > _MAX_CAPTURE_BYTES:
                        raise BackendError("unrecognized or oversized raw Ascend capture")
                    digest = sha256()
                    copied = 0
                    with path.open("xb") as staged:
                        while chunk := source.read(1024 * 1024):
                            copied += len(chunk)
                            if copied > state.st_size:
                                raise BackendError("raw Ascend capture changed during offline staging")
                            digest.update(chunk)
                            staged.write(chunk)
                    if copied != state.st_size:
                        raise BackendError("raw Ascend capture shrank during offline staging")
                os.chown(path, uid, gid)
                total += copied
                manifests.append({"path": path.relative_to(output / raw).as_posix(),
                                  "sha256": digest.hexdigest(), "bytes": copied})
    finally:
        os.close(descriptor)
    destination = output / raw
    if not (destination / "FRAMEWORK").is_dir() or not list(destination.glob("PROF_*")):
        raise BackendError("offline Ascend export requires the original FRAMEWORK and CANN records")
    manifests.sort(key=lambda row: row["path"])
    return destination, {"capture_name": raw, "raw_files": len(manifests), "raw_bytes": total,
                         "raw_manifest_sha256": fingerprint(manifests)}


def export_capture(
    phase: Path, *, python: Path, environment: Mapping[str, str], owner: tuple[int, int, Path], timeout: float,
) -> tuple[Path, Record]:
    uid, gid, binary = owner
    try:
        output = phase / "offline-export"
        output.mkdir(mode=0o700)
        os.chown(output, uid, gid)
        preserved, receipt = _stage_capture(phase / "capture", output, uid, gid)
        env = {name: value for name, value in environment.items()
               if name in {"PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL"}
               or name.startswith(("ASCEND_", "ATB_"))}
        env.update(HOME=str(output), TMPDIR=str(output), ASCEND_WORK_PATH=str(output))
        command = ["/usr/bin/setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", "--no-new-privs",
                   "--", str(python), "-I", "-c", _EXPORT, str(preserved)]
        log = phase / "offline-export.log"
        with log.open("xb") as stream:
            with subprocess.Popen(command, env=env, cwd=output, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    code = process.wait(timeout)
                except subprocess.TimeoutExpired as error:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    raise BackendError(f"offline Ascend export timed out. See {log}") from error
        result = preserved / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
        if code or not result.is_file():
            raise BackendError(f"offline Ascend export did not produce a trace (exit {code}). See {log}")
        return result, {**receipt, "exporter_uid": uid, "exporter_gid": gid, "msprof": str(binary),
                        "log": str(log), "exit_code": code}
    except OSError as error:
        raise BackendError(f"could not stage or export the original Ascend capture: {error}") from error
