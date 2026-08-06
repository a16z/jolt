from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


SEALED_BINARY_SCHEMA = "jolt_sealed_binary_v1"
SEALED_BINARY_SCHEMA_VERSION = 1
SEALED_BINARY_FILE = "runner"
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sealed_binary_token(binary_id: str) -> str:
    return f"{{sealed_binary:{binary_id}}}"


def _read_regular_file(
    path: Path,
    maximum_bytes: int,
    description: str,
    *,
    require_executable: bool = False,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{description} cannot be read: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{description} must be a regular file")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise ValueError(f"{description} size is invalid")
    if require_executable and before.st_mode & 0o111 == 0:
        raise ValueError(f"{description} is not executable")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ValueError(f"{description} changed while it was admitted")
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{description} changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ValueError(f"{description} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def declared_source_sha256(root: Path, source_paths: list[str]) -> str:
    entries = []
    for relative in sorted(source_paths):
        payload = _read_regular_file(
            root / relative,
            MAX_SOURCE_BYTES,
            f"sealed binary source {relative}",
        )
        entries.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
    return sha256(canonical_json(entries))


def prepare_sealed_binary_from_output(
    root: Path,
    binary_id: str,
    contract: dict[str, Any],
    expected_source_sha256: str,
    build_environment_sha256: str,
) -> dict[str, Any]:
    build = contract["build"]
    command = list(build["command"])
    source_paths = list(contract["source_paths"])
    binary = _read_regular_file(
        root / build["output_path"],
        MAX_BINARY_BYTES,
        f"sealed binary {binary_id} output",
        require_executable=True,
    )
    if declared_source_sha256(root, source_paths) != expected_source_sha256:
        raise ValueError(f"sealed binary {binary_id} sources changed after build")
    manifest = {
        "schema": SEALED_BINARY_SCHEMA,
        "schema_version": SEALED_BINARY_SCHEMA_VERSION,
        "id": binary_id,
        "binary_file": SEALED_BINARY_FILE,
        "binary_bytes": len(binary),
        "binary_sha256": sha256(binary),
        "source_sha256": expected_source_sha256,
        "build_command_sha256": sha256(canonical_json(command)),
        "build_environment_sha256": build_environment_sha256,
    }
    return {
        "artifact_sha256": sha256(canonical_json(manifest) + b"\0" + binary),
        "manifest": manifest,
        "binary": binary,
    }


def verify_sealed_binary_store(
    run_dir: Path, *, require_nonwritable: bool = False
) -> Path:
    binaries_dir = run_dir / "binaries"
    try:
        directory = binaries_dir.lstat()
    except OSError as error:
        raise ValueError("sealed binary store cannot be read") from error
    if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
        raise ValueError("sealed binary store must be a directory")
    if binaries_dir.resolve().parent != run_dir.resolve():
        raise ValueError("sealed binary store escapes the run directory")
    if require_nonwritable and directory.st_mode & 0o222:
        raise ValueError("sealed binary store must be nonwritable")
    return binaries_dir


def seal_sealed_binary_store(run_dir: Path) -> None:
    binaries_dir = verify_sealed_binary_store(run_dir)
    os.chmod(binaries_dir, 0o555)
    _fsync_directory(run_dir)
    verify_sealed_binary_store(run_dir, require_nonwritable=True)


def _validate_manifest(manifest: dict[str, Any], binary: bytes) -> None:
    fields = {
        "schema",
        "schema_version",
        "id",
        "binary_file",
        "binary_bytes",
        "binary_sha256",
        "source_sha256",
        "build_command_sha256",
        "build_environment_sha256",
    }
    if set(manifest) != fields:
        raise ValueError("sealed binary manifest fields are invalid")
    if (
        manifest["schema"] != SEALED_BINARY_SCHEMA
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SEALED_BINARY_SCHEMA_VERSION
        or manifest["binary_file"] != SEALED_BINARY_FILE
        or not isinstance(manifest["id"], str)
        or not manifest["id"]
        or type(manifest["binary_bytes"]) is not int
        or manifest["binary_bytes"] != len(binary)
        or manifest["binary_sha256"] != sha256(binary)
    ):
        raise ValueError("sealed binary manifest does not match its executable")
    for field in (
        "binary_sha256",
        "source_sha256",
        "build_command_sha256",
        "build_environment_sha256",
    ):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"sealed binary manifest {field} is invalid")


def verify_sealed_binary(artifact_dir: Path) -> dict[str, Any]:
    try:
        directory = artifact_dir.lstat()
    except OSError as error:
        raise ValueError("sealed binary directory cannot be read") from error
    if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
        raise ValueError("sealed binary path must be a directory")
    if directory.st_mode & 0o222:
        raise ValueError("sealed binary directory must be nonwritable")
    if {path.name for path in artifact_dir.iterdir()} != {
        "manifest.json",
        SEALED_BINARY_FILE,
    }:
        raise ValueError("sealed binary directory has unexpected files")

    manifest_path = artifact_dir / "manifest.json"
    runner_path = artifact_dir / SEALED_BINARY_FILE
    encoded_manifest = _read_regular_file(
        manifest_path, MAX_MANIFEST_BYTES, "sealed binary manifest"
    )
    binary = _read_regular_file(
        runner_path,
        MAX_BINARY_BYTES,
        "sealed binary executable",
        require_executable=True,
    )
    if manifest_path.lstat().st_mode & 0o222:
        raise ValueError("sealed binary manifest must be nonwritable")
    if runner_path.lstat().st_mode & 0o222:
        raise ValueError("sealed binary executable must be nonwritable")
    try:
        manifest = json.loads(encoded_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sealed binary manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or encoded_manifest != canonical_json(manifest):
        raise ValueError("sealed binary manifest is not canonical")
    _validate_manifest(manifest, binary)
    artifact_sha256 = sha256(canonical_json(manifest) + b"\0" + binary)
    if artifact_dir.name != artifact_sha256:
        raise ValueError("sealed binary directory does not match its digest")
    return {
        "artifact_sha256": artifact_sha256,
        "manifest": manifest,
    }


def _write_synced(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("sealed binary write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_sealed_binary(
    run_dir: Path, prepared: dict[str, Any]
) -> dict[str, Any]:
    artifact_sha256 = prepared["artifact_sha256"]
    manifest = prepared["manifest"]
    binary = prepared["binary"]
    _validate_manifest(manifest, binary)
    if artifact_sha256 != sha256(canonical_json(manifest) + b"\0" + binary):
        raise ValueError("prepared sealed binary digest is invalid")
    binaries_dir = verify_sealed_binary_store(run_dir)
    target = binaries_dir / artifact_sha256
    if target.exists():
        record = verify_sealed_binary(target)
        record["artifact_path"] = target.relative_to(run_dir).as_posix()
        return record

    temporary = Path(tempfile.mkdtemp(prefix=".binary-", dir=binaries_dir))
    try:
        _write_synced(temporary / SEALED_BINARY_FILE, binary, 0o555)
        _write_synced(
            temporary / "manifest.json", canonical_json(manifest), 0o444
        )
        os.chmod(temporary, 0o555)
        _fsync_directory(temporary)
        try:
            temporary.rename(target)
        except FileExistsError:
            os.chmod(temporary, 0o755)
        _fsync_directory(binaries_dir)
    finally:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary)
    record = verify_sealed_binary(target)
    record["artifact_path"] = target.relative_to(run_dir).as_posix()
    return record


def verify_sealed_binary_record(
    run_dir: Path, binary_id: str, record: dict[str, Any]
) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "artifact_sha256",
        "artifact_path",
        "manifest",
    }:
        raise ValueError(f"sealed binary {binary_id} record is invalid")
    relative = Path(record["artifact_path"])
    if relative.parts != ("binaries", record["artifact_sha256"]):
        raise ValueError(f"sealed binary {binary_id} path is invalid")
    observed = verify_sealed_binary(run_dir / relative)
    sealed = dict(record)
    sealed.pop("artifact_path")
    if canonical_json(observed) != canonical_json(sealed):
        raise ValueError(f"sealed binary {binary_id} changed")
    if observed["manifest"]["id"] != binary_id:
        raise ValueError(f"sealed binary {binary_id} manifest id is invalid")
    return run_dir / relative / SEALED_BINARY_FILE


def verify_sealed_binary_contract(
    root: Path,
    run_dir: Path,
    binary_id: str,
    contract: dict[str, Any],
    record: dict[str, Any],
) -> Path:
    runner = verify_sealed_binary_record(run_dir, binary_id, record)
    manifest = record["manifest"]
    if manifest["build_command_sha256"] != sha256(
        canonical_json(contract["build"]["command"])
    ):
        raise ValueError(f"sealed binary {binary_id} build contract changed")
    observed_source = declared_source_sha256(root, contract["source_paths"])
    if manifest["source_sha256"] != observed_source:
        raise ValueError(f"sealed binary {binary_id} declared sources changed")
    return runner
