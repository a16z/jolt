from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


OUTER_ARTIFACT_SCHEMA = "jolt_outer_artifact_v2"
OUTER_ARTIFACT_SCHEMA_VERSION = 2
OUTER_SOURCE_FILE = "outer.metal"
MAX_OUTER_SOURCE_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024

_COMMON_ENTRYPOINTS = {
    "dense_bind": "solinas_outer_remainder_bind_and_message",
    "reduction": "solinas_outer_remainder_reduce_columns",
}
OUTER_BINDING_PLANS = {
    "b_only_v1": {
        **_COMMON_ENTRYPOINTS,
        "materialize": "solinas_outer_remainder_materialize_b_and_message",
        "opening": "solinas_outer_remainder_opening_tiles",
        "stream_bind": "solinas_outer_remainder_collapsed_a_stream_bind",
    },
    "b_only_padded_56_v1": {
        **_COMMON_ENTRYPOINTS,
        "materialize": "solinas_outer_remainder_materialize_b_and_message",
        "opening": "solinas_outer_remainder_opening_tiles_padded_56",
        "stream_bind": "solinas_outer_remainder_collapsed_a_stream_bind",
    },
}
OUTER_DISPATCH_PARAMETERS = {
    "materialize_threads": "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS",
    "transition_threads": "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS",
    "opening_threads": "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS",
    "cutoff_log2": "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2",
    "trace_cutoff_log2": "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2",
}
_TRIGRAPH_SUFFIXES = frozenset(b"=/'()!<>-")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular_file(path: Path, maximum_bytes: int, description: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{description} cannot be read: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{description} must be a regular file")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise ValueError(f"{description} size is invalid")

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


def verify_artifact_store(run_dir: Path) -> Path:
    artifacts_dir = run_dir / "artifacts"
    try:
        artifacts_stat = artifacts_dir.lstat()
    except OSError as error:
        raise ValueError("artifact store cannot be read") from error
    if not stat.S_ISDIR(artifacts_stat.st_mode) or stat.S_ISLNK(
        artifacts_stat.st_mode
    ):
        raise ValueError("artifact store must be a directory")
    if artifacts_dir.resolve().parent != run_dir.resolve():
        raise ValueError("artifact store escapes the run directory")
    return artifacts_dir


def _plan_manifest(binding_plan: str) -> dict[str, Any]:
    entrypoints = OUTER_BINDING_PLANS.get(binding_plan)
    if entrypoints is None:
        raise ValueError(f"unsupported Outer binding plan: {binding_plan}")
    layout = {
        "b_only_v1": {
            "row_stride_words": 20,
            "source_row_words": 20,
            "tile_rows": 64,
            "uses_shard_sums": True,
        },
        "b_only_padded_56_v1": {
            "row_stride_words": 21,
            "source_row_words": 20,
            "tile_rows": 56,
            "uses_shard_sums": False,
        },
    }[binding_plan]
    abi = {
        "opening_layout": layout,
        "required_entrypoints": entrypoints,
    }
    return {
        "binding_plan": binding_plan,
        "binding_plan_sha256": sha256(binding_plan.encode()),
        "opening_layout": layout,
        "opening_layout_sha256": sha256(canonical_json(layout)),
        "outer_abi_sha256": sha256(canonical_json(abi)),
        "required_entrypoints": entrypoints,
    }


def outer_dispatch_from_params(params: dict[str, Any]) -> dict[str, Any]:
    try:
        values = {
            field: int(params[parameter])
            for field, parameter in OUTER_DISPATCH_PARAMETERS.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Outer dispatch parameters are incomplete") from error
    for field in ("materialize_threads", "transition_threads", "opening_threads"):
        value = values[field]
        if value < 32 or value > 1024 or value & (value - 1):
            raise ValueError(f"Outer dispatch {field} is invalid")
    for field in ("cutoff_log2", "trace_cutoff_log2"):
        if not 1 <= values[field] <= 62:
            raise ValueError(f"Outer dispatch {field} is invalid")
    return {
        "materialize_threads": values["materialize_threads"],
        "stream_bind_threads": values["transition_threads"],
        "transition_threads": values["transition_threads"],
        "opening_threads": values["opening_threads"],
        "max_threadgroups": 8192,
        "cpu_tail_elements": 1 << values["cutoff_log2"],
        "trace_cutoff_elements": 1 << values["trace_cutoff_log2"],
        "storage_initialization": "full",
    }


def _validate_dispatch(dispatch: dict[str, Any]) -> None:
    fields = {
        "materialize_threads",
        "stream_bind_threads",
        "transition_threads",
        "opening_threads",
        "max_threadgroups",
        "cpu_tail_elements",
        "trace_cutoff_elements",
        "storage_initialization",
    }
    if not isinstance(dispatch, dict) or set(dispatch) != fields:
        raise ValueError("Outer dispatch contract is invalid")
    if dispatch["storage_initialization"] != "full":
        raise ValueError("Outer dispatch storage initialization is invalid")
    for field in fields - {"storage_initialization"}:
        value = dispatch[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Outer dispatch {field} is invalid")
    for field in (
        "materialize_threads",
        "stream_bind_threads",
        "transition_threads",
        "opening_threads",
    ):
        value = dispatch[field]
        if value < 32 or value > 1024 or value & (value - 1):
            raise ValueError(f"Outer dispatch {field} is invalid")
    if dispatch["stream_bind_threads"] != dispatch["transition_threads"]:
        raise ValueError("Outer stream and transition thread counts must match")
    if dispatch["max_threadgroups"] != 8192:
        raise ValueError("Outer max threadgroups is not allowlisted")
    for field in ("cpu_tail_elements", "trace_cutoff_elements"):
        value = dispatch[field]
        if value < 2 or value & (value - 1):
            raise ValueError(f"Outer dispatch {field} is invalid")


def _manifest(
    source: bytes, binding_plan: str, dispatch: dict[str, Any]
) -> dict[str, Any]:
    _validate_dispatch(dispatch)
    return {
        "schema": OUTER_ARTIFACT_SCHEMA,
        "schema_version": OUTER_ARTIFACT_SCHEMA_VERSION,
        "source_file": OUTER_SOURCE_FILE,
        "source_bytes": len(source),
        "outer_source_sha256": sha256(source),
        "dispatch": dispatch,
        "dispatch_sha256": sha256(canonical_json(dispatch)),
        **_plan_manifest(binding_plan),
    }


def _validate_source(source: bytes, description: str) -> None:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8") from error
    if b"\0" in source:
        raise ValueError(f"{description} cannot contain NUL bytes")
    if (
        source.startswith(b"\xef\xbb\xbf")
        or b"\r" in source
        or b'"' in source
        or b"'" in source
    ):
        raise ValueError(f"{description} is not in canonical source form")
    if any(
        source[index : index + 2] == b"??"
        and source[index + 2] in _TRIGRAPH_SUFFIXES
        for index in range(max(0, len(source) - 2))
    ):
        raise ValueError(f"{description} cannot use preprocessor trigraphs")
    del text

    spliced = source.replace(b"\\\n", b"")
    normalized = bytearray()
    index = 0
    while index < len(spliced):
        if spliced[index : index + 2] == b"/*":
            index += 2
            while index < len(spliced) and spliced[index : index + 2] != b"*/":
                index += 1
            index = min(index + 2, len(spliced))
        elif spliced[index : index + 2] == b"//":
            index += 2
            while index < len(spliced) and spliced[index] != ord("\n"):
                index += 1
        elif spliced[index : index + 2] == b"%:":
            normalized.append(ord("#"))
            index += 2
        else:
            normalized.append(spliced[index])
            index += 1

    if b"__has_include" in normalized or b"__has_embed" in normalized:
        raise ValueError(f"{description} cannot include external source")
    for line in normalized.split(b"\n"):
        directive = line.lstrip().removeprefix(b"#").lstrip()
        if line.lstrip().startswith(b"#") and directive.startswith(
            (b"include", b"import", b"embed")
        ):
            raise ValueError(f"{description} cannot include external source")


def _artifact_sha256(manifest: dict[str, Any], source: bytes) -> str:
    return sha256(canonical_json(manifest) + b"\0" + source)


def verify_outer_artifact(artifact_dir: Path) -> dict[str, Any]:
    try:
        directory = artifact_dir.lstat()
    except OSError as error:
        raise ValueError("Outer artifact directory cannot be read") from error
    if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
        raise ValueError("Outer artifact path must be a directory")
    if {path.name for path in artifact_dir.iterdir()} != {
        "manifest.json",
        OUTER_SOURCE_FILE,
    }:
        raise ValueError("Outer artifact directory has unexpected files")

    encoded_manifest = _read_regular_file(
        artifact_dir / "manifest.json", MAX_MANIFEST_BYTES, "Outer manifest"
    )
    try:
        manifest = json.loads(encoded_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Outer artifact manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Outer artifact manifest must be an object")
    if encoded_manifest != canonical_json(manifest):
        raise ValueError("Outer artifact manifest is not canonical")

    source = _read_regular_file(
        artifact_dir / OUTER_SOURCE_FILE,
        MAX_OUTER_SOURCE_BYTES,
        "Outer source",
    )
    _validate_source(source, "Outer source")
    binding_plan = manifest.get("binding_plan")
    if not isinstance(binding_plan, str):
        raise ValueError("Outer artifact binding plan is invalid")
    dispatch = manifest.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("Outer artifact dispatch is invalid")
    expected = _manifest(source, binding_plan, dispatch)
    if manifest != expected:
        raise ValueError("Outer artifact manifest does not match its source and ABI")
    artifact_sha256 = _artifact_sha256(manifest, source)
    if artifact_dir.name != artifact_sha256:
        raise ValueError("Outer artifact directory does not match its digest")
    return {
        "artifact_sha256": artifact_sha256,
        "manifest": manifest,
    }


def _write_synced(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("artifact write made no progress")
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


def materialize_outer_artifact(
    run_dir: Path,
    source_path: Path,
    binding_plan: str,
    dispatch: dict[str, Any],
) -> dict[str, Any]:
    source = _read_regular_file(
        source_path, MAX_OUTER_SOURCE_BYTES, "Outer candidate source"
    )
    _validate_source(source, "Outer candidate source")
    manifest = _manifest(source, binding_plan, dispatch)
    artifact_sha256 = _artifact_sha256(manifest, source)
    artifacts_dir = verify_artifact_store(run_dir)
    target = artifacts_dir / artifact_sha256
    if target.exists():
        record = verify_outer_artifact(target)
        record["artifact_path"] = target.relative_to(run_dir).as_posix()
        return record

    temporary = Path(tempfile.mkdtemp(prefix=".outer-", dir=artifacts_dir))
    try:
        _write_synced(temporary / OUTER_SOURCE_FILE, source)
        _write_synced(temporary / "manifest.json", canonical_json(manifest))
        _fsync_directory(temporary)
        try:
            temporary.rename(target)
        except FileExistsError:
            pass
        _fsync_directory(artifacts_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    record = verify_outer_artifact(target)
    record["artifact_path"] = target.relative_to(run_dir).as_posix()
    return record


RuntimeArtifactContext = dict[str, Any]
RuntimeArtifactMaterializer = Callable[
    [
        Path,
        dict[str, Any],
        Path,
        dict[str, str],
        Path,
        dict[str, str],
    ],
    tuple[dict[str, str], RuntimeArtifactContext],
]


@dataclass(frozen=True)
class _RuntimeArtifactHandler:
    result_adapter: str
    controller_paths: tuple[str, ...]
    validate_contract: Callable[
        [
            Path,
            dict[str, Any],
            set[str],
            dict[str, Any],
            dict[str, Any],
        ],
        None,
    ]
    materialize: RuntimeArtifactMaterializer
    verify: Callable[[Path, dict[str, Any]], None]
    validate_output: Callable[[dict[str, Any], dict[str, Any]], None]


def _validate_outer_runtime_artifact_contract(
    root: Path,
    contract: dict[str, Any],
    editable: set[str],
    search_space: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    fields = {
        "kind",
        "source_path",
        "plan_parameter",
        "plans",
        "tier_id",
    }
    if set(contract) != fields:
        raise ValueError("runtime artifact contract is invalid")
    source_path = contract["source_path"]
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("runtime artifact source path is invalid")
    relative = Path(source_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not (root / relative).is_file()
    ):
        raise ValueError("runtime artifact source path does not exist")
    plan_parameter = contract["plan_parameter"]
    plans = contract["plans"]
    if (
        editable != {source_path}
        or not isinstance(plan_parameter, str)
        or plan_parameter not in search_space
        or not isinstance(plans, list)
        or plans != list(OUTER_BINDING_PLANS)
        or search_space[plan_parameter] != plans
        or baseline[plan_parameter] not in plans
    ):
        raise ValueError("runtime artifact source or plans are not closed")


def _materialize_outer_runtime_artifact_context(
    run_dir: Path,
    contract: dict[str, Any],
    parent_root: Path,
    parent_params: dict[str, str],
    candidate_root: Path,
    candidate_params: dict[str, str],
) -> tuple[dict[str, str], RuntimeArtifactContext]:
    source_path = Path(contract["source_path"])
    plan_parameter = contract["plan_parameter"]
    parent = materialize_outer_artifact(
        run_dir,
        parent_root / source_path,
        str(parent_params[plan_parameter]),
        outer_dispatch_from_params(parent_params),
    )
    candidate = materialize_outer_artifact(
        run_dir,
        candidate_root / source_path,
        str(candidate_params[plan_parameter]),
        outer_dispatch_from_params(candidate_params),
    )
    return (
        {
            "JOLT_AUTORESEARCH_PARENT_ARTIFACT": str(
                (run_dir / parent["artifact_path"]).resolve()
            ),
            "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT": str(
                (run_dir / candidate["artifact_path"]).resolve()
            ),
        },
        {
            "kind": contract["kind"],
            "parent": parent,
            "candidate": candidate,
        },
    )


def _verify_outer_runtime_artifact_context(
    run_dir: Path, context: dict[str, Any]
) -> None:
    if set(context) != {"kind", "parent", "candidate"}:
        raise ValueError("runtime artifact context is invalid")
    verify_artifact_store(run_dir)
    for role in ("parent", "candidate"):
        expected = context[role]
        if not isinstance(expected, dict) or set(expected) != {
            "artifact_sha256",
            "artifact_path",
            "manifest",
        }:
            raise ValueError(f"{role} runtime artifact record is invalid")
        relative = Path(expected["artifact_path"])
        if relative.parts != (
            "artifacts",
            expected["artifact_sha256"],
        ):
            raise ValueError(f"{role} runtime artifact path is invalid")
        observed = verify_outer_artifact(run_dir / relative)
        sealed = dict(expected)
        sealed.pop("artifact_path")
        if canonical_json(observed) != canonical_json(sealed):
            raise ValueError(f"{role} runtime artifact changed")


def _validate_outer_runtime_artifact_output(
    output: dict[str, Any], context: dict[str, Any]
) -> None:
    fingerprint = output.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("runtime artifact fingerprint is missing")
    expected = {
        "parent_artifact_sha256": context["parent"]["artifact_sha256"],
        "candidate_artifact_sha256": context["candidate"][
            "artifact_sha256"
        ],
    }
    if any(fingerprint.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "runtime artifact fingerprint does not match the controller"
        )
    if output.get("schema") != "outer_remainder_successor_v2":
        return

    parent_manifest = context["parent"]["manifest"]
    candidate_manifest = context["candidate"]["manifest"]
    telemetry = output.get("telemetry", {})
    if (
        parent_manifest["dispatch"]["cpu_tail_elements"]
        != candidate_manifest["dispatch"]["cpu_tail_elements"]
        or parent_manifest["dispatch"]["trace_cutoff_elements"]
        != candidate_manifest["dispatch"]["trace_cutoff_elements"]
        or telemetry.get("parent_binding_plan")
        != parent_manifest["binding_plan"]
        or telemetry.get("candidate_binding_plan")
        != candidate_manifest["binding_plan"]
        or telemetry.get("parent_source_sha256")
        != parent_manifest["outer_source_sha256"]
        or telemetry.get("candidate_source_sha256")
        != candidate_manifest["outer_source_sha256"]
    ):
        raise ValueError(
            "runtime artifact telemetry does not match the controller"
        )
    expected_tail = parent_manifest["dispatch"]["cpu_tail_elements"]
    warmup = output.get("excluded_warmup")
    samples = output.get("samples")
    if not isinstance(warmup, dict) or not isinstance(samples, list):
        raise ValueError("runtime artifact arm evidence is missing")
    records = [warmup] + samples
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get(role), dict)
        or record[role].get("tail_elements") != expected_tail
        for record in records
        for role in ("parent", "candidate")
    ):
        raise ValueError(
            "runtime artifact CPU tail does not match the controller"
        )


_RUNTIME_ARTIFACT_HANDLERS = {
    "outer_msl_v1": _RuntimeArtifactHandler(
        result_adapter="outer_remainder_successor_v2",
        controller_paths=("scripts/metal_research/artifacts.py",),
        validate_contract=_validate_outer_runtime_artifact_contract,
        materialize=_materialize_outer_runtime_artifact_context,
        verify=_verify_outer_runtime_artifact_context,
        validate_output=_validate_outer_runtime_artifact_output,
    )
}


def _runtime_artifact_handler(kind: Any) -> _RuntimeArtifactHandler:
    if not isinstance(kind, str):
        raise ValueError("runtime artifact kind is invalid")
    try:
        return _RUNTIME_ARTIFACT_HANDLERS[kind]
    except KeyError as error:
        raise ValueError(f"unsupported runtime artifact kind: {kind}") from error


def validate_runtime_artifact_contract(
    root: Path,
    contract: dict[str, Any],
    editable: set[str],
    search_space: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    if not isinstance(contract, dict):
        raise ValueError("runtime artifact contract is invalid")
    handler = _runtime_artifact_handler(contract.get("kind"))
    handler.validate_contract(root, contract, editable, search_space, baseline)


def runtime_artifact_result_adapter(contract: dict[str, Any]) -> str:
    if not isinstance(contract, dict):
        raise ValueError("runtime artifact contract is invalid")
    return _runtime_artifact_handler(contract.get("kind")).result_adapter


def runtime_artifact_result_adapters() -> set[str]:
    return {
        handler.result_adapter
        for handler in _RUNTIME_ARTIFACT_HANDLERS.values()
    }


def runtime_artifact_controller_paths(
    contract: dict[str, Any],
) -> tuple[str, ...]:
    if not isinstance(contract, dict):
        raise ValueError("runtime artifact contract is invalid")
    return _runtime_artifact_handler(contract.get("kind")).controller_paths


def materialize_runtime_artifact_context(
    run_dir: Path,
    contract: dict[str, Any],
    parent_root: Path,
    parent_params: dict[str, str],
    candidate_root: Path,
    candidate_params: dict[str, str],
) -> tuple[dict[str, str], RuntimeArtifactContext]:
    handler = _runtime_artifact_handler(contract.get("kind"))
    return handler.materialize(
        run_dir,
        contract,
        parent_root,
        parent_params,
        candidate_root,
        candidate_params,
    )


def verify_runtime_artifact_context(
    run_dir: Path,
    expected_kind: Optional[str],
    context: Optional[RuntimeArtifactContext],
) -> None:
    if context is None:
        if expected_kind is not None:
            raise ValueError("required runtime artifact context is missing")
        return
    if not isinstance(context, dict):
        raise ValueError("runtime artifact context is invalid")
    if expected_kind is None:
        raise ValueError("runtime artifact context is unexpected")
    handler = _runtime_artifact_handler(expected_kind)
    if context.get("kind") != expected_kind:
        raise ValueError(
            "runtime artifact context kind does not match the sealed template"
        )
    handler.verify(run_dir, context)


def validate_runtime_artifact_output(
    output: dict[str, Any],
    expected_kind: Optional[str],
    context: Optional[RuntimeArtifactContext],
) -> None:
    if context is None:
        if expected_kind is not None:
            raise ValueError("required runtime artifact context is missing")
        return
    if not isinstance(context, dict):
        raise ValueError("runtime artifact context is invalid")
    if expected_kind is None:
        raise ValueError("runtime artifact context is unexpected")
    handler = _runtime_artifact_handler(expected_kind)
    if context.get("kind") != expected_kind:
        raise ValueError(
            "runtime artifact context kind does not match the sealed template"
        )
    handler.validate_output(output, context)
