#!/usr/bin/env python3
"""Validate the checked-in Metal kernel and research artifact registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
ROOT_KEYS = {
    "schema_version",
    "library",
    "sources",
    "components",
    "slots",
    "handoffs",
    "artifacts",
}
ARTIFACT_KINDS = {
    "benchmarks",
    "evaluators",
    "templates",
    "documents",
    "evidence",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("kernel registry must be a JSON object")
    return value


def resolve_template_binding(
    root: Path, registry: dict[str, Any], template_path: Path
) -> dict[str, str]:
    root = root.resolve()
    template_path = template_path.resolve()
    try:
        relative = template_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("template path must stay within the repository") from error
    matches = [
        artifact
        for artifact in registry["artifacts"]["templates"]
        if artifact["path"] == relative
    ]
    if len(matches) != 1:
        raise ValueError("template path must resolve to exactly one registry artifact")
    artifact = matches[0]
    owning_slots = [
        slot["id"]
        for slot in registry["slots"]
        if artifact["id"] in slot["artifacts"]["templates"]
    ]
    if owning_slots != [artifact["slot_id"]]:
        raise ValueError("template ownership is inconsistent")
    encoded = json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    return {
        "artifact_id": artifact["id"],
        "slot_id": artifact["slot_id"],
        "registry_sha256": sha256(encoded),
    }


def _exact_keys(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} keys do not match the registry schema")


def _ids(records: list[dict[str, Any]], description: str) -> set[str]:
    result: set[str] = set()
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or ID.fullmatch(identifier) is None:
            raise ValueError(f"{description} has an invalid id")
        if identifier in result:
            raise ValueError(f"duplicate {description} id: {identifier}")
        result.add(identifier)
    return result


def _relative_file(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} path must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path must stay within the repository")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing artifact: {value}")
    return path


def _strings(value: Any, description: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{description} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{description} contains duplicates")
    return value


def shader_entry_points(source: str) -> set[str]:
    direct = {
        name
        for name in re.findall(r"^kernel void\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE)
        if name not in {"NAME", "name"}
    }
    direct.update(
        re.findall(r"^DEFINE_CHAIN_PROBE\((solinas_[A-Za-z0-9_]+),", source, re.MULTILINE)
    )
    direct.update(
        re.findall(
            r"^BOOLEANITY_ADDRESS_TILE_ENTRY\((solinas_[A-Za-z0-9_]+),",
            source,
            re.MULTILINE,
        )
    )
    return direct


def _validate_sources(
    root: Path, registry: dict[str, Any]
) -> tuple[set[str], dict[str, str]]:
    sources = registry["sources"]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    source_ids = _ids(sources, "source")
    entry_point_sources: dict[str, str] = {}
    registered_paths: set[Path] = set()
    constants: dict[str, str] = {}
    for source in sources:
        _exact_keys(source, {"id", "path", "source_constant", "entry_points"}, "source")
        path = _relative_file(root, source["path"], "source")
        registered_paths.add(path.resolve())
        source_constant = source["source_constant"]
        if not isinstance(source_constant, str) or not source_constant.endswith("_SOURCE"):
            raise ValueError("source_constant must name a Rust source constant")
        if source_constant in constants:
            raise ValueError(f"duplicate source constant: {source_constant}")
        constants[source_constant] = source["id"]
        declared = set(_strings(source["entry_points"], "source entry_points"))
        duplicate = set(entry_point_sources) & declared
        if duplicate:
            raise ValueError(f"duplicate entry point: {sorted(duplicate)[0]}")
        observed = shader_entry_points(path.read_text())
        if declared != observed:
            raise ValueError(f"entry points do not match shader source {source['id']}")
        entry_point_sources.update({entry_point: source["id"] for entry_point in declared})

    metal_root = root / "crates/jolt-kernels/src/metal"
    observed_paths = {path.resolve() for path in metal_root.rglob("*.metal")}
    if registered_paths != observed_paths:
        raise ValueError("registered Metal sources do not match the repository")

    library = registry["library"]
    _exact_keys(
        library,
        {"rust_path", "context_symbol", "source_order"},
        "library",
    )
    library_path = _relative_file(root, library["rust_path"], "library")
    if not isinstance(library["context_symbol"], str):
        raise ValueError("library context_symbol must be a string")
    library_text = library_path.read_text()
    if library["context_symbol"] not in library_text:
        raise ValueError("library context symbol is missing from its Rust path")
    source_order = _strings(library["source_order"], "library source_order")
    if set(source_order) != source_ids:
        raise ValueError("library source_order must contain every source exactly once")
    builder = re.search(r"let source = format!\((.*?)\n\s*\);", library_text, re.DOTALL)
    if builder is None:
        raise ValueError("library source builder was not found")
    observed_constants = re.findall(r"\{([A-Z][A-Z0-9_]*_SOURCE)\}", builder.group(1))
    try:
        observed_order = [constants[name] for name in observed_constants]
    except KeyError as error:
        raise ValueError(f"library uses an unregistered source constant: {error.args[0]}") from error
    if observed_order != source_order:
        raise ValueError("library source_order does not match SolinasMetal::new")
    return source_ids, entry_point_sources


def _validate_components(
    root: Path,
    registry: dict[str, Any],
    source_ids: set[str],
    entry_point_sources: dict[str, str],
) -> set[str]:
    components = registry["components"]
    if not isinstance(components, list):
        raise ValueError("components must be a list")
    component_ids = _ids(components, "component")
    for component in components:
        _exact_keys(
            component,
            {
                "id",
                "status",
                "rust_path",
                "rust_symbols",
                "source_ids",
                "pipeline_entry_points",
                "uses_components",
            },
            "component",
        )
        if component["status"] not in {"runtime", "diagnostic", "production", "production_shared"}:
            raise ValueError("component has an invalid status")
        rust_path = _relative_file(root, component["rust_path"], "component")
        rust_text = rust_path.read_text()
        for symbol in _strings(component["rust_symbols"], "component rust_symbols"):
            if symbol not in rust_text:
                raise ValueError(f"component symbol {symbol} is missing from {component['rust_path']}")
        if not set(_strings(component["source_ids"], "component source_ids")) <= source_ids:
            raise ValueError("component references an unknown source")
        pipelines = _strings(
            component["pipeline_entry_points"], "component pipeline_entry_points"
        )
        if not set(pipelines) <= set(entry_point_sources):
            raise ValueError("component references an unknown entry point")
        component_sources = set(component["source_ids"])
        for pipeline in pipelines:
            source_id = entry_point_sources[pipeline]
            if source_id not in component_sources:
                raise ValueError(
                    f"component {component['id']} does not include source {source_id} for {pipeline}"
                )
        if not set(_strings(component["uses_components"], "component uses_components")) <= component_ids:
            raise ValueError("component references an unknown component")
    return component_ids


def _validate_artifacts(
    root: Path,
    registry: dict[str, Any],
    component_ids: set[str],
    slot_ids: set[str],
) -> dict[str, set[str]]:
    artifacts = registry["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACT_KINDS:
        raise ValueError("artifact kinds do not match the registry schema")
    artifact_ids: dict[str, set[str]] = {}
    for kind, records in artifacts.items():
        if not isinstance(records, list):
            raise ValueError(f"artifact kind {kind} must be a list")
        artifact_ids[kind] = _ids(records, f"{kind} artifact")
        for artifact in records:
            if not isinstance(artifact, dict) or "path" not in artifact:
                raise ValueError(f"{kind} artifact is missing its path")
            _relative_file(root, artifact["path"], f"{kind} artifact")
            driver = artifact.get("driver")
            if driver is not None:
                _relative_file(root, driver, f"{kind} driver")
            for field in ("covers_components", "component_ids"):
                if field in artifact and not set(_strings(artifact[field], field)) <= component_ids:
                    raise ValueError(f"{kind} artifact references an unknown component")
            for field in ("covers_slots", "slot_ids"):
                if field in artifact and not set(_strings(artifact[field], field)) <= slot_ids:
                    raise ValueError(f"{kind} artifact references an unknown slot")
            slot_id = artifact.get("slot_id")
            if slot_id is not None and slot_id not in slot_ids:
                raise ValueError(f"{kind} artifact references an unknown slot")
    return artifact_ids


def _validate_slots_and_handoffs(
    root: Path,
    registry: dict[str, Any],
    component_ids: set[str],
) -> set[str]:
    slots = registry["slots"]
    if not isinstance(slots, list):
        raise ValueError("slots must be a list")
    slot_ids = _ids(slots, "slot")
    for slot in slots:
        _exact_keys(
            slot,
            {
                "id",
                "stage",
                "host_path",
                "config_field",
                "config_symbol",
                "relation_symbol",
                "trait_kind",
                "components",
                "consumes_state",
                "produces_state",
                "artifacts",
            },
            "slot",
        )
        host_path = _relative_file(root, slot["host_path"], "slot")
        host_text = host_path.read_text()
        for symbol_field in ("config_symbol", "relation_symbol", "trait_kind"):
            symbol = slot[symbol_field]
            if not isinstance(symbol, str) or symbol not in host_text:
                raise ValueError(f"slot {slot['id']} is missing {symbol_field}")
        if slot["config_field"] != slot["id"]:
            raise ValueError("slot config_field must equal its canonical id")
        if not set(_strings(slot["components"], "slot components")) <= component_ids:
            raise ValueError("slot references an unknown component")
        _strings(slot["consumes_state"], "slot consumes_state")
        _strings(slot["produces_state"], "slot produces_state")
        if not isinstance(slot["artifacts"], dict) or set(slot["artifacts"]) != ARTIFACT_KINDS:
            raise ValueError("slot artifact references do not match the registry schema")

    config_path = root / "crates/jolt-kernels/src/metal/instruction_read_raf.rs"
    config_text = config_path.read_text()
    config = re.search(r"pub struct MetalConfig \{(.*?)\n\}", config_text, re.DOTALL)
    if config is None:
        raise ValueError("MetalConfig was not found")
    config_fields = set(re.findall(r"^\s*pub ([a-z][a-z0-9_]*):", config.group(1), re.MULTILINE))
    if config_fields != slot_ids:
        raise ValueError("registry slots do not match MetalConfig")
    installer = re.search(r"pub fn with_metal_compute\(.*?\n\s*\}", config_text, re.DOTALL)
    if installer is None:
        raise ValueError("with_metal_compute was not found")
    installed = set(
        re.findall(r"self\.([a-z][a-z0-9_]*)\s*=\s*Box::new", installer.group(0))
    )
    if installed != slot_ids:
        raise ValueError("registry slots do not match with_metal_compute")

    handoffs = registry["handoffs"]
    if not isinstance(handoffs, list):
        raise ValueError("handoffs must be a list")
    handoff_ids = _ids(handoffs, "handoff")
    state_ids: set[str] = set()
    for handoff in handoffs:
        _exact_keys(
            handoff,
            {"id", "rust_type", "producer_slots", "consumer_slots"},
            "handoff",
        )
        producers = set(_strings(handoff["producer_slots"], "handoff producer_slots"))
        consumers = set(_strings(handoff["consumer_slots"], "handoff consumer_slots"))
        if not producers <= slot_ids or not consumers <= slot_ids:
            raise ValueError("handoff references an unknown slot")
        rust_type = handoff["rust_type"]
        if not isinstance(rust_type, str) or not rust_type:
            raise ValueError("handoff rust_type must be a nonempty string")
        state_ids.add(handoff["id"])
    if handoff_ids != state_ids:
        raise ValueError("handoff ids are inconsistent")
    for slot in slots:
        declared = set(slot["consumes_state"]) | set(slot["produces_state"])
        if not declared <= handoff_ids:
            raise ValueError("slot references an unknown handoff")
    return slot_ids


def _validate_slot_artifacts(
    registry: dict[str, Any], artifact_ids: dict[str, set[str]]
) -> None:
    artifacts = {
        kind: {artifact["id"]: artifact for artifact in records}
        for kind, records in registry["artifacts"].items()
    }
    for slot in registry["slots"]:
        for kind, identifiers in slot["artifacts"].items():
            if not set(_strings(identifiers, f"slot {kind} artifacts")) <= artifact_ids[kind]:
                raise ValueError(f"slot {slot['id']} references an unknown {kind} artifact")
            if kind == "templates" and any(
                artifacts[kind][identifier].get("slot_id") != slot["id"]
                for identifier in identifiers
            ):
                raise ValueError(f"slot {slot['id']} template ownership is inconsistent")
    referenced_templates = {
        identifier
        for slot in registry["slots"]
        for identifier in slot["artifacts"]["templates"]
    }
    if referenced_templates != artifact_ids["templates"]:
        raise ValueError("template ownership does not cover every registered template")


def validate_registry(root: Path, registry: dict[str, Any]) -> None:
    _exact_keys(registry, ROOT_KEYS, "registry")
    if registry["schema_version"] != 1:
        raise ValueError("unsupported kernel registry schema")
    source_ids, entry_point_sources = _validate_sources(root, registry)
    component_ids = _validate_components(
        root, registry, source_ids, entry_point_sources
    )
    slot_ids = _validate_slots_and_handoffs(root, registry, component_ids)
    artifact_ids = _validate_artifacts(root, registry, component_ids, slot_ids)
    _validate_slot_artifacts(registry, artifact_ids)


def registered_paths(registry: dict[str, Any]) -> Iterable[str]:
    for source in registry["sources"]:
        yield source["path"]
    for component in registry["components"]:
        yield component["rust_path"]
    for slot in registry["slots"]:
        yield slot["host_path"]
    for records in registry["artifacts"].values():
        for artifact in records:
            yield artifact["path"]
            if artifact.get("driver") is not None:
                yield artifact["driver"]
