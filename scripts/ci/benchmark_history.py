#!/usr/bin/env python3
"""Collect authenticated benchmark JSON and append it to a local Pages checkout."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
import zipfile


REPOSITORY = "a16z/jolt"
WORKFLOW = "End-to-end benchmark ingest"
WORKFLOW_FILE = ".github/workflows/e2e-bench-ingest.yml"
PLAN_WORKFLOW = "End-to-end prover benchmarks"
PLAN_WORKFLOW_FILE = ".github/workflows/e2e-bench.yml"
APP = "0xandoroid-ai-agent[bot]"
SITE_PATH = Path("dev/prover-history")
ASSETS = Path(__file__).resolve().parents[2] / "web/benchmarks"
METRICS = (
    "e2e_s", "prove_s", "setup_s", "verify_s", "verify_single_s",
    "process_s", "peak_rss_bytes", "proof_bytes", "trace_rows", "padded_trace_rows",
)
CASE_STATUSES = {"complete", "failed", "incomplete", "unsupported", "missing"}
RUN_STATUSES = {"complete", "failed", "incomplete", "disabled", "unsupported"}
MAX_JSON_BYTES = 32 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def read_json(data):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid_constant(value)
        return parsed

    return json.loads(data, parse_constant=invalid_constant, parse_float=finite_float)


def require_fields(value, fields, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    absent = set(fields) - value.keys()
    if absent:
        raise ValueError(f"{label} is missing {', '.join(sorted(absent))}")


def revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamps must be UTC ISO-8601 strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("Timestamps must use UTC")
    return parsed.isoformat()


def validate_file_descriptor(value):
    require_fields(value, ("path", "sha256"), "File descriptor")
    path = value["path"]
    if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("Manifest paths must be relative without parent escapes")
    if not isinstance(value["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None:
        raise ValueError("Manifest hashes must be lowercase SHA-256")


def validate_build(build):
    require_fields(build, ("schema_version", "leg", "revision", "harness_revision", "status", "reason"), "Build")
    if (
        type(build["schema_version"]) is not int or build["schema_version"] != 1
        or build["leg"] not in {"base", "head"} or not revision(build["revision"])
        or not revision(build["harness_revision"])
        or build["status"] not in {"complete", "failed", "incomplete", "unsupported"}
    ):
        raise ValueError("Invalid build schema, identity, or status")
    if build["status"] != "complete":
        return
    require_fields(build, (
        "toolchain", "cargo_version", "target_cpu", "target_triple", "target_features", "profile",
        "rustflags", "source_cargo_lock_sha256", "cargo_lock_sha256", "harness_files", "binaries",
        "guests", "runner", "akita_schedules", "suite_config_sha256", "host",
        "started_at", "finished_at", "build_s", "cache",
    ), "Complete build")
    for name in ("toolchain", "cargo_version", "target_cpu", "target_triple", "profile", "rustflags"):
        if not isinstance(build[name], str):
            raise ValueError(f"Build {name} must be a string")
    if build["target_cpu"] == "native":
        raise ValueError("Build target_cpu must be explicit, not native")
    features = build["target_features"]
    if not isinstance(features, list) or any(not isinstance(value, str) for value in features) or features != sorted(features):
        raise ValueError("Build target_features must be a sorted array of strings")
    for name in ("source_cargo_lock_sha256", "cargo_lock_sha256", "suite_config_sha256"):
        if not isinstance(build[name], str) or re.fullmatch(r"[0-9a-f]{64}", build[name]) is None:
            raise ValueError(f"Build {name} must be lowercase SHA-256")
    require_fields(build["harness_files"], (), "Harness files")
    require_fields(build["binaries"], (), "Binaries")
    require_fields(build["guests"], (), "Guests")
    require_fields(build["host"], (), "Build host")
    require_fields(build["cache"], ("cargo_target_preexisting", "rustc_wrapper", "hit_status"), "Build cache")
    for path, sha in build["harness_files"].items():
        validate_file_descriptor({"path": path, "sha256": sha})
    for pcs, binary in build["binaries"].items():
        if pcs not in {"dory", "akita"}:
            raise ValueError("Unknown binary PCS")
        validate_file_descriptor(binary)
    for guest in build["guests"].values():
        require_fields(guest, ("elf", "inputs"), "Guest")
        validate_file_descriptor(guest["elf"])
        require_fields(guest["inputs"], (), "Guest inputs")
        for count, descriptor in guest["inputs"].items():
            if re.fullmatch(r"[0-9]+", count) is None:
                raise ValueError("Guest inputs must use decimal operation counts")
            validate_file_descriptor(descriptor)
    validate_file_descriptor(build["runner"])
    if not isinstance(build["akita_schedules"], list):
        raise ValueError("Akita schedules must be an array")
    for descriptor in build["akita_schedules"]:
        validate_file_descriptor(descriptor)
    utc_timestamp(build["started_at"])
    utc_timestamp(build["finished_at"])
    if type(build["build_s"]) not in {int, float} or not math.isfinite(build["build_s"]) or build["build_s"] < 0:
        raise ValueError("Complete builds require a finite nonnegative build_s")


def validate_result(data):
    require_fields(data, (
        "schema_version", "run_id", "tier", "kind", "status", "reason", "repository",
        "started_at", "finished_at", "revisions", "provenance", "suite", "host",
        "builds", "wall_time", "cases",
    ), "Result")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported benchmark schema_version; expected 1")
    if data["repository"] != REPOSITORY or not isinstance(data["run_id"], str):
        raise ValueError("Invalid result repository or run_id")
    if data["tier"] not in {"pr", "main", "nightly"}:
        raise ValueError("Unknown benchmark tier")
    if data["kind"] not in {"comparison", "null", "settings", "history"}:
        raise ValueError("Unknown benchmark kind")
    if data["status"] not in RUN_STATUSES:
        raise ValueError("Unknown run status")
    utc_timestamp(data["started_at"])
    utc_timestamp(data["finished_at"])
    require_fields(data["revisions"], ("base", "head", "merge", "harness"), "Revisions")
    for value in (data["revisions"][key] for key in ("base", "head", "merge", "harness")):
        if value is not None and not revision(value):
            raise ValueError("Revisions must be full Git revisions or null")
    require_fields(data["provenance"], (
        "workflow", "workflow_file", "run_id", "run_attempt", "event", "ref", "pr_number",
    ), "Provenance")
    require_fields(data["suite"], (
        "id", "config_sha256", "metric_contract", "warmups", "repetitions",
    ), "Suite")
    if type(data["suite"]["metric_contract"]) is not int or data["suite"]["metric_contract"] != 1:
        raise ValueError("Unsupported metric_contract; expected 1")
    for name, minimum in (("warmups", 0), ("repetitions", 1)):
        value = data["suite"][name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid suite {name}")
    if data["host"] is None and (data["status"] == "complete" or any(case["samples"] for case in data["cases"])):
        raise ValueError("Measured results require a host identity")
    if data["host"] is not None:
        require_fields(data["host"], (
            "id", "architecture", "cpu_model", "kernel", "physical_cores", "logical_cpus",
            "memory_bytes", "topology", "settings",
        ), "Host")
    require_fields(data["wall_time"], ("build_s", "benchmark_s", "total_s", "scope"), "Wall time")
    if not isinstance(data["builds"], list) or not isinstance(data["cases"], list):
        raise ValueError("Builds and cases must be arrays")
    for build in data["builds"]:
        validate_build(build)
    case_ids = set()
    case_points = set()
    for case in data["cases"]:
        require_fields(case, (
            "id", "workload", "pcs", "backend", "physical_cores", "cpu_ids",
            "scale_log2", "input", "status", "reason", "samples",
        ), "Case")
        if not isinstance(case["id"], str) or case["id"] in case_ids:
            raise ValueError("Case ids must be distinct strings")
        case_ids.add(case["id"])
        if case["pcs"] not in {"dory", "akita"} or case["backend"] != "optimized":
            raise ValueError("Unsupported PCS or CPU backend")
        if type(case["physical_cores"]) is not int or case["physical_cores"] not in {1, 8, 24}:
            raise ValueError("Expected exactly 1, 8, or 24 physical cores")
        if case["status"] not in CASE_STATUSES or not isinstance(case["samples"], list):
            raise ValueError("Invalid case status or samples")
        require_fields(case["input"], ("parameters", "seed", "sha256"), "Input")
        point = json.dumps({key: case[key] for key in (
            "workload", "pcs", "backend", "physical_cores", "scale_log2", "input",
        )}, sort_keys=True)
        if point in case_points:
            raise ValueError("Duplicate benchmark point in one run")
        case_points.add(point)
        sample_ids = set()
        for sample in case["samples"]:
            require_fields(sample, (
                "leg", "revision", "phase", "repetition", "pair_index", "order",
                "status", "exit_code", "reason", "metrics",
            ), "Sample")
            if sample["leg"] not in {"base", "head"} or sample["phase"] not in {"warmup", "measure"}:
                raise ValueError("Invalid sample leg or phase")
            if not revision(sample["revision"]) or sample["status"] not in CASE_STATUSES:
                raise ValueError("Invalid sample revision or status")
            if type(sample["repetition"]) is not int or sample["repetition"] < 0:
                raise ValueError("Invalid repetition index")
            identity = (sample["leg"], sample["phase"], sample["repetition"])
            if identity in sample_ids:
                raise ValueError("Duplicate sample repetition")
            sample_ids.add(identity)
            if sample["metrics"] is None and sample["status"] != "complete":
                continue
            require_fields(sample["metrics"], METRICS if sample["status"] == "complete" else (), "Metrics")
            if sample["status"] == "complete" and (type(sample["exit_code"]) is not int or sample["exit_code"] != 0):
                raise ValueError("A complete sample must have exit_code 0")
            for name, value in sample["metrics"].items():
                if name not in METRICS or value is None:
                    continue
                if type(value) not in {int, float}:
                    raise ValueError(f"Metric {name} must be numeric or null")
                try:
                    valid = value >= 0 and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError(f"Metric {name} must be finite and nonnegative")


def gh(path):
    return subprocess.check_output([
        "gh", "api", "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28", path,
    ])


def validate_source(run, run_id, attempt):
    # REST run names expand run-name; the repository and workflow path identify the source.
    if (
        run["id"] != run_id or run["run_attempt"] != attempt
        or run["repository"]["full_name"] != REPOSITORY
        or (run.get("head_repository") or {}).get("full_name") != REPOSITORY
        or run["path"] != WORKFLOW_FILE
        or run["event"] != "workflow_dispatch" or run["head_branch"] != "main"
        or run["actor"]["login"] != APP
        or run["status"] != "completed" or run["conclusion"] != "success" or not revision(run["head_sha"])
    ):
        raise ValueError("Refusing a foreign, untrusted, unfinished, skipped, or mismatched producer run")


def validate_provenance(data, run, measured_revision, plan):
    provenance = data["provenance"]
    expected = {
        "workflow": PLAN_WORKFLOW, "workflow_file": PLAN_WORKFLOW_FILE,
        "event": run["event"], "ref": "refs/heads/main",
        "pr_number": plan["provenance"]["pr_number"],
    }
    if any(provenance[key] != value for key, value in expected.items()):
        raise ValueError("Artifact provenance does not match its GitHub workflow")
    for field, value in (("run_id", run["id"]), ("run_attempt", run["run_attempt"])):
        if str(provenance[field]) != str(value):
            raise ValueError(f"Artifact {field} does not match GitHub")
    if plan["tier"] == "pr":
        item = plan["runs"][0]
        expected_revisions = {"base": item["base"], "head": item["submitted_head"],
                              "merge": item["head"], "harness": plan["harness"]}
        if (data["tier"] != "pr" or data["kind"] != "comparison"
                or data["revisions"] != expected_revisions
                or data.get("artifacts", {}).get("plan_sha256") != plan["sha256"]
                or data["wall_time"]["scope"] != "sequential-build-preparation-through-results-on-this-host"):
            raise ValueError("PR result differs from its trusted admission plan")
        legs = {"base": item["base"], "head": item["head"]}
        for build in data["builds"]:
            if build["revision"] != legs[build["leg"]] or build["harness_revision"] != plan["harness"]:
                raise ValueError("PR build differs from the admitted revisions")
            planned = dict(plan["expected_builds"][build["revision"]])
            if build["native_harness_revision"] != plan["harness"]:
                if build["native_harness_revision"] != build["revision"] or not build.get("harness_fallback_reason"):
                    raise ValueError("PR build uses an unplanned native harness")
                planned["harness_files"] = plan["own_harness_files"][build["revision"]]
            if build["status"] == "complete" and any(build.get(key) != value for key, value in planned.items()):
                raise ValueError("PR build differs from the planned native inputs")
        for case in data["cases"]:
            if any(sample["revision"] != legs[sample["leg"]] for sample in case["samples"]):
                raise ValueError("PR sample differs from the admitted revisions")
        return
    expected_tier = {"push": "main", "schedule": "nightly"}.get(run["event"])
    if (
        data["tier"] not in {"main", "nightly"} or data["kind"] != "history"
        or (expected_tier and data["tier"] != expected_tier)
        or data["revisions"]["head"] != measured_revision
        or data["revisions"]["merge"] not in {None, measured_revision}
        or data["revisions"]["harness"] != plan["harness"]
        or data.get("artifacts") != {"id": plan["dispatch"]["build_artifact_id"],
            "index_sha256": plan["dispatch"]["build_index_sha256"], "plan_sha256": plan["sha256"]}
    ):
        raise ValueError("Artifact tier, purpose, or revision does not match the source run")
    for build in data["builds"]:
        if build["leg"] != "head" or build["revision"] != measured_revision:
            raise ValueError("History build does not match the source revision")
        manifest = {key: value for key, value in build.items() if key != "leg"}
        if digest(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()) != plan["dispatch"]["manifest_digests"][measured_revision]:
            raise ValueError("History build differs from the hosted dispatch digest")
    for case in data["cases"]:
        for sample in case["samples"]:
            if sample["leg"] != "head" or sample["revision"] != measured_revision:
                raise ValueError("History sample does not match the source revision")


def artifact_json(bundle, member):
    entries = [entry for entry in bundle.infolist() if entry.filename == member]
    if not entries:
        return None
    if len(entries) != 1 or entries[0].is_dir():
        raise ValueError(f"Artifact member is duplicated or not a file: {member}")
    if entries[0].file_size > MAX_JSON_BYTES:
        raise ValueError("Benchmark JSON exceeds the 32 MiB publication limit")
    return bundle.read(entries[0])


def producer_run(ingest):
    title = re.fullmatch(r"e2e-bench ingest ([1-9][0-9]*)-([1-9][0-9]*)", ingest["display_title"])
    if title is None:
        raise ValueError("Ingest run title does not identify a producer attempt")
    run_id, attempt = map(int, title.groups())
    run = json.loads(gh(f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}"))
    if (run["id"] != run_id or run["run_attempt"] != attempt
            or run["repository"]["full_name"] != REPOSITORY
            or (run.get("head_repository") or {}).get("full_name") != REPOSITORY
            or run["name"] != PLAN_WORKFLOW or run["path"] != PLAN_WORKFLOW_FILE
            or run["event"] not in {"push", "schedule", "workflow_dispatch", "pull_request_target"}
            or (run["event"] != "pull_request_target" and run["head_branch"] != "main")
            or run["status"] != "completed" or run["conclusion"] != "success"):
        raise ValueError("Ingest names an untrusted plan producer")
    return run


def producer_plan(ingest, bundle, run):
    raw = artifact_json(bundle, "ingest.json")
    if raw is None:
        raise ValueError("Canonical result has no ingest receipt")
    receipt = read_json(raw)
    run_id, attempt = receipt["plan_run_id"], receipt["plan_run_attempt"]
    if (receipt["schema"] != 1 or type(run_id) is not int or run_id <= 0
            or type(attempt) is not int or attempt <= 0
            or receipt["ingest_run_id"] != ingest["id"] or receipt["ingest_run_attempt"] != ingest["run_attempt"]
            or receipt["job_id"] != f"{run_id}-{attempt}"
            or ingest["display_title"] != f"e2e-bench ingest {receipt['job_id']}"):
        raise ValueError("Ingest receipt does not match the App run and job")
    if (run["id"], run["run_attempt"]) != (run_id, attempt):
        raise ValueError("Ingest receipt names a different producer attempt")
    artifacts, page = [], 1
    while True:
        batch = json.loads(gh(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"))["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    documents = {}
    for prefix, member, id_key, hash_key in (
            ("jolt-e2e-plan", "plan.json", "plan_artifact_id", "plan_sha256"),
            ("jolt-e2e-dispatch", "dispatch.json", "dispatch_artifact_id", "dispatch_sha256")):
        matches = [item for item in artifacts if item["name"] == f"{prefix}-{run_id}-{attempt}"]
        if len(matches) != 1:
            raise ValueError("History requires one exact plan-run artifact per attempt")
        artifact = matches[0]
        if (artifact["id"] != receipt[id_key] or artifact["expired"]
                or artifact["workflow_run"]["id"] != run_id or artifact["size_in_bytes"] > 8 * 1024 * 1024):
            raise ValueError("Plan-run artifact differs from the trusted ingest receipt")
        archive = gh(f"repos/{REPOSITORY}/actions/artifacts/{int(artifact['id'])}/zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            content = artifact_json(source, member)
        if content is None or digest(content) != receipt[hash_key]:
            raise ValueError("Plan-run artifact digest differs from the ingest receipt")
        documents[member] = read_json(content)
    plan, dispatch = documents["plan.json"], documents["dispatch.json"]
    provenance = plan["provenance"]
    if (plan["enabled"] is not True or not plan["runs"] or not revision(plan["harness"])
            or dispatch["schema"] != 1 or dispatch["tier"] != plan["tier"]
            or dispatch["run_id"] != run_id or dispatch["run_attempt"] != attempt
            or provenance["workflow_file"] != PLAN_WORKFLOW_FILE or provenance["workflow"] != PLAN_WORKFLOW
            or provenance["event"] != run["event"] or provenance["ref"] != "refs/heads/main"
            or str(provenance["run_id"]) != str(run_id) or str(provenance["run_attempt"]) != str(attempt)):
        raise ValueError("Plan and dispatch do not match their trusted producer attempt")
    if plan["tier"] == "calibration" and run["event"] != "workflow_dispatch":
        raise ValueError("Bootstrap calibration requires a main workflow dispatch")
    if plan["tier"] == "pr":
        item = plan["runs"][0]
        admission = plan["admission"]
        if (run["event"] != "pull_request_target" or len(plan["runs"]) != 1
                or (run.get("pull_requests") and not any(pr["number"] == provenance["pr_number"]
                    and pr["base"]["ref"] == "main" for pr in run["pull_requests"]))):
            raise ValueError("PR plan number is not a pull request of the producer run")
        if (item["submitted_head"] != run["head_sha"] or not revision(item["base"])
                or not revision(item["head"]) or admission["head_ref"] != run["head_branch"]
                or dispatch["check_sha"] != item["submitted_head"]
                or dispatch["pr"] != {"number": provenance["pr_number"], "head_sha": item["submitted_head"],
                    "label_event_id": admission["label_event_id"], "label_actor": admission["label_actor"]}):
            raise ValueError("PR plan does not match its admitted source and label")
        commit = json.loads(gh(f"repos/{REPOSITORY}/commits/{item['head']}"))
        if [parent["sha"] for parent in commit["parents"]] != [item["base"], item["submitted_head"]]:
            raise ValueError("PR merge parents differ from the admitted base and head")
    elif (run["event"] == "pull_request_target" or run["head_branch"] != "main"
            or plan["runs"][-1]["head"] != run["head_sha"] or dispatch["check_sha"] != run["head_sha"]):
        raise ValueError("History plan is not bound to the measured main revision")
    return dict(plan, sha256=receipt["plan_sha256"], artifact_id=receipt["plan_artifact_id"], dispatch=dispatch)


def first_parent_revisions(before, after, oldest):
    base = before
    include_root = False
    if base is None:
        commit = json.loads(gh(f"repos/{REPOSITORY}/commits/{oldest}"))
        if commit["sha"] != oldest:
            raise ValueError("GitHub did not return the declared oldest revision")
        include_root = not commit["parents"]
        base = oldest if include_root else commit["parents"][0]["sha"]
    commits = {}
    page = 1
    while base != after:
        comparison = json.loads(gh(f"repos/{REPOSITORY}/compare/{base}...{after}?per_page=100&page={page}"))
        if comparison["status"] != "ahead" or comparison["base_commit"]["sha"] != base:
            raise ValueError("The recorded push range is not an ancestor comparison")
        batch = comparison["commits"]
        commits.update((commit["sha"], commit) for commit in batch)
        if len(batch) < 100:
            break
        page += 1
    expected = []
    current = after
    while current != base:
        if current not in commits or not commits[current]["parents"]:
            raise ValueError("The recorded lower bound is not on the first-parent chain")
        expected.append(current)
        current = commits[current]["parents"][0]["sha"]
    if include_root:
        expected.append(base)
    expected.reverse()
    if not expected:
        raise ValueError("The recorded range has no new first-parent commits")
    return expected, base


def push_documents(raw_index, run):
    index = read_json(raw_index)
    require_fields(index, ("schema_version", "push_before", "push_after", "documents"), "Push index")
    if (
        type(index["schema_version"]) is not int or index["schema_version"] != 1
        or run["event"] != "push" or index["push_after"] != run["head_sha"]
        or (index["push_before"] is not None and not revision(index["push_before"]))
        or not isinstance(index["documents"], list)
    ):
        raise ValueError("Push index does not match the source main run")
    declared = []
    for document in index["documents"]:
        require_fields(document, ("revision", "path"), "Push document")
        sha = document["revision"]
        if not revision(sha) or sha in declared:
            raise ValueError("Push index revisions must be distinct full revisions")
        member = "benchmark-results.json" if sha == run["head_sha"] else f"commits/{sha}/benchmark-results.json"
        if document["path"] != member:
            raise ValueError("Push index contains a disallowed artifact member path")
        declared.append(sha)
    expected, base = first_parent_revisions(index["push_before"], run["head_sha"], declared[0] if declared else run["head_sha"])
    if declared != [sha for sha in expected if sha in declared]:
        raise ValueError("Push index revisions are not in oldest-first first-parent order")
    return expected, set(declared), {
        "producer_before": index["push_before"], "after": run["head_sha"], "ancestry_base": base,
        "ancestry_verified": True, "event_bound_verified": False, "commits": len(expected),
    }


def source_record(run, publisher_revision, measured_revision, producer):
    return {
        "key": f"github-{measured_revision}-{run['id']}-{run['run_attempt']}",
        "source": {
            "type": "github", "repository": REPOSITORY, "commit": measured_revision,
            "run_head_sha": run["head_sha"], "run_id": run["id"], "run_attempt": run["run_attempt"],
            "event": producer["event"], "workflow": WORKFLOW, "workflow_file": WORKFLOW_FILE,
            "url": f"https://github.com/{REPOSITORY}/actions/runs/{run['id']}/attempts/{run['run_attempt']}",
            "conclusion": run["conclusion"], "artifact": None,
            "plan_run_id": producer["id"], "plan_run_attempt": producer["run_attempt"],
        },
        "started_at": utc_timestamp(producer["run_started_at"]),
        "tier": {"push": "main", "schedule": "nightly", "pull_request_target": "pr"}.get(producer["event"], "unknown"),
        "kind": "comparison" if producer["event"] == "pull_request_target" else "history",
        "status": "missing", "publication_status": "missing",
        "reason": "Producer completed without the expected artifact; the point values are unknown.",
        "benchmark_schema": None, "sha256": None, "push_index": None, "push_index_sha256": None,
        "publisher_revision": publisher_revision,
    }


def collect(run_id, attempt, publisher_revision):
    root = f"repos/{REPOSITORY}/actions"
    run = json.loads(gh(f"{root}/runs/{run_id}/attempts/{attempt}"))
    validate_source(run, run_id, attempt)
    producer = producer_run(run)
    expected_name = f"jolt-e2e-{run_id}-{attempt}"
    artifacts = []
    page = 1
    while True:
        batch = json.loads(gh(f"{root}/runs/{run_id}/artifacts?per_page=100&page={page}"))["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    record = source_record(run, publisher_revision, producer["head_sha"], producer)
    artifacts = [item for item in artifacts if item["name"] == expected_name]
    root_raw = None
    raw_index = None
    if len(artifacts) > 1:
        record.update(publication_status="rejected", status="rejected", reason="Multiple artifacts have the expected run-attempt name.")
    elif artifacts:
        artifact = artifacts[0]
        record["source"]["artifact"] = {
            "id": artifact["id"], "name": artifact["name"], "digest": artifact.get("digest"),
        }
        artifact_run = artifact.get("workflow_run") or {}
        if artifact_run.get("id") != run_id or artifact_run.get("head_sha") != run["head_sha"]:
            record.update(publication_status="rejected", status="rejected", reason="Artifact metadata does not match the source run.")
        elif artifact["expired"]:
            record["reason"] = "The producer artifact expired before publication; the point values are unknown."
        elif artifact["size_in_bytes"] > 128 * 1024 * 1024:
            record.update(publication_status="rejected", status="rejected", reason="Artifact exceeds the 128 MiB download limit.")
        else:
            archive = gh(f"{root}/artifacts/{int(artifact['id'])}/zip")
            try:
                with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                    plan = producer_plan(run, bundle, producer)
                    root_raw = artifact_json(bundle, "benchmark-results.json")
                    raw_index = artifact_json(bundle, "index.json")
                    expected, declared, bounds = [producer["head_sha"]], {producer["head_sha"]}, None
                    if raw_index is not None:
                        record["push_index"] = f"runs/{record['key']}.index.json"
                        record["push_index_sha256"] = digest(raw_index)
                        expected, declared, bounds = push_documents(raw_index, producer)
                    records = []
                    reference_raw = artifact_json(bundle, "calibration/reference.json")
                    if reference_raw is not None:
                        reference = read_json(reference_raw)
                        raw = artifact_json(bundle, "calibration/raw/benchmark-results.json")
                        receipt = read_json(artifact_json(bundle, "ingest.json"))
                        source = reference["source"]
                        canonical = json.dumps(reference, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                        if (plan["tier"] == "pr" or source.get("tier") != plan["tier"]
                                or reference["schema"] != 1 or raw is None
                                or digest(raw) != reference["raw_sha256"] or reference_raw != canonical
                                or any(source.get(name) != receipt[name] for name in receipt)):
                            raise ValueError("Calibration reference does not match its trusted ingest source")
                        sha = digest(reference_raw)
                        package = json_bytes({"reference": reference, "raw": raw.decode()})
                        records.append(({"key": "calibration-" + sha, "sha256": digest(package),
                            "publication_status": "ready", "bootstrap": plan["tier"] == "calibration",
                            "calibration": {"sha256": sha,
                                "raw_sha256": reference["raw_sha256"], "artifact_id": artifact["id"],
                                "ingest_run_id": run_id, "ingest_run_attempt": attempt,
                                "plan_run_id": producer["id"], "plan_run_attempt": producer["run_attempt"],
                                "key_sha256s": [cell["key_sha256"] for cell in reference["cells"]]}}, package))
                    if plan["tier"] == "calibration":
                        return records, None
                    for position, sha in enumerate(expected):
                        item = source_record(run, publisher_revision, sha, producer)
                        item["tier"] = plan["tier"]
                        item["source"].update(plan_artifact_id=plan["artifact_id"], plan_sha256=plan["sha256"])
                        if plan["tier"] == "pr":
                            item["source"].update(pr_number=plan["provenance"]["pr_number"],
                                base_commit=plan["runs"][0]["base"], merge_commit=plan["runs"][0]["head"],
                                harness_revision=plan["harness"], label_event_id=plan["admission"]["label_event_id"],
                                label_actor=plan["admission"]["label_actor"])
                        item["source"]["artifact"] = record["source"]["artifact"]
                        item["source"]["commit_index"] = position
                        if bounds is not None:
                            item["source"]["push_range"] = bounds
                            item["push_index"] = record["push_index"]
                            item["push_index_sha256"] = record["push_index_sha256"]
                        member = "benchmark-results.json" if sha == producer["head_sha"] else f"commits/{sha}/benchmark-results.json"
                        raw = artifact_json(bundle, member)
                        if raw is not None:
                            item["sha256"] = digest(raw)
                        if sha not in declared or raw is None:
                            item["reason"] = f"Expected main revision {sha} has no indexed result document."
                        else:
                            try:
                                data = read_json(raw)
                                if isinstance(data, dict):
                                    item["benchmark_schema"] = data.get("schema_version")
                                validate_result(data)
                                validate_provenance(data, producer, sha, plan)
                                item.update(tier=data["tier"], kind=data["kind"], status=data["status"], publication_status="ready", reason=data["reason"])
                            except (ValueError, TypeError) as error:
                                item.update(publication_status="rejected", status="rejected", reason=str(error))
                        records.append((item, raw))
                    return records, raw_index
            except (ValueError, TypeError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
                record.update(publication_status="rejected", status="rejected", reason=str(error))
    if root_raw is not None:
        record["sha256"] = digest(root_raw)
    return [(record, root_raw)], raw_index


def preview(raw, label, publisher_revision):
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Benchmark JSON exceeds the 32 MiB publication limit")
    data = read_json(raw)
    validate_result(data)
    return {
        "key": f"preview-{digest(raw)}",
        "source": {"type": "manual", "label": label, "commit": data["revisions"]["head"]},
        "started_at": utc_timestamp(data["started_at"]),
        "tier": data["tier"], "kind": data["kind"], "status": data["status"],
        "publication_status": "ready", "reason": data["reason"],
        "benchmark_schema": data["schema_version"], "sha256": digest(raw),
        "push_index": None, "push_index_sha256": None,
        "publisher_revision": publisher_revision,
    }


def safe_path(site, relative):
    root = site.resolve()
    path = root / relative
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError("History path escapes the checkout or contains a symlink")
    return path


def archive_filename(record):
    key = record["key"]
    if not isinstance(key, str) or re.fullmatch(r"(?:calibration-[0-9a-f]{64}|github-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*|preview-[0-9a-f]{64})", key) is None:
        raise ValueError("Invalid publication key")
    return f"{key}.json"


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".history-", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def append_site(site, records=(), raw_index=None, assets=ASSETS):
    index_path = safe_path(site, SITE_PATH / "history.json")
    history = read_json(index_path.read_bytes()) if index_path.exists() else {"format_version": 1, "runs": []}
    if history.get("format_version") != 1 or not isinstance(history.get("runs"), list):
        raise ValueError("Unsupported history index; refusing to overwrite it")
    catalog_path = safe_path(site, SITE_PATH / "calibration/index.json")
    catalog = read_json(catalog_path.read_bytes()) if catalog_path.exists() else {"schema": 1, "references": []}
    if catalog.get("schema") != 1 or not isinstance(catalog.get("references"), list):
        raise ValueError("Unsupported calibration catalog")
    appended = 0
    for record, raw in records:
        key = record["key"]
        filename = archive_filename(record)
        if record["sha256"] != (digest(raw) if raw is not None else None):
            raise ValueError("Receipt and raw JSON digest differ")
        if record.get("calibration"):
            entry = record["calibration"]
            package = read_json(raw)
            payload = json.dumps(package["reference"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            evidence = package["raw"].encode()
            if digest(payload) != entry["sha256"] or digest(evidence) != entry["raw_sha256"]:
                raise ValueError("Calibration archive digest mismatch")
            for name, content in (("reference.json", payload), ("raw.json", evidence)):
                target = safe_path(site, SITE_PATH / "calibration" / entry["sha256"] / name)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError("Conflicting immutable calibration evidence")
                if not target.exists():
                    atomic_write(target, content)
            existing = next((item for item in catalog["references"] if item["sha256"] == entry["sha256"]), None)
            if existing is not None and existing != entry:
                raise ValueError("Conflicting calibration catalog identity")
            if existing is None:
                catalog["references"].append(entry)
                atomic_write(catalog_path, json_bytes(catalog))
            continue
        existing = next((item for item in history["runs"] if item["key"] == key), None)
        if existing:
            # Expired upstream artifacts cannot erase an already archived result.
            if record["sha256"] is not None and existing["sha256"] != record["sha256"]:
                raise ValueError("Conflicting bytes for an immutable run attempt; rerun the producer")
            if record.get("push_index_sha256") is not None and existing.get("push_index_sha256") != record["push_index_sha256"]:
                raise ValueError("Conflicting immutable push index; rerun the producer")
        else:
            record = dict(record)
            record["raw"] = f"runs/{filename}" if raw is not None else None
            if raw is not None:
                raw_path = safe_path(site, SITE_PATH / record["raw"])
                if raw_path.exists() and raw_path.read_bytes() != raw:
                    raise ValueError("Conflicting archived JSON; refusing to overwrite it")
                if not raw_path.exists():
                    atomic_write(raw_path, raw)
            history["runs"].append(record)
            appended += 1
        if record.get("push_index") is not None:
            if raw_index is None or digest(raw_index) != record["push_index_sha256"]:
                raise ValueError("Receipt and push index digest differ")
            if re.fullmatch(r"runs/github-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*\.index\.json", record["push_index"]) is None:
                raise ValueError("Invalid push index path")
            manifest_path = safe_path(site, SITE_PATH / record["push_index"])
            if manifest_path.exists() and manifest_path.read_bytes() != raw_index:
                raise ValueError("Conflicting archived push index; refusing to overwrite it")
            if not manifest_path.exists():
                atomic_write(manifest_path, raw_index)
    if appended or not index_path.exists():
        history["runs"].sort(key=lambda item: (
            item["started_at"], item["source"].get("run_id", 0), item["source"].get("run_attempt", 0),
            item["source"].get("commit_index", 0), item["key"],
        ), reverse=True)
        atomic_write(index_path, json_bytes(history))
    for name in ("index.html", "styles.css", "dashboard.js"):
        destination = safe_path(site, SITE_PATH / name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets / name, destination)
    return {"appended": appended, "runs": len(history["runs"]), "site": str(index_path.parent)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect", help="Read one producer attempt using authenticated GitHub API metadata")
    collector.add_argument("--run-id", type=int, required=True)
    collector.add_argument("--attempt", type=int, required=True)
    collector.add_argument("--out", type=Path, required=True)
    collector.add_argument("--publisher-revision", required=True)
    appender = commands.add_parser("append", help="Append a collected receipt to a local Pages copy; never pushes")
    appender.add_argument("--site", type=Path, required=True)
    appender.add_argument("--record", type=Path)
    appender.add_argument("--assets", type=Path, default=ASSETS)
    manual = commands.add_parser("preview", help="Render manual data separately; never authenticates or publishes it")
    manual.add_argument("--input", type=Path, required=True)
    manual.add_argument("--label", required=True)
    manual.add_argument("--site", type=Path, required=True)
    manual.add_argument("--assets", type=Path, default=ASSETS)
    manual.add_argument("--publisher-revision")
    args = parser.parse_args()
    try:
        if args.command == "collect":
            if args.run_id < 1 or args.attempt < 1 or not revision(args.publisher_revision):
                raise ValueError("Expected positive run/attempt ids and a full publisher revision")
            records, raw_index = collect(args.run_id, args.attempt, args.publisher_revision)
            args.out.mkdir(parents=True, exist_ok=True)
            for record, raw in records:
                if raw is not None:
                    atomic_write(args.out / archive_filename(record), raw)
            if raw_index is not None:
                atomic_write(args.out / "push-index.json", raw_index)
            atomic_write(args.out / "receipt.json", json_bytes([record for record, _ in records]))
            result = {"records": [{"key": record["key"], "publication_status": record["publication_status"]} for record, _ in records]}
        elif args.command == "preview":
            raw = args.input.read_bytes()
            result = append_site(args.site, [(preview(raw, args.label, args.publisher_revision), raw)], assets=args.assets)
        else:
            receipts = read_json((args.record / "receipt.json").read_bytes()) if args.record else []
            records = [(record, (args.record / archive_filename(record)).read_bytes() if record["sha256"] else None) for record in receipts]
            raw_index = (args.record / "push-index.json").read_bytes() if any(record.get("push_index") for record in receipts) else None
            result = append_site(args.site, records, raw_index, args.assets)
        print(json.dumps(result))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"benchmark history: {error}\n")


if __name__ == "__main__":
    main()
