"""Synthetic contract fixtures. These are not measured benchmark results."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "ci/benchmark_history.py"
SPEC = importlib.util.spec_from_file_location("benchmark_history", SCRIPT)
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)
REVISION = "a" * 40
SECOND_REVISION = "b" * 40


def synthetic_result(run_id=101, attempt=1, tier="main", revision=REVISION):
    def sample(phase, repetition, seconds):
        return {
            "leg": "head", "revision": revision, "phase": phase,
            "repetition": repetition, "pair_index": None, "order": 0 if phase == "warmup" else repetition + 1,
            "status": "complete", "exit_code": 0, "reason": None,
            "metrics": {
                "e2e_s": seconds, "prove_s": seconds / 2, "setup_s": 2,
                "verify_s": 0.3, "verify_single_s": 0.5, "process_s": seconds + 1,
                "peak_rss_bytes": 104857600, "proof_bytes": 4096,
                "trace_rows": 65530, "padded_trace_rows": 65536,
            },
        }

    return {
        "schema_version": 1, "run_id": f"SYNTHETIC-FIXTURE-{run_id}",
        "tier": tier, "kind": "history", "status": "complete", "reason": None,
        "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:01:00Z",
        "repository": "a16z/jolt",
        "revisions": {"base": None, "head": revision, "merge": None, "harness": REVISION},
        "provenance": {
            "workflow": "End-to-end prover benchmarks", "workflow_file": ".github/workflows/e2e-bench.yml",
            "run_id": str(run_id), "run_attempt": attempt,
            "event": "schedule" if tier == "nightly" else "push", "ref": "refs/heads/main", "pr_number": None,
        },
        "suite": {"id": "synthetic", "config_sha256": "c" * 64, "metric_contract": 1, "warmups": 1, "repetitions": 2},
        "host": {
            "id": "SYNTHETIC host", "architecture": "x86_64", "cpu_model": "SYNTHETIC CPU",
            "kernel": "synthetic-kernel", "physical_cores": 24, "logical_cpus": 48,
            "memory_bytes": 192000000000, "topology": [{"cpu": i, "core": i % 24, "socket": 0} for i in range(48)],
            "settings": {"governor": "performance", "boost": True, "smt": True},
        },
        "builds": [{
            "schema_version": 1, "leg": "head", "revision": revision, "harness_revision": REVISION,
            "toolchain": "synthetic-rustc", "rustflags": "-C target-cpu=synthetic", "cargo_lock_sha256": "d" * 64,
            "cargo_version": "synthetic-cargo", "target_cpu": "synthetic", "target_triple": "x86_64-unknown-linux-gnu",
            "target_features": ["sse2"], "profile": "release", "source_cargo_lock_sha256": "d" * 64,
            "harness_files": {"profile.rs": "1" * 64}, "suite_config_sha256": "c" * 64,
            "binaries": {pcs: {"path": f"bin/{pcs}", "sha256": revision.ljust(64, "f")} for pcs in ("dory", "akita")},
            "guests": {"synthetic-fibonacci": {"elf": {"path": "guests/fib", "sha256": "2" * 64}, "inputs": {"100": {"path": "inputs/100", "sha256": "3" * 64}}}},
            "runner": {"path": "run.py", "sha256": "4" * 64}, "akita_schedules": [],
            "host": {"id": "SYNTHETIC build host", "architecture": "x86_64"},
            "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:01Z",
            "build_s": 1, "status": "complete", "reason": None,
            "cache": {"cargo_target_preexisting": False, "rustc_wrapper": None, "hit_status": "unestablished"},
        }],
        "wall_time": {"build_s": None, "benchmark_s": 60, "total_s": 60, "scope": "prebuilt execution only"},
        "cases": [{
            "id": "synthetic-fib-dory-1", "workload": "synthetic-fibonacci", "pcs": "dory", "backend": "optimized",
            "physical_cores": 1, "cpu_ids": [0], "scale_log2": 16,
            "input": {"parameters": {"iterations": 100}, "seed": 0, "sha256": "e" * 64},
            "status": "complete", "reason": None,
            "samples": [sample("warmup", 0, 900), sample("measure", 0, 10), sample("measure", 1, 14)],
        }],
    }


def synthetic_run(data):
    return {
        "id": int(data["provenance"]["run_id"]), "run_attempt": data["provenance"]["run_attempt"],
        "repository": {"full_name": "a16z/jolt"}, "head_repository": {"full_name": "a16z/jolt"},
        "path": ".github/workflows/e2e-bench.yml", "name": "End-to-end prover benchmarks",
        "head_branch": "main", "head_sha": data["revisions"]["head"], "event": data["provenance"]["event"],
        "status": "completed", "conclusion": "success", "run_started_at": data["started_at"],
    }


def collected_fixture(data, *, run=None, missing=False, extra_zip=None, artifact_change=None):
    run = run or synthetic_run(data)
    raw = (json.dumps(data, indent=4) + "\n").encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("benchmark-results.json", raw)
        for name, content in (extra_zip or {}).items():
            bundle.writestr(name, content)
    artifact = {
        "id": 42, "name": f"jolt-e2e-{run['id']}-{run['run_attempt']}", "expired": False,
        "size_in_bytes": len(archive.getvalue()), "digest": "sha256:synthetic",
        "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]},
    }
    artifact.update(artifact_change or {})

    def api(path):
        if "/attempts/" in path:
            return json.dumps(run).encode()
        if "/artifacts?" in path:
            return json.dumps({"artifacts": [] if missing else [artifact]}).encode()
        if path.endswith("/artifacts/42/zip"):
            return archive.getvalue()
        raise AssertionError(f"Unexpected API call: {path}")

    with mock.patch.object(history, "gh", side_effect=api):
        records, raw_index = history.collect(run["id"], run["run_attempt"], REVISION)
        assert raw_index is None and len(records) == 1
        return records[0]


def collected_push(*, omit_index_entry=False, bad_path=False, wrong_revision=False):
    before, first, missing, tip, side = (f"{n:040x}" for n in range(1, 6))
    data = synthetic_result(200, revision=tip)
    run = synthetic_run(data)
    documents = [
        {"revision": first, "path": f"commits/{first}/benchmark-results.json"},
        {"revision": missing, "path": f"commits/{missing}/benchmark-results.json"},
        {"revision": tip, "path": "benchmark-results.json"},
    ]
    if omit_index_entry:
        documents.pop(1)
    if bad_path:
        documents[0]["path"] = "../benchmark-results.json"
    raw_index = json.dumps({"schema_version": 1, "push_before": before, "push_after": tip, "documents": documents}).encode()
    earlier = synthetic_result(200, revision=tip if wrong_revision else first)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("index.json", raw_index)
        bundle.writestr("benchmark-results.json", json.dumps(data))
        bundle.writestr(f"commits/{first}/benchmark-results.json", json.dumps(earlier))
    comparison = {
        "status": "ahead", "base_commit": {"sha": before}, "total_commits": 4,
        "commits": [
            {"sha": first, "parents": [{"sha": before}]},
            {"sha": side, "parents": [{"sha": before}]},
            {"sha": missing, "parents": [{"sha": first}]},
            {"sha": tip, "parents": [{"sha": missing}, {"sha": side}]},
        ],
    }
    artifact = {"id": 42, "name": "jolt-e2e-200-1", "expired": False, "size_in_bytes": len(archive.getvalue()),
                "workflow_run": {"id": 200, "head_sha": tip}}

    def api(path):
        if "/attempts/" in path:
            return json.dumps(run).encode()
        if "/artifacts?" in path:
            return json.dumps({"artifacts": [artifact]}).encode()
        if "/compare/" in path:
            return json.dumps(comparison).encode()
        if path.endswith("/artifacts/42/zip"):
            return archive.getvalue()
        raise AssertionError(path)

    with mock.patch.object(history, "gh", side_effect=api):
        return history.collect(200, 1, REVISION)


class BenchmarkHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.site = Path(self.directory.name) / "site"

    def index(self):
        return json.loads((self.site / "dev/prover-history/history.json").read_text())

    def test_append_is_immutable_idempotent_and_preserves_legacy(self):
        legacy = self.site / "dev/bench/data.js"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"window.BENCHMARK_DATA = {old: 'keep exactly'};\n")
        data = synthetic_result()
        record, raw = collected_fixture(data)
        self.assertTrue(history.append_site(self.site, [(record, raw)])["appended"])
        before = {p.relative_to(self.site): p.read_bytes() for p in self.site.rglob("*") if p.is_file()}
        self.assertFalse(history.append_site(self.site, [(record, raw)])["appended"])
        self.assertEqual(before, {p.relative_to(self.site): p.read_bytes() for p in self.site.rglob("*") if p.is_file()})
        self.assertEqual((self.site / "dev/prover-history" / self.index()["runs"][0]["raw"]).read_bytes(), raw)
        data["cases"][0]["samples"][1]["metrics"]["e2e_s"] = 99
        conflicting, changed = collected_fixture(data)
        with self.assertRaisesRegex(ValueError, "immutable"):
            history.append_site(self.site, [(conflicting, changed)])
        self.assertEqual(before, {p.relative_to(self.site): p.read_bytes() for p in self.site.rglob("*") if p.is_file()})
        missing, _ = collected_fixture(data, missing=True)
        self.assertFalse(history.append_site(self.site, [(missing, None)])["appended"])
        self.assertEqual(self.index()["runs"][0]["publication_status"], "ready")

    def test_tiers_inputs_settings_and_rerun_attempts_remain_distinct(self):
        documents = [synthetic_result(), synthetic_result(attempt=2), synthetic_result(102, tier="nightly"), synthetic_result(103)]
        documents[2]["cases"][0]["scale_log2"] = 20
        documents[3]["host"]["settings"]["boost"] = False
        documents[3]["builds"][0]["rustflags"] = "different flags"
        for data in documents:
            record, raw = collected_fixture(data)
            history.append_site(self.site, [(record, raw)])
        entries = self.index()["runs"]
        self.assertEqual(len(entries), 4)
        self.assertEqual({item["source"]["run_attempt"] for item in entries}, {1, 2})
        archived = [json.loads((self.site / "dev/prover-history" / entry["raw"]).read_bytes()) for entry in entries]
        self.assertEqual({item["tier"] for item in archived}, {"main", "nightly"})
        self.assertEqual({item["cases"][0]["scale_log2"] for item in archived}, {16, 20})
        self.assertEqual({item["host"]["settings"]["boost"] for item in archived}, {True, False})

    def test_multi_commit_push_keeps_first_parent_order_and_missing_documents(self):
        records, raw_index = collected_push()
        self.assertEqual([record["source"]["commit"] for record, _ in records], [f"{n:040x}" for n in (2, 3, 4)])
        self.assertEqual([record["publication_status"] for record, _ in records], ["ready", "missing", "ready"])
        self.assertTrue(all(record["source"]["push_range"]["ancestry_verified"] for record, _ in records))
        self.assertTrue(all(not record["source"]["push_range"]["event_bound_verified"] for record, _ in records))
        self.assertEqual(history.append_site(self.site, records, raw_index)["appended"], 3)
        self.assertEqual(history.append_site(self.site, records, raw_index)["appended"], 0)
        self.assertEqual([record["source"]["commit"] for record in self.index()["runs"]], [f"{n:040x}" for n in (4, 3, 2)])
        self.assertEqual((self.site / "dev/prover-history" / self.index()["runs"][0]["push_index"]).read_bytes(), raw_index)
        omitted, _ = collected_push(omit_index_entry=True)
        self.assertEqual([record["publication_status"] for record, _ in omitted], ["ready", "missing", "ready"])

    def test_multi_commit_document_paths_and_measured_revisions_are_bound(self):
        records, _ = collected_push(bad_path=True)
        self.assertEqual(records[0][0]["publication_status"], "rejected")
        records, _ = collected_push(wrong_revision=True)
        self.assertEqual([record["publication_status"] for record, _ in records], ["rejected", "missing", "ready"])

    def test_failed_disabled_and_missing_results_are_archived_without_values(self):
        failed = synthetic_result()
        failed["status"] = failed["cases"][0]["status"] = "failed"
        failed["cases"][0]["samples"][-1].update(status="failed", exit_code=1)
        failed["cases"][0]["samples"][-1]["metrics"] = {name: None for name in history.METRICS}
        failed["cases"].append(dict(copy.deepcopy(failed["cases"][0]), id="missing-case", physical_cores=8, cpu_ids=list(range(8)), status="missing", samples=[]))
        record, raw = collected_fixture(failed)
        history.append_site(self.site, [(record, raw)])
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(json.loads(raw)["cases"][0]["samples"][-1]["metrics"]["e2e_s"])
        disabled = synthetic_result(102)
        disabled.update(status="disabled", cases=[], builds=[], reason="Runner not enabled")
        record, raw = collected_fixture(disabled)
        history.append_site(self.site, [(record, raw)])
        missing, _ = collected_fixture(synthetic_result(103), missing=True)
        history.append_site(self.site, [(missing, None)])
        self.assertEqual({entry["status"] for entry in self.index()["runs"]}, {"failed", "disabled", "missing"})
        self.assertIsNone(next(entry for entry in self.index()["runs"] if entry["status"] == "missing")["raw"])

    def test_github_run_and_artifact_provenance_are_checked(self):
        data = synthetic_result()
        real_run = synthetic_run(data)
        for change in (
            {"event": "pull_request"}, {"head_repository": {"full_name": "outsider/jolt"}},
            {"path": ".github/workflows/other.yml"}, {"status": "in_progress"},
        ):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Refusing"):
                collected_fixture(data, run=dict(real_run, **change))
        data["provenance"]["run_attempt"] = 99
        record, raw = collected_fixture(data, run=real_run)
        self.assertEqual(record["publication_status"], "rejected")
        history.append_site(self.site, [(record, raw)])
        self.assertIsNotNone(self.index()["runs"][0]["raw"])
        record, raw = collected_fixture(synthetic_result(), artifact_change={"workflow_run": {"id": 999, "head_sha": REVISION}})
        self.assertEqual(record["publication_status"], "rejected")
        self.assertIsNone(raw)

    def test_archive_contents_are_never_extracted_or_executed(self):
        data = synthetic_result()
        data["cases"][0]["reason"] = '<img src="https://invalid.example/attack" onerror="alert(1)">'
        record, raw = collected_fixture(data, extra_zip={
            "../../escape.py": "raise RuntimeError('executed')", "index.html": "malicious HTML",
            "post.sh": "exit 99", "build/manifest.json": '{"path":"/etc/passwd"}',
        })
        history.append_site(self.site, [(record, raw)])
        self.assertEqual((self.site / "dev/prover-history/index.html").read_bytes(), (history.ASSETS / "index.html").read_bytes())
        self.assertEqual(len(list(self.site.rglob("*.json"))), 2)
        self.assertFalse((Path(self.directory.name) / "escape.py").exists())
        self.assertIn(b"onerror", raw)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../../benchmark-results.json", raw)
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as bundle:
            self.assertIsNone(history.artifact_json(bundle, "benchmark-results.json"))

    def test_unknown_schema_nonfinite_metrics_and_duplicate_points_are_rejected(self):
        for mutate in (
            lambda data: data.update(schema_version=2),
            lambda data: data["cases"][0]["samples"][1]["metrics"].update(e2e_s=float("nan")),
            lambda data: data["cases"][0]["samples"][1].update(status="ok"),
            lambda data: data["cases"].append(dict(copy.deepcopy(data["cases"][0]), id="duplicate")),
        ):
            data = synthetic_result()
            mutate(data)
            record, raw = collected_fixture(data)
            self.assertEqual(record["publication_status"], "rejected")
            self.assertIsNotNone(raw)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            history.read_json(b'{"value":1e999}')

    def test_history_paths_cannot_follow_symlinks_into_legacy_or_external_files(self):
        (self.site / "dev").mkdir(parents=True)
        for target in (Path(self.directory.name) / "outside", self.site / "dev/bench"):
            with self.subTest(target=target):
                target.mkdir()
                sentinel = target / "index.html"
                sentinel.write_bytes(b"do not overwrite")
                link = self.site / "dev/prover-history"
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaises(ValueError):
                    history.append_site(self.site)
                self.assertEqual(sentinel.read_bytes(), b"do not overwrite")
                self.assertEqual(list(target.iterdir()), [sentinel])
                link.unlink()


if __name__ == "__main__":
    unittest.main()
