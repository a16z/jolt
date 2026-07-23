#!/usr/bin/env python3
"""Coverage gate for the jolt-verifier dependency closure.

Enforces the invariants of specs/test-quality-ci.md:
  1. closure correctness  — the in-scope set is the cargo-metadata normal-dep
     closure of the scope root, union the declared extras
  2. floors completeness  — [floors] and the in-scope set match exactly
  3. coverage floor       — per-crate cumulative line coverage >= floor

Subcommands:
  self-test                          run the embedded unit tests
  print-closure                      resolved in-scope crates as JSON
  check-config                       invariants 1-2 + declared feature paths exist
  plan                               emit the llvm-cov run plan as JSON (one entry
                                     per required invocation; default-only crates
                                     are grouped into a single run)
  enforce --coverage-json PATH ...   invariant 3 against llvm-cov JSON exports
  measure --coverage-json PATH ...   print per-crate coverage, no floor check

A [paths] entry is a list whose items are either the string "default", a
feature string ("prover-fixtures,zk"), or a table
{ features = "prover-fixtures", release = true } for paths that must run in
release mode (e.g. fixture generation via the legacy prover).

`--coverage-json` may be given multiple times — one llvm-cov export per
feature path. Runs are merged at line level from segment data: a line is
instrumented if any run instruments it and covered if any run executes it,
which is the "cumulative across declared feature paths" metric of the spec.
(The separate `cargo llvm-cov report` accumulation flow mismatches rebuilt
binaries across feature sets, so each path gets its own single-shot export.)

Only files under a crate's src/ directory count toward its coverage: floors
measure shipped code, not test/bench/fuzz harnesses.

Stdlib-only (requires Python 3.11+ for tomllib).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"python 3.11+ required for tomllib (running {sys.version.split()[0]})")
import tomllib  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / "coverage-floors.toml"


def load_cargo_metadata(manifest_dir: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=manifest_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def workspace_packages(metadata: dict) -> dict[str, dict]:
    return {p["name"]: p for p in metadata["packages"]}


def dependency_closure(metadata: dict, root: str) -> set[str]:
    """Workspace-member closure of `root` over normal dependencies only."""
    packages = workspace_packages(metadata)
    if root not in packages:
        raise SystemExit(f"scope root '{root}' is not a workspace member")
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for dep in packages[name]["dependencies"]:
            if dep["name"] in packages and dep["kind"] is None:
                stack.append(dep["name"])
    return seen


def normalize_path_entry(entry) -> dict:
    """Normalize a [paths] item to {"features": str | None, "release": bool}."""
    if entry == "default":
        return {"features": None, "release": False}
    if isinstance(entry, str):
        return {"features": entry, "release": False}
    if isinstance(entry, dict):
        return {"features": entry.get("features"), "release": bool(entry.get("release", False))}
    raise SystemExit(f"unrecognized [paths] entry: {entry!r}")


def in_scope_crates(metadata: dict, config: dict) -> dict[str, dict]:
    """Invariant 1: closure ∪ extras, with per-crate source dir and feature paths."""
    packages = workspace_packages(metadata)
    root = config["scope"]["root"]
    closure = dependency_closure(metadata, root)
    if not closure or root not in closure:
        raise SystemExit("closure invariant violated: empty or missing scope root")

    extras = config["scope"].get("extras", [])
    for extra in extras:
        if extra not in packages:
            raise SystemExit(f"extra '{extra}' is not a workspace member")
    scope = sorted(closure | set(extras))

    declared_paths = config.get("paths", {})
    for crate in declared_paths:
        if crate not in scope:
            raise SystemExit(f"[paths] entry '{crate}' is not in scope")

    crates = {}
    for name in scope:
        pkg = packages[name]
        paths = [normalize_path_entry(e) for e in declared_paths.get(name, ["default"])]
        for path in paths:
            if path["features"] is None:
                continue
            for feature in path["features"].split(","):
                if feature not in pkg["features"]:
                    raise SystemExit(
                        f"crate '{name}' declares path '{path['features']}' but "
                        f"has no feature '{feature}'"
                    )
        crates[name] = {
            "dir": str(Path(pkg["manifest_path"]).parent),
            "paths": paths,
        }
    return crates


def plan_runs(crates: dict[str, dict]) -> list[dict]:
    """One llvm-cov invocation per required run.

    Every crate's default path joins one grouped run; each feature path gets
    its own run with package-prefixed feature names so it can be launched
    from the workspace root.
    """
    default_packages = sorted(
        name
        for name, info in crates.items()
        if any(p["features"] is None and not p["release"] for p in info["paths"])
    )
    runs = []
    if default_packages:
        runs.append(
            {"slug": "default", "packages": default_packages, "features": None, "release": False}
        )
    for name in sorted(crates):
        for path in crates[name]["paths"]:
            if path["features"] is None and not path["release"]:
                continue
            features = path["features"]
            prefixed = (
                ",".join(f"{name}/{feature}" for feature in features.split(","))
                if features
                else None
            )
            slug = name
            if features:
                slug += "--" + features.replace(",", "+")
            if path["release"]:
                slug += "--release"
            runs.append(
                {
                    "slug": slug,
                    "packages": [name],
                    "features": prefixed,
                    "release": path["release"],
                }
            )
    return runs


def check_floors(config: dict, scope: set[str]) -> None:
    """Invariant 2: [floors] and the in-scope set match exactly, both directions."""
    floors = set(config["floors"])
    missing = sorted(scope - floors)
    stale = sorted(floors - scope)
    problems = []
    if missing:
        problems.append(f"in-scope crates without a floor entry: {missing}")
    if stale:
        problems.append(f"floor entries for crates no longer in scope: {stale}")
    if problems:
        raise SystemExit(
            "coverage-floors.toml drift (update [floors] deliberately):\n  "
            + "\n  ".join(problems)
        )


def line_execution_counts(file_entry: dict) -> dict[int, int]:
    """Per-line max execution count reconstructed from llvm-cov segments.

    A segment [line, col, count, has_count, is_region_entry, is_gap_region]
    applies from its position up to (exclusive) the next segment's line; gap
    regions terminate ranges but never mark lines themselves.
    """
    segments = file_entry.get("segments", [])
    counts: dict[int, int] = {}
    for idx, seg in enumerate(segments):
        line, _col, count, has_count, _entry, is_gap = seg[:6]
        if not has_count or is_gap:
            continue
        end_line = segments[idx + 1][0] if idx + 1 < len(segments) else line + 1
        for covered_line in range(line, max(end_line, line + 1)):
            counts[covered_line] = max(counts.get(covered_line, 0), count)
    return counts


def merge_line_coverage(coverage_jsons: list[dict]) -> dict[str, tuple[set[int], set[int]]]:
    """filename -> (instrumented lines, covered lines), unioned across runs."""
    merged: dict[str, tuple[set[int], set[int]]] = {}
    for coverage_json in coverage_jsons:
        for file_entry in coverage_json["data"][0]["files"]:
            instrumented, covered = merged.setdefault(file_entry["filename"], (set(), set()))
            for line, count in line_execution_counts(file_entry).items():
                instrumented.add(line)
                if count > 0:
                    covered.add(line)
    return merged


def per_crate_coverage(
    coverage_jsons: list[dict], crates: dict[str, dict]
) -> dict[str, dict]:
    """Attribute merged line coverage to crates; only <crate dir>/src/ counts."""
    merged = merge_line_coverage(coverage_jsons)
    totals = {name: {"covered": 0, "count": 0} for name in crates}
    src_prefixes = {name: str(Path(info["dir"]) / "src") + "/" for name, info in crates.items()}
    for filename, (instrumented, covered) in merged.items():
        for name, prefix in src_prefixes.items():
            if filename.startswith(prefix):
                totals[name]["covered"] += len(covered)
                totals[name]["count"] += len(instrumented)
                break
    for name, t in totals.items():
        t["percent"] = 100.0 * t["covered"] / t["count"] if t["count"] else 0.0
    return totals


def enforce_floors(coverage: dict[str, dict], floors: dict[str, float]) -> list[str]:
    failures = []
    for name in sorted(floors):
        got = coverage.get(name)
        if got is None or got["count"] == 0:
            failures.append(
                f"{name}: no measured source lines — crate missing from the "
                "coverage run (measurement bug, not a pass)"
            )
            continue
        if got["percent"] + 1e-9 < floors[name]:
            failures.append(
                f"{name}: {got['percent']:.1f}% < floor {floors[name]}% "
                f"({got['covered']}/{got['count']} lines)"
            )
    return failures


def cmd_print_closure(metadata: dict, config: dict) -> int:
    crates = in_scope_crates(metadata, config)
    json.dump(crates, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


def cmd_check_config(metadata: dict, config: dict) -> int:
    crates = in_scope_crates(metadata, config)
    check_floors(config, set(crates))
    print(f"config OK: {len(crates)} in-scope crates, floors complete")
    return 0


def cmd_plan(metadata: dict, config: dict) -> int:
    crates = in_scope_crates(metadata, config)
    json.dump(plan_runs(crates), sys.stdout, indent=2)
    print()
    return 0


def cmd_measure(metadata: dict, config: dict, coverage_jsons: list[dict]) -> int:
    crates = in_scope_crates(metadata, config)
    coverage = per_crate_coverage(coverage_jsons, crates)
    for name in sorted(coverage):
        c = coverage[name]
        print(f"{name:24} {c['percent']:6.1f}%  {c['covered']}/{c['count']}")
    return 0


def cmd_enforce(metadata: dict, config: dict, coverage_jsons: list[dict]) -> int:
    crates = in_scope_crates(metadata, config)
    check_floors(config, set(crates))
    coverage = per_crate_coverage(coverage_jsons, crates)
    failures = enforce_floors(coverage, config["floors"])
    for name in sorted(coverage):
        c = coverage[name]
        floor = config["floors"][name]
        status = "FAIL" if any(f.startswith(f"{name}:") for f in failures) else "ok"
        print(f"{status:4} {name:24} {c['percent']:6.1f}%  (floor {floor}%)")
    if failures:
        print("\ncoverage floors violated:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nall {len(coverage)} in-scope crates meet their floors")
    return 0


class GateTests(unittest.TestCase):
    @staticmethod
    def metadata():
        def pkg(name, deps, features=None, dir_=None):
            return {
                "name": name,
                "manifest_path": f"/ws/{dir_ or name}/Cargo.toml",
                "features": features or {},
                "dependencies": deps,
            }

        def dep(name, kind=None):
            return {"name": name, "kind": kind}

        return {
            "packages": [
                pkg("verifier", [dep("claims"), dep("dory", kind="dev")]),
                pkg("claims", [dep("field")], features={"extra": []}),
                pkg("field", [], features={"solinas": []}),
                pkg("dory", []),
                pkg("prover", [dep("verifier"), dep("claims")]),
            ]
        }

    @staticmethod
    def config(**overrides):
        base = {
            "scope": {"root": "verifier", "extras": ["dory"]},
            "paths": {"claims": ["default", "extra"]},
            "floors": {"verifier": 50, "claims": 50, "field": 50, "dory": 50},
        }
        base.update(overrides)
        return base

    def test_closure_follows_normal_deps_only(self):
        closure = dependency_closure(self.metadata(), "verifier")
        self.assertEqual(closure, {"verifier", "claims", "field"})

    def test_scope_includes_extras_not_reverse_deps(self):
        crates = in_scope_crates(self.metadata(), self.config())
        self.assertEqual(set(crates), {"verifier", "claims", "field", "dory"})
        self.assertNotIn("prover", crates)

    def test_unknown_extra_rejected(self):
        cfg = self.config(scope={"root": "verifier", "extras": ["nope"]})
        with self.assertRaises(SystemExit):
            in_scope_crates(self.metadata(), cfg)

    def test_unknown_feature_path_rejected(self):
        cfg = self.config(paths={"claims": ["default", "no-such-feature"]})
        with self.assertRaises(SystemExit):
            in_scope_crates(self.metadata(), cfg)

    def test_paths_entry_outside_scope_rejected(self):
        cfg = self.config(paths={"prover": ["default"]})
        with self.assertRaises(SystemExit):
            in_scope_crates(self.metadata(), cfg)

    def test_plan_groups_default_and_isolates_feature_paths(self):
        cfg = self.config(
            paths={
                "claims": ["default", "extra"],
                "field": ["default", {"features": "solinas", "release": True}],
            }
        )
        runs = plan_runs(in_scope_crates(self.metadata(), cfg))
        by_slug = {run["slug"]: run for run in runs}
        self.assertEqual(
            by_slug["default"]["packages"], ["claims", "dory", "field", "verifier"]
        )
        self.assertEqual(by_slug["claims--extra"]["features"], "claims/extra")
        self.assertFalse(by_slug["claims--extra"]["release"])
        solinas = by_slug["field--solinas--release"]
        self.assertEqual(solinas["features"], "field/solinas")
        self.assertTrue(solinas["release"])

    def test_floor_drift_missing_entry(self):
        cfg = self.config(floors={"verifier": 50, "claims": 50, "field": 50})
        with self.assertRaises(SystemExit):
            check_floors(cfg, {"verifier", "claims", "field", "dory"})

    def test_floor_drift_stale_entry(self):
        cfg = self.config(
            floors={"verifier": 50, "claims": 50, "field": 50, "dory": 50, "gone": 50}
        )
        with self.assertRaises(SystemExit):
            check_floors(cfg, {"verifier", "claims", "field", "dory"})

    @staticmethod
    def entry(filename, covered_lines, uncovered_lines):
        segments = []
        for line in sorted(set(covered_lines) | set(uncovered_lines)):
            count = 1 if line in covered_lines else 0
            segments.append([line, 1, count, True, True, False])
        return {"filename": filename, "segments": segments}

    @classmethod
    def coverage_json(cls):
        return {
            "data": [
                {
                    "files": [
                        cls.entry("/ws/verifier/src/lib.rs", range(1, 91), range(91, 101)),
                        cls.entry("/ws/verifier/tests/e2e.rs", [], range(1, 51)),  # excluded
                        cls.entry("/ws/claims/src/a.rs", range(1, 41), range(41, 101)),
                        cls.entry("/ws/claims/src/b.rs", range(1, 21), range(21, 101)),
                        cls.entry("/ws/field/src/lib.rs", range(1, 101), []),
                        cls.entry("/ws/prover/src/lib.rs", [], range(1, 101)),  # out of scope
                    ]
                }
            ]
        }

    def test_attribution_src_only_and_scoped(self):
        crates = in_scope_crates(self.metadata(), self.config())
        cov = per_crate_coverage([self.coverage_json()], crates)
        self.assertEqual(cov["verifier"], {"covered": 90, "count": 100, "percent": 90.0})
        self.assertEqual(cov["claims"]["percent"], 30.0)
        self.assertEqual(cov["dory"]["count"], 0)

    def test_multi_run_merge_unions_lines(self):
        crates = in_scope_crates(self.metadata(), self.config())
        run2 = {
            "data": [
                {
                    "files": [
                        # second feature path covers lines the first missed
                        self.entry("/ws/verifier/src/lib.rs", range(91, 101), range(1, 91)),
                        # and instruments a feature-gated file
                        self.entry("/ws/verifier/src/zk.rs", range(1, 11), []),
                    ]
                }
            ]
        }
        cov = per_crate_coverage([self.coverage_json(), run2], crates)
        self.assertEqual(cov["verifier"], {"covered": 110, "count": 110, "percent": 100.0})

    def test_gap_regions_do_not_instrument_lines(self):
        entry = {
            "filename": "/ws/field/src/lib.rs",
            "segments": [
                [1, 1, 5, True, True, False],
                [3, 1, 0, True, False, True],  # gap: terminates, marks nothing
                [5, 1, 0, True, True, False],
            ],
        }
        merged = merge_line_coverage([{"data": [{"files": [entry]}]}])
        instrumented, covered = merged["/ws/field/src/lib.rs"]
        self.assertEqual(instrumented, {1, 2, 5})
        self.assertEqual(covered, {1, 2})

    def test_enforce_reports_failures_and_missing_measurement(self):
        crates = in_scope_crates(self.metadata(), self.config())
        cov = per_crate_coverage([self.coverage_json()], crates)
        failures = enforce_floors(cov, {"verifier": 90, "claims": 50, "dory": 10})
        self.assertEqual(len(failures), 2)
        self.assertTrue(any(f.startswith("claims: 30.0%") for f in failures))
        self.assertTrue(any(f.startswith("dory:") for f in failures))

    def test_enforce_passes_at_exact_floor(self):
        crates = in_scope_crates(self.metadata(), self.config())
        cov = per_crate_coverage([self.coverage_json()], crates)
        self.assertEqual(enforce_floors(cov, {"verifier": 90}), [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "print-closure", "check-config", "plan", "enforce", "measure"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coverage-json", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.command == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1

    config = tomllib.loads(args.config.read_text())
    metadata = load_cargo_metadata(args.config.parent.parent)

    if args.command == "print-closure":
        return cmd_print_closure(metadata, config)
    if args.command == "check-config":
        return cmd_check_config(metadata, config)
    if args.command == "plan":
        return cmd_plan(metadata, config)

    if not args.coverage_json:
        parser.error(f"{args.command} requires at least one --coverage-json")
    coverage_jsons = [json.loads(path.read_text()) for path in args.coverage_json]
    if args.command == "measure":
        return cmd_measure(metadata, config, coverage_jsons)
    return cmd_enforce(metadata, config, coverage_jsons)


if __name__ == "__main__":
    sys.exit(main())
