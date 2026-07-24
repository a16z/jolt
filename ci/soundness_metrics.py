#!/usr/bin/env python3
"""Soundness-coverage metrics for the jolt-verifier dependency closure.

Implements invariant 5 of specs/test-quality-ci.md:

* **error-variant coverage** — the fraction of error-enum variants whose
  reference sites execute during the coverage run. A variant counts as
  exercised when any line mentioning `EnumName::Variant` (outside the enum's
  own definition) has a nonzero llvm-cov execution count. Approximate by
  construction: variants with no locatable reference site are reported as
  `unlocatable` and excluded from the denominator rather than silently
  counted either way.

* **tamper-manifest active ratio** — the fraction of `TamperTarget`s in
  jolt-verifier's tamper manifest whose coverage is `TamperCoverage::Active`
  (vs `IgnoredUntilFixture` / `Deferred`).

Subcommands:
  self-test
  error-variants --coverage-json PATH    report + enforce [soundness.error_variants]
  tamper-ratio                           report + enforce tamper_active_ratio_min

Stdlib-only (requires Python 3.11+ for tomllib).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"python 3.11+ required for tomllib (running {sys.version.split()[0]})")
import tomllib  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import coverage_gate  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / "coverage-floors.toml"

ENUM_RE = re.compile(r"^\s*pub enum ([A-Za-z0-9_]*Error)\b")
VARIANT_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9_]*)\s*(?:\{|\(|,|$)")


def find_error_enums(source: str) -> dict[str, tuple[int, int, list[str]]]:
    """Map enum name -> (start line, end line, variants), 1-indexed inclusive."""
    lines = source.splitlines()
    enums: dict[str, tuple[int, int, list[str]]] = {}
    i = 0
    while i < len(lines):
        match = ENUM_RE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1)
        start = i + 1
        depth = 0
        opened = False
        variants: list[str] = []
        j = i
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                opened = True
            if opened and depth == 1 and j > i:
                stripped = lines[j].strip()
                if not stripped.startswith(("#", "//", "/*", "*")):
                    variant = VARIANT_RE.match(lines[j])
                    if variant:
                        variants.append(variant.group(1))
            if opened and depth == 0:
                break
            j += 1
        enums[name] = (start, j + 1, variants)
        i = j + 1
    return enums


line_execution_counts = coverage_gate.line_execution_counts


def merged_line_counts(coverage_jsons: list[dict]) -> dict[str, dict[int, int]]:
    """filename -> per-line max execution count, merged across runs."""
    merged: dict[str, dict[int, int]] = {}
    for coverage_json in coverage_jsons:
        for file_entry in coverage_json["data"][0]["files"]:
            counts = merged.setdefault(file_entry["filename"], defaultdict(int))
            for line, count in line_execution_counts(file_entry).items():
                counts[line] = max(counts[line], count)
    return merged


def crate_error_variant_report(
    crate_dir: Path,
    coverage_by_file: dict[str, dict[int, int]],
    read_text=lambda p: p.read_text(),
    src_files: list[Path] | None = None,
) -> dict:
    """For one crate: {variant id: exercised} + unlocatable variants."""
    if src_files is None:
        src_files = sorted((crate_dir / "src").rglob("*.rs"))
    sources = {path: read_text(path) for path in src_files}

    enums: dict[str, tuple[Path, int, int, list[str]]] = {}
    for path, source in sources.items():
        for name, (start, end, variants) in find_error_enums(source).items():
            enums[name] = (path, start, end, variants)

    exercised: dict[str, bool] = {}
    unlocatable: list[str] = []
    for name, (def_path, start, end, variants) in enums.items():
        for variant in variants:
            ref_re = re.compile(rf"\b{name}::{variant}\b")
            hit = False
            located = False
            for path, source in sources.items():
                for lineno, line in enumerate(source.splitlines(), start=1):
                    if path == def_path and start <= lineno <= end:
                        continue
                    if not ref_re.search(line):
                        continue
                    located = True
                    if coverage_by_file.get(str(path), {}).get(lineno, 0) > 0:
                        hit = True
                        break
                if hit:
                    break
            key = f"{name}::{variant}"
            if located:
                exercised[key] = hit
            else:
                unlocatable.append(key)
    return {"exercised": exercised, "unlocatable": unlocatable}


def tamper_ratio(manifest_source: str) -> tuple[int, int]:
    """(active, total) TamperCoverage occurrences, ignoring comparisons."""
    active = total = 0
    for line in manifest_source.splitlines():
        if "==" in line or "matches!" in line:
            continue
        for match in re.finditer(r"TamperCoverage::(Active|IgnoredUntilFixture|Deferred)\b", line):
            total += 1
            if match.group(1) == "Active":
                active += 1
    return active, total


def cmd_error_variants(config: dict, coverage_jsons: list[dict], repo_root: Path) -> int:
    metadata = coverage_gate.load_cargo_metadata(repo_root)
    crates = coverage_gate.in_scope_crates(metadata, config)
    coverage_by_file = merged_line_counts(coverage_jsons)

    floors = config.get("soundness", {}).get("error_variants", {})
    failures = []
    for name in sorted(crates):
        report = crate_error_variant_report(Path(crates[name]["dir"]), coverage_by_file)
        total = len(report["exercised"])
        if total == 0 and not report["unlocatable"]:
            continue
        hit = sum(report["exercised"].values())
        pct = 100.0 * hit / total if total else 0.0
        floor = floors.get(name)
        status = "ok"
        if floor is not None and pct + 1e-9 < floor:
            status = "FAIL"
            failures.append(f"{name}: {pct:.1f}% < floor {floor}%")
        floor_txt = f"(floor {floor}%)" if floor is not None else "(report-only)"
        print(f"{status:4} {name:24} {pct:5.1f}%  {hit}/{total} variants exercised {floor_txt}")
        for key in sorted(k for k, v in report["exercised"].items() if not v):
            print(f"       unexercised: {key}")
        for key in report["unlocatable"]:
            print(f"       unlocatable: {key}")
    if failures:
        print("\nerror-variant floors violated:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def cmd_tamper_ratio(config: dict, repo_root: Path) -> int:
    manifest = repo_root / "crates/jolt-verifier/tests/support/tamper_manifest.rs"
    active, total = tamper_ratio(manifest.read_text())
    if total == 0:
        print(f"no TamperCoverage entries found in {manifest}", file=sys.stderr)
        return 1
    ratio = active / total
    floor = config.get("soundness", {}).get("tamper_active_ratio_min", 0.0)
    print(f"tamper manifest: {active}/{total} targets active ({100 * ratio:.1f}%, floor {100 * floor:.1f}%)")
    if ratio + 1e-9 < floor:
        print("tamper-manifest active ratio below floor", file=sys.stderr)
        return 1
    return 0


class SoundnessTests(unittest.TestCase):
    SOURCE = """\
pub enum VerifyError {
    /// doc comment
    BadProof,
    #[serde(skip)]
    WrongStage { got: usize },
    Unreferenced(u32),
}

fn check(ok: bool) -> Result<(), VerifyError> {
    if !ok {
        return Err(VerifyError::BadProof);
    }
    Err(VerifyError::WrongStage { got: 1 })
}
"""

    def test_enum_parsing_skips_attrs_and_docs(self):
        enums = find_error_enums(self.SOURCE)
        self.assertEqual(list(enums), ["VerifyError"])
        _start, end, variants = enums["VerifyError"]
        self.assertEqual(variants, ["BadProof", "WrongStage", "Unreferenced"])
        self.assertEqual(end, 7)

    def test_variant_report_exercised_vs_not(self):
        path = Path("/fake/src/error.rs")
        # line 11 (Err BadProof) executed, line 13 (WrongStage) not
        coverage = {str(path): {11: 3, 13: 0}}
        report = crate_error_variant_report(
            Path("/fake"),
            coverage,
            read_text=lambda p: self.SOURCE,
            src_files=[path],
        )
        self.assertEqual(
            report["exercised"],
            {"VerifyError::BadProof": True, "VerifyError::WrongStage": False},
        )
        self.assertEqual(report["unlocatable"], ["VerifyError::Unreferenced"])

    def test_definition_lines_not_counted_as_references(self):
        path = Path("/fake/src/error.rs")
        # mark the whole file executed: definition lines must still not count
        coverage = {str(path): {n: 1 for n in range(1, 20)}}
        report = crate_error_variant_report(
            Path("/fake"),
            coverage,
            read_text=lambda p: self.SOURCE,
            src_files=[path],
        )
        self.assertIn("VerifyError::Unreferenced", report["unlocatable"])

    def test_merged_line_counts_takes_max_across_runs(self):
        def run(count):
            return {
                "data": [
                    {
                        "files": [
                            {
                                "filename": "/f.rs",
                                "segments": [[5, 1, count, True, True, False]],
                            }
                        ]
                    }
                ]
            }

        merged = merged_line_counts([run(0), run(9)])
        self.assertEqual(merged["/f.rs"][5], 9)

    def test_tamper_ratio_ignores_comparisons(self):
        source = """\
            coverage: TamperCoverage::Active,
            coverage: TamperCoverage::Deferred,
            coverage: TamperCoverage::Active,
            assert!(t.coverage == TamperCoverage::Active);
        """
        self.assertEqual(tamper_ratio(source), (2, 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "error-variants", "tamper-ratio"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coverage-json", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.command == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SoundnessTests)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1

    config = tomllib.loads(args.config.read_text())
    repo_root = args.config.parent.parent

    if args.command == "tamper-ratio":
        return cmd_tamper_ratio(config, repo_root)

    if not args.coverage_json:
        parser.error("error-variants requires at least one --coverage-json")
    coverage_jsons = [json.loads(path.read_text()) for path in args.coverage_json]
    return cmd_error_variants(config, coverage_jsons, repo_root)


if __name__ == "__main__":
    sys.exit(main())
