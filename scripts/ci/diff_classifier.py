#!/usr/bin/env python3
"""Classify a PR diff's added/removed lines into code / tests / docs /
fixtures / helper scripts.

Every changed line gets exactly one category, so the per-category counts sum
to the total diff. Classification order (first match wins):

1. fixtures — Cargo.lock, `*.snap` anywhere, and data-extension files
   (.json, .bin, .txt, ...) living under a fixture-ish directory component
   (fixtures/, testdata/, golden/, snapshots/, corpus/, ...) or under a
   tests/ / benches/ directory.
2. docs — `*.md`/`*.mdx` anywhere, and anything under a docs/, specs/, or
   book/ directory component.
3. tests (file level) — anything under a tests/ or benches/ directory
   component, or source files named like tests (test.rs, tests.rs,
   test_*.py, *_test.rs, *_tests.rs, bench*.rs, *_bench.rs, ...). Tests
   beat helper scripts: a test-named file under scripts/ counts as tests.
4. helper scripts (path level) — anything under a scripts/ or tools/
   directory component, including Rust helpers living there.
5. actual code — Rust/jolt code only: `.rs` files (line level: lines inside
   `#[cfg(test)]` items / `#[test]`-attributed functions are tests, with a
   test region winning over doc comments inside it; remaining pure
   doc-comment lines `///` / `//!` are docs; the rest is code), plus
   Cargo.toml manifests and native guest sources (.c/.h/.S/.ld/...).
6. helper scripts (catch-all) — everything else: CI workflows (.yml/.yaml),
   .py/.sh tooling, justfile/Makefile-type files, lint/deploy configs, and
   any other non-Rust support files.

Added lines are classified against the NEW file (head revision), removed
lines against the OLD file (merge-base revision), so `#[cfg(test)]` region
membership is resolved at the revision where the line actually exists.

Stdlib-only; used by .github/workflows/diff-classifier.yml.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field

MINUS = "−"  # typographic minus used in the plain-text summary

CATEGORIES = ("code", "tests", "docs", "fixtures", "helper")

CATEGORY_LABELS = {
    "code": "Diff in the actual code",
    "tests": "Diff in tests",
    "docs": "Diff in docs",
    "fixtures": "Diff in fixtures",
    "helper": "Diff in helper scripts",
}

# GitHub-rendered labels for the sticky PR comment.
MARKDOWN_CATEGORIES = {
    "code": ("⌨️", "Actual code"),
    "tests": ("🧪", "Tests"),
    "docs": ("📚", "Docs"),
    "fixtures": ("📦", "Fixtures"),
    "helper": ("🔧", "Helper scripts"),
}

COMMENT_MARKER = "<!-- diff-classifier -->"

# Data files under these directory components are fixtures.
FIXTURE_DIR_COMPONENTS = {
    "fixtures",
    "fixture",
    "testdata",
    "test_data",
    "test-data",
    "golden",
    "snapshots",
    "snapshot",
    "corpus",
    "vectors",
}

# Directory components whose data files count as fixtures too (test inputs).
TEST_DIR_COMPONENTS = {"tests", "benches"}

DATA_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".bin",
    ".snap",
    ".txt",
    ".csv",
    ".tsv",
    ".hex",
    ".dat",
    ".elf",
    ".wasm",
    ".gz",
    ".zst",
    ".tar",
    ".proof",
}

DOC_DIR_COMPONENTS = {"docs", "specs", "book"}

DOC_EXTENSIONS = {".md", ".mdx"}

# Source files named like tests (any implementation language).
TEST_FILE_STEM = re.compile(
    r"^(?:tests?|test_.*|.*_tests?|bench(?:es)?|bench_.*|.*_bench)$"
)
TEST_FILE_EXTENSIONS = {".rs", ".py", ".sh", ".go", ".js", ".ts"}

# Tooling directories: their contents are helper scripts even when Rust.
HELPER_DIR_COMPONENTS = {"scripts", "tools"}

# "Actual code" beyond .rs: manifests and native guest sources.
CODE_BASENAMES = {"Cargo.toml"}
CODE_EXTENSIONS = {".s", ".c", ".h", ".cc", ".cpp", ".hpp", ".ld", ".x"}

# #[test], #[tokio::test], #[test_case(...)], #[rstest], ... — an attribute
# whose final path segment is `test`-like marks the following item as a test.
TEST_ATTR = re.compile(
    r"^#\s*\[\s*(?:[A-Za-z_]\w*\s*::\s*)*(?:test|rstest|test_case|quickcheck|proptest)\s*(?:\]|\()"
)


def file_extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    return name[dot:].lower() if dot > 0 else ""


def classify_path(path: str) -> str | None:
    """File-level category, or None when a Rust file needs line-level analysis."""
    components = path.split("/")
    basename = components[-1]
    dirs = set(components[:-1])
    ext = file_extension(path)

    # 1. fixtures
    if basename == "Cargo.lock" or ext == ".snap":
        return "fixtures"
    if ext in DATA_EXTENSIONS and dirs & (FIXTURE_DIR_COMPONENTS | TEST_DIR_COMPONENTS):
        return "fixtures"

    # 2. docs
    if ext in DOC_EXTENSIONS or dirs & DOC_DIR_COMPONENTS:
        return "docs"

    # 3. tests (file level) — beats helper scripts for test-named files
    if dirs & TEST_DIR_COMPONENTS:
        return "tests"
    if ext in TEST_FILE_EXTENSIONS and TEST_FILE_STEM.match(basename[: -len(ext)]):
        return "tests"

    # 4. helper scripts by path — beats code, so Rust files under scripts/
    # count as helpers, not actual code
    if dirs & HELPER_DIR_COMPONENTS:
        return "helper"

    # 5. actual code: mixed Rust files defer to line-level analysis;
    # manifests and native guest sources are code as-is
    if ext == ".rs":
        return None
    if basename in CODE_BASENAMES or ext in CODE_EXTENSIONS:
        return "code"

    # 6. helper scripts catch-all
    return "helper"


# --- Rust line-level analysis -------------------------------------------------
#
# A line-oriented scanner (ported from rust-loc-treemap) tracks block-comment /
# string / raw-string state across lines so braces inside strings or comments
# never affect item-boundary tracking. On top of it, a small state machine
# marks the line ranges covered by test-only items: a (possibly multiline)
# `#[cfg(test)]` / `#[cfg(all(test, ...))]` attribute, any stacked attributes
# after it, and the following item through its closing brace (or trailing
# semicolon for `mod tests;`-style items, or trailing comma for non-item
# targets such as struct fields, struct-literal fields, enum variants, and
# brace-less match arms). `#[test]`-style fn attributes get the same
# treatment, and a file-inner `#![cfg(test)]` marks the whole file.


@dataclass
class LineScan:
    has_code: bool = False
    open_braces: int = 0
    close_braces: int = 0
    code: str = ""


@dataclass
class _CfgAttr:
    code: str = ""
    square_depth: int = 0
    started: bool = False
    inner: bool = False  # #![cfg(...)] file/module-inner attribute
    lines: list[int] = field(default_factory=list)

    def push(self, scan: LineScan, lineno: int) -> None:
        self.lines.append(lineno)
        self.code += scan.code
        for ch in scan.code:
            if ch == "[":
                self.started = True
                self.square_depth += 1
            elif ch == "]":
                self.square_depth -= 1

    def is_complete(self) -> bool:
        return self.started and self.square_depth <= 0

    def is_test_only(self) -> bool:
        arg = _cfg_arg(self.code)
        return arg is not None and _is_test_only_cfg_expr(arg)


# Item-declaration keywords (with optional visibility/qualifier prefixes).
# A test attribute on one of these tracks braces to the item's end; anything
# else (struct field, struct-literal field, brace-less match arm, enum
# variant, statement) is comma-or-semicolon-terminated.
_ITEM_KEYWORD = re.compile(
    r"^(?:pub\s*(?:\([^)]*\))?\s+)?(?:default\s+)?(?:unsafe\s+)?(?:async\s+)?"
    r'(?:unsafe\s+)?(?:extern\s*(?:"[^"]*")?\s+)?'
    r"(?:fn|mod|struct|enum|union|impl|trait|const|static|type|use|macro_rules|macro)\b"
)


@dataclass
class _Item:
    keyword_item: bool = True
    saw_brace: bool = False
    brace_depth: int = 0
    group_depth: int = 0  # unclosed parens/brackets across lines

    def consume_line(self, scan: LineScan) -> bool:
        """Feed one line; True when the item ends on this line."""
        if not scan.has_code:
            return False
        if scan.open_braces > 0:
            self.saw_brace = True
        if self.saw_brace:
            self.brace_depth += scan.open_braces - scan.close_braces
            return self.brace_depth <= 0
        if not self.keyword_item and scan.close_braces > scan.open_braces:
            # The enclosing block closed before the item terminated (e.g. a
            # brace-less final match arm without a trailing comma). End here;
            # counting the enclosing `}` line is an acceptable one-line
            # imprecision, unbounded leakage is not.
            return True
        for ch in scan.code:
            if ch in "([":
                self.group_depth += 1
            elif ch in ")]":
                self.group_depth -= 1
        if self.group_depth > 0:
            return False
        code = scan.code.rstrip()
        if code.endswith(";"):
            return True
        # Comma-terminated targets: struct fields, struct-literal fields,
        # enum variants, match arms. Item declarations (fn/mod/struct/...)
        # never end at a comma — a multiline fn signature has `,`-terminated
        # parameter lines but is either inside unclosed parens or followed by
        # its braced body.
        return not self.keyword_item and code.endswith(",")


def _starts_cfg_attr(code: str) -> bool:
    code = code.lstrip()
    return code.startswith("#[cfg") or code.startswith("#![cfg")


def _cfg_arg(attr: str) -> str | None:
    idx = attr.find("cfg")
    if idx < 0:
        return None
    after = attr[idx + 3 :]
    open_paren = after.find("(")
    if open_paren < 0:
        return None
    start = idx + 3 + open_paren + 1
    depth = 1
    for offset, ch in enumerate(attr[start:]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return attr[start : start + offset]
    return None


def _call_args(expr: str, name: str) -> str | None:
    expr = expr.strip()
    if not expr.startswith(name):
        return None
    rest = expr[len(name) :].lstrip()
    if not rest.startswith("("):
        return None
    rest = rest[1:]
    depth = 1
    for offset, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return rest[:offset]
    return None


def _split_top_level_args(args: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    for index, ch in enumerate(args):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(args[start:index].strip())
            start = index + 1
    parts.append(args[start:].strip())
    return parts


def _is_test_only_cfg_expr(expr: str) -> bool:
    expr = expr.strip()
    if expr == "test":
        return True
    # `any(test, ...)` can compile in non-test builds, so it is NOT test-only.
    all_args = _call_args(expr, "all")
    if all_args is not None:
        return any(_is_test_only_cfg_expr(a) for a in _split_top_level_args(all_args))
    return False


class _Scanner:
    """Cross-line lexical state: block comments, strings, raw strings."""

    def __init__(self) -> None:
        self.block_comment_depth = 0
        self.in_string = False
        self.in_raw_string = False
        self.raw_hash_count = 0

    def scan_line(self, line: str) -> LineScan:
        scan = LineScan()
        chars = line
        length = len(chars)
        i = 0
        while i < length:
            c = chars[i]
            nxt = chars[i + 1] if i + 1 < length else ""

            if self.block_comment_depth > 0:
                if c == "/" and nxt == "*":
                    self.block_comment_depth += 1
                    i += 2
                elif c == "*" and nxt == "/":
                    self.block_comment_depth -= 1
                    i += 2
                else:
                    i += 1
                continue

            if self.in_string:
                scan.has_code = True
                if c == "\\":
                    i += 2
                elif c == '"':
                    self.in_string = False
                    i += 1
                else:
                    i += 1
                continue

            if self.in_raw_string:
                scan.has_code = True
                if c == '"':
                    closing = 0
                    while (
                        i + 1 + closing < length and chars[i + 1 + closing] == "#"
                    ):
                        closing += 1
                    if closing >= self.raw_hash_count:
                        self.in_raw_string = False
                        i += 1 + self.raw_hash_count
                    else:
                        i += 1
                else:
                    i += 1
                continue

            if c == "/" and nxt == "/":
                break  # line comment: rest of line is not code
            if c == "/" and nxt == "*":
                self.block_comment_depth = 1
                i += 2
            elif c == "r" and nxt in ('"', "#"):
                scan.has_code = True
                scan.code += c
                hashes = 0
                j = i + 1
                while j < length and chars[j] == "#":
                    hashes += 1
                    j += 1
                if j < length and chars[j] == '"':
                    self.in_raw_string = True
                    self.raw_hash_count = hashes
                    i = j + 1
                else:
                    i += 1
            elif c == '"':
                self.in_string = True
                scan.has_code = True
                scan.code += c
                i += 1
            elif c == "'":
                scan.has_code = True
                scan.code += c
                if i + 2 < length and chars[i + 1] != "\\" and chars[i + 2] == "'":
                    i += 3
                elif i + 3 < length and chars[i + 1] == "\\" and chars[i + 3] == "'":
                    i += 4
                else:
                    i += 1  # lifetime or lone quote
            elif c == "{":
                scan.has_code = True
                scan.open_braces += 1
                scan.code += c
                i += 1
            elif c == "}":
                scan.has_code = True
                scan.close_braces += 1
                scan.code += c
                i += 1
            elif c.isspace():
                scan.code += c
                i += 1
            else:
                scan.has_code = True
                scan.code += c
                i += 1

        return scan


@dataclass
class RustRegions:
    """1-based line-number sets for a Rust source file."""

    test_lines: set[int]
    doc_lines: set[int]

    def category(self, lineno: int) -> str:
        # Test region wins over a doc comment inside it.
        if lineno in self.test_lines:
            return "tests"
        if lineno in self.doc_lines:
            return "docs"
        return "code"


def analyze_rust(content: str) -> RustRegions:
    scanner = _Scanner()
    test_lines: set[int] = set()
    doc_lines: set[int] = set()

    pending_cfg: _CfgAttr | None = None
    await_item = False  # a test-only attribute was seen; consume attrs then item
    item: _Item | None = None
    rest_of_file_is_test = False

    lines = content.splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        # Doc-comment detection uses the state at line start: a `///` inside a
        # block comment or multiline string is not a doc comment.
        clean_start = (
            scanner.block_comment_depth == 0
            and not scanner.in_string
            and not scanner.in_raw_string
        )
        if clean_start and (stripped.startswith("///") or stripped.startswith("//!")):
            doc_lines.add(lineno)

        scan = scanner.scan_line(raw_line)

        if rest_of_file_is_test:
            test_lines.add(lineno)
            continue

        if item is not None:
            test_lines.add(lineno)
            if item.consume_line(scan):
                item = None
            continue

        if pending_cfg is not None:
            pending_cfg.push(scan, lineno)
            if pending_cfg.is_complete():
                if pending_cfg.is_test_only():
                    test_lines.update(pending_cfg.lines)
                    if pending_cfg.inner:
                        rest_of_file_is_test = True
                    else:
                        await_item = True
                pending_cfg = None
            continue

        code = scan.code.strip()

        if await_item:
            if code.startswith("#"):
                test_lines.add(lineno)  # stacked attribute on the test item
                continue
            if scan.has_code:
                test_lines.add(lineno)
                new_item = _Item(keyword_item=bool(_ITEM_KEYWORD.match(code)))
                if not new_item.consume_line(scan):
                    item = new_item
                await_item = False
            else:
                test_lines.add(lineno)  # blank/comment line between attr and item
            continue

        if _starts_cfg_attr(code):
            pending_cfg = _CfgAttr(inner=code.startswith("#!"))
            pending_cfg.push(scan, lineno)
            if pending_cfg.is_complete():
                if pending_cfg.is_test_only():
                    test_lines.update(pending_cfg.lines)
                    if pending_cfg.inner:
                        rest_of_file_is_test = True
                    else:
                        await_item = True
                pending_cfg = None
            continue

        if TEST_ATTR.match(code):
            test_lines.add(lineno)
            await_item = True
            continue

    return RustRegions(test_lines=test_lines, doc_lines=doc_lines)


# --- Diff parsing ---------------------------------------------------------


@dataclass
class FileDiff:
    old_path: str | None  # None for added files
    new_path: str | None  # None for deleted files
    added: list[int] = field(default_factory=list)  # 1-based lines in new file
    removed: list[int] = field(default_factory=list)  # 1-based lines in old file


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _unquote_git_path(path: str) -> str:
    if not (path.startswith('"') and path.endswith('"')):
        return path
    body = path[1:-1]
    out = []
    i = 0
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in escapes:
                out.append(escapes[nxt])
                i += 2
                continue
            if body[i + 1 : i + 4].isdigit() and len(body) >= i + 4:
                out.append(chr(int(body[i + 1 : i + 4], 8)))
                i += 4
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _strip_prefix(path: str, prefix: str) -> str | None:
    path = _unquote_git_path(path)
    if path == "/dev/null":
        return None
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def parse_diff(diff_text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    old_line = new_line = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current = FileDiff(old_path=None, new_path=None)
            files.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        if not in_hunk:
            if line.startswith("--- "):
                current.old_path = _strip_prefix(line[4:], "a/")
                continue
            if line.startswith("+++ "):
                current.new_path = _strip_prefix(line[4:], "b/")
                continue
            if line.startswith("rename from "):
                current.old_path = _unquote_git_path(line[len("rename from ") :])
                continue
            if line.startswith("rename to "):
                current.new_path = _unquote_git_path(line[len("rename to ") :])
                continue
        match = _HUNK_HEADER.match(line)
        if match:
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            current.removed.append(old_line)
            old_line += 1
        elif line.startswith("+"):
            current.added.append(new_line)
            new_line += 1
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file"
        elif line.startswith(" "):  # context line (not emitted with -U0)
            old_line += 1
            new_line += 1

    return files


# --- Orchestration ----------------------------------------------------------


def _git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


class DiffClassifier:
    def __init__(self, repo: str, old_rev: str, new_rev: str) -> None:
        self.repo = repo
        self.old_rev = old_rev
        self.new_rev = new_rev
        self._region_cache: dict[tuple[str, str], RustRegions | None] = {}

    def _regions(self, rev: str, path: str) -> RustRegions | None:
        key = (rev, path)
        if key not in self._region_cache:
            try:
                content = _git(self.repo, "show", f"{rev}:{path}")
            except subprocess.CalledProcessError:
                self._region_cache[key] = None
            else:
                self._region_cache[key] = analyze_rust(content)
        return self._region_cache[key]

    def _line_category(self, rev: str, path: str, lineno: int) -> str:
        file_cat = classify_path(path)
        if file_cat is not None:
            return file_cat
        regions = self._regions(rev, path)
        if regions is None:
            return "code"  # unreadable content: fall back to plain code
        return regions.category(lineno)

    def classify(self) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        """Returns (totals, per_file) where totals[category] = {added, removed}."""
        # WARNING: -U0 can pick a slightly different hunk alignment than the
        # default-context diff, so the grand total may drift a couple of lines
        # from `git diff --shortstat` on rare inputs. Categories still exactly
        # partition the -U0 diff this classifier actually parses.
        diff_text = _git(
            self.repo,
            "-c",
            "core.quotepath=off",
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            "-U0",
            self.old_rev,
            self.new_rev,
        )
        totals = {cat: {"added": 0, "removed": 0} for cat in CATEGORIES}
        per_file: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for file_diff in parse_diff(diff_text):
            display = file_diff.new_path or file_diff.old_path or "?"
            if file_diff.new_path is not None:
                for lineno in file_diff.added:
                    cat = self._line_category(
                        self.new_rev, file_diff.new_path, lineno
                    )
                    totals[cat]["added"] += 1
                    per_file[display][f"{cat}+"] += 1
            if file_diff.old_path is not None:
                for lineno in file_diff.removed:
                    cat = self._line_category(
                        self.old_rev, file_diff.old_path, lineno
                    )
                    totals[cat]["removed"] += 1
                    per_file[display][f"{cat}-"] += 1

        return totals, dict(per_file)


def format_summary(totals: dict[str, dict[str, int]]) -> str:
    lines = []
    total_added = sum(totals[cat]["added"] for cat in CATEGORIES)
    total_removed = sum(totals[cat]["removed"] for cat in CATEGORIES)
    for cat in CATEGORIES:
        added = totals[cat]["added"]
        removed = totals[cat]["removed"]
        lines.append(f"{CATEGORY_LABELS[cat]}: +{added} {MINUS}{removed}")
    lines.append(f"Total diff: +{total_added} {MINUS}{total_removed}")
    return "\n".join(lines)


def format_markdown(totals: dict[str, dict[str, int]]) -> str:
    def delta(value: int, prefix: str) -> str:
        return f"{prefix}{value:,}" if value else "0"

    total_added = sum(totals[cat]["added"] for cat in CATEGORIES)
    total_removed = sum(totals[cat]["removed"] for cat in CATEGORIES)
    code_added = totals["code"]["added"]
    code_removed = totals["code"]["removed"]
    code_changed = code_added + code_removed
    code_unit = "line" if code_changed == 1 else "lines"

    lines = [
        COMMENT_MARKER,
        "## 📏 PR diff",
        "",
        "> [!NOTE]",
        f"> ### Actual code changed: **{code_changed:,} {code_unit}**",
        f"> 🟢 **{delta(code_added, '+')} added** &nbsp;&nbsp; "
        f"🔴 **{delta(code_removed, MINUS)} removed**",
        ">",
        "> Tests, docs, fixtures, and helper scripts excluded.",
        "",
        "| Category | 🟢 Added | 🔴 Removed |",
        "|:--|--:|--:|",
    ]
    for cat in CATEGORIES:
        icon, label = MARKDOWN_CATEGORIES[cat]
        added = totals[cat]["added"]
        removed = totals[cat]["removed"]
        if cat == "code":
            lines.append(
                f"| {icon} **{label}** | **{delta(added, '+')}** | "
                f"**{delta(removed, MINUS)}** |"
            )
        else:
            lines.append(
                f"| {icon} {label} | {delta(added, '+')} | "
                f"{delta(removed, MINUS)} |"
            )
    lines.extend(
        [
            f"| **Total diff** | **{delta(total_added, '+')}** | "
            f"**{delta(total_removed, MINUS)}** |",
            "",
            "<sub>Every changed line is classified once.</sub>",
            "",
        ]
    )
    return "\n".join(lines)


def format_details(per_file: dict[str, dict[str, int]]) -> str:
    lines = ["", "Per-file breakdown (category+added/-removed):"]
    for path in sorted(per_file):
        counts = per_file[path]
        parts = []
        for cat in CATEGORIES:
            added = counts.get(f"{cat}+", 0)
            removed = counts.get(f"{cat}-", 0)
            if added or removed:
                parts.append(f"{cat} +{added}/-{removed}")
        lines.append(f"  {path}: {', '.join(parts)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository path")
    parser.add_argument("--base", required=True, help="base rev (e.g. origin/main)")
    parser.add_argument("--head", required=True, help="head rev (e.g. HEAD)")
    parser.add_argument(
        "--no-merge-base",
        action="store_true",
        help="diff base..head directly instead of merge-base(base, head)..head",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    parser.add_argument(
        "--output", help="also write the formatted result to this file"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print a per-file breakdown to stdout",
    )
    args = parser.parse_args()

    old_rev = args.base
    if not args.no_merge_base:
        old_rev = _git(args.repo, "merge-base", args.base, args.head).strip()

    classifier = DiffClassifier(args.repo, old_rev, args.head)
    totals, per_file = classifier.classify()

    if args.format == "json":
        output = json.dumps(totals, indent=2)
    elif args.format == "markdown":
        output = format_markdown(totals)
    else:
        output = format_summary(totals)

    print(output)
    if args.details:
        print(format_details(per_file))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — never let CI surface a stack trace as failure
        print(f"diff_classifier error: {exc}", file=sys.stderr)
        sys.exit(2)
