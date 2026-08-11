#!/usr/bin/env python3
"""Unit tests for diff_classifier.py: `python3 scripts/ci/test_diff_classifier.py`."""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_classifier import (  # noqa: E402
    DiffClassifier,
    analyze_rust,
    classify_path,
    parse_diff,
)


def categories(content: str) -> dict[int, str]:
    regions = analyze_rust(content)
    return {
        lineno: regions.category(lineno)
        for lineno in range(1, content.count("\n") + 2)
    }


class ClassifyPathTests(unittest.TestCase):
    def test_fixtures(self):
        self.assertEqual(classify_path("Cargo.lock"), "fixtures")
        self.assertEqual(classify_path("crates/foo/Cargo.lock"), "fixtures")
        self.assertEqual(classify_path("crates/foo/src/snapshots/x.snap"), "fixtures")
        self.assertEqual(classify_path("x.snap"), "fixtures")
        self.assertEqual(
            classify_path("crates/jolt-profiling/tests/fixtures/simple_trace.json"),
            "fixtures",
        )
        self.assertEqual(
            classify_path(
                "jolt-inlines/fixtures/fixtures/registered_inline_expand_parity_hashes.jsonl"
            ),
            "fixtures",
        )
        self.assertEqual(
            classify_path("crates/jolt-program/src/expand/fixtures/hashes.json"),
            "fixtures",
        )
        # Data file directly under a tests/ dir is test data, not test code.
        self.assertEqual(classify_path("tests/arch-tests/skip.txt"), "fixtures")
        self.assertEqual(classify_path("tests/arch-tests/jolt/sail.json"), "fixtures")

    def test_fixture_crate_source_is_not_fixture(self):
        # jolt-inlines/fixtures is a crate named "fixtures"; its Rust source
        # is real code and must not be swallowed by the fixtures category.
        self.assertIsNone(classify_path("jolt-inlines/fixtures/src/lib.rs"))

    def test_docs(self):
        self.assertEqual(classify_path("README.md"), "docs")
        self.assertEqual(classify_path("book/src/usage/profiling.md"), "docs")
        self.assertEqual(classify_path("book/book.toml"), "docs")
        self.assertEqual(classify_path("specs/clean-slate-prover.md"), "docs")
        # docs beats tests for non-data files under tests/.
        self.assertEqual(classify_path("crates/foo/tests/README.md"), "docs")

    def test_tests(self):
        self.assertEqual(classify_path("crates/jolt-prover/tests/e2e.rs"), "tests")
        self.assertEqual(classify_path("crates/foo/benches/bench.rs"), "tests")
        self.assertEqual(classify_path("crates/foo/src/foo_test.rs"), "tests")
        self.assertEqual(classify_path("crates/foo/src/test_utils.rs"), "tests")
        self.assertEqual(classify_path("crates/foo/src/tests.rs"), "tests")
        # Bare test.rs modules (e.g. zkvm/lookup_table/test.rs) are test
        # helper infrastructure even when not cfg(test)-gated.
        self.assertEqual(classify_path("tracer/src/instruction/test.rs"), "tests")
        self.assertEqual(classify_path("crates/foo/src/bench.rs"), "tests")
        self.assertIsNone(classify_path("crates/foo/src/attest.rs"))
        self.assertIsNone(classify_path("crates/foo/src/latest.rs"))
        # Non-data files under tests/ are test code.
        self.assertEqual(classify_path("tests/arch-tests/run.sh"), "tests")

    def test_code_and_mixed(self):
        self.assertIsNone(classify_path("crates/jolt-poly/src/lib.rs"))
        self.assertEqual(classify_path("Cargo.toml"), "code")
        self.assertEqual(classify_path(".github/workflows/rust.yml"), "code")
        self.assertEqual(classify_path("scripts/ci/fs-soundness.sh"), "code")
        # Data extension outside fixture-ish dirs is code.
        self.assertEqual(classify_path("firebase.json"), "code")


class AnalyzeRustTests(unittest.TestCase):
    def test_cfg_test_module_bounds(self):
        code = """\
fn real_code() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(true);
    }
}

fn more_real_code() {}
"""
        cats = categories(code)
        self.assertEqual(cats[1], "code")
        self.assertEqual(cats[3], "tests")  # #[cfg(test)] attribute line
        self.assertEqual(cats[4], "tests")  # mod tests {
        self.assertEqual(cats[8], "tests")  # fn body
        self.assertEqual(cats[11], "tests")  # closing brace
        self.assertEqual(cats[13], "code")  # after the module

    def test_multiline_cfg_attr(self):
        code = """\
fn real() {}
#[cfg(
    test
)]
mod tests {
    fn helper() {}
}
fn after() {}
"""
        cats = categories(code)
        self.assertEqual(cats[1], "code")
        for line in range(2, 8):
            self.assertEqual(cats[line], "tests", f"line {line}")
        self.assertEqual(cats[8], "code")

    def test_cfg_all_test_is_test_only(self):
        code = """\
#[cfg(all(test, feature = "slow"))]
mod slow_tests {
    fn t() {}
}
fn real() {}
"""
        cats = categories(code)
        for line in range(1, 5):
            self.assertEqual(cats[line], "tests", f"line {line}")
        self.assertEqual(cats[5], "code")

    def test_cfg_any_test_is_not_test_only(self):
        code = """\
#[cfg(any(test, feature = "support"))]
mod support {
    fn s() {}
}
"""
        cats = categories(code)
        for line in range(1, 5):
            self.assertEqual(cats[line], "code", f"line {line}")

    def test_test_attribute_fn(self):
        code = """\
fn real() {}

#[test]
fn standalone_test() {
    assert_eq!(1, 1);
}

#[tokio::test]
async fn async_test() {
    assert!(true);
}

fn after() {}
"""
        cats = categories(code)
        self.assertEqual(cats[1], "code")
        for line in range(3, 7):
            self.assertEqual(cats[line], "tests", f"line {line}")
        for line in range(8, 12):
            self.assertEqual(cats[line], "tests", f"line {line}")
        self.assertEqual(cats[13], "code")

    def test_stacked_attributes_after_cfg_test(self):
        code = """\
#[cfg(test)]
#[allow(dead_code)]
mod tests {
    fn t() {}
}
fn real() {}
"""
        cats = categories(code)
        for line in range(1, 6):
            self.assertEqual(cats[line], "tests", f"line {line}")
        self.assertEqual(cats[6], "code")

    def test_semicolon_item(self):
        code = """\
#[cfg(test)]
mod tests;
fn real() {}
"""
        cats = categories(code)
        self.assertEqual(cats[1], "tests")
        self.assertEqual(cats[2], "tests")
        self.assertEqual(cats[3], "code")

    def test_braces_in_strings_and_comments(self):
        code = '''\
#[cfg(test)]
mod tests {
    fn tricky() {
        let s = "} not a close";
        let raw = r#"{ not an open }"#;
        // } ignored
        /* { ignored */
        assert_eq!(s, "} not a close");
    }
}
fn real() {}
'''
        cats = categories(code)
        for line in range(1, 11):
            self.assertEqual(cats[line], "tests", f"line {line}")
        self.assertEqual(cats[11], "code")

    def test_inner_cfg_test_marks_whole_file(self):
        code = """\
#![cfg(test)]

use foo::bar;

fn helper() {}
"""
        cats = categories(code)
        for line in range(1, 6):
            self.assertEqual(cats[line], "tests", f"line {line}")

    def test_doc_comments(self):
        code = """\
//! Module docs.

/// Documents the function.
/// Second doc line.
fn documented() {}

// Plain comment is code-adjacent, not docs.
let x = 1; /// not a pure doc line
"""
        cats = categories(code)
        self.assertEqual(cats[1], "docs")
        self.assertEqual(cats[3], "docs")
        self.assertEqual(cats[4], "docs")
        self.assertEqual(cats[5], "code")
        self.assertEqual(cats[7], "code")
        self.assertEqual(cats[8], "code")

    def test_doc_comment_inside_test_region_is_tests(self):
        code = """\
#[cfg(test)]
mod tests {
    /// Doc comment on a test helper.
    fn helper() {}
}
"""
        cats = categories(code)
        self.assertEqual(cats[3], "tests")

    def test_doc_like_line_inside_block_comment_is_not_docs(self):
        code = """\
/*
/// looks like a doc comment
*/
fn real() {}
"""
        cats = categories(code)
        self.assertEqual(cats[2], "code")
        self.assertEqual(cats[4], "code")

    def test_cfg_not_test_is_code(self):
        code = """\
#[cfg(not(test))]
fn shipping() {}
"""
        cats = categories(code)
        self.assertEqual(cats[1], "code")
        self.assertEqual(cats[2], "code")


class ParseDiffTests(unittest.TestCase):
    def test_add_remove_and_rename(self):
        diff = """\
diff --git a/src/lib.rs b/src/lib.rs
index 111..222 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,2 +10 @@ fn ctx()
-old line one
-old line two
+new line
@@ -20 +19,3 @@
-gone
+a
+b
+c
\\ No newline at end of file
diff --git a/old_name.rs b/new_name.rs
similarity index 90%
rename from old_name.rs
rename to new_name.rs
index 333..444 100644
--- a/old_name.rs
+++ b/new_name.rs
@@ -5 +5 @@
-x
+y
diff --git a/added.md b/added.md
new file mode 100644
index 000..555
--- /dev/null
+++ b/added.md
@@ -0,0 +1,2 @@
+hello
+world
"""
        files = parse_diff(diff)
        self.assertEqual(len(files), 3)

        lib = files[0]
        self.assertEqual(lib.old_path, "src/lib.rs")
        self.assertEqual(lib.new_path, "src/lib.rs")
        self.assertEqual(lib.removed, [10, 11, 20])
        self.assertEqual(lib.added, [10, 19, 20, 21])

        renamed = files[1]
        self.assertEqual(renamed.old_path, "old_name.rs")
        self.assertEqual(renamed.new_path, "new_name.rs")
        self.assertEqual(renamed.removed, [5])
        self.assertEqual(renamed.added, [5])

        added = files[2]
        self.assertIsNone(added.old_path)
        self.assertEqual(added.new_path, "added.md")
        self.assertEqual(added.added, [1, 2])
        self.assertEqual(added.removed, [])


class EndToEndTests(unittest.TestCase):
    """Build a real git repo and classify an actual diff."""

    def _git(self, repo, *args):
        subprocess.run(
            ["git", "-C", repo, *args],
            check=True,
            capture_output=True,
            env={
                **os.environ,
                # Isolate from user/system git config (hooks, signing, ...).
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )

    def test_mixed_change(self):
        with tempfile.TemporaryDirectory() as repo:
            def write(path, content):
                full = os.path.join(repo, path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(content)

            self._git(repo, "init", "-q", "-b", "main")
            write(
                "src/lib.rs",
                """\
/// Old doc.
fn old_code() {}

#[cfg(test)]
mod tests {
    #[test]
    fn old_test() {
        assert!(true);
    }
}
""",
            )
            write("Cargo.lock", "version = 1\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-q", "-m", "base")

            write(
                "src/lib.rs",
                """\
/// Old doc.
/// New doc line.
fn old_code() {}

fn new_code() {}

#[cfg(test)]
mod tests {
    #[test]
    fn old_test() {
        assert!(true);
        assert!(cfg!(test));
    }
}
""",
            )
            write("Cargo.lock", "version = 2\n")
            write("crates/foo/tests/fixtures/data.json", "{}\n")
            write("specs/new-spec.md", "# Spec\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-q", "-m", "change")

            classifier = DiffClassifier(repo, "HEAD~1", "HEAD")
            totals, _ = classifier.classify()

            # +: 1 doc line, 2 code lines (new fn + blank line before it),
            #    1 test assert, 1 fixture json, 1 spec md, 1 Cargo.lock
            self.assertEqual(totals["docs"], {"added": 2, "removed": 0})
            self.assertEqual(totals["code"], {"added": 2, "removed": 0})
            self.assertEqual(totals["tests"], {"added": 1, "removed": 0})
            self.assertEqual(totals["fixtures"], {"added": 2, "removed": 1})

            total_added = sum(totals[c]["added"] for c in totals)
            total_removed = sum(totals[c]["removed"] for c in totals)
            self.assertEqual(total_added, 7)
            self.assertEqual(total_removed, 1)


if __name__ == "__main__":
    unittest.main()
