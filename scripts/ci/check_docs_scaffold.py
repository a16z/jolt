#!/usr/bin/env python3
"""Check that onboarding docs stay aligned with the `jolt new` scaffold."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
MAIN_RS = ROOT / "src" / "main.rs"
QUICKSTART = ROOT / "book" / "src" / "usage" / "quickstart.md"
HOSTS = ROOT / "book" / "src" / "usage" / "guests_hosts" / "hosts.md"
SUMMARY = ROOT / "book" / "src" / "SUMMARY.md"
README = ROOT / "README.md"
AGENT_WORKFLOW = ROOT / "book" / "src" / "usage" / "agent_workflow.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize(source: str) -> str:
    return source.replace("\r\n", "\n").strip("\n")


def raw_rust_const(source: str, name: str) -> str:
    pattern = re.compile(rf'const {name}: &str = r#"(.*?)"#;', re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise AssertionError(f"could not find raw Rust constant {name}")
    return match.group(1)


def rust_block_after(markdown: str, marker: str) -> str:
    marker_start = markdown.find(marker)
    if marker_start == -1:
        raise AssertionError(f"could not find marker: {marker}")

    after_marker = markdown[marker_start + len(marker) :]
    fence_start = after_marker.find("```rust")
    if fence_start == -1:
        raise AssertionError(f"could not find Rust fence after marker: {marker}")

    code = after_marker[fence_start + len("```rust") :]
    fence_end = code.find("```")
    if fence_end == -1:
        raise AssertionError(f"could not find closing fence after marker: {marker}")
    return code[:fence_end]


def assert_equal(name: str, actual: str, expected: str) -> None:
    if normalize(actual) != normalize(expected):
        raise AssertionError(f"{name} is out of sync with the generated scaffold")


def assert_contains(name: str, source: str, needle: str) -> None:
    if needle not in source:
        raise AssertionError(f"{name} is missing expected text: {needle}")


def assert_summary_link_exists(summary: str) -> None:
    for link in re.findall(r"\]\((\./[^)]+)\)", summary):
        target = (SUMMARY.parent / link).resolve()
        if not target.exists():
            raise AssertionError(f"SUMMARY.md links to missing file: {link}")


def main() -> int:
    main_rs = read(MAIN_RS)
    quickstart = read(QUICKSTART)
    hosts = read(HOSTS)
    summary = read(SUMMARY)
    readme = read(README)
    agent_workflow = read(AGENT_WORKFLOW)

    assert_equal(
        "quickstart guest snippet",
        rust_block_after(
            quickstart,
            "We can view the guest code in `guest/src/lib.rs`.",
        ),
        raw_rust_const(main_rs, "GUEST_LIB"),
    )
    assert_equal(
        "quickstart host snippet",
        rust_block_after(quickstart, "Next let's take a look at the host code"),
        raw_rust_const(main_rs, "HOST_MAIN"),
    )

    assert_contains(
        "hosts guide",
        hosts,
        "`preprocess_prover_sha3(shared)` and "
        "`preprocess_verifier_sha3(shared, verifier_setup, blindfold_setup)`",
    )
    assert_contains(
        "hosts guide",
        hosts,
        "guest::preprocess_verifier_sha3(shared_preprocessing, verifier_setup, None)",
    )

    assert_contains("SUMMARY.md", summary, "./usage/agent_workflow.md")
    assert_summary_link_exists(summary)
    assert_contains("README.md", readme, "usage/agent_workflow.html")

    for heading in (
        "## What to Ask For",
        "## Scaffold First",
        "## Signature Adaptation",
        "## Host Pipeline",
        "## Public, Advice, and Private Inputs",
        "## Agent Checklist",
        "## Fast Debug Loop",
    ):
        assert_contains("agent workflow", agent_workflow, heading)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"docs scaffold check failed: {error}", file=sys.stderr)
        raise SystemExit(1)

