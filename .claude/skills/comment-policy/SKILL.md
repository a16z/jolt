---
name: comment-policy
description: Cleanup pass on a diff or file set that deletes comments carrying no information the code cannot, removes lint suppressions that hide real bugs, and flags the symbol that made a comment necessary as MUST KILL. Report and delete only; never writes application code. Use for "strip comments", "comment pass", "clean up comments on this diff".
---

# Comment policy

Delete comments. Report what was deleted, what was kept and why, and which symbols need a refactor so the comment is no longer needed. Touch comments and lint suppressions only. Never write, move, or reshape application code; that is the `MUST KILL` list's job, handed to a separate change.

## Posture

- Default is delete. A comment survives only by matching one clause of the keep list below. If unsure whether a clause applies, delete.
- `IMPORTANT`, `do not remove`, `too risky`, `fine for now`, and any long justification are signals to read the nearby code before judging. If no keep clause applies, the comment is covering for the code: delete it and flag the symbol.
- A kept comment must be true today: the enforcement point, bug, coupled file, or external constraint it names exists on a live path. Unproven after reading the code: delete.
- Never shorten a bad comment into a better-sounding justification. Delete it, or keep it whole.
- Every flag names a real symbol inside scope at a real `file:line`. Invent nothing.

## Keep list

Only these survive. Everything else is deleted.

1. License or legal headers.
2. Non-obvious behaviour forced by an external dependency, platform, vendor, or protocol the repo cannot reshape (arkworks fork behaviour, RISC-V spec corner cases, guest toolchain quirks).
3. Directives a tool reads: `#[rustfmt::skip]`, `# fmt: skip`, shebangs, `# shellcheck shell=`/`source=`, encoding and schema pragmas. Lint suppressions are judged below, not here.
4. Doc comments that define a public API contract (`///`, `//!`, Python docstrings, JavaScript `/** */` on exported items). One that only restates the signature is not a contract.
5. Issue or RFC links explaining a constraint the code cannot express, including `TODO(#issue)` only when the linked issue explains that constraint; algorithm explanations, with paper/spec links where applicable.
6. Invariants or contracts the type system cannot express, stating WHERE they are enforced (name the checking function or specific fixture). Without the where, delete. Never keep a comment that calls a property constraint-enforced when it holds only for the honest prover or under a `debug_assert`.
7. Why-not notes and warnings for non-obvious gotchas that prevent reintroducing a known bug, soundness gap, deadlock, or race.
8. Pointers to non-local coupling: keep-in-sync with another file, crate, or format (relation ordering owned by `jolt-claims`, serialized enum discriminants, transcript absorption order mirrored by prover and verifier, constants mirrored in `common`).
9. Soundness or security reasoning (why this bound check suffices, why this input is trusted). `// SAFETY:` on `unsafe` belongs here.

Delete on sight: narration (`bind the polynomial`), section banners, commented-out code, history and migration notes (`used to be…`, `renamed from…`, `after the refactor…`), behaviour descriptions that no longer match the code, TODOs not meeting clause 5, restated signatures and type names, end-of-block markers.

## MUST KILL

When a deleted comment was carrying weight the code should carry, flag the symbol:

```
MUST KILL <exact symbol> — <one line: what would make the behaviour obvious without prose> — <file:line>
```

Rename, extract, type, or restructure are the usual fixes; name the one that applies. The flag ends the job on that symbol. Do not apply it.

## Lint suppressions

Every suppression claims the rule is wrong at that site. The Rust workspace denies `allow_attributes`; use `#[expect(..)]`, not `#[allow(..)]`. Inspect both forms, including inner attributes. Look up the rule before judging.

- Stays: the rule is style, pedantic, or naming only, or it is a known false positive, linked or evident at that site. A `reason` on it is fine.
- Stays: test-inappropriate lints on test modules, including panic-family rules (`unwrap_used`, `expect_used`, `panic`, `indexing_slicing`), as permitted by `CLAUDE.md`; a panic is how a test fails.
- Stays: function-level `#[expect(clippy::unwrap_used)]`/`expect_used` where avoiding the call significantly hurts readability (e.g. infallible conversions), as permitted by the Lint Policy in `CLAUDE.md`.
- Stays: in the jolt-verifier runtime closure crates (`specs/verifier-closure-lints.md`), an `#[expect(clippy::.., reason = "..")]` at the narrowest scope with a verified justification for that rule (e.g. a lossless cast, structurally guaranteed indexing, or a fail-closed wildcard). Apply the clause 6/9 evidence standard; a vague or missing reason: delete + MUST KILL.
- Everything else is deleted with the guilty symbol flagged `MUST KILL`: correctness, safety, suspicious, complexity, and perf rules, and any rule that cannot be identified.
- Rule levels in config files (Cargo `[lints]`, crate-root `#![deny(..)]`/`#![forbid(..)]`, `clippy.toml`) are repo policy, not suppressions: leave them.

## Language matrix

| Language | Comment forms | Suppressions | Verdict on suppressions |
|---|---|---|---|
| Rust | `//` `///` `//!` `/* */` | `#[expect(.., reason = "..")]`, inner `#![expect(..)]` | `clippy::style`/`pedantic` may stay; outside tests `clippy::correctness`/`suspicious`/`complexity`/`perf`, `unwrap_used`, `expect_used`, `panic`, `indexing_slicing`, `unreachable` → delete + MUST KILL unless an exception above applies |
| Python | `#` `"""docstring"""` | `# noqa`, `# type: ignore`, `# pylint: disable` | `# type: ignore` hides real bugs → delete + MUST KILL unless the stub is broken (clause 2); public-contract docstrings stay under clause 4 |
| JavaScript | `//` `/* */` `/** */` (including scripts in HTML) | `// eslint-disable*`, `@ts-ignore`, `@ts-expect-error`, `// biome-ignore lint/...` | Type-error suppressions → delete + MUST KILL unless third-party types are broken (clause 2); otherwise by rule class |
| HTML/CSS | `<!-- -->` `/* */` | `/* stylelint-disable */` | By rule class |
| Shell | `#` | `# shellcheck disable=SCxxxx` | SC2086 and quoting rules are correctness → delete + MUST KILL; style codes may stay |
| TOML/YAML | `#` | `# yamllint disable*` | By rule class |

## Procedure

1. **Scope.** The files or diff given by the caller. Otherwise `git diff origin/main --name-only`. For a diff, judge only comments and suppressions on changed lines. Never widen scope.
2. **Per-file pass.** Read the whole file, not just the comment lines: a comment's truth is decided by the code around it. Tag each comment with a keep clause number or `delete`. Tag each suppression with its rule and class.
3. **Delete.** Remove comment lines and dead suppressions; for a trailing comment strip only the comment. Leave the code untouched, including blank lines that were not the comment's. Do not reflow.
4. **Prove comment-only.** For touched Rust, run sequentially:
   ```bash
   cargo fmt -q --message-format=short --all --check
   cargo clippy --all --features host -q --message-format=short --all-targets -- -D warnings
   cargo clippy --all --features host,zk -q --message-format=short --all-targets -- -D warnings
   ```
   For other touched languages, use the repository's configured check commands; do not invent a runner or apply Rust checks to Python, shell, or HTML. Record unavailable checks as skips, not passes.
   Never let the formatter rewrite code. A lint failure caused only by a removed suppression is expected: keep the deletion, it is already a MUST KILL. Then run `git diff --stat` and `git diff --no-ext-diff --no-color --ignore-blank-lines --word-diff` on the touched files: every removed hunk must be comment or suppression text. Revert any other hunk.
5. **Report.** Never send raw diffs.

## Report

- **Touched files** — list with `file:line` ranges.
- **Deleted** — number of comment lines removed, number of suppressions removed.
- **MUST KILL** — the flag lines, one per symbol.
- **Kept** — tally by keep clause (`clause 6: 3, clause 8: 1`); list any kept suppression with its rule.
- **Skips** — files or hunks left alone and why (generated code, vendored, out of scope, did not build).
