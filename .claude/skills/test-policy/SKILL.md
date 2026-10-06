---
name: test-policy
description: Cleanup pass on a diff, crate, or test tree that deletes tests with no independent failure signal, flags missing contract tests and measured slow tests, and reports test time saved. Report and delete only; never writes tests or application code. Use for "prune tests", "test pass", "remove agent-written tests".
---

# Test policy

Delete tests. Touch tests and their unused support only. Never write tests or change application code; flags are separate tasks.

## Posture

- Default is delete. A test survives only by matching one keep clause below. If unsure whether a clause applies, delete.
- Exception: a (b) candidate below whose bug is unconfirmed → keep and say so. False keeps are cheap; lost regressions are not.
- A surviving test carries a failure signal a maintainer wants independent of how the code is currently written.
- Keep a qualifying test whole. Never rewrite a bad test into a better one in this pass.
- Every flag names a real symbol or contract inside scope at a real `file:line`. Invent nothing.

## Keep list

Only these survive. Dedupe still applies, except to regressions. Tally each kept test under one primary clause.

1. **(a)** Public contract / API behaviour (SDK, prover, and verifier entry points; the `e2e_matrix` guest × mode table).
2. **(b)** Regression test pinned to a fixed bug — keep, and link the issue or commit if findable.
3. **(c)** Invariant or algebraic property the types cannot express (sumcheck round identities, polynomial binding, field laws), asserted against independent ground truth.
4. **(d)** Protocol / wire / format compatibility: verifier fixtures, frozen wire or transcript digests, spec vectors, golden files.
5. **(e)** Soundness / security checks: tampered proofs, malformed inputs, and mode mismatches the verifier must reject.

For (b), find the introducing commit with `git log -S '<test name or failing expression>' -- <path>` and `git blame <path>`; add `--diff-filter=A` only for a new file.
A test is a (b) candidate only if that commit fixes a bug or names an issue/failing input; feature-added tests are not candidates.
Keep candidates without a confirmed bug link under (b), marked `unconfirmed`, with the missing evidence. Put found links in the report, not new comments.
Proptest properties, fuzz targets and their seed corpora are (c)/(e); never delete them.

Delete when no keep clause applies:

- Implementation restatements: tests whose oracle is a second implementation of the same rule, mock-everything tests, assertions that an internal call was made.
- Old-vs-new equivalence tests that keep superseded logic alive as the oracle, including `#[cfg(test)]` copies of deleted production code.
- Tests serving only an agent's self-validation of one small change.
- Duplicate coverage of the same path with different literals and no distinct failure signal.
- Incidental details: exact log or error strings, or ordering that is not a contract.
- Scaffolding: `#[ignore]` tests with no stated run condition, development probes, parity harnesses, temporary benchmarks, diagnostic counters.
- Tests whose only failure mode is "someone refactored".

Judge the failure signal, not the shape of the test.
Clear/ZK/verifier-fixture coverage required by `CLAUDE.md` for changes to stage order, public-value derivation, or transcript absorption is (d)/(e).

## MUST DESIGN

Flag an untested contract a maintainer needs checked:

```text
MUST DESIGN <symbol/contract> — <one-line test sketch> — <file:line>
```

Name the input and observable failure the test would catch. Sketch only; writing the test is a separate task.
Search existing tests and fixtures before claiming a gap. Missing branch coverage alone is not a missing contract test.

## MUST KILL

If a deleted test was the last user of a test hook compiled into non-test builds, flag it:

```text
MUST KILL <exact symbol> — <one line: remove the unused test hook> — <file:line>
```

Search all callers across the workspace, not just the deletion scope. Leave the hook untouched.

## SLOW

```text
SLOW <test> — <wall> — <signal it buys>
```

Read per-test wall durations from nextest output (`PASS [   12.345s] <crate> <test>`) of a real run. Never guess wall time.
Compare measured cost with the failure signal; there is no universal seconds cutoff. A qualifying test stays.

## Test forms

`#[test]`, `#[tokio::test]`, proptest, `#[serial]` tests, and `tests/*.rs` integration binaries, with their support: test data, golden files, verifier fixtures, `#[cfg(test)]` modules, and `prover-fixtures`-gated helpers. Run only touched tests: `cargo nextest run -p <crate> --cargo-quiet -E 'test(<name>)'`, adding the features (`prover-fixtures`, `zk`, `akita`, `field-inline`) the test needs.

## Procedure

1. **Scope.** Use the caller's files, diff, or crates; default to the whole workspace test tree, including embedded test modules.
   For a diff, judge tests touched by changed lines. Inspect dependencies outside scope; do not delete outside scope.
   Skip generated and vendored tests. Record the starting diff so unrelated edits remain untouched.
2. **Per-test pass.** Read each test and the code under test. Identify the observable failure, then assign (a)–(e) or `delete`.
   Record missing-contract sketches and measured slow tests while the relevant code is open.
3. **Dedupe.** Group tests by exercised path and failure signal, not similar names.
   Same path, different literals: keep the existing case with the widest or most hostile input; delete the redundant cases.
   Different boundaries, failure modes, fixed bugs, modes (clear/ZK/Akita), or contracts are not duplicates.
4. **Delete.** Remove whole test functions with their attributes; never leave an empty assertion body.
   Drop a now-empty `mod tests`. Remove fixtures, imports, and test-only helpers only when these deletions removed their last caller.
   Check indirect fixture loading and callers across the workspace before removing support.
   Never touch manifests, `.config/nextest.toml`, CI workflows, or dependencies.
5. **Prove tests-only.** Inspect `git diff --stat` and `git diff --no-ext-diff --no-color` against the starting diff.
   Changed paths must be test files/fixtures, or hunks inside items gated by plain `#[cfg(test)]`. Undo any hunk outside that boundary.
   Anything compiled into non-test builds (`cfg(any(test, feature = "..."))`, `pub`/`#[doc(hidden)]` hooks, fixture features) is application code: flag it `MUST KILL`.
   Run sequentially, per touched crate: `cargo clippy -p <crate> --features <crate features> -q --all-targets -- -D warnings` (repeat with `zk` where the crate has it), then the touched crate's tests with `cargo nextest run -p <crate> --cargo-quiet`.
   Lint and touched tests must pass. Undo deletions that break them; report pre-existing failures or unavailable checks as skips, not a pass. The full suite runs in CI only.
6. **Measure.** Saved test time = sum of deleted tests' durations from a nextest run of the same selection before deletion. No data → `unmeasured`; report test-seconds, labelled local.
   CI wall saved requires comparable before/after `gh run view <run> --json jobs` timings; otherwise `unmeasured`.
7. **Report.** Never send raw diffs.

## Report

- **Touched files** — paths and relevant `file:line` locations.
- **Deleted** — test count; one line each: `<test name> — <reason>`; list removed support separately.
- **Kept** — tally by (a)–(e); regression links and `unconfirmed` candidates with the missing evidence.
- **MUST DESIGN** — flag lines, one per untested contract.
- **MUST KILL** — flag lines, one per unused test hook in application code.
- **SLOW** — flag lines with timing source, or `none measured`.
- **Test time** — test-seconds saved; before/after CI wall times with run references, or `unmeasured`.
- **Skips** — out-of-scope/generated files, unavailable history/timings, and failed or unavailable checks with reasons.
