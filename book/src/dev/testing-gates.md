# Testing and coverage gates

CI enforces test quality and coverage over the verifier-side protocol code —
the `jolt-verifier` dependency closure plus the verification-relevant extras
(`jolt-dory`, `jolt-akita`, `jolt-hyperkzg`, `tracer`). The design rationale
and acceptance criteria live in
[`specs/test-quality-ci.md`](https://github.com/a16z/jolt/blob/main/specs/test-quality-ci.md);
prover-side code is exempt because a broken prover can only produce proofs the
verifier rejects, never unsound acceptances.

## Coverage floors

`.github/workflows/coverage.yml` fails a PR when any in-scope crate's line
coverage drops below its floor in [`ci/coverage-floors.toml`](https://github.com/a16z/jolt/blob/main/ci/coverage-floors.toml).

- The in-scope set is computed at CI runtime from `cargo metadata`, so
  dependency refactors track automatically; the gate fails loudly if the
  floors file and the computed set drift apart.
- Coverage is **cumulative across each crate's declared feature paths**
  (e.g. `jolt-verifier` merges its `default`, `prover-fixtures`, and
  `prover-fixtures,zk` runs), measured over `src/` files only.
- Floors are static minimums under a manual ratchet: CI never moves them;
  raising one is a deliberate, reviewed edit to the TOML.

Run the gate locally:

```bash
# one llvm-cov export per plan entry (see the plan)
python3 ci/coverage_gate.py plan
cargo llvm-cov nextest -p <crates...> [--features ...] --cargo-quiet \
    --json --output-path cov-default.json
# then enforce
python3 ci/coverage_gate.py enforce --coverage-json cov-default.json [...]
python3 ci/coverage_gate.py self-test   # the gate's own unit tests
```

## Soundness metrics

The same workflow tracks two soundness-specific numbers
(`ci/soundness_metrics.py`):

- **Error-variant coverage** — the fraction of error-enum variants whose
  `Err(...)` construction sites execute during tests. This measures "is there
  a negative test for every rejection path," which line coverage alone hides.
- **Tamper-manifest active ratio** — the fraction of
  `TamperTarget`s in `jolt-verifier`'s tamper manifest with
  `TamperCoverage::Active`. May not regress below its floor.

The tamper harness also asserts *where* a rejection fires: each manifest
target documents the verifier phase that is its last line of defense, and
`assert_verifier_fixture_tamper_rejects` fails if the observed rejection maps
to a later phase than documented.

Nightly, `.github/workflows/coverage-nightly.yml` runs
[`cargo-mutants`](https://mutants.rs) over the soundness-critical modules in
`ci/mutants-modules.toml` (report-only) — the only automated way to catch
"covered but not actually verified" code.

## Test-quality review

`.github/workflows/test-quality.yml` fires a hosted Claude routine on PRs
whose diff adds or modifies tests. Each test is scored 1–10 against
[`ci/test-quality-rubric.md`](https://github.com/a16z/jolt/blob/main/ci/test-quality-rubric.md)
— five dimensions: oracle strength, adversarial reach, failure specificity,
independence & determinism, and property clarity, with crypto-specific
tripwires (notably: a prover/verifier round-trip is *not* an oracle when both
sides share the code under test). Tests scoring below the threshold in
`ci/test-quality.toml` get a rework-or-delete suggestion; the gate is
advisory until `enforce = true` is flipped.
