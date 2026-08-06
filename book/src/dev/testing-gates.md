# Testing and coverage gates

CI enforces test coverage over the verifier-side protocol code —
the `jolt-verifier` dependency closure plus the verification-relevant extras
(`jolt-dory`, `jolt-akita`, `jolt-hyperkzg`, `tracer`). Prover-side code is
exempt because a broken prover can only produce proofs the verifier rejects,
never unsound acceptances.

## Coverage floors

`.github/workflows/coverage.yml` fails a PR when any in-scope crate's line
coverage drops below its floor in [`scripts/ci/coverage-floors.toml`](https://github.com/a16z/jolt/blob/main/scripts/ci/coverage-floors.toml).

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
python3 scripts/ci/coverage_gate.py plan
cargo llvm-cov nextest -p <crates...> [--features ...] --cargo-quiet \
    --json --output-path cov-default.json
# then enforce
python3 scripts/ci/coverage_gate.py enforce --coverage-json cov-default.json [...]
python3 scripts/ci/coverage_gate.py self-test   # the gate's own unit tests
```

## Soundness metrics

The same workflow tracks two soundness-specific numbers
(`scripts/ci/soundness_metrics.py`):

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
`scripts/ci/mutants-modules.toml` (report-only) — the only automated way to catch
"covered but not actually verified" code. A mutant is only "missed" relative to
the tests that actually run, so targets whose strongest oracle is feature-gated
declare those features in the config (`jolt-verifier` runs with
`prover-fixtures` so the completeness and tamper suites participate in the kill
signal; fixtures are disk-cached after the baseline, keeping per-mutant test
time flat).

## Fiat-Shamir soundness

The `fs-obligations` and `fs-attacks-smoke` jobs protect the Fiat-Shamir
soundness of `jolt-verifier`. They run independently for Dory clear, Dory ZK,
and Akita clear. Akita ZK is compile-probed and must be added to the matrix as
soon as that verifier combination becomes supported.

An attack test is valid only when all four steps hold:

1. The original fixture verifies and records typed challenge calls.
2. A coordinated mutation makes an individual protocol claim false.
3. Verification with the recorded challenges replayed succeeds. This proves the
   mutation preserves the verifier's algebraic checks when Fiat-Shamir binding
   is removed.
4. Verification with the production transcript changes a relevant challenge and
   rejects.

Production acceptance is a soundness finding. Frozen-challenge rejection is a
coverage failure: another check prevented the test from isolating the claimed
Fiat-Shamir defense. A changed transcript schedule or prover/verifier agreement
is not a security oracle.

Run the complete local matrix with:

```bash
scripts/ci/fs-soundness.sh
```

`fs_obligations` assigns stable identities to transcript absorption expressions,
challenge-shaped calls, scope annotations, generated sumcheck batching draws,
and serialized verifier inputs in the production dependency closure. The
absorption inventory includes normalized call expressions, so changing either a
label or its bound value requires review. A reviewed protocol change may
regenerate the inventories with:

```bash
JOLT_FS_BLESS=1 cargo nextest run -p jolt-verifier \
  --test fs_obligations --features fs-audit --cargo-quiet
```

Review the resulting diff as a list of new or removed security obligations.
Never regenerate it merely to make CI pass.
