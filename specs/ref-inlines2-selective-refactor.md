# Spec: ref-inlines2-selective-refactor

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-06-16                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Selectively port the useful ideas from the obsolete `ref/inlines2` branch into current main's inline expansion architecture. The work adds typed semantic contracts and shared inline harness checks first, then gates smaller builder helper and curve field-wrapper refactors on fixture and performance evidence. The branch itself must not be rebased or cherry-picked because current main moved expansion into `jolt-program`, added registered inline expansion fixtures, added P-256, and changed the host SDK surface.

## Intent

### Goal

Add `InlineReference` and test-focused `InlineSpec` abstractions to `jolt-inlines-sdk`, migrate inline tests to those shared contracts, and only then apply narrowly-scoped builder helper or curve field-wrapper refactors that preserve generated inline behavior.

`InlineReference` is the pure semantic contract for an inline operation: typed input, typed output, and deterministic reference evaluation. It must not depend on `rand`, `tracer/test-utils`, `InlineTestHarness`, runtime CPU state, or inline advice queues.

`InlineSpec` builds on `InlineReference` and `InlineOp` for tests: seeded/random input generation, harness construction, input loading, output reading, instruction selection, and a shared reference-vs-harness assertion. It must live behind a `jolt-inlines-sdk/test-utils` feature so normal `host` users do not inherit test harness dependencies. Inline crates should enable this feature only from dev-dependencies or test-only targets.

### Invariants

Existing `jolt-eval` invariants:

- Preserve `source_to_jolt_expansion_equivalence`; semantic/test refactors must not change canonical expansion output.
- Preserve `field_mul_scalar`, `split_eq_bind_*`, `transcript_prover_verifier_consistency_*`, and `soundness`; this spec should not modify those domains.

New or local invariants:

- Every converted inline's harness execution output must equal `InlineReference::reference(input)` for fixed edge cases and random cases.
- `InlineOp::build_sequence` remains deterministic and tracer-free; runtime CPU state and concrete advice values remain isolated to `build_advice`.
- `jolt-inlines-fixtures` registered expansion `row_count` and `output_sha256` values stay unchanged for `InlineReference`/`InlineSpec`, typed-advice evaluation, and non-performance refactors.
- Bigint multiplication is the only intentional expansion change in this PR: removing redundant result-accumulator zero-init rows changes each fixture scenario from 222 rows to 214 rows.
- Any intentional expansion change must document the before/after row count and why the change is worth the fixture update.
- Normal `jolt-inlines-sdk/host` dependencies must not grow to include `tracer/test-utils`, `rand/std_rng`, or harness-only utilities unless explicitly accepted in the implementation PR.
- Curve field-wrapper refactors must preserve guest custom opcode/funct selector behavior for multiplication, squaring, division, and advice consumption.

Candidate `/new-invariant` after the first two packages are converted: `inline_reference_harness_equivalence`, parameterized by registered `InlineSpec` cases, checking that typed reference output matches inline harness execution.

### Non-Goals

- Do not rebase or cherry-pick `ref/inlines2` as a branch.
- Do not revive old `InstrAssembler` APIs or branch-era expansion signatures.
- Do not replace `jolt-inlines-fixtures` with branch-style tests.
- Do not preserve the branch `Cargo.lock` delta.
- Do not add typed `InlineOp::Advice` as a standalone cleanup before stronger wins land.
- Do not add EC operator overloads as a standalone refactor.
- Do not rewrite sequence builders for style only.
- Do not unify secp256k1, Grumpkin, and P-256 into one cross-curve abstraction in the first field-wrapper pass.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-inlines-sdk` exposes a pure `InlineReference` trait that is usable without harness/test dependencies.
- [ ] `jolt-inlines-sdk` exposes `InlineSpec` and shared reference-vs-harness helpers behind a `test-utils` feature.
- [ ] `test-utils` keeps existing `host` dependency shape unchanged unless the implementation PR explicitly justifies the change.
- [ ] Bigint multiplication is converted first; the only accepted fixture drift is the documented 222-to-214 row reduction from removing redundant accumulator zero-init rows.
- [ ] SHA-2, Keccak, Blake2, and Blake3 tests are converted after the bigint conversion stabilizes.
- [ ] Per-crate reference/test utility code is deleted only after equivalent `InlineSpec` tests pass.
- [ ] Missing builder range helpers are added only when needed by at least two current-main builders or when they remove a concrete correctness hazard.
- [ ] Builder helper conversions preserve emitted instruction order unless the PR intentionally updates fixtures with documented row-count changes.
- [ ] Field-wrapper generic work starts with exactly one of secp256k1 or Grumpkin, using current P-256 as local prior art.
- [ ] Typed advice, operator overloads, and sequence-builder mining remain deferred unless their gate criteria in this spec are met.

### Testing Strategy

Run cargo commands sequentially.

Baseline before implementation:

```bash
cargo nextest run -p jolt-inlines-fixtures --cargo-quiet --features host
cargo nextest run -p jolt-inlines-bigint --cargo-quiet --features host
cargo nextest run -p jolt-inlines-sha2 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-keccak256 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-blake2 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-blake3 --cargo-quiet --features host
```

After `InlineReference`/`InlineSpec` conversion:

```bash
cargo nextest run -p jolt-inlines-bigint --cargo-quiet --features host
cargo nextest run -p jolt-inlines-sha2 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-keccak256 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-blake2 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-blake3 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-fixtures --cargo-quiet --features host
```

If curve field-wrapper code changes:

```bash
cargo nextest run -p jolt-inlines-secp256k1 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-grumpkin --cargo-quiet --features host
cargo nextest run -p jolt-inlines-p256 --cargo-quiet --features host
cargo nextest run -p jolt-inlines-fixtures --cargo-quiet --features host
```

Final handoff checks:

```bash
cargo fmt -q
cargo clippy -p jolt-inlines-sdk --features host --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-inlines-fixtures --features host --message-format=short -q --all-targets -- -D warnings
cargo nextest run -p jolt-inlines-fixtures --cargo-quiet --features host
```

No `--features host,zk` coverage is required unless an implementation touches prover/verifier code outside host-only inline expansion and SDK tests.

### Performance

`InlineReference`/`InlineSpec` must be guest/prover performance neutral. Fixture row counts and hashes are the primary guardrail.

Bigint multiplication intentionally drops eight rows by relying on inline finalization to clear released inline registers instead of emitting eight explicit accumulator zero-init ADD rows. The fixture update records the resulting 222-to-214 row-count change, and `bigint_mul_keeps_zero_init_row_reduction` guards that performance win.

Builder helpers are acceptable when they preserve row count and order, or when they reduce row count with an explicit fixture update. Sequence-builder mining is acceptable only one candidate at a time after recording pre-change fixture row counts.

Curve field-wrapper refactors must be neutral for generated guest behavior. If secp256k1 ECDSA paths are touched, use `jolt-eval`'s `prover_time_secp256k1_ecdsa_verify` benchmark as the relevant existing objective. A useful future objective is `registered_inline_row_count_total`, computed from `jolt-inlines/fixtures/fixtures/registered_inline_expand_parity_hashes.jsonl`.

## Design

### Architecture

Add a new SDK module, likely `jolt-inlines/sdk/src/spec.rs`, and re-export it only under the chosen test gate.

Expected trait shape:

```rust
pub trait InlineReference {
    type Input;
    type Output: PartialEq + core::fmt::Debug;

    fn reference(input: &Self::Input) -> Self::Output;
}

#[cfg(feature = "test-utils")]
pub trait InlineSpec: InlineReference + host::InlineOp {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input>;
    fn random(rng: &mut impl rand::RngCore) -> Self::Input;
    fn harness() -> tracer::utils::inline_test_harness::InlineTestHarness;
    fn load(harness: &mut tracer::utils::inline_test_harness::InlineTestHarness, input: &Self::Input);
    fn read(harness: &tracer::utils::inline_test_harness::InlineTestHarness) -> Self::Output;
}
```

The final API may differ, but it must preserve the dependency split: pure reference logic outside harness dependencies, shared harness checks behind test utilities.

The first migration target is `jolt-inlines/bigint`, because it has a simple typed input/output shape and existing duplicated reference/test utility code. Hash and compression packages follow after the helper API is stable.

Builder helper work extends `InlineBuilderExt` in `jolt-inlines/sdk/src/host.rs` for current `InlineExpansionBuilder`; it must not introduce old assembler abstractions.

Field-wrapper work uses `jolt-inlines/p256/src/sdk.rs` as current-main prior art for config-based field types. Start with secp256k1 or Grumpkin only, preserve public aliases where practical, and leave cross-curve unification for a later spec.

### Alternatives Considered

Rebasing `ref/inlines2` was rejected because the branch predates the current expansion architecture and would reintroduce stale assembler assumptions.

Keeping only per-crate tests was rejected because it leaves duplicated reference/harness plumbing and makes future P-256-style inlines more expensive to add.

Relying only on `jolt-inlines-fixtures` was rejected because fixtures catch expansion drift, not semantic reference-vs-execution drift.

Making typed advice the next step was deferred because it touches every inline for mostly boilerplate reduction.

Adding EC operator overloads was deferred because it is readability churn in a crypto hot path unless it naturally falls out of a field-wrapper refactor.

## Documentation

No Jolt book changes are required for the first phase because this is internal SDK/test infrastructure. If the final `InlineReference`/`InlineSpec` API is intended for downstream inline authors, add or update inline-author documentation near the SDK docs instead of the user-facing zkVM book.

## Execution

1. Record baseline fixture and package test results.
2. Add the SDK reference/spec module and feature/cfg gating.
3. Convert bigint multiplication first.
4. Convert hash packages after the API stabilizes.
5. Delete replaced per-crate utilities only after equivalent tests pass.
6. Evaluate builder helpers with fixture row-count review.
7. Evaluate one curve field-wrapper refactor after lower-risk phases land.
8. Reconsider typed advice only if there is a concrete provider/registration cleanup beyond aesthetics.
9. Mine sequence-builder rewrites only with pre-change row counts and an explicit performance win.

## References

- `ref-inlines2-refactor-plan.md`
- Branch reviewed in the plan: `ref/inlines2`, base `dd9810685`, head `4c67133ec`
- Current expansion code: `crates/jolt-program/src/expand/`
- Inline SDK host surface: `jolt-inlines/sdk/src/host.rs`
- Inline fixture guardrail: `jolt-inlines/fixtures/`
- Existing expansion invariant: `jolt-eval/src/invariant/source_to_jolt_expansion_equivalence.rs`
