# Spec: Test Quality & Coverage CI

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @markosg04                     |
| Created     | 2026-07-23                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Line coverage across the workspace is 70.5% (2026-07-23 measurement) but unevenly distributed and, in places, misleading: soundness-relevant code ships with zero test reach (the `jolt-blindfold` prover is never executed by any test — its suite validates the verifier against a reimplemented parallel prover), some negative tests assert `Err(_)` without checking which rejection fired, and the strongest suites are feature-gated (`prover-fixtures`, `solinas`, `zk`) and never run in CI. This spec adds CI gates that make coverage both enforced and meaningful, scoped to the code that defines protocol correctness: `jolt-verifier` is the protocol-defining crate, and prover-side bugs cannot affect soundness (a broken prover produces proofs the verifier rejects; it can never produce unsound acceptances). Enforcement therefore targets the `jolt-verifier` dependency closure plus verification-relevant extras. Three gates: (1) per-crate coverage floors measured cumulatively across each crate's meaningful feature paths; (2) an AI test-quality routine scoring every new or modified test 1–10 against a systems-Rust/cryptography rubric, with sub-7 tests flagged for rework or deletion; (3) quantified soundness metrics (error-variant coverage, tamper-manifest active ratio, nightly mutation testing) so negative-test coverage is a tracked number, not an impression. The same PR that lands the gates performs a sweep bringing every in-scope crate above its target floor, so the merged baseline is the enforced bar and future PRs can only maintain or raise it.

## Intent

### Goal

Add CI enforcement making verifier-side protocol code's test coverage quantitatively enforced and qualitatively meaningful: per-crate `cargo llvm-cov` floors over the `cargo metadata`-computed `jolt-verifier` workspace dependency closure (plus a declared extras list), measured cumulatively across each crate's declared feature paths; an AI test-quality routine scoring new/modified tests 1–10 against a checked-in rubric; and tracked soundness metrics with their own floors.

Key artifacts introduced:

- `ci/coverage-floors.toml` — per-crate floors, per-crate feature paths, extras list, soundness-metric floors
- `ci/coverage_gate.py` — closure computation, config-drift validation, llvm-cov aggregation, floor enforcement (stdlib-only Python, self-tested)
- `ci/soundness_metrics.py` — error-variant coverage and tamper-manifest ratio computation
- `ci/test-quality-rubric.md` — the scoring rubric consumed by the hosted routine
- `ci/test-quality.toml` — routine threshold (7) and `enforce` flag (ships `false`)
- `.github/workflows/coverage.yml` (PR gate), `.github/workflows/coverage-nightly.yml` (mutation testing + trend report), `.github/workflows/test-quality.yml` (fires the hosted routine)

### Scope: the in-scope crate set

The enforced set is computed at CI runtime, never hardcoded:

- **Closure** (16 crates today): the workspace normal-dependency closure of `jolt-verifier` — `jolt-verifier`, `jolt-blindfold`, `jolt-claims`, `jolt-claims-derive`, `jolt-verifier-derive`, `jolt-crypto`, `jolt-field`, `jolt-lookup-tables`, `jolt-openings`, `jolt-poly`, `jolt-program`, `jolt-r1cs`, `jolt-riscv`, `jolt-sumcheck`, `jolt-transcript`, `common`
- **Extras** (declared in `coverage-floors.toml`): `jolt-dory`, `jolt-akita`, `jolt-hyperkzg`, `tracer` — verification-relevant but reachable only as dev-dependencies or outside `crates/`

### Invariants

No existing `jolt-eval` invariant changes. The gates are CI tooling; their trustworthiness invariants are enforced as script self-tests that run inside the coverage job itself (not `/new-invariant` entries, since they are properties of CI, not of the zkVM):

1. **Closure correctness** — the enforced crate set equals the `cargo metadata` normal-dependency closure of `jolt-verifier` union the declared extras; it is never empty and always contains `jolt-verifier`.
2. **Floors completeness** — every in-scope crate has exactly one floor entry, and every floor entry corresponds to an in-scope crate. Drift in either direction (a refactor adds/removes a closure crate) fails the job loudly rather than silently un-gating code.
3. **Coverage floor** — for every in-scope crate, cumulative line coverage across its declared feature paths ≥ its configured floor. Floors are static minimums; editing them is ordinary code review (manual ratchet, no meta-enforcement on the config file).
4. **Score-output validity** — the AI routine's output conforms to a JSON schema (per-test integer scores 1–10 with per-dimension breakdown); malformed output fails the quality job rather than silently passing.
5. **Soundness-metric floors** — error-variant coverage per closure crate ≥ its configured floor; tamper-manifest active ratio ≥ its configured baseline.

### Non-Goals

- **No enforcement on prover-side crates**: `jolt-prover`, `jolt-prover-legacy`, `jolt-witness`, `jolt-kernels`, `jolt-lean-gen`, `jolt-profiling`, `jolt-sdk`, `jolt-inlines`, `jolt-core-svm`. Prover-side code cannot affect protocol soundness; witness-generation crates are consumed by the prover only.
- **No auto-ratchet**: CI never persists or raises floors itself. Floors move only via reviewed edits to `coverage-floors.toml`.
- **AI gate is not blocking at introduction**: it ships advisory (`enforce = false`) and is flipped to blocking in a follow-up once it has scored a few real PRs without misfiring.
- **Mutation testing is nightly report-only**, never a merge gate (runtime cost).
- **No external coverage service** (Codecov etc.): thresholds are versioned in-repo, no data egress, no third-party dependency.
- **No wholesale proptest adoption**: the sweep adds property-based tests only where they beat the existing differential-vs-reference house style.
- **No new `jolt-eval` invariants or objectives are required** for this feature (candidates noted under Performance are optional follow-ups).

## Evaluation

### Acceptance Criteria

Infrastructure:

- [x] `coverage.yml` runs on every PR: computes the closure at runtime, validates config completeness (invariants 1–2), runs `cargo llvm-cov nextest` for each in-scope crate under each of its declared feature paths, merges profiles per crate, and fails if any crate's cumulative line coverage is below its floor (invariant 3).
- [x] `jolt-verifier`'s `prover-fixtures` path runs in the PR job with fixture caching keyed on the legacy-prover fingerprint (lockfile hash + `crates/jolt-prover-legacy` tree hash), so warm runs skip proof generation.
- [x] `coverage_gate.py` and `soundness_metrics.py` self-tests run at the start of the coverage job and gate it.
- [ ] `coverage-nightly.yml` runs `cargo mutants` over the module list in `ci/mutants-modules.toml` and publishes a report (step summary + artifact); it never gates merges.
- [ ] `test-quality.yml` fires the hosted routine on PRs whose diff touches test code (paths filter: `**/tests/**` plus files with modified `#[cfg(test)]`/`#[test]` hunks); the routine posts a PR review with per-test scores, dimension breakdowns, and rework-or-delete suggestions for scores < 7; schema-validated (invariant 4); advisory at merge.
- [x] Soundness metrics are computed per closure crate and enforced against floors (invariant 5): error-variant coverage (fraction of error-enum variants whose `Err(...)` construction sites execute during tests, from llvm-cov region data) and tamper-manifest active ratio (from `jolt-verifier/tests/support/tamper_manifest.rs` dispositions).

Sweep (same PR, after infra):

- [ ] Every in-scope crate reaches ≥ 80% cumulative line coverage across its declared feature paths, or carries a documented exception comment in `coverage-floors.toml` explaining why the target is unreachable and what the achievable floor is.
- [ ] Floors in `coverage-floors.toml` are pinned at post-sweep measured values (rounded down to the integer), and the coverage job passes on the final commit.
- [x] `jolt-blindfold`: the real `prove()` is exercised — prover↔verifier round-trip tests, a differential test against the harness prover in `tests/support/`, and a bad-witness-rejected test (`validate_witness` / `ensure_row_capacity`).
- [x] `jolt-verifier`: tamper tests assert the expected `VerifierError` variant/stage recorded in the manifest's `checked_at`, not bare `Err(_)`; the 16 `Deferred`/`IgnoredUntilFixture` tamper targets are activated or carry per-target justification.
- [x] `jolt-transcript`: pinned known-answer vectors for `LegacyBlake2bTranscript` (state chaining, `challenge_bytes` chunking, label packing) and for the `challenge_scalar` decode path.
- [x] `jolt-riscv`: RVC decode battery covering every `uncompress.rs` expansion arm plus the illegal-encoding fallthrough.
- [x] `jolt-program`: malformed/truncated-ELF and ELF32 rejection tests, `merge_ranges` unit tests, immediate sign-extension and B/J/S/U reassembly tests in `image/decode.rs`, and the RAM-domain error branches (`HeapBelowLowest`, `DomainTooLarge`).
- [ ] *(deferred — the `solinas` feature is not on `main` yet; these land with the Solinas field stack branch)* `jolt-field`: randomized `x.square() == x * x` checks for the Fp128 squaring kernels, pseudo-Mersenne registry self-consistency (`modulus == 2^bits − offset`, alias agreement), `MontgomeryConstants` invariant check, and the `from_bytes` fuzz target compiles again.
- [x] `jolt-poly`: tests for `gruen_poly_deg_3`, `gruen_poly_from_evals`, `e_out_in_for_window`, `e_active_for_window`, and the `new_with_scaling` path in `split_eq.rs`.
- [x] `jolt-openings`: soundness-negative tests for the ZK batch path (tampered commitment, wrong point) — not just witness-count validation.
- [x] `jolt-r1cs`: rv64 constraint satisfaction tests with realistic (non-noop) instruction witnesses.
- [x] `jolt-crypto`: GLV decomposition reconstruction identity (`k0 + k1·λ (+ k2·λ² + k3·λ³) ≡ k mod r`) including near-boundary scalars.
- [x] `jolt-claims`: tests for `outer_uniskip.rs` / `product_uniskip.rs` relations and `symbolic.rs`; serde round-trips for the claim types crossing the proof boundary.
- [x] `jolt-sumcheck`: compressed-encoding tamper test; `BatchedCommittedSumcheckConsistency` offset/overflow error branches.
- [x] `jolt-dory` / `jolt-hyperkzg` / `jolt-akita`: transcript-adapter byte-layout golden test; `WrongEvaluationWidth` negative test; Jolt↔Akita basis-order index KAT.
- [ ] All new sweep tests score ≥ 7 under the rubric (the routine's first full workout is this PR's own sweep).
- [ ] Full workspace `cargo nextest run --cargo-quiet` passes; `muldiv` e2e passes under `--features host` and `--features host,zk`; clippy is clean in both modes.

### Testing Strategy

- **Existing tests**: the full workspace nextest suite, the `muldiv` e2e in both standard and ZK modes, and both clippy modes must keep passing throughout. No existing test is deleted without a rubric score justifying it.
- **New tests**: sweep tests land in their home crates following each crate's existing harness patterns (e.g. the lookup-table macro harnesses, the tamper-manifest framework, seeded `ChaCha20Rng` differential style). CI scripts get unit tests colocated with the scripts (`python -m unittest` in-job).
- **Feature paths** (initial declaration in `coverage-floors.toml`; the sweep calibrates the final set):
  - `jolt-verifier`: `default`, `prover-fixtures`, `prover-fixtures,zk`
  - `jolt-field`: `default` (bn254), `solinas`
  - `jolt-sumcheck`: `default`, `r1cs`
  - `jolt-claims`, `jolt-r1cs`: `default`, `field-inline`
  - all others: `default`
- **Both modes**: ZK-specific verification paths are covered via `jolt-verifier`'s `prover-fixtures,zk` path; the `zk` feature's BlindFold integration continues to be guarded by the `muldiv` e2e in `--features host,zk`.

### Performance

No existing `jolt-eval` objective is expected to move: `lloc`, `cognitive_complexity_avg`, and `halstead_bugs` measure `crates/jolt-prover-legacy/src/`, which this feature does not touch; runtime code is unchanged (the sweep adds tests only), so `bind_parallel_*` and `prover_time_*` benchmarks are unaffected.

Candidate new objectives (optional, via `/new-objective` during or after implementation, so the optimizer can track them): `closure_coverage_min` (minimum per-crate cumulative coverage over the in-scope set), `error_variant_coverage`, `mutation_score_soundness`. The CI scripts remain the source of truth either way.

CI budgets:

- PR coverage job: ≤ 25 min p50 with warm caches (Rust build cache + prover-fixture cache). Cold-cache worst case (fixture regeneration) is accepted and expected to be rare.
- Test-quality routine: advisory; no wall-clock constraint on merges.
- Nightly job: ≤ 6 h (mutation testing dominates; the module list is sized to fit).

## Design

### Architecture

```
PR opened/updated
│
├─ coverage.yml
│   1. self-test ci/coverage_gate.py, ci/soundness_metrics.py
│   2. closure = cargo metadata closure(jolt-verifier) ∪ extras   ── invariant 1
│   3. validate coverage-floors.toml completeness                 ── invariant 2
│   4. for crate in closure: for path in paths(crate):
│        cargo llvm-cov nextest -p crate --features path --no-report
│      cargo llvm-cov report --json          (profiles merge cumulatively)
│   5. enforce per-crate floors                                   ── invariant 3
│   6. soundness metrics from region data + tamper manifest       ── invariant 5
│
├─ test-quality.yml (paths-filtered)
│   └─ _fire-claude-routine.yml → hosted routine
│        reads ci/test-quality-rubric.md + test hunks of the diff
│        emits schema-validated JSON                              ── invariant 4
│        posts PR review: per-test score table, <7 ⇒ rework-or-delete
│        gate behavior from ci/test-quality.toml (enforce=false at merge)
│
└─ nightly: coverage-nightly.yml
    cargo mutants over ci/mutants-modules.toml → report artifact + summary
```

- **Coverage measurement**: `cargo llvm-cov` accumulates profraw across successive `--no-report` runs in the same target dir, so per-crate cumulative coverage over multiple feature paths is a single merged report; the gate script filters the JSON per crate and compares against floors. Uninstantiated generics and cfg'd-out code are invisible to llvm-cov — floors are therefore honest about *compiled* code per declared path, which is exactly why paths are declared per crate.
- **Fixture caching**: the `prover-fixtures` tests already cache generated fixtures behind a lock file; CI persists that cache keyed on `Cargo.lock` + `crates/jolt-prover-legacy` content hash.
- **Error-variant coverage**: `soundness_metrics.py` inventories error-enum variants in closure crates (regex over `error.rs`-pattern files), locates `Err(Variant...)` construction sites, and checks execution via llvm-cov region data at those lines. Approximate by construction; misses are surfaced as "unlocatable variants" in the report rather than silently counted.
- **Tamper-manifest ratio**: parsed from `TamperTarget` dispositions in `jolt-verifier/tests/support/tamper_manifest.rs`; the manifest's own self-audit tests remain the semantic guard.
- **Routine wiring**: a third hosted routine alongside the existing code/spec-review routines, fired through the existing `_fire-claude-routine.yml` reusable workflow; the routine ID is minted by the repo owner and the trigger is `pull_request_target` with a paths filter instead of a label.
- **Rubric** (`ci/test-quality-rubric.md`): five dimensions scored 0–2, summed to 0–10 (floored at 1). Oracle strength; adversarial reach; failure specificity; independence & determinism; property clarity. Crypto-specific tripwires the routine must check: prover/verifier round-trips do not count as an oracle when both sides share the code under test (the self-consistency trap); transcript tests require pinned vectors; tamper tests must mutate the object the targeted check actually guards; randomized tests must be seeded.

### Alternatives Considered

- **cargo-tarpaulin instead of llvm-cov**: rejected — llvm-cov's region-level JSON is required for the error-variant metric, and it is accurate on macOS/Linux alike.
- **One global coverage number**: rejected — a large well-covered crate masks regressions in small soundness-critical ones; per-crate floors localize accountability.
- **Auto-ratchet (floor = high-water mark − ε)**: rejected in favor of manual ratchet — CI stays stateless, floors move only by reviewed edits; the cost (coverage may drift down toward a floor) is acceptable given floors are pinned at post-sweep values.
- **External coverage service**: rejected — thresholds belong in the repo next to the code they gate; no third-party data flow.
- **Blocking AI gate from day one**: rejected — the rubric needs calibration against real PRs before it can fail them; advisory-then-flip captures the value without the friction.
- **Mutation testing as a PR gate**: rejected — runtime is unbounded relative to PR latency budgets; nightly with a curated module list captures most of the signal.
- **Headless `claude` CLI in Actions instead of a hosted routine**: rejected — the repo already standardizes on hosted routines (`_fire-claude-routine.yml`); one pattern, one secret-handling story.
- **Enforcing the prover side too**: rejected by scoping rationale — prover bugs cannot produce unsound acceptances, and prover-side code is churning toward `jolt-prover`.

## Documentation

- New Jolt book page under the contributor documentation: how the gates work, how to run them locally (`cargo llvm-cov nextest -p <crate> --features <path>`, `python3 ci/coverage_gate.py --local`), how floors are raised, and the rubric's dimensions with examples of 9-scoring vs 3-scoring tests.
- `CLAUDE.md`: add the local gate commands to Essential Commands so agents run the same checks CI does.
- No user-facing zkVM behavior changes; no protocol documentation impact.

## Execution

Phased inside the single PR:

1. **Infra**: `ci/` scripts + configs with provisional floors at today's measured values; `coverage.yml`; validate on the PR itself.
2. **Soundness metrics**: error-variant + tamper-ratio computation wired into the coverage job with provisional floors.
3. **Routine**: rubric checked in; owner mints the routine ID; `test-quality.yml` wired advisory.
4. **Sweep** (priority order from the 2026-07-23 analysis): jolt-blindfold prover round-trips → jolt-verifier tamper hardening → transcript KATs → riscv/program decoder batteries → field solinas gaps → poly/openings/r1cs/crypto/claims/sumcheck/dory/hyperkzg/akita items → tracer/common to floor.
5. **Pin floors** at post-sweep measured values (target ≥ 80; documented exceptions otherwise); nightly mutation job enabled.
6. Merge with coverage enforcement **on** and AI gate **advisory**; flip `enforce = true` in a follow-up after the routine has scored several real PRs cleanly.

## References

- Internal coverage analysis, 2026-07-23 (session artifact: per-crate llvm-cov table + six-cluster qualitative review; headline numbers reproduced in Summary/Scope above)
- `jolt-verifier/tests/support/tamper_manifest.rs` — the manifest pattern the soundness metrics generalize
- [`jolt-eval/README.md`](../jolt-eval/README.md) — invariant/objective framework referenced under Invariants and Performance
- `.github/workflows/_fire-claude-routine.yml` — hosted-routine reusable workflow
- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov), [cargo-nextest](https://nexte.st/), [cargo-mutants](https://mutants.rs/)
- [specs/1370-spec-driven-workflow.md](1370-spec-driven-workflow.md) — the process this spec follows
