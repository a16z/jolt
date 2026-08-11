# Spec: Verifier-Closure Lint Hardening

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Markos Georghiades             |
| Created     | 2026-07-26                     |
| Status      | implemented                    |
| PR          | #1703                          |

## Summary

`jolt-verifier` is the protocol-defining crate and consumes untrusted proof bytes; in WASM and embedded deployments a panic is a denial of service. The workspace lints already deny `panic!`/`unwrap`/`expect`/`todo!`/`unimplemented!` and undocumented unsafe, but leave the remaining panic and UB sources unconstrained: slice indexing (`a[i]`), `unreachable!`, silently wrapping release-mode arithmetic, and `unsafe` itself. This spec adds mechanically enforced, per-crate lint discipline over the verifier's runtime crate set, and fixes or explicitly justifies every existing violation. Survey (2026-07-26, clippy 1.95, lib targets only):

| crate | indexing+slicing | unreachable | unsafe sites | arithmetic |
|---|---|---|---|---|
| jolt-poly | 246 | 0 | 3 | 313 |
| jolt-lookup-tables | 175 | 8 | 1 | 388 |
| jolt-claims | 105 | 0 | 0 | 317 |
| jolt-field | 92 | 3 | 0 | 95 |
| jolt-verifier | 42 | 2 | 0 | 73 |
| jolt-blindfold | 41 | 0 | 0 | 67 |
| jolt-openings | 40 | 0 | 0 | 58 |
| jolt-hyperkzg | 32 | 0 | 0 | 36 |
| jolt-r1cs | 28 | 0 | 0 | 59 |
| jolt-crypto | 18 | 1 | 10 | 61 |
| jolt-transcript | 17 | 0 | 0 | 15 |
| jolt-dory | 16 | 1 | 7 | 11 |
| jolt-program | 11 | 2 | 0 | 29 |
| jolt-sumcheck | 9 | 0 | 0 | 20 |
| common | 5 | 0 | 0 | 13 |
| jolt-akita | 3 | 8 | 0 | 3 |
| jolt-riscv | 0 | 0 | 0 | 41 |
| jolt-claims-derive | 0 | 1 | 0 | 0 |
| jolt-verifier-derive | 0 | 0 | 0 | 0 |

## Intent

### Goal

Enforce panic- and unsafe-discipline via crate-root lint attributes on the 19 verifier-runtime crates: the 16-crate in-workspace normal-dependency closure of `jolt-verifier` plus `jolt-dory`, `jolt-akita`, and `jolt-hyperkzg`, which are dev-dependencies of the generic library but linked by every production deployment as the concrete PCS.

Four enforcement tiers:

1. **All 19 crates**: `#![deny(clippy::get_unwrap, clippy::string_slice, clippy::fallible_impl_from, clippy::mem_forget, clippy::exit, clippy::panic_in_result_fn, clippy::let_underscore_must_use, clippy::host_endian_bytes, clippy::wildcard_enum_match_arm)]`. Existing violations: 1 `get_unwrap`, 1 `panic_in_result_fn`, 5 `wildcard_enum_match_arm`; the last four lints are otherwise at zero and act as regression guards — `host_endian_bytes` blocks platform-dependent serialization (WASM parity), `wildcard_enum_match_arm` forces every match over a protocol enum to break at compile time when a variant is added instead of silently absorbing it into `_ =>`. jolt-claims also omits `wildcard_enum_match_arm` (79 hits, all fail-closed resolvers returning `Err`/`None` or sparse test evaluation maps; no wrong-success path exists for the lint to catch). The two proc-macro crates omit `wildcard_enum_match_arm`: they match foreign syn AST enums, where wildcard fallbacks to `Err`/`None` are the correct, version-stable idiom.

   **`clippy::unreachable` is denied in jolt-verifier only**, not across the closure. The hole it closes is real and specific: `unreachable!` is the only one of the four abort macros that escapes both `clippy::panic` and `clippy::panic_in_result_fn`, and base `jolt-dory/src/transcript.rs` used it as the `unimplemented!()` substitute that compiled. But across the closure it does not pay: of 29 pre-existing sites it eliminated 4, all provably-dead branches the compiler could not see (`k % 4` over literal-only callers, a `windows(2)` postcondition, two stage8 or-pattern artifacts), leaving 23 production `#[expect]`s in crates where the branch is compile-time impossible (const-generic rotation dispatch in jolt-lookup-tables, masked-bit invariants in jolt-field) or settled by setup-time dispatch validation (jolt-akita). Those annotations are unverifiable prose, not a checked invariant, and the lint is structurally blind to the failure that matters — it flags the macro, never the truth of the claim in its argument. jolt-verifier went 2 → 0 in this campaign and carries no hatch, so the deny is free exactly where untrusted bytes arrive. Reconsider extending it if a crate's `unreachable!` count reaches zero on its own.
2. **Unsafe containment**: `#![forbid(unsafe_code)]` on the 15 crates with zero unsafe; `#![deny(unsafe_op_in_unsafe_fn)]` on the 4 with unsafe (jolt-crypto 10 sites, jolt-dory 7, jolt-poly 3, jolt-lookup-tables 1). Each retained site is assessed: removed where a safe form has no measured cost, kept with its SAFETY comment otherwise.
3. **Control-plane crates** (all except jolt-poly, jolt-lookup-tables, jolt-claims, jolt-field): `#![deny(clippy::indexing_slicing)]`, fixing ~260 sites. Proof-input-reachable paths return typed errors; structurally-guaranteed indexing is rewritten with iterators/`get` or annotated with a reason.
4. **jolt-verifier only — numeric discipline**: `#![deny(clippy::arithmetic_side_effects, clippy::as_conversions, clippy::integer_division)]`, fixing 139 sites (73 arithmetic, 64 `as` casts, 2 integer divisions). The workspace deliberately allows the cast lints for field-arithmetic code, which also switched them off in the verifier; here `as` on untrusted lengths and indices truncates silently, so casts become `From`/`TryFrom` with error returns and divisions become explicit (`div_ceil` or documented floor). A `crates/jolt-verifier/clippy.toml` adds `disallowed-methods` for `wrapping_*`/`unchecked_*` integer ops (zero current uses; closes the loophole where explicit wrapping evades `arithmetic_side_effects`) and `disallowed-types` for `HashMap`/`HashSet`/`SystemTime`/`Instant` (zero current uses; verification must be a deterministic pure function). Extending numeric discipline to dep crates is examined after this lands.

### Invariants

- Proof acceptance is unchanged: every fix is behavior-preserving on paths honest proofs exercise; panic-to-`Err` conversions may only affect inputs that previously crashed. `muldiv` e2e passes in both `host` and `host,zk`, and the jolt-verifier suites pass unchanged.
- Every `#[expect]` introduced carries a `reason`.
- Prover performance is untouched: shared-crate changes (jolt-poly, jolt-field, jolt-lookup-tables, jolt-claims) are limited to non-loop code in this campaign; their indexing stays warn-level (see Non-Goals).
- New error variants are constructed under test (forward-compatible with the error-variant coverage floor from `specs/test-quality-ci.md`).

### Non-Goals

- Hot-path indexing in jolt-poly (246), jolt-lookup-tables (175), jolt-claims (105), jolt-field (92): loop-heavy code shared with the prover. Deferred to per-crate follow-up PRs gated on criterion benchmarks; blanket `#[expect]` annotation is explicitly rejected as noise.
- Numeric lints outside jolt-verifier: `arithmetic_side_effects` (1,476 remaining sites) and `as` casts (856 remaining) pervade modular-arithmetic code and need a per-crate strategy.
- Variable-amount bit shifts have no clippy lint; debug/test builds panic on shift overflow (overflow-checks) but release wraps. Accepted residual risk, noted for review attention.
- Out-of-workspace proof-byte-reachable code: the vendored `dory`/dory-pcs crates and the arkworks fork are outside crate-root attribute reach — they are exactly the dependency-surface tier of the byte-boundary hardening campaign (#1674). Likewise `jolt-sdk` (`deserialize_verifier_object`, the WASM byte entry) sits outside the 19-crate closure; it is a thin fallible bincode wrapper today but is #1674's primary target. Resource-exhaustion/allocation amplification is lint-invisible and also belongs to #1674's plan. The two campaigns should cross-reference to avoid drift.
- The registers `rw_config` phase split is validated eagerly at both stage-4 fronts (`validate_phase_split` before any round-count accessor); jolt-claims' geometry accessors themselves stay outside the numeric tier with the rest of the crate (see the numeric-lints Non-Goal above).
- Fallible `MemoryLayout` construction: `common` was found to be outside the workspace lint regime entirely (no `[lints] workspace = true`); this campaign adds it, and its layout-overflow `expect`s are documented rather than converted, since a `Result`-returning constructor ripples into prover-side callers. Tracked as follow-up.
- Prover-side crates (jolt-prover, jolt-prover-legacy, tracer, jolt-witness, jolt-kernels).
- Test code: unit-test modules may `#[expect]` the strict lints wholesale per the existing test lint policy; `tests/` directories are separate targets and out of scope.

## Evaluation

### Acceptance Criteria

- [x] `cargo clippy --all --features host --all-targets` and `--features host,zk` pass with all attributes in place (no CI workflow changes needed; crate-root attributes bind locally and in CI identically).
- [x] 15 crates carry `#![forbid(unsafe_code)]`; total unsafe sites in scope ≤ 21, each with a SAFETY comment; any removal in jolt-crypto/jolt-poly is benchmarked.
- [x] Zero `indexing_slicing` violations in the 15 control-plane crates' lib code; zero `arithmetic_side_effects`/`as_conversions`/`integer_division` violations in jolt-verifier lib code; `crates/jolt-verifier/clippy.toml` disallows wrapping/unchecked integer ops and nondeterministic collections/clocks.
- [x] Zero `unreachable!` in jolt-verifier, and zero `#[expect(clippy::unreachable)]` anywhere in the closure — the lint is denied only where its hatch count is zero.
- [x] Every `#[expect(clippy::...)]` added has a non-empty `reason` (grep-checkable).
- [x] `muldiv` e2e passes in both modes; full workspace `cargo nextest run` is green.

### Testing Strategy

Existing suites are the primary guard, in both standard and ZK modes. Each new `VerifierError` variant introduced by a panic-to-`Err` conversion gets a rejection test that constructs the triggering input. No new test infrastructure.

### Performance

Control-plane fixes are off hot paths; no benchmark movement expected. Any unsafe removal or arithmetic change in jolt-crypto (GLV/MSM paths) or jolt-poly is validated against the existing criterion benchmarks with no regression beyond noise; a regression means the site keeps its unsafe/unchecked form with documentation.

## Design

### Architecture

Enforcement lives in crate-root attributes in each crate's `lib.rs`, not in Cargo lint tables or CI flags. Rationale: workspace-level denies would hit prover crates whose hot loops index legitimately; per-crate `[lints]` tables would replace `workspace = true` and fork the shared config; CI-only flags drift from what developers see locally. Attributes are visible in the code they govern, compose with the inherited workspace lints, and `forbid` resists future local overrides.

### Alternatives Considered

- Workspace-wide deny + `#[expect]` across prover code: rejected, hundreds of annotations in crates this campaign does not own.
- A shared `#![doc(include)]`-style macro or build-rs injection of attributes: rejected, indirection for 8 lines of attributes per crate.
- Fixing hot-path indexing now: rejected per scope decision; iterator refactors in sumcheck inner loops carry prover-regression risk that needs benchmark-gated, per-crate PRs.

## Documentation

One paragraph in CLAUDE.md's Lint Policy section stating which crates are strict, why (verifier-runtime closure), and that new code in them must fix or `#[expect]`-with-reason. A book subsection is deferred: the book has no dev section on main yet (`specs/test-quality-ci.md` adds one); add the lint-policy page there once that PR lands.

## Execution

Implementation slices, each leaving clippy green in both feature modes:

1. Attributes on all 19 crates + fix the 33 tier-1 violations + `forbid(unsafe_code)` on the 15 clean crates.
2. Unsafe assessment in jolt-crypto, jolt-dory, jolt-poly, jolt-lookup-tables (remove or document; benchmark where hot).
3. `indexing_slicing` per crate, smallest first: jolt-riscv, common, jolt-akita, jolt-sumcheck, jolt-program, jolt-dory, jolt-transcript, jolt-crypto, jolt-r1cs, jolt-hyperkzg, jolt-openings, jolt-blindfold, jolt-verifier.
4. Verifier numeric discipline: `arithmetic_side_effects` + `as_conversions` + `integer_division` (139 sites) + `clippy.toml` disallow lists.

Slices 3–4 may split into follow-up PRs if the diff outgrows review; the attribute in a crate lands in the same commit as that crate's fixes, never before.

## References

- `specs/test-quality-ci.md` — error-variant coverage floor this campaign feeds.
- Survey method: `cargo clippy --lib -- -W clippy::indexing_slicing -W clippy::string_slice -W clippy::unreachable -W clippy::fallible_impl_from -W clippy::arithmetic_side_effects -W clippy::mem_forget -W clippy::exit -W clippy::get_unwrap` on this branch's base (main @ 3ab638dd7).
