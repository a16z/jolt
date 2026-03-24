# jolt-poly Review

**Crate:** jolt-poly (Level 2)
**LOC:** 4,880
**Baseline:** 0 clippy warnings, 160 tests passing (was broken before fixes)
**Rating:** 8/10

## Overview

Polynomial types and evaluation primitives for Jolt. Provides `Polynomial<T>`
(evaluation tables over Boolean hypercube), `EqPolynomial`, `UnivariatePoly`,
`CompressedUnivariatePoly`, and traits `MultilinearPoly<F>` / `MultilinearEvaluation<F>`.
Also provides `RlcSource` for streaming PCS access and `LagrangeBasis` for
interpolation. Core data structure for the entire proving pipeline — used by
7+ downstream crates.

**Verdict:** Solid polynomial library with excellent generic scalar support
(`Polynomial<T>` where T ranges from bool to i128 to field elements). The
evaluation-table-as-struct pattern is clean and well-tested. The major naming
bug (`coefficients` meaning evaluations) has been fixed. bincode v2 migration
and `is_multiple_of` ambiguity resolved all compilation issues. 160 tests
now pass including a new LowToHigh binding correctness test.

---

## Findings

### [CQ-1.1] Field named `coefficients` stores evaluations, not coefficients
**File:** `src/cpu_polynomial.rs`
**Severity:** HIGH
**Finding:** `Polynomial<T>::coefficients` stores evaluation-table entries on the Boolean
hypercube, not polynomial coefficients. Accessor `coefficients()` propagated this
misnaming to 12 downstream call sites across 6 crates.
**Status:** RESOLVED — Renamed field to `evals`, accessor to `evals()`. Updated all 12
downstream call sites in jolt-sumcheck, jolt-blindfold, jolt-spartan, jolt-zkvm.

### [CQ-2.1] bincode v1 API usage causes compilation failure
**File:** `src/univariate.rs`, `src/compressed_univariate.rs`, `src/cpu_polynomial.rs`, `tests/integration.rs`
**Severity:** HIGH
**Finding:** Workspace uses bincode v2 but source files use v1 API. Tests fail to compile.
**Status:** RESOLVED — Migrated all 4 files to bincode v2 `encode_to_vec`/`decode_from_slice`.

### [CQ-2.2] `is_multiple_of` compilation error
**File:** `src/one_hot.rs:193`
**Severity:** HIGH
**Finding:** `i.is_multiple_of(3)` requires `num-integer::Integer` trait which isn't imported.
**Status:** RESOLVED — Changed to `i % 3 == 0`.

### [CQ-3.1] Missing LowToHigh binding test
**File:** `src/cpu_polynomial.rs`
**Severity:** LOW
**Finding:** The `bind` method has a `reverse: bool` parameter but no test for LowToHigh mode.
**Status:** RESOLVED — Added `low_to_high_binding_produces_correct_evaluation` test verifying
both binding orders produce identical evaluations.

### [CQ-4.1] Misleading lagrange.rs comment about batch inversion
**File:** `src/lagrange.rs`
**Severity:** LOW
**Finding:** Doc comment claims "a single batch inversion" but implementation uses per-element `inverse()`.
**Status:** RESOLVED — Updated to "$O(N)$ per-element inversions" and "$O(N^2)$ weight computation".

### [CD-1.1] EqPlusOnePolynomial::point field is public
**File:** `src/eq_plus_one.rs`
**Severity:** LOW
**Finding:** Internal field `point` exposed as `pub` with no need for external access.
**Status:** RESOLVED — Made private.

### [CD-2.1] RlcSource::for_each_row duplication
**File:** `src/rlc.rs`
**Severity:** LOW
**Finding:** Potential duplication between `for_each_row` and `fold_rows` on streaming types.
**Status:** PASS — Investigated; these are distinct methods with different semantics (callback vs fold).

### [CD-3.1] MultilinearPoly vs MultilinearEvaluation trait split undocumented
**File:** `src/cpu_polynomial.rs`
**Severity:** LOW
**Finding:** The distinction between these two traits is not obvious from names alone.
**Status:** DEFERRED — Documentation improvement, not blocking.

### [CD-4.1] Cargo.toml metadata incomplete
**File:** `Cargo.toml`
**Severity:** LOW
**Finding:** Missing authors, repository, keywords, categories; license was MIT-only.
**Status:** RESOLVED — Updated to dual MIT OR Apache-2.0, added all metadata.

---

## Summary

| Category | Pass | Resolved | Deferred | Total |
|----------|------|----------|----------|-------|
| CQ       | 0    | 5        | 0        | 5     |
| CD       | 1    | 2        | 1        | 4     |
| NIT      | 0    | 0        | 0        | 0     |
| **Total**| **1**| **7**    | **1**    | **9** |
