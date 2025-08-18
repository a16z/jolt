# SVO Type System for Compile-Time R1CS and Streaming Witnesses

This document updates the SVO type-system design to match the current R1CS architecture:
- Compile-time uniform constraints via `LC` and `Constraint` in `jolt-core/src/zkvm/r1cs/constraints.rs` with `UNIFORM_R1CS`.
- Streaming witness access via `WitnessRowAccessor` in `jolt-core/src/zkvm/r1cs/inputs.rs` (no materialized `input_polys`).
- Interleaved Spartan sumcheck with SVO precompute in `jolt-core/src/poly/spartan_interleaved_poly.rs` and `utils/small_value.rs`.

## 1. Problem Restatement

In the first SVO rounds of the Spartan sumcheck we evaluate terms of the form:
`e_in * Az_ext * Bz_ext`, where `e_in` is a full-width field element, and `Az_ext`, `Bz_ext` are extended (eval(1) − eval(0)) combinations across many constraint rows. These extended evaluations can exceed single-row ranges. We must avoid overflow and unnecessary modular reductions, while integrating with compile-time constraints and streaming witnesses.

## 2. Design Overview

We keep the two-level representation, but adapt it to the compile-time R1CS and streaming access patterns.

### Level 1: Per-row operand domains (single constraint wires)

We treat each of A, B, C as a const-friendly linear combination `LC` of `JoltR1CSInputs` with small integer coefficients. At row `t`, the evaluation is done via `LC::evaluate_row_with(&accessor, t)` without materializing polynomials. For SVO we need tighter numeric domains for per-row values before aggregation:

```rust
// New module: jolt-core/src/zkvm/r1cs/svo_types.rs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    // Small signed values from A-combos at a single row
    U5(u8), // encode |x| <= 31 with separate sign bit below
    U64AndSign { magnitude: u64, is_positive: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    // B can be larger per-row due to inclusion of immediate and register values
    U64(u64),
    U64AndSign { magnitude: u64, is_positive: bool },
    U128AndSign { magnitude: u128, is_positive: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzValue {
    Zero,
    U64(u64),
    U64AndSign { magnitude: u64, is_positive: bool },
    U128AndSign { magnitude: u128, is_positive: bool },
}
```

Notes:
- We avoid `i128` storage in favor of magnitude-plus-sign for uniformity and carry logic.
- `AzValue::U5` is justified by the structure of `UNIFORM_R1CS`: `A` LCs are mostly small sums/differences of a few inputs and flags; promote to `U64AndSign` on overflow.

Conversion adapters from `F` to these enums live next to `svo_types` and are used only in SVO code paths (not in generic field evaluation).

### Level 2: Extended evaluations and typed products

Extended evaluations are formed by recursive eval(1) − eval(0) aggregations over rows matching partial assignments in the first SVO rounds. We maintain compact, sign-aware accumulators with bounded limb widths:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzExtendedEval {
    I8(i8), // fast path when range is provably tiny
    U64AndSign { magnitude: u64, is_positive: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzExtendedEval {
    L1 { val: u64, is_positive: bool },      // <= 64 bits
    L2 { val: [u64; 2], is_positive: bool }, // <= 128 bits
    L3 { val: [u64; 3], is_positive: bool }, // <= 192 bits
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SVOProductValue {
    L1 { val: u64, is_positive: bool },
    L2 { val: [u64; 2], is_positive: bool },
    L3 { val: [u64; 3], is_positive: bool },
    L4 { val: [u64; 4], is_positive: bool }, // up to 256 bits pre-field-mul
}

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = ark_ff::BigInt<8>; // 512-bit unsigned integer
```

Rationale:
- `AzExtendedEval` stays extremely compact; promote to `U64AndSign` if proof-driven range analysis detects possible overflow out of `i8`.
- `BzExtendedEval` is limb-based to enable carry-safe multiplication without reduction.
- `SVOProductValue` covers the product space of `Az_ext * Bz_ext` without losing sign.

## 3. Streaming extended-eval with compile-time constraints

We do not materialize `A(z), B(z), C(z)` as multilinears. Instead, in `SpartanInterleavedPolynomial::new_with_precompute`, when iterating over ternary points for SVO, we:
1. Identify the set of witness rows consistent with the current partial assignment via the split `EqPolynomial` (`GruenSplitEqPolynomial`).
2. For each such row `t`, evaluate `LC` via `LC::evaluate_row_with(&accessor, t)` and convert to `AzValue` / `BzValue` using cheap, branchless range clamps.
3. Accumulate `(+/-)` into `AzExtendedEval` and `BzExtendedEval` using limb add/sub without reduction.
4. Form `SVOProductValue = Az_ext * Bz_ext` with a minimal limb width chosen by table-driven rules.
5. Multiply by `e_in` (4 limbs for 255–256-bit fields) into an `UnreducedProduct` with separate positive and negative accumulators; reduce once per block when required by the existing SVO accumulator boundaries.

This keeps the hot path zero-copy and uses the streaming accessor exclusively.

## 4. Concrete code change plan (no code changes in this doc; this is a checklist)

- New module: `jolt-core/src/zkvm/r1cs/svo_types.rs`
  - Define `AzValue`, `BzValue`, `CzValue`, `AzExtendedEval`, `BzExtendedEval`, `SVOProductValue`, and `UnreducedProduct`.
  - Provide lightweight constructors and `promote_*` helpers to move up variants when needed.
  - Provide `mul_limbs_{1,2,3,4}x{4}` helpers using `ark_ff::BigInt` limbs for carry-safe ops.

- New module: `jolt-core/src/zkvm/r1cs/svo_eval.rs`
  - `compute_extended_eval_az_row(acc: &mut AzExtendedEval, a_lc: &LC, accessor: &dyn WitnessRowAccessor<F>, t: usize)`; fast-path i8 accumulation with saturating promote.
  - `compute_extended_eval_bz_row(acc: &mut BzExtendedEval, b_lc: &LC, accessor: &dyn WitnessRowAccessor<F>, t: usize)`; limb accumulation with sign.
  - `combine_to_product(az: AzExtendedEval, bz: BzExtendedEval) -> SVOProductValue`.
  - `mul_with_field(product: SVOProductValue, e_in_4limbs: [u64; 4]) -> UnreducedProduct`.

- Integrate in `jolt-core/src/poly/spartan_interleaved_poly.rs`
  - In `SpartanInterleavedPolynomial::<NUM_SVO_ROUNDS, F>::new_with_precompute`, replace interim F-typed accumulators for SVO precompute with typed accumulators:
    - Maintain `pos_sum: UnreducedProduct` and `neg_sum: UnreducedProduct` per SVO accumulator slot.
    - Use streaming `LC::evaluate_row_with` and `svo_eval` helpers inside the existing loops over ternary points.
  - Ensure single modular reduction at the end of SVO precompute block, preserving the current API outputs.

- Touchpoints in `jolt-core/src/utils/small_value.rs`
  - Keep the shape of accumulators and distribution logic unchanged, but let their element type carry `UnreducedProduct` during precompute; normalize to `F` only at the boundary where the protocol requires field elements.
  - Where existing functions assume `F`, add overloads or generic adapters that accept typed accumulators and perform a final `mod p` when needed.

- Minor changes in `jolt-core/src/zkvm/r1cs/constraints.rs`
  - No changes to `UNIFORM_R1CS` construction.
  - Optionally add a feature-gated utility that returns conservative per-LC range hints (max absolute sum of coefficients times max witness magnitude) to drive `AzExtendedEval` fast-path selection.

## 5. API sketches

```rust
// jolt-core/src/zkvm/r1cs/svo_types.rs
pub fn add_with_sign_u64(mag_a: u64, sign_a: bool, mag_b: u64, sign_b: bool) -> (u64, bool);
pub fn add_limbs<const N: usize>(a: ([u64; N], bool), b: ([u64; N], bool)) -> ([u64; N], bool);
pub fn mul_limbs_az_bz(az: AzExtendedEval, bz: BzExtendedEval) -> SVOProductValue;
pub fn mul_product_by_field(prod: SVOProductValue, field_4limbs_le: [u64; 4]) -> UnreducedProduct;
```

```rust
// jolt-core/src/zkvm/r1cs/svo_eval.rs
pub fn eval_lc_row_typed<F: JoltField>(lc: &LC, accessor: &dyn WitnessRowAccessor<F>, t: usize)
    -> (AzValue, BzValue, CzValue);

pub fn accumulate_extended_az(az_acc: &mut AzExtendedEval, az_val: AzValue, sign: bool);
pub fn accumulate_extended_bz(bz_acc: &mut BzExtendedEval, bz_val: BzValue, sign: bool);
```

## 6. Correctness and overflow guarantees

- All intermediate additions/multiplications are performed on magnitudes with explicit sign, never casting to wider signed integers.
- Limb widths are chosen so that `max(Az_ext_bits) + max(Bz_ext_bits) <= 256` prior to the final multiply by a 256-bit field element, resulting in at most 512 bits in `UnreducedProduct`.
- One-time modular reduction per SVO block matches the protocol’s algebra, since additions commute and reductions can be deferred.

## 7. Performance notes

- Fast paths keep `AzExtendedEval::I8` and `BzExtendedEval::L1` as long as possible; promote lazily on overflow boundaries.
- Streaming access via `WitnessRowAccessor` removes polynomial materialization and improves cache locality.
- Limb helpers should use `ark_ff::BigInt` arithmetic to leverage existing intrinsics.

## 8. Testing strategy

- Unit tests for limb add/mul and sign handling.
- Property tests comparing typed SVO precompute with the baseline field-reduced implementation for small instances.
- Integration tests using existing examples (e.g., `examples/sha2-ex`) to ensure no transcript changes beyond expected SVO differences.

## 9. Implementation order

1. Add `svo_types.rs` and limb helpers; add unit tests.
2. Add `svo_eval.rs` row-eval and accumulation helpers.
3. Wire `new_with_precompute` to typed accumulators behind a feature flag (e.g., `svo_typed`).
4. Flip default to typed once tests and benches are green.

## 10. Non-goals

- No changes to the semantics or count of `UNIFORM_R1CS` constraints.
- No changes to transcript structure beyond the internal precompute arithmetic representation.
