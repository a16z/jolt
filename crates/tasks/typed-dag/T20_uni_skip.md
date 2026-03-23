# T20: Univariate Skip for S1 and S2

**Status**: `[ ]` Not started
**Depends on**: T07 (S1), T08 (S2)
**Blocks**: Fiat-Shamir transcript parity with jolt-core
**Crate**: `jolt-zkvm`, `jolt-sumcheck`
**Estimated scope**: Medium (~300 lines)

## Objective

Implement univariate skip (uni-skip) for the Spartan outer sumcheck (S1)
and Product Virtual sumcheck (S2). Uni-skip computes an analytic first-round
polynomial that skips the Lagrange-kernel multiplication, saving one
sumcheck round's worth of pair-wise reduction.

## Background

In jolt-core, stages 1 and 2 use uni-skip:
- **S1 (Outer)**: `OuterUniSkipProver` computes `s1(Y) = L(τ_high, Y) · t1(Y)`
  where t1 is evaluated at extended symmetric points {−D..D}. Degree 3D ≈ 21.
- **S2 (PV)**: `ProductVirtualUniSkipProver` computes a similar analytic
  first-round poly over the product constraints.

In the typed DAG, uni-skip is an implementation detail of `SumcheckCompute`:
the `first_round_polynomial()` method returns an analytic poly instead of
`None`, and the prover uses it for round 0.

## Existing Infrastructure

- **`SumcheckCompute::first_round_polynomial()`** — already in the trait.
  When it returns `Some(poly)`, `BatchedSumcheckProver` uses it for round 0.
- **`KernelEvaluator::set_first_round_override()`** — already exists.
  Sets the precomputed first-round polynomial.
- **`UnivariatePoly::interpolate()`** — Lagrange interpolation.
- **`jolt-core/src/subprotocols/univariate_skip.rs`** — reference impl of
  `build_uniskip_first_round_poly()`.

## Deliverables

### 1. Port `build_uniskip_first_round_poly` to jolt-zkvm

Create `crates/jolt-zkvm/src/evaluators/uni_skip.rs`:

```rust
/// Computes the analytic first-round polynomial for univariate skip.
///
/// Given base_evals (evaluations at the base window) and extended_evals
/// (evaluations at the extended symmetric window), reconstructs
/// t1(Y) on the full window and multiplies by L(tau_high, Y).
pub fn build_uniskip_first_round_poly<F: Field>(
    base_evals: &[F],
    extended_evals: &[F],
    tau_high: F,
    domain_size: usize,
) -> UnivariatePoly<F>
```

The function:
1. Merges base and extended evals into the full symmetric window
2. Interpolates t1 via Lagrange
3. Computes Lagrange kernel L(τ_high, Y) coefficients
4. Returns the product polynomial (coefficient convolution)

### 2. Integrate into S1 (Spartan outer)

In `prove_spartan()`:
1. After the Spartan outer sumcheck produces its proof, extract the
   uni-skip first-round data
2. OR: implement a custom `SumcheckCompute` that wraps the Spartan
   outer sumcheck and provides the uni-skip first round

Since jolt-spartan's `prove_dense_with_challenges` already handles
uni-skip internally, the main work is ensuring the proof carries the
uni-skip polynomial data and the verifier can check it.

### 3. Integrate into S2 (Product Virtual)

In `prove_stage2()`:
1. Before creating the PV `KernelEvaluator`, compute the uni-skip
   first-round polynomial from the product constraint evaluations
2. Call `evaluator.set_first_round_override(poly)` on the PV witness
3. The `BatchedSumcheckProver` will use it for round 0

Base evaluations come from the 5 product constraints at the base
domain (size 5). Extended evaluations require computing the fused
product witness `left_z(x) · right_z(x)` at extended grid points.

### 4. Update `SumcheckStageProof` if needed

The proof may need to carry uni-skip data separately from round
polynomials (the uni-skip poly has different degree than regular
rounds). Check if `SumcheckProof` handles this or if a wrapper
is needed.

## Key Details

### Outer Uni-Skip Domain
- Base domain: {-2, -1, 0, 1, 2} (5 points)
- Extended domain: {-D, ..., D} \ base (where D = 7 typically)
- Full domain: 2D + 1 = 15 points
- First-round poly degree: 3D ≈ 21

### PV Uni-Skip Domain
- Same domain structure as outer
- Base evaluations: 5 product constraint claimed sums
- Extended evaluations: product of fused left/right witnesses at shifted grids

### Fiat-Shamir Impact
- Uni-skip squeezes `tau_high` BEFORE the remaining sumcheck rounds
- The first-round poly coefficients are appended to the transcript
- Then the verifier samples r_0 from it

## Reference

- Uni-skip theory: `jolt-core/src/subprotocols/univariate_skip.rs`
- Outer uni-skip prover: `jolt-core/src/zkvm/spartan/outer.rs:149-312`
- PV uni-skip prover: `jolt-core/src/zkvm/spartan/product.rs:179-330`
- Lagrange helpers: `jolt-core/src/subprotocols/lagrange.rs`

## Acceptance Criteria

- [ ] `build_uniskip_first_round_poly` implemented and tested
- [ ] S2 PV stage uses uni-skip first-round override
- [ ] Fiat-Shamir transcript includes uni-skip poly before regular rounds
- [ ] Uni-skip proof verifiable by the verifier
- [ ] `cargo clippy -p jolt-zkvm` clean
