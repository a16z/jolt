# Task 18: Route Polynomial Arithmetic Through Backend

## Status: TODO

## Anti-Pattern
Three op handlers do polynomial arithmetic on CPU:

1. **AbsorbRoundPoly — Uniskip encoding** (runtime.rs ~70 lines): Lagrange basis computation, `interpolate_to_coeffs`, O(k^2) `poly_mul` convolution, and result resizing. This runs every Uniskip round (stage 1).

2. **AbsorbRoundPoly — Compressed encoding** (runtime.rs ~15 lines): `UnivariatePoly::interpolate` on round evals. Runs every non-Uniskip round.

3. **BatchAccumulateInstance — degree extension** (runtime.rs ~20 lines): Conditional `UnivariatePoly::interpolate` + evaluate when instance degree < max_evals. Runs every round of every batched sumcheck.

4. **BatchRoundBegin — claim update** (runtime.rs ~15 lines): `UnivariatePoly::interpolate` + evaluate to update per-instance claims. Same pattern as #3.

## Fix
Add to `ComputeBackend`:

```rust
/// Encode round polynomial evaluations into transcript-ready coefficients (Uniskip path).
fn uniskip_encode<F: Field>(
    &self, raw_evals: &[F], domain_size: usize, domain_start: i64,
    tau: F, zero_base: bool, num_coeffs: usize,
) -> Vec<F>;

/// Encode round polynomial evaluations into transcript-ready coefficients (Compressed path).
fn compressed_encode<F: Field>(&self, evals: &[F]) -> Vec<F>;

/// Interpolate evaluations at {0,1,...,n-1} and evaluate at a point.
fn interpolate_evaluate<F: Field>(&self, evals: &[F], point: F) -> F;

/// Extend evaluations at {0,...,n-1} to {0,...,target-1} via interpolation.
fn extend_evals<F: Field>(&self, evals: &[F], target_len: usize) -> Vec<F>;
```

`interpolate_evaluate` is shared by BatchRoundBegin and BatchAccumulateInstance.

## Risk: Medium
Uniskip encoding feeds directly into transcript absorption — correctness is critical. The polynomials are small (degree 2-8 typically), so GPU benefit is marginal, but the abstraction is needed for the "no CPU escape hatch" guarantee.

## Dependencies: None
