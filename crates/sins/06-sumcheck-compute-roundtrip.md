# Sin 6: SumcheckCompute Returns Heap-Allocated UnivariatePoly

## The Problem

```rust
pub trait SumcheckCompute<F: Field>: Send + Sync {
    fn round_polynomial(&self) -> UnivariatePoly<F>;  // heap-allocated Vec<F>
    fn bind(&mut self, challenge: F);
}
```

Every round, the witness produces `UnivariatePoly<F>` which is a `Vec<F>` (heap
allocation of D+1 field elements). The protocol flow per round:

1. **Device**: `pairwise_reduce` → D+1 scalars → download to CPU  (already heap via `Vec<F>`)
2. **CPU**: Reconstruct `UnivariatePoly` (claim derivation, interpolation)
3. **CPU**: Append coefficients to transcript, squeeze challenge
4. **Device**: Upload challenge, `interpolate_pairs` (bind)

Step 2 allocates. For D=4, that's 5 × 32 = 160 bytes per round per instance. Over 20
rounds × 6 stages × multiple instances, it's thousands of small allocations. Not
performance-critical compared to the O(N) bind, but it's design debt.

## Why This Is a (Minor) Sin

The trait interface is "CPU return" shaped, not "write to caller" shaped. This means:

1. **Unnecessary allocation**: D+1 coefficients could be returned as `[F; D]` (stack) or
   written into a caller-provided buffer. But the trait is generic over degree.

2. **Forces CPU roundtrip in the contract**: Even if the transcript lived on-device
   (future GPU transcript), the trait would force a download to construct `UnivariatePoly`
   on the host.

3. **Prevents fusion**: The transcript could absorb raw evaluations and interpolate
   internally, but the trait requires the witness to do interpolation before returning.

## Why It's Minor

- D+1 field elements (128-1024 bytes) is noise compared to the O(N) bind cost
- The transcript MUST run on CPU (Fiat-Shamir is sequential, hash-based)
- GPU→CPU bandwidth for D+1 elements is negligible (~1 PCIe transaction)
- `pairwise_reduce` already returns `Vec<F>`, so the allocation happens anyway

## Possible Future Direction

If this ever matters (it probably won't), the trait could use a buffer protocol:

```rust
pub trait SumcheckCompute<F: Field>: Send + Sync {
    /// Maximum degree (compile-time upper bound for stack allocation).
    const MAX_DEGREE: usize;

    /// Write round polynomial evaluations into `out[0..degree+1]`.
    /// Returns the actual degree.
    fn round_polynomial_into(&self, out: &mut [F]) -> usize;

    fn bind(&mut self, challenge: F);
}
```

But this adds complexity for near-zero benefit. **Recommended: leave as-is, document
the design rationale, revisit only if profiling shows allocation pressure.**

## Verdict

This is a conscious tradeoff, not a real sin. The heap allocation is correct, simple,
and not on any hot path. Including it here for completeness and as documentation of
the decision.
