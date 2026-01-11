# Virtual Claim Optimization

## Key Insight

Stage 2 (virtualization) **already is** the concatenation sumcheck that Justin Thaler described. We're just sending unnecessary data to the verifier.

## Current Flow

```
Stage 1: Produce claims p₀(r_x)=v₀, p₁(r_x)=v₁, ..., p_{n-1}(r_x)=v_{n-1}
         ↓
         Send all n claims to verifier (e.g., 1,024 field elements)
         ↓
Stage 2: Prove Σₛ eq(r_s, s) · M(s, r_x) = v where v = Σᵢ eq(r_s, i) · vᵢ
```

## Optimized Flow

```
Stage 1: Produce claims p₀(r_x)=v₀, p₁(r_x)=v₁, ..., p_{n-1}(r_x)=v_{n-1}
         ↓
         Prover computes v = Σᵢ eq(r_s, i) · vᵢ locally
         ↓
         Send only v to verifier (1 field element)
         ↓
Stage 2: Prove Σₛ eq(r_s, s) · M(s, r_x) = v (unchanged)
```

## Why This Works

1. **Stage 2 already proves the aggregation**: The virtualization sumcheck proves that `v` is the correct weighted sum of all virtual claims
2. **No security loss**: The verifier doesn't need individual claims if Stage 2 proves the aggregation
3. **r_s is public**: Both prover and verifier can compute `eq(r_s, i)` coefficients

## Implementation Change

```rust
// Current: Store all virtual claims
pub struct RecursionProof {
    pub opening_claims: Openings<F>, // Contains all n virtual claims
    // ...
}

// Optimized: Store only aggregated claim
pub struct RecursionProof {
    pub aggregated_claim: F,  // Single value: Σᵢ eq(r_s, i) · vᵢ
    pub r_s: Vec<F>,         // Challenge used for aggregation
    // ...
}
```

## Concrete Savings

For GT exponentiation with 256-bit scalar:
- **Current**: 1,024 field elements (32 KB)
- **Optimized**: 1 field element (32 bytes)
- **Reduction**: 99.9%

## Why We Didn't See This Earlier

The current design sends virtual claims because:
1. It follows the standard sumcheck pattern of outputting all final evaluations
2. The connection between Stage 1 outputs and Stage 2 inputs wasn't viewed as redundant
3. Stage 2 was seen as "organizing" claims rather than "proving an aggregation"

But Stage 2 **is** proving the aggregation, so the individual claims are redundant.

## Compatibility

This optimization is compatible with all other optimizations:
- **With current approach**: 1,024 → 1 claim
- **With unified polynomials**: 4 → 1 claim
- **Works with any number of virtual polynomials**

## Summary

We don't need a new concatenation sumcheck - Stage 2 already does exactly that. We just need to stop sending the intermediate virtual claims that Stage 2 is going to aggregate anyway.