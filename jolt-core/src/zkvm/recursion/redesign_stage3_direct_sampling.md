# Redesigning Stage 3 for Direct Sampling Protocol

## Executive Summary

This document explains how to adjust Stage 3 of the recursion SNARK to work with directly sampled points from the new one-round protocol, replacing the sumcheck-derived points from the old Stage 2.

## The Core Issue

### Old Protocol Flow
1. Stage 2 ran a sumcheck over `s` variables: `Σ_s eq(r_s_init, s) · M(s, r_x) = Σ_i eq(r_s_init, i) · v_i`
2. Through sumcheck rounds, it bound `s` variables sequentially, producing `r_s_final`
3. Stage 3 received: `M(r_s_final, r_x) = v_sparse` where `r_s_final` came from the binding process
4. Stage 3 proved: `v_sparse = Σ_i q(i) · f_jagged(r_s_final, r_x, i)`

### New Protocol Flow
1. Directly sample all of `r_s` at once from the transcript
2. Compute and verify: `M(r_s, r_x) = Σ_i eq(r_s, i) · v_i`
3. Stage 3 receives: `M(r_s, r_x) = v_sparse` where `r_s` was directly sampled
4. Stage 3 must prove: `v_sparse = Σ_i q(i) · f_jagged(r_s, r_x, i)`

## The Mathematical Adjustment

The key insight is that Stage 3's correctness doesn't fundamentally depend on HOW the evaluation point was derived, but rather on the mathematical relationship between the sparse and dense representations.

### Current Stage 3 Mathematics

The jagged transform proves:
```
M(r_s, r_x) = Σ_{i∈{0,1}^m} q(i) · f_jagged(r_s, r_x, i)
```

Where:
- `M(s,x)` is the sparse constraint matrix
- `q(i)` is the dense polynomial containing non-redundant values
- `f_jagged(r_s, r_x, i) = eq(row(i), r_s) · eq(col(i), r_x)` for boolean `i`
- `row(i)` and `col(i)` map dense index `i` to sparse coordinates

### Why This Still Works

The jagged indicator function `f_jagged` implements the bijection between:
- Dense indices `i ∈ {0,1}^m`
- Sparse positions `(s,x)` where values are non-zero

This bijection is independent of how `r_s` was obtained. Whether `r_s` came from:
- Sequential sumcheck binding (old protocol)
- Direct sampling (new protocol)

The mathematical relationship remains valid:
```
M̃(r_s, r_x) = Σ_{i∈{0,1}^m} q̃(i) · f̃_jagged(r_s, r_x, i)
```

## Required Adjustments

### 1. No Changes to Prover Algorithm

The Stage 3 prover remains unchanged because:
- It receives `(r_s, r_x)` as the evaluation point
- It computes `eq(row(i), r_s)` and `eq(col(i), r_x)` the same way
- The sumcheck protocol proceeds identically

### 2. No Changes to Verifier Algorithm

The Stage 3 verifier remains unchanged because:
- It receives the same sparse claim `M(r_s, r_x) = v`
- It verifies the sumcheck using the same jagged indicator evaluation
- The final dense polynomial opening is verified identically

### 3. Key Implementation Details

The critical parts that remain correct:

```rust
// In Stage 3 prover construction
let eq_s_cache = vec![F::zero(); num_polynomials];
for poly_idx in 0..num_polynomials {
    let matrix_row = matrix_rows[poly_idx];
    let row_bits = index_to_binary_vec(matrix_row, num_s_vars);
    // This works correctly whether r_s is from sumcheck or direct sampling
    eq_s_cache[poly_idx] = EqPolynomial::mle(&row_bits, &r_s);
}
```

## Soundness Analysis

### Why Soundness is Preserved

1. **Schwartz-Zippel for Stage 2**: The direct evaluation check ensures that with high probability, all individual polynomial claims from Stage 1 are correct.

2. **Stage 3 Sumcheck Soundness**: The jagged sumcheck proves the correct relationship between sparse and dense evaluations, regardless of how the evaluation point was derived.

3. **Composition**: The overall soundness comes from:
   - Stage 1: Individual constraint sumchecks are sound
   - New Stage 2: Linear combination check via Schwartz-Zippel
   - Stage 3: Jagged transform sumcheck relates sparse to dense

### Potential Attack Vectors (and why they fail)

**Attack 1**: Prover tries to cheat by providing false Stage 1 claims
- **Defense**: The direct evaluation check in Stage 2 catches this with overwhelming probability

**Attack 2**: Prover tries to provide incorrect dense polynomial in Stage 3
- **Defense**: The jagged sumcheck ensures the dense polynomial evaluation matches the sparse claim

**Attack 3**: Prover tries to exploit the different sampling method
- **Defense**: The evaluation point `r_s` is still uniformly random, providing the same security

## Implementation Path

### Phase 1: Verification
1. Run both old and new protocols in parallel
2. Verify that Stage 3 produces identical results with both evaluation points
3. Confirm sumcheck polynomials match

### Phase 2: Integration
1. Remove Stage 2 sumcheck code
2. Pass directly sampled `r_s` to Stage 3
3. No modifications needed to Stage 3 code

### Phase 3: Optimization
1. Consider optimizing Stage 3 for the new structure
2. Potential for batching or parallelization improvements

## Mathematical Proof of Correctness

**Theorem**: The Stage 3 jagged transform sumcheck remains sound when using directly sampled `r_s` instead of sumcheck-derived `r_s_final`.

**Proof**:
1. Let `π_old` be the old protocol with sumcheck-derived `r_s_final`
2. Let `π_new` be the new protocol with directly sampled `r_s`

For both protocols:
- The sparse matrix `M` is the same
- The dense polynomial `q` is the same
- The bijection functions `row()` and `col()` are the same

The only difference is the evaluation point. In both cases:
- The point is uniformly random over the field
- The jagged indicator function is evaluated correctly
- The sumcheck verifies the same mathematical relationship

Therefore, if a cheating prover could break `π_new`, they could also break `π_old` by simulating the direct sampling internally, contradicting the assumed security of `π_old`. ∎

## Benefits of the Adjustment

1. **Efficiency**: Eliminates entire Stage 2 sumcheck
2. **Simplicity**: Fewer rounds of interaction
3. **Modularity**: Stage 3 becomes more self-contained
4. **Proof Size**: Smaller proofs without Stage 2 sumcheck data

## Conclusion

The Stage 3 jagged transform can work correctly with directly sampled points without any code changes. The mathematical relationship it verifies remains valid regardless of how the evaluation point was obtained. This allows us to safely eliminate Stage 2 sumcheck while maintaining the overall soundness of the recursion SNARK.

The key insight is that the jagged transform is proving a structural property of the sparse-to-dense mapping, not a property specific to sumcheck-derived points. As long as the evaluation point is uniformly random (which both protocols ensure), the security guarantees remain intact.