# Shift Sum-Check Optimization for Packed GT Exponentiation

## Overview

This document describes an optimization to the packed GT exponentiation protocol that eliminates the need to commit to the `rho_next` polynomial by using a specialized shift sum-check protocol.

## Current Approach

In the current packed GT exponentiation implementation, we commit to three polynomials per GT exponentiation:

1. `rho(s,x)` - intermediate results at each step
2. `rho_next(s,x)` - shifted intermediate results where `rho_next(s,x) = rho(s+1,x)`
3. `quotient(s,x)` - quotient polynomials for ring switching

The constraint that must hold is:
```
C(s,x) = rho_next(s,x) - rho(s,x)² × base(x)^{bit(s)} - quotient(s,x) × g(x) = 0
```

## The Problem

Committing to `rho_next` is redundant since it's completely determined by `rho` through the shift relationship `rho_next(s,x) = rho(s+1,x)`. This adds unnecessary commitment overhead and proof size.

## Integration with Recursion SNARK Stages

The recursion SNARK has four stages:

1. **Stage 1**: Constraint sum-checks (GT Exp, GT Mul, G1 Scalar Mul) → virtual polynomial claims at r_x
2. **Stage 2**: Direct evaluation protocol → combines all claims into matrix M(r_s, r_x)
3. **Stage 3**: Jagged transform → sparse M to dense q
4. **Stage 4**: Opening proof (Hyrax over Grumpkin)

Currently, the packed GT constraint sum-check in Stage 1 produces these virtual claims:
- `PackedGtExpRho(i)`: ρ(r_s*, r_x*)
- `PackedGtExpRhoNext(i)`: ρ_next(r_s*, r_x*)
- `PackedGtExpQuotient(i)`: Q(r_s*, r_x*)

## Proposed Optimization: Shift Sum-Check

Instead of committing to `rho_next`, we can use a sum-check protocol to prove that a claimed value equals `rho` at a shifted position. The shift sum-check would run at the **end of Stage 1**, after the packed GT constraint sum-check completes but before Stage 2 begins.

### Core Idea

Given:
- A committed polynomial `rho(s,x)` (12 variables: 8 step + 4 element)
- A challenge point `(r_s*, r_x*)`
- A claimed value `v` that supposedly equals `rho_next(r_s*, r_x*)`

We prove that `v = rho(r_s*+1, r_x*)` using the sum-check relation:

```
v = Σ_{s∈{0,1}^8, x∈{0,1}^4} EqPlusOne(r_s*, s) × Eq(r_x*, x) × rho(s,x)
```

Where:
- `EqPlusOne(r_s*, s)` = 1 if `s = r_s* + 1`, and 0 otherwise
- `Eq(r_x*, x)` = 1 if `x = r_x*`, and 0 otherwise

### Why This Works

The sum evaluates to:
- When `s = r_s*+1` and `x = r_x*`: contributes `1 × 1 × rho(r_s*+1, r_x*) = rho(r_s*+1, r_x*)`
- All other terms: contribute `0` (either `EqPlusOne` or `Eq` is 0)

Therefore, the sum equals exactly `rho(r_s*+1, r_x*)`, which is `rho_next(r_s*, r_x*)` by definition.

## Modified Protocol Flow with Recursion SNARK

### Current Flow (Stage 1)
1. Packed GT constraint sum-check runs (12 rounds)
2. After binding all variables to `(r_s*, r_x*)`:
   - Cache opening: `PackedGtExpRho(i)` → ρ(r_s*, r_x*)
   - Cache opening: `PackedGtExpRhoNext(i)` → ρ_next(r_s*, r_x*)
   - Cache opening: `PackedGtExpQuotient(i)` → Q(r_s*, r_x*)
3. These claims feed into Stage 2's direct evaluation

### Optimized Flow (Stage 1 + Shift Sum-Check)
1. **Stage 1a**: Packed GT constraint sum-check runs (12 rounds)
   - Prover commits only to `rho` and `quotient` (not `rho_next`)
   - After binding all variables to `(r_s*, r_x*)`:
     - Cache opening: `PackedGtExpRho(i)` → ρ(r_s*, r_x*)
     - Cache opening: `PackedGtExpQuotient(i)` → Q(r_s*, r_x*)
     - Prover claims: v = ρ_next(r_s*, r_x*)

2. **Stage 1b**: Shift sum-check (12 rounds)
   - Proves that v = ρ(r_s*+1, r_x*) using the relation:
     ```
     v = Σ_{s,x} EqPlusOne(r_s*, s) × Eq(r_x*, x) × rho(s,x)
     ```
   - After verification, cache the verified claim:
     - `PackedGtExpRhoNext(i)` → v (now verified)

3. **Stage 2**: Proceeds normally with all virtual claims
   - Direct evaluation combines all claims (including verified `rho_next`)
   - No changes needed to Stage 2 protocol

### Detailed Stage Integration

The shift sum-check acts as a "Stage 1.5" that bridges the constraint sum-check and Stage 2:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1a: Packed GT Constraint Sum-Check                   │
│  - 12 rounds over (s,x)                                      │
│  - Commits: rho, quotient (NOT rho_next)                    │
│  - Output: claims at (r_s*, r_x*) + unverified rho_next claim│
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1b: Shift Sum-Check (NEW)                            │
│  - 12 rounds over (s,x)                                      │
│  - Proves: claimed rho_next = rho(r_s*+1, r_x*)            │
│  - Output: verified rho_next claim                          │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Direct Evaluation Protocol                        │
│  - Uses all virtual claims (including verified rho_next)    │
│  - No protocol changes needed                               │
└─────────────────────────────────────────────────────────────┘
```

## Shift Sum-Check Details

### Round Structure
The shift sum-check has 12 rounds (8 for step variables, 4 for element variables):

**Rounds 1-8 (Step variables)**:
- Bind the `s` variables of both `EqPlusOne(r_s*, s)` and `rho(s,x)`
- `Eq(r_x*, x)` remains constant for each `x` block

**Rounds 9-12 (Element variables)**:
- `EqPlusOne` is fully bound (constant)
- Bind the `x` variables of both `Eq(r_x*, x)` and `rho(s,x)`

### Polynomial Evaluations

For round `j`, the prover sends a univariate polynomial `p_j(X_j)` of degree at most 2, where:
- Degree 1 from `EqPlusOne` or `Eq` (linear in each variable)
- Degree 1 from `rho` (multilinear)
- Total degree = 1 + 1 = 2

### Handling Boundary Cases

When `r_s* = 2^8-1` (maximum step value), there is no "next" step. In this case:
- The protocol should define `rho_next(2^8-1, x) = 0` for all `x`
- The shift sum-check would prove this equals 0

## Integration with Packed GT Constraint Sum-Check

The packed GT constraint sum-check needs to evaluate:
```
Σ_{s,x} eq_s(r_s, s) × eq_x(r_x, x) × C(s,x) = 0
```

Where `C(s,x)` requires both `rho(s,x)` and `rho_next(s,x)`.

### Modified Constraint Sum-Check

1. During the constraint sum-check, when evaluating the constraint polynomial:
   - Use the committed `rho(s,x)` directly
   - For `rho_next(s,x)`, use the shift relationship

2. After all rounds, at the final challenge point `(r_s*, r_x*)`:
   - Get `rho(r_s*, r_x*)` from opening proof
   - Run shift sum-check to get verified `rho_next(r_s*, r_x*)`
   - Get `quotient(r_s*, r_x*)` from opening proof
   - Verify the constraint equation

## Benefits

1. **Reduced Commitments**: Save one polynomial commitment per GT exponentiation
2. **Smaller Proof Size**: One less opening proof to transmit
3. **Conceptual Clarity**: The shift relationship is proven rather than trusted
4. **Reusability**: The shift sum-check technique could apply to other shifted polynomials

## Trade-offs

1. **Additional Rounds**: Adds 12 sum-check rounds for the shift verification
2. **Complexity**: More complex verifier implementation
3. **Prover Work**: Additional sum-check computation (though saves commitment work)

## Security and Soundness Analysis

### Soundness Preservation
The shift sum-check maintains the same soundness error as a standard sum-check:
- Soundness error per round: deg/|𝔽| = 2/p ≈ 2^{-254}
- Total soundness error for 12 rounds: ≤ 12 × 2^{-254} ≈ 2^{-250}
- This adds to the existing constraint sum-check error, maintaining overall security

### Transcript Integration
The shift sum-check must be properly integrated into the Fiat-Shamir transcript:
1. After Stage 1a completes, prover sends claimed `rho_next` value
2. This claim is appended to the transcript before shift sum-check begins
3. All shift sum-check messages are added to the transcript
4. The verified claim is then used in Stage 2

### Verifier Complexity
The verifier must:
1. Run the standard packed GT constraint sum-check verification
2. Run the shift sum-check verification (12 additional rounds)
3. Check that the verified `rho_next` claim satisfies the constraint equation

Total verifier work increases by O(12 × degree) = O(24) field operations per round.

## Implementation Considerations

1. **Batching**: Multiple shift sum-checks (if multiple GT exps) can be batched:
   ```
   Σ_i γ^i × (v_i - Σ_{s,x} EqPlusOne(r_s*_i, s) × Eq(r_x*_i, x) × rho_i(s,x)) = 0
   ```

2. **Polynomial Access**: During the shift sum-check, the prover needs access to:
   - The committed `rho` polynomial (already available)
   - Ability to evaluate `EqPlusOne` and `Eq` (standard sum-check operations)

3. **Stage Boundaries**: Clear interfaces between stages:
   - Stage 1a output: Standard virtual claims + unverified `rho_next` claim
   - Stage 1b output: Verified `rho_next` claim
   - Stage 2 input: All virtual claims (verified)

## Performance Analysis

### Costs
- **Additional Rounds**: +12 sum-check rounds
- **Verifier Work**: +O(24) field operations per round
- **Proof Size**: +12 field elements (one per round)

### Savings
- **Commitment Reduction**: -1 polynomial commitment per GT exp
- **Opening Proof**: -1 opening proof in Stage 4
- **Proof Size**: Approximately -160 bytes per GT exp

### Net Benefit
For applications with many GT exponentiations:
- Break-even: When commitment + opening cost > 12 rounds of sum-check
- Significant benefit: When batching multiple GT exps (amortized shift sum-check cost)

## Conclusion

The shift sum-check optimization provides a concrete way to reduce commitment overhead in the packed GT exponentiation protocol. By adding a specialized sum-check between Stage 1a and Stage 2, we can eliminate the redundant `rho_next` commitment while maintaining the same security guarantees.

The technique integrates cleanly with the existing recursion SNARK architecture, requiring no changes to Stage 2 or beyond. The optimization is particularly beneficial when multiple GT exponentiations are performed, as the shift sum-checks can be batched together.

This approach demonstrates a general principle: redundant committed data that has a known algebraic relationship to other committed data can often be replaced with a proof of that relationship, trading computation for reduced proof size.