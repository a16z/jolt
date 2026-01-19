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

## Implementation with Accumulator Pattern

### Accumulator Communication Flow

The recursion SNARK uses an `OpeningAccumulator` to track polynomial claims across stages. Here's how the shift sumcheck integrates:

#### Current Flow (Without Optimization)

```rust
// Stage 1a: Packed GT constraint sumcheck
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRho(i),
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    rho_claim,
);
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRhoNext(i),  // Committed polynomial
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    rho_next_claim,
);
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpQuotient(i),
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    quotient_claim,
);
```

#### Optimized Flow (With Shift Sumcheck)

```rust
// Stage 1a: Packed GT constraint sumcheck
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRho(i),
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    rho_claim,
);
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpQuotient(i),
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    quotient_claim,
);

// Prover claims rho_next value (unverified)
let rho_next_claimed = prover.evaluate_rho_at_shifted_point(r_s_star, r_x_star);
pending_shift_claims.push(ShiftClaim {
    constraint_idx: i,
    source_poly: VirtualPolynomial::PackedGtExpRho(i),
    shift_type: ShiftType::StepPlusOne,
    point: (r_s_star, r_x_star),
    claimed_value: rho_next_claimed,
});

// Stage 1b: Shift sumcheck verifies the claim
shift_sumcheck_prover.prove(pending_shift_claims, accumulator, transcript);

// After verification, add to accumulator as virtual polynomial
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRhoNext(i),
    SumcheckId::ShiftSumcheck,  // Different sumcheck ID!
    (r_s_star, r_x_star),
    rho_next_claimed,  // Now verified
);
```

### Key Design: Virtualization Pattern

The shift sumcheck transforms `rho_next` from a **committed polynomial** to a **virtual polynomial**:

- **Committed polynomial**: Requires commitment and opening proof in Stage 4
- **Virtual polynomial**: Verified algebraically through sumcheck, no commitment needed

This leverages the existing virtual polynomial infrastructure:

```rust
enum VirtualPolynomial {
    // Original virtual polynomials from constraint sumchecks
    PackedGtExpRho(usize),
    PackedGtExpQuotient(usize),

    // Virtual polynomial verified through shift sumcheck
    PackedGtExpRhoNext(usize),  // Same variant, different verification path!
}

enum SumcheckId {
    PackedGtExp,      // For constraint sumcheck
    ShiftSumcheck,    // For shift verification
}
```

### Stage 2 Integration - No Changes Required

Stage 2's direct evaluation protocol treats all virtual claims uniformly:

```rust
// Stage 2 doesn't care about the verification method
let (point, value) = accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::PackedGtExpRhoNext(i),
    SumcheckId::ShiftSumcheck,  // Just uses different sumcheck ID
);

// Computes M(r_s, r_x) = Σ_i eq(r_s, i) · v_i exactly as before
```

This is the beauty of the virtualization pattern - downstream stages don't need to know whether a claim was:
- Directly evaluated from a committed polynomial
- Verified through the shift sumcheck relation

### Complete Communication Flow

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1a: Packed GT Constraint Sumcheck                  │
│ - Commits: rho, quotient (NOT rho_next)                  │
│ - To accumulator: rho, quotient virtual claims           │
│ - To Stage 1b: pending_shift_claims                      │
└─────────────────────────┬────────────────────────────────┘
                          │ pending_shift_claims
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1b: Shift Sumcheck                                 │
│ - Input: pending_shift_claims + accumulator reference    │
│ - Verifies: each claim v_i = rho_i(r_s*+1, r_x*)        │
│ - To accumulator: verified rho_next virtual claims       │
└─────────────────────────┬────────────────────────────────┘
                          │ accumulator now complete
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Direct Evaluation                               │
│ - Input: accumulator (unchanged interface!)              │
│ - Process: Combines all virtual claims uniformly         │
│ - Output: M(r_s, r_x) claim                              │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Jagged Transform                                │
│ - No changes needed                                      │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Opening Proof                                   │
│ - Opens fewer polynomials (no rho_next commitments)      │
└─────────────────────────────────────────────────────────┘
```

### Implementation Details

1. **Shift Claim Structure**:
   ```rust
   struct ShiftClaim {
       constraint_idx: usize,
       source_poly: VirtualPolynomial,
       shift_type: ShiftType,
       point: (Vec<F>, Vec<F>),
       claimed_value: F,
   }

   enum ShiftType {
       StepPlusOne,  // For rho_next(s,x) = rho(s+1,x)
       // Could extend for other shift patterns
   }
   ```

2. **Batching**: Multiple shift sum-checks (if multiple GT exps) can be batched:
   ```rust
   // Batch coefficient from transcript
   let gamma = transcript.challenge();

   // Prove: Σ_i γ^i × (v_i - Σ_{s,x} EqPlusOne(r_s*_i, s) × Eq(r_x*_i, x) × rho_i(s,x)) = 0
   ```

3. **Verifier Adjustment**:
   ```rust
   // Verify Stage 1a (expects one less polynomial)
   let stage1a_claims = verify_packed_gt_sumcheck(proof.stage1a, transcript)?;

   // Verify Stage 1b shift sumcheck
   let shift_claims = verify_shift_sumcheck(
       proof.stage1b,
       &stage1a_claims.pending_shifts,
       transcript
   )?;

   // Stage 2+ proceed unchanged
   ```

4. **Accumulator Extension** (Optional for debugging):
   ```rust
   impl OpeningAccumulator {
       // Track shift relationships for verification
       shift_relations: Vec<ShiftRelation>,

       fn append_shift_verified(
           &mut self,
           poly: VirtualPolynomial,
           sumcheck_id: SumcheckId,
           point: (Vec<F>, Vec<F>),
           value: F,
           source: VirtualPolynomial,
           shift_type: ShiftType,
       ) {
           self.append_virtual(poly, sumcheck_id, point, value);
           self.shift_relations.push(ShiftRelation {
               verified_poly: poly,
               source_poly: source,
               shift_type,
           });
       }
   }
   ```

## The Virtualization Pattern

### Understanding Virtual vs Committed Polynomials

The key insight enabling this optimization is the distinction between **committed** and **virtual** polynomials in the recursion SNARK:

**Committed Polynomials**:
- Have explicit commitments computed by the prover
- Require opening proofs in Stage 4 (Hyrax)
- Add to proof size and commitment computation

**Virtual Polynomials**:
- Exist only as claimed evaluations at specific points
- Verified through algebraic relations (sumchecks)
- No commitment overhead

The shift sumcheck transforms `rho_next` from committed to virtual by proving the algebraic relation `rho_next(s,x) = rho(s+1,x)`.

### Why This Works with Existing Infrastructure

The recursion SNARK's `OpeningAccumulator` already handles both types uniformly:

```rust
// Both committed and virtual polynomials stored in same structure
virtual_openings: HashMap<(VirtualPolynomial, SumcheckId), (Point, Value)>
```

Stage 2 (Direct Evaluation) processes all virtual claims identically:
```rust
// Doesn't distinguish between verification methods
M(r_s, r_x) = Σ_i eq(r_s, i) × v_i
```

This means we can change a polynomial from committed to virtual without affecting downstream stages!

### Matrix Organization Impact

With the optimization, the matrix row count changes:

**Before**: PackedGtExpRho, PackedGtExpRhoNext, PackedGtExpQuotient (3 rows per GT exp)
**After**: PackedGtExpRho, PackedGtExpQuotient (2 rows per GT exp) + virtual rho_next

The virtual rho_next claims still participate in Stage 2's matrix computation, but don't require commitment storage or opening proofs.

## Performance Analysis

### Costs
- **Additional Rounds**: +12 sum-check rounds
- **Verifier Work**: +O(24) field operations per round
- **Proof Size**: +12 field elements (one per round)

### Savings
- **Commitment Reduction**: -1 polynomial commitment per GT exp
- **Opening Proof**: -1 opening proof in Stage 4
- **Proof Size**: Approximately -160 bytes per GT exp
- **Memory**: No need to store rho_next polynomial (saves ~10MB for 256-bit exp)

### Net Benefit
For applications with many GT exponentiations:
- Break-even: When commitment + opening cost > 12 rounds of sum-check
- Significant benefit: When batching multiple GT exps (amortized shift sum-check cost)

## Implementation Clarifications

### Design Decisions

1. **Constraint Evaluation During Sumcheck**:
   - Build `rho_next` array during witness generation (as currently done)
   - Do NOT add it to the constraint system or commit to it
   - Use the precomputed array for efficient constraint evaluation during sumcheck
   - This maintains performance while eliminating the commitment

2. **Shift Sumcheck Integration**:
   - Implement as separate `ShiftRhoProver/ShiftRhoVerifier` structs
   - Place in new file `shift_rho.rs` following existing patterns
   - Clear separation of concerns from constraint sumcheck

3. **Batching Multiple GT Exps**:
   - Run ONE batched shift sumcheck after ALL GT exp constraint sumchecks complete
   - Use batching coefficient γ to combine all rho_next claims
   - More efficient than individual shift sumchecks

4. **Matrix Row Indexing**:
   - Completely remove `PackedGtExpRhoNext` from polynomial type enum
   - Renumber subsequent types (no backwards compatibility needed)
   - This is strictly better - no need to maintain old indexing

5. **Stage 3 Jagged Transform**:
   - Dense polynomial includes only committed polynomials
   - Virtual rho_next claims do not participate in jagged transform
   - They only exist as point evaluations verified through sumcheck

### No Backwards Compatibility

Since this optimization is strictly better:
- Remove all `rho_next` commitment code paths
- Update all tests to use new flow
- No feature flags or compatibility modes needed

## Conclusion

The shift sum-check optimization provides a concrete way to reduce commitment overhead in the packed GT exponentiation protocol. By adding a specialized sum-check between Stage 1a and Stage 2, we can eliminate the redundant `rho_next` commitment while maintaining the same security guarantees.

The technique integrates cleanly with the existing recursion SNARK architecture, requiring no changes to Stage 2 or beyond. The optimization is particularly beneficial when multiple GT exponentiations are performed, as the shift sum-checks can be batched together.

This approach demonstrates a general principle: redundant committed data that has a known algebraic relationship to other committed data can often be replaced with a proof of that relationship, trading computation for reduced proof size.