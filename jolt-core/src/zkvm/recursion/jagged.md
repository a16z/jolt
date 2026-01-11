# Jagged Polynomial Commitments - Row-wise Perspective

## Overview

This document describes the jagged polynomial commitment scheme from a row-wise perspective (transposed from the column-wise description in the original paper).

## 1. Jagged Definition and Bijection

### Row-wise Jagged Definition

A **row-wise jagged function** is a function `p : {0,1}^k × {0,1}^n → F` where:
- Each row `x ∈ {0,1}^k` has a particular width `w_x ∈ [0, ..., 2^n - 1]`
- `p(x, y) = 0` for every `y ≥ w_x`

In this row-wise perspective:
- `k` bits index the rows (instead of columns)
- `n` bits index the columns within each row (instead of rows within columns)
- `w_x` represents the width of row `x` (instead of height of columns)

### Bijection Between Sparse and Dense Representations

Given a row-wise jagged function with widths `(w_x)_{x∈{0,1}^k}`, we construct a bijection between the dense and sparse representations:

1. **Cumulative widths**: Define `t_x = ∑_{x'<x} w_{x'}` for each row `x`
   - `t_x` represents the cumulative width up to (but not including) row `x`
   - The total number of non-zero entries is `M = ∑_x w_x`
   - Assume `M = 2^m` for some integer `m`

2. **Bijection functions**: For index `i ∈ {0,1}^m`, define:
   - `row_t(i)`: returns the row `x` such that `t_x ≤ i < t_{x+1}`
   - `col_t(i)`: returns the column position within the row as `i - t_{row_t(i)}`

3. **Dense representation**: Define `q : {0,1}^m → F` as:
   ```
   q(i) = p(row_t(i), col_t(i))
   ```

This creates a bijection between:
- Dense indices `i ∈ {0,1}^m`
- Non-zero positions `(x, y)` where `x ∈ {0,1}^k`, `y ∈ {0,1}^n`, and `y < w_x`

## 2. Evaluation Expression

### Multilinear Extension Evaluation

For the multilinear extensions `p̂` and `q̂`, given evaluation point `(z_row, z_col) ∈ F^k × F^n`:

```
p̂(z_row, z_col) = ∑_{i∈{0,1}^m} q(i) · eq(row_t(i), z_row) · eq(col_t(i), z_col)
```

Where:
- `eq(a, b) = ∏_j eq_1(a_j, b_j)` is the equality polynomial
- `eq_1(a, b) = a·b + (1-a)·(1-b)`

### Reduction to Dense Evaluation

The jagged PCS protocol reduces claims about the sparse polynomial `p̂` to claims about the dense polynomial `q̂`:

1. **Claim**: `p̂(z_row, z_col) = v`
2. **Reduction**: Run sumcheck on:
   ```
   v = ∑_{i∈{0,1}^m} q̂(i) · f̂_t(z_row, z_col, i)
   ```
   where `f̂_t(z_row, z_col, i) = eq(row_t(i), z_row) · eq(col_t(i), z_col)`

3. **Output**: A claim `q̂(i*) = v*` for some `i* ∈ F^m` and `v* ∈ F`

### Efficient Computation of f̂_t

The function `f̂_t` can be computed efficiently because:
- It can be expressed using a width-4 read-once branching program
- The multilinear extension of such functions can be evaluated in `O(m)` time
- For row-wise jagged with `2^k` rows, the verifier complexity is `O(m · 2^k)`

### Critical Verification Detail: Summing Over All Possible Rows

**Important**: Based on Claim 3.2.1 from the jagged polynomial paper, the verifier must sum over **ALL** possible k-bit strings, not just existing rows/polynomials. The correct formula is:

```
f̂_t(z_row, z_col, i) = ∑_{y∈{0,1}^k} eq(z_col, y) · ĝ(z_row, i, t_{y-1}, t_y)
```

Where:
- The sum is over **all** `2^k` possible row indices `y ∈ {0,1}^k`
- This includes rows with zero width (`w_y = 0`)
- For rows with `w_y = 0`, we have `t_y = t_{y-1}` (cumulative width doesn't increase)

**Common Implementation Error**: It is incorrect to iterate only over existing polynomials or non-empty rows. The mathematics requires summing over the entire domain `{0,1}^k` to maintain the algebraic properties of the multilinear extension.

**Performance Implication**: The `O(m · 2^k)` verifier complexity comes from this requirement to iterate over all `2^k` rows, even though many may be empty.

## Key Properties

1. **Prover efficiency**: `O(M)` field operations for commitment and evaluation proofs
2. **Verifier efficiency**: `O(m · 2^k)` arithmetic operations (can be reduced using batching techniques)
3. **No additional commitments**: Unlike general sparse PCS schemes, only the dense polynomial `q` needs to be committed
4. **Arithmetic circuit verifier**: The verifier can be implemented as a pure arithmetic circuit depending only on `m`

## Implementation Complexity: Mapping Between Polynomial and Matrix Indices

### The Challenge

In our implementation, we have an additional layer of indirection that is not present in the theoretical description above. The bijection functions `row_t(i)` and `col_t(i)` return **polynomial indices**, not actual matrix row indices. This is because:

1. **Multiple polynomial types per constraint**: Each constraint in our system contributes multiple polynomials (e.g., GT exponentiation contributes 4 polynomials: base, rho_prev, rho_curr, quotient).

2. **Jaggedness at polynomial level**: Different polynomial types have different widths (4-var vs 8-var MLEs), but they all belong to the same constraint.

3. **Matrix organization**: The constraint matrix organizes rows by polynomial type and constraint index, not by a simple linear ordering.

### The Mapping Architecture

Our implementation uses a three-level mapping:

```
Dense Index → (Polynomial Index, Evaluation Index) → (Constraint Index, Polynomial Type) → Matrix Row
```

1. **Bijection Level** (`VarCountJaggedBijection`):
   - Maps dense index `i` to polynomial index and evaluation index
   - `row(i)` returns the polynomial index (not the matrix row!)
   - `col(i)` returns the evaluation index within that polynomial

2. **Mapping Level** (`ConstraintSystemJaggedMapping`):
   - Decodes polynomial index to constraint index and polynomial type
   - `decode(poly_idx)` returns `(constraint_idx, poly_type)`

3. **Matrix Level** (`DoryMultilinearMatrix`):
   - Converts constraint index and polynomial type to actual matrix row
   - `row_index(poly_type, constraint_idx)` returns the matrix row index

### Implementation Requirements

For the jagged sumcheck to work correctly:

1. **Prover**: Must have access to the mapping to convert polynomial indices to matrix rows when computing `eq(row_t(i), z_row)`.

2. **Verifier**: Also needs the mapping for the same conversion when verifying the sumcheck relation.

3. **Precomputation**: To avoid passing complex data structures, we precompute all matrix row indices:
   ```rust
   let mut matrix_rows = Vec::with_capacity(num_polynomials);
   for poly_idx in 0..num_polynomials {
       let (constraint_idx, poly_type) = mapping.decode(poly_idx);
       let matrix_row = matrix.row_index(poly_type, constraint_idx);
       matrix_rows.push(matrix_row);
   }
   ```

### Why This Complexity?

This design allows us to:
- Handle different polynomial widths (4-var vs 8-var) efficiently
- Group related polynomials by constraint for better cache locality
- Maintain a clean separation between the bijection logic and the constraint system structure

The theoretical jagged polynomial scheme assumes a simple row-wise sparse matrix, but real constraint systems have more structure that requires this additional mapping layer.

## Correct Verification Implementation

### The Bug

Our current implementation **incorrectly** iterates only over existing polynomials:

```rust
// WRONG: Only iterates over existing polynomials
for poly_idx in 0..self.bijection.num_polynomials() {
    let t_prev = self.bijection.cumulative_size_before(poly_idx);
    let t_curr = self.bijection.cumulative_size(poly_idx);
    // ... compute g_mle and accumulate ...
}
```

### The Fix

According to Claim 3.2.1, we must iterate over **all** possible row indices:

```rust
// CORRECT: Iterates over all 2^k possible rows
let num_rows = 1 << self.params.num_s_vars; // 2^k rows
for row_idx in 0..num_rows {
    // Need to map row_idx to cumulative sizes
    // For non-existent rows, t_prev == t_curr
    let t_prev = get_cumulative_size_before(row_idx);
    let t_curr = get_cumulative_size_at(row_idx);
    // ... compute g_mle and accumulate ...
}
```

### Implementation Strategy

To fix this efficiently:

1. **Precompute cumulative sizes for all 2^k rows**: Most will have zero width, so `t_y = t_{y-1}`.

2. **Sparse representation**: Store only non-zero widths and compute cumulative sizes on demand.

3. **Equality polynomial optimization**: Since we're summing over all rows, we can use the fact that `∑_{y∈{0,1}^k} eq(z_col, y) = 1` for certain optimizations.

### Why This Matters

The incorrect implementation breaks the algebraic properties of the multilinear extension. By only summing over existing polynomials, we're effectively evaluating a different polynomial than what the prover committed to. This is why verification takes ~6 seconds instead of milliseconds - we're computing the wrong thing entirely.

## Proposed Solution: Stage 4 for Batch Verification

Instead of trying to fix Stage 3 to sum over all 2^k rows directly, we propose adding a new Stage 4 that uses the "Jagged Assist" protocol from the paper (Section 5 / Theorem 1.5).

### Architecture Change

Current (broken):
```
Stage 1 → Stage 2 → Stage 3 (tries to sum over 2^k rows) → PCS
                         ↑
                    [BOTTLENECK: wrong sum]
```

Proposed:
```
Stage 1 → Stage 2 → Stage 3 → Stage 4 → PCS
                      ↓          ↓
                   [claim]   [batch verify]
```

### How Stage 4 Works

#### The Problem
Stage 3 needs to compute:
```
Σ_{y∈{0,1}^k} eq(zc, y) · ĝ(zr, i, t_{y-1}, t_y)
```
This requires 2^k evaluations of the branching program ĝ.

#### The Solution: Batch Verification Protocol

1. **Stage 3 (modified)** outputs:
   - A claim: `Σ_{y∈{0,1}^k} eq(zc, y) · ĝ(zr, i, t_{y-1}, t_y) = v`
   - The 2^k individual values: `{ĝ(zr, i, t_{y-1}, t_y)}_{y∈{0,1}^k}`

2. **Stage 4** verifies all 2^k evaluations using batch verification:
   - Verifier chooses random coefficients r₁, ..., r_{2^k}
   - Reduces to single claim via sumcheck
   - Verifier only evaluates ĝ ONCE at a random point
   - Total verifier work: O(m·2^k) arithmetic operations

### Implementation Plan

#### Stage 4 Verifier
```rust
fn verify_stage4(
    proof: &Stage4Proof,
    transcript: &mut T,
    accumulator: &mut VerifierOpeningAccumulator,
    zr: &[F],      // from Stage 1
    zc: &[F],      // from Stage 2
    i: &[F],       // from Stage 3
    claimed_sum: F, // from Stage 3
) -> Result<Vec<F>, Box<dyn Error>> {
    let K = 1 << num_s_vars; // 2^k

    // 1. Verify the sum matches claimed values
    let computed_sum: F = (0..K)
        .map(|y| eq(zc, y) * proof.g_evaluations[y])
        .sum();
    assert_eq!(computed_sum, claimed_sum);

    // 2. Setup batch verification for K evaluations
    let points: Vec<_> = (0..K)
        .map(|y| (zr, i, cumulative_sizes[y], cumulative_sizes[y+1]))
        .collect();

    // 3. Run batch MLE verification (reduces to 1 evaluation)
    BatchedSumcheck::verify(
        &proof.batch_proof,
        &points,
        &proof.g_evaluations,
        accumulator,
        transcript
    )
}
```

#### Stage 4 Prover
```rust
fn prove_stage4(
    transcript: &mut T,
    branching_program: &JaggedBranchingProgram,
    r_stage1: &[F],
    r_stage3: &[F],
) -> Stage4Proof {
    let K = 1 << num_s_vars;

    // 1. Compute all 2^k evaluations
    let g_evaluations: Vec<F> = (0..K)
        .map(|y| {
            evaluate_branching_program(
                branching_program,
                r_stage1,
                r_stage3,
                cumulative_sizes[y],
                cumulative_sizes[y+1]
            )
        })
        .collect();

    // 2. Create batch proof
    let batch_proof = BatchedSumcheck::prove(...);

    Stage4Proof { g_evaluations, batch_proof }
}
```

### Benefits

1. **Clean separation**: Stage 3 handles jagged transform, Stage 4 handles batch verification
2. **No major refactoring**: Stage 3 keeps most current structure
3. **Correct mathematics**: We properly sum over all 2^k rows
4. **Efficient verification**: O(m·2^k) work with only ONE branching program evaluation
5. **Better debugging**: Stage 4 can be tested independently

### Preprocessing Requirements

Need to precompute and store:
- Cumulative sizes for all 2^k possible rows (most will have size 0)
- This is a vector of size 2^k + 1

### TODO

1. Implement `compute_all_cumulative_sizes()` that handles non-existent rows
2. Create Stage4Proof structure
3. Implement Stage 4 prover and verifier
4. Modify Stage 3 to output claims instead of computing the sum
5. Update RecursionProof to include Stage 4
6. Add Stage 4 to the verification pipeline