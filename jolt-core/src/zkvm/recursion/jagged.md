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