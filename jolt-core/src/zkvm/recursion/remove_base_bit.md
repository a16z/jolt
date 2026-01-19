# Removing Base and Bit Polynomial Commitments from Packed GT Exp

## Overview

The current packed GT exponentiation sum-check requires the prover to commit to **5 polynomials** per GT exp witness:

| Polynomial | Variables | Size | Description |
|------------|-----------|------|-------------|
| `rho` | 12-var | 4096 | Intermediate square-and-multiply results |
| `rho_next` | 12-var | 4096 | Shifted intermediates (ρ_{s+1}) |
| `quotient` | 12-var | 4096 | Quotient polynomials for Fq12 reduction |
| `bit` | 8-var → 12-var | 4096 | Scalar exponent bits (replicated) |
| `base` | 4-var → 12-var | 4096 | Base element (replicated) |

However, **both `base` and `bit` are derived from public inputs** (the base element and scalar exponent), meaning the verifier can reconstruct and evaluate them directly.

## Why We Can Remove Them

From `spec.md` (Section 2.2):

```
| Role         | Symbol              | Description        |
|--------------|---------------------|--------------------|
| Public Input | a ∈ G_T             | Base (as Fq12)     |
| Public Input | k ∈ F_r             | Exponent           |
| Public Output| b ∈ G_T             | Result b = a^k     |
```

The base `a` and scalar `k` are **public inputs** to the GT exponentiation constraint. The verifier already knows:
- The base Fq12 element → can construct `base(x)` MLE (16 values)
- The scalar exponent → can extract bits and construct `bit(s)` MLE (256 values)

### Precedent: The g Polynomial

We already do this for the irreducible polynomial `g(x)`. Looking at `packed_gt_exp.rs:761-766`:

```rust
// Verifier computes g(r_x*) directly - no commitment needed
let g_eval: F = {
    let g_mle_4var = get_g_mle();  // Hardcoded constant
    let g_poly_fq = MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_4var));
    let g_eval_fq = g_poly_fq.evaluate_dot_product(r_x_star_fq);
    unsafe { std::mem::transmute_copy(&g_eval_fq) }
};
```

The `g` polynomial is a known constant (the irreducible polynomial of Fq12), so the verifier evaluates it directly at the challenge point rather than receiving a claim. The same principle applies to `base` and `bit`.

## How It Would Work

### Current Verifier Flow

```rust
// Verifier receives 5 claims per witness from accumulator
let rho_claim = claims[0];
let rho_next_claim = claims[1];
let quotient_claim = claims[2];
let bit_claim = claims[3];      // ← Received from prover
let base_claim = claims[4];     // ← Received from prover

// Use in constraint check
let base_power = F::one() - bit_claim + bit_claim * base_claim;
let constraint = rho_next_claim - rho_claim * rho_claim * base_power - quotient_claim * g_eval;
```

### Proposed Verifier Flow

```rust
// Verifier receives only 3 claims per witness
let rho_claim = claims[0];
let rho_next_claim = claims[1];
let quotient_claim = claims[2];

// Verifier computes bit(r_s*) directly from public scalar
let bit_eval = evaluate_bit_mle(&scalar_bits, &r_s_star);  // 256-point MLE

// Verifier computes base(r_x*) directly from public base element
let base_eval = evaluate_base_mle(&base_fq12, &r_x_star);  // 16-point MLE

// Use in constraint check (same formula)
let base_power = F::one() - bit_eval + bit_eval * base_eval;
let constraint = rho_next_claim - rho_claim * rho_claim * base_power - quotient_claim * g_eval;
```

### MLE Evaluation Functions

Both evaluations are tiny and fast:

```rust
/// Evaluate bit MLE at challenge point r_s* (8-variable MLE, 256 points)
fn evaluate_bit_mle(scalar_bits: &[bool], r_s_star: &[F]) -> F {
    // bit(s) = scalar_bits[s] for s ∈ {0,1}^8
    // MLE evaluation: Σ_s eq(r_s*, s) · bit_s
    let eq_evals = EqPolynomial::evals(r_s_star);
    scalar_bits.iter()
        .zip(eq_evals.iter())
        .map(|(b, eq)| if *b { *eq } else { F::zero() })
        .sum()
}

/// Evaluate base MLE at challenge point r_x* (4-variable MLE, 16 points)
fn evaluate_base_mle(base_fq12: &Fq12, r_x_star: &[F]) -> F {
    // base(x) = fq12_to_multilinear_evals(base_fq12)[x] for x ∈ {0,1}^4
    let base_mle = fq12_to_multilinear_evals(base_fq12);  // 16 Fq values
    DensePolynomial::new(base_mle).evaluate(r_x_star)
}
```

## Impact on Data Committed

### Per GT Exponentiation

| Component | Current | Proposed | Reduction |
|-----------|---------|----------|-----------|
| Witness polynomials | 5 | 3 | 40% fewer |
| Virtual claims | 5 | 3 | 40% fewer |
| Field elements in witness | 20,480 | 12,288 | 8,192 fewer |

### Prover Changes

1. **Remove `bit_packed` and `base_packed` from `PackedGtExpWitness`**
2. **Remove `bit_polys` and `base_polys` from `PackedGtExpProver`**
3. **Prover still needs these during sumcheck computation** - but computes them locally from public data, doesn't commit
4. **Remove `bit_claims` and `base_claims` from final claim caching**

### Verifier Changes

1. **Verifier receives public inputs**: `(base: Fq12, scalar: Fr)` per GT exp
2. **Compute evaluations directly** in `expected_output_claim()`
3. **Reduce virtual polynomial claims** from 5 to 3

### Constraint System Changes

The `VirtualPolynomial` enum can remove:
- `PackedGtExpBit(usize)`
- `PackedGtExpBase(usize)`

The `PolyType` enum in the constraint system can remove:
- `Bit`
- `Base`

This simplifies Stage 2 virtualization as there are fewer polynomial types to handle.

## Why This Matters for the Larger Protocol

### Stage 2: Virtualization Sum-Check

Stage 2 combines all virtual polynomial claims into a matrix evaluation. Fewer polynomial types means:
- Smaller constraint matrix
- Fewer rows to virtualize
- Simpler indexing logic

### Stage 3: Jagged Transform

The jagged transform maps sparse constraint indices to dense polynomial indices. Removing 2 polynomial types per GT exp:
- Reduces the sparse matrix size
- Speeds up the bijection computation
- Smaller opening proofs

### Stage 4: Opening Proof (Hyrax)

Fewer polynomials = fewer commitment openings = smaller proof size and faster verification.

## Implementation Checklist

1. **`packed_gt_exp.rs`**:
   - [ ] Remove `bit_packed`, `base_packed` from `PackedGtExpWitness`
   - [ ] Remove `bit_polys`, `base_polys` from `PackedGtExpProver`
   - [ ] Update `from_steps()` to not pack bit/base
   - [ ] Update `compute_message()` to compute bit/base values on-the-fly from public data
   - [ ] Remove `bit_claims`, `base_claims`
   - [ ] Update `cache_openings()` to only cache 3 claims
   - [ ] Add `base: Fq12` and `scalar_bits: Vec<bool>` as public inputs to prover
   - [ ] Update `PackedGtExpVerifier` to compute bit/base evaluations directly
   - [ ] Update `expected_output_claim()` to use computed values

2. **`witness.rs`**:
   - [ ] Ensure `GTExpOpWitness` provides access to public `base` and `exponent`

3. **`constraints_sys.rs`**:
   - [ ] Update `PolyType` enum (remove Bit/Base if applicable)
   - [ ] Update matrix construction to not include bit/base rows

4. **Virtual polynomial utilities**:
   - [ ] Remove `PackedGtExpBit` and `PackedGtExpBase` variants
   - [ ] Update claim retrieval logic

5. **Tests**:
   - [ ] Update all tests to reflect new claim count
   - [ ] Verify constraint satisfaction still holds
   - [ ] Benchmark to confirm performance improvement

## Security Considerations

This optimization is **sound** because:

1. **No information leak**: The base and scalar are already public inputs - the verifier knows them
2. **Same constraint**: The constraint equation is unchanged; only the source of `bit` and `base` evaluations changes
3. **Deterministic computation**: Both prover and verifier compute identical MLE evaluations from the same public data
4. **Follows established pattern**: Same approach used for the irreducible polynomial `g(x)`

The prover cannot cheat by providing false bit/base claims because the verifier computes them independently from public inputs.

## Summary

By recognizing that `base` and `bit` polynomials are derived entirely from public inputs, we can eliminate them from the prover's commitments:

- **5 → 3 polynomials** per GT exponentiation
- **40% reduction** in virtual claims
- **8,192 fewer field elements** committed per GT exp
- **Simpler constraint system** for Stages 2-4
- **No security impact** - verifier computes same values from public data

This follows the same pattern already used for the irreducible polynomial `g(x)`, which the verifier evaluates directly via `get_g_mle()` rather than receiving as a claim.
