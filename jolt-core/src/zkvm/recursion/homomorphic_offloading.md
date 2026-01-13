# Homomorphic Offloading for `combine_commitments`

## Overview

Offload the expensive GT operations in `combine_commitments` to the recursion SNARK, reusing existing GT exp/mul constraint infrastructure.

## Current Implementation

```rust
fn combine_commitments(commitments: &[GT], coeffs: &[Fr]) -> GT {
    coeffs.iter()
        .zip(commitments)
        .map(|(coeff, comm)| coeff * comm)  // GT exponentiation
        .reduce(|a, b| a + b)                // GT multiplication
}
```

This performs **n exponentiations** and **n-1 multiplications** in GT (Fq12) - extremely expensive operations.

## Witness Structure

Simple linear reduction - no tree, just accumulate left-to-right:

```rust
/// Witness for combine_commitments
pub struct GTCombineWitness {
    /// Exponentiation witnesses: scaled[i] = coeff[i] * commitment[i]
    pub exp_witnesses: Vec<GTExpWitness>,

    /// Multiplication witnesses for linear fold: acc[i] = acc[i-1] * scaled[i]
    pub mul_witnesses: Vec<GTMulWitness>,
}
```

### Witness Generation

```rust
impl GTCombineWitness {
    pub fn generate(
        commitments: &[Fq12],
        coefficients: &[Fr],
    ) -> (Self, Fq12) {
        assert!(!commitments.is_empty());

        // Step 1: Generate exponentiation witnesses
        let exp_witnesses: Vec<_> = commitments
            .iter()
            .zip(coefficients)
            .map(|(comm, coeff)| GTExpWitness::generate(comm, coeff))
            .collect();

        // Step 2: Linear fold with multiplication witnesses
        let mut mul_witnesses = Vec::with_capacity(exp_witnesses.len() - 1);
        let mut accumulator = exp_witnesses[0].result;

        for exp_wit in &exp_witnesses[1..] {
            let mul_wit = GTMulWitness::generate(&accumulator, &exp_wit.result);
            accumulator = mul_wit.result;
            mul_witnesses.push(mul_wit);
        }

        (Self { exp_witnesses, mul_witnesses }, accumulator)
    }
}
```

## Constraint Count

For n commitments:
- **n** GT exponentiation constraints
- **n-1** GT multiplication constraints
- **Total**: 2n-1 constraints

## Integration

1. Prover generates `GTCombineWitness` during Stage 8
2. Verifier receives result as hint
3. Recursion prover adds exp/mul constraints from witness

No new constraint types needed - reuses existing `GtExp` and `GtMul`.