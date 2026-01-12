# Homomorphic Offloading for `combine_commitments`

## Overview

This document describes a proposed optimization to offload the expensive GT operations in `combine_commitments` to the recursion SNARK, following the same pattern used for Dory verifier operations like GT exponentiation and multiplication.

## Current Implementation

The `combine_commitments` function in Dory performs homomorphic combining of GT commitments:

```rust
// /jolt-core/src/poly/commitment/dory/commitment_scheme.rs:224-239
fn combine_commitments<C: Borrow<Self::Commitment>>(
    commitments: &[C],
    coeffs: &[Self::Field],
) -> Self::Commitment {
    coeffs
        .par_iter()
        .zip(commitments)
        .map(|(coeff, commitment)| {
            ark_coeff * **commitment  // GT scalar multiplication (exponentiation)
        })
        .reduce(ArkGT::identity, |a, b| a + b)  // GT addition (field multiplication)
}
```

This is used in both prover and verifier during Stage 8 (Dory batch opening) to compute a joint commitment for polynomial batching.

## Why We Want Homomorphic Offloading

### Performance Impact

For n polynomial commitments being combined:
- **n GT exponentiations**: Each `coeff * commitment` is an expensive GT scalar multiplication
- **(n-1) GT multiplications**: The reduction combines results via GT group operations

These operations are among the most expensive in the verifier, involving:
- Operations in the 12-degree field extension Fq^12
- Complex modular arithmetic
- No native hardware acceleration

### Current Recursion Success

The recursion SNARK already achieves ~150× speedup by offloading:
- GT exponentiations (square-and-multiply)
- GT multiplications
- G1 scalar multiplications

Adding `combine_commitments` offloading would extend these benefits to the polynomial batching phase.

## Implementation Approach

### 1. Witness Structure

Create a new witness type for homomorphic combining:

```rust
/// Witness for homomorphic combination of GT commitments
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTCombineWitness {
    /// Input commitments (GT elements)
    pub commitments: Vec<Fq12>,

    /// Scalar coefficients
    pub coefficients: Vec<Fr>,

    /// Witnesses for exponentiations: result_i = coeff_i^commitment_i
    pub exp_witnesses: Vec<GTExpWitness>,

    /// Tree reduction witnesses for combining results
    pub tree_levels: Vec<TreeLevel>,

    /// Final combined result
    pub result: Fq12,
}

#[derive(Clone, Debug)]
pub struct TreeLevel {
    /// GT multiplication witnesses for this tree level
    pub multiplications: Vec<GTMulWitness>,
}
```

### 2. Tree Reduction for Efficiency

Implement tree reduction for the GT multiplications to minimize depth:

```rust
impl GTCombineWitness {
    pub fn generate(
        commitments: Vec<Fq12>,
        coefficients: Vec<Fr>
    ) -> Self {
        // Step 1: Parallel GT exponentiations
        let mut exp_results = vec![];
        let exp_witnesses: Vec<GTExpWitness> = commitments
            .par_iter()
            .zip(&coefficients)
            .map(|(comm, coeff)| {
                let witness = generate_gt_exp_witness(comm, coeff);
                exp_results.push(witness.result);
                witness
            })
            .collect();

        // Step 2: Tree reduction for GT multiplications
        let tree_levels = generate_tree_reduction(exp_results);
        let result = tree_levels.last().unwrap()
            .multiplications[0].result;

        Self {
            commitments,
            coefficients,
            exp_witnesses,
            tree_levels,
            result,
        }
    }
}

fn generate_tree_reduction(mut elements: Vec<Fq12>) -> Vec<TreeLevel> {
    let mut levels = vec![];

    while elements.len() > 1 {
        let mut next_level = vec![];
        let mut level_muls = vec![];

        // Process pairs in parallel
        for chunk in elements.chunks(2) {
            match chunk.len() {
                2 => {
                    let witness = generate_gt_mul_witness(&chunk[0], &chunk[1]);
                    next_level.push(witness.result);
                    level_muls.push(witness);
                }
                1 => next_level.push(chunk[0]),
                _ => unreachable!()
            }
        }

        levels.push(TreeLevel { multiplications: level_muls });
        elements = next_level;
    }

    levels
}
```

### 3. Integration with Existing Constraints

The beauty of this approach is that it reuses existing constraint types:

```rust
enum ConstraintType {
    GtExp { bit: bool },  // For coeff^commitment operations
    GtMul,                // For tree reduction multiplications
    // ... existing types ...
}
```

No new constraint sumcheck implementations needed!

### 4. Modified Prover Flow

```rust
// In Stage 8 batch opening
impl<F: JoltField, PCS: DoryCommitmentScheme> Prover<F, PCS> {
    fn prove_stage8_with_offloading(&mut self) {
        // Compute RLC coefficients
        let (coeffs, commitments) = /* ... */;

        // Generate witness instead of computing directly
        let combine_witness = GTCombineWitness::generate(
            commitments,
            coeffs
        );

        // Use precomputed result
        let joint_commitment = combine_witness.result;

        // Store witness for recursion prover
        self.gt_combine_witnesses.push(combine_witness);

        // Continue with opening proof...
    }
}
```

### 5. Modified Verifier Flow

```rust
impl<F: JoltField, PCS: DoryCommitmentScheme> Verifier<F, PCS> {
    fn verify_stage8_with_hints(&mut self, hints: &HintMap) {
        // Instead of expensive computation:
        // let joint = PCS::combine_commitments(&commitments, &coeffs);

        // Use precomputed hint
        let joint_commitment = hints.get_combine_commitment_hint();

        // The recursion SNARK will prove this was computed correctly
    }
}
```

### 6. Constraint Count Analysis

For n polynomial commitments:
- **n GT exponentiation constraints** (one per coefficient-commitment pair)
- **(n-1) GT multiplication constraints** (tree reduction)
- **Total**: 2n-1 constraints

With typical batches of 10-50 polynomials, this is very manageable.

## Benefits

1. **Verification Speedup**: Eliminate expensive GT operations from the critical path
2. **Reuse Infrastructure**: Leverages existing GT exp/mul constraint implementations
3. **Parallelization**: Tree reduction enables parallel witness generation
4. **Scalability**: Cost grows logarithmically with tree depth, not linearly

## Implementation Steps

1. **Add witness types** to `witness.rs`
2. **Extend `DoryRecursionWitness`** with `gt_combine_witness` field
3. **Modify Stage 8** in prover.rs to generate witnesses
4. **Update verifier** to use hints
5. **Extend `RecursionProver`** to include combine constraints
6. **Add tests** verifying correctness of offloaded computation

## Testing Strategy

```rust
#[test]
fn test_combine_commitments_offloading() {
    // 1. Generate random commitments and coefficients
    let commitments = /* random GT elements */;
    let coeffs = /* random scalars */;

    // 2. Compute directly
    let direct_result = DoryCommitmentScheme::combine_commitments(
        &commitments,
        &coeffs
    );

    // 3. Compute via witness generation
    let witness = GTCombineWitness::generate(commitments, coeffs);

    // 4. Verify result matches
    assert_eq!(direct_result, witness.result);

    // 5. Verify all constraints are satisfied
    verify_exp_constraints(&witness.exp_witnesses);
    verify_tree_constraints(&witness.tree_levels);
}
```

## Future Optimizations

1. **Batched Combining**: If multiple `combine_commitments` calls occur, batch them together
2. **Specialized Tree Structure**: Optimize for common sizes (e.g., binary tree for power-of-2 sizes)
3. **Streaming Witnesses**: Generate witnesses on-the-fly during polynomial evaluation
4. **Depth vs Width Tradeoffs**: Analyze optimal tree shapes for different batch sizes

## Conclusion

Homomorphic offloading of `combine_commitments` is a natural extension of the existing recursion framework that would provide significant verification speedup with minimal implementation complexity. By reusing the existing GT exp/mul constraints and following the established pattern, we can achieve the same ~150× speedup benefits for this critical operation.