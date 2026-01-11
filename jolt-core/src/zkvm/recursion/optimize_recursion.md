# Recursion Performance Optimization Plan

Based on trace analysis of fibonacci_18.json, this document analyzes each major recursion span, identifies unexpected performance bottlenecks, and proposes optimization strategies.

## Performance Summary

Total recursion time: ~47.4 seconds
- Prover-side operations: ~15.3s
- Verifier-side operations: ~9.2s
- Hyrax commitment: ~4.9s
- Other overhead: ~18s

## Span-by-Span Analysis

### 1. verify_recursion_stage3 (8,909 ms) ❌ UNEXPECTED

**What it does**: Verifies the Stage 3 jagged transform sumcheck, which proves the bijection between the sparse "jagged" constraint matrix and its dense representation.

**Should be non-negligible?**: Yes, but not THIS much. The prover side takes only 2,205 ms. Verification should be faster than proving.

**Deep dive - Why is this so slow?**

The Stage 3 verifier must:
1. Compute expected claims using the jagged bijection
2. Evaluate polynomials at sumcheck challenge points
3. Verify the sumcheck proof

The 4x slowdown suggests the verifier is doing expensive recomputation that the prover avoids.

**Specific optimization plan**:
1. **Profile the hot path**:
   ```rust
   // In RecursionVerifier::verify_stage3
   // Add fine-grained tracing to identify which part is slow
   ```

2. **Cache bijection lookups**:
   ```rust
   // Current: Recomputes bijection on every polynomial access
   let (poly_idx, eval_idx) = bijection.sparse_to_dense(row, col);

   // Better: Pre-compute and cache the mapping
   let cached_mapping = bijection.precompute_stage3_mapping();
   ```

3. **Optimize polynomial evaluation**:
   - The verifier might be evaluating the full sparse matrix instead of just the dense representation
   - Ensure we're using the dense polynomial, not reconstructing sparse

4. **Memory access pattern**:
   - The jagged bijection causes cache-unfriendly access patterns
   - Consider reorganizing data layout for better cache utilization

5. **Algorithmic fix**:
   - The verifier might be checking more constraints than necessary
   - Review if we're verifying both the forward and backward bijection when only one is needed

### 2. hyrax_commit (4,874 ms) ✓ EXPECTED

**What it does**: Commits to the dense polynomial after the jagged transform using the Hyrax polynomial commitment scheme.

**Should be non-negligible?**: Yes. Polynomial commitment is inherently expensive, especially for large polynomials.

**Optimization plan**:
- Investigate if we can reduce the polynomial size through better constraint packing
- Consider parallelizing the commitment computation
- Explore alternative commitment schemes (though this is a major change)

### 3. create_recursion_prover (2,618 ms) ❌ UNEXPECTED

**What it does**: Creates RecursionProver from pre-generated Dory witnesses, including:
- Converting witness formats
- Building constraint system
- Setting up polynomial representations

**Should be non-negligible?**: No. This is just data structure initialization.

**Optimization plan**:
- **Profile witness conversion**: The unsafe transmutes suggest format mismatches
- Eliminate the witness format conversion by unifying Dory and Jolt witness types
- Pre-allocate memory for constraint system instead of growing vectors
- Consider lazy initialization of polynomial representations
- Remove the `witnesses_to_dory_recursion` conversion step entirely

### 4. recursion_stage11_3_jagged (2,205 ms) ✓ EXPECTED

**What it does**: Prover-side Stage 3 sumcheck for the jagged transform.

**Should be non-negligible?**: Yes. This is a sumcheck protocol with complex polynomial evaluations.

**Optimization plan**:
- Already reasonably optimized
- Minor gains possible through better vectorization
- Ensure we're using parallel evaluation where possible

### 5. recursion_stage11_2_virtualization (2,108 ms) ✓ EXPECTED

**What it does**: Prover-side Stage 2 sumcheck for virtualization of the constraint matrix.

**Should be non-negligible?**: Yes. Another sumcheck protocol.

**Optimization plan**:
- Performance is reasonable
- Check if constraint evaluation can be optimized
- Verify parallel execution is utilized

### 6. extract_constraint_metadata (1,753 ms) ❌ UNEXPECTED

**What it does**: Extracts metadata from the constraint system for the verifier, including:
- Constraint types
- Jagged bijection information
- Matrix row indices

**Should be non-negligible?**: No. This is just data extraction and reorganization.

**Optimization plan**:
- **Move to preprocessing**: This metadata is static for a given constraint system
- Cache the metadata instead of recomputing
- The `RecursionMetadataBuilder` is doing too much work:
  ```rust
  // Current: Rebuilds everything
  RecursionMetadataBuilder::from_constraint_system(constraint_system.clone())

  // Better: Extract pre-computed metadata
  constraint_system.get_cached_metadata()
  ```
- Remove the expensive `clone()` of the entire constraint system

### 7. recursion_stage11_1_constraints (1,567 ms) ✓ EXPECTED

**What it does**: Prover-side Stage 1 sumchecks for GT exp, GT mul, and G1 scalar mul constraints.

**Should be non-negligible?**: Yes. Multiple sumcheck protocols running.

**Optimization plan**:
- Performance is acceptable
- Minor optimizations in constraint evaluation possible
- Ensure SIMD operations are used where applicable

### 8. build_dense_polynomial (62 ms) ✓ REASONABLE

**What it does**: Builds the dense polynomial from the jagged constraint matrix.

**Should be non-negligible?**: Somewhat. Involves data transformation.

**Optimization plan**:
- Current performance is acceptable
- Could benefit from better memory layout

### 9. verify_recursion_stage1 & stage2 (61 ms, 54 ms) ✓ REASONABLE

**What it does**: Verifies Stage 1 and Stage 2 sumchecks.

**Should be non-negligible?**: Yes, but these are appropriately fast.

**Optimization plan**:
- No optimization needed
- Good examples of expected verifier performance

## Priority Optimization Targets

### 1. **Stage 3 Verification** (Potential savings: 6-7 seconds)
- Fix the 4x slowdown compared to prover
- Move bijection computation to preprocessing
- Cache polynomial evaluation points

### 2. **Witness Conversion** (Potential savings: 2+ seconds)
- Eliminate format conversions
- Direct Dory → RecursionProver pipeline
- Remove unnecessary cloning and transmutes

### 3. **Metadata Extraction** (Potential savings: 1.5 seconds)
- Move to preprocessing phase
- Cache static metadata
- Remove constraint system cloning

## Implementation Recommendations

### Short Term (1-2 days)
1. Profile `verify_recursion_stage3` to identify the bottleneck
2. Remove the `constraint_system.clone()` in metadata extraction
3. Add caching for static bijection data

### Medium Term (3-5 days)
1. Refactor witness conversion to eliminate format mismatches
2. Move metadata extraction to preprocessing
3. Optimize Stage 3 verification algorithm

### Long Term (1+ week)
1. Unify Dory and Jolt witness formats
2. Investigate alternative polynomial commitment schemes
3. Consider GPU acceleration for polynomial operations

## Additional Observations

### Repeated Function Calls
The trace shows multiple calls to:
- `RecursionVirtualization::compute_message` (6 calls totaling ~7.6 ms)
- `RecursionVirtualization::ingest_challenge` (8 calls totaling ~7.8 ms)

While individually small, these suggest the sumcheck rounds are properly instrumented and the overhead is minimal.

### Negligible Operations (Working as Expected)
These operations take < 1ms and are properly optimized:
- `sample_recursion_challenges` (0.011 ms)
- `extract_opening_claims` (0.010 ms)
- `init_opening_accumulator` (0.008 ms)
- `create_polynomial_map` (0.008 ms)
- `convert_witness_collection` (0.007 ms)

## Expected Impact

With these optimizations, we could reduce recursion overhead from ~47 seconds to ~25-30 seconds:
- Stage 3 verification: 8.9s → 2.5s
- Witness conversion: 2.6s → 0.5s
- Metadata extraction: 1.8s → 0.1s
- Total savings: ~13 seconds (27% improvement)

## Implementation Status

### Completed Optimizations

1. **Stage 3 Verification** (✅ DONE)
   - Precomputed equality polynomial values in JaggedSumcheckVerifier
   - Added caching for branching program and Point objects
   - Eliminated redundant sparse-to-dense conversions in hot path

2. **Witness Conversion** (✅ DONE)
   - Added preallocation for witness vectors
   - Eliminated unnecessary cloning operations
   - Optimized vector capacity calculations

3. **Metadata Extraction** (✅ DONE)
   - Modified RecursionMetadataBuilder to accept reference instead of ownership
   - Eliminated expensive constraint_system.clone() call
   - No functional changes, just removed unnecessary copying

The sumcheck and commitment operations would remain as the dominant costs, which is architecturally correct.

## Validation Strategy

After implementing optimizations:
1. Run the same fibonacci_18 benchmark
2. Verify correctness with existing tests
3. Compare trace timings for each optimized span
4. Ensure no regression in other areas
5. Test on larger instances to verify scalability