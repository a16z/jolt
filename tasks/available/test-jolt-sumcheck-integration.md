# test-jolt-sumcheck-integration: Integration tests for jolt-sumcheck

**Scope:** crates/jolt-sumcheck/tests/

**Depends:** impl-jolt-sumcheck, test-jolt-sumcheck

**Verifier:** ./verifiers/scoped.sh /workdir jolt-sumcheck

**Context:**

Write integration tests for the `jolt-sumcheck` crate that verify the sumcheck protocol engine from an external user's perspective.

### Integration Test Files

Create the following test files in `crates/jolt-sumcheck/tests/`:

#### 1. `protocol.rs` - Test complete protocol round-trips

Test end-to-end sumcheck protocol execution:
- Implement simple polynomial witnesses and verify sumcheck
- Test with polynomials of varying degrees and variable counts
- Verify soundness: incorrect claims fail verification
- Test completeness: correct claims always verify

#### 2. `batching.rs` - Test batched sumcheck

Test the batched sumcheck functionality:
- Multiple claims with same number of variables
- Claims with different numbers of variables (front-loaded batching)
- Verify each claim in batch could verify independently
- Test batch sizes from 2 to 10+ claims
- Verify efficiency gains from batching

#### 3. `streaming.rs` - Test streaming variant

Test memory-efficient streaming sumcheck:
- Large polynomial witnesses that stream evaluations
- Verify same result as non-streaming version
- Test memory usage remains bounded
- Test with various chunk sizes

### Implementation Examples

**Custom Test Witness:**
```rust
/// Simple polynomial witness for testing: f(x) = sum of coordinates
struct CoordinateSumWitness {
    num_vars: usize,
    current_vars_bound: Vec<F>,
}

impl<F: Field> SumcheckInstanceProver<F> for CoordinateSumWitness {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        // Compute round polynomial for coordinate sum
        // s_i(X) = sum over free vars of (bound_sum + X + free_sum)
        let n_free = self.num_vars - self.current_vars_bound.len() - 1;
        let count = 1 << n_free;

        // Coefficients for degree-1 polynomial
        let c0 = /* compute constant term */;
        let c1 = /* compute linear term */;

        UnivariatePoly::new(vec![c0, c1])
    }

    fn bind(&mut self, challenge: F) {
        self.current_vars_bound.push(challenge);
    }
}
```

**Protocol Round-Trip:**
```rust
#[test]
fn test_coordinate_sum_sumcheck() {
    let num_vars = 4;
    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: compute_coordinate_sum(num_vars),
    };

    let mut witness = CoordinateSumWitness {
        num_vars,
        current_vars_bound: vec![],
    };

    let mut transcript = TestTranscript::new();

    // Prove
    let proof = SumcheckProver::prove(
        &claim,
        &mut witness,
        &mut transcript
    );

    // Verify
    let mut verify_transcript = TestTranscript::new();
    let result = SumcheckVerifier::verify(
        &claim,
        &proof,
        &mut verify_transcript
    );

    assert!(result.is_ok());
}
```

**Batching Test:**
```rust
#[test]
fn test_batched_different_sizes() {
    // Create claims with different numbers of variables
    let claims = vec![
        SumcheckClaim { num_vars: 3, degree: 2, claimed_sum: sum1 },
        SumcheckClaim { num_vars: 5, degree: 1, claimed_sum: sum2 },
        SumcheckClaim { num_vars: 4, degree: 2, claimed_sum: sum3 },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckInstanceProver<F>>> = vec![
        Box::new(witness1),
        Box::new(witness2),
        Box::new(witness3),
    ];

    let mut transcript = TestTranscript::new();

    // Prove batched
    let batched_proof = SumcheckProver::prove_batched(
        &claims,
        &mut witnesses,
        &mut transcript
    );

    // Verify batched
    let mut verify_transcript = TestTranscript::new();
    let result = SumcheckVerifier::verify_batched(
        &claims,
        &batched_proof,
        &mut verify_transcript
    );

    assert!(result.is_ok());

    // Also verify each claim works independently
    // ...
}
```

**Streaming Test:**
```rust
struct StreamingPolynomialWitness {
    chunk_size: usize,
    // ... streaming state
}

#[test]
fn test_streaming_large_polynomial() {
    let num_vars = 20; // 2^20 evaluations
    let chunk_size = 1 << 10; // 1024 elements at a time

    let mut streaming_witness = StreamingPolynomialWitness::new(
        num_vars,
        chunk_size
    );

    // Run sumcheck with streaming witness
    // Verify memory usage stays bounded
    // Compare result with non-streaming version
}
```

### Test Properties

**Soundness Tests:**
- Incorrect claimed sum → verification fails
- Modified proof transcript → verification fails
- Wrong degree claim → verification fails

**Completeness Tests:**
- All honest executions verify
- Random polynomials sum correctly
- Edge cases (zero polynomial, constant, single variable)

**Efficiency Tests:**
- Batching reduces proof size vs individual proofs
- Streaming uses less memory than full materialization

### Current Progress

| File | Status | Notes |
|------|--------|-------|
| `tests/integration.rs` | Exists (143 lines, 2 tests) | `blake2b_and_keccak_both_verify`, `evaluate_then_prove_then_verify` |
| `tests/batching.rs` | Not started | |
| `tests/streaming.rs` | Not started | |

**Remaining work:**
- Expand `integration.rs` with more protocol round-trip tests (varying degrees and variable counts)
- Create `batching.rs` with batched sumcheck integration tests
- Create `streaming.rs` with streaming variant tests
- Add negative test cases (soundness) to integration tests

**Note:** The code samples above reference `SumcheckInstanceProver` — the actual trait is `SumcheckCompute`. Update samples if using this task as a reference.

### Acceptance Criteria

- Three integration test files created
- At least 5 tests per file covering different aspects
- Custom witness implementations for testing
- Both positive and negative test cases
- Streaming functionality tested
- All tests pass with `cargo nextest run -p jolt-sumcheck`
- Well-documented test cases
- No source code modifications