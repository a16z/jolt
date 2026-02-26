# test-jolt-dory-integration: Integration tests for jolt-dory

**Scope:** crates/jolt-dory/tests/

**Depends:** impl-jolt-dory, test-jolt-dory

**Verifier:** ./verifiers/scoped.sh /workdir jolt-dory

**Context:**

Write integration tests for the `jolt-dory` crate that verify the Dory polynomial commitment scheme implementation from an external user's perspective.

### Integration Test Files

Create the following test files in `crates/jolt-dory/tests/`:

#### 1. `commitment.rs` - Test commitment round-trips

Test complete commitment, opening, and verification flows:
- Small polynomials (< 10 variables)
- Large polynomials (15-20 variables)
- Test all trait methods from `CommitmentScheme`
- Verify proofs with correct and incorrect evaluations
- Test parameter boundaries

#### 2. `streaming.rs` - Test streaming commitment

Test the streaming commitment functionality:
- Stream large polynomials in chunks
- Verify streaming produces same commitment as non-streaming
- Test various chunk sizes
- Test interruption/resumption if supported

#### 3. `batching.rs` - Test batched operations

Test homomorphic batching capabilities:
- Batch multiple polynomial openings
- Test linear combination of commitments
- Verify batch proof size efficiency
- Test RLC (random linear combination) security

### Implementation Examples

**Basic Commitment Test:**
```rust
#[test]
fn test_dory_commitment_round_trip() {
    // Setup Dory with specific parameters
    let params = DoryParams {
        t: 8,
        max_num_rows: 1024,
        num_columns: 256,
    };
    let dory = DoryScheme::new(params);

    // Create test polynomial
    let num_vars = 10;
    let poly = DensePolynomial::<ark_bn254::Fr>::random(num_vars, &mut rng);

    // Setup
    let prover_setup = dory.setup_prover(1 << num_vars);
    let verifier_setup = dory.setup_verifier(1 << num_vars);

    // Commit
    let commitment = dory.commit(&poly, &prover_setup);

    // Create opening at random point
    let point: Vec<_> = (0..num_vars)
        .map(|_| ark_bn254::Fr::random(&mut rng))
        .collect();

    let eval = poly.evaluate(&point);

    // Prove
    let mut prover_transcript = Blake2bTranscript::new(b"test");
    let proof = dory.prove(
        &poly,
        &point,
        eval,
        &prover_setup,
        &mut prover_transcript
    );

    // Verify
    let mut verifier_transcript = Blake2bTranscript::new(b"test");
    let result = dory.verify(
        &commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript
    );

    assert!(result.is_ok());
}

#[test]
fn test_incorrect_evaluation_fails() {
    let params = DoryParams { /* ... */ };
    let dory = DoryScheme::new(params);

    // ... setup and commit ...

    let correct_eval = poly.evaluate(&point);
    let incorrect_eval = correct_eval + ark_bn254::Fr::one();

    // Prove with incorrect eval
    let proof = dory.prove(
        &poly,
        &point,
        incorrect_eval, // Wrong!
        &prover_setup,
        &mut prover_transcript
    );

    // Verification should fail
    let result = dory.verify(
        &commitment,
        &point,
        incorrect_eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript
    );

    assert!(result.is_err());
}
```

**Streaming Commitment:**
```rust
#[test]
fn test_streaming_commitment() {
    let params = DoryParams { /* ... */ };
    let dory = DoryScheme::new(params);

    // Large polynomial
    let num_vars = 18; // 2^18 = 262,144 evaluations
    let poly = DensePolynomial::<ark_bn254::Fr>::random(num_vars, &mut rng);

    let prover_setup = dory.setup_prover(1 << num_vars);

    // Non-streaming commitment
    let direct_commitment = dory.commit(&poly, &prover_setup);

    // Streaming commitment
    let mut partial = dory.begin_streaming(&prover_setup);

    let chunk_size = 1 << 10; // 1024 elements per chunk
    let evaluations = poly.evaluations();

    for chunk in evaluations.chunks(chunk_size) {
        dory.stream_chunk(&mut partial, chunk);
    }

    let streaming_commitment = dory.finalize_streaming(partial);

    // Should produce identical commitments
    assert_eq!(direct_commitment, streaming_commitment);
}
```

**Batched Operations:**
```rust
#[test]
fn test_homomorphic_batching() {
    let params = DoryParams { /* ... */ };
    let dory = DoryScheme::new(params);

    // Create multiple polynomials
    let num_vars = 8;
    let poly1 = DensePolynomial::random(num_vars, &mut rng);
    let poly2 = DensePolynomial::random(num_vars, &mut rng);
    let poly3 = DensePolynomial::random(num_vars, &mut rng);

    let prover_setup = dory.setup_prover(1 << num_vars);

    // Commit to each
    let c1 = dory.commit(&poly1, &prover_setup);
    let c2 = dory.commit(&poly2, &prover_setup);
    let c3 = dory.commit(&poly3, &prover_setup);

    // Test homomorphic property: commitment is linear
    let alpha = ark_bn254::Fr::random(&mut rng);
    let beta = ark_bn254::Fr::random(&mut rng);
    let gamma = ark_bn254::Fr::random(&mut rng);

    // Combine commitments
    let combined_commitment = dory.combine_commitments(
        &[c1.clone(), c2.clone(), c3.clone()],
        &[alpha, beta, gamma]
    );

    // Create combined polynomial
    let combined_poly = poly1.scale(alpha)
        .add(&poly2.scale(beta))
        .add(&poly3.scale(gamma));

    // Commitment to combination should equal combination of commitments
    let direct_combined = dory.commit(&combined_poly, &prover_setup);
    assert_eq!(combined_commitment, direct_combined);
}

#[test]
fn test_batch_proving() {
    let params = DoryParams { /* ... */ };
    let dory = DoryScheme::new(params);

    // Multiple polynomials and points
    let polys: Vec<_> = (0..5)
        .map(|_| DensePolynomial::random(10, &mut rng))
        .collect();

    let points: Vec<Vec<_>> = (0..5)
        .map(|_| {
            (0..10)
                .map(|_| ark_bn254::Fr::random(&mut rng))
                .collect()
        })
        .collect();

    let evals: Vec<_> = polys.iter()
        .zip(&points)
        .map(|(p, pt)| p.evaluate(pt))
        .collect();

    // Batch prove
    let mut transcript = Blake2bTranscript::new(b"batch-test");
    let batch_proof = dory.batch_prove(
        &polys.iter().map(|p| p as &dyn MultilinearPolynomial<_>).collect::<Vec<_>>(),
        &points,
        &evals,
        &prover_setup,
        &mut transcript
    );

    // Batch verify
    let commitments: Vec<_> = polys.iter()
        .map(|p| dory.commit(p, &prover_setup))
        .collect();

    let mut verify_transcript = Blake2bTranscript::new(b"batch-test");
    let result = dory.batch_verify(
        &commitments,
        &points,
        &evals,
        &batch_proof,
        &verifier_setup,
        &mut verify_transcript
    );

    assert!(result.is_ok());
}
```

**Parameter Testing:**
```rust
#[test]
fn test_parameter_boundaries() {
    // Test various parameter combinations
    let test_params = vec![
        DoryParams { t: 4, max_num_rows: 256, num_columns: 64 },
        DoryParams { t: 8, max_num_rows: 1024, num_columns: 256 },
        DoryParams { t: 16, max_num_rows: 4096, num_columns: 1024 },
    ];

    for params in test_params {
        let dory = DoryScheme::new(params.clone());

        // Test with polynomial at max size
        let max_vars = params.t + params.num_columns.ilog2() as usize;
        let poly = DensePolynomial::random(max_vars, &mut rng);

        // Should work at boundary
        let prover_setup = dory.setup_prover(1 << max_vars);
        let commitment = dory.commit(&poly, &prover_setup);

        // ... verify round-trip works
    }
}
```

### Edge Cases to Test

- Empty polynomial (0 variables)
- Maximum size polynomial for given parameters
- Commitments with all-zero evaluations
- Points with special structure (all 0s, all 1s)
- Proof verification with tampered commitments

### Acceptance Criteria

- Three integration test files created
- Round-trip commitment/opening/verification tested
- Streaming functionality verified
- Homomorphic properties tested
- Batch operations verified
- Parameter boundaries tested
- All tests use public API only
- Tests pass with `cargo nextest run -p jolt-dory`
- Well-documented test cases
- No source code modifications