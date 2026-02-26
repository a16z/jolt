# test-jolt-openings-integration: Integration tests for jolt-openings

**Scope:** crates/jolt-openings/tests/

**Depends:** impl-jolt-openings, test-jolt-openings

**Verifier:** ./verifiers/scoped.sh /workdir jolt-openings

**Context:**

Write integration tests for the `jolt-openings` crate that verify the commitment scheme traits and opening accumulator functionality from an external perspective.

### Integration Test Files

Create the following test files in `crates/jolt-openings/tests/`:

#### 1. `commitment_api.rs` - Test trait implementations

Test commitment scheme trait usage patterns:
- Mock implementations of `CommitmentScheme` trait
- Test homomorphic operations with mock `HomomorphicCommitmentScheme`
- Verify streaming commitment workflow
- Test trait object usage and type erasure

#### 2. `accumulator.rs` - Test accumulator round-trips

Test opening accumulator functionality:
- Accumulate multiple openings and verify batch reduction
- Test accumulator with different commitment schemes
- Verify accumulator produces minimal proof sets
- Test edge cases (empty accumulator, single opening)

#### 3. `batching.rs` - Test batch operations

Test batched opening proofs:
- Multiple polynomials at same point
- Same polynomial at multiple points
- Different polynomials at different points
- Verify batch proof size vs individual proofs
- Test RLC (random linear combination) correctness

### Implementation Examples

**Mock Commitment Scheme:**
```rust
/// Simple mock commitment scheme for testing
struct MockCommitmentScheme;

#[derive(Clone, Debug)]
struct MockCommitment(u64); // Simple mock

#[derive(Clone)]
struct MockProof {
    poly_hash: u64,
    point_hash: u64,
    eval: TestField,
}

impl CommitmentScheme for MockCommitmentScheme {
    type Field = TestField;
    type Commitment = MockCommitment;
    type Proof = MockProof;
    type ProverSetup = ();
    type VerifierSetup = ();

    fn protocol_name() -> &'static str { "mock" }

    fn setup_prover(_max_size: usize) -> Self::ProverSetup { () }
    fn setup_verifier(_max_size: usize) -> Self::VerifierSetup { () }

    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        // Simple hash of evaluations for testing
        MockCommitment(hash_polynomial(poly))
    }

    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript,
    ) -> Self::Proof {
        MockProof {
            poly_hash: hash_polynomial(poly),
            point_hash: hash_point(point),
            eval,
        }
    }

    fn verify(
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        // Verify mock proof
        if proof.poly_hash != commitment.0 {
            return Err(OpeningsError::CommitmentMismatch { ... });
        }
        // ... additional checks
        Ok(())
    }
}
```

**Accumulator Test:**
```rust
#[test]
fn test_accumulator_batch_reduction() {
    let mut prover_acc = ProverOpeningAccumulator::<TestField>::new();

    // Create test polynomials
    let poly1 = TestPolynomial::new(vec![1, 2, 3, 4]);
    let poly2 = TestPolynomial::new(vec![5, 6, 7, 8]);
    let poly3 = TestPolynomial::new(vec![9, 10, 11, 12]);

    let point1 = vec![TestField::from(1), TestField::from(0)];
    let point2 = vec![TestField::from(0), TestField::from(1)];

    // Accumulate openings
    prover_acc.accumulate(&poly1, point1.clone(), poly1.evaluate(&point1));
    prover_acc.accumulate(&poly2, point1.clone(), poly2.evaluate(&point1));
    prover_acc.accumulate(&poly3, point2.clone(), poly3.evaluate(&point2));

    // Reduce and prove
    let setup = MockCommitmentScheme::setup_prover(16);
    let mut transcript = TestTranscript::new();

    let proofs = prover_acc.reduce_and_prove::<MockHomomorphicScheme>(
        &setup,
        &mut transcript
    );

    // Should produce 2 batch proofs (grouped by point)
    assert_eq!(proofs.len(), 2);
}
```

**Homomorphic Operations:**
```rust
impl HomomorphicCommitmentScheme for MockHomomorphicScheme {
    type BatchedProof = Vec<MockProof>;

    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment {
        // Linear combination of mock commitments
        let combined_hash = commitments.iter()
            .zip(scalars)
            .map(|(c, s)| c.0 * s.to_u64())
            .sum();

        MockCommitment(combined_hash)
    }

    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::BatchedProof {
        // Create batch proof
        polynomials.iter()
            .zip(points.iter())
            .zip(evals.iter())
            .map(|((poly, point), eval)| {
                Self::prove(*poly, point, *eval, setup, transcript)
            })
            .collect()
    }

    // ... batch_verify
}
```

**Streaming Test:**
```rust
#[test]
fn test_streaming_commitment() {
    struct MockStreamingScheme;

    impl StreamingCommitmentScheme for MockStreamingScheme {
        type PartialCommitment = Vec<u64>;

        fn begin_streaming(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
            vec![]
        }

        fn stream_chunk(partial: &mut Self::PartialCommitment, chunk: &[Self::Field]) {
            partial.push(hash_chunk(chunk));
        }

        fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment {
            MockCommitment(partial.iter().sum())
        }
    }

    // Test streaming workflow
    let mut partial = MockStreamingScheme::begin_streaming(&());

    // Stream chunks
    for chunk in large_poly_chunks {
        MockStreamingScheme::stream_chunk(&mut partial, &chunk);
    }

    let commitment = MockStreamingScheme::finalize_streaming(partial);

    // Verify matches non-streaming commitment
    // ...
}
```

### Test Cases

**Type Erasure:**
```rust
#[test]
fn test_type_erased_accumulator() {
    let mut verifier_acc = VerifierOpeningAccumulator::<TestField>::new();

    // Can accumulate different commitment types via type erasure
    let mock_commitment = MockCommitment(42);
    let another_commitment = AnotherScheme::Commitment { ... };

    verifier_acc.accumulate(&mock_commitment, point1, eval1);
    verifier_acc.accumulate(&another_commitment, point2, eval2);

    // Verification dispatches to correct scheme
    // ...
}
```

### Acceptance Criteria

- Three integration test files created
- Mock commitment schemes implemented for testing
- Accumulator functionality thoroughly tested
- Homomorphic operations verified
- Streaming commitment tested
- Type erasure and trait object usage demonstrated
- All tests pass with `cargo nextest run -p jolt-openings`
- Well-documented test implementations
- No source code modifications