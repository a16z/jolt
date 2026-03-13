# fuzz-jolt-openings: Fuzz testing for jolt-openings

**Scope:** crates/jolt-openings/fuzz/

**Depends:** impl-jolt-openings, test-jolt-openings, test-jolt-openings-integration

**Verifier:** ./verifiers/scoped.sh /workdir jolt-openings

**Context:**

Create fuzz tests for the `jolt-openings` crate to test commitment scheme traits, opening proof verification, and accumulator operations.

### Setup

1. Initialize fuzzing:
```bash
cd crates/jolt-openings
cargo fuzz init
```

2. Configure `fuzz/Cargo.toml`:
```toml
[dependencies]
libfuzzer-sys = "0.4"
jolt-openings = { path = ".." }
jolt-field = { path = "../../jolt-field" }
jolt-poly = { path = "../../jolt-poly" }
jolt-transcript = { path = "../../jolt-transcript" }
arbitrary = { version = "1", features = ["derive"] }
```

### Fuzz Targets

#### 1. `verify_proof.rs` - Fuzz opening proof verification

Test that arbitrary proofs are handled correctly by the verifier.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_openings::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_transcript::Blake2bTranscript;
use arbitrary::{Arbitrary, Unstructured};

/// Mock commitment scheme for fuzzing
#[derive(Clone)]
struct FuzzCommitmentScheme;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FuzzCommitment(Vec<u8>);

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct FuzzProof {
    data: Vec<u8>,
    claimed_eval: TestField,
}

impl CommitmentScheme for FuzzCommitmentScheme {
    type Field = TestField;
    type Commitment = FuzzCommitment;
    type Proof = FuzzProof;
    type ProverSetup = ();
    type VerifierSetup = ();

    fn protocol_name() -> &'static str { "fuzz" }

    fn setup_prover(_: usize) -> Self::ProverSetup { () }
    fn setup_verifier(_: usize) -> Self::VerifierSetup { () }

    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        _: &Self::ProverSetup,
    ) -> Self::Commitment {
        // Simple hash of evaluations
        let mut hash = 0u64;
        for (i, eval) in poly.evaluations().iter().enumerate() {
            hash = hash.wrapping_mul(31).wrapping_add(i as u64);
            // Mix in field element (simplified)
            hash ^= eval.to_bytes()[0] as u64;
        }
        FuzzCommitment(hash.to_le_bytes().to_vec())
    }

    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        point: &[Self::Field],
        eval: Self::Field,
        _: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof {
        // Append to transcript
        transcript.append_scalars(&point);
        transcript.append_scalar(&eval);

        FuzzProof {
            data: vec![poly.num_vars() as u8],
            claimed_eval: eval,
        }
    }

    fn verify(
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        // Append to transcript
        transcript.append_scalars(&point);
        transcript.append_scalar(&eval);

        // Simple verification logic
        if proof.claimed_eval != eval {
            return Err(OpeningsError::VerificationFailed);
        }

        if commitment.0.is_empty() {
            return Err(OpeningsError::InvalidSetup("empty commitment".into()));
        }

        Ok(())
    }
}

#[derive(Debug, Arbitrary)]
struct VerifyInput {
    commitment_data: Vec<u8>,
    point_data: Vec<u8>,
    eval_data: [u8; 32],
    proof_data: Vec<u8>,
    corrupt_commitment: bool,
    corrupt_eval: bool,
}

fuzz_target!(|input: VerifyInput| {
    // Create commitment from fuzz data
    let mut commitment = FuzzCommitment(input.commitment_data);
    if commitment.0.is_empty() {
        commitment.0 = vec![0]; // Ensure non-empty
    }

    // Create point
    let num_vars = (input.point_data.len() % 10) + 1;
    let point: Vec<TestField> = input.point_data
        .chunks(32)
        .take(num_vars)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    // Create eval
    let mut eval = TestField::from_random_bytes(&input.eval_data).unwrap_or(TestField::zero());

    // Create proof
    let proof = FuzzProof {
        data: input.proof_data,
        claimed_eval: eval,
    };

    // Corrupt if requested
    if input.corrupt_commitment {
        commitment.0[0] ^= 1;
    }
    if input.corrupt_eval {
        eval += TestField::one();
    }

    // Verify - should not panic
    let mut transcript = Blake2bTranscript::new(b"fuzz");
    let result = FuzzCommitmentScheme::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &(),
        &mut transcript
    );

    // Check result makes sense
    if input.corrupt_eval {
        assert!(result.is_err());
    }
});
```

#### 2. `accumulator.rs` - Fuzz opening accumulator

Test accumulator with various opening combinations.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_openings::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};

#[derive(Debug, arbitrary::Arbitrary)]
struct AccumulatorInput {
    num_openings: u8,
    openings: Vec<OpeningData>,
}

#[derive(Debug, arbitrary::Arbitrary)]
struct OpeningData {
    poly_size_log: u8,
    poly_seed: u64,
    point_data: Vec<u8>,
}

// Mock homomorphic scheme for testing
struct MockHomomorphicScheme;

impl CommitmentScheme for MockHomomorphicScheme {
    type Field = TestField;
    type Commitment = u64; // Simple mock
    type Proof = Vec<u8>;
    type ProverSetup = ();
    type VerifierSetup = ();

    fn protocol_name() -> &'static str { "mock-homomorphic" }
    fn setup_prover(_: usize) -> Self::ProverSetup { () }
    fn setup_verifier(_: usize) -> Self::VerifierSetup { () }

    fn commit(poly: &impl MultilinearPolynomial<Self::Field>, _: &Self::ProverSetup) -> Self::Commitment {
        // Simple hash
        poly.evaluations().len() as u64
    }

    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        point: &[Self::Field],
        eval: Self::Field,
        _: &Self::ProverSetup,
        _: &mut impl Transcript,
    ) -> Self::Proof {
        vec![poly.num_vars() as u8]
    }

    fn verify(
        _: &Self::Commitment,
        _: &[Self::Field],
        _: Self::Field,
        _: &Self::Proof,
        _: &Self::VerifierSetup,
        _: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        Ok(())
    }
}

impl HomomorphicCommitmentScheme for MockHomomorphicScheme {
    type BatchedProof = Vec<Vec<u8>>;

    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment {
        commitments.iter()
            .zip(scalars)
            .map(|(c, s)| c * s.to_bytes()[0] as u64)
            .sum()
    }

    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        _: &Self::ProverSetup,
        _: &mut impl Transcript,
    ) -> Self::BatchedProof {
        polynomials.iter()
            .map(|p| vec![p.num_vars() as u8])
            .collect()
    }

    fn batch_verify(
        _: &[Self::Commitment],
        _: &[Vec<Self::Field>],
        _: &[Self::Field],
        _: &Self::BatchedProof,
        _: &Self::VerifierSetup,
        _: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        Ok(())
    }
}

fuzz_target!(|input: AccumulatorInput| {
    let num_openings = (input.num_openings % 20) as usize + 1;

    let mut prover_acc = ProverOpeningAccumulator::<TestField>::new();

    // Accumulate openings
    for i in 0..num_openings {
        let opening_data = &input.openings[i % input.openings.len()];

        // Create polynomial
        let num_vars = (opening_data.poly_size_log % 8) as usize + 1;
        let poly_size = 1 << num_vars;

        let evaluations: Vec<TestField> = (0..poly_size)
            .map(|j| {
                let value = opening_data.poly_seed
                    .wrapping_mul(j as u64 + 1)
                    .wrapping_add(i as u64);
                TestField::from(value % 97)
            })
            .collect();

        let poly = DensePolynomial::new(evaluations);

        // Create point
        let point: Vec<TestField> = opening_data.point_data
            .chunks(32)
            .take(num_vars)
            .map(|chunk| {
                let mut bytes = [0u8; 32];
                bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
                TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
            })
            .collect();

        let point = if point.len() == num_vars {
            point
        } else {
            vec![TestField::zero(); num_vars]
        };

        let eval = poly.evaluate(&point);

        // Accumulate
        prover_acc.accumulate(&poly, point, eval);
    }

    // Reduce and prove - should not panic
    let mut transcript = Blake2bTranscript::new(b"accumulator-fuzz");
    let proofs = prover_acc.reduce_and_prove::<MockHomomorphicScheme>(
        &(),
        &mut transcript
    );

    // Verify we got some proofs
    assert!(!proofs.is_empty());
});
```

#### 3. `batching.rs` - Fuzz batch operations

Test homomorphic batching with various combinations.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_openings::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_transcript::Blake2bTranscript;

#[derive(Debug, arbitrary::Arbitrary)]
struct BatchingInput {
    num_commitments: u8,
    scalars: Vec<u8>,
    batch_size: u8,
    point_variations: Vec<Vec<u8>>,
}

fuzz_target!(|input: BatchingInput| {
    let num_commitments = ((input.num_commitments % 10) + 1) as usize;
    let batch_size = ((input.batch_size % 5) + 1) as usize;

    // Create commitments
    let commitments: Vec<u64> = (0..num_commitments)
        .map(|i| (i * 17 + 5) as u64)
        .collect();

    // Create scalars
    let scalars: Vec<TestField> = input.scalars
        .iter()
        .take(num_commitments)
        .map(|&s| TestField::from(s as u64))
        .collect();

    // Pad if needed
    let scalars = if scalars.len() < num_commitments {
        let mut s = scalars;
        while s.len() < num_commitments {
            s.push(TestField::one());
        }
        s
    } else {
        scalars
    };

    // Test combining commitments
    let combined = MockHomomorphicScheme::combine_commitments(&commitments, &scalars);

    // Verify homomorphic property
    let expected: u64 = commitments.iter()
        .zip(&scalars)
        .map(|(c, s)| c * s.to_bytes()[0] as u64)
        .sum();

    assert_eq!(combined, expected);

    // Create polynomials for batch proving
    let mut polynomials: Vec<DensePolynomial<TestField>> = Vec::new();
    let mut points: Vec<Vec<TestField>> = Vec::new();

    for i in 0..batch_size {
        let num_vars = 3;
        let poly_size = 1 << num_vars;

        let evaluations: Vec<TestField> = (0..poly_size)
            .map(|j| TestField::from(((i * 100 + j * 7) % 97) as u64))
            .collect();

        polynomials.push(DensePolynomial::new(evaluations));

        // Create point from fuzz data
        let point_data = &input.point_variations[i % input.point_variations.len()];
        let point: Vec<TestField> = point_data
            .chunks(32)
            .take(num_vars)
            .map(|chunk| {
                let mut bytes = [0u8; 32];
                bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
                TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
            })
            .collect();

        let point = if point.len() == num_vars {
            point
        } else {
            vec![TestField::zero(); num_vars]
        };

        points.push(point);
    }

    // Compute evaluations
    let evals: Vec<TestField> = polynomials.iter()
        .zip(&points)
        .map(|(p, pt)| p.evaluate(pt))
        .collect();

    // Batch prove
    let poly_refs: Vec<&dyn MultilinearPolynomial<TestField>> = polynomials.iter()
        .map(|p| p as &dyn MultilinearPolynomial<TestField>)
        .collect();

    let mut transcript = Blake2bTranscript::new(b"batch-fuzz");
    let batch_proof = MockHomomorphicScheme::batch_prove(
        &poly_refs,
        &points,
        &evals,
        &(),
        &mut transcript
    );

    // Should produce proof for each polynomial
    assert_eq!(batch_proof.len(), batch_size);
});
```

#### 4. `streaming_commitment.rs` - Fuzz streaming operations

Test streaming commitment with various chunk sizes.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_openings::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::DensePolynomial;

struct MockStreamingScheme;

impl CommitmentScheme for MockStreamingScheme {
    type Field = TestField;
    type Commitment = u64;
    type Proof = ();
    type ProverSetup = ();
    type VerifierSetup = ();

    fn protocol_name() -> &'static str { "mock-streaming" }
    fn setup_prover(_: usize) -> Self::ProverSetup { () }
    fn setup_verifier(_: usize) -> Self::VerifierSetup { () }

    fn commit(poly: &impl MultilinearPolynomial<Self::Field>, _: &Self::ProverSetup) -> Self::Commitment {
        poly.evaluations().len() as u64
    }

    fn prove(
        _: &impl MultilinearPolynomial<Self::Field>,
        _: &[Self::Field],
        _: Self::Field,
        _: &Self::ProverSetup,
        _: &mut impl Transcript,
    ) -> Self::Proof {
        ()
    }

    fn verify(
        _: &Self::Commitment,
        _: &[Self::Field],
        _: Self::Field,
        _: &Self::Proof,
        _: &Self::VerifierSetup,
        _: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        Ok(())
    }
}

impl StreamingCommitmentScheme for MockStreamingScheme {
    type PartialCommitment = Vec<u64>;

    fn begin_streaming(_: &Self::ProverSetup) -> Self::PartialCommitment {
        vec![]
    }

    fn stream_chunk(partial: &mut Self::PartialCommitment, chunk: &[Self::Field]) {
        // Simple hash of chunk
        let hash = chunk.len() as u64 * 31 + chunk.first().map(|f| f.to_bytes()[0] as u64).unwrap_or(0);
        partial.push(hash);
    }

    fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment {
        partial.iter().sum()
    }
}

#[derive(Debug, arbitrary::Arbitrary)]
struct StreamingInput {
    poly_size_log: u8,
    chunk_size_log: u8,
    poly_seed: u64,
}

fuzz_target!(|input: StreamingInput| {
    let num_vars = ((input.poly_size_log % 12) + 1) as usize;
    let poly_size = 1 << num_vars;
    let chunk_size = 1 << ((input.chunk_size_log % 8) as usize + 1);

    // Create polynomial
    let evaluations: Vec<TestField> = (0..poly_size)
        .map(|i| {
            let value = input.poly_seed.wrapping_mul(i as u64 + 1);
            TestField::from(value % 97)
        })
        .collect();

    let poly = DensePolynomial::new(evaluations);

    // Non-streaming commitment
    let direct_commitment = MockStreamingScheme::commit(&poly, &());

    // Streaming commitment
    let mut partial = MockStreamingScheme::begin_streaming(&());

    let all_evals = poly.evaluations();
    for chunk in all_evals.chunks(chunk_size) {
        MockStreamingScheme::stream_chunk(&mut partial, chunk);
    }

    let streaming_commitment = MockStreamingScheme::finalize_streaming(partial);

    // In this mock, they won't be equal, but we're testing for panics
    let _ = streaming_commitment;
    let _ = direct_commitment;
});
```

### Running Script

Create `fuzz/run-all.sh`:

```bash
#!/bin/bash
set -e

echo "Running jolt-openings fuzz tests..."

FUZZ_TIME=300

echo "Fuzzing verify_proof..."
cargo +nightly fuzz run verify_proof -- -max_total_time=$FUZZ_TIME

echo "Fuzzing accumulator..."
cargo +nightly fuzz run accumulator -- -max_total_time=$FUZZ_TIME

echo "Fuzzing batching..."
cargo +nightly fuzz run batching -- -max_total_time=$FUZZ_TIME

echo "Fuzzing streaming_commitment..."
cargo +nightly fuzz run streaming_commitment -- -max_total_time=$FUZZ_TIME

echo "All fuzz tests completed!"
```

### Acceptance Criteria

- Four fuzz targets created and compile
- Each runs for 5 minutes without crashes
- Mock commitment schemes implemented for testing
- Accumulator operations thoroughly fuzzed
- Homomorphic properties tested
- Streaming functionality tested
- `run-all.sh` script provided
- No source code modifications