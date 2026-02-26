# fuzz-jolt-sumcheck: Fuzz testing for jolt-sumcheck

**Scope:** crates/jolt-sumcheck/fuzz/

**Depends:** impl-jolt-sumcheck, test-jolt-sumcheck, test-jolt-sumcheck-integration

**Verifier:** ./verifiers/scoped.sh /workdir jolt-sumcheck

**Context:**

Create fuzz tests for the `jolt-sumcheck` crate to test soundness properties and find edge cases in the sumcheck protocol implementation.

### Setup

1. Initialize fuzzing:
```bash
cd crates/jolt-sumcheck
cargo fuzz init
```

2. Configure `fuzz/Cargo.toml`:
```toml
[dependencies]
libfuzzer-sys = "0.4"
jolt-sumcheck = { path = ".." }
jolt-field = { path = "../../jolt-field" }
jolt-poly = { path = "../../jolt-poly" }
jolt-transcript = { path = "../../jolt-transcript" }
arbitrary = { version = "1", features = ["derive"] }
```

### Fuzz Targets

#### 1. `round_polys.rs` - Fuzz arbitrary round polynomials

Test that the verifier correctly rejects invalid round polynomials.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_sumcheck::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Blake2bTranscript;
use arbitrary::{Arbitrary, Unstructured};

#[derive(Debug, Arbitrary)]
struct FuzzRoundPolys {
    num_vars: u8,
    degree: u8,
    claimed_sum: u64,
    round_polys: Vec<Vec<u64>>, // Coefficients for each round
}

fuzz_target!(|input: FuzzRoundPolys| {
    // Limit to reasonable sizes
    let num_vars = ((input.num_vars % 8) + 1) as usize;
    let degree = ((input.degree % 4) + 1) as usize;

    let claim = SumcheckClaim {
        num_vars,
        degree,
        claimed_sum: TestField::from(input.claimed_sum),
    };

    // Create round polynomials from fuzz input
    let mut round_polynomials = Vec::new();

    for (round, coeffs) in input.round_polys.iter().enumerate() {
        if round >= num_vars {
            break;
        }

        // Create polynomial of correct degree
        let poly_coeffs: Vec<TestField> = coeffs
            .iter()
            .take(degree + 1)
            .map(|&c| TestField::from(c))
            .collect();

        if poly_coeffs.len() < degree + 1 {
            // Pad with zeros if not enough coefficients
            let mut padded = poly_coeffs;
            padded.resize(degree + 1, TestField::zero());
            round_polynomials.push(UnivariatePoly::new(padded));
        } else {
            round_polynomials.push(UnivariatePoly::new(poly_coeffs));
        }
    }

    // Pad with valid polynomials if not enough rounds
    while round_polynomials.len() < num_vars {
        round_polynomials.push(UnivariatePoly::new(vec![TestField::zero(); degree + 1]));
    }

    let proof = SumcheckProof { round_polynomials };

    // Verify - should handle arbitrary input gracefully
    let mut transcript = Blake2bTranscript::new(b"fuzz-test");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut transcript);

    // Most random proofs should fail verification
    // We're testing that verification doesn't panic
    let _ = result;
});
```

#### 2. `verify.rs` - Fuzz complete sumcheck verification

Test the complete protocol with controlled cheating attempts.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_sumcheck::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::{DensePolynomial, MultilinearPolynomial, UnivariatePoly};

/// Malicious prover that can cheat in specific ways
struct CheatingProver {
    poly: DensePolynomial<TestField>,
    cheat_round: Option<usize>,
    cheat_amount: TestField,
}

impl SumcheckInstanceProver<TestField> for CheatingProver {
    fn round_polynomial(&self) -> UnivariatePoly<TestField> {
        let num_remaining = self.poly.num_vars();
        let round = self.poly.len().trailing_zeros() as usize - num_remaining;

        // Compute honest round polynomial
        let mut coeffs = vec![TestField::zero(); 2]; // degree 1 for multilinear

        // Sum over all possible assignments to remaining variables
        for i in 0..self.poly.len() / 2 {
            let eval0 = self.poly.evaluations()[2 * i];
            let eval1 = self.poly.evaluations()[2 * i + 1];

            coeffs[0] += eval0;
            coeffs[1] += eval1 - eval0;
        }

        // Cheat if this is the cheat round
        if Some(round) == self.cheat_round {
            coeffs[0] += self.cheat_amount;
        }

        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: TestField) {
        self.poly.bind_in_place(challenge);
    }
}

#[derive(Debug, arbitrary::Arbitrary)]
struct CheatingInput {
    poly_size_log: u8,
    cheat_round: Option<u8>,
    cheat_amount: u64,
}

fuzz_target!(|input: CheatingInput| {
    let num_vars = (input.poly_size_log % 6) as usize + 1;
    let poly_size = 1 << num_vars;

    // Create a random polynomial
    let evaluations: Vec<TestField> = (0..poly_size)
        .map(|i| TestField::from((i * 17 + 5) as u64))
        .collect();

    let poly = DensePolynomial::new(evaluations);

    // Compute honest sum
    let honest_sum: TestField = poly.evaluations().iter().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1, // multilinear
        claimed_sum: honest_sum,
    };

    // Create cheating prover
    let cheat_round = input.cheat_round.map(|r| (r as usize) % num_vars);
    let mut witness = CheatingProver {
        poly,
        cheat_round,
        cheat_amount: TestField::from(input.cheat_amount),
    };

    // Run protocol
    let mut prover_transcript = Blake2bTranscript::new(b"fuzz");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut prover_transcript);

    // Verify
    let mut verifier_transcript = Blake2bTranscript::new(b"fuzz");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut verifier_transcript);

    // If prover cheated, verification should fail
    if cheat_round.is_some() && input.cheat_amount != 0 {
        assert!(result.is_err(), "Cheating prover should be caught");
    } else {
        assert!(result.is_ok(), "Honest prover should pass");
    }
});
```

#### 3. `batched_sumcheck.rs` - Fuzz batched sumcheck

Test batched sumcheck with various claim combinations.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_sumcheck::*;
use jolt_field::ark_bn254::Fr as TestField;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};

#[derive(Debug, arbitrary::Arbitrary)]
struct BatchedInput {
    num_claims: u8,
    claim_configs: Vec<(u8, u8)>, // (num_vars, degree) for each claim
}

struct SimpleWitness {
    poly: DensePolynomial<TestField>,
}

impl SumcheckInstanceProver<TestField> for SimpleWitness {
    fn round_polynomial(&self) -> UnivariatePoly<TestField> {
        // Simple implementation for testing
        let degree = 1; // multilinear
        let mut coeffs = vec![TestField::zero(); degree + 1];

        for i in 0..self.poly.len() / 2 {
            let eval0 = self.poly.evaluations()[2 * i];
            let eval1 = self.poly.evaluations()[2 * i + 1];
            coeffs[0] += eval0;
            coeffs[1] += eval1 - eval0;
        }

        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: TestField) {
        self.poly.bind_in_place(challenge);
    }
}

fuzz_target!(|input: BatchedInput| {
    let num_claims = ((input.num_claims % 5) + 1) as usize;

    let mut claims = Vec::new();
    let mut witnesses: Vec<Box<dyn SumcheckInstanceProver<TestField>>> = Vec::new();

    // Create claims from fuzz input
    for i in 0..num_claims {
        let (num_vars, degree) = if i < input.claim_configs.len() {
            let (v, d) = input.claim_configs[i];
            ((v % 6) as usize + 1, (d % 3) as usize + 1)
        } else {
            (3, 1) // default
        };

        // Create polynomial for this claim
        let poly_size = 1 << num_vars;
        let evaluations: Vec<TestField> = (0..poly_size)
            .map(|j| TestField::from(((i * 100 + j * 7) % 97) as u64))
            .collect();

        let poly = DensePolynomial::new(evaluations);
        let sum: TestField = poly.evaluations().iter().sum();

        claims.push(SumcheckClaim {
            num_vars,
            degree,
            claimed_sum: sum,
        });

        witnesses.push(Box::new(SimpleWitness { poly }));
    }

    // Prove batched
    let mut transcript = Blake2bTranscript::new(b"batched-fuzz");
    let proof = SumcheckProver::prove_batched(&claims, &mut witnesses, &mut transcript);

    // Verify batched
    let mut verify_transcript = Blake2bTranscript::new(b"batched-fuzz");
    let result = SumcheckVerifier::verify_batched(&claims, &proof, &mut verify_transcript);

    // Should succeed for honest proofs
    assert!(result.is_ok());
});
```

#### 4. `streaming_sumcheck.rs` - Fuzz streaming variant

Test streaming sumcheck with memory constraints.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_sumcheck::*;
use jolt_field::ark_bn254::Fr as TestField;

/// Streaming witness that generates polynomial evaluations on demand
struct StreamingWitness {
    num_vars: usize,
    seed: u64,
    current_round: usize,
    bound_challenges: Vec<TestField>,
}

impl StreamingWitness {
    fn compute_eval(&self, index: usize) -> TestField {
        // Deterministic evaluation based on seed
        let mut value = self.seed;
        for i in 0..self.num_vars {
            if index & (1 << i) != 0 {
                value = value.wrapping_mul(31).wrapping_add(i as u64);
            }
        }
        TestField::from(value % 97)
    }
}

impl SumcheckInstanceProver<TestField> for StreamingWitness {
    fn round_polynomial(&self) -> UnivariatePoly<TestField> {
        let remaining_vars = self.num_vars - self.current_round;
        let chunk_size = 1 << (remaining_vars - 1);

        let mut coeffs = vec![TestField::zero(); 2];

        // Stream through evaluations without storing all in memory
        for chunk_idx in 0..chunk_size {
            let base_idx = chunk_idx << 1;

            // Apply bound challenges to get actual indices
            let idx0 = self.apply_bindings(base_idx);
            let idx1 = self.apply_bindings(base_idx | 1);

            let eval0 = self.compute_eval(idx0);
            let eval1 = self.compute_eval(idx1);

            coeffs[0] += eval0;
            coeffs[1] += eval1 - eval0;
        }

        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: TestField) {
        self.bound_challenges.push(challenge);
        self.current_round += 1;
    }
}

impl StreamingWitness {
    fn apply_bindings(&self, partial_index: usize) -> usize {
        let mut full_index = 0;
        let mut partial_bit = 0;

        for i in 0..self.num_vars {
            if i < self.bound_challenges.len() {
                // This variable is bound
                if self.bound_challenges[i] == TestField::one() {
                    full_index |= 1 << i;
                }
            } else {
                // Take bit from partial_index
                if partial_index & (1 << partial_bit) != 0 {
                    full_index |= 1 << i;
                }
                partial_bit += 1;
            }
        }

        full_index
    }
}

#[derive(Debug, arbitrary::Arbitrary)]
struct StreamingInput {
    num_vars: u8,
    seed: u64,
}

fuzz_target!(|input: StreamingInput| {
    let num_vars = ((input.num_vars % 10) + 1) as usize;

    // Compute sum using streaming witness
    let mut streaming_sum = TestField::zero();
    let witness = StreamingWitness {
        num_vars,
        seed: input.seed,
        current_round: 0,
        bound_challenges: Vec::new(),
    };

    // Compute sum by streaming
    for i in 0..(1 << num_vars) {
        streaming_sum += witness.compute_eval(i);
    }

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: streaming_sum,
    };

    let mut witness = StreamingWitness {
        num_vars,
        seed: input.seed,
        current_round: 0,
        bound_challenges: Vec::new(),
    };

    // Prove with streaming
    let mut transcript = Blake2bTranscript::new(b"streaming");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut transcript);

    // Verify
    let mut verify_transcript = Blake2bTranscript::new(b"streaming");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut verify_transcript);

    assert!(result.is_ok(), "Streaming witness should produce valid proof");
});
```

### Running Script

Create `fuzz/run-all.sh`:

```bash
#!/bin/bash
set -e

echo "Running jolt-sumcheck fuzz tests..."

FUZZ_TIME=300

echo "Fuzzing round_polys..."
cargo +nightly fuzz run round_polys -- -max_total_time=$FUZZ_TIME

echo "Fuzzing verify..."
cargo +nightly fuzz run verify -- -max_total_time=$FUZZ_TIME

echo "Fuzzing batched_sumcheck..."
cargo +nightly fuzz run batched_sumcheck -- -max_total_time=$FUZZ_TIME

echo "Fuzzing streaming_sumcheck..."
cargo +nightly fuzz run streaming_sumcheck -- -max_total_time=$FUZZ_TIME

echo "All fuzz tests completed!"
```

### Acceptance Criteria

- Four fuzz targets created and compile
- Each runs for 5 minutes without crashes
- Soundness properties tested (cheating detection)
- Batched and streaming variants tested
- Clear invariant checks in each target
- `run-all.sh` script provided
- Corpus directories configured
- No source code modifications