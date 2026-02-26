# cross-crate-integration-1: Spartan + Dory Integration Testing

**Scope:** workspace-level integration tests

**Depends:** impl-jolt-spartan, impl-jolt-dory, test-jolt-spartan-integration, test-jolt-dory-integration

**Verifier:** ./verifiers/scoped.sh /workdir jolt-spartan jolt-dory

**Context:**

Create integration tests that verify `jolt-spartan` and `jolt-dory` work correctly together. This tests the interface between the R1CS proving system and the Dory polynomial commitment scheme.

### Test Location

Create tests in a workspace-level test directory:
- `tests/spartan_dory_integration.rs`

### Integration Tests

#### 1. Basic R1CS with Dory Commitments

```rust
use jolt_spartan::*;
use jolt_dory::*;
use jolt_field::ark_bn254::Fr as BN254Fr;
use jolt_poly::DensePolynomial;
use jolt_transcript::Blake2bTranscript;

/// Simple R1CS instance for testing: x² + x = y
struct QuadraticR1CS;

impl R1CS<BN254Fr> for QuadraticR1CS {
    fn num_constraints(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 3 } // z = [1, x, y]

    fn multiply_witness(&self, witness: &[BN254Fr]) -> (Vec<BN254Fr>, Vec<BN254Fr>, Vec<BN254Fr>) {
        // First constraint: x * x = x²
        // Second constraint: (x² + x) * 1 = y

        let x = witness[1];
        let y = witness[2];
        let x_squared = x * x;

        // A matrix evaluations
        let az = vec![x, x_squared + x];

        // B matrix evaluations
        let bz = vec![x, BN254Fr::one()];

        // C matrix evaluations
        let cz = vec![x_squared, y];

        (az, bz, cz)
    }
}

#[test]
fn test_spartan_with_dory_simple_r1cs() {
    // Create R1CS instance
    let r1cs = QuadraticR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);

    // Create witness: x = 3, y = 12 (since 3² + 3 = 12)
    let witness = vec![BN254Fr::from(1), BN254Fr::from(3), BN254Fr::from(12)];

    // Setup Dory
    let dory_params = DoryParams {
        t: 8,
        max_num_rows: 1024,
        num_columns: 256,
    };
    let dory = DoryScheme::new(dory_params);

    // Get max polynomial size for Spartan
    let max_poly_size = calculate_max_poly_size(&r1cs);
    let prover_setup = dory.setup_prover(max_poly_size);

    // Create transcript
    let mut prover_transcript = Blake2bTranscript::new(b"spartan-dory-test");

    // Generate Spartan proof using Dory
    let proof = SpartanProver::prove::<BN254Fr, DoryScheme>(
        &key,
        &witness,
        &prover_setup,
        &mut prover_transcript
    ).expect("Proving should succeed");

    // Verify
    let verifier_setup = dory.setup_verifier(max_poly_size);
    let mut verifier_transcript = Blake2bTranscript::new(b"spartan-dory-test");

    let result = SpartanVerifier::verify::<BN254Fr, DoryScheme>(
        &key,
        &proof,
        &verifier_setup,
        &mut verifier_transcript
    );

    assert!(result.is_ok(), "Proof verification failed");
}
```

#### 2. Larger R1CS System

```rust
/// R1CS for system of equations:
/// x₁ + x₂ = x₃
/// x₁ * x₂ = x₄
/// x₃ * x₃ = x₅
struct SystemR1CS;

impl R1CS<BN254Fr> for SystemR1CS {
    fn num_constraints(&self) -> usize { 3 }
    fn num_variables(&self) -> usize { 6 } // z = [1, x₁, x₂, x₃, x₄, x₅]

    fn multiply_witness(&self, witness: &[BN254Fr]) -> (Vec<BN254Fr>, Vec<BN254Fr>, Vec<BN254Fr>) {
        let x1 = witness[1];
        let x2 = witness[2];
        let x3 = witness[3];
        let x4 = witness[4];
        let x5 = witness[5];

        // Constraint 1: (x₁ + x₂) * 1 = x₃
        // Constraint 2: x₁ * x₂ = x₄
        // Constraint 3: x₃ * x₃ = x₅

        let az = vec![x1 + x2, x1, x3];
        let bz = vec![BN254Fr::one(), x2, x3];
        let cz = vec![x3, x4, x5];

        (az, bz, cz)
    }
}

#[test]
fn test_spartan_dory_larger_system() {
    let r1cs = SystemR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);

    // Witness: x₁=2, x₂=3, x₃=5, x₄=6, x₅=25
    let witness = vec![
        BN254Fr::from(1),
        BN254Fr::from(2),
        BN254Fr::from(3),
        BN254Fr::from(5),
        BN254Fr::from(6),
        BN254Fr::from(25),
    ];

    // Use larger Dory parameters
    let dory_params = DoryParams {
        t: 10,
        max_num_rows: 2048,
        num_columns: 512,
    };
    let dory = DoryScheme::new(dory_params);

    // Prove and verify
    test_prove_and_verify(&r1cs, &witness, dory);
}
```

#### 3. Uniform R1CS with Dory

```rust
#[test]
fn test_uniform_r1cs_with_dory() {
    let block_size = 4;
    let num_blocks = 50;

    // Create uniform R1CS with repeating pattern
    let uniform_r1cs = UniformR1CS::<BN254Fr>::new(
        block_size,
        num_blocks,
        |block_idx| {
            // Each block computes: a² + b² = c, a * b = d
            UniformBlock {
                constraints: vec![
                    // Constraint pattern for each block
                    (0, 0, 2), // a * a = a²
                    (1, 1, 3), // b * b = b²
                    (4, 5, 2), // (a² + b²) * 1 = c
                    (0, 1, 3), // a * b = d
                ],
            }
        }
    );

    let key = SpartanKey::from_r1cs(&uniform_r1cs);

    // Generate satisfying witness
    let witness = generate_uniform_witness(&uniform_r1cs);

    // Configure Dory for larger uniform instance
    let dory_params = DoryParams {
        t: 12,
        max_num_rows: 4096,
        num_columns: 1024,
    };
    let dory = DoryScheme::new(dory_params);

    // Time the proving
    let start = std::time::Instant::now();

    let proof = prove_with_dory(&key, &witness, dory.clone());

    let proving_time = start.elapsed();
    println!("Uniform R1CS proving time: {:?}", proving_time);

    // Verify
    assert!(verify_with_dory(&key, &proof, dory).is_ok());
}
```

#### 4. Test Univariate Skip with Dory

```rust
#[test]
fn test_univariate_skip_with_dory() {
    let r1cs = create_test_r1cs(100); // 100 constraints
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = create_satisfying_witness(&r1cs);

    let dory = create_standard_dory();

    // Test with standard first round
    let proof_standard = prove_with_strategy(
        &key,
        &witness,
        &dory,
        FirstRoundStrategy::Standard
    );

    // Test with univariate skip
    let proof_skip = prove_with_strategy(
        &key,
        &witness,
        &dory,
        FirstRoundStrategy::UnivariateSkip { domain_size: 256 }
    );

    // Both should verify
    assert!(verify_with_dory(&key, &proof_standard, &dory).is_ok());
    assert!(verify_with_dory(&key, &proof_skip, &dory).is_ok());

    // Compare proof sizes
    let size_standard = bincode::serialize(&proof_standard).unwrap().len();
    let size_skip = bincode::serialize(&proof_skip).unwrap().len();

    println!("Standard proof size: {} bytes", size_standard);
    println!("Uni-skip proof size: {} bytes", size_skip);
}
```

#### 5. Stress Test with Many Constraints

```rust
#[test]
#[ignore] // Run with --ignored flag for stress tests
fn test_spartan_dory_stress_test() {
    // Create R1CS with many constraints
    for num_constraints in [10, 100, 1000, 5000] {
        println!("\nTesting with {} constraints", num_constraints);

        let r1cs = create_random_r1cs(num_constraints);
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = create_random_satisfying_witness(&r1cs);

        // Scale Dory parameters with constraint count
        let dory_params = DoryParams {
            t: 12 + (num_constraints.ilog2() / 4) as usize,
            max_num_rows: (num_constraints * 4).next_power_of_two(),
            num_columns: 1024,
        };
        let dory = DoryScheme::new(dory_params);

        let start = std::time::Instant::now();
        let proof = prove_with_dory(&key, &witness, &dory);
        let proving_time = start.elapsed();

        let proof_size = bincode::serialize(&proof).unwrap().len();

        println!("  Proving time: {:?}", proving_time);
        println!("  Proof size: {} KB", proof_size / 1024);

        assert!(verify_with_dory(&key, &proof, &dory).is_ok());
    }
}
```

#### 6. Error Cases

```rust
#[test]
fn test_spartan_dory_invalid_witness() {
    let r1cs = QuadraticR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);

    // Invalid witness: x = 3, y = 10 (should be 12)
    let invalid_witness = vec![
        BN254Fr::from(1),
        BN254Fr::from(3),
        BN254Fr::from(10), // Wrong!
    ];

    let dory = create_standard_dory();

    // Proving should fail with constraint violation
    let result = SpartanProver::prove::<BN254Fr, DoryScheme>(
        &key,
        &invalid_witness,
        &dory.setup_prover(1024),
        &mut Blake2bTranscript::new(b"test")
    );

    assert!(matches!(result, Err(SpartanError::ConstraintViolation(_))));
}

#[test]
fn test_dory_polynomial_too_large() {
    let dory_params = DoryParams {
        t: 4,
        max_num_rows: 16,
        num_columns: 16,
    };
    let dory = DoryScheme::new(dory_params);

    // Try to commit to polynomial larger than setup
    let large_poly = DensePolynomial::<BN254Fr>::random(10, &mut rand::thread_rng());

    let setup = dory.setup_prover(256); // Setup for smaller size

    // This should fail
    let result = std::panic::catch_unwind(|| {
        dory.commit(&large_poly, &setup)
    });

    assert!(result.is_err());
}
```

### Helper Functions

Create helper functions for common operations:

```rust
fn create_standard_dory() -> DoryScheme {
    DoryScheme::new(DoryParams {
        t: 8,
        max_num_rows: 1024,
        num_columns: 256,
    })
}

fn prove_with_dory<R: R1CS<BN254Fr>>(
    key: &SpartanKey<BN254Fr>,
    witness: &[BN254Fr],
    dory: DoryScheme,
) -> SpartanProof<BN254Fr, DoryScheme> {
    let max_size = calculate_max_poly_size(key);
    let setup = dory.setup_prover(max_size);
    let mut transcript = Blake2bTranscript::new(b"test");

    SpartanProver::prove(key, witness, &setup, &mut transcript)
        .expect("Proving should succeed")
}

fn verify_with_dory(
    key: &SpartanKey<BN254Fr>,
    proof: &SpartanProof<BN254Fr, DoryScheme>,
    dory: DoryScheme,
) -> Result<(), SpartanError> {
    let max_size = calculate_max_poly_size(key);
    let setup = dory.setup_verifier(max_size);
    let mut transcript = Blake2bTranscript::new(b"test");

    SpartanVerifier::verify(key, proof, &setup, &mut transcript)
}

fn calculate_max_poly_size<F: Field>(r1cs: &impl R1CS<F>) -> usize {
    // Calculate based on R1CS dimensions
    let vars = r1cs.num_variables();
    let constraints = r1cs.num_constraints();
    (vars * constraints).next_power_of_two()
}
```

### Test Configuration

Add to workspace `Cargo.toml`:

```toml
[[test]]
name = "spartan_dory_integration"
path = "tests/spartan_dory_integration.rs"

[dev-dependencies]
jolt-spartan = { path = "crates/jolt-spartan" }
jolt-dory = { path = "crates/jolt-dory" }
jolt-field = { path = "crates/jolt-field" }
jolt-poly = { path = "crates/jolt-poly" }
jolt-transcript = { path = "crates/jolt-transcript" }
bincode = "1.3"
rand = "0.8"
```

### Acceptance Criteria

- Integration test file created at workspace level
- Tests basic R1CS proving with Dory
- Tests larger systems and uniform R1CS
- Tests univariate skip optimization
- Includes stress tests (marked as ignored)
- Tests error cases
- Helper functions reduce code duplication
- All tests pass
- Performance metrics collected
- No modifications to crate source code