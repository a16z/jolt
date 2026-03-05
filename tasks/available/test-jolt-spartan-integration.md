# test-jolt-spartan-integration: Integration tests for jolt-spartan

**Scope:** crates/jolt-spartan/tests/

**Depends:** impl-jolt-spartan, test-jolt-spartan

**Verifier:** ./verifiers/scoped.sh /workdir jolt-spartan

**Context:**

Write integration tests for the `jolt-spartan` crate that verify the R1CS proving system from an external user's perspective.

### Integration Test Files

Create the following test files in `crates/jolt-spartan/tests/`:

#### 1. `r1cs_proving.rs` - Test R1CS satisfaction proving

Test end-to-end R1CS proving and verification:
- Simple R1CS instances (e.g., x² = y)
- Complex R1CS with many constraints
- Test with satisfying and non-satisfying witnesses
- Verify proof serialization/deserialization
- Test with different commitment schemes

#### 2. `uniform_r1cs.rs` - Test uniform R1CS

Test the uniform (structured) R1CS functionality:
- Create uniform R1CS instances
- Verify efficiency gains vs general R1CS
- Test repeating constraint patterns
- Verify key generation for uniform structures

#### 3. `uni_skip.rs` - Test univariate skip optimization

Test the univariate skip optimization:
- Compare standard vs uni-skip proving time
- Verify same proof/verification result
- Test with different R1CS sizes
- Measure actual performance improvement

### Implementation Examples

**Simple R1CS Instance:**
```rust
/// R1CS for x² = y (single constraint)
struct SquareR1CS;

impl<F: Field> R1CS<F> for SquareR1CS {
    fn num_constraints(&self) -> usize { 1 }
    fn num_variables(&self) -> usize { 3 } // z = [1, x, y]

    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        // A: selects x (index 1)
        let az = vec![witness[1]];
        // B: selects x (index 1)
        let bz = vec![witness[1]];
        // C: selects y (index 2)
        let cz = vec![witness[2]];

        (az, bz, cz)
    }
}

#[test]
fn test_square_r1cs_proving() {
    let r1cs = SquareR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);

    // Satisfying witness: x=3, y=9
    let witness = vec![F::from(1), F::from(3), F::from(9)];

    // Setup mock PCS
    let pcs_setup = MockPCS::setup_prover(10);
    let mut transcript = TestTranscript::new();

    // Prove
    let proof = SpartanProver::prove::<F, MockPCS>(
        &key,
        &witness,
        &pcs_setup,
        &mut transcript
    ).unwrap();

    // Verify
    let verifier_setup = MockPCS::setup_verifier(10);
    let mut verify_transcript = TestTranscript::new();

    let result = SpartanVerifier::verify::<F, MockPCS>(
        &key,
        &proof,
        &verifier_setup,
        &mut verify_transcript
    );

    assert!(result.is_ok());
}

#[test]
fn test_unsatisfying_witness_fails() {
    let r1cs = SquareR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);

    // Unsatisfying witness: x=3, y=10 (not 9)
    let witness = vec![F::from(1), F::from(3), F::from(10)];

    let pcs_setup = MockPCS::setup_prover(10);
    let mut transcript = TestTranscript::new();

    // Prove should fail
    let result = SpartanProver::prove::<F, MockPCS>(
        &key,
        &witness,
        &pcs_setup,
        &mut transcript
    );

    assert!(matches!(result, Err(SpartanError::ConstraintViolation(_))));
}
```

**Complex R1CS:**
```rust
/// R1CS for system: x + y = z, x * y = w
struct SystemR1CS;

impl<F: Field> R1CS<F> for SystemR1CS {
    fn num_constraints(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 5 } // z = [1, x, y, z, w]

    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        // First constraint: (x + y) * 1 = z
        // A = [x + y], B = [1], C = [z]

        // Second constraint: x * y = w
        // A = [x], B = [y], C = [w]

        let az = vec![witness[1] + witness[2], witness[1]];
        let bz = vec![F::one(), witness[2]];
        let cz = vec![witness[3], witness[4]];

        (az, bz, cz)
    }
}
```

**Uniform R1CS Test:**
```rust
#[test]
fn test_uniform_r1cs_efficiency() {
    // Create a uniform R1CS with repeating structure
    let block_size = 4;
    let num_blocks = 100;

    let uniform_r1cs = UniformR1CS::<F>::new(
        block_size,
        num_blocks,
        |block_idx| {
            // Define repeating constraint pattern
            // ...
        }
    );

    let key = SpartanKey::from_r1cs(&uniform_r1cs);

    // Measure proving time for uniform vs general R1CS
    // Verify uniform is more efficient
}
```

**Univariate Skip Test:**
```rust
#[test]
fn test_univariate_skip_optimization() {
    let r1cs = create_large_r1cs(1000); // 1000 constraints
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = create_satisfying_witness(&r1cs);

    // Prove with standard strategy
    let start = Instant::now();
    let proof_standard = prove_with_strategy(
        &key,
        &witness,
        FirstRoundStrategy::Standard
    );
    let time_standard = start.elapsed();

    // Prove with univariate skip
    let start = Instant::now();
    let proof_skip = prove_with_strategy(
        &key,
        &witness,
        FirstRoundStrategy::UnivariateSkip { domain_size: 256 }
    );
    let time_skip = start.elapsed();

    // Verify both proofs are valid
    assert!(verify_proof(&key, &proof_standard).is_ok());
    assert!(verify_proof(&key, &proof_skip).is_ok());

    // Skip should be faster
    assert!(time_skip < time_standard);
}
```

**Proof Serialization:**
```rust
#[test]
fn test_proof_serialization() {
    let r1cs = SquareR1CS;
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = vec![F::from(1), F::from(5), F::from(25)];

    // Generate proof
    let proof = generate_proof(&key, &witness);

    // Serialize
    let serialized = serde_json::to_string(&proof).unwrap();

    // Deserialize
    let deserialized: SpartanProof<F, MockPCS> =
        serde_json::from_str(&serialized).unwrap();

    // Verify deserialized proof
    assert!(verify_proof(&key, &deserialized).is_ok());
}
```

### Test Cases to Cover

- Small R1CS instances (< 10 constraints)
- Medium R1CS instances (100-1000 constraints)
- Edge cases: 0 constraints, 1 variable
- Different witness sizes
- Proof size scaling with R1CS size
- Integration with real commitment schemes (when available)

### Acceptance Criteria

- Three integration test files created
- Various R1CS instances tested
- Both satisfying and unsatisfying witnesses tested
- Uniform R1CS functionality verified
- Univariate skip optimization tested
- Proof serialization tested
- Performance comparisons included
- All tests pass with `cargo nextest run -p jolt-spartan`
- Well-documented test cases
- No source code modifications