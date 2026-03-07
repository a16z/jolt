//! End-to-end integration: SymbolicField → AstBundle → SpartanAstEmitter →
//! SpartanCircuit → SimpleR1CS → SpartanKey → prove → verify.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_spartan::{
    FirstRoundStrategy, SimpleR1CS, SpartanError, SpartanKey, SpartanProver, SpartanVerifier,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_wrapper::arena::ArenaSession;
use jolt_wrapper::bundle::VarAllocator;
use jolt_wrapper::spartan::SpartanAstEmitter;
use jolt_wrapper::symbolic::SymbolicField;

type MockPCS = MockCommitmentScheme<Fr>;

/// Converts a `SpartanCircuit<Fr>` into a `SimpleR1CS<Fr>` and runs
/// `SpartanProver::prove` → `SpartanVerifier::verify`.
fn prove_and_verify(
    circuit: &jolt_wrapper::spartan::SpartanCircuit<Fr>,
    witness: &[Fr],
    label: &'static [u8],
) {
    let (a, b, c) = circuit.sparse_entries();
    let r1cs = SimpleR1CS::new(
        circuit.num_constraints(),
        circuit.num_variables as usize,
        a,
        b,
        c,
    );
    let key = SpartanKey::from_r1cs(&r1cs);

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof = SpartanProver::prove::<MockPCS, _>(
        &r1cs,
        &key,
        witness,
        &(),
        &mut prover_transcript,
        FirstRoundStrategy::Standard,
    )
    .expect("proving should succeed");

    let mut verifier_transcript = Blake2bTranscript::new(label);
    SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut verifier_transcript)
        .expect("verification should succeed");
}

/// Same as `prove_and_verify` but expects the prover to fail with a
/// constraint violation.
fn prove_should_fail(
    circuit: &jolt_wrapper::spartan::SpartanCircuit<Fr>,
    witness: &[Fr],
    label: &'static [u8],
) {
    let (a, b, c) = circuit.sparse_entries();
    let r1cs = SimpleR1CS::new(
        circuit.num_constraints(),
        circuit.num_variables as usize,
        a,
        b,
        c,
    );
    let key = SpartanKey::from_r1cs(&r1cs);

    let mut transcript = Blake2bTranscript::new(label);
    let result = SpartanProver::prove::<MockPCS, _>(
        &r1cs,
        &key,
        witness,
        &(),
        &mut transcript,
        FirstRoundStrategy::Standard,
    );
    assert!(
        matches!(result, Err(SpartanError::ConstraintViolation(_))),
        "expected constraint violation, got Ok or different error"
    );
}


#[test]
fn spartan_e2e_simple_mul() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero((x * y).into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // x=0, y=42 → 0*42 = 0 ✓
    let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(42))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-simple-mul");
}

#[test]
fn spartan_e2e_simple_mul_unsatisfied() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero((x * y).into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // x=3, y=7 → 21 ≠ 0 ✗
    let witness = circuit.build_witness(&[(0, Fr::from_u64(3)), (1, Fr::from_u64(7))]);
    assert!(!circuit.is_satisfied(&witness));
    prove_should_fail(&circuit, &witness, b"wrapper-simple-mul-bad");
}

#[test]
fn spartan_e2e_booleanity() {
    let _session = ArenaSession::new();

    let h = SymbolicField::variable(0, "H");
    let gamma = SymbolicField::variable(1, "gamma");
    let constraint = gamma * (h * h - h);

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("H");
    let _ = alloc.input("gamma");
    alloc.assert_zero(constraint.into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // H=1 is boolean → satisfied for any gamma
    let witness = circuit.build_witness(&[(0, Fr::from_u64(1)), (1, Fr::from_u64(99))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-bool-h1");

    // H=0 is boolean → satisfied
    let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(99))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-bool-h0");
}

#[test]
fn spartan_e2e_booleanity_violated() {
    let _session = ArenaSession::new();

    let h = SymbolicField::variable(0, "H");
    let gamma = SymbolicField::variable(1, "gamma");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("H");
    let _ = alloc.input("gamma");
    alloc.assert_zero((gamma * (h * h - h)).into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // H=2 is not boolean, gamma=1 → 1*(4-2)=2 ≠ 0
    let witness = circuit.build_witness(&[(0, Fr::from_u64(2)), (1, Fr::from_u64(1))]);
    assert!(!circuit.is_satisfied(&witness));
    prove_should_fail(&circuit, &witness, b"wrapper-bool-bad");
}

#[test]
fn spartan_e2e_assert_equal() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_equal(x.into_edge(), y.into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    let witness = circuit.build_witness(&[(0, Fr::from_u64(42)), (1, Fr::from_u64(42))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-eq");
}

#[test]
fn spartan_e2e_chained_mul() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let z = SymbolicField::variable(2, "z");
    let expected = SymbolicField::variable(3, "expected");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    let _ = alloc.input("z");
    let _ = alloc.input("expected");
    alloc.assert_equal((x * y * z).into_edge(), expected.into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // 2*3*5 = 30
    let witness = circuit.build_witness(&[
        (0, Fr::from_u64(2)),
        (1, Fr::from_u64(3)),
        (2, Fr::from_u64(5)),
        (3, Fr::from_u64(30)),
    ]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-chain-mul");
}

#[test]
fn spartan_e2e_weighted_sum() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let three = SymbolicField::from_u64(3);
    let seven = SymbolicField::from_u64(7);
    let forty_two = SymbolicField::from_u64(42);
    let constraint = three * x + seven * y - forty_two;

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero(constraint.into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // 3*7 + 7*3 = 21+21 = 42 ✓
    let witness = circuit.build_witness(&[(0, Fr::from_u64(7)), (1, Fr::from_u64(3))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-weighted-sum");
}

#[test]
fn spartan_e2e_randomized() {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    let mut rng = ChaCha8Rng::seed_from_u64(0xcafe_babe);

    for i in 0..10 {
        let _session = ArenaSession::new();

        let a_val = Fr::random(&mut rng);
        let b_val = Fr::random(&mut rng);
        let c_val = a_val * b_val;

        let a = SymbolicField::variable(0, "a");
        let b = SymbolicField::variable(1, "b");
        let c = SymbolicField::variable(2, "c");

        let mut alloc = VarAllocator::new();
        let _ = alloc.input("a");
        let _ = alloc.input("b");
        let _ = alloc.input("c");
        alloc.assert_zero((a * b - c).into_edge());
        let bundle = alloc.finish();

        let mut emitter = SpartanAstEmitter::<Fr>::new();
        bundle.emit(&mut emitter);
        let circuit = emitter.finish();

        let witness = circuit.build_witness(&[(0, a_val), (1, b_val), (2, c_val)]);
        assert!(
            circuit.is_satisfied(&witness),
            "iteration {i} should satisfy"
        );

        // Use a unique label per iteration for independent Fiat-Shamir transcripts
        let label: &[u8] = b"wrapper-rand";
        prove_and_verify(&circuit, &witness, label);
    }
}

#[test]
fn spartan_e2e_multiple_assertions() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    // x * y = 0
    alloc.assert_zero((x * y).into_edge());
    // x = y
    alloc.assert_equal(x.into_edge(), y.into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // x=0, y=0 satisfies both
    let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(0))]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-multi-assert");
}

#[test]
fn spartan_e2e_computed_assert_equal() {
    let _session = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let z = SymbolicField::variable(2, "z");
    let w = SymbolicField::variable(3, "w");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    let _ = alloc.input("z");
    let _ = alloc.input("w");
    alloc.assert_equal((x * y).into_edge(), (z * w).into_edge());
    let bundle = alloc.finish();

    let mut emitter = SpartanAstEmitter::<Fr>::new();
    bundle.emit(&mut emitter);
    let circuit = emitter.finish();

    // 3*7 = 21 = 21*1
    let witness = circuit.build_witness(&[
        (0, Fr::from_u64(3)),
        (1, Fr::from_u64(7)),
        (2, Fr::from_u64(21)),
        (3, Fr::from_u64(1)),
    ]);
    assert!(circuit.is_satisfied(&witness));
    prove_and_verify(&circuit, &witness, b"wrapper-computed-eq");
}
