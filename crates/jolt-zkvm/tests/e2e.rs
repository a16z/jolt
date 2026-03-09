//! End-to-end integration tests for the Jolt proving pipeline.
//!
//! Tests exercise:
//! 1. Uniform Spartan with the actual Jolt R1CS (24 constraints × 41 vars/cycle)
//! 2. Full `prove()` pipeline with uniform Spartan + sumcheck stages + openings

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_spartan::UniformSpartanKey;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zkvm::preprocessing::{interleave_witnesses, preprocess, JoltConfig};
use jolt_zkvm::proof::JoltProof;
use jolt_zkvm::prover::prove;
use jolt_zkvm::r1cs;
use jolt_zkvm::stage::ProverStage;
use jolt_zkvm::stages::s1_spartan::{UniformSpartanResult, UniformSpartanStage};
use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

/// Build a NOP cycle witness at a given PC.
///
/// Variable 0 = constant 1, flags/operands = 0. PC update is straight-line
/// (NextUnexpandedPC = UnexpandedPC + 4) to satisfy constraint 16.
fn nop_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);
    w
}

/// Build an ADD cycle witness at a given PC.
///
/// rs1=7, rs2=3 → lookup_output=10 → rd_write=10.
/// Mirrors the `add_witness()` from `r1cs.rs` unit tests.
fn add_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);

    // Flags
    w[r1cs::V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
    w[r1cs::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
    w[r1cs::V_IS_RD_NOT_ZERO] = Fr::from_u64(1);

    // Instruction I/O
    w[r1cs::V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(7);
    w[r1cs::V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(3);
    w[r1cs::V_PRODUCT] = Fr::from_u64(21); // 7 * 3

    // Lookup operands (add mode)
    w[r1cs::V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(0);
    w[r1cs::V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(10); // 7 + 3
    w[r1cs::V_LOOKUP_OUTPUT] = Fr::from_u64(10);

    // Product-derived booleans
    w[r1cs::V_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1); // IsRdNotZero(1) * Flag(1)
    w[r1cs::V_WRITE_PC_TO_RD] = Fr::from_u64(0); // IsRdNotZero(1) * Jump(0)

    // Register write
    w[r1cs::V_RD_WRITE_VALUE] = Fr::from_u64(10); // == LookupOutput

    // PC update: straight-line
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);

    w
}

/// Commit witness and append commitment to transcript.
fn commit_and_append<PCS: CommitmentScheme<Field = Fr>>(
    flat_witness: &[Fr],
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut Blake2bTranscript,
) {
    let (commitment, _hint) = PCS::commit(flat_witness, pcs_setup);
    transcript.append_bytes(format!("{commitment:?}").as_bytes());
}

/// Run S1 (Uniform Spartan PIOP) with the Jolt R1CS.
fn run_s1_uniform(
    key: &UniformSpartanKey<Fr>,
    flat_witness: &[Fr],
    transcript: &mut Blake2bTranscript,
) -> UniformSpartanResult<Fr> {
    UniformSpartanStage::prove(key, flat_witness, flat_witness, transcript)
        .expect("S1 proving should succeed")
}

/// Build a synthetic claim reduction stage from Spartan's r_y.
fn build_reduction_stage(r_y: &[Fr], rng: &mut ChaCha20Rng) -> ClaimReductionStage<Fr> {
    let num_vars = r_y.len();
    let n = 1usize << num_vars;

    let poly_a: Vec<Fr> = (0..n).map(|_| Fr::random(rng)).collect();
    let poly_b: Vec<Fr> = (0..n).map(|_| Fr::random(rng)).collect();
    let c0 = Fr::random(rng);
    let c1 = Fr::random(rng);

    ClaimReductionStage::increment(poly_a, poly_b, r_y.to_vec(), c0, c1)
}

/// Full E2E test with the actual Jolt R1CS.
fn run_jolt_r1cs_e2e<PCS: AdditivelyHomomorphic<Field = Fr>>(
    prover_setup: &PCS::ProverSetup,
    _verifier_setup: &PCS::VerifierSetup,
) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // 4 cycles: NOP, ADD, NOP, NOP
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, PCS>(&config, |_| (prover_setup.clone(), _verifier_setup.clone()));

    let cycle_witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];
    let flat_witness = interleave_witnesses(&key.spartan_key, &cycle_witnesses);

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    // Commit witness and append to transcript before S1
    commit_and_append::<PCS>(&flat_witness, prover_setup, &mut pt);

    let spartan_result = run_s1_uniform(&key.spartan_key, &flat_witness, &mut pt);

    let r_y = spartan_result.r_y.clone();

    let reduction_stage = build_reduction_stage(&r_y, &mut rng);
    let mut stages: Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> =
        vec![Box::new(reduction_stage)];

    let commitments: Vec<PCS::Output> = Vec::new();
    let proof: JoltProof<Fr, PCS> = prove::<PCS, Blake2bTranscript>(
        spartan_result,
        &mut stages,
        &key,
        commitments,
        4,
        &mut pt,
        challenge_fn,
    );

    assert_eq!(
        proof.stage_proofs.len(),
        1,
        "one stage → one sumcheck proof"
    );
    assert!(
        !proof.opening_proofs.proofs.is_empty(),
        "should have at least one opening proof"
    );

    // Verify S1
    let mut vt = Blake2bTranscript::new(b"jolt-e2e");
    commit_and_append::<PCS>(&flat_witness, prover_setup, &mut vt);
    let _ = UniformSpartanStage::verify(&key.spartan_key, &proof.spartan_proof, &mut vt)
        .expect("S1 verification should succeed");
}

#[test]
fn e2e_jolt_r1cs_mock_pcs() {
    run_jolt_r1cs_e2e::<MockPCS>(&(), &());
}

/// Verify that the Jolt R1CS key has expected dimensions.
#[test]
fn jolt_r1cs_key_dimensions() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    assert_eq!(
        key.spartan_key.num_constraints,
        r1cs::NUM_CONSTRAINTS_PER_CYCLE
    );
    assert_eq!(key.spartan_key.num_vars, r1cs::NUM_VARS_PER_CYCLE);
    assert_eq!(key.spartan_key.num_cycles, 4);
}

/// Verify uniform Spartan prove+verify with NOP-only cycles.
#[test]
fn uniform_spartan_nop_only() {
    let config = JoltConfig { num_cycles: 2 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![nop_cycle_witness(0, 0), nop_cycle_witness(4, 1)];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"nop-only");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("NOP-only proving should succeed");

    let mut vt = Blake2bTranscript::new(b"nop-only");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("NOP-only verification should succeed");
}

/// Verify uniform Spartan prove+verify with mixed cycle types.
#[test]
fn uniform_spartan_mixed_cycles() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        add_cycle_witness(12, 3),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"mixed-cycles");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("mixed-cycle proving should succeed");

    let mut vt = Blake2bTranscript::new(b"mixed-cycles");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("mixed-cycle verification should succeed");
}

/// Verify that challenge vectors from S1 have correct dimensions.
#[test]
fn s1_challenge_vector_dimensions() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"s1-dims");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result =
        UniformSpartanStage::prove(&key, &flat, &flat, &mut pt).expect("proving should succeed");

    assert_eq!(result.r_x.len(), key.num_row_vars());
    assert_eq!(result.r_y.len(), key.num_col_vars());
    assert_eq!(result.witness_opening_claim.point, result.r_y);
    assert_eq!(result.witness_opening_claim.eval, result.proof.witness_eval);
}

mod dory {
    use super::*;
    use jolt_dory::DoryScheme;

    #[test]
    fn e2e_jolt_r1cs_dory() {
        let config = JoltConfig { num_cycles: 4 };
        let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;
        let num_vars = key.num_col_vars();

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        run_jolt_r1cs_e2e::<DoryScheme>(&prover_setup, &verifier_setup);
    }
}
