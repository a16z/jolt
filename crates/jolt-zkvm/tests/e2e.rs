//! End-to-end integration tests for the Jolt proving pipeline.
//!
//! Tests exercise:
//! 1. Uniform Spartan with the actual Jolt R1CS (24 constraints × 41 vars/cycle)
//! 2. Full `prove()` pipeline with uniform Spartan + sumcheck stages + openings
//! 3. Multiple stage configurations and error cases

use std::sync::Arc;

use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_openings::mock::{MockCommitment, MockCommitmentScheme};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::EqPolynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::StageDescriptor;
use jolt_zkvm::preprocessing::{interleave_witnesses, preprocess, JoltConfig};
use jolt_zkvm::prover::prove;
use jolt_zkvm::r1cs;
use jolt_zkvm::stage::ProverStage;
use jolt_zkvm::stages::s1_spartan::UniformSpartanStage;
use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
use rand_chacha::ChaCha20Rng;

fn cpu() -> Arc<CpuBackend> {
    Arc::new(CpuBackend)
}
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

/// NOP cycle: all zeros except constant=1 and PC update.
fn nop_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);
    w
}

/// ADD cycle: left + right → lookup output → rd_write.
///
/// Satisfies constraints: Product = left*right, LeftLookup = 0 (ADD mode),
/// RightLookup = left+right, LookupOutput = sum, RdWrite = sum.
fn add_cycle_witness(unexpanded_pc: u64, pc: u64, left: u64, right: u64) -> Vec<Fr> {
    let sum = left + right;
    let product = left * right;

    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);

    w[r1cs::V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
    w[r1cs::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
    w[r1cs::V_IS_RD_NOT_ZERO] = Fr::from_u64(1);

    w[r1cs::V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(left);
    w[r1cs::V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(right);
    w[r1cs::V_PRODUCT] = Fr::from_u64(product);

    w[r1cs::V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(0);
    w[r1cs::V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(sum);
    w[r1cs::V_LOOKUP_OUTPUT] = Fr::from_u64(sum);

    // Product-derived boolean: IsRdNotZero(1) * FlagWriteLookupOutputToRd(1) = 1
    w[r1cs::V_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);

    w[r1cs::V_RD_WRITE_VALUE] = Fr::from_u64(sum);

    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);

    w
}

/// LOAD cycle: RAM[rs1 + imm] → rd_write.
///
/// Satisfies constraints: RamAddr = Rs1+Imm, RamRead = RamWrite (load),
/// RamRead = RdWrite (load writes loaded value to register).
fn load_cycle_witness(unexpanded_pc: u64, pc: u64, rs1: u64, imm: u64, ram_value: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);

    w[r1cs::V_FLAG_LOAD] = Fr::from_u64(1);
    w[r1cs::V_IS_RD_NOT_ZERO] = Fr::from_u64(1);

    w[r1cs::V_RS1_VALUE] = Fr::from_u64(rs1);
    w[r1cs::V_IMM] = Fr::from_u64(imm);
    w[r1cs::V_RAM_ADDRESS] = Fr::from_u64(rs1 + imm);
    w[r1cs::V_RAM_READ_VALUE] = Fr::from_u64(ram_value);
    w[r1cs::V_RAM_WRITE_VALUE] = Fr::from_u64(ram_value);
    w[r1cs::V_RD_WRITE_VALUE] = Fr::from_u64(ram_value);

    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);

    w
}

/// STORE cycle: rs2 → RAM[rs1 + imm].
///
/// Satisfies constraints: RamAddr = Rs1+Imm, RamWrite = Rs2 (store).
/// No register write (IsRdNotZero = 0).
fn store_cycle_witness(unexpanded_pc: u64, pc: u64, rs1: u64, imm: u64, rs2_value: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);

    w[r1cs::V_FLAG_STORE] = Fr::from_u64(1);

    w[r1cs::V_RS1_VALUE] = Fr::from_u64(rs1);
    w[r1cs::V_IMM] = Fr::from_u64(imm);
    w[r1cs::V_RAM_ADDRESS] = Fr::from_u64(rs1 + imm);
    w[r1cs::V_RS2_VALUE] = Fr::from_u64(rs2_value);
    w[r1cs::V_RAM_WRITE_VALUE] = Fr::from_u64(rs2_value);

    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);

    w
}

/// Random polynomial pair with linear combination coefficients.
struct SyntheticReduction {
    poly_a: Vec<Fr>,
    poly_b: Vec<Fr>,
    c0: Fr,
    c1: Fr,
}

fn build_reduction_data(num_vars: usize, rng: &mut ChaCha20Rng) -> SyntheticReduction {
    let n = 1usize << num_vars;
    SyntheticReduction {
        poly_a: (0..n).map(|_| Fr::random(rng)).collect(),
        poly_b: (0..n).map(|_| Fr::random(rng)).collect(),
        c0: Fr::random(rng),
        c1: Fr::random(rng),
    }
}

fn build_prover_stage(
    data: &SyntheticReduction,
    r_y: Vec<Fr>,
) -> ClaimReductionStage<Fr, CpuBackend> {
    ClaimReductionStage::increment(
        data.poly_a.clone(),
        data.poly_b.clone(),
        r_y,
        data.c0,
        data.c1,
        cpu(),
    )
}

fn build_verifier_descriptor(
    data: &SyntheticReduction,
    eq_point: &[Fr],
    base_index: usize,
) -> StageDescriptor<Fr> {
    let n = data.poly_a.len();
    let eq_table = EqPolynomial::new(eq_point.to_vec()).evaluations();

    let claimed_sum: Fr = (0..n)
        .map(|j| eq_table[j] * (data.c0 * data.poly_a[j] + data.c1 * data.poly_b[j]))
        .sum();

    StageDescriptor::claim_reduction(
        eq_point.to_vec(),
        vec![data.c0, data.c1],
        claimed_sum,
        vec![base_index, base_index + 1],
    )
    .with_reverse_challenges()
}

/// Runs prove → verify for a given program (cycle witnesses) with one claim reduction stage.
fn run_e2e<PCS: AdditivelyHomomorphic<Field = Fr>>(
    cycle_witnesses: &[Vec<Fr>],
    setup_fn: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig {
        num_cycles: cycle_witnesses.len(),
    };
    let key = preprocess::<Fr, PCS>(&config, setup_fn);

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);

    let (com_a, _) = PCS::commit(&reduction_data.poly_a, &key.pcs_prover_setup);
    let (com_b, _) = PCS::commit(&reduction_data.poly_b, &key.pcs_prover_setup);
    let poly_commitments: Vec<PCS::Output> = vec![com_a, com_b];

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let proof = prove::<PCS, Blake2bTranscript>(
        &key,
        cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
        challenge_fn,
    )
    .expect("proving should succeed");

    assert_eq!(proof.stage_proofs.len(), 1);
    assert!(!proof.opening_proofs.is_empty());

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: key.pcs_verifier_setup.clone(),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-e2e");

    let (v_r_x, v_r_y) = jolt_verifier::verify::<PCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
        &mut vt,
        challenge_fn,
    )
    .expect("verification should succeed");

    assert_eq!(v_r_x.len(), key.spartan_key.num_row_vars());
    assert_eq!(v_r_y.len(), key.spartan_key.num_col_vars());
}

fn run_mock_e2e(cycle_witnesses: &[Vec<Fr>]) {
    run_e2e::<MockPCS>(cycle_witnesses, |_| ((), ()));
}

/// 4 NOPs: trivial straight-line execution.
#[test]
fn e2e_nop_program() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ]);
}

/// 4 ADDs with different operands.
#[test]
fn e2e_add_program() {
    run_mock_e2e(&[
        add_cycle_witness(0, 0, 7, 3),
        add_cycle_witness(4, 1, 10, 5),
        add_cycle_witness(8, 2, 100, 200),
        add_cycle_witness(12, 3, 1, 1),
    ]);
}

/// LOAD → ADD → STORE → NOP: load-compute-store pattern.
#[test]
fn e2e_load_compute_store() {
    run_mock_e2e(&[
        load_cycle_witness(0, 0, 100, 20, 42),
        add_cycle_witness(4, 1, 42, 8),
        store_cycle_witness(8, 2, 180, 20, 50),
        nop_cycle_witness(12, 3),
    ]);
}

/// All four instruction types in one program.
#[test]
fn e2e_mixed_instructions() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        load_cycle_witness(8, 2, 100, 20, 42),
        store_cycle_witness(12, 3, 180, 20, 55),
    ]);
}

/// 8-cycle mixed workload: load → compute → store, repeated.
#[test]
fn e2e_eight_cycle_program() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        load_cycle_witness(4, 1, 0, 8, 100),
        add_cycle_witness(8, 2, 100, 50),
        store_cycle_witness(12, 3, 0, 16, 150),
        load_cycle_witness(16, 4, 0, 24, 200),
        add_cycle_witness(20, 5, 200, 150),
        store_cycle_witness(24, 6, 0, 32, 350),
        nop_cycle_witness(28, 7),
    ]);
}

/// Single-cycle program (minimum size).
#[test]
fn e2e_single_cycle() {
    run_mock_e2e(&[nop_cycle_witness(0, 0)]);
}

/// Two-cycle program (non-power-of-two padded).
#[test]
fn e2e_two_cycles() {
    run_mock_e2e(&[add_cycle_witness(0, 0, 3, 4), add_cycle_witness(4, 1, 5, 6)]);
}

/// Load → compute → store with a trailing NOP (4 cycles).
#[test]
fn e2e_load_add_store_nop() {
    run_mock_e2e(&[
        load_cycle_witness(0, 0, 50, 10, 99),
        add_cycle_witness(4, 1, 99, 1),
        store_cycle_witness(8, 2, 60, 0, 100),
        nop_cycle_witness(12, 3),
    ]);
}

/// Two independent claim reduction stages through the full pipeline.
#[test]
fn e2e_multi_stage_pipeline() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let cycle_witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let data1 = build_reduction_data(num_col_vars, &mut rng);
    let data2 = build_reduction_data(num_col_vars, &mut rng);

    let (com_a1, ()) = MockPCS::commit(&data1.poly_a, &());
    let (com_b1, ()) = MockPCS::commit(&data1.poly_b, &());
    let (com_a2, ()) = MockPCS::commit(&data2.poly_a, &());
    let (com_b2, ()) = MockPCS::commit(&data2.poly_b, &());
    let poly_commitments: Vec<MockCommitment<Fr>> = vec![com_a1, com_b1, com_a2, com_b2];

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, _t| {
            let s1 = build_prover_stage(&data1, r_y.to_vec());
            let s2 = build_prover_stage(&data2, r_y.to_vec());
            vec![Box::new(s1), Box::new(s2)]
        },
        &mut pt,
        challenge_fn,
    )
    .expect("proving should succeed");

    assert_eq!(proof.stage_proofs.len(), 2);

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-e2e");

    let (v_r_x, v_r_y) = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| {
            vec![
                build_verifier_descriptor(&data1, r_y, 0),
                build_verifier_descriptor(&data2, r_y, 2),
            ]
        },
        &mut vt,
        challenge_fn,
    )
    .expect("verification should succeed");

    assert_eq!(v_r_x.len(), key.spartan_key.num_row_vars());
    assert_eq!(v_r_y.len(), key.spartan_key.num_col_vars());
}

/// Bad witness (violated constraint) causes prover to fail.
#[test]
fn e2e_bad_witness_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    // Corrupt the ADD witness: wrong product value violates constraint 19.
    let mut bad_add = add_cycle_witness(4, 1, 7, 3);
    bad_add[r1cs::V_PRODUCT] = Fr::from_u64(999);

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        bad_add,
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
        challenge_fn,
    );

    assert!(result.is_err(), "proving with bad witness should fail");
}

/// Tampered proof is rejected by the verifier.
#[test]
fn e2e_tampered_proof_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let mut proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
        challenge_fn,
    )
    .expect("proving should succeed");

    // Tamper: shift witness_eval to invalidate the Spartan inner sumcheck check.
    proof.spartan_proof.witness_eval += Fr::from_u64(1);

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-e2e");

    let result = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
        &mut vt,
        challenge_fn,
    );

    assert!(result.is_err(), "tampered proof should fail verification");
}

/// Helper to commit and append witness to transcript (for Spartan-only tests).
fn commit_and_append<PCS: CommitmentScheme<Field = Fr>>(
    flat_witness: &[Fr],
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut Blake2bTranscript,
) {
    let (commitment, _hint) = PCS::commit(flat_witness, pcs_setup);
    transcript.append_bytes(format!("{commitment:?}").as_bytes());
}

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

#[test]
fn uniform_spartan_mixed_cycles() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        add_cycle_witness(12, 3, 10, 5),
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

/// Spartan-only: LOAD and STORE cycles.
#[test]
fn uniform_spartan_load_store() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        load_cycle_witness(0, 0, 100, 20, 42),
        store_cycle_witness(4, 1, 200, 50, 42),
        load_cycle_witness(8, 2, 300, 0, 77),
        store_cycle_witness(12, 3, 400, 10, 77),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"load-store");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("LOAD/STORE proving should succeed");

    let mut vt = Blake2bTranscript::new(b"load-store");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("LOAD/STORE verification should succeed");
}

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

#[test]
fn claim_reduction_prove_verify_standalone() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let num_vars = 4;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let c0 = Fr::random(&mut rng);
    let c1 = Fr::random(&mut rng);

    let poly_a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let poly_b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = (0..n)
        .map(|j| eq_table[j] * (c0 * poly_a[j] + c1 * poly_b[j]))
        .sum();

    let prover_stage =
        ClaimReductionStage::increment(poly_a, poly_b, eq_point.clone(), c0, c1, cpu());
    let mut prover_stages: Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> =
        vec![Box::new(prover_stage)];

    let mut pt = Blake2bTranscript::new(b"cr-standalone");
    let (stage_proofs, _opening_claims) =
        jolt_zkvm::pipeline::prove_stages(&mut prover_stages, &mut pt, challenge_fn);

    assert_eq!(stage_proofs.len(), 1);

    let desc = StageDescriptor::claim_reduction(eq_point, vec![c0, c1], claimed_sum, vec![0, 1])
        .with_reverse_challenges();

    let stage_proof = &stage_proofs[0];

    let mut vt = Blake2bTranscript::new(b"cr-standalone");

    // Build and verify sumcheck claim.
    let claims = [jolt_sumcheck::SumcheckClaim {
        num_vars: desc.num_vars,
        degree: desc.degree,
        claimed_sum: desc.claimed_sum,
    }];

    let (final_eval, challenges) = jolt_sumcheck::BatchedSumcheckVerifier::verify(
        &claims,
        &stage_proof.sumcheck_proof,
        &mut vt,
        challenge_fn,
    )
    .expect("sumcheck verification should succeed");

    // LowToHigh: eval_point = challenges.reverse()
    let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
    let eq_eval = EqPolynomial::new(desc.eq_point.clone()).evaluate(&eval_point);
    let g_eval: Fr = desc
        .output_expr
        .evaluate(&stage_proof.evaluations, &desc.output_challenges);
    let expected = eq_eval * g_eval;

    assert_eq!(expected, final_eval, "eq * g should match final_eval");
    assert_eq!(stage_proof.evaluations.len(), 2);
}

mod trace_based {
    use super::*;
    use jolt_zkvm::witness::generate_witnesses;
    use tracer::instruction::{
        add::ADD,
        format::format_j::{FormatJ, RegisterStateFormatJ},
        format::format_r::{FormatR, RegisterStateFormatR},
        jal::JAL,
        sub::SUB,
        RISCVCycle,
    };

    fn make_add_cycle(addr: u64, rs1: u64, rs2: u64) -> tracer::instruction::Cycle {
        tracer::instruction::Cycle::from(RISCVCycle {
            instruction: ADD {
                address: addr,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                ..ADD::default()
            },
            register_state: RegisterStateFormatR {
                rd: (0, rs1.wrapping_add(rs2)),
                rs1,
                rs2,
            },
            ram_access: (),
        })
    }

    fn make_sub_cycle(addr: u64, rs1: u64, rs2: u64) -> tracer::instruction::Cycle {
        tracer::instruction::Cycle::from(RISCVCycle {
            instruction: SUB {
                address: addr,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                ..SUB::default()
            },
            register_state: RegisterStateFormatR {
                rd: (0, rs1.wrapping_sub(rs2)),
                rs1,
                rs2,
            },
            ram_access: (),
        })
    }

    /// JAL-to-self terminal cycle (rd=0, imm=0).
    /// Sets FLAG_JUMP=1 which zeros constraint 16's guard, allowing
    /// NextUnexpPC=0 when this is the last cycle (next=None).
    fn make_jal_terminal(addr: u64) -> tracer::instruction::Cycle {
        tracer::instruction::Cycle::from(RISCVCycle {
            instruction: JAL {
                address: addr,
                operands: FormatJ { rd: 0, imm: 0 },
                ..JAL::default()
            },
            register_state: RegisterStateFormatJ { rd: (0, 0) },
            ram_access: (),
        })
    }

    /// Runs generate_witnesses → prove → verify for a trace.
    fn run_trace_e2e(trace: Vec<tracer::instruction::Cycle>, label: &'static [u8]) {
        let output = generate_witnesses::<Fr>(&trace);

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let config = JoltConfig {
            num_cycles: output.cycle_witnesses.len(),
        };
        let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

        let num_col_vars = key.spartan_key.num_col_vars();
        let reduction_data = build_reduction_data(num_col_vars, &mut rng);

        let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
        let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

        let mut pt = Blake2bTranscript::new(label);

        let proof = prove::<MockPCS, Blake2bTranscript>(
            &key,
            &output.cycle_witnesses,
            vec![com_a, com_b],
            |_r_x, r_y, _t| {
                let stage = build_prover_stage(&reduction_data, r_y.to_vec());
                vec![Box::new(stage)]
            },
            &mut pt,
            challenge_fn,
        )
        .expect("trace-based proving should succeed");

        assert_eq!(proof.stage_proofs.len(), 1);

        let vk = jolt_verifier::JoltVerifyingKey {
            spartan_key: key.spartan_key.clone(),
            pcs_setup: (),
        };

        let mut vt = Blake2bTranscript::new(label);

        let _ = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
            &proof,
            &vk,
            |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
            &mut vt,
            challenge_fn,
        )
        .expect("trace-based verification should succeed");
    }

    /// ADDs terminated by JAL-to-self: straight-line compute with valid termination.
    #[test]
    fn trace_add_pipeline() {
        run_trace_e2e(
            vec![
                make_add_cycle(0x1000, 3, 4),
                make_add_cycle(0x1004, 10, 20),
                make_add_cycle(0x1008, 100, 200),
                make_jal_terminal(0x100C),
            ],
            b"jolt-trace-add",
        );
    }

    /// Two NOOPs as Cycle::NoOp through the full pipeline.
    #[test]
    fn trace_noop_pipeline() {
        run_trace_e2e(
            vec![
                tracer::instruction::Cycle::NoOp,
                tracer::instruction::Cycle::NoOp,
            ],
            b"jolt-trace-noop",
        );
    }

    /// Mixed instruction types (ADD + SUB) with valid PC sequencing and JAL termination.
    #[test]
    fn trace_mixed_pipeline() {
        run_trace_e2e(
            vec![
                make_add_cycle(0x1000, 7, 3),
                make_sub_cycle(0x1004, 42, 8),
                make_add_cycle(0x1008, 1, 2),
                make_jal_terminal(0x100C),
            ],
            b"jolt-trace-mixed",
        );
    }
}

mod dory {
    use super::*;
    use jolt_dory::DoryScheme;

    #[test]
    fn e2e_dory_mixed() {
        run_e2e::<DoryScheme>(
            &[
                nop_cycle_witness(0, 0),
                add_cycle_witness(4, 1, 7, 3),
                nop_cycle_witness(8, 2),
                nop_cycle_witness(12, 3),
            ],
            |num_vars| {
                let prover_setup = DoryScheme::setup_prover(num_vars);
                let verifier_setup = DoryScheme::setup_verifier(num_vars);
                (prover_setup, verifier_setup)
            },
        );
    }

    #[test]
    fn e2e_dory_load_store() {
        run_e2e::<DoryScheme>(
            &[
                load_cycle_witness(0, 0, 100, 20, 42),
                add_cycle_witness(4, 1, 42, 8),
                store_cycle_witness(8, 2, 180, 20, 50),
                nop_cycle_witness(12, 3),
            ],
            |num_vars| {
                let prover_setup = DoryScheme::setup_prover(num_vars);
                let verifier_setup = DoryScheme::setup_verifier(num_vars);
                (prover_setup, verifier_setup)
            },
        );
    }
}

mod real_program {
    use super::*;
    use jolt_host::Program;
    use jolt_zkvm::witness::generate_witnesses;

    /// Runs trace → generate_witnesses → prove → verify for a real guest program.
    fn run_real_program_e2e(guest_name: &str, inputs: &[u8], label: &'static [u8]) {
        let mut program = Program::new(guest_name);

        let (_, trace, _, _io_device) = program.trace(inputs, &[], &[]);

        // generate_witnesses pads the trace to power-of-two with Cycle::NoOp internally
        let output = generate_witnesses::<Fr>(&trace);

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let config = JoltConfig {
            num_cycles: output.cycle_witnesses.len(),
        };
        let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

        let num_col_vars = key.spartan_key.num_col_vars();
        let reduction_data = build_reduction_data(num_col_vars, &mut rng);

        let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
        let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

        let mut pt = Blake2bTranscript::new(label);

        let proof = prove::<MockPCS, Blake2bTranscript>(
            &key,
            &output.cycle_witnesses,
            vec![com_a, com_b],
            |_r_x, r_y, _t| {
                let stage = build_prover_stage(&reduction_data, r_y.to_vec());
                vec![Box::new(stage)]
            },
            &mut pt,
            challenge_fn,
        )
        .expect("proving should succeed");

        assert_eq!(proof.stage_proofs.len(), 1);

        let vk = jolt_verifier::JoltVerifyingKey {
            spartan_key: key.spartan_key.clone(),
            pcs_setup: (),
        };

        let mut vt = Blake2bTranscript::new(label);

        let _ = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
            &proof,
            &vk,
            |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
            &mut vt,
            challenge_fn,
        )
        .expect("verification should succeed");
    }

    /// Debug test: just Spartan prove/verify (no stages) on real trace.
    #[test]
    fn muldiv_spartan_only() {
        use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
        use jolt_transcript::Transcript;

        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        let mut program = Program::new("muldiv-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);
        let output = generate_witnesses::<Fr>(&trace);

        eprintln!("num_cycles = {}", output.cycle_witnesses.len());
        let key = r1cs::build_jolt_spartan_key::<Fr>(output.cycle_witnesses.len());

        let total_cols_padded = key.total_cols().next_power_of_two();
        let mut flat = vec![Fr::from_u64(0); total_cols_padded];
        for (c, w) in output.cycle_witnesses.iter().enumerate() {
            let base = c * key.num_vars_padded;
            for (v, &val) in w.iter().enumerate().take(key.num_vars) {
                flat[base + v] = val;
            }
        }

        let (commitment, ()) = MockPCS::commit(&flat, &());

        let mut pt = Blake2bTranscript::new(b"spartan-only");
        pt.append_bytes(format!("{commitment:?}").as_bytes());
        let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut pt)
            .expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"spartan-only");
        vt.append_bytes(format!("{commitment:?}").as_bytes());
        UniformSpartanVerifier::verify(&key, &proof, &mut vt).expect("verification should succeed");
    }

    /// Bisect: find the cycle count where Spartan starts failing.
    #[test]
    fn nop_spartan_size_bisect() {
        use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
        use jolt_transcript::Transcript;
        use jolt_zkvm::witness::bytecode::BytecodePreprocessing;
        use jolt_zkvm::witness::r1cs_inputs;

        for &n in &[2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let trace: Vec<tracer::instruction::Cycle> = vec![tracer::instruction::Cycle::NoOp; n];
            let bytecode = BytecodePreprocessing::new(&trace);
            let witnesses: Vec<Vec<Fr>> = trace
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let next = trace.get(i + 1);
                    r1cs_inputs::cycle_to_witness(c, next, &bytecode)
                })
                .collect();

            let key = r1cs::build_jolt_spartan_key::<Fr>(n);
            let total_cols_padded = key.total_cols().next_power_of_two();
            let mut flat = vec![Fr::from_u64(0); total_cols_padded];
            for (c, w) in witnesses.iter().enumerate() {
                let base = c * key.num_vars_padded;
                for (v, &val) in w.iter().enumerate().take(key.num_vars) {
                    flat[base + v] = val;
                }
            }

            let (commitment, ()) = MockPCS::commit(&flat, &());

            let mut pt = Blake2bTranscript::new(b"nop-bisect");
            pt.append_bytes(format!("{commitment:?}").as_bytes());
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut pt)
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"nop-bisect");
            vt.append_bytes(format!("{commitment:?}").as_bytes());
            let result = UniformSpartanVerifier::verify(&key, &proof, &mut vt);
            eprintln!("n={n}: {}", if result.is_ok() { "PASS" } else { "FAIL" });
        }
    }

    /// Debug test: checks every cycle's witness against all 24 R1CS constraints.
    #[test]
    fn muldiv_r1cs_satisfaction() {
        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        let mut program = Program::new("muldiv-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);
        let output = generate_witnesses::<Fr>(&trace);

        let key = r1cs::build_jolt_spartan_key::<Fr>(output.cycle_witnesses.len());
        let mut violations = Vec::new();

        for (cycle_idx, w) in output.cycle_witnesses.iter().enumerate() {
            for k in 0..r1cs::NUM_CONSTRAINTS_PER_CYCLE {
                let a_val: Fr = key.a_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                let b_val: Fr = key.b_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                let c_val: Fr = key.c_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                if a_val * b_val != c_val {
                    violations.push((cycle_idx, k));
                }
            }
        }

        if !violations.is_empty() {
            let total = output.cycle_witnesses.len();
            // Show first 20 violations
            for &(c, k) in violations.iter().take(20) {
                let w = &output.cycle_witnesses[c];
                eprintln!(
                    "cycle {c}/{total} constraint {k}: flags=[add={}, sub={}, mul={}, load={}, store={}, jump={}, doNotUpdate={}, isNoop(next)={}] unexpPC={:?} nextUnexpPC={:?}",
                    w[r1cs::V_FLAG_ADD_OPERANDS],
                    w[r1cs::V_FLAG_SUBTRACT_OPERANDS],
                    w[r1cs::V_FLAG_MULTIPLY_OPERANDS],
                    w[r1cs::V_FLAG_LOAD],
                    w[r1cs::V_FLAG_STORE],
                    w[r1cs::V_FLAG_JUMP],
                    w[r1cs::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC],
                    w[r1cs::V_NEXT_IS_NOOP],
                    w[r1cs::V_UNEXPANDED_PC],
                    w[r1cs::V_NEXT_UNEXPANDED_PC],
                );
            }
            panic!(
                "{} R1CS violations in {} cycles (first: cycle {} constraint {})",
                violations.len(),
                total,
                violations[0].0,
                violations[0].1
            );
        }
    }

    /// Compiles the muldiv guest via `jolt` CLI, traces execution with real inputs,
    /// runs generate_witnesses → prove → verify through the Spartan pipeline.
    ///
    /// This is the primary real-program correctness check: it exercises the full
    /// path from RISC-V ELF → tracer → witness gen → R1CS → Spartan prove/verify.
    #[test]
    fn muldiv_trace_spartan() {
        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        run_real_program_e2e("muldiv-guest", &inputs, b"jolt-muldiv");
    }

    /// Fibonacci guest: exercises longer traces with loop-heavy execution.
    #[test]
    fn fibonacci_trace_spartan() {
        let inputs = postcard::to_stdvec(&10u32).unwrap();
        run_real_program_e2e("fibonacci-guest", &inputs, b"jolt-fib");
    }
}

/// Host-layer E2E tests using [`prove_trace`] + [`verify_proof`].
///
/// These exercise the unified pipeline: trace → witness gen → preprocess →
/// commit real polynomials → prove (with real S3 increment reduction) → verify.
mod host_pipeline {
    use jolt_host::Program;
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_field::Fr;
    use jolt_zkvm::host::{prove_trace, verify_proof};

    type MockPCS = MockCommitmentScheme<Fr>;

    /// Full pipeline: muldiv trace → prove_trace → verify_proof with MockPCS.
    #[test]
    fn muldiv_host_pipeline() {
        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        let mut program = Program::new("muldiv-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);

        let output = prove_trace::<MockPCS>(&trace, |_| ((), ()))
            .expect("prove_trace should succeed");

        verify_proof::<MockPCS>(&output)
            .expect("verify_proof should succeed");
    }

    /// Full pipeline: fibonacci trace → prove_trace → verify_proof with MockPCS.
    #[test]
    fn fibonacci_host_pipeline() {
        let inputs = postcard::to_stdvec(&10u32).unwrap();
        let mut program = Program::new("fibonacci-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);

        let output = prove_trace::<MockPCS>(&trace, |_| ((), ()))
            .expect("prove_trace should succeed");

        verify_proof::<MockPCS>(&output)
            .expect("verify_proof should succeed");
    }
}
