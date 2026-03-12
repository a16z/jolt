//! Shared test utilities for jolt-zkvm integration tests.

#![allow(dead_code)]

use std::sync::Arc;

use jolt_cpu::CpuBackend;
pub use jolt_field::{Field, Fr};
pub use jolt_openings::mock::MockCommitmentScheme;
pub use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::EqPolynomial;
pub use jolt_transcript::{Blake2bTranscript, Transcript};
pub use jolt_verifier::StageDescriptor;
pub use jolt_zkvm::preprocessing::{preprocess, JoltConfig};
pub use jolt_zkvm::prover::prove;
pub use jolt_zkvm::r1cs;
pub use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
pub use rand_chacha::ChaCha20Rng;
pub use rand_core::SeedableRng;

pub type MockPCS = MockCommitmentScheme<Fr>;

pub fn cpu() -> Arc<CpuBackend> {
    Arc::new(CpuBackend)
}

// ── Synthetic witness builders ──────────────────────────────────────────

/// NOP cycle: all zeros except constant=1 and PC update.
pub fn nop_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);
    w
}

/// ADD cycle: left + right → lookup output → rd_write.
pub fn add_cycle_witness(unexpanded_pc: u64, pc: u64, left: u64, right: u64) -> Vec<Fr> {
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

    w[r1cs::V_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
    w[r1cs::V_RD_WRITE_VALUE] = Fr::from_u64(sum);

    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);

    w
}

/// LOAD cycle: RAM[rs1 + imm] → rd_write.
pub fn load_cycle_witness(
    unexpanded_pc: u64,
    pc: u64,
    rs1: u64,
    imm: u64,
    ram_value: u64,
) -> Vec<Fr> {
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
pub fn store_cycle_witness(
    unexpanded_pc: u64,
    pc: u64,
    rs1: u64,
    imm: u64,
    rs2_value: u64,
) -> Vec<Fr> {
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

// ── Synthetic claim reduction data ──────────────────────────────────────

pub struct SyntheticReduction {
    pub poly_a: Vec<Fr>,
    pub poly_b: Vec<Fr>,
    pub c0: Fr,
    pub c1: Fr,
}

pub fn build_reduction_data(num_vars: usize, rng: &mut ChaCha20Rng) -> SyntheticReduction {
    let n = 1usize << num_vars;
    SyntheticReduction {
        poly_a: (0..n).map(|_| Fr::random(rng)).collect(),
        poly_b: (0..n).map(|_| Fr::random(rng)).collect(),
        c0: Fr::random(rng),
        c1: Fr::random(rng),
    }
}

pub fn build_prover_stage(
    data: &SyntheticReduction,
    r_y: Vec<Fr>,
) -> ClaimReductionStage<Fr, CpuBackend> {
    build_prover_stage_with_backend(data, r_y, cpu())
}

pub fn build_prover_stage_with_backend<B: jolt_compute::ComputeBackend>(
    data: &SyntheticReduction,
    r_y: Vec<Fr>,
    backend: Arc<B>,
) -> ClaimReductionStage<Fr, B> {
    ClaimReductionStage::increment(
        data.poly_a.clone(),
        data.poly_b.clone(),
        r_y,
        data.c0,
        data.c1,
        backend,
    )
}

pub fn build_verifier_descriptor(
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

// ── Full pipeline runners ───────────────────────────────────────────────

/// Runs prove → verify for given cycle witnesses with one claim reduction stage.
pub fn run_e2e<PCS: AdditivelyHomomorphic<Field = Fr>>(
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
    )
    .expect("verification should succeed");

    assert_eq!(v_r_x.len(), key.spartan_key.num_row_vars());
    assert_eq!(v_r_y.len(), key.spartan_key.num_col_vars());
}

pub fn run_mock_e2e(cycle_witnesses: &[Vec<Fr>]) {
    run_e2e::<MockPCS>(cycle_witnesses, |_| ((), ()));
}

/// Commits a witness and appends commitment to transcript (for Spartan-only tests).
pub fn commit_and_append<PCS: CommitmentScheme<Field = Fr>>(
    flat_witness: &[Fr],
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut Blake2bTranscript,
) {
    let (commitment, _hint) = PCS::commit(flat_witness, pcs_setup);
    transcript.append_bytes(format!("{commitment:?}").as_bytes());
}

// ── Tracer cycle builders ───────────────────────────────────────────────

use tracer::instruction::{
    add::ADD,
    format::format_j::{FormatJ, RegisterStateFormatJ},
    format::format_r::{FormatR, RegisterStateFormatR},
    jal::JAL,
    sub::SUB,
    RISCVCycle,
};

pub fn make_add_cycle(addr: u64, rs1: u64, rs2: u64) -> tracer::instruction::Cycle {
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

pub fn make_sub_cycle(addr: u64, rs1: u64, rs2: u64) -> tracer::instruction::Cycle {
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
/// FLAG_JUMP=1 zeros constraint 16's guard, allowing NextUnexpPC=0
/// when this is the last cycle (next=None).
pub fn make_jal_terminal(addr: u64) -> tracer::instruction::Cycle {
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

/// Runs generate_witnesses → prove → verify for a tracer-level trace.
pub fn run_trace_e2e(trace: Vec<tracer::instruction::Cycle>, label: &'static [u8]) {
    use jolt_zkvm::witness::generate_witnesses;

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
    )
    .expect("trace-based verification should succeed");
}

// ── Real program helpers ────────────────────────────────────────────────

/// Traces a guest program and runs prove_trace → verify_proof with Dory.
pub fn prove_and_verify_guest(guest_name: &str, inputs: &[u8]) {
    use jolt_dory::DoryScheme;
    use jolt_host::Program;
    use jolt_zkvm::host::{prove_trace, verify_proof};

    let mut program = Program::new(guest_name);
    let (_, trace, _, _) = program.trace(inputs, &[], &[]);

    let output = prove_trace::<DoryScheme, _>(
        &trace,
        |num_vars| {
            (
                DoryScheme::setup_prover(num_vars),
                DoryScheme::setup_verifier(num_vars),
            )
        },
        cpu(),
    )
    .expect("prove_trace should succeed");

    verify_proof::<DoryScheme>(&output).expect("verify_proof should succeed");
}

/// Traces a guest program with a specific function name, proves with Dory.
pub fn prove_and_verify_guest_func(guest_name: &str, func: &str, inputs: &[u8]) {
    use jolt_dory::DoryScheme;
    use jolt_host::Program;
    use jolt_zkvm::host::{prove_trace, verify_proof};

    let mut program = Program::new(guest_name);
    program.set_func(func);
    let (_, trace, _, _) = program.trace(inputs, &[], &[]);

    let output = prove_trace::<DoryScheme, _>(
        &trace,
        |num_vars| {
            (
                DoryScheme::setup_prover(num_vars),
                DoryScheme::setup_verifier(num_vars),
            )
        },
        cpu(),
    )
    .expect("prove_trace should succeed");

    verify_proof::<DoryScheme>(&output).expect("verify_proof should succeed");
}

/// Traces a guest program and runs through the synthetic-stage pipeline.
pub fn run_real_program_synthetic(guest_name: &str, inputs: &[u8], label: &'static [u8]) {
    run_real_program_synthetic_inner(guest_name, None, inputs, label);
}

/// Same as `run_real_program_synthetic` but selects a specific function.
pub fn run_real_program_synthetic_func(
    guest_name: &str,
    func: &str,
    inputs: &[u8],
    label: &'static [u8],
) {
    run_real_program_synthetic_inner(guest_name, Some(func), inputs, label);
}

fn run_real_program_synthetic_inner(
    guest_name: &str,
    func: Option<&str>,
    inputs: &[u8],
    label: &'static [u8],
) {
    use jolt_host::Program;
    use jolt_zkvm::witness::generate_witnesses;

    let mut program = Program::new(guest_name);
    if let Some(f) = func {
        program.set_func(f);
    }
    let (_, trace, _, _) = program.trace(inputs, &[], &[]);

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
    )
    .expect("verification should succeed");
}
