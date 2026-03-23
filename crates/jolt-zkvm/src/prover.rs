//! Top-level proving API.
//!
//! [`prove`] takes a RISC-V execution trace and produces a complete Jolt proof
//! by running the typed DAG pipeline: S1 (Spartan) → S2–S7 (sumcheck stages)
//! → S8 (PCS opening).

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_instructions::flags::InstructionFlags;
use jolt_ir::zkvm::tags::poly;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningReduction, ProverClaim, RlcReduction,
};
use jolt_spartan::SpartanError;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use tracer::instruction::Cycle;

use crate::preprocessing::{preprocess, JoltConfig};
use crate::proof::JoltProvingKey;
use crate::tables::PolynomialTables;
use crate::witness::generate::generate_witnesses;

#[derive(Debug)]
pub enum ProveError {
    Spartan(SpartanError),
}

impl From<SpartanError> for ProveError {
    fn from(e: SpartanError) -> Self {
        Self::Spartan(e)
    }
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spartan(e) => write!(f, "spartan: {e}"),
        }
    }
}

impl std::error::Error for ProveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spartan(e) => Some(e),
        }
    }
}

pub struct ProveOutput<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub proof: crate::proof::JoltProof<F, PCS>,
    pub verifying_key: jolt_verifier::JoltVerifyingKey<F, PCS>,
    pub tables: PolynomialTables<F>,
}

#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<PCS, B>(
    trace: &[Cycle],
    pcs_setup: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
    backend: Arc<B>,
) -> Result<ProveOutput<PCS::Field, PCS>, ProveError>
where
    PCS: AdditivelyHomomorphic,
    B: ComputeBackend,
{
    let output = generate_witnesses::<PCS::Field>(trace);
    let config = output.config.clone();
    let ohp = config.one_hot_params_from_config();
    let jolt_config = JoltConfig { num_cycles: output.cycle_witnesses.len() };
    let key: JoltProvingKey<PCS::Field, PCS> = preprocess(&jolt_config, pcs_setup);

    let tables = build_tables::<PCS::Field>(trace, &output, &ohp);

    let mut transcript = jolt_transcript::Blake2bTranscript::<PCS::Field>::new(b"jolt-v2");

    // S0: commit witness
    let flat_witness =
        crate::preprocessing::interleave_witnesses(&key.spartan_key, &output.cycle_witnesses);
    let (witness_commitment, _) = PCS::commit(&flat_witness, &key.pcs_prover_setup);
    transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());

    // S1–S7: typed DAG stages.
    // After each stage, append evaluations to the transcript for Fiat-Shamir
    // binding. The verifier must do the same between verify_sumcheck calls.
    use jolt_transcript::AppendToTranscript;
    use jolt_verifier::protocol::types::PackEvals;

    let flush = |evals: &[PCS::Field], t: &mut jolt_transcript::Blake2bTranscript<PCS::Field>| {
        for &e in evals { e.append_to_transcript(t); }
    };

    let s1 = crate::stages::prove_spartan(&tables, &config, &key.spartan_key, &flat_witness, &mut transcript, &backend)?;

    let s2 = crate::stages::prove_stage2(&s1, &tables, &config, &mut transcript, &backend);
    flush(&s2.evals.pack(), &mut transcript);

    let s3 = crate::stages::prove_stage3(&s1, &s2.evals, &tables, &config, &mut transcript, &backend);
    flush(&s3.evals.pack(), &mut transcript);

    let s4 = crate::stages::prove_stage4(&s2.evals, &s3.evals, &tables, &config, &mut transcript, &backend);
    flush(&s4.evals.pack(), &mut transcript);

    let s5 = crate::stages::prove_stage5(&s2.evals, &s4.evals, &tables, &config, &mut transcript, &backend);
    flush(&s5.evals.pack(), &mut transcript);

    let s6 = crate::stages::prove_stage6(&s2.evals, &s4.evals, &s5.evals, &tables, &config, &mut transcript, &backend);
    flush(&s6.evals.pack(), &mut transcript);

    let s7 = crate::stages::prove_stage7(&s5.evals, &s6.evals, &tables, &config, &mut transcript, &backend);
    flush(&s7.evals.pack(), &mut transcript);

    let sp = |proof: &SumcheckProof<PCS::Field>, evals: Vec<PCS::Field>| {
        jolt_verifier::StageProof { round_polys: proof.clone(), evals }
    };
    let stage_proofs = vec![
        sp(&s2.proof, s2.evals.pack()),
        sp(&s3.proof, s3.evals.pack()),
        sp(&s4.proof, s4.evals.pack()),
        sp(&s5.proof, s5.evals.pack()),
        sp(&s6.proof, s6.evals.pack()),
        sp(&s7.proof, s7.evals.pack()),
    ];

    // S8: PCS opening
    let unified = &s7.evals.unified_point;
    let lagrange = lagrange_zero_selector(&unified[..ohp.log_k_chunk]);

    let commitments = commit_polynomials::<PCS>(&tables, &key.pcs_prover_setup);
    let pcs_claims = build_pcs_claims(&tables, &s1, &s6.evals, &s7.evals, unified, lagrange, &flat_witness);
    let (reduced, ()) =
        <RlcReduction as OpeningReduction<PCS>>::reduce_prover(pcs_claims, &mut transcript);

    let opening_proofs = reduced
        .into_iter()
        .map(|c| {
            let poly: PCS::Polynomial = c.evaluations.into();
            PCS::open(&poly, &c.point, c.eval, &key.pcs_prover_setup, None, &mut transcript)
        })
        .collect();

    Ok(ProveOutput {
        proof: crate::proof::JoltProof {
            config,
            spartan_proof: s1.proof,
            stage_proofs,
            opening_proofs,
            witness_commitment,
            commitments,
        },
        verifying_key: jolt_verifier::JoltVerifyingKey {
            spartan_key: key.spartan_key,
            pcs_setup: key.pcs_verifier_setup,
        },
        tables,
    })
}

fn lagrange_zero_selector<F: Field>(r: &[F]) -> F {
    r.iter().fold(F::one(), |acc, &ri| acc * (F::one() - ri))
}

fn commit_polynomials<PCS: AdditivelyHomomorphic>(
    tables: &PolynomialTables<PCS::Field>,
    setup: &PCS::ProverSetup,
) -> Vec<PCS::Output> {
    let mut out = Vec::new();
    let commit = |data: &[PCS::Field]| PCS::commit(data, setup).0;
    out.push(commit(&tables.ram_inc));
    out.push(commit(&tables.rd_inc));
    for ra in &tables.instruction_ra { out.push(commit(ra)); }
    for ra in &tables.bytecode_ra { out.push(commit(ra)); }
    for ra in &tables.ram_ra { out.push(commit(ra)); }
    out
}

fn build_pcs_claims<F: Field>(
    tables: &PolynomialTables<F>,
    s1: &jolt_verifier::protocol::types::SpartanOutput<F>,
    s6: &jolt_verifier::protocol::types::S6Evals<F>,
    s7: &jolt_verifier::protocol::types::S7Evals<F>,
    unified: &[F],
    lagrange: F,
    flat_witness: &[F],
) -> Vec<ProverClaim<F>> {
    let mut claims = Vec::new();
    let pcs = |table: Vec<F>, point: Vec<F>, eval: F| ProverClaim { evaluations: table, point, eval };

    // Dense: Lagrange-normalized to unified point
    claims.push(pcs(tables.ram_inc.clone(), unified.to_vec(), s6.ram_inc_reduced.eval * lagrange));
    claims.push(pcs(tables.rd_inc.clone(), unified.to_vec(), s6.rd_inc_reduced.eval * lagrange));

    // RA polys at unified point
    for (i, e) in s7.instruction_ra.iter().enumerate() {
        claims.push(pcs(tables.instruction_ra[i].clone(), unified.to_vec(), e.eval));
    }
    for (i, e) in s7.bytecode_ra.iter().enumerate() {
        claims.push(pcs(tables.bytecode_ra[i].clone(), unified.to_vec(), e.eval));
    }
    for (i, e) in s7.ram_ra.iter().enumerate() {
        claims.push(pcs(tables.ram_ra[i].clone(), unified.to_vec(), e.eval));
    }

    // Spartan witness
    claims.push(pcs(flat_witness.to_vec(), s1.r_y.clone(), s1.proof.witness_eval));
    claims
}

fn build_tables<F: Field>(
    trace: &[Cycle],
    output: &crate::witness::generate::WitnessOutput<F>,
    ohp: &jolt_verifier::OneHotParams,
) -> PolynomialTables<F> {
    let cw = &output.cycle_witnesses;
    let n = cw.len();
    let col = |idx: usize| -> Vec<F> { cw.iter().map(|w| w[idx]).collect() };
    let flag = |f: InstructionFlags| extract_instruction_flag_poly(trace, n, f as usize);

    PolynomialTables {
        ram_inc: output.witness_store.get(poly::RAM_INC).to_vec(),
        rd_inc: output.witness_store.get(poly::RD_INC).to_vec(),
        instruction_ra: (0..ohp.instruction_d).map(|i| output.witness_store.get(poly::instruction_ra(i)).to_vec()).collect(),
        bytecode_ra: (0..ohp.bytecode_d).map(|i| output.witness_store.get(poly::bytecode_ra(i)).to_vec()).collect(),
        ram_ra: (0..ohp.ram_d).map(|i| output.witness_store.get(poly::ram_ra_committed(i)).to_vec()).collect(),
        rd_write_value: col(crate::r1cs::V_RD_WRITE_VALUE),
        rs1_value: col(crate::r1cs::V_RS1_VALUE),
        rs2_value: col(crate::r1cs::V_RS2_VALUE),
        hamming_weight: cw.iter().map(|w| w[crate::r1cs::V_FLAG_LOAD] + w[crate::r1cs::V_FLAG_STORE]).collect(),
        ram_address: col(crate::r1cs::V_RAM_ADDRESS),
        ram_read_value: col(crate::r1cs::V_RAM_READ_VALUE),
        ram_write_value: col(crate::r1cs::V_RAM_WRITE_VALUE),
        lookup_output: col(crate::r1cs::V_LOOKUP_OUTPUT),
        left_instruction_input: col(crate::r1cs::V_LEFT_INSTRUCTION_INPUT),
        right_instruction_input: col(crate::r1cs::V_RIGHT_INSTRUCTION_INPUT),
        is_rd_not_zero: flag(InstructionFlags::IsRdNotZero),
        write_lookup_to_rd_flag: col(crate::r1cs::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD),
        jump_flag: col(crate::r1cs::V_FLAG_JUMP),
        branch_flag: col(crate::r1cs::V_BRANCH),
        next_is_noop: col(crate::r1cs::V_NEXT_IS_NOOP),
        left_is_rs1: flag(InstructionFlags::LeftOperandIsRs1Value),
        left_is_pc: flag(InstructionFlags::LeftOperandIsPC),
        right_is_rs2: flag(InstructionFlags::RightOperandIsRs2Value),
        right_is_imm: flag(InstructionFlags::RightOperandIsImm),
        unexpanded_pc: col(crate::r1cs::V_UNEXPANDED_PC),
        imm: col(crate::r1cs::V_IMM),
        rs1_ra: extract_register_addr(trace, n, |c| c.rs1_read().map(|(a, _)| a as usize)),
        rs2_ra: extract_register_addr(trace, n, |c| c.rs2_read().map(|(a, _)| a as usize)),
        rd_wa: extract_register_addr(trace, n, |c| c.rd_write().map(|(a, _, _)| a as usize)),
        next_unexpanded_pc: col(crate::r1cs::V_NEXT_UNEXPANDED_PC),
        next_pc: col(crate::r1cs::V_NEXT_PC),
        next_is_virtual: col(crate::r1cs::V_NEXT_IS_VIRTUAL),
        next_is_first_in_sequence: col(crate::r1cs::V_NEXT_IS_FIRST_IN_SEQUENCE),
    }
}

fn extract_register_addr<F: Field>(
    trace: &[Cycle],
    padded_len: usize,
    accessor: impl Fn(&Cycle) -> Option<usize>,
) -> Vec<F> {
    let mut poly: Vec<F> = trace
        .iter()
        .map(|c| accessor(c).map_or(F::zero(), |a| F::from_u64(a as u64)))
        .collect();
    poly.resize(padded_len, F::zero());
    poly
}

fn extract_instruction_flag_poly<F: Field>(
    trace: &[Cycle],
    padded_len: usize,
    flag_idx: usize,
) -> Vec<F> {
    let noop_flags =
        crate::witness::flags::instruction_flags(&tracer::instruction::Instruction::NoOp);
    let pad = if noop_flags[flag_idx] { F::one() } else { F::zero() };
    let bool_to_f = |b: bool| if b { F::one() } else { F::zero() };

    let mut poly: Vec<F> = trace
        .iter()
        .map(|c| bool_to_f(crate::witness::flags::instruction_flags(&c.instruction())[flag_idx]))
        .collect();
    poly.resize(padded_len, pad);
    poly
}
