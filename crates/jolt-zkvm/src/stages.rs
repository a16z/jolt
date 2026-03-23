//! Typed DAG stage functions for the Jolt proving pipeline.
//!
//! Each stage proves a batched sumcheck, consuming typed outputs from upstream
//! stages and producing typed outputs for downstream. The orchestrator in
//! [`crate::prover`] calls these in DAG order: S1 → S2 → … → S7.

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_openings::{CommittedEval, VirtualEval};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{BatchedSumcheckProver, CaptureHandler, SumcheckClaim, SumcheckProof};
use jolt_transcript::Transcript;
use jolt_verifier::ProverConfig;

use crate::evaluators::catalog::{self, Term};
use crate::evaluators::kernel::KernelEvaluator;
use jolt_verifier::protocol::types::*;
use jolt_verifier::protocol::claims;
use crate::tables::PolynomialTables;

/// Prover stage result: sumcheck proof + typed evaluations.
pub struct StageResult<F: Field, E> {
    pub proof: SumcheckProof<F>,
    pub evals: E,
}

pub(crate) fn eval_poly<F: Field>(table: &[F], point: &[F]) -> F {
    jolt_poly::Polynomial::new(table.to_vec()).evaluate(point)
}

pub(crate) use jolt_verifier::verify::gamma_powers;

/// Builds a `SumcheckCompute` witness for `Σ w(x) · g(x)` where `w` is the
/// weighting polynomial (eq, eq+1, LT, or unit) and `g` is a sum-of-products
/// over `poly_tables`.
///
/// `formula_descriptor` wraps the formula as `opening(0) · Σ_k c_k · Π opening(factor+1)`,
/// so `w_table` becomes `inputs[0]` and term factor indices are 0-based into `poly_tables`.
fn weighted_witness<F: Field, B: ComputeBackend>(
    w_table: &[F],
    poly_tables: &[&[F]],
    terms: &[Term<F>],
    degree: usize,
    backend: &Arc<B>,
) -> Box<dyn jolt_sumcheck::SumcheckCompute<F>> {
    let (desc, challenges) = catalog::formula_descriptor(terms, poly_tables.len(), degree);
    let kernel = backend.compile_kernel_with_challenges::<F>(&desc, &challenges);

    let mut inputs = vec![backend.upload(w_table)];
    inputs.extend(poly_tables.iter().map(|t| backend.upload(t)));

    Box::new(KernelEvaluator::with_unit_weights(
        inputs,
        kernel,
        desc.degree + 1,
        Arc::clone(backend),
    ))
}

/// Runs a batched sumcheck and returns the proof, challenges, and BigEndian eval point.
fn prove_batch<F: Field, T: Transcript<Challenge = F>>(
    claims: &[SumcheckClaim<F>],
    witnesses: &mut [Box<dyn jolt_sumcheck::SumcheckCompute<F>>],
    transcript: &mut T,
) -> (SumcheckProof<F>, Vec<F>, Vec<F>) {
    let max_vars = claims.iter().map(|c| c.num_vars).max().unwrap_or(0);
    let captured = BatchedSumcheckProver::prove_with_handler(
        claims,
        witnesses,
        transcript,
        CaptureHandler::with_capacity(max_vars),
    );
    let eval_point: Vec<F> = captured.challenges.iter().rev().copied().collect();
    (captured.proof, captured.challenges, eval_point)
}

fn committed_eval<F: Field>(table: &[F], point: &[F]) -> CommittedEval<F> {
    CommittedEval {
        point: point.to_vec(),
        eval: eval_poly(table, point),
    }
}


pub fn prove_spartan<F, T, B>(
    tables: &PolynomialTables<F>,
    _config: &ProverConfig,
    key: &jolt_spartan::UniformSpartanKey<F>,
    flat_witness: &[F],
    transcript: &mut T,
    _backend: &Arc<B>,
) -> Result<SpartanOutput<F>, jolt_spartan::SpartanError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let (proof, r_x, r_y) =
        jolt_spartan::UniformSpartanProver::prove_dense_with_challenges(
            key,
            flat_witness,
            transcript,
        )?;

    let r_cycle = &r_y[..tables.log_num_cycles()];
    let evals = SpartanVirtualEvals {
        ram_read_value: VirtualEval(eval_poly(&tables.ram_read_value, r_cycle)),
        ram_write_value: VirtualEval(eval_poly(&tables.ram_write_value, r_cycle)),
        ram_address: VirtualEval(eval_poly(&tables.ram_address, r_cycle)),
        lookup_output: VirtualEval(eval_poly(&tables.lookup_output, r_cycle)),
        left_operand: VirtualEval(F::zero()),
        right_operand: VirtualEval(F::zero()),
        left_instruction_input: VirtualEval(eval_poly(&tables.left_instruction_input, r_cycle)),
        right_instruction_input: VirtualEval(eval_poly(&tables.right_instruction_input, r_cycle)),
        rd_write_value: VirtualEval(eval_poly(&tables.rd_write_value, r_cycle)),
        rs1_value: VirtualEval(eval_poly(&tables.rs1_value, r_cycle)),
        rs2_value: VirtualEval(eval_poly(&tables.rs2_value, r_cycle)),
    };

    Ok(SpartanOutput { proof, r_x, r_y, evals })
}

/// S2: Product Virtual (single-instance for now).
///
/// TODO(T19): Add RamRW, InstrLookupsCR, RamRaf, OutputCheck instances.
pub fn prove_stage2<F, T, B>(
    s1: &SpartanOutput<F>,
    tables: &PolynomialTables<F>,
    _config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S2Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = tables.log_num_cycles();
    let r_cycle = &s1.r_y[..log_t];

    let gp = gamma_powers(transcript.challenge(), 5);
    let eq = EqPolynomial::new(r_cycle.to_vec()).evaluations();

    let pv_claimed_sum: F = (0..n)
        .map(|j| {
            let t = &tables;
            eq[j]
                * (gp[0] * t.left_instruction_input[j] * t.right_instruction_input[j]
                    + gp[1] * t.is_rd_not_zero[j] * t.write_lookup_to_rd_flag[j]
                    + gp[2] * t.is_rd_not_zero[j] * t.jump_flag[j]
                    + gp[3] * t.lookup_output[j] * t.branch_flag[j]
                    + gp[4] * t.jump_flag[j] * (F::one() - t.next_is_noop[j]))
        })
        .sum();

    let factor_tables: Vec<&[F]> = vec![
        &tables.left_instruction_input,
        &tables.right_instruction_input,
        &tables.is_rd_not_zero,
        &tables.write_lookup_to_rd_flag,
        &tables.jump_flag,
        &tables.lookup_output,
        &tables.branch_flag,
        &tables.next_is_noop,
    ];

    let terms = vec![
        Term { coeff: gp[0], factors: vec![0, 1] },
        Term { coeff: gp[1], factors: vec![2, 3] },
        Term { coeff: gp[2], factors: vec![2, 4] },
        Term { coeff: gp[3], factors: vec![5, 6] },
        Term { coeff: gp[4], factors: vec![4] },
        Term { coeff: -gp[4], factors: vec![4, 7] },
    ];

    let (pv_desc, pv_chal) = catalog::formula_descriptor(&terms, 8, 4);
    let pv_kernel = backend.compile_kernel_with_challenges::<F>(&pv_desc, &pv_chal);
    let mut pv_inputs: Vec<_> = vec![backend.upload(&eq)];
    pv_inputs.extend(factor_tables.iter().map(|t| backend.upload(t)));

    let pv_witness: Box<dyn jolt_sumcheck::SumcheckCompute<F>> =
        Box::new(KernelEvaluator::with_unit_weights(
            pv_inputs,
            pv_kernel,
            pv_desc.degree + 1,
            Arc::clone(backend),
        ));

    let claims = [SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: pv_claimed_sum }];
    let (proof, _challenges, ep) = prove_batch(&claims, &mut [pv_witness], transcript);

    let ev = |t: &[F]| VirtualEval(eval_poly(t, &ep));

    StageResult {
        proof,
        evals: S2Evals {
            eval_point: ep.clone(),
            ram_val: VirtualEval(F::zero()),
            ram_inc: committed_eval(&tables.ram_inc, &ep),
            next_is_noop: ev(&tables.next_is_noop),
            left_instr_input: ev(&tables.left_instruction_input),
            right_instr_input: ev(&tables.right_instruction_input),
            lookup_output: ev(&tables.lookup_output),
            left_operand: VirtualEval(F::zero()),
            right_operand: VirtualEval(F::zero()),
            ram_raf_eval: ev(&tables.ram_address),
            ram_val_final: ev(&tables.ram_write_value),
        },
    }
}

/// S3: Shift + InstructionInput + RegistersCR.
pub fn prove_stage3<F, T, B>(
    s1: &SpartanOutput<F>,
    s2: &S2Evals<F>,
    tables: &PolynomialTables<F>,
    _config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S3Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = tables.log_num_cycles();
    let r_cycle = &s1.r_y[..log_t];

    let shift_gp = gamma_powers(transcript.challenge(), 5);
    let instr_gamma: F = transcript.challenge();
    let reg_gamma: F = transcript.challenge();
    let reg_gamma_sq = reg_gamma * reg_gamma;

    // Shift: combined eq+1 from outer and product points
    let (_, epo_outer) = jolt_poly::EqPlusOnePolynomial::evals(r_cycle, None);
    let (_, epo_product) = jolt_poly::EqPlusOnePolynomial::evals(&s2.eval_point, None);

    let noop_complement: Vec<F> = tables.next_is_noop.iter().map(|&v| F::one() - v).collect();
    let shift_eq: Vec<F> = (0..n).map(|j| epo_outer[j] + epo_product[j]).collect();

    let shift_sum: F = (0..n)
        .map(|j| {
            epo_outer[j]
                * (shift_gp[0] * tables.next_unexpanded_pc[j]
                    + shift_gp[1] * tables.next_pc[j]
                    + shift_gp[2] * tables.next_is_virtual[j]
                    + shift_gp[3] * tables.next_is_first_in_sequence[j])
                + epo_product[j] * shift_gp[4] * noop_complement[j]
        })
        .sum();

    let shift_witness = weighted_witness(
        &shift_eq,
        &[
            &tables.next_unexpanded_pc,
            &tables.next_pc,
            &tables.next_is_virtual,
            &tables.next_is_first_in_sequence,
            &noop_complement,
        ],
        &[
            Term { coeff: shift_gp[0], factors: vec![0] },
            Term { coeff: shift_gp[1], factors: vec![1] },
            Term { coeff: shift_gp[2], factors: vec![2] },
            Term { coeff: shift_gp[3], factors: vec![3] },
            Term { coeff: shift_gp[4], factors: vec![4] },
        ],
        3,
        backend,
    );

    // InstructionInput: eq · (right + γ·left)
    let eq = EqPolynomial::new(r_cycle.to_vec()).evaluations();

    let instr_sum = claims::s3_instruction_input(s2, instr_gamma);

    let instr_witness = weighted_witness(
        &eq,
        &[
            &tables.right_is_rs2, &tables.rs2_value,
            &tables.right_is_imm, &tables.imm,
            &tables.left_is_rs1, &tables.rs1_value,
            &tables.left_is_pc, &tables.unexpanded_pc,
        ],
        &[
            Term { coeff: F::one(), factors: vec![0, 1] },
            Term { coeff: F::one(), factors: vec![2, 3] },
            Term { coeff: instr_gamma, factors: vec![4, 5] },
            Term { coeff: instr_gamma, factors: vec![6, 7] },
        ],
        4,
        backend,
    );

    let reg_sum = claims::s3_registers_cr(&s1.evals, reg_gamma);

    let reg_witness = weighted_witness(
        &eq,
        &[&tables.rd_write_value, &tables.rs1_value, &tables.rs2_value],
        &[
            Term { coeff: F::one(), factors: vec![0] },
            Term { coeff: reg_gamma, factors: vec![1] },
            Term { coeff: reg_gamma_sq, factors: vec![2] },
        ],
        3,
        backend,
    );

    let claims = [
        SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: shift_sum },
        SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: instr_sum },
        SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: reg_sum },
    ];
    let (proof, _challenges, ep) =
        prove_batch(&claims, &mut [shift_witness, instr_witness, reg_witness], transcript);

    let ev = |t: &[F]| VirtualEval(eval_poly(t, &ep));
    StageResult {
        proof,
        evals: S3Evals {
            eval_point: ep.clone(),
            rs1_value: ev(&tables.rs1_value),
            rs2_value: ev(&tables.rs2_value),
            rd_write_value: ev(&tables.rd_write_value),
        },
    }
}

/// S4: RegistersRW + RamValCheck.
pub fn prove_stage4<F, T, B>(
    s2: &S2Evals<F>,
    s3: &S3Evals<F>,
    tables: &PolynomialTables<F>,
    _config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S4Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = tables.log_num_cycles();

    let reg_gamma: F = transcript.challenge();
    let reg_gamma_sq = reg_gamma * reg_gamma;
    let ram_gamma: F = transcript.challenge();

    let reg_sum = claims::s4_registers_rw(s3, reg_gamma);
    let eq_reg = EqPolynomial::new(s3.eval_point.clone()).evaluations();

    let reg_witness = weighted_witness(
        &eq_reg,
        &[&tables.rd_wa, &tables.rd_inc, &tables.rd_write_value, &tables.rs1_ra, &tables.rs2_ra],
        &[
            Term { coeff: F::one(), factors: vec![0, 1] },
            Term { coeff: F::one(), factors: vec![0, 2] },
            Term { coeff: reg_gamma, factors: vec![3, 2] },
            Term { coeff: reg_gamma_sq, factors: vec![4, 2] },
        ],
        4,
        backend,
    );

    // RamValCheck: (eq·(LT+γ)) · inc · addr
    let eq_ram = EqPolynomial::new(s2.eval_point.clone()).evaluations();
    let lt = jolt_poly::LtPolynomial::evaluations(&s2.eval_point);
    let ram_w: Vec<F> = (0..n).map(|j| eq_ram[j] * (lt[j] + ram_gamma)).collect();
    let ram_sum: F = (0..n)
        .map(|j| ram_w[j] * tables.ram_inc[j] * tables.ram_address[j])
        .sum();

    let ram_witness = weighted_witness(
        &ram_w,
        &[&tables.ram_inc, &tables.ram_address],
        &[Term { coeff: F::one(), factors: vec![0, 1] }],
        4,
        backend,
    );

    let claims = [
        SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: reg_sum },
        SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: ram_sum },
    ];
    let (proof, _challenges, ep) =
        prove_batch(&claims, &mut [reg_witness, ram_witness], transcript);

    StageResult {
        proof,
        evals: S4Evals {
            eval_point: ep.clone(),
            ram_inc: committed_eval(&tables.ram_inc, &ep),
            rd_inc: committed_eval(&tables.rd_inc, &ep),
        },
    }
}

/// S5: RegistersValEval (RamRaCR and InstructionReadRaf deferred to T19).
pub fn prove_stage5<F, T, B>(
    s2: &S2Evals<F>,
    _s4: &S4Evals<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S5Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = tables.log_num_cycles();
    let _ohp = config.one_hot_params_from_config();

    let _instr_raf_gamma: F = transcript.challenge();
    let _ram_ra_gamma: F = transcript.challenge();
    let reg_val_r: Vec<F> = (0..log_t).map(|_| transcript.challenge()).collect();

    // Brute-force claimed sum for deferred RamRaCR (keeps transcript in sync)
    let _eq_rw = EqPolynomial::new(s2.eval_point.clone()).evaluations();

    // RegistersValEval: Σ inc·wa·LT(r_cycle, j)
    let lt = jolt_poly::LtPolynomial::evaluations(&reg_val_r);
    let rv_sum: F = (0..n).map(|j| tables.rd_inc[j] * tables.rd_wa[j] * lt[j]).sum();

    let rv_witness = weighted_witness(
        &lt,
        &[&tables.rd_inc, &tables.rd_wa],
        &[Term { coeff: F::one(), factors: vec![0, 1] }],
        4,
        backend,
    );

    let claims = [SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: rv_sum }];
    let (proof, _challenges, ep) = prove_batch(&claims, &mut [rv_witness], transcript);

    StageResult {
        proof,
        evals: S5Evals {
            eval_point: ep.clone(),
            rd_inc: committed_eval(&tables.rd_inc, &ep),
        },
    }
}

/// S6: IncCR + HammingBooleanity (Booleanity, RA virtual, BytecodeReadRaf deferred to T19/T21).
pub fn prove_stage6<F, T, B>(
    s2: &S2Evals<F>,
    s4: &S4Evals<F>,
    s5: &S5Evals<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S6Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = tables.log_num_cycles();
    let _ohp = config.one_hot_params_from_config();

    // Squeeze placeholders for deferred instances (keeps transcript order)
    let _bytecode_gamma: F = transcript.challenge();
    let _bool_eq: Vec<F> = (0..log_t).map(|_| transcript.challenge()).collect();
    let _bool_gamma: F = transcript.challenge();
    let h_eq_point: Vec<F> = (0..log_t).map(|_| transcript.challenge()).collect();
    let _ra_virt_gamma: F = transcript.challenge();
    let _instr_ra_gamma: F = transcript.challenge();
    let inc_gamma: F = transcript.challenge();

    // IncCR: precompute combined eq weights from 4 prior stage points
    let eq_r2 = EqPolynomial::new(s2.ram_inc.point.clone()).evaluations();
    let eq_r4r = EqPolynomial::new(s4.ram_inc.point.clone()).evaluations();
    let eq_r4d = EqPolynomial::new(s4.rd_inc.point.clone()).evaluations();
    let eq_r5 = EqPolynomial::new(s5.rd_inc.point.clone()).evaluations();

    let g2 = inc_gamma * inc_gamma;
    let g3 = g2 * inc_gamma;
    let weighted_inc: Vec<F> = (0..n)
        .map(|j| {
            (eq_r2[j] + inc_gamma * eq_r4r[j]) * tables.ram_inc[j]
                + (g2 * eq_r4d[j] + g3 * eq_r5[j]) * tables.rd_inc[j]
        })
        .collect();
    let inc_sum = claims::s6_inc_cr(s2, s4, s5, inc_gamma);
    debug_assert_eq!(inc_sum, weighted_inc.iter().copied().sum::<F>(), "IncCR brute-force mismatch");

    let unit_eq: Vec<F> = vec![F::one(); n];
    let inc_witness = weighted_witness(
        &unit_eq,
        &[&weighted_inc],
        &[Term { coeff: F::one(), factors: vec![0] }],
        3,
        backend,
    );

    // HammingBooleanity: eq · h · (h − 1) = 0
    let h_eq = EqPolynomial::new(h_eq_point).evaluations();
    let h_witness = weighted_witness(
        &h_eq,
        &[&tables.hamming_weight],
        &[
            Term { coeff: F::one(), factors: vec![0, 0] },
            Term { coeff: -F::one(), factors: vec![0] },
        ],
        4,
        backend,
    );

    let claims = [
        SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: inc_sum },
        SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: F::zero() },
    ];
    let (proof, _challenges, ep) = prove_batch(&claims, &mut [inc_witness, h_witness], transcript);

    StageResult {
        proof,
        evals: S6Evals {
            r_cycle: ep.clone(),
            ram_inc_reduced: committed_eval(&tables.ram_inc, &ep),
            rd_inc_reduced: committed_eval(&tables.rd_inc, &ep),
        },
    }
}

/// S7: HammingWeightCR → unified opening point `(r_addr || r_cycle)`.
pub fn prove_stage7<F, T, B>(
    _s5: &S5Evals<F>,
    s6: &S6Evals<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> StageResult<F, S7Evals<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let ohp = config.one_hot_params_from_config();
    let log_k = ohp.log_k_chunk;
    let total_d = ohp.instruction_d + ohp.bytecode_d + ohp.ram_d;

    let gp = gamma_powers(transcript.challenge(), total_d);

    // Pushforward G_i(k) = Σ_j eq(r_cycle_s6, j) · ra_i(k, j)
    let eq_cycle = EqPolynomial::new(s6.r_cycle.clone()).evaluations();
    let n_cycle = tables.num_cycles();
    let n_addr = ohp.k_chunk;
    let all_ra = tables.all_ra_polys();

    let g_polys: Vec<Vec<F>> = all_ra
        .iter()
        .map(|ra| {
            let mut g = vec![F::zero(); n_addr];
            for j in 0..n_cycle {
                for k in 0..n_addr {
                    g[k] += eq_cycle[j] * ra[k * n_cycle + j];
                }
            }
            g
        })
        .collect();

    let hw_sum: F = (0..n_addr)
        .map(|k| (0..total_d).map(|i| gp[i] * g_polys[i][k]).sum::<F>())
        .sum();

    let hw_terms: Vec<Term<F>> = (0..total_d)
        .map(|i| Term { coeff: gp[i], factors: vec![i] })
        .collect();
    let unit_eq: Vec<F> = vec![F::one(); n_addr];
    let g_refs: Vec<&[F]> = g_polys.iter().map(|g| g.as_slice()).collect();
    let hw_witness = weighted_witness(&unit_eq, &g_refs, &hw_terms, 3, backend);

    let claims = [SumcheckClaim { num_vars: log_k, degree: 2, claimed_sum: hw_sum }];
    let (proof, _challenges, r_addr) = prove_batch(&claims, &mut [hw_witness], transcript);

    let mut unified = r_addr;
    unified.extend_from_slice(&s6.r_cycle);

    StageResult {
        proof,
        evals: S7Evals {
            unified_point: unified.clone(),
            instruction_ra: (0..ohp.instruction_d)
                .map(|i| committed_eval(&tables.instruction_ra[i], &unified))
                .collect(),
            bytecode_ra: (0..ohp.bytecode_d)
                .map(|i| committed_eval(&tables.bytecode_ra[i], &unified))
                .collect(),
            ram_ra: (0..ohp.ram_d)
                .map(|i| committed_eval(&tables.ram_ra[i], &unified))
                .collect(),
        },
    }
}
