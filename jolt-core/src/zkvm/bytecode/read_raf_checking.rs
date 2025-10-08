use std::{cell::RefCell, iter::once, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable, math::Math, small_scalar::SmallScalar,
        thread::unsafe_allocate_zero_vec,
    },
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{
            CircuitFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::{REGISTER_COUNT, XLEN};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::NormalizedInstruction;

/// Number of batched read-checking sumchecks bespokely
const STAGES: usize = 5;

#[derive(Allocative)]
struct ReadCheckingProverState<F: JoltField> {
    F: [MultilinearPolynomial<F>; STAGES],
    ra: Vec<MultilinearPolynomial<F>>,
    v: Vec<ExpandingTable<F>>,
    eq_polys: [MultilinearPolynomial<F>; STAGES],
    val_gamma: Option<[F; STAGES]>,
    pc: Vec<usize>,
}

#[derive(Allocative)]
pub struct ReadRafSumcheck<F: JoltField> {
    gamma: [F; STAGES],
    gamma_cub: F,
    gamma_sqr: F,
    gamma_four: F,
    gamma_five: F,
    rv_claim: F,
    log_K_chunk: usize,
    K_chunk: usize,
    log_K: usize,
    log_T: usize,
    d: usize,
    prover_state: Option<ReadCheckingProverState<F>>,
    val_polys: [MultilinearPolynomial<F>; STAGES],
    int_poly: IdentityPolynomial<F>,
}

#[derive(Debug, Clone, Copy)]
enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Jump flag from ShouldJumpVirtualization
    Stage2,
    /// PCSumcheck, Instruction Lookups
    Stage3,
    /// Registers from read-write sumcheck (rd, rs1, rs2)
    Stage4,
    /// Registers val evaluation sumcheck
    Stage5,
}

impl<F: JoltField> ReadRafSumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_prover_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_prover_data().0.shared.bytecode.d;
        let log_T = sm.get_prover_data().1.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;
        let gamma_four = gamma_sqr.square();
        let gamma_five = gamma_four * gamma;
        let gamma_six = gamma_five * gamma;
        let (val_1, rv_claim_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let (val_2, rv_claim_2) = Self::compute_val_rv(sm, ReadCheckingValType::Stage2);
        let (val_3, rv_claim_3) = Self::compute_val_rv(sm, ReadCheckingValType::Stage3);
        let (val_4, rv_claim_4) = Self::compute_val_rv(sm, ReadCheckingValType::Stage4);
        let (val_5, rv_claim_5) = Self::compute_val_rv(sm, ReadCheckingValType::Stage5);
        let r_cycles = Self::get_r_cycle(&sm.get_prover_accumulator());
        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let int_poly = IdentityPolynomial::<F>::new(log_K);
        let (preprocessing, trace, _, _) = sm.get_prover_data();

        let rv_claim = rv_claim_1
            + gamma * rv_claim_2
            + gamma_sqr * rv_claim_3
            + gamma_cub * rv_claim_4
            + gamma_four * rv_claim_5
            + gamma_five * raf_claim
            + gamma_six * raf_shift_claim;

        let eq_evals = [
            EqPolynomial::evals(&r_cycles[0]),
            EqPolynomial::evals(&r_cycles[1]),
            EqPolynomial::evals(&r_cycles[2]),
            EqPolynomial::evals(&r_cycles[3]),
            EqPolynomial::evals(&r_cycles[4]),
        ];

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let F = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result_1: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_2: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_3: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_4: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_5: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let pc = preprocessing.shared.bytecode.get_pc(cycle);
                    result_1[pc] += eq_evals[0][j];
                    result_2[pc] += eq_evals[1][j];
                    result_3[pc] += eq_evals[2][j];
                    result_4[pc] += eq_evals[3][j];
                    result_5[pc] += eq_evals[4][j];
                    j += 1;
                }
                (result_1, result_2, result_3, result_4, result_5)
            })
            .reduce(
                || {
                    (
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                    )
                },
                |(mut running_1, mut running_2, mut running_3, mut running_4, mut running_5),
                 (new_1, new_2, new_3, new_4, new_5)| {
                    running_1
                        .par_iter_mut()
                        .zip(new_1.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_2
                        .par_iter_mut()
                        .zip(new_2.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_3
                        .par_iter_mut()
                        .zip(new_3.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_4
                        .par_iter_mut()
                        .zip(new_4.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_5
                        .par_iter_mut()
                        .zip(new_5.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    (running_1, running_2, running_3, running_4, running_5)
                },
            );

        #[cfg(test)]
        {
            // Verify that for each stage i: sum(val_i[k] * F_i[k] * eq_i[k]) = rv_claim_i
            let rv_claims = [rv_claim_1, rv_claim_2, rv_claim_3, rv_claim_4, rv_claim_5];
            let val_evals = [&val_1, &val_2, &val_3, &val_4, &val_5];
            let F_evals = [&F.0, &F.1, &F.2, &F.3, &F.4];
            for i in 0..STAGES {
                let computed_claim: F = (0..K)
                    .into_par_iter()
                    .map(|k| {
                        let val_k = val_evals[i][k];
                        let F_k = F_evals[i][k];
                        val_k * F_k
                    })
                    .sum();
                assert_eq!(
                    computed_claim,
                    rv_claims[i],
                    "Stage {} mismatch: computed {} != expected {}",
                    i + 1,
                    computed_claim,
                    rv_claims[i]
                );
            }
        }

        let eq_polys = eq_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let F = [
            MultilinearPolynomial::from(F.0),
            MultilinearPolynomial::from(F.1),
            MultilinearPolynomial::from(F.2),
            MultilinearPolynomial::from(F.3),
            MultilinearPolynomial::from(F.4),
        ];

        let mut v = (0..d)
            .map(|_| ExpandingTable::new(K_chunk))
            .collect::<Vec<_>>();
        v.par_iter_mut().for_each(|v| v.reset(F::one()));

        let pc = trace
            .par_iter()
            .map(|cycle| preprocessing.shared.bytecode.get_pc(cycle))
            .collect();

        Self {
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk,
            d,
            log_T,
            prover_state: Some(ReadCheckingProverState {
                F,
                ra: Vec::with_capacity(d),
                v,
                eq_polys,
                val_gamma: None,
                pc,
            }),
            val_polys: [
                MultilinearPolynomial::from(val_1),
                MultilinearPolynomial::from(val_2),
                MultilinearPolynomial::from(val_3),
                MultilinearPolynomial::from(val_4),
                MultilinearPolynomial::from(val_5),
            ],
            int_poly,
            // r_cycles: [r_cycle_1, r_cycle_2, r_cycle_3, r_cycle_4, r_cycle_5],
            gamma: [F::one(), gamma, gamma_sqr, gamma_cub, gamma_four],
            gamma_sqr,
            gamma_cub,
            gamma_four,
            gamma_five,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_verifier_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_T = sm.get_verifier_data().2.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;
        let gamma_four = gamma_sqr.square();
        let gamma_five = gamma_four * gamma;
        let gamma_six = gamma_five * gamma;
        let (val_1, rv_claim_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let (val_2, rv_claim_2) = Self::compute_val_rv(sm, ReadCheckingValType::Stage2);
        let (val_3, rv_claim_3) = Self::compute_val_rv(sm, ReadCheckingValType::Stage3);
        let (val_4, rv_claim_4) = Self::compute_val_rv(sm, ReadCheckingValType::Stage4);
        let (val_5, rv_claim_5) = Self::compute_val_rv(sm, ReadCheckingValType::Stage5);
        let int_poly = IdentityPolynomial::new(log_K);

        let val_polys = [
            MultilinearPolynomial::from(val_1),
            MultilinearPolynomial::from(val_2),
            MultilinearPolynomial::from(val_3),
            MultilinearPolynomial::from(val_4),
            MultilinearPolynomial::from(val_5),
        ];

        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let rv_claim = rv_claim_1
            + gamma * rv_claim_2
            + gamma_sqr * rv_claim_3
            + gamma_cub * rv_claim_4
            + gamma_four * rv_claim_5
            + gamma_five * raf_claim
            + gamma_six * raf_shift_claim;

        Self {
            gamma: [F::one(), gamma, gamma_sqr, gamma_cub, gamma_four],
            gamma_sqr,
            gamma_cub,
            gamma_four,
            gamma_five,
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk: 1 << log_K_chunk,
            d,
            log_T,
            prover_state: None,
            val_polys,
            int_poly,
            // r_cycles: [r_cycle_1, r_cycle_2, r_cycle_3, r_cycle_4, r_cycle_5],
        }
    }

    fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> (Vec<F>, F) {
        match val_type {
            ReadCheckingValType::Stage1 => {
                let gamma_powers = get_gamma_powers(
                    &mut *sm.get_transcript().borrow_mut(),
                    3 + NUM_CIRCUIT_FLAGS,
                );
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage2 => {
                let gamma_powers = get_gamma_powers(&mut *sm.get_transcript().borrow_mut(), 6);
                (
                    Self::compute_val_2(sm, &gamma_powers),
                    Self::compute_rv_claim_2(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage3 => {
                let gamma_powers = get_gamma_powers(
                    &mut *sm.get_transcript().borrow_mut(),
                    4 + NUM_LOOKUP_TABLES,
                );
                (
                    Self::compute_val_3(sm, &gamma_powers),
                    Self::compute_rv_claim_3(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage4 => {
                let gamma_powers = get_gamma_powers(&mut *sm.get_transcript().borrow_mut(), 3);
                (
                    Self::compute_val_4(sm, &gamma_powers),
                    Self::compute_rv_claim_4(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage5 => {
                let gamma_powers = get_gamma_powers(&mut *sm.get_transcript().borrow_mut(), 3);
                (
                    Self::compute_val_5(sm, &gamma_powers),
                    Self::compute_rv_claim_5(sm, &gamma_powers),
                )
            }
        }
    }

    fn get_r_cycle(acc: &Rc<RefCell<ProverOpeningAccumulator<F>>>) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r_cycle_2, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::ShouldJumpVirtualization,
        );
        let (r_cycle_3, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_4) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_5) = r.split_at((REGISTER_COUNT as usize).log_2());
        [
            r_cycle_1.r,
            r_cycle_2.r,
            r_cycle_3.r,
            r_cycle_4.r,
            r_cycle_5.r,
        ]
    }

    fn get_r_cycle_verif(
        acc: &Rc<RefCell<VerifierOpeningAccumulator<F>>>,
    ) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::ShouldJumpVirtualization,
        );
        let r_cycle_2 = r.r;
        // Stage 3: Get r_cycle from PCSumcheck or other stage 3 sumchecks
        let (r_cycle_3, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_4) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_5) = r.split_at((REGISTER_COUNT as usize).log_2());
        [
            r_cycle_1.r,
            r_cycle_2,
            r_cycle_3.r,
            r_cycle_4.r,
            r_cycle_5.r,
        ]
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let NormalizedInstruction {
                    address: unexpanded_pc,
                    operands,
                    ..
                } = instruction.normalize();

                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(unexpanded_pc as u64);
                linear_combination += operands.imm.field_mul(gamma_powers[1]);
                linear_combination += (operands.rd as u64).field_mul(gamma_powers[2]);
                let flags = instruction.circuit_flags();
                // sanity check
                assert!(
                    !flags[CircuitFlags::IsCompressed]
                        || !flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                );
                for (flag, gamma_power) in flags.iter().zip(gamma_powers[3..].iter()) {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Rd, SumcheckId::SpartanOuter);

        let mut sum = unexpanded_pc_claim * gamma_powers[0]
            + imm_claim * gamma_powers[1]
            + rd_claim * gamma_powers[2];

        let mut gamma_idx = 3;

        // Add circuit flag claims from SpartanOuter
        for flag in CircuitFlags::iter() {
            let (_, claim) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(flag),
                SumcheckId::SpartanOuter,
            );
            sum += claim * gamma_powers[gamma_idx];
            gamma_idx += 1;
        }

        sum
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = jump_flag(k) + gamma * branch_flag(k) + gamma^2 * rd_addr(k) + gamma^3 * jump_flag(k)
    ///             + gamma^4 * rd_addr(k) + gamma^5 * write_lookup_output_to_rd_flag(k)
    /// where jump_flag(k) = 1 if instruction k is a jump, 0 otherwise
    ///       branch_flag(k) = 1 if instruction k is a branch, 0 otherwise
    ///       rd_addr(k) = rd address for instruction k
    ///       write_lookup_output_to_rd_flag(k) = 1 if instruction k writes lookup output to rd, 0 otherwise
    /// This particular Val virtualizes the flag claims from
    /// ShouldJumpVirtualization, ShouldBranchVirtualization, WritePCtoRDVirtualization, and WriteLookupOutputToRDVirtualization.
    /// Note: Jump flag appears twice (gamma^0 for ShouldJump, gamma^3 for WritePCtoRD).
    /// Note: Rd addr appears twice (gamma^2 for WritePCtoRD, gamma^4 for WriteLookupOutputToRD).
    fn compute_val_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let flags = instruction.circuit_flags();
                let instr = instruction.normalize();
                let jump_val = if flags[CircuitFlags::Jump] {
                    F::one()
                } else {
                    F::zero()
                };
                let branch_val = if flags[CircuitFlags::Branch] {
                    F::one()
                } else {
                    F::zero()
                };
                let write_lookup_output_to_rd_val = if flags[CircuitFlags::WriteLookupOutputToRD] {
                    F::one()
                } else {
                    F::zero()
                };
                let rd_addr_val = F::from_u64(instr.operands.rd as u64);
                jump_val * gamma_powers[0]
                    + branch_val * gamma_powers[1]
                    + rd_addr_val * gamma_powers[2]
                    + jump_val * gamma_powers[3]
                    + rd_addr_val * gamma_powers[4]
                    + write_lookup_output_to_rd_val * gamma_powers[5]
            })
            .collect()
    }

    fn compute_rv_claim_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, jump_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::ShouldJumpVirtualization,
        );

        let (_, branch_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Branch),
            SumcheckId::ShouldBranchVirtualization,
        );

        let (_, rd_wa_claim_write_pc) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::WritePCtoRDVirtualization,
        );

        let (_, jump_claim_write_pc) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::WritePCtoRDVirtualization,
        );

        let (_, rd_wa_claim_write_lookup) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::WriteLookupOutputToRDVirtualization,
        );

        let (_, write_lookup_output_to_rd_flag_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            SumcheckId::WriteLookupOutputToRDVirtualization,
        );

        jump_claim * gamma_powers[0]
            + branch_claim * gamma_powers[1]
            + rd_wa_claim_write_pc * gamma_powers[2]
            + jump_claim_write_pc * gamma_powers[3]
            + rd_wa_claim_write_lookup * gamma_powers[4]
            + write_lookup_output_to_rd_flag_claim * gamma_powers[5]
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * instr_raf_flag(k)
    ///             + gamma^2 * lookup_table_flag[0](k)
    ///             + gamma^3 * lookup_table_flag[1](k) + ...
    /// This particular Val virtualizes claims output by the PCSumcheck
    /// and the instruction lookups sumcheck (but NOT registers val evaluation).
    fn compute_val_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();
                let flags = instruction.circuit_flags();
                let unexpanded_pc = instr.address;

                let mut linear_combination: F = F::zero();

                linear_combination += F::from_u64(unexpanded_pc as u64);
                if flags[CircuitFlags::IsNoop] {
                    linear_combination += gamma_powers[1];
                }
                if !flags.is_interleaved_operands() {
                    linear_combination += gamma_powers[2];
                }

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[3 + table_index];
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, raf_flag_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        std::iter::empty()
            .chain(once(unexpanded_pc_claim))
            .chain(once(is_noop_claim))
            .chain(once(raf_flag_claim))
            .chain((0..LookupTables::<XLEN>::COUNT).map(|i| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(i),
                    SumcheckId::InstructionReadRaf,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register) + gamma * rs1(k, r_register) + gamma^2 * rs2(k, r_register)
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// and analogously for rs1(k, k') and rs2(k, k').
    /// This particular Val virtualizes claims output by the registers read/write checking sumcheck.
    fn compute_val_4(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        let r_register = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        let r_register = &r_register[..(REGISTER_COUNT as usize).log_2()];
        let eq_r_register = EqPolynomial::<F>::evals(r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);

        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();

                std::iter::empty()
                    .chain(once(instr.operands.rd))
                    .chain(once(instr.operands.rs1))
                    .chain(once(instr.operands.rs2))
                    .map(|r| eq_r_register[r as usize])
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum::<F>()
            })
            .collect()
    }

    fn compute_rv_claim_4(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        std::iter::empty()
            .chain(once(VirtualPolynomial::RdWa))
            .chain(once(VirtualPolynomial::Rs1Ra))
            .chain(once(VirtualPolynomial::Rs2Ra))
            .map(|vp| {
                sm.get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum::<F>()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register)
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes the claim output by the registers val-evaluation sumcheck.
    fn compute_val_5(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        _gamma_powers: &[F],
    ) -> Vec<F> {
        let r_register = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        let r_register: Vec<_> = r_register[..(REGISTER_COUNT as usize).log_2()].to_vec();
        let eq_r_register = EqPolynomial::evals(&r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);

        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();
                eq_r_register[instr.operands.rd as usize]
            })
            .collect()
    }

    fn compute_rv_claim_5(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        _gamma_powers: &[F],
    ) -> F {
        let (_, rd_wa_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        rd_wa_claim
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        if round < self.log_K {
            const DEGREE: usize = 2;

            let univariate_poly_evals: [F; DEGREE] = (0..self.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = ps.F.iter().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow)
                    });
                    let int_evals =
                        self.int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                    // We have a separate Val polynomial for each stage
                    // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
                    // So we would have:
                    // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
                    // Stage 2: gamma^1 * (Val_2)
                    // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
                    // Stage 4: gamma^3 * (Val_4)
                    // Stage 5: gamma^4 * (Val_5)
                    // Which matches with the input claim:
                    // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
                    let val_evals = self
                        .val_polys
                        .iter()
                        // Val polynomials
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow))
                        // Here are the RAF polynomials and their powers
                        .zip([Some(&int_evals), None, Some(&int_evals), None, None])
                        .zip([Some(self.gamma_five), None, Some(self.gamma_four), None, None])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                        int_evals[j] * gamma.unwrap()
                                    })
                            })
                        });

                    // Compute ra * val * gamma, and sum together
                    ra_evals
                        .zip(val_evals)
                        .zip(self.gamma.iter())
                        .map(|((ra_evals, val_evals), gamma)| {
                            std::array::from_fn(|j| ra_evals[j] * val_evals[j] * gamma)
                        })
                        .fold([F::zero(); DEGREE], |mut running, new: [F; DEGREE]| {
                            for i in 0..DEGREE {
                                running[i] += new[i];
                            }
                            running
                        })
                })
                .reduce(
                    || [F::zero(); DEGREE],
                    |mut running, new| {
                        for i in 0..DEGREE {
                            running[i] += new[i];
                        }
                        running
                    },
                );

            univariate_poly_evals.to_vec()
        } else {
            let degree = <Self as SumcheckInstance<F, T>>::degree(self);
            (0..ps.ra[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_polys
                        .iter()
                        .map(|eq| eq.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let ra_evals = ps
                        .ra
                        .iter()
                        .map(|ra| ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let eq_times_val = eq_evals
                        .zip(ps.val_gamma.as_ref().unwrap().iter())
                        .map(|(eq_evals, val_evals)| {
                            eq_evals
                                .into_iter()
                                .map(|eq_eval| eq_eval * val_evals)
                                .collect()
                        })
                        .fold(
                            vec![F::zero(); degree],
                            |mut running: Vec<F>, new: Vec<F>| {
                                for i in 0..degree {
                                    running[i] += new[i];
                                }
                                running
                            },
                        );

                    ra_evals.fold(eq_times_val, |mut running: Vec<F>, new: Vec<F>| {
                        for i in 0..degree {
                            running[i] *= new[i];
                        }
                        running
                    })
                })
                .reduce(
                    || vec![F::zero(); degree],
                    |mut running, new| {
                        for i in 0..degree {
                            running[i] += new[i];
                        }
                        running
                    },
                )
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_K {
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.val_polys
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                });
                s.spawn(|_| {
                    self.int_poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
                s.spawn(|_| {
                    ps.F.par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| {
                    ps.v[round / self.log_K_chunk].update(r_j);
                });
            });
            if round == self.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            ps.ra
                .par_iter_mut()
                .chain(ps.eq_polys.par_iter_mut())
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (r_address_prime, r_cycle_prime) = r.split_at(self.log_K);
        // r_cycle is bound LowToHigh, so reverse
        let r_cycle_prime = r_cycle_prime
            .iter()
            .rev()
            .copied()
            .collect::<Vec<F::Challenge>>();

        let int_poly = self.int_poly.evaluate(r_address_prime);

        let ra_claims = (0..self.d).map(|i| {
            accumulator
                .borrow()
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });
        let r_cycles = Self::get_r_cycle_verif(accumulator);

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^4 * raf_3
        let val = self
            .val_polys
            .iter()
            .zip(r_cycles.iter())
            .zip(self.gamma.iter())
            .zip([
                int_poly * self.gamma_five, // RAF for Stage1
                F::zero(),                  // There's no raf for Stage2
                int_poly * self.gamma_four, // RAF for Stage3
                F::zero(),                  // There's no raf for Stage4
                F::zero(),                  // There's no raf for Stage5
            ])
            .map(|(((val, r_cycle), gamma), int_poly)| {
                (val.evaluate(r_address_prime) + int_poly)
                    * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime)
                    * gamma
            })
            .sum::<F>();

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = opening_point.to_vec();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let (r_address, r_cycle) = opening_point.clone().split_at(self.log_K);

        for i in 0..self.d {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address.to_vec(),
                r_cycle.clone().into(),
                vec![ps.ra[i].final_sumcheck_claim()],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_address, r_cycle) = opening_point.split_at(self.log_K);
        (0..self.d).for_each(|i| {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                [r_address, &r_cycle.r].concat(),
            );
        });
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> ReadRafSumcheck<F> {
    fn init_log_t_rounds(&mut self) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let int_poly = self.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^4 * raf_3
        ps.val_gamma = Some(
            self.val_polys
                .iter()
                .zip(self.gamma.iter())
                .zip([
                    int_poly * self.gamma_five,
                    F::zero(),
                    int_poly * self.gamma_four,
                    F::zero(),
                    F::zero(),
                ])
                .map(|((poly, gamma), int_poly)| (poly.final_sumcheck_claim() + int_poly) * gamma)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        ps.v.par_iter()
            .enumerate()
            .map(|(i, v)| {
                let ra_i: Vec<F> = ps
                    .pc
                    .par_iter()
                    .map(|k| {
                        let k = (k >> (self.log_K_chunk * (self.d - i - 1))) % self.K_chunk;
                        v[k]
                    })
                    .collect();
                MultilinearPolynomial::from(ra_i)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|ra| {
                ps.ra.push(ra);
            });
    }
}

fn get_gamma_powers<F: JoltField>(transcript: &mut impl Transcript, amount: usize) -> Vec<F> {
    let mut gamma_powers = vec![F::one()];
    let gamma: F = transcript.challenge_scalar();
    for _ in 1..amount {
        gamma_powers.push(gamma * gamma_powers.last().unwrap());
    }
    gamma_powers
}
