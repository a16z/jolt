use std::{cell::RefCell, iter::once, rc::Rc};

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::{
        instruction::{CircuitFlags, InstructionFlags, InstructionLookup, NUM_CIRCUIT_FLAGS},
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        vm::instruction_lookups::WORD_SIZE,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    utils::{math::Math, transcript::Transcript},
};
use common::constants::REGISTER_COUNT;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::NormalizedInstruction;

struct ReadCheckingProverState<F: JoltField> {
    r: Vec<F>,
    ra_poly: MultilinearPolynomial<F>,
    unbound_ra_poly: MultilinearPolynomial<F>,
}

pub struct ReadCheckingSumcheck<F: JoltField> {
    rv_claim: F,
    K: usize,
    prover_state: Option<ReadCheckingProverState<F>>,
    val_poly: MultilinearPolynomial<F>,
    sumcheck_id: SumcheckId,
    r_cycle: Vec<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Registers read-write sumcheck
    Stage2,
    /// Registers val sumcheck wa, PCSumcheck, Instruction Lookups
    Stage3,
}

impl<F: JoltField> ReadCheckingSumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: Vec<F>,
        unbound_ra_poly: MultilinearPolynomial<F>,
        val_type: ReadCheckingValType,
    ) -> Self {
        let K = sm.get_bytecode().len();
        let (val, rv_claim, r_cycle, sumcheck_id) = Self::compute_val_rv(sm, val_type);

        let ra_poly = MultilinearPolynomial::from(F);
        let val_poly = MultilinearPolynomial::from(val);

        Self {
            rv_claim,
            K,
            prover_state: Some(ReadCheckingProverState {
                r: Vec::with_capacity(K.log_2()),
                ra_poly,
                unbound_ra_poly,
            }),
            val_poly,
            sumcheck_id,
            r_cycle,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> Self {
        let K = sm.get_bytecode().len();
        let (val, rv_claim, r_cycle, sumcheck_id) = Self::compute_val_rv(sm, val_type);
        let val_poly = MultilinearPolynomial::from(val);

        Self {
            rv_claim,
            K,
            prover_state: None,
            val_poly,
            sumcheck_id,
            r_cycle,
        }
    }

    pub fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> (Vec<F>, F, Vec<F>, SumcheckId) {
        match val_type {
            ReadCheckingValType::Stage1 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_CIRCUIT_FLAGS + 1 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Imm,
                    SumcheckId::SpartanOuter,
                );
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                    r_cycle.r,
                    SumcheckId::BytecodeReadChecking(ReadCheckingValType::Stage1 as usize),
                )
            }
            ReadCheckingValType::Stage2 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..2 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Rs1Ra,
                    SumcheckId::RegistersReadWriteChecking,
                );
                let (_, r_cycle) = r.split_at((REGISTER_COUNT as usize).log_2());
                (
                    Self::compute_val_2(sm, &gamma_powers),
                    Self::compute_rv_claim_2(sm, &gamma_powers),
                    r_cycle.r,
                    SumcheckId::BytecodeReadChecking(ReadCheckingValType::Stage2 as usize),
                )
            }
            ReadCheckingValType::Stage3 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_LOOKUP_TABLES + 1 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::RdWa,
                    SumcheckId::RegistersValEvaluation,
                );
                let (_, r_cycle) = r.split_at((REGISTER_COUNT as usize).log_2());
                (
                    Self::compute_val_3(sm, &gamma_powers),
                    Self::compute_rv_claim_3(sm, &gamma_powers),
                    r_cycle.r,
                    SumcheckId::BytecodeReadChecking(ReadCheckingValType::Stage3 as usize),
                )
            }
        }
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    pub fn compute_val_1(
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
                for (flag, gamma_power) in instruction
                    .circuit_flags()
                    .iter()
                    .zip(gamma_powers[2..].iter())
                {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    pub fn compute_rv_claim_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);

        once(unexpanded_pc_claim)
            .chain(once(imm_claim))
            .chain(CircuitFlags::iter().map(|flag| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::OpFlags(flag),
                    SumcheckId::SpartanOuter,
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
    pub fn compute_val_2(
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
        let eq_r_register = EqPolynomial::evals(r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();

                std::iter::empty()
                    .chain(once(instr.operands.rd))
                    .chain(once(instr.operands.rs1))
                    .chain(once(instr.operands.rs2))
                    .map(|r| eq_r_register[r])
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum::<F>()
            })
            .collect()
    }

    pub fn compute_rv_claim_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, rd_wa_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        std::iter::empty()
            .chain(once(rd_wa_claim))
            .chain(once(rs1_ra_claim))
            .chain(once(rs2_ra_claim))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * rd(k, r_register) + gamma^2 * lookup_table_flag[0](k)
    ///             + gamma^3 * lookup_table_flag[1](k) + ...
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes claims output by the PCSumcheck,
    /// the registers val-evaluation sumcheck, and the instruction lookups sumcheck.
    pub fn compute_val_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
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
                let unexpanded_pc = instr.address;

                let mut linear_combination: F = std::iter::empty()
                    .chain(once(F::from_u64(unexpanded_pc as u64)))
                    .chain(once(eq_r_register[instr.operands.rd]))
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum();

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[2 + table_index];
                }

                linear_combination
            })
            .collect()
    }

    pub fn compute_rv_claim_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, rd_wa_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        std::iter::empty()
            .chain(once(unexpanded_pc_claim))
            .chain(once(rd_wa_claim))
            .chain((0..LookupTables::<WORD_SIZE>::COUNT).map(|i| {
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
}

impl<F: JoltField> SumcheckInstance<F> for ReadCheckingSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = prover_state
                    .ra_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let val_evals = self
                    .val_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                // Compute ra[i] * val[i] for points 0 and 2
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |mut running, new| {
                    for i in 0..2 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        ps.r.push(r_j);

        rayon::join(
            || ps.ra_poly.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.val_poly.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let (_, ra_claim) = accumulator
            .as_ref()
            .unwrap()
            .borrow()
            .get_committed_polynomial_opening(CommittedPolynomial::BytecodeRa, self.sumcheck_id);

        let r: Vec<_> = r.iter().rev().copied().collect();

        // Verify sumcheck_claim = ra_claim * val_eval
        ra_claim * self.val_poly.evaluate(&r)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_claim = prover_state.ra_poly.final_sumcheck_claim();
        accumulator.borrow_mut().append_sparse(
            vec![CommittedPolynomial::BytecodeRa],
            self.sumcheck_id,
            opening_point.r,
            self.r_cycle.clone(),
            vec![ra_claim],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(self.r_cycle.clone())
            .collect::<Vec<_>>();
        accumulator.borrow_mut().append_sparse(
            vec![CommittedPolynomial::BytecodeRa],
            self.sumcheck_id,
            r,
        );
    }
}
