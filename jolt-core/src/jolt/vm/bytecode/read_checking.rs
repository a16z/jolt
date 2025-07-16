use std::{cell::RefCell, iter::once, rc::Rc};

use crate::{
    dag::{stage::StagedSumcheck, state_manager::StateManager},
    field::JoltField,
    jolt::{
        instruction::{CircuitFlags, InstructionFlags, InstructionLookup, NUM_CIRCUIT_FLAGS},
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        vm::instruction_lookups::WORD_SIZE,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::{NormalizedInstruction, RV32IMInstruction};

struct ReadCheckingProverState<F: JoltField> {
    r: Vec<F>,
    ra_poly: MultilinearPolynomial<F>,
    unbound_ra_poly: MultilinearPolynomial<F>,
}

pub struct ReadCheckingSumcheck<F: JoltField> {
    /// Input claim = rv(r_cycle)
    rv_claim: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<ReadCheckingProverState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim: Option<F>,
    val_poly: MultilinearPolynomial<F>,
    op_key: OpeningsKeys,
    r_cycle: Vec<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadCheckingValTypes {
    /// Spartan outer sumcheck
    Stage1,
    /// Spartan shift sumcheck, instructions, registers
    Stage2,
    /// Registers val sumcheck wa
    Stage3,
}

impl<F: JoltField> ReadCheckingSumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: Vec<F>,
        unbound_ra_poly: MultilinearPolynomial<F>,
        val_type: ReadCheckingValTypes,
    ) -> Self {
        let K = sm.get_bytecode().len();
        let (val, rv_claim, r_cycle, op_key) = Self::compute_val_rv(sm, val_type);

        let rv_claim_check = F.iter().zip(val.iter()).map(|(f, v)| *f * v).sum::<F>();
        assert_eq!(rv_claim, rv_claim_check, "failed in {val_type:?}");

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
            ra_claim: None,
            val_poly,
            op_key,
            r_cycle,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValTypes,
    ) -> Self {
        let K = sm.get_bytecode().len();
        let (val, rv_claim, r_cycle, op_key) = Self::compute_val_rv(sm, val_type);
        let val_poly = MultilinearPolynomial::from(val);
        let ra_claim = sm.get_opening(op_key);

        Self {
            rv_claim,
            K,
            prover_state: None,
            ra_claim: Some(ra_claim),
            val_poly,
            op_key,
            r_cycle,
        }
    }

    pub fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValTypes,
    ) -> (Vec<F>, F, Vec<F>, OpeningsKeys) {
        match val_type {
            ReadCheckingValTypes::Stage1 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_CIRCUIT_FLAGS + 1 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                    sm.get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm))
                        .unwrap()
                        .r,
                    OpeningsKeys::BytecodeStage1Ra,
                )
            }
            ReadCheckingValTypes::Stage2 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_LOOKUP_TABLES + 3 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                (
                    Self::compute_val_2(sm, &gamma_powers),
                    Self::compute_rv_claim_2(sm, &gamma_powers),
                    sm.get_opening_point(OpeningsKeys::PCSumcheckPC).unwrap().r,
                    OpeningsKeys::BytecodeStage2Ra,
                )
            }
            ReadCheckingValTypes::Stage3 => (
                Self::compute_val_3(sm),
                Self::compute_rv_claim_3(sm),
                sm.get_opening_point(OpeningsKeys::RegistersValEvaluationWa)
                    .unwrap()
                    .r,
                OpeningsKeys::BytecodeStage3Ra,
            ),
        }
    }

    /// Returns a boxed closure that computes:
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
                // TODO: Check if this is in fact correct and not failing because imm is i64
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
        once(sm.get_spartan_z(JoltR1CSInputs::UnexpandedPC))
            .chain(once(sm.get_spartan_z(JoltR1CSInputs::Imm)))
            .chain(CircuitFlags::iter().map(|f| sm.get_spartan_z(JoltR1CSInputs::OpFlags(f))))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a boxed closure that computes:
    ///    Val(k) = unexpanded_pc(k) + gamma * rd(k, r_register) + gamma^2 * rs1(k, r_register)
    ///             + gamma^3 * rs2(k, r_register) + gamma^4 * lookup_table_flags[0](k)
    ///             + gamma^5 * lookup_table_flags[1](k)...
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// and analogously for rs1(k, k') and rs2(k, k').
    /// This particular Val virtualizes claims output by Spartan's "shift" sumcheck,
    /// the instruction execution raf-evaluation sumcheck, the instruction execution
    /// read checking sumcheck, and the registers read/write checking sumcheck.
    pub fn compute_val_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        // TODO: this len shouldn't be caluclated like tihs
        let log_T = sm
            .get_opening_point(OpeningsKeys::PCSumcheckUnexpandedPC)
            .unwrap()
            .r
            .len();
        let r_register = sm
            .get_opening_point(OpeningsKeys::RegistersReadWriteRdWa)
            .unwrap()
            .r;
        let r_register = &r_register[0..r_register.len() - log_T];
        let eq_r_register = EqPolynomial::evals(r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();
                let unexpanded_pc = instr.address;

                let mut linear_combination: F = std::iter::empty()
                    .chain(once(F::from_u64(unexpanded_pc as u64)))
                    // .chain(once(eq_r_register[instr.operands.rd]))
                    // .chain(once(eq_r_register[instr.operands.rs1]))
                    // .chain(once(eq_r_register[instr.operands.rs2]))
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum();

                // linear_combination += eq_r_register[instr.operands.rd] * gamma_powers[1];
                // linear_combination += eq_r_register[instr.operands.rs1] * gamma_powers[2];
                // linear_combination += eq_r_register[instr.operands.rs2] * gamma_powers[3];

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[1 + table_index];
                }

                linear_combination
            })
            .collect()
    }

    pub fn compute_rv_claim_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        once(sm.get_opening(OpeningsKeys::PCSumcheckUnexpandedPC))
            // .chain(once(sm.get_opening(OpeningsKeys::RegistersReadWriteRdWa)))
            // .chain(once(sm.get_opening(OpeningsKeys::RegistersReadWriteRs1Ra)))
            // .chain(once(sm.get_opening(OpeningsKeys::RegistersReadWriteRs2Ra)))
            .chain(
                (0..LookupTables::<WORD_SIZE>::COUNT)
                    .map(|i| sm.get_opening(OpeningsKeys::LookupTableFlag(i))),
            )
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a boxed closure that computes:
    ///    Val(k) = rd(k, r_register)
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes claims output by the Val-evaluation sumcheck
    /// for registers, which outputs a claim of the form rd_wa(r_register, r_cycle)
    pub fn compute_val_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Vec<F> {
        let r_register = sm
            .get_opening_point(OpeningsKeys::RegistersValEvaluationWa)
            .unwrap()
            .r;
        let r_register: Vec<_> = r_register[..(REGISTER_COUNT as usize).log_2()].to_vec();
        let eq_r_register = EqPolynomial::evals(&r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let rd = instruction.normalize().operands.rd;
                eq_r_register[rd]
            })
            .collect()
    }

    pub fn compute_rv_claim_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> F {
        sm.get_opening(OpeningsKeys::RegistersValEvaluationWa)
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for ReadCheckingSumcheck<F> {
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

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim = self.ra_claim.as_ref().expect("ra_claim not set");
        let r: Vec<_> = r.iter().rev().copied().collect();

        // Verify sumcheck_claim = ra_claim * val_eval
        *ra_claim * self.val_poly.evaluate(&r)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for ReadCheckingSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let ra_claim = ps.ra_poly.final_sumcheck_claim();
        let accumulator = accumulator.expect("accumulator should be set");
        accumulator.borrow_mut().append_sparse(
            vec![std::mem::take(&mut ps.unbound_ra_poly)],
            opening_point.r,
            self.r_cycle.clone(),
            vec![ra_claim],
            Some(vec![self.op_key]),
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(self.r_cycle.clone())
            .collect::<Vec<_>>()
            .into();
        let accumulator = accumulator.expect("should be set");
        accumulator
            .borrow_mut()
            .populate_claim_opening(self.op_key, r);
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for ReadCheckingSumcheck<F>
{
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_claim: F,
    rv_claims: [F; 3],
}

impl<F: JoltField, ProofTranscript: Transcript> ReadCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadCheckingProof::prove")]
    pub fn prove(
        _bytecode: &[RV32IMInstruction],
        _F: Vec<F>,
        _K: usize,
        _transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, MultilinearPolynomial<F>) {
        todo!()
        // Used to combine the various fields in each instruction into a single
        // // field element.
        // let gamma_1: F = transcript.challenge_scalar();
        // let compute_val_1 = Self::compute_val_1(gamma_1);
        //
        // // TODO: Get r_register from the registers read/write checking and/or
        // // Val-evaluation sumcheck
        // let r_register: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        // let eq_r_register = EqPolynomial::evals(&r_register);
        //
        // let gamma_2: F = transcript.challenge_scalar();
        // let compute_val_2 = Self::compute_val_2(gamma_2, eq_r_register.clone());
        //
        // let compute_val_3 = Self::compute_val_3(eq_r_register);
        //
        // let (mut read_checking_sumcheck_1, rv_claim_1) =
        //     ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_1);
        //
        // let (mut read_checking_sumcheck_2, rv_claim_2) =
        //     ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_2);
        //
        // let (mut read_checking_sumcheck_3, rv_claim_3) =
        //     ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_3);
        //
        // let (sumcheck_proof, r_address) = BatchedSumcheck::prove(
        //     vec![
        //         &mut read_checking_sumcheck_1,
        //         &mut read_checking_sumcheck_2,
        //         &mut read_checking_sumcheck_3,
        //     ],
        //     transcript,
        // );
        // // BatchedSumcheck::cache_openings(
        // //     vec![
        // //         &mut read_checking_sumcheck_1,
        // //         &mut read_checking_sumcheck_2,
        // //         &mut read_checking_sumcheck_3,
        // //     ],
        // //     openings,
        // //     accumulator,
        // // );
        //
        // let ra_claim = read_checking_sumcheck_1
        //     .ra_claim
        //     .expect("ra_claim should be set after prove_single");
        //
        // let proof = Self {
        //     sumcheck_proof,
        //     ra_claim,
        //     rv_claims: [rv_claim_1, rv_claim_2, rv_claim_3],
        // };
        //
        // let raf_ra = MultilinearPolynomial::from(F);
        // (proof, r_address, raf_ra)
    }

    pub fn verify(
        &self,
        _bytecode: &[RV32IMInstruction],
        _K: usize,
        _transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        todo!()
        // // Used to combine the various fields in each instruction into a single
        // // field element.
        // let gamma_1: F = transcript.challenge_scalar();
        // let compute_val_1 = Self::compute_val_1(gamma_1);
        //
        // // TODO: Get r_register from the registers read/write checking and/or
        // // Val-evaluation sumcheck
        // let r_register: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        // let eq_r_register = EqPolynomial::evals(&r_register);
        //
        // let gamma_2: F = transcript.challenge_scalar();
        // let compute_val_2 = Self::compute_val_2(gamma_2, eq_r_register.clone());
        //
        // let compute_val_3 = Self::compute_val_3(eq_r_register);
        //
        // read_checking_sumcheck_1.ra_claim = Some(self.ra_claim);
        // read_checking_sumcheck_2.ra_claim = Some(self.ra_claim);
        // read_checking_sumcheck_3.ra_claim = Some(self.ra_claim);
        //
        // let r_address = BatchedSumcheck::verify(
        //     &self.sumcheck_proof,
        //     vec![
        //         &read_checking_sumcheck_1,
        //         &read_checking_sumcheck_2,
        //         &read_checking_sumcheck_3,
        //     ],
        //     transcript,
        // )?;
        //
        // Ok(r_address)
    }
}
