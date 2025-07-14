use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    jolt::{
        instruction::{InstructionFlags, InstructionLookup, NUM_CIRCUIT_FLAGS},
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, BatchedSumcheck, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use rayon::prelude::*;
use tracer::instruction::{NormalizedInstruction, RV32IMInstruction};

struct ReadCheckingProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
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
}

impl<F: JoltField> ReadCheckingSumcheck<F> {
    pub fn new_prover(
        bytecode: &[RV32IMInstruction],
        F: Vec<F>,
        K: usize,
        compute_val: Box<dyn Fn(&[RV32IMInstruction]) -> Vec<F> + Sync>,
    ) -> (Self, F) {
        let val = compute_val(bytecode);
        let rv_claim: F = F
            .par_iter()
            .zip(val.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();

        let ra_poly = MultilinearPolynomial::from(F);
        let val_poly = MultilinearPolynomial::from(val);

        let prover = Self {
            rv_claim,
            K,
            prover_state: Some(ReadCheckingProverState { ra_poly }),
            ra_claim: None,
            val_poly,
        };

        (prover, rv_claim)
    }

    pub fn new_verifier(
        bytecode: &[RV32IMInstruction],
        rv_claim: F,
        K: usize,
        compute_val: Box<dyn Fn(&[RV32IMInstruction]) -> Vec<F> + Sync>,
    ) -> Self {
        let val = compute_val(bytecode);
        let val_poly = MultilinearPolynomial::from(val);

        Self {
            rv_claim,
            K,
            prover_state: None,
            ra_claim: None,
            val_poly,
        }
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
        let degree = <Self as BatchableSumcheckInstance<F>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    prover_state
                        .ra_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let val_evals = self
                    .val_poly
                    .sumcheck_evals(i, degree, BindingOrder::LowToHigh);

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
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || {
                prover_state
                    .ra_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
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
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
    ) {
        debug_assert!(self.ra_claim.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim = Some(prover_state.ra_poly.final_sumcheck_claim());
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_claim: F,
    rv_claims: [F; 3],
}

impl<F: JoltField, ProofTranscript: Transcript> ReadCheckingProof<F, ProofTranscript> {
    /// Returns a boxed closure that computes:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(gamma: F) -> Box<dyn Fn(&[RV32IMInstruction]) -> Vec<F> + Sync> {
        let mut gamma_powers = vec![F::one()];
        for _ in 0..NUM_CIRCUIT_FLAGS + 1 {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }

        let closure = move |bytecode: &[RV32IMInstruction]| {
            bytecode
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
        };
        Box::new(closure)
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
    fn compute_val_2(
        gamma: F,
        eq_r_register: Vec<F>,
    ) -> Box<dyn Fn(&[RV32IMInstruction]) -> Vec<F> + Sync> {
        let mut gamma_powers = vec![F::one()];
        for _ in 0..NUM_LOOKUP_TABLES + 3 {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }

        let closure = move |bytecode: &[RV32IMInstruction]| {
            bytecode
                .par_iter()
                .map(|instruction| {
                    let instr = instruction.normalize();
                    let unexpanded_pc = instr.address;

                    let mut linear_combination = F::zero();
                    linear_combination += F::from_u64(unexpanded_pc as u64);

                    linear_combination += eq_r_register[instr.operands.rd] * gamma_powers[1];
                    linear_combination += eq_r_register[instr.operands.rs1] * gamma_powers[2];
                    linear_combination += eq_r_register[instr.operands.rs2] * gamma_powers[3];

                    if let Some(table) = instruction.lookup_table() {
                        let table_index = LookupTables::enum_index(&table);
                        linear_combination += gamma_powers[4 + table_index];
                    }

                    linear_combination
                })
                .collect()
        };
        Box::new(closure)
    }

    /// Returns a boxed closure that computes:
    ///    Val(k) = rd(k, r_register)
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes claims output by the Val-evaluation sumcheck
    /// for registers, which outputs a claim of the form rd_wa(r_register, r_cycle)
    fn compute_val_3(eq_r_register: Vec<F>) -> Box<dyn Fn(&[RV32IMInstruction]) -> Vec<F> + Sync> {
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        let closure = move |bytecode: &[RV32IMInstruction]| {
            bytecode
                .par_iter()
                .map(|instruction| {
                    let rd = instruction.normalize().operands.rd;
                    eq_r_register[rd]
                })
                .collect()
        };
        Box::new(closure)
    }

    #[tracing::instrument(skip_all, name = "ReadCheckingProof::prove")]
    pub fn prove(
        bytecode: &[RV32IMInstruction],
        F: Vec<F>,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, MultilinearPolynomial<F>) {
        // Used to combine the various fields in each instruction into a single
        // field element.
        let gamma_1: F = transcript.challenge_scalar();
        let compute_val_1 = Self::compute_val_1(gamma_1);

        // TODO: Get r_register from the registers read/write checking and/or
        // Val-evaluation sumcheck
        let r_register: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        let eq_r_register = EqPolynomial::evals(&r_register);

        let gamma_2: F = transcript.challenge_scalar();
        let compute_val_2 = Self::compute_val_2(gamma_2, eq_r_register.clone());

        let compute_val_3 = Self::compute_val_3(eq_r_register);

        let (mut read_checking_sumcheck_1, rv_claim_1) =
            ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_1);

        let (mut read_checking_sumcheck_2, rv_claim_2) =
            ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_2);

        let (mut read_checking_sumcheck_3, rv_claim_3) =
            ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_3);

        let (sumcheck_proof, r_address) = BatchedSumcheck::prove(
            vec![
                &mut read_checking_sumcheck_1,
                &mut read_checking_sumcheck_2,
                &mut read_checking_sumcheck_3,
            ],
            transcript,
        );
        // BatchedSumcheck::cache_openings(
        //     vec![
        //         &mut read_checking_sumcheck_1,
        //         &mut read_checking_sumcheck_2,
        //         &mut read_checking_sumcheck_3,
        //     ],
        //     openings,
        //     accumulator,
        // );

        let ra_claim = read_checking_sumcheck_1
            .ra_claim
            .expect("ra_claim should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim,
            rv_claims: [rv_claim_1, rv_claim_2, rv_claim_3],
        };

        let raf_ra = MultilinearPolynomial::from(F);
        (proof, r_address, raf_ra)
    }

    pub fn verify(
        &self,
        bytecode: &[RV32IMInstruction],
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        // Used to combine the various fields in each instruction into a single
        // field element.
        let gamma_1: F = transcript.challenge_scalar();
        let compute_val_1 = Self::compute_val_1(gamma_1);

        // TODO: Get r_register from the registers read/write checking and/or
        // Val-evaluation sumcheck
        let r_register: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        let eq_r_register = EqPolynomial::evals(&r_register);

        let gamma_2: F = transcript.challenge_scalar();
        let compute_val_2 = Self::compute_val_2(gamma_2, eq_r_register.clone());

        let compute_val_3 = Self::compute_val_3(eq_r_register);

        let mut read_checking_sumcheck_1 =
            ReadCheckingSumcheck::new_verifier(bytecode, self.rv_claims[0], K, compute_val_1);

        let mut read_checking_sumcheck_2 =
            ReadCheckingSumcheck::new_verifier(bytecode, self.rv_claims[1], K, compute_val_2);

        let mut read_checking_sumcheck_3 =
            ReadCheckingSumcheck::new_verifier(bytecode, self.rv_claims[2], K, compute_val_3);

        read_checking_sumcheck_1.ra_claim = Some(self.ra_claim);
        read_checking_sumcheck_2.ra_claim = Some(self.ra_claim);
        read_checking_sumcheck_3.ra_claim = Some(self.ra_claim);

        let r_address = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![
                &read_checking_sumcheck_1,
                &read_checking_sumcheck_2,
                &read_checking_sumcheck_3,
            ],
            transcript,
        )?;

        Ok(r_address)
    }
}
