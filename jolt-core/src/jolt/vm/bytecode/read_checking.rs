use crate::{
    field::JoltField,
    jolt::{
        instruction::{InstructionFlags, InstructionLookup, NUM_CIRCUIT_FLAGS},
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
    },
    poly::{
        compact_polynomial::SmallScalar,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
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
        compute_val: Box<dyn Fn(&RV32IMInstruction) -> F + Sync>,
    ) -> (Self, F) {
        let val: Vec<_> = bytecode
            .par_iter()
            .map(|instruction| compute_val(instruction))
            .collect();
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
        compute_val: Box<dyn Fn(&RV32IMInstruction) -> F + Sync>,
    ) -> Self {
        let val: Vec<_> = bytecode
            .par_iter()
            .map(|instruction| compute_val(instruction))
            .collect();
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

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ReadCheckingSumcheck<F>
{
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
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

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

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claim.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim = Some(prover_state.ra_poly.final_sumcheck_claim());
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim = self.ra_claim.as_ref().expect("ra_claim not set");
        let r: Vec<_> = r.iter().rev().copied().collect();

        // Verify sumcheck_claim = ra_claim * val_eval
        *ra_claim * self.val_poly.evaluate(&r)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_claim: F,
    rv_claims: [F; 2],
}

impl<F: JoltField, ProofTranscript: Transcript> ReadCheckingProof<F, ProofTranscript> {
    /// Returns a boxed closure that computes:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(gamma: F) -> Box<dyn Fn(&RV32IMInstruction) -> F + Sync> {
        let mut gamma_powers = vec![F::one()];
        for _ in 0..NUM_CIRCUIT_FLAGS + 1 {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }

        let closure = move |instruction: &RV32IMInstruction| {
            let NormalizedInstruction {
                address: unexpanded_pc,
                operands,
                ..
            } = instruction.normalize();

            let mut linear_combination = F::zero();
            linear_combination += (unexpanded_pc as u64).field_mul(gamma_powers[0]);
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
        };
        Box::new(closure)
    }

    /// Returns a boxed closure that computes:
    ///    Val(k) = unexpanded_pc(k) + gamma * lookup_table_flags[0](k)
    ///             + gamma^2 * lookup_table_flags[1](k)...
    /// This particular Val virtualizes claims output by Spartan's "shift" sumcheck,
    /// the instruction execution raf-evaluation sumcheck, the instruction execution
    /// read checking sumcheck.
    fn compute_val_2(gamma: F) -> Box<dyn Fn(&RV32IMInstruction) -> F + Sync> {
        let mut gamma_powers = vec![F::one()];
        for _ in 0..NUM_LOOKUP_TABLES {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }

        let closure = move |instruction: &RV32IMInstruction| {
            let NormalizedInstruction {
                address: unexpanded_pc,
                ..
            } = instruction.normalize();

            let mut linear_combination = F::zero();
            linear_combination += (unexpanded_pc as u64).field_mul(gamma_powers[0]);

            if let Some(table) = instruction.lookup_table() {
                let table_index = LookupTables::enum_index(&table);
                linear_combination += gamma_powers[1 + table_index];
            }

            linear_combination
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

        let gamma_2: F = transcript.challenge_scalar();
        let compute_val_2 = Self::compute_val_2(gamma_2);

        let (mut read_checking_sumcheck_1, rv_claim_1) =
            ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_1);

        let (mut read_checking_sumcheck_2, rv_claim_2) =
            ReadCheckingSumcheck::new_prover(bytecode, F.clone(), K, compute_val_2);

        let (sumcheck_proof, r_address) = BatchedSumcheck::prove(
            vec![&mut read_checking_sumcheck_1, &mut read_checking_sumcheck_2],
            transcript,
        );

        let ra_claim = read_checking_sumcheck_1
            .ra_claim
            .expect("ra_claim should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim,
            rv_claims: [rv_claim_1, rv_claim_2],
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

        let gamma_2: F = transcript.challenge_scalar();
        let compute_val_2 = Self::compute_val_2(gamma_2);

        let mut read_checking_sumcheck_1 =
            ReadCheckingSumcheck::new_verifier(bytecode, self.rv_claims[0], K, compute_val_1);

        let mut read_checking_sumcheck_2 =
            ReadCheckingSumcheck::new_verifier(bytecode, self.rv_claims[1], K, compute_val_2);

        read_checking_sumcheck_1.ra_claim = Some(self.ra_claim);
        read_checking_sumcheck_2.ra_claim = Some(self.ra_claim);

        let r_address = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![&read_checking_sumcheck_1, &read_checking_sumcheck_2],
            transcript,
        )?;

        Ok(r_address)
    }
}
