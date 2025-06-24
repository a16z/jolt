use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;
use crate::jolt::vm::registers::{ReadWriteCheckingProof, RegistersTwistProof, ValEvaluationProof};

impl<F: JoltField, ProofTranscript: Transcript> RegistersTwistProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_T = T.log_2();
        let r: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let r_cycle = self
            .read_write_checking_proof
            .verify(r, r_prime, transcript);

        let (sumcheck_claim, r_cycle_prime) = self.val_evaluation_proof.sumcheck_proof.verify(
            self.read_write_checking_proof.val_claim,
            log_T,
            2,
            transcript,
        )?;

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().rev().zip(r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        assert_eq!(
            sumcheck_claim,
            lt_eval * self.val_evaluation_proof.inc_claim,
            "Val evaluation sumcheck failed"
        );

        // TODO: Append Inc claim to opening proof accumulator

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    pub fn verify(&self, r: Vec<F>, r_prime: Vec<F>, transcript: &mut ProofTranscript) -> Vec<F> {
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        let z: F = transcript.challenge_scalar();

        let (sumcheck_claim, r_sumcheck) = self
            .sumcheck_proof
            .verify(
                self.inc_claim + z * self.rs1_rv_claim + z.square() * self.rs2_rv_claim,
                T.log_2() + K.log_2(),
                3,
                transcript,
            )
            .unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[self.sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..self.sumcheck_switch_index].iter().rev());
        // Final log(K) rounds bind address variables
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::new(r).evaluate(&r_address);

        assert_eq!(
            eq_eval_address
                * eq_eval_cycle
                * self.rd_wa_claim
                * (self.rd_wv_claim - self.val_claim)
                + z * eq_eval_cycle * self.rs1_ra_claim * self.val_claim
                + z.square() * eq_eval_cycle * self.rs2_ra_claim * self.val_claim,
            sumcheck_claim,
            "Read/write-checking sumcheck failed"
        );

        r_cycle
    }
}