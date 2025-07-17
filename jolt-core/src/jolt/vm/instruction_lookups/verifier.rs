use crate::field::JoltField;
use crate::jolt::vm::instruction_lookups::LookupsProof;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::CommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::shout::verify_sparse_dense_shout;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::Transcript;

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn verify(
        &self,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &self.read_checking_proof.sumcheck_proof,
            self.log_T,
            r_cycle.clone(),
            self.read_checking_proof.rv_claim,
            self.read_checking_proof.ra_claims,
            self.read_checking_proof.add_sub_mul_flag_claim,
            &self.read_checking_proof.flag_claims,
            transcript,
        )?;

        let mut r_address: Vec<F> = transcript.challenge_vector(16);
        let z_booleanity: F = transcript.challenge_scalar();
        let z_booleanity_squared: F = z_booleanity.square();
        let z_booleanity_cubed: F = z_booleanity_squared * z_booleanity;
        let (sumcheck_claim, r_booleanity) = self.booleanity_proof.sumcheck_proof.verify(
            F::zero(),
            16 + self.log_T,
            3,
            transcript,
        )?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(16);

        r_address = r_address.into_iter().rev().collect();
        let eq_eval_address = EqPolynomial::mle(&r_address, r_address_prime);
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&r_cycle_rev, r_cycle_prime);

        assert_eq!(
            eq_eval_address
                * eq_eval_cycle
                * ((self.booleanity_proof.ra_claims[0].square()
                    - self.booleanity_proof.ra_claims[0])
                    + z_booleanity
                        * (self.booleanity_proof.ra_claims[1].square()
                            - self.booleanity_proof.ra_claims[1])
                    + z_booleanity_squared
                        * (self.booleanity_proof.ra_claims[2].square()
                            - self.booleanity_proof.ra_claims[2])
                    + z_booleanity_cubed
                        * (self.booleanity_proof.ra_claims[3].square()
                            - self.booleanity_proof.ra_claims[3])),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        let z_hamming_weight: F = transcript.challenge_scalar();
        let z_hamming_weight_squared: F = z_hamming_weight.square();
        let z_hamming_weight_cubed: F = z_hamming_weight_squared * z_hamming_weight;
        let (sumcheck_claim, r_hamming_weight) = self.hamming_weight_proof.sumcheck_proof.verify(
            F::one() + z_hamming_weight + z_hamming_weight_squared + z_hamming_weight_cubed,
            16,
            1,
            transcript,
        )?;

        assert_eq!(
            self.hamming_weight_proof.ra_claims[0]
                + z_hamming_weight * self.hamming_weight_proof.ra_claims[1]
                + z_hamming_weight_squared * self.hamming_weight_proof.ra_claims[2]
                + z_hamming_weight_cubed * self.hamming_weight_proof.ra_claims[3],
            sumcheck_claim,
            "Hamming weight sumcheck failed"
        );

        let r_hamming_weight: Vec<_> = r_hamming_weight.iter().copied().rev().collect();
        for i in 0..4 {
            opening_accumulator.append(
                &[&commitments.commitments[CommittedPolynomials::InstructionRa(i).to_index()]],
                [r_hamming_weight.as_slice(), r_cycle.as_slice()].concat(),
                &[self.hamming_weight_proof.ra_claims[i]],
                transcript,
            );
        }

        Ok(())
    }
}
