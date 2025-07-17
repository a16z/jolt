use crate::jolt::vm::registers::{
    RegistersTwistProof, ValEvaluationSumcheck, ValEvaluationSumcheckClaims,
    ValEvaluationVerifierState,
};
use crate::jolt::vm::registers_read_write_checking::RegistersReadWriteChecking;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::CommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::sumcheck::BatchableSumcheckInstance;
use crate::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};

impl<F: JoltField, ProofTranscript: Transcript> RegistersTwistProof<F, ProofTranscript> {
    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        T: usize,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_T = T.log_2();
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) = RegistersReadWriteChecking::verify(
            &self.read_write_checking_proof,
            &r_prime,
            transcript,
        )?;

        let sumcheck_instance = ValEvaluationSumcheck {
            claimed_evaluation: self.read_write_checking_proof.claims.val_claim,
            prover_state: None,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: log_T,
                r_address,
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim: self.val_evaluation_proof.inc_claim,
                wa_claim: self.val_evaluation_proof.wa_claim,
            }),
        };

        let mut r_cycle_prime = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
            &self.val_evaluation_proof.sumcheck_proof,
            transcript,
        )?;

        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_commitment = &commitments.commitments[CommittedPolynomials::RdInc.to_index()];
        opening_accumulator.append(
            &[inc_commitment],
            r_cycle_prime,
            &[self.val_evaluation_proof.inc_claim],
            transcript,
        );

        // TODO: Append Inc claim to opening proof accumulator

        Ok(())
    }
}
