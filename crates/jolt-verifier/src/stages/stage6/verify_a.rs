//! Stage 6a verifier helpers.

use jolt_claims::protocols::jolt::{JoltRelationClaims, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, BatchedEvaluationClaim, BatchedSumcheckVerifier,
    SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::FsTranscript;

use super::inputs::Stage6Claims;
use crate::{
    proof::JoltProof,
    stages::zk::{committed, outputs::CommittedOutputClaimOutput},
    verifier::CheckedInputs,
    VerifierError,
};

pub(super) struct Stage6AZkOutput<F: Field, C> {
    pub address_phase_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub address_phase_output_claims: CommittedOutputClaimOutput<C>,
    pub bytecode_address_point: Vec<F>,
    pub bytecode_r_address: Vec<F>,
    pub booleanity_address_point: Vec<F>,
    pub booleanity_r_address: Vec<F>,
}

pub(super) struct Stage6AClearOutput<F: Field> {
    pub address_batch: BatchedEvaluationClaim<F>,
    pub bytecode_address_point: Vec<F>,
    pub bytecode_r_address: Vec<F>,
    pub booleanity_address_point: Vec<F>,
    pub booleanity_r_address: Vec<F>,
    pub bytecode_read_raf_input: F,
    pub booleanity_input: F,
    pub expected_final_claim: F,
}

pub(super) fn verify_zk<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    bytecode_address_claims: &JoltRelationClaims<PCS::Field>,
    booleanity_address_claims: &JoltRelationClaims<PCS::Field>,
) -> Result<Stage6AZkOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsTranscript<PCS::Field>,
{
    let address_statements = vec![
        SumcheckStatement::new(
            bytecode_address_claims.sumcheck.rounds,
            bytecode_address_claims.sumcheck.degree,
        ),
        SumcheckStatement::new(
            booleanity_address_claims.sumcheck.rounds,
            booleanity_address_claims.sumcheck.degree,
        ),
    ];
    let address_phase_consistency = BatchedSumcheckVerifier::verify_committed_consistency(
        &address_statements,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;
    let address_phase_output_claims =
        committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
            checked,
            proof: &proof.stages.stage6a_sumcheck_proof,
            proof_label: "stage6a_sumcheck_proof",
            output_claim_count: 2,
            stage: JoltRelationId::BytecodeReadRaf,
        })?;

    let bytecode_address_point = address_phase_consistency
        .try_instance_point(bytecode_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let booleanity_address_point = address_phase_consistency
        .try_instance_point(booleanity_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_r_address = booleanity_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();

    Ok(Stage6AZkOutput {
        address_phase_consistency,
        address_phase_output_claims,
        bytecode_address_point,
        bytecode_r_address,
        booleanity_address_point,
        booleanity_r_address,
    })
}

pub(super) fn verify_clear<PCS, VC, T, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    claims: &Stage6Claims<PCS::Field>,
    bytecode_address_claims: &JoltRelationClaims<PCS::Field>,
    booleanity_address_claims: &JoltRelationClaims<PCS::Field>,
    bytecode_read_raf_input: PCS::Field,
    booleanity_input: PCS::Field,
) -> Result<Stage6AClearOutput<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsTranscript<PCS::Field>,
{
    let address_sumcheck_claims = vec![
        SumcheckClaim::new(
            bytecode_address_claims.sumcheck.rounds,
            bytecode_address_claims.sumcheck.degree,
            bytecode_read_raf_input,
        ),
        SumcheckClaim::new(
            booleanity_address_claims.sumcheck.rounds,
            booleanity_address_claims.sumcheck.degree,
            booleanity_input,
        ),
    ];
    let address_batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &address_sumcheck_claims,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_address_point = address_batch
        .try_instance_point(bytecode_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?
        .to_vec();
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let booleanity_address_point = address_batch
        .try_instance_point(booleanity_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?
        .to_vec();
    let booleanity_r_address = booleanity_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let address_expected_outputs = [
        claims.address_phase.bytecode_read_raf,
        claims.address_phase.booleanity,
    ];
    if address_batch.batching_coefficients.len() != address_expected_outputs.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 address batch verifier returned {} coefficients for {} instances",
                address_batch.batching_coefficients.len(),
                address_expected_outputs.len()
            ),
        });
    }
    let expected_final_claim = address_batch
        .batching_coefficients
        .iter()
        .zip(address_expected_outputs)
        .map(|(coefficient, output)| *coefficient * output)
        .sum();
    if address_batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }

    append_opening_claims(transcript, claims);

    Ok(Stage6AClearOutput {
        address_batch,
        bytecode_address_point,
        bytecode_r_address,
        booleanity_address_point,
        booleanity_r_address,
        bytecode_read_raf_input,
        booleanity_input,
        expected_final_claim,
    })
}

pub(super) fn append_opening_claims<F, T>(transcript: &mut T, claims: &Stage6Claims<F>)
where
    F: Field,
    T: FsTranscript<F>,
{
    transcript.absorb_field(&claims.address_phase.bytecode_read_raf);
    transcript.absorb_field(&claims.address_phase.booleanity);
}
