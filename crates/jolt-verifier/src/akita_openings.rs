use crate::{
    akita::{
        AkitaClearVectorCommitment, AkitaJoltProof, AkitaPackingBatchProof,
        AkitaPackingProverSetup, AkitaPackingWitnessArtifacts, AkitaVerifierPreprocessing,
    },
    akita_packing::AkitaPackingScheme,
    akita_validation::validate_akita_artifacts_for_proof,
    akita_validity::{attach_akita_packing_validity_proof, prove_akita_jolt_packed_validity},
    stages::stage8::{Stage8BatchStatement, Stage8OpeningId},
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_openings::{
    BatchOpeningScheme, BatchOpeningStatement, PackingWitnessSource, PhysicalView,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;

#[derive(Clone, Copy, Debug)]
pub struct AkitaPrecommittedOpeningInput<'a> {
    pub polynomial: &'a Polynomial<AkitaField>,
    pub hint: &'a AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaStage8ClearOpeningProofs {
    pub packed: AkitaPackingBatchProof,
    pub precommitted: Vec<AkitaPackingBatchProof>,
}

pub fn prove_akita_packing_openings<T, OpeningId, RelationId, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packing opening source layout does not match committed artifact"
                .to_string(),
        });
    }
    if statement.layout_digest != artifacts.layout.digest {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason:
                "Akita packing opening statement layout digest does not match committed artifact"
                    .to_string(),
        });
    }
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packing opening artifacts do not carry a lattice payload".to_string(),
        })?;
    for claim in &statement.claims {
        if claim.commitment != payload.packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: "Akita packing opening statement references a non-artifact commitment"
                    .to_string(),
            });
        }
    }

    AkitaPackingScheme::prove_packing_source_batch(
        setup,
        transcript,
        statement,
        source,
        artifacts.hint.clone(),
    )
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })
}

pub fn prove_akita_stage8_clear_openings<T, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    prove_akita_stage8_clear_openings_with_precommitted(
        setup,
        transcript,
        artifacts,
        source,
        statement,
        &[],
    )
    .map(|proofs| proofs.packed)
}

pub fn prove_akita_stage8_clear_openings_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    let Stage8BatchStatement::Clear(statement) = statement else {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packing opening proving requires a clear Stage 8 statement".to_string(),
        });
    };
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packing opening artifacts do not carry a lattice payload".to_string(),
        })?;
    validate_akita_precommitted_opening_inputs(
        &payload.packed_witness,
        &statement.precommitted_statements,
        precommitted_inputs,
    )?;
    let packed =
        prove_akita_packing_openings(setup, transcript, artifacts, source, &statement.statement)?;
    let precommitted = prove_akita_precommitted_opening_batches(
        setup,
        transcript,
        &payload.packed_witness,
        &statement.precommitted_statements,
        precommitted_inputs,
    )?;
    Ok(AkitaStage8ClearOpeningProofs {
        packed,
        precommitted,
    })
}

fn prove_akita_precommitted_opening_batches<T>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    packed_witness: &AkitaCommitment,
    statements: &[BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >],
    inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<Vec<AkitaPackingBatchProof>, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
{
    validate_akita_precommitted_opening_inputs(packed_witness, statements, inputs)?;

    statements
        .iter()
        .zip(inputs)
        .map(|(statement, input)| {
            AkitaPackingScheme::prove_batch(
                setup,
                transcript,
                statement,
                std::slice::from_ref(input.polynomial),
                vec![input.hint.clone()],
            )
            .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            })
        })
        .collect()
}

fn validate_akita_precommitted_opening_inputs(
    packed_witness: &AkitaCommitment,
    statements: &[BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >],
    inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<(), VerifierError> {
    if statements.len() != inputs.len() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "expected {} Akita precommitted opening inputs, got {}",
                statements.len(),
                inputs.len()
            ),
        });
    }

    for (index, (statement, input)) in statements.iter().zip(inputs).enumerate() {
        validate_akita_precommitted_opening_input(index, packed_witness, statement, input)?;
    }
    Ok(())
}

fn validate_akita_precommitted_opening_input(
    index: usize,
    packed_witness: &AkitaCommitment,
    statement: &BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >,
    input: &AkitaPrecommittedOpeningInput<'_>,
) -> Result<(), VerifierError> {
    if statement.claims.is_empty() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!("Akita precommitted opening statement {index} has no claims"),
        });
    }
    if input.hint.matches_commitment(packed_witness) {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "Akita precommitted opening input {index} must not use the packed witness hint"
            ),
        });
    }
    for claim in &statement.claims {
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "Akita precommitted opening statement {index} must use direct physical views"
                ),
            });
        }
        if claim.commitment == *packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "Akita precommitted opening statement {index} must target a separate precommitted commitment"
                ),
            });
        }
        if !input.hint.matches_commitment(&claim.commitment) {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "Akita precommitted opening input {index} does not match statement commitment"
                ),
            });
        }
    }
    Ok(())
}

pub fn prove_and_attach_akita_opening_proofs<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    prove_and_attach_akita_opening_proofs_with_precommitted::<T, S>(
        setup,
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        artifacts,
        source,
        &[],
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "prover helper mirrors proof attachment inputs and adds precommitted openings"
)]
pub fn prove_and_attach_akita_opening_proofs_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    let mut candidate = proof.clone();
    let validity = prove_akita_jolt_packed_validity::<T, S>(
        setup,
        preprocessing,
        public_io,
        &candidate,
        trusted_advice_commitment,
        artifacts,
        source,
    )?;
    attach_akita_packing_validity_proof(&mut candidate, validity)?;
    let opening_proofs = prove_akita_jolt_final_openings_with_precommitted::<T, S>(
        setup,
        preprocessing,
        public_io,
        &candidate,
        trusted_advice_commitment,
        artifacts,
        source,
        precommitted_inputs,
    )?;
    candidate.joint_opening_proof = opening_proofs.packed;
    candidate.lattice_precommitted_opening_proofs = opening_proofs.precommitted;
    *proof = candidate;
    Ok(())
}

pub fn prove_akita_jolt_final_openings<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::prover_support::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackingScheme,
            AkitaClearVectorCommitment,
            T,
            _,
        >(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            &artifacts.protocol,
        )?;
    prove_akita_stage8_clear_openings(setup, &mut transcript, artifacts, source, &statement)
}

#[expect(
    clippy::too_many_arguments,
    reason = "prover helper mirrors final opening inputs and adds precommitted openings"
)]
pub fn prove_akita_jolt_final_openings_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::prover_support::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackingScheme,
            AkitaClearVectorCommitment,
            T,
            _,
        >(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            &artifacts.protocol,
        )?;
    prove_akita_stage8_clear_openings_with_precommitted(
        setup,
        &mut transcript,
        artifacts,
        source,
        &statement,
        precommitted_inputs,
    )
}
