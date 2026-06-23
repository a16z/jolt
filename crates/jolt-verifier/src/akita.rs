//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    akita_packed::AkitaPackedScheme,
    akita_validation::{
        validate_akita_advice_commitment_aliases, validate_akita_artifacts_for_proof,
        validate_akita_packed_opening_proof_payload_shape,
        validate_akita_precommitted_commitment_aliases,
        validate_akita_precommitted_opening_proof_payload_shapes,
        validate_akita_proof_payload_shape, validate_akita_verifier_setup_config,
    },
    akita_witness::JoltPackedWitnessBuilder,
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{
        ClearOnlyCommitment, ClearOnlyVectorCommitment, CommitmentPayload, JoltProof,
        JoltProofClaims, LatticeCommitmentPayload,
    },
    stages::{
        stage7::inputs::LatticePackedValidityOutputClaims,
        stage8::{
            build_lattice_packed_validity_batch, derive_lattice_packed_validity_requirements,
            derive_lattice_packed_validity_statements, field_element_canonical_factors,
            field_element_canonical_value_from_openings, lattice_packed_validity_claims,
            lattice_packed_validity_opening_count, lattice_packing_family_id,
            sample_lattice_packed_validity_eq_points,
            validate_lattice_packed_witness_layout_config, FieldCanonicalFactor,
            LatticePackedValidityStatement, LatticePackedValidityStatementKind,
            Stage8BatchStatement, Stage8OpeningId,
        },
        PrecommittedSchedule,
    },
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_claims::protocols::jolt::{
    lattice_packed_validity_digest, JoltAdviceKind, LatticePackedFamilyId,
    LatticePackedValidityKind, LatticePackedValidityRequirement,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::{
    BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, PackedAdviceKind,
    PackedFactDomain, PackedFamilyId, PackedLinearBatchProof, PackedWitnessLayout,
    PackedWitnessSource, PhysicalView, SparsePackedWitness,
};
use jolt_poly::{try_eq_mle, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
use jolt_sumcheck::{
    append_sumcheck_claim, BatchedEvaluationClaim, ClearProof, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, EvaluationClaim, RoundMessage, SumcheckProof,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

#[cfg(test)]
use crate::akita_validation::{
    validate_akita_opening_proof_payload_shape, validate_akita_precommitted_commitment_is_separate,
    validate_akita_verifier_setup_layout,
};

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaPackedBatchProof = PackedLinearBatchProof<AkitaBatchProof>;
pub type AkitaPackedProverSetup = <AkitaPackedScheme as CommitmentScheme>::ProverSetup;
pub type AkitaPackedVerifierSetup = <AkitaPackedScheme as CommitmentScheme>::VerifierSetup;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackedScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackedScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackedWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaPackedJoltWitnessInput<'a> {
    pub layout: PackedWitnessLayout,
    pub trace_rows: &'a [JoltTraceRow],
    pub log_k_chunk: usize,
    pub instruction_lookup_indices: &'a [u128],
    pub untrusted_advice: Option<&'a [u8]>,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackedWitnessArtifacts,
    pub witness: SparsePackedWitness<AkitaField>,
}

#[derive(Clone, Copy, Debug)]
pub struct AkitaPrecommittedOpeningInput<'a> {
    pub polynomial: &'a Polynomial<AkitaField>,
    pub hint: &'a AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaStage8ClearOpeningProofs {
    pub packed: AkitaPackedBatchProof,
    pub precommitted: Vec<AkitaPackedBatchProof>,
}

#[derive(Clone, Debug)]
pub struct AkitaPackedValidityProofArtifacts {
    pub sumcheck_proof: SumcheckProof<AkitaField, ClearOnlyCommitment>,
    pub opening_claims: LatticePackedValidityOutputClaims<AkitaField>,
    pub opening_proof: AkitaPackedBatchProof,
}

impl AkitaPackedWitnessArtifacts {
    pub fn payload(&self) -> Option<&LatticeCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_lattice()
    }
}

pub fn build_akita_packed_jolt_witness(
    input: AkitaPackedJoltWitnessInput<'_>,
) -> Result<SparsePackedWitness<AkitaField>, VerifierError> {
    validate_akita_jolt_packed_witness_layout(&input.layout)?;
    let protocol = akita_lattice_protocol_config_for_layout(&input.layout);
    validate_lattice_packed_witness_layout_config(&protocol, &input.layout)?;

    if input.instruction_lookup_indices.len() != input.trace_rows.len() {
        return Err(akita_witness_error(format!(
            "instruction lookup index count {} does not match trace row count {}",
            input.instruction_lookup_indices.len(),
            input.trace_rows.len()
        )));
    }

    let mut builder = JoltPackedWitnessBuilder::new(input.layout.clone());
    builder
        .pack_trace_rows(
            input.trace_rows,
            input.log_k_chunk,
            |row, _| input.instruction_lookup_indices[row],
            |_, row| (row.is_load() || row.is_store()).then(|| row.ram_address()),
        )
        .map(|_| ())
        .map_err(akita_witness_error)?;

    pack_untrusted_advice_bytes(&mut builder, input.untrusted_advice)?;

    builder.finish().map_err(akita_witness_error)
}

pub fn commit_akita_packed_jolt_witness(
    setup: &AkitaPackedProverSetup,
    input: AkitaPackedJoltWitnessInput<'_>,
) -> Result<AkitaCommittedPackedJoltWitness, VerifierError> {
    let witness = build_akita_packed_jolt_witness(input)?;
    let artifacts = commit_akita_packed_witness(setup, &witness)?;
    Ok(AkitaCommittedPackedJoltWitness { artifacts, witness })
}

fn validate_akita_jolt_packed_witness_layout(
    layout: &PackedWitnessLayout,
) -> Result<(), VerifierError> {
    for family in &layout.families {
        if jolt_packed_witness_family_is_precommitted(&family.id) {
            return Err(VerifierError::InvalidProtocolConfig {
                reason: format!(
                    "precommitted family {:?} cannot be included in the Akita packed witness layout",
                    family.id
                ),
            });
        }
    }
    Ok(())
}

fn jolt_packed_witness_family_is_precommitted(family: &PackedFamilyId) -> bool {
    matches!(
        family,
        PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Trusted,
            ..
        } | PackedFamilyId::BytecodeChunk { .. }
            | PackedFamilyId::BytecodeRegisterSelector { .. }
            | PackedFamilyId::BytecodeCircuitFlag { .. }
            | PackedFamilyId::BytecodeInstructionFlag { .. }
            | PackedFamilyId::BytecodeLookupSelector { .. }
            | PackedFamilyId::BytecodeRafFlag { .. }
            | PackedFamilyId::BytecodeUnexpandedPcBytes { .. }
            | PackedFamilyId::BytecodeImmBytes { .. }
            | PackedFamilyId::ProgramImageInit
    )
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackedWitnessLayout,
) -> JoltProtocolConfig {
    let validity_requirements = akita_lattice_validity_requirements_for_layout(layout);
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            validity_digest: Some(lattice_packed_validity_digest(&validity_requirements)),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: false,
            untrusted: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
    };
    config
}

pub fn akita_lattice_validity_requirements_for_layout(
    layout: &PackedWitnessLayout,
) -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = layout
        .families
        .iter()
        .filter_map(|family| {
            let limbs = family.limbs;
            let alphabet_size = family.alphabet.size();
            match family.id {
                PackedFamilyId::UnsignedIncChunk { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::UnsignedIncChunk { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::UnsignedIncMsb => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::UnsignedIncMsb,
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackedFamilyId::FieldRdIncByte { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::FieldRdIncByte { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::AdviceBytes { kind, index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::AdviceBytes {
                            kind: jolt_advice_kind(kind),
                            index,
                        },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackedFamilyId::BytecodeLookupSelector { chunk } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeLookupSelector { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::BytecodeRafFlag { chunk } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeRafFlag { chunk },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::BytecodeImmBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeImmBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::ProgramImageInit => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::ProgramImageInit,
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::InstructionRa { .. }
                | PackedFamilyId::BytecodeRa { .. }
                | PackedFamilyId::RamRa { .. }
                | PackedFamilyId::FieldRdIncSign
                | PackedFamilyId::BytecodeChunk { .. }
                | PackedFamilyId::Custom { .. } => None,
            }
        })
        .collect::<Vec<_>>();
    for family in &layout.families {
        let PackedFamilyId::BytecodeCircuitFlag { chunk, flag } = &family.id else {
            continue;
        };
        let chunk = *chunk;
        if *flag == CircuitFlags::Store as usize
            && layout
                .family(&PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 })
                .is_some()
        {
            requirements.push(LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk));
        }
    }
    requirements
}

pub fn commit_akita_packed_witness<S>(
    setup: &AkitaPackedProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packed_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packed_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaPackedProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_lattice_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackedScheme::commit_packed_source(setup, source).map_err(|error| {
            VerifierError::AkitaCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = LatticeCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_lattice_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackedWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Lattice(payload),
        hint,
    })
}

pub fn prove_akita_packed_openings<T, OpeningId, RelationId, S>(
    setup: &AkitaPackedProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<AkitaPackedBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening source layout does not match committed artifact"
                .to_string(),
        });
    }
    if statement.layout_digest != artifacts.layout.digest {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason:
                "Akita packed opening statement layout digest does not match committed artifact"
                    .to_string(),
        });
    }
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening artifacts do not carry a lattice payload".to_string(),
        })?;
    for claim in &statement.claims {
        if claim.commitment != payload.packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: "Akita packed opening statement references a non-artifact commitment"
                    .to_string(),
            });
        }
    }

    AkitaPackedScheme::prove_packed_source_batch(
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
    setup: &AkitaPackedProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
) -> Result<AkitaPackedBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
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
    setup: &AkitaPackedProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    let Stage8BatchStatement::Clear(statement) = statement else {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening proving requires a clear Stage 8 statement".to_string(),
        });
    };
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening artifacts do not carry a lattice payload".to_string(),
        })?;
    validate_akita_precommitted_opening_inputs(
        &payload.packed_witness,
        &statement.precommitted_statements,
        precommitted_inputs,
    )?;
    let packed =
        prove_akita_packed_openings(setup, transcript, artifacts, source, &statement.statement)?;
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
    setup: &AkitaPackedProverSetup,
    transcript: &mut T,
    packed_witness: &AkitaCommitment,
    statements: &[BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >],
    inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<Vec<AkitaPackedBatchProof>, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
{
    validate_akita_precommitted_opening_inputs(packed_witness, statements, inputs)?;

    statements
        .iter()
        .zip(inputs)
        .map(|(statement, input)| {
            AkitaPackedScheme::prove_batch(
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

pub fn prove_akita_packed_validity<T, S>(
    setup: &AkitaPackedProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<AkitaPackedValidityProofArtifacts, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(
            VerifierError::LatticePackedValidityOpeningVerificationFailed {
                reason: "Akita packed validity source layout does not match committed artifact"
                    .to_string(),
            },
        );
    }

    let requirements = derive_lattice_packed_validity_requirements(
        &artifacts.protocol,
        log_k_chunk,
        precommitted,
    )?;
    let statements = derive_lattice_packed_validity_statements(&artifacts.layout, &requirements)?;
    let eq_points =
        sample_lattice_packed_validity_eq_points(transcript, &artifacts.layout, &statements);
    let sumcheck_claims = lattice_packed_validity_claims::<AkitaField>(&statements);
    for claim in &sumcheck_claims {
        append_sumcheck_claim(transcript, &claim.claimed_sum);
    }
    let batching_coefficients = (0..statements.len())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let max_num_vars = statements
        .iter()
        .map(|statement| statement.num_vars)
        .max()
        .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "cannot prove an empty Akita packed validity batch".to_string(),
        })?;
    let max_degree = statements
        .iter()
        .map(|statement| statement.degree)
        .max()
        .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "cannot prove an empty Akita packed validity batch".to_string(),
        })?;

    let (compressed, reduction) = prove_combined_validity_sumcheck(
        source,
        &statements,
        &eq_points,
        &batching_coefficients,
        max_num_vars,
        max_degree,
        transcript,
    )?;
    let mut opening_claims = Vec::with_capacity(lattice_packed_validity_opening_count(&statements));
    for statement in &statements {
        let point = reduction
            .try_instance_point(statement.num_vars)
            .map_err(|error| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: error.to_string(),
            })?;
        opening_claims.extend(validity_opening_values(source, statement, point)?);
    }
    let batch = build_lattice_packed_validity_batch(
        &artifacts.layout,
        &statements,
        artifacts
            .payload()
            .ok_or_else(
                || VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: "Akita packed validity artifacts do not carry a lattice payload"
                        .to_string(),
                },
            )?
            .packed_witness
            .clone(),
        &eq_points,
        &reduction,
        &opening_claims,
    )?;
    if reduction.reduction.value != batch.expected_final_claim {
        return Err(VerifierError::LatticePackedValidityOutputMismatch);
    }
    let opening_proof =
        prove_akita_packed_openings(setup, transcript, artifacts, source, &batch.statement)?;

    Ok(AkitaPackedValidityProofArtifacts {
        sumcheck_proof: SumcheckProof::Clear(ClearProof::Compressed(compressed)),
        opening_claims: LatticePackedValidityOutputClaims { opening_claims },
        opening_proof,
    })
}

pub fn attach_akita_packed_validity_proof(
    proof: &mut AkitaJoltProof,
    validity: AkitaPackedValidityProofArtifacts,
) -> Result<(), VerifierError> {
    proof.stages.lattice_packed_validity_sumcheck_proof = Some(validity.sumcheck_proof);
    proof.lattice_packed_validity_opening_proof = Some(validity.opening_proof);
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(VerifierError::UnexpectedBlindFoldProof);
    };
    claims.stage7.lattice_packed_validity = Some(validity.opening_claims);
    Ok(())
}

pub fn prove_akita_jolt_packed_validity<T, S>(
    setup: &AkitaPackedProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
) -> Result<AkitaPackedValidityProofArtifacts, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (checked, mut transcript) =
        crate::verifier::lattice_packed_validity_transcript_with_config::<
            AkitaField,
            AkitaPackedScheme,
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
    prove_akita_packed_validity(
        setup,
        &mut transcript,
        artifacts,
        source,
        proof.one_hot_config.committed_chunk_bits(),
        &checked.precommitted,
    )
}

pub fn prove_and_attach_akita_opening_proofs<T, S>(
    setup: &AkitaPackedProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
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
    setup: &AkitaPackedProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
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
    attach_akita_packed_validity_proof(&mut candidate, validity)?;
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
    setup: &AkitaPackedProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
) -> Result<AkitaPackedBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::verifier::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackedScheme,
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
    setup: &AkitaPackedProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::verifier::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackedScheme,
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

pub fn verify_akita_clear<T>(
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    <AkitaField as WithAccumulator>::Accumulator: RingAccumulator<Element = AkitaField>,
{
    validate_akita_verifier_setup_config(&preprocessing.pcs_setup, config)?;
    validate_akita_proof_payload_shape(&preprocessing.pcs_setup, &proof.commitments)?;
    validate_akita_packed_opening_proof_payload_shape(
        &proof.commitments,
        &proof.joint_opening_proof,
        "Akita joint opening proof",
    )?;
    if let Some(opening_proof) = &proof.lattice_packed_validity_opening_proof {
        validate_akita_packed_opening_proof_payload_shape(
            &proof.commitments,
            opening_proof,
            "Akita lattice packed validity opening proof",
        )?;
    }
    validate_akita_precommitted_opening_proof_payload_shapes(
        &proof.commitments,
        &proof.lattice_precommitted_opening_proofs,
    )?;
    validate_akita_advice_commitment_aliases(
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
    )?;
    validate_akita_precommitted_commitment_aliases(
        preprocessing,
        &proof.commitments,
        trusted_advice_commitment,
    )?;
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackedScheme,
        AkitaClearVectorCommitment,
        T,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
}

fn pack_untrusted_advice_bytes(
    builder: &mut JoltPackedWitnessBuilder,
    bytes: Option<&[u8]>,
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| {
            matches!(
                id,
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Untrusted,
                    index: 0,
                }
            )
        },
        "untrusted advice bytes",
    )?;
    let Some(expected) = expected else {
        if bytes.is_none_or(<[u8]>::is_empty) {
            return Ok(());
        }
        return Err(akita_witness_error(format!(
            "{} were supplied but the packed layout has no matching advice family",
            "untrusted advice bytes"
        )));
    };
    let padded = padded_slice(
        bytes.unwrap_or_default(),
        expected,
        "untrusted advice bytes",
    )?;
    builder
        .pack_untrusted_advice_bytes(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn expected_rows_for_family(
    layout: &PackedWitnessLayout,
    mut matches_family: impl FnMut(&PackedFamilyId) -> bool,
    domain: &'static str,
) -> Result<Option<usize>, VerifierError> {
    let mut rows = None;
    for family in &layout.families {
        if !matches_family(&family.id) {
            continue;
        }
        let got = packed_domain_rows(family.domain)?;
        match rows {
            Some(expected) if expected != got => {
                return Err(akita_witness_error(format!(
                    "{domain} layout row count mismatch: expected {expected}, got {got}"
                )));
            }
            Some(_) => {}
            None => rows = Some(got),
        }
    }
    Ok(rows)
}

fn packed_domain_rows(domain: PackedFactDomain) -> Result<usize, VerifierError> {
    let log_rows = match domain {
        PackedFactDomain::TraceRows { log_t } => log_t,
        PackedFactDomain::BytecodeRows { log_bytecode } => log_bytecode,
        PackedFactDomain::ProgramImageWords { log_words } => log_words,
        PackedFactDomain::AdviceBytes { log_bytes, .. } => log_bytes,
    };
    1usize
        .checked_shl(log_rows as u32)
        .ok_or_else(|| akita_witness_error("packed witness domain row count overflow"))
}

fn padded_slice<T: Clone + Default>(
    values: &[T],
    expected: usize,
    domain: &'static str,
) -> Result<Vec<T>, VerifierError> {
    if values.len() > expected {
        return Err(akita_witness_error(format!(
            "{domain} length {} exceeds packed layout size {expected}",
            values.len()
        )));
    }
    let mut padded = values.to_vec();
    padded.resize_with(expected, T::default);
    Ok(padded)
}

fn prove_combined_validity_sumcheck<T, S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    max_degree: usize,
    transcript: &mut T,
) -> Result<
    (
        CompressedSumcheckProof<AkitaField>,
        BatchedEvaluationClaim<AkitaField>,
    ),
    VerifierError,
>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    let mut challenges = Vec::with_capacity(max_num_vars);
    let mut round_polynomials = Vec::with_capacity(max_num_vars);
    for _ in 0..max_num_vars {
        let remaining = max_num_vars - challenges.len() - 1;
        let round_evals = (0..=max_degree)
            .map(|point| {
                let mut prefix = challenges.clone();
                prefix.push(AkitaField::from_u64(point as u64));
                sum_combined_validity_over_suffix(
                    source,
                    statements,
                    eq_points,
                    batching_coefficients,
                    max_num_vars,
                    &prefix,
                    remaining,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let round_poly = UnivariatePoly::from_evals(&round_evals);
        let compressed =
            CompressedLabeledRoundPoly::new(&round_poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, AkitaField> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        let challenge = transcript.challenge();
        round_polynomials.push(round_poly.compress());
        challenges.push(challenge);
    }

    let value = combined_validity_value(
        source,
        statements,
        eq_points,
        batching_coefficients,
        max_num_vars,
        &challenges,
    )?;

    Ok((
        CompressedSumcheckProof { round_polynomials },
        BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(challenges, value),
            batching_coefficients: batching_coefficients.to_vec(),
            max_num_vars,
            max_degree,
        },
    ))
}

fn sum_combined_validity_over_suffix<S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    prefix: &[AkitaField],
    remaining: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let suffix_count = checked_power_of_two(remaining, "packed validity suffix")?;
    let mut sum = AkitaField::zero();
    for suffix in 0..suffix_count {
        let mut point = prefix.to_vec();
        append_boolean_bits(&mut point, suffix, remaining);
        sum += combined_validity_value(
            source,
            statements,
            eq_points,
            batching_coefficients,
            max_num_vars,
            &point,
        )?;
    }
    Ok(sum)
}

fn combined_validity_value<S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    for ((statement, eq_point), coefficient) in
        statements.iter().zip(eq_points).zip(batching_coefficients)
    {
        let offset = max_num_vars
            .checked_sub(statement.num_vars)
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity statement has more variables than the combined batch"
                    .to_string(),
            })?;
        let instance_point = point
            .get(offset..offset + statement.num_vars)
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity instance point is out of range".to_string(),
            })?;
        value += *coefficient * validity_value(source, statement, eq_point, instance_point)?;
    }
    Ok(value)
}

fn validity_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    eq_point: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let eq_mask = try_eq_mle(point, eq_point).map_err(|error| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: error.to_string(),
        }
    })?;
    let value = match statement.kind {
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            let openings = validity_opening_values(source, statement, point)?;
            openings[0] * openings[1]
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            let openings = validity_opening_values(source, statement, point)?;
            field_element_canonical_value_from_openings(statement, &openings)?
        }
        _ => validity_violation(
            statement.kind,
            validity_opening_value(source, statement, point)?,
        ),
    };
    Ok(eq_mask * value)
}

fn validity_opening_values<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<Vec<AkitaField>, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    if statement.kind == LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint {
        return Ok(vec![
            bytecode_store_rd_disjoint_factor_value(source, statement, point, 0)?,
            bytecode_store_rd_disjoint_factor_value(source, statement, point, 1)?,
        ]);
    }
    if statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes {
        let factors = field_element_canonical_factors(&statement.requirement)?;
        return factors
            .into_iter()
            .map(|factor| field_element_canonical_factor_value(source, point, factor))
            .collect();
    }
    validity_opening_value(source, statement, point).map(|value| vec![value])
}

fn validity_opening_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let family_id = lattice_packing_family_id(&statement.requirement.family);
    let shape = validity_statement_shape(source.layout(), statement, &family_id)?;
    let point_parts = split_validity_point(statement.kind, point, shape)?;
    let row_weights = EqPolynomial::<AkitaField>::evals(point_parts.row, None);
    let limb_weights = EqPolynomial::<AkitaField>::evals(point_parts.limb, None);
    match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity => {
            let symbol_weights = EqPolynomial::<AkitaField>::evals(point_parts.symbol, None);
            weighted_family_value(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::Point(&symbol_weights),
            )
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => weighted_family_value(
            source,
            &family_id,
            &row_weights,
            &limb_weights,
            SymbolWeights::All,
        ),
        LatticePackedValidityStatementKind::BooleanIndicator => {
            let LatticePackedValidityKind::BooleanIndicator { symbol } = statement.requirement.kind
            else {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "boolean-indicator validity statement has non-indicator requirement"
                        .to_string(),
                });
            };
            weighted_family_value(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::Fixed(symbol),
            )
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            Err(VerifierError::InvalidProtocolConfig {
                reason: "bytecode Store/Rd disjointness has multiple opening factors".to_string(),
            })
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            Err(VerifierError::InvalidProtocolConfig {
                reason: "field-element canonical-byte validity has multiple opening factors"
                    .to_string(),
            })
        }
    }
}

fn validity_violation(kind: LatticePackedValidityStatementKind, opening: AkitaField) -> AkitaField {
    match kind {
        LatticePackedValidityStatementKind::CellBooleanity
        | LatticePackedValidityStatementKind::BooleanIndicator
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            opening * (opening - AkitaField::one())
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum => {
            let difference = opening - AkitaField::one();
            difference * difference
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => opening,
    }
}

fn weighted_direct_symbol_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    weighted_direct_limb_symbol_value(source, family_id, row_weights, 0, symbol)
}

fn weighted_direct_limb_symbol_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    let mut error = None;
    source.for_each_nonzero(|rank, cell| {
        if error.is_some() {
            return;
        }
        let Some(address) = source.layout().unrank(rank) else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity source emitted out-of-layout rank {rank}"),
                },
            );
            return;
        };
        if &address.family != family_id || address.limb != limb || address.symbol != symbol {
            return;
        }
        let Some(row_weight) = row_weights.get(address.row).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", address.row),
                },
            );
            return;
        };
        value += row_weight * cell;
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(value)
}

fn field_element_canonical_factor_value<S>(
    source: &S,
    point: &[AkitaField],
    factor: FieldCanonicalFactor,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let (family, limb, symbol_filter) = match factor {
        FieldCanonicalFactor::Eq {
            family,
            limb,
            symbol,
            ..
        } => {
            return weighted_field_canonical_symbol_value(source, point, &family, limb, symbol);
        }
        FieldCanonicalFactor::Range {
            family,
            limb,
            start_symbol,
            ..
        } => (family, limb, start_symbol..256),
    };

    let mut value = AkitaField::zero();
    for symbol in symbol_filter {
        value += weighted_field_canonical_symbol_value(source, point, &family, limb, symbol)?;
    }
    Ok(value)
}

fn weighted_field_canonical_symbol_value<S>(
    source: &S,
    point: &[AkitaField],
    family_id: &PackedFamilyId,
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let family =
        source
            .layout()
            .family(family_id)
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: format!("field-element canonical-byte factor requires {family_id:?}"),
            })?;
    if limb >= family.limbs || family.alphabet.size() != 256 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "field-element canonical-byte factor {family_id:?} must be a byte family"
            ),
        });
    }
    let rows = family
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!(
                "field-element canonical-byte factor {family_id:?} has invalid row domain: {error}"
            ),
        })?;
    let row_vars = power_of_two_log(rows, "field-element canonical-byte row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "field-element canonical-byte point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    weighted_direct_limb_symbol_value(source, family_id, &row_weights, limb, symbol)
}

fn bytecode_store_rd_disjoint_factor_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
    factor: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let chunk = bytecode_store_rd_disjoint_chunk(&statement.requirement)?;
    let store_id = PackedFamilyId::BytecodeCircuitFlag {
        chunk,
        flag: CircuitFlags::Store as usize,
    };
    let store =
        source
            .layout()
            .family(&store_id)
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: format!("bytecode Store/Rd disjointness requires {store_id:?}"),
            })?;
    let rows = store
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("bytecode Store/Rd disjointness row domain is invalid: {error}"),
        })?;
    let row_vars = power_of_two_log(rows, "bytecode Store/Rd disjointness row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "bytecode Store/Rd disjointness point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    match factor {
        0 => weighted_direct_symbol_value(source, &store_id, &row_weights, 1),
        1 => {
            let rd_id = PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
            let rd = source.layout().family(&rd_id).ok_or_else(|| {
                VerifierError::InvalidProtocolConfig {
                    reason: format!("bytecode Store/Rd disjointness requires {rd_id:?}"),
                }
            })?;
            if rd.domain != store.domain || rd.limbs != 1 {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "bytecode Store/Rd disjointness rd selector layout mismatch"
                        .to_string(),
                });
            }
            let limb_weights = [AkitaField::one()];
            weighted_family_value(
                source,
                &rd_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::All,
            )
        }
        _ => Err(VerifierError::InvalidProtocolConfig {
            reason: format!("bytecode Store/Rd disjointness has no opening factor {factor}"),
        }),
    }
}

fn bytecode_store_rd_disjoint_chunk(
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } = &requirement.family else {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "bytecode Store/Rd disjointness must be anchored on the Store circuit flag"
                .to_string(),
        });
    };
    if *flag != CircuitFlags::Store as usize
        || requirement.limbs != 1
        || requirement.alphabet_size != 2
    {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "bytecode Store/Rd disjointness must be anchored on a boolean Store circuit flag"
                    .to_string(),
        });
    }
    Ok(*chunk)
}

#[derive(Clone, Copy)]
struct ValidityStatementShape {
    row: usize,
    limb: usize,
    symbol: usize,
}

struct ValidityPointParts<'a> {
    row: &'a [AkitaField],
    limb: &'a [AkitaField],
    symbol: &'a [AkitaField],
}

fn validity_statement_shape(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    family_id: &PackedFamilyId,
) -> Result<ValidityStatementShape, VerifierError> {
    let family = layout
        .family(family_id)
        .ok_or_else(|| VerifierError::InvalidProtocolConfig {
            reason: format!("packed validity statement references missing family {family_id:?}"),
        })?;
    if family.limbs != statement.requirement.limbs {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "packed validity family {family_id:?} limb count mismatch: layout has {}, statement has {}",
                family.limbs, statement.requirement.limbs
            ),
        });
    }
    if family.alphabet.size() != statement.requirement.alphabet_size {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "packed validity family {family_id:?} alphabet mismatch: layout has {}, statement has {}",
                family.alphabet.size(),
                statement.requirement.alphabet_size
            ),
        });
    }
    let rows = family
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("packed validity family {family_id:?} has invalid row domain: {error}"),
        })?;
    Ok(ValidityStatementShape {
        row: power_of_two_log(rows, "packed validity row count")?,
        limb: power_of_two_log(statement.requirement.limbs, "packed validity limb count")?,
        symbol: power_of_two_log(
            statement.requirement.alphabet_size,
            "packed validity alphabet size",
        )?,
    })
}

fn split_validity_point(
    kind: LatticePackedValidityStatementKind,
    point: &[AkitaField],
    shape: ValidityStatementShape,
) -> Result<ValidityPointParts<'_>, VerifierError> {
    let expected = match kind {
        LatticePackedValidityStatementKind::CellBooleanity => shape.row + shape.limb + shape.symbol,
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum
        | LatticePackedValidityStatementKind::BooleanIndicator => shape.row + shape.limb,
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => shape.row,
    };
    if point.len() != expected {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "packed validity point has {} variables but statement requires {expected}",
                point.len()
            ),
        });
    }
    let row_end = shape.row;
    let limb_end = row_end + shape.limb;
    let symbol_end = limb_end + shape.symbol;
    let symbol = if matches!(kind, LatticePackedValidityStatementKind::CellBooleanity) {
        &point[limb_end..symbol_end]
    } else {
        &[]
    };
    Ok(ValidityPointParts {
        row: &point[..row_end],
        limb: &point[row_end..limb_end],
        symbol,
    })
}

enum SymbolWeights<'a> {
    Point(&'a [AkitaField]),
    All,
    Fixed(usize),
}

fn weighted_family_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    limb_weights: &[AkitaField],
    symbol_weights: SymbolWeights<'_>,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    let mut error = None;
    source.for_each_nonzero(|rank, cell| {
        if error.is_some() {
            return;
        }
        let Some(address) = source.layout().unrank(rank) else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity source emitted out-of-layout rank {rank}"),
                },
            );
            return;
        };
        if &address.family != family_id {
            return;
        }
        let Some(row_weight) = row_weights.get(address.row).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", address.row),
                },
            );
            return;
        };
        let Some(limb_weight) = limb_weights.get(address.limb).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity limb {} is outside limb weights",
                        address.limb
                    ),
                },
            );
            return;
        };
        let symbol_weight = match symbol_weights {
            SymbolWeights::Point(weights) => {
                let Some(weight) = weights.get(address.symbol).copied() else {
                    error = Some(
                        VerifierError::LatticePackedValidityOpeningVerificationFailed {
                            reason: format!(
                                "packed validity symbol {} is outside symbol weights",
                                address.symbol
                            ),
                        },
                    );
                    return;
                };
                weight
            }
            SymbolWeights::All => AkitaField::one(),
            SymbolWeights::Fixed(symbol) => {
                if address.symbol == symbol {
                    AkitaField::one()
                } else {
                    AkitaField::zero()
                }
            }
        };
        value += row_weight * limb_weight * symbol_weight * cell;
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(value)
}

fn append_boolean_bits(point: &mut Vec<AkitaField>, index: usize, bits: usize) {
    for bit in 0..bits {
        let shift = bits - 1 - bit;
        point.push(AkitaField::from_u64(((index >> shift) & 1) as u64));
    }
}

fn checked_power_of_two(bits: usize, name: &'static str) -> Result<usize, VerifierError> {
    1usize.checked_shl(bits as u32).ok_or_else(|| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!("{name} dimension is too large"),
        }
    })
}

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{name} must be a power of two, got {value}"),
        })
    }
}

fn akita_witness_error(reason: impl ToString) -> VerifierError {
    VerifierError::AkitaCommitmentFailed {
        reason: format!(
            "Akita packed witness packing failed: {}",
            reason.to_string()
        ),
    }
}

fn jolt_advice_kind(kind: PackedAdviceKind) -> JoltAdviceKind {
    match kind {
        PackedAdviceKind::Trusted => JoltAdviceKind::Trusted,
        PackedAdviceKind::Untrusted => JoltAdviceKind::Untrusted,
    }
}

fn layout_has_field_rd_inc(layout: &PackedWitnessLayout) -> bool {
    layout
        .families
        .iter()
        .any(|family| matches!(family.id, PackedFamilyId::FieldRdIncByte { .. }))
}

fn layout_has_advice(layout: &PackedWitnessLayout, kind: PackedAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackedFamilyId::AdviceBytes {
                kind: family_kind,
                ..
            } if family_kind == kind
        )
    })
}

#[cfg(test)]
#[path = "akita_tests.rs"]
mod tests;
