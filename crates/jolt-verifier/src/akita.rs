//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    akita_witness::JoltPackedWitnessBuilder,
    config::{
        validate_protocol_config, AdviceLatticeConfig, FieldInlineLatticeConfig,
        IncrementCommitmentMode, JoltProtocolConfig, LatticeConfig, PackedWitnessConfig, PcsFamily,
        ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{
        ClearOnlyCommitment, ClearOnlyVectorCommitment, CommitmentPayload, JoltProof,
        JoltProofClaims, LatticeCommitmentPayload,
    },
    stages::{
        stage7::inputs::LatticePackedValidityOutputClaims,
        stage8::{
            akita_packed_family_id, build_lattice_packed_validity_batch,
            derive_akita_packed_validity_requirements, derive_akita_packed_validity_statements,
            field_element_canonical_factors, field_element_canonical_value_from_openings,
            lattice_packed_validity_claims, lattice_packed_validity_opening_count,
            sample_lattice_packed_validity_eq_points, validate_akita_packed_witness_layout_config,
            FieldCanonicalFactor, LatticePackedValidityStatement,
            LatticePackedValidityStatementKind, Stage8BatchStatement, Stage8OpeningId,
        },
        PrecommittedSchedule,
    },
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaPackedBatchProof, AkitaPackedScheme, AkitaProverHint,
    AkitaProverSetup, AkitaVerifierSetup,
};
use jolt_claims::protocols::jolt::{
    lattice_packed_validity_digest, JoltAdviceKind, LatticePackedFamilyId,
    LatticePackedValidityKind, LatticePackedValidityRequirement,
};
use jolt_field::{FixedByteSize, RingAccumulator, WithAccumulator};
use jolt_openings::{
    BatchOpeningScheme, BatchOpeningStatement, PackedAdviceKind, PackedFactDomain, PackedFamilyId,
    PackedWitnessLayout, PackedWitnessSource, PhysicalView, SparsePackedWitness,
};
use jolt_poly::{try_eq_mle, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
use jolt_sumcheck::{
    append_sumcheck_claim, BatchedEvaluationClaim, ClearProof, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, EvaluationClaim, RoundMessage, SumcheckProof,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
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
    validate_akita_packed_witness_layout_config(&protocol, &input.layout)?;

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
    setup: &AkitaProverSetup,
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
            field_rd_inc_family: layout_has_field_rd_inc(layout),
            trusted_advice_family: false,
            untrusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Untrusted),
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_akita_packed_witness_layout_config(&protocol, &layout)?;
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
    setup: &AkitaProverSetup,
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
            reason: "Akita packed opening artifacts do not carry an Akita payload".to_string(),
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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
            reason: "Akita packed opening artifacts do not carry an Akita payload".to_string(),
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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
            VerifierError::AkitaPackedValidityOpeningVerificationFailed {
                reason: "Akita packed validity source layout does not match committed artifact"
                    .to_string(),
            },
        );
    }

    let requirements =
        derive_akita_packed_validity_requirements(&artifacts.protocol, log_k_chunk, precommitted)?;
    let statements = derive_akita_packed_validity_statements(&artifacts.layout, &requirements)?;
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
        .ok_or_else(|| VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: "cannot prove an empty Akita packed validity batch".to_string(),
        })?;
    let max_degree = statements
        .iter()
        .map(|statement| statement.degree)
        .max()
        .ok_or_else(|| VerifierError::AkitaPackedValiditySumcheckFailed {
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
            .map_err(|error| VerifierError::AkitaPackedValiditySumcheckFailed {
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
                || VerifierError::AkitaPackedValidityOpeningVerificationFailed {
                    reason: "Akita packed validity artifacts do not carry an Akita payload"
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
        return Err(VerifierError::AkitaPackedValidityOutputMismatch);
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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
    setup: &AkitaProverSetup,
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

fn validate_akita_artifacts_for_proof(
    setup: &AkitaVerifierSetup,
    proof_protocol: &JoltProtocolConfig,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    artifacts: &AkitaPackedWitnessArtifacts,
) -> Result<(), VerifierError> {
    validate_akita_verifier_setup_layout(setup, &artifacts.layout)?;
    validate_akita_packed_witness_layout_config(&artifacts.protocol, &artifacts.layout)?;
    if proof_protocol != &artifacts.protocol {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: artifacts.protocol,
            got: *proof_protocol,
        });
    }
    let artifact_payload =
        artifacts
            .payload()
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita proof assembly requires Akita packed witness artifacts".to_string(),
            })?;
    let proof_payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if proof_payload != artifact_payload {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita proof commitments do not match packed witness artifacts".to_string(),
        });
    }
    Ok(())
}

fn validate_akita_verifier_setup_config(
    setup: &AkitaVerifierSetup,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup requires lattice PCS mode".to_string(),
        });
    }

    let packed_witness = config.lattice.packed_witness;
    let expected_digest =
        packed_witness
            .layout_digest
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires a packed witness layout digest".to_string(),
            })?;
    let expected_dimension =
        packed_witness
            .d_pack
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires D_pack".to_string(),
            })?;
    validate_akita_verifier_setup_shape(setup, expected_digest, expected_dimension)?;

    let setup_layout =
        setup
            .packed_layout
            .as_ref()
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires a packed witness layout".to_string(),
            })?;
    if setup_layout.digest != expected_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout digest does not match protocol config".to_string(),
        });
    }
    if setup_layout.dimension != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout dimension does not match protocol D_pack"
                .to_string(),
        });
    }

    Ok(())
}

fn validate_akita_proof_payload_shape(
    setup: &AkitaVerifierSetup,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    validate_akita_verifier_setup_shape(setup, payload.layout_digest, payload.d_pack)?;
    if payload.packed_witness.layout_digest != payload.layout_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita packed witness commitment layout digest does not match proof payload"
                .to_string(),
        });
    }
    if payload.packed_witness.num_vars != payload.d_pack {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita packed witness commitment dimension does not match proof payload D_pack"
                .to_string(),
        });
    }
    if payload.packed_witness.poly_count != 1 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita packed witness commitment must contain exactly one polynomial"
                .to_string(),
        });
    }
    validate_akita_commitment_bytes(&payload.packed_witness)?;
    Ok(())
}

fn validate_akita_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackedBatchProof,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if opening_proof.native.commitment != payload.packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof commitment does not match packed witness payload"
                .to_string(),
        });
    }
    validate_akita_commitment_bytes(&opening_proof.native.commitment)?;
    if opening_proof.native.statement_bridge.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing statement bridge bytes".to_string(),
        });
    }
    if opening_proof.native.proof_shape.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof shape bytes".to_string(),
        });
    }
    if opening_proof.native.proof.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof bytes".to_string(),
        });
    }
    if let Some(reduction) = &opening_proof.reduction {
        validate_akita_field_bytes(
            "Akita packed reduction opening eval",
            &reduction.opening_eval,
        )?;
        for round in &reduction.rounds {
            for eval in round {
                validate_akita_field_bytes("Akita packed reduction round eval", eval)?;
            }
        }
    }
    Ok(())
}

fn validate_akita_packed_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackedBatchProof,
    field: &'static str,
) -> Result<(), VerifierError> {
    validate_akita_opening_proof_payload_shape(proof_commitments, opening_proof)?;
    if opening_proof.reduction.is_none() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{field} must include a packed reduction"),
        });
    }
    Ok(())
}

fn validate_akita_precommitted_opening_proof_payload_shapes(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proofs: &[AkitaPackedBatchProof],
) -> Result<(), VerifierError> {
    for opening_proof in opening_proofs {
        validate_akita_precommitted_opening_proof_payload_shape(proof_commitments, opening_proof)?;
    }
    Ok(())
}

fn validate_akita_precommitted_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackedBatchProof,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if opening_proof.native.commitment == payload.packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "Akita precommitted opening proof must target a separate precommitted commitment"
                    .to_string(),
        });
    }
    if opening_proof.reduction.is_some() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita precommitted opening proof must not include a packed reduction"
                .to_string(),
        });
    }
    validate_akita_native_opening_proof_payload_shape(&opening_proof.native)
}

fn validate_akita_native_opening_proof_payload_shape(
    opening_proof: &jolt_akita::AkitaBatchProof,
) -> Result<(), VerifierError> {
    validate_akita_commitment_bytes(&opening_proof.commitment)?;
    if opening_proof.statement_bridge.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing statement bridge bytes".to_string(),
        });
    }
    if opening_proof.proof_shape.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof shape bytes".to_string(),
        });
    }
    if opening_proof.proof.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof bytes".to_string(),
        });
    }
    Ok(())
}

fn validate_akita_commitment_bytes(commitment: &AkitaCommitment) -> Result<(), VerifierError> {
    if commitment.native.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita commitment is missing native commitment bytes".to_string(),
        });
    }
    Ok(())
}

fn validate_akita_field_bytes(label: &'static str, bytes: &[u8]) -> Result<(), VerifierError> {
    if bytes.len() != AkitaField::NUM_BYTES {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "{label} has {} bytes but expected {}",
                bytes.len(),
                AkitaField::NUM_BYTES
            ),
        });
    }
    let value = u128::from_le_bytes(bytes.try_into().map_err(|_| {
        VerifierError::InvalidProtocolConfig {
            reason: format!("{label} must be exactly {} bytes", AkitaField::NUM_BYTES),
        }
    })?);
    if value >= jolt_akita::AKITA_FIELD_MODULUS {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{label} is not a canonical Akita field encoding"),
        });
    }
    Ok(())
}

fn validate_akita_advice_commitment_aliases(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    untrusted_advice_commitment: Option<&AkitaCommitment>,
    trusted_advice_commitment: Option<&AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if untrusted_advice_commitment.is_some_and(|commitment| commitment != &payload.packed_witness) {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita untrusted advice commitment must alias the packed witness commitment"
                .to_string(),
        });
    }
    if let Some(commitment) = trusted_advice_commitment {
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            commitment,
            "trusted advice",
        )?;
    }
    Ok(())
}

fn validate_akita_precommitted_commitment_aliases(
    preprocessing: &AkitaVerifierPreprocessing,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    trusted_advice_commitment: Option<&AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if let Some(commitment) = trusted_advice_commitment {
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            commitment,
            "trusted advice",
        )?;
    }
    if let Some(committed) = preprocessing.program.committed() {
        for commitment in &committed.bytecode_chunk_commitments {
            validate_akita_precommitted_commitment_is_separate(
                &payload.packed_witness,
                commitment,
                "bytecode chunk",
            )?;
        }
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            &committed.program_image_commitment,
            "program image",
        )?;
    }
    Ok(())
}

fn validate_akita_precommitted_commitment_is_separate(
    packed_witness: &AkitaCommitment,
    precommitted: &AkitaCommitment,
    label: &'static str,
) -> Result<(), VerifierError> {
    if precommitted == packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("Akita {label} commitment must be separate from packed witness"),
        });
    }
    validate_akita_commitment_bytes(precommitted)
}

fn validate_akita_verifier_setup_layout(
    setup: &AkitaVerifierSetup,
    layout: &PackedWitnessLayout,
) -> Result<(), VerifierError> {
    validate_akita_verifier_setup_shape(setup, layout.digest, layout.dimension)?;
    let setup_layout =
        setup
            .packed_layout
            .as_ref()
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires a packed witness layout".to_string(),
            })?;
    if setup_layout != layout {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout does not match packed witness artifact layout"
                .to_string(),
        });
    }

    Ok(())
}

fn validate_akita_verifier_setup_shape(
    setup: &AkitaVerifierSetup,
    expected_digest: [u8; 32],
    expected_dimension: usize,
) -> Result<(), VerifierError> {
    if setup.default_layout_digest != expected_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout digest does not match packed witness layout"
                .to_string(),
        });
    }
    if setup.max_num_vars != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup max_num_vars does not match packed witness dimension"
                .to_string(),
        });
    }
    if setup.max_num_polys_per_commitment_group == 0 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "Akita verifier setup must support at least one polynomial per commitment group"
                    .to_string(),
        });
    }
    if setup.native.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup is missing native setup bytes".to_string(),
        });
    }

    Ok(())
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
            .ok_or_else(|| VerifierError::AkitaPackedValiditySumcheckFailed {
                reason: "packed validity statement has more variables than the combined batch"
                    .to_string(),
            })?;
        let instance_point = point
            .get(offset..offset + statement.num_vars)
            .ok_or_else(|| VerifierError::AkitaPackedValiditySumcheckFailed {
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
        VerifierError::AkitaPackedValiditySumcheckFailed {
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
    let family_id = akita_packed_family_id(&statement.requirement.family);
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
                VerifierError::AkitaPackedValidityOpeningVerificationFailed {
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
                VerifierError::AkitaPackedValidityOpeningVerificationFailed {
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
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
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
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
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
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
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
                VerifierError::AkitaPackedValidityOpeningVerificationFailed {
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
                VerifierError::AkitaPackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", address.row),
                },
            );
            return;
        };
        let Some(limb_weight) = limb_weights.get(address.limb).copied() else {
            error = Some(
                VerifierError::AkitaPackedValidityOpeningVerificationFailed {
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
                        VerifierError::AkitaPackedValidityOpeningVerificationFailed {
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
        VerifierError::AkitaPackedValiditySumcheckFailed {
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
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful artifact construction"
    )]

    use super::*;
    use crate::stages::stage8::{
        Stage8BatchStatement, Stage8ClearBatchStatement, Stage8LogicalManifest, Stage8OpeningId,
        Stage8PhysicalManifest,
    };
    use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use jolt_akita::{AkitaScheme, AkitaSetupParams, AKITA_FIELD_MODULUS};
    use jolt_claims::protocols::jolt::{
        bytecode_imm_canonical_bytes_requirement,
        formulas::{
            dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
            ra::JoltRaPolynomialLayout,
        },
        unsigned_inc_msb_opening, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
        PackedAlphabet, PackedCellAddress, PackedFactDomain, PackedFamilySpec, PackedLinearTerm,
        PhysicalView, SparsePackedWitness,
    };
    use jolt_poly::Point;
    use jolt_riscv::{
        CapturedState, CircuitFlags, JoltInstructionKind, JoltInstructionRow, JoltTraceRow,
        NonMemoryState, NormalizedOperands, StoreState,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> PackedWitnessLayout {
        let specs = vec![
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncMsb,
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Bit,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::FieldRdIncByte { index },
                    PackedFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackedAlphabet::Byte,
                )
            }));
            specs
        };
        PackedWitnessLayout::new(specs).expect("layout should build")
    }

    fn packed_cell(family: PackedFamilyId, symbol: usize) -> PackedCellAddress {
        packed_cell_at(family, 0, 0, symbol)
    }

    fn packed_cell_at(
        family: PackedFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> PackedCellAddress {
        PackedCellAddress {
            family,
            row,
            limb,
            symbol,
        }
    }

    fn instruction(
        kind: JoltInstructionKind,
        address: usize,
        operands: NormalizedOperands,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: kind,
            address,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn trace_row(
        kind: JoltInstructionKind,
        operands: NormalizedOperands,
        state: CapturedState,
        bytecode_pc: u32,
    ) -> JoltTraceRow {
        JoltTraceRow::from_components(
            state,
            &instruction(kind, 0x8000_0000 + (bytecode_pc as usize * 4), operands),
            bytecode_pc,
        )
        .expect("trace row should build")
    }

    fn af(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    #[test]
    fn protocol_config_binds_layout_digest_and_dimension() {
        let layout = tiny_layout();

        let config = akita_lattice_protocol_config_for_layout(&layout);

        assert_eq!(
            config.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
        assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
        assert_eq!(
            config.lattice.packed_witness.validity_digest,
            Some(lattice_packed_validity_digest(
                &akita_lattice_validity_requirements_for_layout(&layout)
            ))
        );
        assert_eq!(config.lattice.program_mode, ProgramMode::Committed);
        assert_eq!(
            config.lattice.increment_mode,
            IncrementCommitmentMode::FusedOneHot
        );
    }

    #[test]
    fn commits_packed_witness_and_returns_verifier_payload() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(
            layout.clone(),
            vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
        )
        .expect("source should build");

        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");

        assert_eq!(artifact.layout, layout);
        let payload = artifact
            .payload()
            .expect("artifact should carry lattice payload");
        assert_eq!(payload.layout_digest, layout.digest);
        assert_eq!(payload.d_pack, layout.dimension);
        assert_eq!(payload.packed_witness.layout_digest, layout.digest);
        assert_eq!(payload.packed_witness.num_vars, layout.dimension);
        assert_eq!(
            artifact.protocol.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
    }

    #[test]
    fn commits_jolt_packed_witness_inputs_with_padding() {
        let specs = vec![
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::RamRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Untrusted,
                    index: 0,
                },
                PackedFactDomain::AdviceBytes {
                    kind: PackedAdviceKind::Untrusted,
                    log_bytes: 2,
                },
                1,
                PackedAlphabet::Byte,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::FieldRdIncByte { index },
                    PackedFactDomain::TraceRows { log_t: 1 },
                    1,
                    PackedAlphabet::Byte,
                )
            }));
            specs
        };
        let layout = PackedWitnessLayout::new(specs).expect("layout should build");
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 4,
                    rd_write_value: 7,
                }),
                0,
            ),
            trace_row(
                JoltInstructionKind::SD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: None,
                    imm: 8,
                },
                CapturedState::Store(StoreState {
                    rs1_value: 1,
                    rs2_value: 11,
                    ram_read_value: 10,
                    ram_address: 0x34,
                }),
                1,
            ),
        ];
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);

        let committed = commit_akita_packed_jolt_witness(
            &prover_setup,
            AkitaPackedJoltWitnessInput {
                layout: layout.clone(),
                trace_rows: &rows,
                log_k_chunk: 8,
                instruction_lookup_indices: &[0xaa, 0xbb],
                untrusted_advice: Some(&[7, 8]),
            },
        )
        .expect("Jolt packed witness should build and commit");

        assert_eq!(committed.artifacts.layout, layout);
        let payload = committed
            .artifacts
            .payload()
            .expect("artifact should carry lattice payload");
        assert_eq!(payload.layout_digest, layout.digest);

        let witness = &committed.witness;
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::InstructionRa { index: 0 },
                    0,
                    0,
                    0xaa,
                ))
                .expect("instruction RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::BytecodeRa { index: 0 },
                    1,
                    0,
                    1,
                ))
                .expect("bytecode RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::RamRa { index: 0 },
                    1,
                    0,
                    0x34
                ))
                .expect("RAM RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::UnsignedIncChunk { index: 0 },
                    0,
                    0,
                    3
                ))
                .expect("increment cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::AdviceBytes {
                        kind: PackedAdviceKind::Untrusted,
                        index: 0,
                    },
                    2,
                    0,
                    0,
                ))
                .expect("padded untrusted advice byte should exist"),
            AkitaField::one()
        );
    }

    #[test]
    fn build_jolt_packed_witness_rejects_precommitted_layout_families() {
        let forbidden_specs = [
            PackedFamilySpec::direct(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    index: 0,
                },
                PackedFactDomain::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    log_bytes: 0,
                },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeChunk { index: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackedAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::ProgramImageInit,
                PackedFactDomain::ProgramImageWords { log_words: 0 },
                8,
                PackedAlphabet::Byte,
            ),
        ];

        for spec in forbidden_specs {
            let layout =
                PackedWitnessLayout::new([spec]).expect("forbidden layout should still parse");

            let error = build_akita_packed_jolt_witness(AkitaPackedJoltWitnessInput {
                layout,
                trace_rows: &[],
                log_k_chunk: 8,
                instruction_lookup_indices: &[],
                untrusted_advice: None,
            })
            .expect_err("precommitted packed-witness layout should reject");

            assert!(
                matches!(
                    error,
                    VerifierError::InvalidProtocolConfig { ref reason }
                        if reason.contains("precommitted family")
                ),
                "unexpected error: {error:?}"
            );
        }
    }

    #[test]
    fn packed_witness_artifacts_feed_akita_packed_batch_verifier() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let instruction_family = PackedFamilyId::InstructionRa { index: 0 };
        let sign_family = PackedFamilyId::UnsignedIncMsb;
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(instruction_family.clone(), 7),
                    AkitaField::one(),
                ),
                (packed_cell(sign_family.clone(), 1), AkitaField::one()),
            ],
        )
        .expect("source should build");
        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifact
            .payload()
            .expect("artifact should carry lattice payload")
            .packed_witness
            .clone();
        let instruction_claim = AkitaField::from_u64(2);
        let sign_claim = AkitaField::from_u64(3);
        let instruction_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(0),
            JoltRelationId::HammingWeightClaimReduction,
        ));
        let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
        let statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![
                BatchOpeningClaim {
                    id: instruction_id,
                    relation: instruction_id,
                    commitment: commitment.clone(),
                    claim: instruction_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(2),
                            instruction_family.physical_ref(),
                            0,
                            7,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(3),
                },
                BatchOpeningClaim {
                    id: sign_id,
                    relation: sign_id,
                    commitment: commitment.clone(),
                    claim: sign_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(3),
                            sign_family.physical_ref(),
                            0,
                            1,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(7),
                },
            ],
        };
        let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
            logical_manifest: Stage8LogicalManifest {
                openings: Vec::new(),
                pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            },
            physical_manifest: Stage8PhysicalManifest {
                openings: Vec::new(),
                layout_digest: layout.digest,
            },
            opening_ids: vec![instruction_id, sign_id],
            opening_claims: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            statement: statement.clone(),
            precommitted_statements: Vec::new(),
        });

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let proof = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &stage8_statement,
        )
        .expect("packed batch proof should be produced");
        validate_akita_opening_proof_payload_shape(&artifact.commitments, &proof)
            .expect("fresh packed batch proof shape should pass preflight");

        let mut wrong_stage8_statement = stage8_statement.clone();
        let Stage8BatchStatement::Clear(wrong_statement) = &mut wrong_stage8_statement else {
            unreachable!("test statement is clear");
        };
        wrong_statement.statement.claims[0].commitment.layout_digest = [9; 32];
        let mut wrong_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let error = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut wrong_transcript,
            &artifact,
            &source,
            &wrong_stage8_statement,
        )
        .expect_err("non-artifact commitment should reject");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { .. }
        ));

        let mut wrong_commitment_proof = proof.clone();
        wrong_commitment_proof.native.commitment.layout_digest = [9; 32];
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(
                &artifact.commitments,
                &wrong_commitment_proof,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("opening proof commitment")
        ));

        let mut missing_native_proof = proof.clone();
        missing_native_proof.native.proof.clear();
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(&artifact.commitments, &missing_native_proof),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native proof bytes")
        ));

        let mut missing_reduction = proof.clone();
        missing_reduction.reduction = None;
        assert!(matches!(
            validate_akita_packed_opening_proof_payload_shape(
                &artifact.commitments,
                &missing_reduction,
                "Akita joint opening proof",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("packed reduction")
        ));

        let mut missing_reduction_eval = proof.clone();
        missing_reduction_eval
            .reduction
            .as_mut()
            .expect("packed proof should contain a reduction")
            .opening_eval
            .clear();
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(
                &artifact.commitments,
                &missing_reduction_eval,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("packed reduction opening eval")
        ));

        let mut noncanonical_reduction_eval = proof.clone();
        noncanonical_reduction_eval
            .reduction
            .as_mut()
            .expect("packed proof should contain a reduction")
            .opening_eval = AKITA_FIELD_MODULUS.to_le_bytes().to_vec();
        assert!(matches!(
            validate_akita_packed_opening_proof_payload_shape(
                &artifact.commitments,
                &noncanonical_reduction_eval,
                "Akita joint opening proof",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("canonical Akita field encoding")
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed batch proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(
            result.reduced_opening,
            result.coefficients[0] * instruction_claim + result.coefficients[1] * sign_claim
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn stage8_clear_openings_prove_separate_precommitted_batches() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let sign_family = PackedFamilyId::UnsignedIncMsb;
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [(packed_cell(sign_family.clone(), 1), AkitaField::one())],
        )
        .expect("source should build");
        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let packed_commitment = artifact
            .payload()
            .expect("artifact should carry lattice payload")
            .packed_witness
            .clone();
        let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
        let packed_statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![BatchOpeningClaim {
                id: sign_id,
                relation: sign_id,
                commitment: packed_commitment.clone(),
                claim: AkitaField::from_u64(3),
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: vec![PackedLinearTerm::new(
                        AkitaField::from_u64(3),
                        sign_family.physical_ref(),
                        0,
                        1,
                    )
                    .with_row_point(Vec::new())],
                },
                scale: AkitaField::from_u64(7),
            }],
        };

        let precommitted_point = vec![AkitaField::zero(); layout.dimension];
        let mut precommitted_evals = vec![AkitaField::zero(); 1usize << layout.dimension];
        precommitted_evals[0] = AkitaField::from_u64(19);
        let precommitted_poly = Polynomial::new(precommitted_evals);
        let precommitted_digest = [11; 32];
        let (precommitted_commitment, precommitted_hint) = AkitaScheme::commit_group(
            &prover_setup,
            precommitted_digest,
            std::slice::from_ref(&precommitted_poly),
        )
        .expect("precommitted commitment should commit");
        let precommitted_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::TrustedAdvice,
            JoltRelationId::AdviceClaimReduction,
        ));
        let precommitted_statement = BatchOpeningStatement {
            logical_point: precommitted_point.clone(),
            pcs_point: precommitted_point,
            layout_digest: precommitted_digest,
            claims: vec![BatchOpeningClaim {
                id: precommitted_id,
                relation: precommitted_id,
                commitment: precommitted_commitment.clone(),
                claim: AkitaField::from_u64(19),
                view: PhysicalView::Direct,
                scale: AkitaField::from_u64(2),
            }],
        };
        assert_eq!(
            precommitted_statement.layout_digest,
            precommitted_commitment.layout_digest
        );
        let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
            logical_manifest: Stage8LogicalManifest {
                openings: Vec::new(),
                pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            },
            physical_manifest: Stage8PhysicalManifest {
                openings: Vec::new(),
                layout_digest: layout.digest,
            },
            opening_ids: vec![sign_id, precommitted_id],
            opening_claims: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            statement: packed_statement.clone(),
            precommitted_statements: vec![precommitted_statement.clone()],
        });
        let precommitted_inputs = [AkitaPrecommittedOpeningInput {
            polynomial: &precommitted_poly,
            hint: &precommitted_hint,
        }];

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let proofs = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &stage8_statement,
            &precommitted_inputs,
        )
        .expect("stage8 proofs should be produced");
        assert_eq!(proofs.precommitted.len(), 1);
        validate_akita_precommitted_opening_proof_payload_shapes(
            &artifact.commitments,
            &proofs.precommitted,
        )
        .expect("fresh precommitted proof payload should pass preflight");

        let mut packed_target_precommitted_proof = proofs.precommitted[0].clone();
        packed_target_precommitted_proof.native.commitment = packed_commitment.clone();
        assert!(matches!(
            validate_akita_precommitted_opening_proof_payload_shapes(
                &artifact.commitments,
                std::slice::from_ref(&packed_target_precommitted_proof),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("precommitted commitment")
        ));

        let mut packed_reduction_precommitted_proof = proofs.precommitted[0].clone();
        packed_reduction_precommitted_proof.reduction =
            Some(jolt_akita::AkitaPackedReductionProof {
                rounds: Vec::new(),
                opening_eval: vec![0; AkitaField::NUM_BYTES],
            });
        assert!(matches!(
            validate_akita_precommitted_opening_proof_payload_shapes(
                &artifact.commitments,
                std::slice::from_ref(&packed_reduction_precommitted_proof),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("packed reduction")
        ));

        let mut packed_target_statement = stage8_statement.clone();
        let Stage8BatchStatement::Clear(clear_statement) = &mut packed_target_statement else {
            unreachable!("test statement is clear");
        };
        clear_statement.precommitted_statements[0].claims[0].commitment = packed_commitment.clone();
        let mut packed_target_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut packed_target_transcript,
            &artifact,
            &source,
            &packed_target_statement,
            &precommitted_inputs,
        )
        .expect_err("precommitted statement targeting W_pack should fail");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("separate precommitted commitment")
        ));

        let packed_hint_inputs = [AkitaPrecommittedOpeningInput {
            polynomial: &precommitted_poly,
            hint: &artifact.hint,
        }];
        let mut packed_hint_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut packed_hint_transcript,
            &artifact,
            &source,
            &stage8_statement,
            &packed_hint_inputs,
        )
        .expect_err("precommitted input using W_pack hint should fail");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("packed witness hint")
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let _ = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &packed_statement,
            &proofs.packed,
        )
        .expect("packed proof should verify");
        let _ = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &precommitted_statement,
            &proofs.precommitted[0],
        )
        .expect("precommitted proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut missing_input_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut missing_input_transcript,
            &artifact,
            &source,
            &stage8_statement,
        )
        .expect_err("precommitted statement requires input");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("expected 1 Akita precommitted opening inputs")
        ));
    }

    #[test]
    #[cfg_attr(
        feature = "field-inline",
        ignore = "field-inline canonical-byte validity makes the real Akita proof fixture expensive; run explicitly with --run-ignored"
    )]
    fn packed_validity_helper_proves_real_akita_opening_proof() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);
            #[cfg(feature = "field-inline")]
            {
                config.lattice.field_inline.enabled = true;
                config.lattice.packed_witness.field_rd_inc_family = true;
            }

            let layout = crate::stages::stage8::derive_akita_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_akita_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));
            let source = validity_default_source(&layout, &requirements);
            let params = AkitaSetupParams::from_packed_layout(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
            let artifacts = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
                .expect("valid packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packed_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("validity proof should prove");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect("validity proof should verify");
            assert_eq!(prover_transcript.state(), verifier_transcript.state());

            let mut tampered = validity.clone();
            tampered.opening_claims.opening_claims[0] += AkitaField::one();
            let mut tampered_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut tampered_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &tampered,
            )
            .expect_err("tampered validity opening claim should reject");
            assert!(matches!(
                error,
                VerifierError::AkitaPackedValidityOutputMismatch
                    | VerifierError::AkitaPackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[cfg(feature = "field-inline")]
    #[test]
    #[ignore = "real Akita negative canonical-byte proof takes over two minutes; run explicitly with --run-ignored"]
    fn packed_validity_rejects_noncanonical_field_rd_inc_bytes() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);

            let layout = crate::stages::stage8::derive_akita_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_akita_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));

            let modulus_bytes = AKITA_FIELD_MODULUS.to_le_bytes();
            let source =
                validity_source_with_field_rd_inc_bytes(&layout, &requirements, &modulus_bytes);
            let params = AkitaSetupParams::from_packed_layout(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
            let artifacts = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
                .expect("packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packed_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("invalid packed witness can still produce a proof transcript");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect_err("noncanonical field bytes should reject");
            assert!(matches!(
                error,
                VerifierError::AkitaPackedValidityOutputMismatch
                    | VerifierError::AkitaPackedValiditySumcheckFailed { .. }
                    | VerifierError::AkitaPackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn packed_validity_value_detects_noncanonical_field_rd_inc_bytes() {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);

        let layout = crate::stages::stage8::derive_akita_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        let requirements =
            derive_akita_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        let statements = derive_akita_packed_validity_statements(&layout, &requirements)
            .expect("validity statements should derive");
        let source = validity_source_with_field_rd_inc_bytes(
            &layout,
            &requirements,
            &AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let statement = statements
            .iter()
            .find(|statement| {
                matches!(
                    statement.requirement.family,
                    LatticePackedFamilyId::FieldRdIncByte { index: 0 }
                ) && statement.kind
                    == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
            })
            .expect("FieldRdInc canonical-byte statement should exist");
        let point = vec![AkitaField::zero(); statement.num_vars];
        let value = validity_value(&source, statement, &point, &point)
            .expect("validity value should evaluate");

        assert_ne!(value, AkitaField::zero());
    }

    #[test]
    fn packed_validity_value_detects_noncanonical_bytecode_imm_bytes() {
        let (layout, statements, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let statement = statements
            .iter()
            .find(|statement| {
                matches!(
                    statement.requirement.family,
                    LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 }
                ) && statement.kind
                    == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
            })
            .expect("bytecode immediate canonical-byte statement should exist");
        let point = vec![AkitaField::zero(); statement.num_vars];
        let value = validity_value(&source, statement, &point, &point)
            .expect("validity value should evaluate");

        assert_ne!(value, AkitaField::zero());
    }

    #[test]
    fn packed_validity_value_detects_malformed_advice_byte_onehot() {
        let (layout, statements) = small_validity_context();
        let family = PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Untrusted,
            index: 0,
        };
        let source = SparsePackedWitness::try_from_cells(
            layout,
            [
                (packed_cell_at(family.clone(), 0, 0, 7), AkitaField::one()),
                (packed_cell_at(family, 0, 0, 8), AkitaField::one()),
            ],
        )
        .expect("malformed advice source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Untrusted,
                index: 0,
            },
            LatticePackedValidityStatementKind::ExactOneHotRowSum,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    #[test]
    fn packed_validity_value_detects_malformed_bytecode_optional_selector() {
        let (layout, statements, _) = small_bytecode_validity_context();
        let family = PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        };
        let source = SparsePackedWitness::try_from_cells(
            layout,
            [
                (packed_cell_at(family.clone(), 0, 0, 3), AkitaField::one()),
                (packed_cell_at(family, 0, 0, 4), AkitaField::one()),
            ],
        )
        .expect("malformed bytecode selector source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            LatticePackedValidityStatementKind::OptionalOneHotRowSum,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    #[test]
    fn packed_validity_value_detects_malformed_bytecode_boolean_flag() {
        let (layout, statements, _) = small_bytecode_validity_context();
        let flag = CircuitFlags::Store as usize;
        let family = PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag };
        let source =
            SparsePackedWitness::try_from_cells(layout, [(packed_cell_at(family, 0, 0, 1), af(2))])
                .expect("malformed bytecode flag source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag },
            LatticePackedValidityStatementKind::BooleanIndicator,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    #[test]
    fn packed_validity_rejects_precommitted_bytecode_layout_config() {
        let (layout, _, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);

        let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect_err("precommitted bytecode families should reject");

        assert!(matches!(
            error,
            VerifierError::InvalidProtocolConfig { reason }
                if reason.contains("precommitted family")
        ));
    }

    fn validity_default_source(
        layout: &PackedWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
    ) -> SparsePackedWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |_, _| 0)
    }

    fn small_validity_context() -> (PackedWitnessLayout, Vec<LatticePackedValidityStatement>) {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            Some(1),
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.advice.untrusted = true;
        config.lattice.packed_witness.untrusted_advice_family = true;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }

        let layout = crate::stages::stage8::derive_akita_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        let requirements =
            derive_akita_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        let statements = derive_akita_packed_validity_statements(&layout, &requirements)
            .expect("validity statements should derive");
        (layout, statements)
    }

    fn small_bytecode_validity_context() -> (
        PackedWitnessLayout,
        Vec<LatticePackedValidityStatement>,
        Vec<LatticePackedValidityRequirement>,
    ) {
        let specs = vec![
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 0,
                },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackedAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackedAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 0 },
                AkitaField::NUM_BYTES,
                PackedAlphabet::Byte,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::FieldRdIncByte { index },
                    PackedFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackedAlphabet::Byte,
                )
            }));
            specs
        };
        let layout =
            PackedWitnessLayout::new(specs).expect("manual bytecode validity layout should build");
        let mut requirements = akita_lattice_validity_requirements_for_layout(&layout);
        requirements.push(bytecode_imm_canonical_bytes_requirement(
            0,
            AkitaField::NUM_BYTES,
            AKITA_FIELD_MODULUS,
        ));
        let statements = derive_akita_packed_validity_statements(&layout, &requirements)
            .expect("manual bytecode validity statements should derive");
        (layout, statements, requirements)
    }

    fn validity_statement(
        statements: &[LatticePackedValidityStatement],
        family: LatticePackedFamilyId,
        kind: LatticePackedValidityStatementKind,
    ) -> &LatticePackedValidityStatement {
        statements
            .iter()
            .find(|statement| statement.requirement.family == family && statement.kind == kind)
            .expect("validity statement should exist")
    }

    fn validity_value_at_zero(
        source: &SparsePackedWitness<AkitaField>,
        statement: &LatticePackedValidityStatement,
    ) -> AkitaField {
        let point = vec![AkitaField::zero(); statement.num_vars];
        validity_value(source, statement, &point, &point).expect("validity value should evaluate")
    }

    #[cfg(feature = "field-inline")]
    fn validity_source_with_field_rd_inc_bytes(
        layout: &PackedWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackedWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, _| match family {
            LatticePackedFamilyId::FieldRdIncByte { index } => bytes[*index] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_bytecode_imm_bytes(
        layout: &PackedWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackedWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, limb| match family {
            LatticePackedFamilyId::BytecodeImmBytes { .. } => bytes[limb] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_symbols(
        layout: &PackedWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        mut symbol_for: impl FnMut(&LatticePackedFamilyId, usize) -> usize,
    ) -> SparsePackedWitness<AkitaField> {
        let mut cells = Vec::new();
        for requirement in requirements {
            let family_id = akita_packed_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .expect("validity family should exist");
            let rows = family.domain.rows().expect("family rows should derive");
            if !matches!(requirement.kind, LatticePackedValidityKind::ExactOneHot) {
                continue;
            }
            for row in 0..rows {
                for limb in 0..requirement.limbs {
                    let symbol = symbol_for(&requirement.family, limb);
                    cells.push((
                        PackedCellAddress {
                            family: family_id.clone(),
                            row,
                            limb,
                            symbol,
                        },
                        AkitaField::one(),
                    ));
                }
            }
        }
        SparsePackedWitness::try_from_cells(layout.clone(), cells)
            .expect("validity source should build")
    }

    fn verify_validity_artifacts<T>(
        setup: &AkitaVerifierSetup,
        transcript: &mut T,
        artifacts: &AkitaPackedWitnessArtifacts,
        log_k_chunk: usize,
        precommitted: &PrecommittedSchedule,
        validity: &AkitaPackedValidityProofArtifacts,
    ) -> Result<(), VerifierError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        crate::stages::stage8::verify_lattice_packed_validity_proof::<
            AkitaField,
            AkitaPackedScheme,
            T,
            ClearOnlyCommitment,
        >(
            setup,
            transcript,
            &artifacts.protocol,
            log_k_chunk,
            precommitted,
            &artifacts.layout,
            artifacts
                .payload()
                .expect("artifact should carry lattice payload")
                .packed_witness
                .clone(),
            &validity.sumcheck_proof,
            &validity.opening_claims.opening_claims,
            &validity.opening_proof,
        )
    }

    #[test]
    fn akita_clear_verifier_surface_is_nameable() {
        type TestTranscript = Blake2bTranscript<AkitaField>;
        type VerifyFn = fn(
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &AkitaJoltProof,
            Option<&AkitaCommitment>,
            &JoltProtocolConfig,
        ) -> Result<(), VerifierError>;
        let _verify: VerifyFn = verify_akita_clear::<TestTranscript>;
        type ProveFn = fn(
            &AkitaProverSetup,
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &AkitaJoltProof,
            Option<&AkitaCommitment>,
            &AkitaPackedWitnessArtifacts,
            &SparsePackedWitness<AkitaField>,
        ) -> Result<AkitaPackedBatchProof, VerifierError>;
        let _prove: ProveFn =
            prove_akita_jolt_final_openings::<TestTranscript, SparsePackedWitness<AkitaField>>;
        type ProveValidityFn = fn(
            &AkitaProverSetup,
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &AkitaJoltProof,
            Option<&AkitaCommitment>,
            &AkitaPackedWitnessArtifacts,
            &SparsePackedWitness<AkitaField>,
        )
            -> Result<AkitaPackedValidityProofArtifacts, VerifierError>;
        let _prove_validity: ProveValidityFn =
            prove_akita_jolt_packed_validity::<TestTranscript, SparsePackedWitness<AkitaField>>;
        type AttachOpeningsFn = fn(
            &AkitaProverSetup,
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &mut AkitaJoltProof,
            Option<&AkitaCommitment>,
            &AkitaPackedWitnessArtifacts,
            &SparsePackedWitness<AkitaField>,
        ) -> Result<(), VerifierError>;
        let _attach_openings: AttachOpeningsFn = prove_and_attach_akita_opening_proofs::<
            TestTranscript,
            SparsePackedWitness<AkitaField>,
        >;
    }

    #[test]
    fn akita_verifier_setup_binds_protocol_config() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (_, verifier_setup) = AkitaPackedScheme::setup(params);
        let config = akita_lattice_protocol_config_for_layout(&layout);

        validate_akita_verifier_setup_config(&verifier_setup, &config)
            .expect("setup should match generated Akita protocol config");

        let mut wrong_digest = config;
        let mut digest = layout.digest;
        digest[0] ^= 1;
        wrong_digest.lattice.packed_witness.layout_digest = Some(digest);
        assert!(matches!(
            validate_akita_verifier_setup_config(&verifier_setup, &wrong_digest),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut wrong_dimension = config;
        wrong_dimension.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
        assert!(matches!(
            validate_akita_verifier_setup_config(&verifier_setup, &wrong_dimension),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut missing_layout = verifier_setup.clone();
        missing_layout.packed_layout = None;
        assert!(matches!(
            validate_akita_verifier_setup_config(&missing_layout, &config),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut missing_native = verifier_setup;
        missing_native.native.clear();
        assert!(matches!(
            validate_akita_verifier_setup_config(&missing_native, &config),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native setup bytes")
        ));
    }

    #[test]
    fn akita_verifier_setup_binds_artifact_layout() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (_, verifier_setup) = AkitaPackedScheme::setup(params);

        validate_akita_verifier_setup_layout(&verifier_setup, &layout)
            .expect("setup should match generated Akita packed layout");

        let other_layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 1 },
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Byte,
        )])
        .expect("layout should build");
        assert!(matches!(
            validate_akita_verifier_setup_layout(&verifier_setup, &other_layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut zero_group_setup = verifier_setup;
        zero_group_setup.max_num_polys_per_commitment_group = 0;
        assert!(matches!(
            validate_akita_verifier_setup_layout(&zero_group_setup, &layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn akita_verifier_payload_shape_binds_inner_commitment_metadata() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
            .expect("empty sparse source should build");
        let artifacts = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        validate_akita_proof_payload_shape(&verifier_setup, &artifacts.commitments)
            .expect("matching payload shape should pass");
        let payload = artifacts
            .commitments
            .as_lattice()
            .expect("artifact should carry lattice payload");

        let mut wrong_commitment_digest = payload.clone();
        wrong_commitment_digest.packed_witness.layout_digest = [9; 32];
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_commitment_digest),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("commitment layout digest")
        ));

        let mut wrong_commitment_dimension = payload.clone();
        wrong_commitment_dimension.packed_witness.num_vars = layout.dimension + 1;
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_commitment_dimension),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("commitment dimension")
        ));

        let mut wrong_poly_count = payload.clone();
        wrong_poly_count.packed_witness.poly_count = 2;
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_poly_count),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("exactly one polynomial")
        ));

        let mut missing_native_commitment = payload.clone();
        missing_native_commitment.packed_witness.native.clear();
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(missing_native_commitment),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native commitment bytes")
        ));
    }

    #[test]
    fn akita_untrusted_advice_aliases_packed_witness_but_trusted_must_be_separate() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout, Vec::new())
            .expect("empty sparse source should build");
        let artifacts = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let payload = artifacts
            .commitments
            .as_lattice()
            .expect("artifact should carry lattice payload");
        let packed_witness = &payload.packed_witness;
        validate_akita_advice_commitment_aliases(&artifacts.commitments, None, None)
            .expect("absent advice commitments should pass");
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            Some(packed_witness),
            None,
        )
        .expect("packed-witness untrusted advice alias should pass");
        assert!(matches!(
            validate_akita_advice_commitment_aliases(
                &artifacts.commitments,
                None,
                Some(packed_witness),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("trusted advice commitment must be separate")
        ));

        let mut other_commitment = packed_witness.clone();
        other_commitment.layout_digest[0] ^= 1;
        assert!(matches!(
            validate_akita_advice_commitment_aliases(
                &artifacts.commitments,
                Some(&other_commitment),
                None,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("untrusted advice commitment")
        ));
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            None,
            Some(&other_commitment),
        )
        .expect("trusted advice may use a separate precommitted commitment");
    }

    #[test]
    fn akita_precommitted_commitments_must_not_alias_packed_witness() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout, Vec::new())
            .expect("empty sparse source should build");
        let artifacts = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let packed_witness = &artifacts
            .commitments
            .as_lattice()
            .expect("artifact should carry lattice payload")
            .packed_witness;

        assert!(matches!(
            validate_akita_precommitted_commitment_is_separate(
                packed_witness,
                packed_witness,
                "bytecode chunk",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("bytecode chunk commitment must be separate")
        ));

        let mut separate_commitment = packed_witness.clone();
        separate_commitment.layout_digest[0] ^= 1;
        validate_akita_precommitted_commitment_is_separate(
            packed_witness,
            &separate_commitment,
            "program image",
        )
        .expect("separate precommitted commitment should pass");
    }

    #[test]
    fn configured_layout_mismatch_rejects_before_commit() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
            .expect("empty sparse source should build");
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.layout_digest = Some([9; 32]);

        let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect_err("layout mismatch should reject");

        assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
    }

    #[test]
    fn akita_artifact_preflight_rejects_stale_protocol_and_commitments() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(PackedFamilyId::InstructionRa { index: 0 }, 7),
                    AkitaField::one(),
                ),
                (
                    packed_cell(PackedFamilyId::UnsignedIncMsb, 1),
                    AkitaField::one(),
                ),
            ],
        )
        .expect("source should build");
        let other_source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(PackedFamilyId::InstructionRa { index: 0 }, 8),
                    AkitaField::one(),
                ),
                (
                    packed_cell(PackedFamilyId::UnsignedIncMsb, 0),
                    AkitaField::one(),
                ),
            ],
        )
        .expect("other source should build");
        let artifacts = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let other_artifacts = commit_akita_packed_witness(&prover_setup, &other_source)
            .expect("other packed witness should commit");

        validate_akita_artifacts_for_proof(
            &verifier_setup,
            &artifacts.protocol,
            &artifacts.commitments,
            &artifacts,
        )
        .expect("matching artifacts should pass preflight");

        let mut stale_protocol = artifacts.protocol;
        stale_protocol.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
        assert!(matches!(
            validate_akita_artifacts_for_proof(
                &verifier_setup,
                &stale_protocol,
                &artifacts.commitments,
                &artifacts,
            ),
            Err(VerifierError::ProtocolConfigMismatch { expected, got })
                if expected == artifacts.protocol && got == stale_protocol
        ));

        assert!(matches!(
            validate_akita_artifacts_for_proof(
                &verifier_setup,
                &artifacts.protocol,
                &other_artifacts.commitments,
                &artifacts,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("do not match packed witness artifacts")
        ));
    }
}
