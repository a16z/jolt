//! Prover-facing helpers for assembling Akita verifier artifacts.

use std::collections::BTreeMap;

use crate::{
    config::{
        validate_protocol_config, AdviceLatticeConfig, FieldInlineLatticeConfig,
        IncrementCommitmentMode, JoltProtocolConfig, LatticeConfig, PackedWitnessConfig, PcsFamily,
        ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{
        AkitaCommitmentPayload, ClearOnlyCommitment, ClearOnlyVectorCommitment, CommitmentPayload,
        JoltProof, JoltProofClaims,
    },
    stages::{
        stage6::{
            inputs::{FusedIncrementSourceLinkOutputClaims, FusedIncrementTranslationOutputClaims},
            outputs::VerifiedBytecodeReadRafSumcheck,
        },
        stage7::inputs::LatticePackedValidityOutputClaims,
        stage8::{
            akita_packed_family_id, akita_packed_view_formula, build_lattice_packed_validity_batch,
            derive_akita_packed_validity_requirements, derive_akita_packed_validity_statements,
            jolt_lattice_view_formula, lattice_packed_validity_claims,
            lattice_packed_validity_opening_count, sample_lattice_packed_validity_eq_points,
            validate_akita_packed_witness_layout_config, LatticePackedValidityStatement,
            LatticePackedValidityStatementKind, Stage8BatchStatement,
        },
        PrecommittedSchedule,
    },
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaPackedBatchProof, AkitaPackedScheme, AkitaProverHint,
    AkitaProverSetup, AkitaVerifierSetup, JoltPackedWitnessBuilder, PackedAdviceKind,
    PackedFactDomain, PackedFamilyId, PackedViewFormula, PackedWitnessLayout, PackedWitnessSource,
    SparsePackedWitness,
};
use jolt_claims::protocols::jolt::{
    fused_increment_bytecode_source_opening, fused_increment_magnitude_lattice_view_formula,
    fused_increment_sign_lattice_view_formula, lattice_packed_validity_digest, JoltAdviceKind,
    JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, LatticeFusedIncrementTarget,
    LatticePackedFamilyId, LatticePackedValidityKind, LatticePackedValidityRequirement,
    FUSED_INCREMENT_BYTE_LIMBS,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::BatchOpeningStatement;
use jolt_poly::{try_eq_mle, EqPolynomial, UnivariatePoly};
use jolt_riscv::{JoltInstructionRow, JoltTraceRow};
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
    pub bytecode_rows: &'a [JoltInstructionRow],
    pub program_image_words: &'a [u64],
    pub trusted_advice: Option<&'a [u8]>,
    pub untrusted_advice: Option<&'a [u8]>,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackedWitnessArtifacts,
    pub witness: SparsePackedWitness<AkitaField>,
}

#[derive(Clone, Debug)]
pub struct AkitaPackedValidityProofArtifacts {
    pub sumcheck_proof: SumcheckProof<AkitaField, ClearOnlyCommitment>,
    pub opening_claims: LatticePackedValidityOutputClaims<AkitaField>,
    pub opening_proof: AkitaPackedBatchProof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AkitaFusedIncrementTranslationSources {
    pub ram_source: AkitaField,
    pub rd_source: AkitaField,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AkitaFusedIncrementStage6Claims {
    pub translation: FusedIncrementTranslationOutputClaims<AkitaField>,
    pub source_link: FusedIncrementSourceLinkOutputClaims<AkitaField>,
}

impl AkitaPackedWitnessArtifacts {
    pub fn payload(&self) -> Option<&AkitaCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_akita()
    }
}

pub fn build_akita_packed_jolt_witness(
    input: AkitaPackedJoltWitnessInput<'_>,
) -> Result<SparsePackedWitness<AkitaField>, VerifierError> {
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

    pack_bytecode_rows(&mut builder, input.bytecode_rows)?;
    pack_program_image_words(&mut builder, input.program_image_words)?;
    pack_advice_bytes(
        &mut builder,
        PackedAdviceKind::Trusted,
        input.trusted_advice,
    )?;
    pack_advice_bytes(
        &mut builder,
        PackedAdviceKind::Untrusted,
        input.untrusted_advice,
    )?;

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

pub fn derive_akita_fused_increment_stage6_claims<S>(
    source: &S,
    precommitted: &PrecommittedSchedule,
    log_k_chunk: usize,
    translation_opening_point: &[AkitaField],
    source_link: &VerifiedBytecodeReadRafSumcheck<AkitaField>,
    translation_sources: AkitaFusedIncrementTranslationSources,
) -> Result<AkitaFusedIncrementStage6Claims, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let magnitude = eval_akita_packed_formula(
        source,
        fused_increment_magnitude_lattice_view_formula(),
        translation_opening_point,
        JoltRelationId::FusedIncrementTranslation,
    )?;
    let sign = eval_akita_packed_formula(
        source,
        fused_increment_sign_lattice_view_formula(),
        translation_opening_point,
        JoltRelationId::FusedIncrementTranslation,
    )?;
    let translation = FusedIncrementTranslationOutputClaims {
        ram_source: translation_sources.ram_source,
        magnitude,
        sign,
        rd_source: translation_sources.rd_source,
    };

    let bytecode_ra = source_link
        .bytecode_ra_opening_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let row_point = point.get(log_k_chunk..).ok_or_else(|| {
                akita_fused_claim_error(
                    JoltRelationId::FusedIncrementSourceLink,
                    format!(
                        "BytecodeRa({index}) source-link opening point has {} variables but needs at least {log_k_chunk}",
                        point.len()
                    ),
                )
            })?;
            eval_akita_jolt_lattice_opening(
                source,
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(index),
                    JoltRelationId::FusedIncrementSourceLink,
                ),
                point,
                row_point,
                log_k_chunk,
                precommitted,
                JoltRelationId::FusedIncrementSourceLink,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let bytecode_layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        akita_fused_claim_error(
            JoltRelationId::FusedIncrementSourceLink,
            "source-link derivation requires committed-bytecode layout",
        )
    })?;
    let source_address = bytecode_layout
        .split_address_point(&source_link.r_address)
        .map_err(|error| {
            akita_fused_claim_error(JoltRelationId::FusedIncrementSourceLink, error)
        })?;
    let store_flag = eval_akita_jolt_lattice_opening(
        source,
        fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
        &source_link.r_address,
        &source_address.r_bc,
        log_k_chunk,
        precommitted,
        JoltRelationId::FusedIncrementSourceLink,
    )?;
    let rd_present = eval_akita_jolt_lattice_opening(
        source,
        fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
        &source_link.r_address,
        &source_address.r_bc,
        log_k_chunk,
        precommitted,
        JoltRelationId::FusedIncrementSourceLink,
    )?;
    let source_link = FusedIncrementSourceLinkOutputClaims {
        bytecode_ra,
        store_flag,
        rd_present,
    };

    Ok(AkitaFusedIncrementStage6Claims {
        translation,
        source_link,
    })
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
            trusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        zk: false,
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
                PackedFamilyId::IncByte { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::IncByte { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackedFamilyId::IncSign => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::IncSign,
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
    if layout.family(&PackedFamilyId::IncSign).is_some()
        && (0..FUSED_INCREMENT_BYTE_LIMBS)
            .all(|index| layout.family(&PackedFamilyId::IncByte { index }).is_some())
    {
        requirements.push(LatticePackedValidityRequirement::fused_increment_canonical_zero());
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
    let payload = AkitaCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_akita_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackedWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Akita(payload),
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
    let Stage8BatchStatement::Clear(statement) = statement else {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening proving requires a clear Stage 8 statement".to_string(),
        });
    };
    prove_akita_packed_openings(setup, transcript, artifacts, source, &statement.statement)
}

pub fn prove_akita_packed_validity<T, S>(
    setup: &AkitaProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
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
        derive_akita_packed_validity_requirements(&artifacts.protocol, precommitted)?;
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
    candidate.joint_opening_proof = prove_akita_jolt_final_openings::<T, S>(
        setup,
        preprocessing,
        public_io,
        &candidate,
        trusted_advice_commitment,
        artifacts,
        source,
    )?;
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
            .as_akita()
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

    Ok(())
}

fn pack_bytecode_rows(
    builder: &mut JoltPackedWitnessBuilder,
    rows: &[JoltInstructionRow],
) -> Result<(), VerifierError> {
    let expected = expected_bytecode_rows(builder.layout())?;
    let Some(expected) = expected else {
        if rows.is_empty() {
            return Ok(());
        }
        return Err(akita_witness_error(
            "bytecode rows were supplied but the packed layout has no bytecode families",
        ));
    };
    let padded = padded_slice(rows, expected, "bytecode rows")?;
    builder
        .pack_bytecode_rows(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn pack_program_image_words(
    builder: &mut JoltPackedWitnessBuilder,
    words: &[u64],
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| matches!(id, PackedFamilyId::ProgramImageInit),
        "program image words",
    )?;
    let Some(expected) = expected else {
        if words.is_empty() {
            return Ok(());
        }
        return Err(akita_witness_error(
            "program image words were supplied but the packed layout has no program image family",
        ));
    };
    let padded = padded_slice(words, expected, "program image words")?;
    builder
        .pack_program_image_words(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn pack_advice_bytes(
    builder: &mut JoltPackedWitnessBuilder,
    kind: PackedAdviceKind,
    bytes: Option<&[u8]>,
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| matches!(id, PackedFamilyId::AdviceBytes { kind: family_kind, index: 0 } if *family_kind == kind),
        advice_domain_name(kind),
    )?;
    let Some(expected) = expected else {
        if bytes.is_none_or(<[u8]>::is_empty) {
            return Ok(());
        }
        return Err(akita_witness_error(format!(
            "{} were supplied but the packed layout has no matching advice family",
            advice_domain_name(kind)
        )));
    };
    let padded = padded_slice(
        bytes.unwrap_or_default(),
        expected,
        advice_domain_name(kind),
    )?;
    builder
        .pack_advice_bytes(kind, &padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn expected_bytecode_rows(layout: &PackedWitnessLayout) -> Result<Option<usize>, VerifierError> {
    let mut chunks = BTreeMap::<usize, usize>::new();
    for family in &layout.families {
        let chunk = match family.id {
            PackedFamilyId::BytecodeChunk { index }
            | PackedFamilyId::BytecodeRegisterSelector { chunk: index, .. }
            | PackedFamilyId::BytecodeCircuitFlag { chunk: index, .. }
            | PackedFamilyId::BytecodeInstructionFlag { chunk: index, .. }
            | PackedFamilyId::BytecodeLookupSelector { chunk: index }
            | PackedFamilyId::BytecodeRafFlag { chunk: index }
            | PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: index }
            | PackedFamilyId::BytecodeImmBytes { chunk: index } => index,
            _ => continue,
        };
        let rows = packed_domain_rows(family.domain)?;
        match chunks.insert(chunk, rows) {
            Some(existing) if existing != rows => {
                return Err(akita_witness_error(format!(
                    "bytecode rows layout chunk {chunk} has inconsistent row counts {existing} and {rows}"
                )));
            }
            _ => {}
        }
    }
    let Some((&max_chunk, &chunk_rows)) = chunks.last_key_value() else {
        return Ok(None);
    };
    for chunk in 0..=max_chunk {
        if !chunks.contains_key(&chunk) {
            return Err(akita_witness_error(format!(
                "bytecode rows layout is missing chunk {chunk}",
            )));
        }
    }
    max_chunk
        .checked_add(1)
        .and_then(|chunk_count| chunk_count.checked_mul(chunk_rows))
        .ok_or_else(|| akita_witness_error("bytecode rows layout size overflow"))
        .map(Some)
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

fn advice_domain_name(kind: PackedAdviceKind) -> &'static str {
    match kind {
        PackedAdviceKind::Trusted => "trusted advice bytes",
        PackedAdviceKind::Untrusted => "untrusted advice bytes",
    }
}

fn eval_akita_jolt_lattice_opening<S>(
    source: &S,
    id: JoltOpeningId,
    formula_point: &[AkitaField],
    row_point: &[AkitaField],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
    stage: JoltRelationId,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let formula = jolt_lattice_view_formula(id, formula_point, log_k_chunk, precommitted)?;
    eval_akita_packed_formula(source, formula, row_point, stage)
}

fn eval_akita_packed_formula<S>(
    source: &S,
    formula: jolt_claims::protocols::jolt::LatticePackedViewFormula<AkitaField>,
    row_point: &[AkitaField],
    stage: JoltRelationId,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let formula = akita_packed_view_formula(&formula)
        .map_err(|error| akita_fused_claim_error(stage, error))?;
    eval_akita_packed_view(source, &formula, row_point, stage)
}

fn eval_akita_packed_view<S>(
    source: &S,
    formula: &PackedViewFormula<AkitaField>,
    row_point: &[AkitaField],
    stage: JoltRelationId,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    formula
        .eval_row_point(source, row_point)
        .map_err(|error| akita_fused_claim_error(stage, error))
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
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => {
            validity_opening_values(source, statement, point)?
                .into_iter()
                .fold(AkitaField::one(), |acc, opening| acc * opening)
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
    if statement.kind != LatticePackedValidityStatementKind::FusedIncrementCanonicalZero {
        return validity_opening_value(source, statement, point).map(|value| vec![value]);
    }

    let mut values = Vec::with_capacity(FUSED_INCREMENT_BYTE_LIMBS + 1);
    values.push(fused_increment_canonical_zero_factor_value(
        source, point, 0,
    )?);
    for factor in 1..=FUSED_INCREMENT_BYTE_LIMBS {
        values.push(fused_increment_canonical_zero_factor_value(
            source, point, factor,
        )?);
    }
    Ok(values)
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
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => {
            Err(VerifierError::InvalidProtocolConfig {
                reason: "canonical-zero validity has multiple opening factors".to_string(),
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
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => opening,
    }
}

fn fused_increment_canonical_zero_factor_value<S>(
    source: &S,
    point: &[AkitaField],
    factor: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let (family_id, symbol) = if factor == 0 {
        (PackedFamilyId::IncSign, 1)
    } else {
        let byte_index = factor - 1;
        if byte_index >= FUSED_INCREMENT_BYTE_LIMBS {
            return Err(VerifierError::InvalidProtocolConfig {
                reason: format!("fused increment canonical-zero has no opening factor {factor}"),
            });
        }
        (PackedFamilyId::IncByte { index: byte_index }, 0)
    };
    let family =
        source
            .layout()
            .family(&family_id)
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: format!("fused increment canonical-zero factor requires {family_id:?}"),
            })?;
    let rows = family
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!(
                "fused increment canonical-zero factor {family_id:?} has invalid row domain: {error}"
            ),
        })?;
    let row_vars = power_of_two_log(rows, "fused increment canonical-zero row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "fused increment canonical-zero point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    weighted_direct_symbol_value(source, &family_id, &row_weights, symbol)
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
        if &address.family != family_id || address.limb != 0 || address.symbol != symbol {
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
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => shape.row,
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

fn akita_fused_claim_error(stage: JoltRelationId, reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: format!(
            "Akita fused increment claim derivation failed: {}",
            reason.to_string()
        ),
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
    use jolt_akita::{
        AkitaSetupParams, PackedAlphabet, PackedCellAddress, PackedFactDomain, PackedFamilySpec,
        SparsePackedWitness,
    };
    use jolt_claims::protocols::jolt::{
        formulas::{
            dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
            ra::JoltRaPolynomialLayout,
        },
        fused_increment_bytecode_source_opening, fused_increment_magnitude_opening,
        fused_increment_sign_opening, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
        LatticeFusedIncrementTarget,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
        OpeningsError, PackedLinearTerm, PhysicalView,
    };
    use jolt_poly::{EqPolynomial, Point};
    use jolt_riscv::{
        CapturedState, CircuitFlags, JoltInstructionKind, JoltInstructionRow, JoltTraceRow,
        NonMemoryState, NormalizedOperands, StoreState,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
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
            .expect("artifact should carry Akita payload");
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
        let layout = PackedWitnessLayout::new([
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
                PackedFamilyId::IncByte { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 1 },
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 1 },
                AkitaField::NUM_BYTES,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::ProgramImageInit,
                PackedFactDomain::ProgramImageWords { log_words: 1 },
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    index: 0,
                },
                PackedFactDomain::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    log_bytes: 2,
                },
                1,
                PackedAlphabet::Byte,
            ),
        ])
        .expect("layout should build");
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
        let bytecode = [instruction(
            JoltInstructionKind::ADDI,
            0x8000_0000,
            NormalizedOperands {
                rs1: Some(1),
                rs2: None,
                rd: Some(5),
                imm: 7,
            },
        )];
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);

        let committed = commit_akita_packed_jolt_witness(
            &prover_setup,
            AkitaPackedJoltWitnessInput {
                layout: layout.clone(),
                trace_rows: &rows,
                log_k_chunk: 8,
                instruction_lookup_indices: &[0xaa, 0xbb],
                bytecode_rows: &bytecode,
                program_image_words: &[0x0201],
                trusted_advice: Some(&[7, 8]),
                untrusted_advice: None,
            },
        )
        .expect("Jolt packed witness should build and commit");

        assert_eq!(committed.artifacts.layout, layout);
        let payload = committed
            .artifacts
            .payload()
            .expect("artifact should carry Akita payload");
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
                    PackedFamilyId::IncByte { index: 0 },
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
                    PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                    0,
                    0,
                    7,
                ))
                .expect("bytecode immediate cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                    1,
                    0,
                    0,
                ))
                .expect("padded bytecode row should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(PackedFamilyId::ProgramImageInit, 1, 0, 0))
                .expect("padded program image word should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::AdviceBytes {
                        kind: PackedAdviceKind::Trusted,
                        index: 0,
                    },
                    2,
                    0,
                    0,
                ))
                .expect("padded trusted advice byte should exist"),
            AkitaField::one()
        );
    }

    #[test]
    fn derives_fused_increment_stage6_claims_from_packed_witness() {
        let log_t = 1;
        let log_k_chunk = 2;
        let bytecode_log_rows = 1;
        let mut specs = (0..8)
            .map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::IncByte { index },
                    PackedFactDomain::TraceRows { log_t },
                    1,
                    PackedAlphabet::Byte,
                )
            })
            .collect::<Vec<_>>();
        specs.extend([
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t },
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRa { index: 0 },
                PackedFactDomain::TraceRows { log_t },
                1,
                PackedAlphabet::Fixed {
                    size: 1 << log_k_chunk,
                },
            ),
        ]);
        for chunk in 0..2 {
            specs.extend([
                PackedFamilySpec::direct(
                    PackedFamilyId::BytecodeCircuitFlag {
                        chunk,
                        flag: CircuitFlags::Store as usize,
                    },
                    PackedFactDomain::BytecodeRows {
                        log_bytecode: bytecode_log_rows,
                    },
                    1,
                    PackedAlphabet::Bit,
                ),
                PackedFamilySpec::direct(
                    PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
                    PackedFactDomain::BytecodeRows {
                        log_bytecode: bytecode_log_rows,
                    },
                    1,
                    PackedAlphabet::Fixed {
                        size: 1 << REGISTER_ADDRESS_BITS,
                    },
                ),
            ]);
        }
        let layout = PackedWitnessLayout::new(specs).expect("layout should build");
        let source = SparsePackedWitness::try_from_cells(
            layout,
            [
                (
                    packed_cell_at(PackedFamilyId::IncByte { index: 0 }, 0, 0, 5),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(PackedFamilyId::IncByte { index: 0 }, 1, 0, 9),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(PackedFamilyId::IncSign, 1, 0, 1),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(PackedFamilyId::BytecodeRa { index: 0 }, 0, 0, 1),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(PackedFamilyId::BytecodeRa { index: 0 }, 1, 0, 3),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(
                        PackedFamilyId::BytecodeCircuitFlag {
                            chunk: 0,
                            flag: CircuitFlags::Store as usize,
                        },
                        0,
                        0,
                        1,
                    ),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(
                        PackedFamilyId::BytecodeCircuitFlag {
                            chunk: 1,
                            flag: CircuitFlags::Store as usize,
                        },
                        1,
                        0,
                        1,
                    ),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(
                        PackedFamilyId::BytecodeRegisterSelector {
                            chunk: 0,
                            selector: 2,
                        },
                        0,
                        0,
                        5,
                    ),
                    AkitaField::one(),
                ),
                (
                    packed_cell_at(
                        PackedFamilyId::BytecodeRegisterSelector {
                            chunk: 1,
                            selector: 2,
                        },
                        1,
                        0,
                        7,
                    ),
                    AkitaField::one(),
                ),
            ],
        )
        .expect("source should build");
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 4,
                bytecode_chunk_count: 2,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let translation_point = [af(3)];
        let source_link = VerifiedBytecodeReadRafSumcheck {
            input_claim: AkitaField::zero(),
            sumcheck_point: Vec::new(),
            r_address: vec![af(11), af(13)],
            r_cycle: vec![af(7)],
            full_opening_point: vec![af(11), af(13), af(7)],
            bytecode_ra_opening_points: vec![vec![af(2), af(5), af(7)]],
            expected_output_claim: AkitaField::zero(),
        };

        let claims = derive_akita_fused_increment_stage6_claims(
            &source,
            &precommitted,
            log_k_chunk,
            &translation_point,
            &source_link,
            AkitaFusedIncrementTranslationSources {
                ram_source: af(17),
                rd_source: af(19),
            },
        )
        .expect("fused increment claims should derive");

        let trace_weights = EqPolynomial::<AkitaField>::evals(&translation_point, None);
        assert_eq!(
            claims.translation,
            FusedIncrementTranslationOutputClaims {
                ram_source: af(17),
                magnitude: trace_weights[0] * af(5) + trace_weights[1] * af(9),
                sign: trace_weights[1],
                rd_source: af(19),
            }
        );
        let bytecode_ra_address_weights = EqPolynomial::<AkitaField>::evals(
            &source_link.bytecode_ra_opening_points[0][..2],
            None,
        );
        let bytecode_ra_cycle_weights = EqPolynomial::<AkitaField>::evals(
            &source_link.bytecode_ra_opening_points[0][2..],
            None,
        );
        let expected_bytecode_ra = bytecode_ra_cycle_weights[0] * bytecode_ra_address_weights[1]
            + bytecode_ra_cycle_weights[1] * bytecode_ra_address_weights[3];
        let chunk_weights = EqPolynomial::<AkitaField>::evals(&source_link.r_address[..1], None);
        let bytecode_row_weights =
            EqPolynomial::<AkitaField>::evals(&source_link.r_address[1..], None);
        let expected_source =
            chunk_weights[0] * bytecode_row_weights[0] + chunk_weights[1] * bytecode_row_weights[1];
        assert_eq!(
            claims.source_link,
            FusedIncrementSourceLinkOutputClaims {
                bytecode_ra: vec![expected_bytecode_ra],
                store_flag: expected_source,
                rd_present: expected_source,
            }
        );

        let params = AkitaSetupParams::from_packed_layout(source.layout(), 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let artifacts = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifacts
            .payload()
            .expect("artifact should carry Akita payload")
            .packed_witness
            .clone();
        let source_address = precommitted
            .bytecode
            .as_ref()
            .expect("bytecode layout should exist")
            .split_address_point(&source_link.r_address)
            .expect("source address should split");
        let magnitude_view =
            akita_packed_view_formula(&fused_increment_magnitude_lattice_view_formula())
                .expect("magnitude formula should lower")
                .physical_view_at(source.layout(), &translation_point)
                .expect("magnitude view should bind row point");
        let sign_view = akita_packed_view_formula(&fused_increment_sign_lattice_view_formula())
            .expect("sign formula should lower")
            .physical_view_at(source.layout(), &translation_point)
            .expect("sign view should bind row point");
        let bytecode_ra_formula = jolt_lattice_view_formula(
            JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltRelationId::FusedIncrementSourceLink,
            ),
            &source_link.bytecode_ra_opening_points[0],
            log_k_chunk,
            &precommitted,
        )
        .expect("bytecode RA formula should resolve");
        let bytecode_ra_view = akita_packed_view_formula(&bytecode_ra_formula)
            .expect("bytecode RA formula should lower")
            .physical_view_at(
                source.layout(),
                &source_link.bytecode_ra_opening_points[0][log_k_chunk..],
            )
            .expect("bytecode RA view should bind row point");
        let store_view = akita_packed_view_formula(
            &jolt_lattice_view_formula(
                fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
                &source_link.r_address,
                log_k_chunk,
                &precommitted,
            )
            .expect("store source formula should resolve"),
        )
        .expect("store source formula should lower")
        .physical_view_at(source.layout(), &source_address.r_bc)
        .expect("store source view should bind row point");
        let rd_view = akita_packed_view_formula(
            &jolt_lattice_view_formula(
                fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
                &source_link.r_address,
                log_k_chunk,
                &precommitted,
            )
            .expect("rd source formula should resolve"),
        )
        .expect("rd source formula should lower")
        .physical_view_at(source.layout(), &source_address.r_bc)
        .expect("rd source view should bind row point");
        let magnitude_id = Stage8OpeningId::from(fused_increment_magnitude_opening());
        let sign_id = Stage8OpeningId::from(fused_increment_sign_opening());
        let bytecode_ra_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltRelationId::FusedIncrementSourceLink,
        ));
        let store_id = Stage8OpeningId::from(fused_increment_bytecode_source_opening(
            LatticeFusedIncrementTarget::Ram,
        ));
        let rd_id = Stage8OpeningId::from(fused_increment_bytecode_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
        let statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: source.layout().digest,
            claims: vec![
                BatchOpeningClaim {
                    id: magnitude_id,
                    relation: magnitude_id,
                    commitment: commitment.clone(),
                    claim: claims.translation.magnitude,
                    view: magnitude_view,
                    scale: AkitaField::one(),
                },
                BatchOpeningClaim {
                    id: sign_id,
                    relation: sign_id,
                    commitment: commitment.clone(),
                    claim: claims.translation.sign,
                    view: sign_view,
                    scale: AkitaField::one(),
                },
                BatchOpeningClaim {
                    id: bytecode_ra_id,
                    relation: bytecode_ra_id,
                    commitment: commitment.clone(),
                    claim: claims.source_link.bytecode_ra[0],
                    view: bytecode_ra_view,
                    scale: AkitaField::one(),
                },
                BatchOpeningClaim {
                    id: store_id,
                    relation: store_id,
                    commitment: commitment.clone(),
                    claim: claims.source_link.store_flag,
                    view: store_view,
                    scale: AkitaField::one(),
                },
                BatchOpeningClaim {
                    id: rd_id,
                    relation: rd_id,
                    commitment: commitment.clone(),
                    claim: claims.source_link.rd_present,
                    view: rd_view,
                    scale: AkitaField::one(),
                },
            ],
        };
        let opening_ids = statement
            .claims
            .iter()
            .map(|claim| claim.id)
            .collect::<Vec<_>>();
        let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
            logical_manifest: Stage8LogicalManifest {
                openings: Vec::new(),
                pcs_opening_point: Point::high_to_low(Vec::new()),
            },
            physical_manifest: Stage8PhysicalManifest {
                openings: Vec::new(),
                layout_digest: source.layout().digest,
            },
            opening_ids,
            opening_claims: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::new()),
            statement: statement.clone(),
        });

        let mut prover_transcript = Blake2bTranscript::new(b"derived-fused-stage6");
        let proof = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut prover_transcript,
            &artifacts,
            &source,
            &stage8_statement,
        )
        .expect("derived fused claims should prove as packed openings");
        let mut verifier_transcript = Blake2bTranscript::new(b"derived-fused-stage6");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("derived fused packed openings should verify");
        assert_eq!(result.joint_commitment, commitment);

        let mut tampered_statement = statement.clone();
        tampered_statement
            .claims
            .iter_mut()
            .find(|claim| claim.id == store_id)
            .expect("store source claim should exist")
            .claim += AkitaField::one();
        let mut tampered_transcript = Blake2bTranscript::new(b"derived-fused-stage6");
        let error = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut tampered_transcript,
            &tampered_statement,
            &proof,
        )
        .expect_err("tampered fused source claim should not verify");
        assert!(matches!(error, OpeningsError::VerificationFailed));
    }

    #[test]
    fn packed_witness_artifacts_feed_akita_packed_batch_verifier() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let instruction_family = PackedFamilyId::InstructionRa { index: 0 };
        let sign_family = PackedFamilyId::IncSign;
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(instruction_family.clone(), 7),
                    AkitaField::from_u64(11),
                ),
                (packed_cell(sign_family.clone(), 1), AkitaField::from_u64(5)),
            ],
        )
        .expect("source should build");
        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifact
            .payload()
            .expect("artifact should carry Akita payload")
            .packed_witness
            .clone();
        let instruction_claim = AkitaField::from_u64(22);
        let sign_claim = AkitaField::from_u64(15);
        let instruction_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(0),
            JoltRelationId::HammingWeightClaimReduction,
        ));
        let sign_id = Stage8OpeningId::from(fused_increment_sign_opening());
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
            let requirements = derive_akita_packed_validity_requirements(&config, &precommitted)
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
                &precommitted,
            )
            .expect("validity proof should prove");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
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

    fn validity_default_source(
        layout: &PackedWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
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
                    cells.push((
                        PackedCellAddress {
                            family: family_id.clone(),
                            row,
                            limb,
                            symbol: 0,
                        },
                        AkitaField::one(),
                    ));
                }
            }
        }
        SparsePackedWitness::try_from_cells(layout.clone(), cells)
            .expect("default validity source should build")
    }

    fn verify_validity_artifacts<T>(
        setup: &AkitaVerifierSetup,
        transcript: &mut T,
        artifacts: &AkitaPackedWitnessArtifacts,
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
            precommitted,
            &artifacts.layout,
            artifacts
                .payload()
                .expect("artifact should carry Akita payload")
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
                (packed_cell(PackedFamilyId::IncSign, 1), AkitaField::one()),
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
                (packed_cell(PackedFamilyId::IncSign, 0), AkitaField::one()),
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
