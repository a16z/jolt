use super::{
    inputs::Deps,
    outputs::{
        Stage8BatchStatement, Stage8ClearBatchStatement, Stage8ClearOutput, Stage8LogicalManifest,
        Stage8LogicalOpening, Stage8OpeningId, Stage8Output, Stage8PhysicalManifest,
        Stage8ZkBatchStatement, Stage8ZkOutput,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{CommitmentPayload, JoltCommitments, JoltProof},
    stages::{
        stage6::{
            inputs::Stage6Claims,
            outputs::{VerifiedBytecodeReadRafSumcheck, VerifiedStage6Sumcheck},
        },
        stage7::{inputs::Stage7Claims, outputs::PrecommittedFinalOpening},
    },
    verifier::CheckedInputs,
    VerifierError,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::claim_reductions::increments as field_increments;
use jolt_claims::protocols::jolt::{
    formulas::{
        committed_openings::{
            commitment_embedding_scale, final_opening_id, final_opening_point,
            final_opening_polynomial_order, FinalOpeningPointInputs,
        },
        dimensions::JoltFormulaDimensions,
        ra::JoltRaPolynomialLayout,
    },
    fused_increment_bytecode_source_opening, fused_increment_magnitude_opening,
    fused_increment_sign_opening, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId,
    JoltRelationId, LatticeFusedIncrementTarget,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
    EvaluationClaim, VerifierOpeningClaim, ZkBatchOpeningScheme,
};
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_transcript::Transcript;

struct Stage8BatchEntry<'a, F: Field, C> {
    id: Stage8OpeningId,
    commitment: &'a C,
    /// `None` in ZK mode, where opening claims stay committed.
    opening_claim: Option<F>,
    /// Point where this logical opening was produced before Stage 8 embedding.
    own_point: Vec<F>,
    /// Lagrange factor embedding this polynomial's own opening point into the
    /// unified opening point.
    scale: F,
}

#[cfg(feature = "field-inline")]
const fn field_inline_final_opening_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_final_opening_count() -> usize {
    0
}

pub fn verify<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    deps: Deps<'_, F, VC::Output>,
) -> Result<Stage8Output<F, PCS::Output, VC::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>
        + BatchOpeningScheme
        + ZkBatchOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    match batch_statement(
        checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        deps,
    )? {
        Stage8BatchStatement::Zk(batch) => {
            if !checked.zk {
                return Err(VerifierError::ExpectedClearProof { field: "stage8" });
            }
            let batch_result = PCS::verify_batch_zk(
                &preprocessing.pcs_setup,
                transcript,
                &batch.statement,
                &proof.joint_opening_proof,
            )
            .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?;

            Ok(Stage8Output::Zk(Stage8ZkOutput {
                opening_ids: batch.opening_ids,
                constraint_coefficients: batch_result.coefficients,
                pcs_opening_point: batch.pcs_opening_point,
                joint_commitment: batch_result.joint_commitment,
                hiding_evaluation_commitment: batch_result.reduced_opening,
            }))
        }
        Stage8BatchStatement::Clear(batch) => {
            if checked.zk {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage8" });
            }
            let batch_result = PCS::verify_batch(
                &preprocessing.pcs_setup,
                transcript,
                &batch.statement,
                &proof.joint_opening_proof,
            )
            .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?;

            Ok(Stage8Output::Clear(Stage8ClearOutput {
                opening_claims: batch.opening_claims,
                opening_ids: batch.opening_ids,
                constraint_coefficients: batch_result.coefficients,
                pcs_opening_point: batch.pcs_opening_point,
                joint_claim: batch_result.reduced_opening,
                joint_commitment: batch_result.joint_commitment,
            }))
        }
    }
}

pub fn verify_clear<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    deps: Deps<'_, F, VC::Output>,
) -> Result<Stage8ClearOutput<F, PCS::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    if checked.zk {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    }

    let Stage8BatchStatement::Clear(batch) = batch_statement(
        checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        deps,
    )?
    else {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    };

    let batch_result = PCS::verify_batch(
        &preprocessing.pcs_setup,
        transcript,
        &batch.statement,
        &proof.joint_opening_proof,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })?;

    Ok(Stage8ClearOutput {
        opening_claims: batch.opening_claims,
        opening_ids: batch.opening_ids,
        constraint_coefficients: batch_result.coefficients,
        pcs_opening_point: batch.pcs_opening_point,
        joint_claim: batch_result.reduced_opening,
        joint_commitment: batch_result.joint_commitment,
    })
}

pub fn batch_statement<F, PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    deps: Deps<'_, F, VC::Output>,
) -> Result<Stage8BatchStatement<F, PCS::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage8" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage8" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let layout = formula_dimensions.ra_layout;
    let (
        hamming_opening_point,
        inc_opening_point,
        precommitted_finals,
        clear_claims,
        fused_increment_translation,
        fused_increment_source_link,
    ) = match deps {
        Deps::Clear { stage6, stage7 } => (
            stage7
                .batch
                .hamming_weight_claim_reduction
                .opening_point
                .as_slice(),
            stage6.batch.inc_claim_reduction.opening_point.as_slice(),
            stage7.precommitted_final_openings.as_slice(),
            Some((&stage6.output_claims, &stage7.output_claims)),
            stage6.batch.fused_increment_translation.as_ref(),
            stage6.batch.fused_increment_source_link.as_ref(),
        ),
        Deps::Zk { stage6, stage7 } => (
            stage7
                .hamming_weight_claim_reduction
                .opening_point
                .as_slice(),
            stage6.inc_claim_reduction.opening_point.as_slice(),
            stage7.precommitted_final_openings.as_slice(),
            None,
            None,
            None,
        ),
    };

    let anchor_points: Vec<&[F]> = precommitted_finals
        .iter()
        .map(|opening| opening.point.as_slice())
        .collect();
    let opening_point = final_opening_point(FinalOpeningPointInputs {
        log_t,
        log_k_chunk: proof.one_hot_config.committed_chunk_bits(),
        trace_order: proof.trace_polynomial_order,
        hamming_weight_opening_point: hamming_opening_point,
        inc_claim_reduction_opening_point: inc_opening_point,
        precommitted_anchor_points: &anchor_points,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let pcs_opening_point = Point::high_to_low(opening_point.clone());
    let committed_bytecode_chunk_count = preprocessing
        .program
        .committed()
        .map(|committed| committed.bytecode_chunk_count());

    let entries = match &proof.commitments {
        CommitmentPayload::Dory(commitments) => {
            require_commitment_layout(commitments, layout)?;
            batch_entries(
                layout,
                committed_bytecode_chunk_count,
                &opening_point,
                hamming_opening_point,
                inc_opening_point,
                precommitted_finals,
                clear_claims,
                false,
                |polynomial| {
                    dory_final_commitment(
                        preprocessing,
                        commitments,
                        proof.untrusted_advice_commitment.as_ref(),
                        trusted_advice_commitment,
                        polynomial,
                    )
                },
                #[cfg(feature = "field-inline")]
                Some(&commitments.field_inline.field_registers.rd_inc),
            )?
        }
        CommitmentPayload::Akita(payload) => {
            let mut entries = akita_fused_increment_entries(
                &opening_point,
                &payload.packed_witness,
                fused_increment_translation,
                fused_increment_source_link,
                clear_claims.map(|(stage6, _)| stage6),
            )?;
            entries.extend(batch_entries(
                layout,
                committed_bytecode_chunk_count,
                &opening_point,
                hamming_opening_point,
                inc_opening_point,
                precommitted_finals,
                clear_claims,
                true,
                |_| Ok(&payload.packed_witness),
                #[cfg(feature = "field-inline")]
                Some(&payload.packed_witness),
            )?);
            entries
        }
    };
    let logical_manifest = logical_manifest(&entries, pcs_opening_point.clone());
    let opening_ids = logical_manifest.opening_ids();
    let (physical_manifest, layout_digest) = match &proof.commitments {
        CommitmentPayload::Dory(_) => {
            let layout_digest = stage8_layout_digest(preprocessing);
            (
                Stage8PhysicalManifest::direct(&logical_manifest, layout_digest),
                layout_digest,
            )
        }
        CommitmentPayload::Akita(_) => akita_stage8_physical_manifest(
            &proof.protocol,
            checked,
            proof,
            layout,
            &logical_manifest,
        )?,
    };
    let point = pcs_opening_point.as_slice().to_vec();

    if checked.zk {
        let claims = entries
            .iter()
            .zip(&physical_manifest.openings)
            .map(|(entry, physical)| BatchOpeningClaim {
                id: entry.id,
                relation: physical.relation,
                commitment: entry.commitment.clone(),
                claim: (),
                view: physical.view.clone(),
                scale: entry.scale,
            })
            .collect::<Vec<_>>();
        return Ok(Stage8BatchStatement::Zk(Stage8ZkBatchStatement {
            logical_manifest,
            physical_manifest,
            opening_ids,
            pcs_opening_point,
            statement: BatchOpeningStatement {
                logical_point: point.clone(),
                pcs_point: point,
                layout_digest,
                claims,
            },
        }));
    }

    let mut opening_claims = Vec::with_capacity(entries.len());
    let mut claims = Vec::with_capacity(entries.len());
    for (entry, physical) in entries.iter().zip(&physical_manifest.openings) {
        let opening_claim =
            entry
                .opening_claim
                .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                    reason: "missing clear opening claim in final batch".to_string(),
                })?;
        opening_claims.push(VerifierOpeningClaim {
            commitment: entry.commitment.clone(),
            evaluation: EvaluationClaim::new(
                pcs_opening_point.clone(),
                opening_claim * entry.scale,
            ),
        });
        claims.push(BatchOpeningClaim {
            id: entry.id,
            relation: physical.relation,
            commitment: entry.commitment.clone(),
            claim: opening_claim,
            view: physical.view.clone(),
            scale: entry.scale,
        });
    }
    Ok(Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
        logical_manifest,
        physical_manifest,
        opening_ids,
        opening_claims,
        pcs_opening_point,
        statement: BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point,
            layout_digest,
            claims,
        },
    }))
}

/// Builds the final PCS batch in the canonical order from
/// [`final_opening_polynomial_order`], resolving each polynomial's commitment,
/// opening claim (clear mode only), and unified-point embedding scale.
#[expect(
    clippy::too_many_arguments,
    reason = "gathers per-polynomial sources from several stages"
)]
fn batch_entries<'a, F, C, CommitmentFor>(
    layout: JoltRaPolynomialLayout,
    committed_bytecode_chunk_count: Option<usize>,
    opening_point: &[F],
    hamming_opening_point: &[F],
    inc_opening_point: &[F],
    precommitted_finals: &'a [PrecommittedFinalOpening<F>],
    clear_claims: Option<(&Stage6Claims<F>, &Stage7Claims<F>)>,
    skip_increment_openings: bool,
    mut commitment_for: CommitmentFor,
    #[cfg(feature = "field-inline")] field_rd_inc_commitment: &'a C,
) -> Result<Vec<Stage8BatchEntry<'a, F, C>>, VerifierError>
where
    F: Field,
    CommitmentFor: FnMut(JoltCommittedPolynomial) -> Result<&'a C, VerifierError>,
{
    let precommitted_final = |polynomial: JoltCommittedPolynomial| {
        precommitted_finals
            .iter()
            .find(|opening| opening.polynomial == polynomial)
    };
    let include_trusted = precommitted_final(JoltCommittedPolynomial::TrustedAdvice).is_some();
    let include_untrusted = precommitted_final(JoltCommittedPolynomial::UntrustedAdvice).is_some();
    let order = final_opening_polynomial_order(
        layout,
        include_trusted,
        include_untrusted,
        committed_bytecode_chunk_count,
    );

    let mut entries = Vec::with_capacity(order.len() + field_inline_final_opening_count());
    // Core's final PCS batch order intentionally differs from proof payload order.
    for polynomial in order {
        if skip_increment_openings
            && matches!(
                polynomial,
                JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc
            )
        {
            continue;
        }
        let id = final_opening_id(polynomial);
        let (own_point, opening_claim): (&[F], Option<F>) = match polynomial {
            JoltCommittedPolynomial::RamInc => (
                inc_opening_point,
                clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.ram_inc),
            ),
            JoltCommittedPolynomial::RdInc => (
                inc_opening_point,
                clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.rd_inc),
            ),
            JoltCommittedPolynomial::InstructionRa(index) => (
                hamming_opening_point,
                match clear_claims {
                    Some((_, stage7)) => Some(
                        *stage7
                            .hamming_weight_claim_reduction
                            .instruction_ra
                            .get(index)
                            .ok_or(VerifierError::MissingOpeningClaim { id })?,
                    ),
                    None => None,
                },
            ),
            JoltCommittedPolynomial::BytecodeRa(index) => (
                hamming_opening_point,
                match clear_claims {
                    Some((_, stage7)) => Some(
                        *stage7
                            .hamming_weight_claim_reduction
                            .bytecode_ra
                            .get(index)
                            .ok_or(VerifierError::MissingOpeningClaim { id })?,
                    ),
                    None => None,
                },
            ),
            JoltCommittedPolynomial::RamRa(index) => (
                hamming_opening_point,
                match clear_claims {
                    Some((_, stage7)) => Some(
                        *stage7
                            .hamming_weight_claim_reduction
                            .ram_ra
                            .get(index)
                            .ok_or(VerifierError::MissingOpeningClaim { id })?,
                    ),
                    None => None,
                },
            ),
            JoltCommittedPolynomial::TrustedAdvice => {
                let opening = precommitted_final(polynomial)
                    .ok_or(VerifierError::MissingOpeningClaim { id })?;
                (opening.point.as_slice(), opening.opening_claim)
            }
            JoltCommittedPolynomial::UntrustedAdvice => {
                let opening = precommitted_final(polynomial)
                    .ok_or(VerifierError::MissingOpeningClaim { id })?;
                (opening.point.as_slice(), opening.opening_claim)
            }
            JoltCommittedPolynomial::BytecodeChunk(_) => {
                let opening = precommitted_final(polynomial)
                    .ok_or(VerifierError::MissingOpeningClaim { id })?;
                (opening.point.as_slice(), opening.opening_claim)
            }
            JoltCommittedPolynomial::ProgramImageInit => {
                let opening = precommitted_final(polynomial)
                    .ok_or(VerifierError::MissingOpeningClaim { id })?;
                (opening.point.as_slice(), opening.opening_claim)
            }
        };
        let commitment = commitment_for(polynomial)?;
        entries.push(Stage8BatchEntry {
            id: id.into(),
            commitment,
            opening_claim,
            own_point: own_point.to_vec(),
            scale: commitment_embedding_scale(opening_point, own_point),
        });
        #[cfg(feature = "field-inline")]
        if polynomial == JoltCommittedPolynomial::RdInc {
            entries.push(Stage8BatchEntry {
                id: field_increments::field_rd_inc_reduced_opening().into(),
                commitment: field_rd_inc_commitment,
                opening_claim: clear_claims.map(|(stage6, _)| {
                    stage6
                        .field_inline
                        .field_registers_inc_claim_reduction
                        .field_rd_inc
                }),
                own_point: inc_opening_point.to_vec(),
                scale: commitment_embedding_scale(opening_point, inc_opening_point),
            });
        }
    }
    Ok(entries)
}

fn akita_fused_increment_entries<'a, F, C>(
    opening_point: &[F],
    packed_witness: &'a C,
    translation: Option<&VerifiedStage6Sumcheck<F>>,
    source_link: Option<&VerifiedBytecodeReadRafSumcheck<F>>,
    stage6_claims: Option<&Stage6Claims<F>>,
) -> Result<Vec<Stage8BatchEntry<'a, F, C>>, VerifierError>
where
    F: Field,
{
    let translation = translation.ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
        reason: "Akita final batch requires verified fused increment translation claims"
            .to_string(),
    })?;
    let source_link = source_link.ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
        reason: "Akita final batch requires verified fused increment source-link claims"
            .to_string(),
    })?;
    let translation_output_claims = stage6_claims
        .and_then(|stage6| stage6.fused_increment_translation.as_ref())
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita final batch requires fused increment translation output claims"
                .to_string(),
        })?;
    let source_link_output_claims = stage6_claims
        .and_then(|stage6| stage6.fused_increment_source_link.as_ref())
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita final batch requires fused increment source-link output claims"
                .to_string(),
        })?;
    if source_link_output_claims.bytecode_ra.len() != source_link.bytecode_ra_opening_points.len() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "fused increment source-link bytecode RA claim count mismatch: expected {}, got {}",
                source_link.bytecode_ra_opening_points.len(),
                source_link_output_claims.bytecode_ra.len()
            ),
        });
    }

    let translation_point = translation.opening_point.as_slice();
    let translation_scale = commitment_embedding_scale(opening_point, translation_point);
    let source_point = source_link.r_address.as_slice();
    let source_scale = commitment_embedding_scale(opening_point, source_point);
    let mut entries = vec![
        Stage8BatchEntry {
            id: fused_increment_magnitude_opening().into(),
            commitment: packed_witness,
            opening_claim: Some(translation_output_claims.magnitude),
            own_point: translation_point.to_vec(),
            scale: translation_scale,
        },
        Stage8BatchEntry {
            id: fused_increment_sign_opening().into(),
            commitment: packed_witness,
            opening_claim: Some(translation_output_claims.sign),
            own_point: translation_point.to_vec(),
            scale: translation_scale,
        },
    ];

    for (index, (claim, own_point)) in source_link_output_claims
        .bytecode_ra
        .iter()
        .copied()
        .zip(&source_link.bytecode_ra_opening_points)
        .enumerate()
    {
        entries.push(Stage8BatchEntry {
            id: JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(index)),
                relation: JoltRelationId::FusedIncrementSourceLink,
            }
            .into(),
            commitment: packed_witness,
            opening_claim: Some(claim),
            own_point: own_point.clone(),
            scale: commitment_embedding_scale(opening_point, own_point),
        });
    }

    entries.extend([
        Stage8BatchEntry {
            id: fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram).into(),
            commitment: packed_witness,
            opening_claim: Some(source_link_output_claims.store_flag),
            own_point: source_point.to_vec(),
            scale: source_scale,
        },
        Stage8BatchEntry {
            id: fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd).into(),
            commitment: packed_witness,
            opening_claim: Some(source_link_output_claims.rd_present),
            own_point: source_point.to_vec(),
            scale: source_scale,
        },
    ]);

    Ok(entries)
}

fn dory_final_commitment<'a, PCS, VC>(
    preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    commitments: &'a JoltCommitments<PCS::Output>,
    untrusted_advice_commitment: Option<&'a PCS::Output>,
    trusted_advice_commitment: Option<&'a PCS::Output>,
    polynomial: JoltCommittedPolynomial,
) -> Result<&'a PCS::Output, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    match polynomial {
        JoltCommittedPolynomial::RamInc => Ok(&commitments.ram_inc),
        JoltCommittedPolynomial::RdInc => Ok(&commitments.rd_inc),
        JoltCommittedPolynomial::InstructionRa(index) => commitments
            .ra
            .instruction
            .get(index)
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::BytecodeRa(index) => commitments
            .ra
            .bytecode
            .get(index)
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::RamRa(index) => commitments
            .ra
            .ram
            .get(index)
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::TrustedAdvice => trusted_advice_commitment
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::UntrustedAdvice => untrusted_advice_commitment
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::BytecodeChunk(index) => preprocessing
            .program
            .committed()
            .and_then(|committed| committed.bytecode_chunk_commitments.get(index))
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
        JoltCommittedPolynomial::ProgramImageInit => preprocessing
            .program
            .committed()
            .map(|committed| &committed.program_image_commitment)
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial }),
    }
}

#[cfg(feature = "akita")]
fn akita_stage8_physical_manifest<F, PCS, VC, ZkProof>(
    config: &crate::config::JoltProtocolConfig,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    layout: JoltRaPolynomialLayout,
    logical: &Stage8LogicalManifest<F>,
) -> Result<(Stage8PhysicalManifest<F>, [u8; 32]), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let packed_layout = super::derive_akita_packed_witness_layout(
        config,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
        layout,
        &checked.precommitted,
    )?;
    super::validate_akita_packed_witness_layout_config(config, &packed_layout)?;
    let physical = super::jolt_lattice_physical_manifest(
        logical,
        &packed_layout,
        proof.one_hot_config.committed_chunk_bits(),
        &checked.precommitted,
    )?;
    Ok((physical, packed_layout.digest))
}

#[cfg(not(feature = "akita"))]
fn akita_stage8_physical_manifest<F, PCS, VC, ZkProof>(
    config: &crate::config::JoltProtocolConfig,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    layout: JoltRaPolynomialLayout,
    logical: &Stage8LogicalManifest<F>,
) -> Result<(Stage8PhysicalManifest<F>, [u8; 32]), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let _ = (config, checked, proof, layout, logical);
    Err(VerifierError::InvalidProtocolConfig {
        reason: "lattice Stage 8 requires the jolt-verifier akita feature".to_string(),
    })
}

fn logical_manifest<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
    pcs_opening_point: Point<HIGH_TO_LOW, F>,
) -> Stage8LogicalManifest<F>
where
    F: Field,
{
    Stage8LogicalManifest {
        openings: entries
            .iter()
            .map(|entry| Stage8LogicalOpening {
                id: entry.id,
                point: entry.own_point.clone(),
                claim: entry.opening_claim,
                scale: entry.scale,
            })
            .collect(),
        pcs_opening_point,
    }
}

fn stage8_layout_digest<PCS, VC>(preprocessing: &JoltVerifierPreprocessing<PCS, VC>) -> [u8; 32]
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    preprocessing.preprocessing_digest
}

fn require_commitment_layout<C>(
    commitments: &JoltCommitments<C>,
    layout: JoltRaPolynomialLayout,
) -> Result<(), VerifierError> {
    let expected = 2 + layout.total();
    let got = 2
        + commitments.ra.instruction.len()
        + commitments.ra.bytecode.len()
        + commitments.ra.ram.len();
    if got != expected {
        return Err(VerifierError::InvalidCommitmentCount { expected, got });
    }
    if commitments.ra.instruction.len() != layout.instruction()
        || commitments.ra.bytecode.len() != layout.bytecode()
        || commitments.ra.ram.len() != layout.ram()
    {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "commitment layout mismatch: expected instruction={}, bytecode={}, ram={}; got instruction={}, bytecode={}, ram={}",
                layout.instruction(),
                layout.bytecode(),
                layout.ram(),
                commitments.ra.instruction.len(),
                commitments.ra.bytecode.len(),
                commitments.ra.ram.len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test setup should fail loudly when helper contracts change"
    )]

    use super::*;
    use crate::stages::stage6::inputs::{
        BooleanityOutputOpeningClaims, BytecodeReadRafOutputOpeningClaims,
        FusedIncrementSourceLinkOutputClaims, FusedIncrementTranslationOutputClaims,
        IncClaimReductionOutputOpeningClaims, InstructionRaVirtualizationOutputOpeningClaims,
        RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
        Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn akita_fused_increment_entries_use_verified_translation_outputs() {
        let packed_witness = 9_u64;
        let opening_point = (1..=22).map(Fr::from_u64).collect::<Vec<_>>();
        let translation = VerifiedStage6Sumcheck {
            input_claim: Fr::from_u64(4),
            sumcheck_point: vec![Fr::from_u64(5), Fr::from_u64(6)],
            opening_point: vec![Fr::from_u64(1), Fr::from_u64(2)],
            expected_output_claim: Fr::from_u64(7),
        };
        let source_link = VerifiedBytecodeReadRafSumcheck {
            input_claim: Fr::from_u64(14),
            sumcheck_point: vec![Fr::from_u64(15), Fr::from_u64(16), Fr::from_u64(17)],
            r_address: vec![Fr::from_u64(18), Fr::from_u64(19)],
            r_cycle: vec![Fr::from_u64(20)],
            full_opening_point: vec![Fr::from_u64(18), Fr::from_u64(19), Fr::from_u64(20)],
            bytecode_ra_opening_points: vec![vec![Fr::from_u64(21), Fr::from_u64(22)]],
            expected_output_claim: Fr::from_u64(23),
        };
        let claims = stage6_claims_with_fused_outputs(
            Fr::from_u64(10),
            Fr::from_u64(11),
            Fr::from_u64(12),
            Fr::from_u64(13),
        );

        let entries = akita_fused_increment_entries(
            &opening_point,
            &packed_witness,
            Some(&translation),
            Some(&source_link),
            Some(&claims),
        )
        .expect("fused entries should build");

        assert_eq!(
            entries.iter().map(|entry| entry.id).collect::<Vec<_>>(),
            vec![
                fused_increment_magnitude_opening().into(),
                fused_increment_sign_opening().into(),
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(0)),
                    relation: JoltRelationId::FusedIncrementSourceLink,
                }
                .into(),
                fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram).into(),
                fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd).into(),
            ]
        );
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.opening_claim)
                .collect::<Vec<_>>(),
            vec![
                Some(Fr::from_u64(11)),
                Some(Fr::from_u64(12)),
                Some(Fr::from_u64(30)),
                Some(Fr::from_u64(31)),
                Some(Fr::from_u64(32)),
            ]
        );
        assert!(entries
            .iter()
            .all(|entry| entry.commitment == &packed_witness));
        assert_eq!(entries[0].own_point, translation.opening_point);
        assert_eq!(entries[1].own_point, translation.opening_point);
        assert_eq!(
            entries[2].own_point,
            source_link.bytecode_ra_opening_points[0]
        );
        assert_eq!(entries[3].own_point, source_link.r_address);
        assert_eq!(entries[4].own_point, source_link.r_address);
    }

    #[test]
    fn akita_fused_increment_entries_require_verified_stage6_outputs() {
        let packed_witness = 9_u64;
        let claims = stage6_claims_with_fused_outputs(
            Fr::from_u64(10),
            Fr::from_u64(11),
            Fr::from_u64(12),
            Fr::from_u64(13),
        );

        let error = akita_fused_increment_entries::<Fr, _>(
            &[Fr::from_u64(1)],
            &packed_witness,
            None,
            None,
            Some(&claims),
        )
        .err()
        .expect("missing verified sumcheck should fail");
        assert!(error
            .to_string()
            .contains("verified fused increment translation claims"));
    }

    #[test]
    fn committed_program_batch_entries_require_final_openings() {
        let layout =
            JoltRaPolynomialLayout::new(1, 0, 0).expect("test RA layout should be valid");
        let opening_point = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let hamming_opening_point = vec![Fr::from_u64(1)];
        let inc_opening_point = vec![Fr::from_u64(1)];
        let commitment = 9_u64;

        let program_image_only = vec![PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::ProgramImageInit,
            point: vec![Fr::from_u64(3)],
            opening_claim: Some(Fr::from_u64(30)),
        }];
        let error = batch_entries(
            layout,
            Some(1),
            &opening_point,
            &hamming_opening_point,
            &inc_opening_point,
            &program_image_only,
            None,
            true,
            |_| Ok(&commitment),
            #[cfg(feature = "field-inline")]
            &commitment,
        )
        .err()
        .expect("missing bytecode chunk opening should fail");
        assert!(matches!(
            error,
            VerifierError::MissingOpeningClaim { id }
                if id == final_opening_id(JoltCommittedPolynomial::BytecodeChunk(0))
        ));

        let bytecode_only = vec![PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::BytecodeChunk(0),
            point: vec![Fr::from_u64(2)],
            opening_claim: Some(Fr::from_u64(20)),
        }];
        let error = batch_entries(
            layout,
            Some(1),
            &opening_point,
            &hamming_opening_point,
            &inc_opening_point,
            &bytecode_only,
            None,
            true,
            |_| Ok(&commitment),
            #[cfg(feature = "field-inline")]
            &commitment,
        )
        .err()
        .expect("missing program image opening should fail");
        assert!(matches!(
            error,
            VerifierError::MissingOpeningClaim { id }
                if id == final_opening_id(JoltCommittedPolynomial::ProgramImageInit)
        ));
    }

    fn stage6_claims_with_fused_outputs(
        ram_source: Fr,
        magnitude: Fr,
        sign: Fr,
        rd_source: Fr,
    ) -> Stage6Claims<Fr> {
        let zero = Fr::from_u64(0);
        Stage6Claims {
            address_phase: Stage6AddressPhaseClaims {
                bytecode_read_raf: zero,
                booleanity: zero,
                bytecode_val_stages: None,
            },
            bytecode_read_raf: BytecodeReadRafOutputOpeningClaims {
                bytecode_ra: Vec::new(),
            },
            booleanity: BooleanityOutputOpeningClaims {
                instruction_ra: Vec::new(),
                bytecode_ra: Vec::new(),
                ram_ra: Vec::new(),
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims {
                ram_hamming_weight: zero,
            },
            ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims { ram_ra: Vec::new() },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims {
                committed_instruction_ra: Vec::new(),
            },
            inc_claim_reduction: IncClaimReductionOutputOpeningClaims {
                ram_inc: zero,
                rd_inc: zero,
            },
            fused_increment_translation: Some(FusedIncrementTranslationOutputClaims {
                ram_source,
                magnitude,
                sign,
                rd_source,
            }),
            fused_increment_source_link: Some(FusedIncrementSourceLinkOutputClaims {
                bytecode_ra: vec![Fr::from_u64(30)],
                store_flag: Fr::from_u64(31),
                rd_present: Fr::from_u64(32),
            }),
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        }
    }
}
