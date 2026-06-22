use super::{
    inputs::Deps,
    outputs::{
        Stage8BatchStatement, Stage8ClearBatchStatement, Stage8ClearOutput, Stage8LogicalManifest,
        Stage8LogicalOpening, Stage8OpeningId, Stage8OpeningStatement, Stage8Output,
        Stage8PhysicalManifest, Stage8PhysicalOpening, Stage8ZkBatchStatement, Stage8ZkOutput,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{CommitmentPayload, JoltCommitments, JoltProof},
    stages::{
        stage6::inputs::Stage6Claims,
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
        lattice as lattice_formulas,
        ra::JoltRaPolynomialLayout,
    },
    JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentLayoutDigest,
    CommitmentScheme, EvaluationClaim, PhysicalView, VerifierOpeningClaim, ZkBatchOpeningScheme,
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

struct LatticeUnsignedIncFinalOpenings<'a, F: Field> {
    chunk_point: &'a [F],
    chunk_claims: Option<&'a [F]>,
    msb_point: &'a [F],
    msb_claim: Option<F>,
}

type Stage8ClearBatchClaim<F, C> = BatchOpeningClaim<F, C, Stage8OpeningId, Stage8OpeningId, F>;
type Stage8ClearClaimBuild<F, C> = (
    Vec<VerifierOpeningClaim<F, C>>,
    Vec<Stage8ClearBatchClaim<F, C>>,
);
type Stage8PrecommittedStatementBuild<F, C> = (
    Vec<VerifierOpeningClaim<F, C>>,
    Vec<Stage8OpeningStatement<F, C, F>>,
);
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
    PCS::Output: Clone + CommitmentLayoutDigest,
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
            let mut coefficients = batch_result.coefficients;
            coefficients.extend(verify_precommitted_opening_batches::<PCS, _>(
                &preprocessing.pcs_setup,
                transcript,
                &batch.precommitted_statements,
                &proof.lattice_precommitted_opening_proofs,
            )?);

            Ok(Stage8Output::Clear(Stage8ClearOutput {
                opening_claims: batch.opening_claims,
                opening_ids: batch.opening_ids,
                constraint_coefficients: coefficients,
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
    PCS::Output: Clone + CommitmentLayoutDigest,
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
    let mut coefficients = batch_result.coefficients;
    coefficients.extend(verify_precommitted_opening_batches::<PCS, _>(
        &preprocessing.pcs_setup,
        transcript,
        &batch.precommitted_statements,
        &proof.lattice_precommitted_opening_proofs,
    )?);

    Ok(Stage8ClearOutput {
        opening_claims: batch.opening_claims,
        opening_ids: batch.opening_ids,
        constraint_coefficients: coefficients,
        pcs_opening_point: batch.pcs_opening_point,
        joint_claim: batch_result.reduced_opening,
        joint_commitment: batch_result.joint_commitment,
    })
}

fn verify_precommitted_opening_batches<PCS, T>(
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
    statements: &[Stage8OpeningStatement<PCS::Field, PCS::Output, PCS::Field>],
    proofs: &[PCS::Proof],
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: BatchOpeningScheme,
    T: Transcript<Challenge = PCS::Field>,
{
    if statements.len() != proofs.len() {
        return Err(VerifierError::FinalOpeningVerificationFailed {
            reason: format!(
                "expected {} precommitted opening proofs, got {}",
                statements.len(),
                proofs.len()
            ),
        });
    }

    let mut coefficients = Vec::new();
    for (statement, proof) in statements.iter().zip(proofs) {
        let result = PCS::verify_batch(setup, transcript, statement, proof).map_err(|error| {
            VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            }
        })?;
        coefficients.extend(result.coefficients);
    }
    Ok(coefficients)
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
    PCS::Output: Clone + CommitmentLayoutDigest,
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
        unsigned_inc_finals,
    ) = match deps {
        Deps::Clear { stage6, stage7 } => {
            let inc_anchor = stage6
                .batch
                .inc_claim_reduction
                .as_ref()
                .or(stage6.batch.unsigned_inc_claim_reduction.as_ref())
                .ok_or(VerifierError::MissingOpeningClaim {
                    id: lattice_formulas::unsigned_inc_opening(),
                })?;
            let unsigned_inc_finals = stage6
                .batch
                .unsigned_inc_msb_booleanity
                .as_ref()
                .zip(stage6.output_claims.unsigned_inc_claim_reduction.as_ref())
                .zip(
                    stage7.batch.unsigned_inc_chunk_reconstruction.as_ref().zip(
                        stage7
                            .output_claims
                            .unsigned_inc_chunk_reconstruction
                            .as_ref(),
                    ),
                )
                .map(|((msb_source, msb_claims), (chunk_source, chunk_claims))| {
                    LatticeUnsignedIncFinalOpenings {
                        chunk_point: chunk_source.opening_point.as_slice(),
                        chunk_claims: Some(chunk_claims.chunks.as_slice()),
                        msb_point: msb_source.opening_point.as_slice(),
                        msb_claim: Some(msb_claims.unsigned_inc_msb),
                    }
                });
            (
                stage7
                    .batch
                    .hamming_weight_claim_reduction
                    .opening_point
                    .as_slice(),
                inc_anchor.opening_point.as_slice(),
                stage7.precommitted_final_openings.as_slice(),
                Some((&stage6.output_claims, &stage7.output_claims)),
                unsigned_inc_finals,
            )
        }
        Deps::Zk { stage6, stage7 } => {
            let unsigned_inc_finals = stage6
                .unsigned_inc_msb_booleanity
                .as_ref()
                .zip(stage7.unsigned_inc_chunk_reconstruction.as_ref())
                .map(
                    |(msb_source, chunk_source)| LatticeUnsignedIncFinalOpenings {
                        chunk_point: chunk_source.opening_point.as_slice(),
                        chunk_claims: None,
                        msb_point: msb_source.opening_point.as_slice(),
                        msb_claim: None,
                    },
                );
            (
                stage7
                    .hamming_weight_claim_reduction
                    .opening_point
                    .as_slice(),
                stage6.inc_claim_reduction.opening_point.as_slice(),
                stage7.precommitted_final_openings.as_slice(),
                None,
                unsigned_inc_finals,
            )
        }
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

    let (entries, precommitted_entries) = match &proof.commitments {
        CommitmentPayload::Dory(commitments) => {
            require_commitment_layout(commitments, layout)?;
            (
                batch_entries(
                    layout,
                    committed_bytecode_chunk_count,
                    checked.precommitted.trusted_advice.is_some(),
                    checked.precommitted.untrusted_advice.is_some(),
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
                    &commitments.field_inline.field_registers.rd_inc,
                )?,
                Vec::new(),
            )
        }
        CommitmentPayload::Akita(payload) => {
            let mut final_entries = batch_entries(
                layout,
                committed_bytecode_chunk_count,
                checked.precommitted.trusted_advice.is_some(),
                checked.precommitted.untrusted_advice.is_some(),
                &opening_point,
                hamming_opening_point,
                inc_opening_point,
                precommitted_finals,
                clear_claims,
                true,
                |polynomial| {
                    if akita_requires_precommitted_opening(polynomial) {
                        precommitted_final_commitment(
                            preprocessing,
                            trusted_advice_commitment,
                            polynomial,
                        )
                    } else {
                        Ok(&payload.packed_witness)
                    }
                },
                #[cfg(feature = "field-inline")]
                &payload.packed_witness,
            )?;
            final_entries.extend(akita_unsigned_inc_batch_entries(
                proof.one_hot_config.committed_chunk_bits(),
                &opening_point,
                unsigned_inc_finals.as_ref(),
                &payload.packed_witness,
            )?);
            let (entries, precommitted_entries): (Vec<_>, Vec<_>) = final_entries
                .into_iter()
                .partition(|entry| !akita_precommitted_stage8_opening(entry.id));
            (entries, precommitted_entries)
        }
    };
    let logical_manifest = logical_manifest_with_precommitted(
        &entries,
        &precommitted_entries,
        pcs_opening_point.clone(),
    );
    let opening_ids = logical_manifest.opening_ids();
    let (physical_manifest, layout_digest) = match &proof.commitments {
        CommitmentPayload::Dory(_) => {
            let layout_digest = stage8_layout_digest(preprocessing);
            (
                Stage8PhysicalManifest::direct(&logical_manifest, layout_digest),
                layout_digest,
            )
        }
        CommitmentPayload::Akita(_) => {
            let packed_logical_manifest =
                stage8_logical_manifest(&entries, pcs_opening_point.clone());
            let (mut packed_manifest, layout_digest) = akita_stage8_physical_manifest(
                &proof.protocol,
                checked,
                proof,
                layout,
                &packed_logical_manifest,
            )?;
            packed_manifest
                .openings
                .extend(direct_physical_openings(&precommitted_entries));
            (packed_manifest, layout_digest)
        }
    };
    let point = pcs_opening_point.as_slice().to_vec();

    if checked.zk && !precommitted_entries.is_empty() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "precommitted lattice openings are not supported in ZK mode".to_string(),
        });
    }

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

    let (mut opening_claims, claims) = clear_batch_claims(
        &entries,
        &physical_manifest.openings[..entries.len()],
        &pcs_opening_point,
    )?;
    let (precommitted_opening_claims, precommitted_statements) = precommitted_clear_statements(
        &precommitted_entries,
        stage8_layout_digest(preprocessing),
        &point,
        &pcs_opening_point,
    )?;
    opening_claims.extend(precommitted_opening_claims);
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
        precommitted_statements,
    }))
}

fn clear_batch_claims<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
    physical_openings: &[Stage8PhysicalOpening<F>],
    pcs_opening_point: &Point<HIGH_TO_LOW, F>,
) -> Result<Stage8ClearClaimBuild<F, C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    if entries.len() != physical_openings.len() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "entry/physical opening count mismatch: {} entries, {} physical openings",
                entries.len(),
                physical_openings.len()
            ),
        });
    }

    let mut opening_claims = Vec::with_capacity(entries.len());
    let mut claims = Vec::with_capacity(entries.len());
    for (entry, physical) in entries.iter().zip(physical_openings) {
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

    Ok((opening_claims, claims))
}

fn precommitted_clear_statements<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
    default_layout_digest: [u8; 32],
    point: &[F],
    pcs_opening_point: &Point<HIGH_TO_LOW, F>,
) -> Result<Stage8PrecommittedStatementBuild<F, C>, VerifierError>
where
    F: Field,
    C: Clone + CommitmentLayoutDigest,
{
    let mut opening_claims = Vec::with_capacity(entries.len());
    let mut statements = Vec::with_capacity(entries.len());
    for entry in entries {
        let physical = Stage8PhysicalOpening {
            id: entry.id,
            relation: entry.id,
            view: PhysicalView::Direct,
        };
        let (mut entry_claims, claims) = clear_batch_claims(
            std::slice::from_ref(entry),
            std::slice::from_ref(&physical),
            pcs_opening_point,
        )?;
        opening_claims.append(&mut entry_claims);
        statements.push(BatchOpeningStatement {
            logical_point: point.to_vec(),
            pcs_point: point.to_vec(),
            layout_digest: direct_statement_layout_digest(entry.commitment, default_layout_digest),
            claims,
        });
    }

    Ok((opening_claims, statements))
}

fn direct_statement_layout_digest<C: CommitmentLayoutDigest>(
    commitment: &C,
    default_layout_digest: [u8; 32],
) -> [u8; 32] {
    commitment.layout_digest().unwrap_or(default_layout_digest)
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
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
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
    let order = final_opening_polynomial_order(
        layout,
        include_trusted_advice,
        include_untrusted_advice,
        committed_bytecode_chunk_count,
    );

    let mut entries = Vec::with_capacity(order.len() + field_inline_final_opening_count());
    // Core's final PCS batch order intentionally differs from proof payload order.
    for polynomial in order {
        let skip_base_opening = skip_increment_openings
            && matches!(
                polynomial,
                JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc
            );
        if !skip_base_opening {
            let id = final_opening_id(polynomial);
            let (own_point, opening_claim): (&[F], Option<F>) = match polynomial {
                JoltCommittedPolynomial::RamInc => (
                    inc_opening_point,
                    clear_claims.and_then(|(stage6, _)| {
                        stage6
                            .inc_claim_reduction
                            .as_ref()
                            .map(|claims| claims.ram_inc)
                    }),
                ),
                JoltCommittedPolynomial::RdInc => (
                    inc_opening_point,
                    clear_claims.and_then(|(stage6, _)| {
                        stage6
                            .inc_claim_reduction
                            .as_ref()
                            .map(|claims| claims.rd_inc)
                    }),
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
        }
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

fn akita_unsigned_inc_batch_entries<'a, F, C>(
    log_k_chunk: usize,
    opening_point: &[F],
    sources: Option<&LatticeUnsignedIncFinalOpenings<'_, F>>,
    commitment: &'a C,
) -> Result<Vec<Stage8BatchEntry<'a, F, C>>, VerifierError>
where
    F: Field,
{
    let sources = sources.ok_or(VerifierError::MissingOpeningClaim {
        id: lattice_formulas::unsigned_inc_chunk_opening(0),
    })?;
    let chunk_count =
        lattice_formulas::unsigned_inc_lower_chunk_count(log_k_chunk).ok_or_else(|| {
            VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "unsigned increment chunk size must evenly divide 64 bits, got {log_k_chunk}"
                ),
            }
        })?;
    if let Some(chunk_claims) = sources.chunk_claims {
        if chunk_claims.len() != chunk_count {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "unsigned increment final chunk opening count mismatch: expected {chunk_count}, got {}",
                    chunk_claims.len()
                ),
            });
        }
    }

    let mut entries = Vec::with_capacity(chunk_count + 1);
    for index in 0..chunk_count {
        entries.push(Stage8BatchEntry {
            id: Stage8OpeningId::from(lattice_formulas::unsigned_inc_chunk_opening(index)),
            commitment,
            opening_claim: sources.chunk_claims.map(|claims| claims[index]),
            own_point: sources.chunk_point.to_vec(),
            scale: commitment_embedding_scale(opening_point, sources.chunk_point),
        });
    }
    entries.push(Stage8BatchEntry {
        id: Stage8OpeningId::from(lattice_formulas::unsigned_inc_msb_opening()),
        commitment,
        opening_claim: sources.msb_claim,
        own_point: sources.msb_point.to_vec(),
        scale: commitment_embedding_scale(opening_point, sources.msb_point),
    });
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

fn precommitted_final_commitment<'a, PCS, VC>(
    preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    trusted_advice_commitment: Option<&'a PCS::Output>,
    polynomial: JoltCommittedPolynomial,
) -> Result<&'a PCS::Output, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    match polynomial {
        JoltCommittedPolynomial::TrustedAdvice => trusted_advice_commitment
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
        _ => Err(VerifierError::MissingFinalOpeningCommitment { polynomial }),
    }
}

fn akita_requires_precommitted_opening(polynomial: JoltCommittedPolynomial) -> bool {
    matches!(
        polynomial,
        JoltCommittedPolynomial::TrustedAdvice
            | JoltCommittedPolynomial::BytecodeChunk(_)
            | JoltCommittedPolynomial::ProgramImageInit
    )
}

fn akita_precommitted_stage8_opening(id: Stage8OpeningId) -> bool {
    matches!(
        id,
        Stage8OpeningId::Jolt(
            JoltOpeningId::TrustedAdvice {
                relation: JoltRelationId::AdviceClaimReduction,
            } | JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(
                    JoltCommittedPolynomial::BytecodeChunk(_)
                        | JoltCommittedPolynomial::ProgramImageInit,
                ),
                ..
            },
        )
    )
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
    let log_k_chunk = proof.one_hot_config.committed_chunk_bits();
    let validity_requirements = super::derive_akita_packed_validity_requirements(
        config,
        log_k_chunk,
        &checked.precommitted,
    )?;
    super::validate_akita_packed_witness_validity_config(
        config,
        log_k_chunk,
        &checked.precommitted,
    )?;
    let physical = super::jolt_lattice_physical_manifest_with_validity(
        logical,
        &packed_layout,
        proof.one_hot_config.committed_chunk_bits(),
        &checked.precommitted,
        &validity_requirements,
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

fn stage8_logical_manifest<F, C>(
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

fn logical_manifest_with_precommitted<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
    precommitted_entries: &[Stage8BatchEntry<'_, F, C>],
    pcs_opening_point: Point<HIGH_TO_LOW, F>,
) -> Stage8LogicalManifest<F>
where
    F: Field,
{
    let mut openings = logical_openings(entries);
    openings.extend(logical_openings(precommitted_entries));
    Stage8LogicalManifest {
        openings,
        pcs_opening_point,
    }
}

fn logical_openings<F, C>(entries: &[Stage8BatchEntry<'_, F, C>]) -> Vec<Stage8LogicalOpening<F>>
where
    F: Field,
{
    entries
        .iter()
        .map(|entry| Stage8LogicalOpening {
            id: entry.id,
            point: entry.own_point.clone(),
            claim: entry.opening_claim,
            scale: entry.scale,
        })
        .collect()
}

fn direct_physical_openings<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
) -> Vec<Stage8PhysicalOpening<F>>
where
    F: Field,
{
    entries
        .iter()
        .map(|entry| Stage8PhysicalOpening {
            id: entry.id,
            relation: entry.id,
            view: PhysicalView::Direct,
        })
        .collect()
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
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{BatchOpeningResult, OpeningsError};
    use jolt_poly::{MultilinearPoly, Polynomial};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ProofCheckingPcs;

    impl jolt_crypto::Commitment for ProofCheckingPcs {
        type Output = u64;
    }

    impl CommitmentScheme for ProofCheckingPcs {
        type Field = Fr;
        type Proof = Fr;
        type ProverSetup = ();
        type VerifierSetup = ();
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = ();

        fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            ((), ())
        }

        fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            (0, ())
        }

        fn open(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            eval
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            eval: Self::Field,
            proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            if eval == *proof {
                Ok(())
            } else {
                Err(OpeningsError::VerificationFailed)
            }
        }

        fn bind_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _eval: &Self::Field,
        ) {
        }
    }

    impl BatchOpeningScheme for ProofCheckingPcs {
        fn prove_batch<T, OpeningId, RelationId>(
            _setup: &Self::ProverSetup,
            _transcript: &mut T,
            statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _polynomials: &[Self::Polynomial],
            _hints: Vec<Self::OpeningHint>,
        ) -> Result<Self::Proof, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            statement
                .claims
                .first()
                .map(|claim| claim.claim)
                .ok_or(OpeningsError::VerificationFailed)
        }

        fn verify_batch<T, OpeningId, RelationId>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut T,
            statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            proof: &Self::Proof,
        ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            let claim = statement
                .claims
                .first()
                .ok_or(OpeningsError::VerificationFailed)?;
            if claim.claim != *proof {
                return Err(OpeningsError::VerificationFailed);
            }
            Ok(BatchOpeningResult {
                coefficients: vec![claim.scale],
                joint_commitment: claim.commitment,
                reduced_opening: claim.claim * claim.scale,
            })
        }
    }

    fn proof_checking_precommitted_statement(
        id: Stage8OpeningId,
        claim: Fr,
    ) -> Stage8OpeningStatement<Fr, u64, Fr> {
        BatchOpeningStatement {
            logical_point: vec![Fr::from_u64(0)],
            pcs_point: vec![Fr::from_u64(0)],
            layout_digest: [17; 32],
            claims: vec![BatchOpeningClaim {
                id,
                relation: id,
                commitment: 3,
                claim,
                view: PhysicalView::Direct,
                scale: Fr::from_u64(1),
            }],
        }
    }

    #[test]
    fn precommitted_opening_batches_require_exact_ordered_proofs() {
        let statements = vec![
            proof_checking_precommitted_statement(
                final_opening_id(JoltCommittedPolynomial::BytecodeChunk(0)).into(),
                Fr::from_u64(3),
            ),
            proof_checking_precommitted_statement(
                final_opening_id(JoltCommittedPolynomial::ProgramImageInit).into(),
                Fr::from_u64(5),
            ),
        ];
        let proofs = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-order");
        let coefficients = verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
            &(),
            &mut transcript,
            &statements,
            &proofs,
        )
        .expect("ordered precommitted proofs should verify");
        assert_eq!(coefficients, vec![Fr::from_u64(1), Fr::from_u64(1)]);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-extra");
        let mut extra_proofs = proofs.clone();
        extra_proofs.push(Fr::from_u64(8));
        assert!(matches!(
            verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
                &(),
                &mut transcript,
                &statements,
                &extra_proofs,
            ),
            Err(VerifierError::FinalOpeningVerificationFailed { reason })
                if reason.contains("expected 2 precommitted opening proofs, got 3")
        ));

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-reorder");
        let reordered = vec![Fr::from_u64(5), Fr::from_u64(3)];
        assert!(matches!(
            verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
                &(),
                &mut transcript,
                &statements,
                &reordered,
            ),
            Err(VerifierError::FinalOpeningVerificationFailed { .. })
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn precommitted_statements_use_akita_commitment_layout_digest() {
        let digest = [23; 32];
        let default_digest = [17; 32];
        let commitment = jolt_akita::AkitaCommitment {
            layout_digest: digest,
            num_vars: 1,
            poly_count: 1,
            native: vec![1],
        };
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::TrustedAdvice,
            JoltRelationId::AdviceClaimReduction,
        ));
        let entry = Stage8BatchEntry {
            id,
            commitment: &commitment,
            opening_claim: Some(Fr::from_u64(7)),
            own_point: vec![Fr::from_u64(0)],
            scale: Fr::from_u64(1),
        };
        let point = vec![Fr::from_u64(0)];
        let pcs_opening_point = Point::high_to_low(point.clone());
        let (_, statements) =
            precommitted_clear_statements(&[entry], default_digest, &point, &pcs_opening_point)
                .expect("precommitted statement should build");

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].layout_digest, digest);
        assert_ne!(statements[0].layout_digest, default_digest);
    }

    #[test]
    fn committed_program_batch_entries_require_final_openings() {
        let layout = JoltRaPolynomialLayout::new(1, 0, 0).expect("test RA layout should be valid");
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
            false,
            false,
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
            false,
            false,
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

    #[test]
    fn akita_unsigned_increment_entries_require_sources_and_chunk_count() {
        let commitment = 9_u64;
        let opening_point = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let error = akita_unsigned_inc_batch_entries::<Fr, _>(8, &opening_point, None, &commitment)
            .err()
            .expect("lattice unsigned increment openings require stage sources");
        assert!(matches!(
            error,
            VerifierError::MissingOpeningClaim { id }
                if id == lattice_formulas::unsigned_inc_chunk_opening(0)
        ));

        let short_chunks = vec![Fr::from_u64(0); 7];
        let sources = LatticeUnsignedIncFinalOpenings {
            chunk_point: &opening_point,
            chunk_claims: Some(&short_chunks),
            msb_point: &opening_point,
            msb_claim: Some(Fr::from_u64(0)),
        };
        let error = akita_unsigned_inc_batch_entries::<Fr, _>(
            8,
            &opening_point,
            Some(&sources),
            &commitment,
        )
        .err()
        .expect("lattice unsigned increment openings require every lower chunk");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("unsigned increment final chunk opening count mismatch")
        ));
    }

    #[test]
    fn akita_unsigned_increment_entries_use_chunk_and_msb_points() {
        let commitment = 9_u64;
        let opening_point = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
            Fr::from_u64(5),
        ];
        let chunk_point = vec![Fr::from_u64(3), Fr::from_u64(4)];
        let msb_point = vec![Fr::from_u64(5)];
        let chunks = vec![Fr::from_u64(0); 8];
        let sources = LatticeUnsignedIncFinalOpenings {
            chunk_point: &chunk_point,
            chunk_claims: Some(&chunks),
            msb_point: &msb_point,
            msb_claim: Some(Fr::from_u64(1)),
        };

        let entries = akita_unsigned_inc_batch_entries::<Fr, _>(
            8,
            &opening_point,
            Some(&sources),
            &commitment,
        )
        .expect("complete lattice unsigned increment sources should produce entries");

        assert_eq!(entries.len(), 9);
        for (index, entry) in entries.iter().take(8).enumerate() {
            assert_eq!(
                entry.id,
                Stage8OpeningId::from(lattice_formulas::unsigned_inc_chunk_opening(index))
            );
            assert_eq!(entry.own_point, chunk_point);
        }
        let msb_entry = entries.last().expect("MSB entry should be last");
        assert_eq!(
            msb_entry.id,
            Stage8OpeningId::from(lattice_formulas::unsigned_inc_msb_opening())
        );
        assert_eq!(msb_entry.own_point, msb_point);
    }
}
