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
        ra::JoltRaPolynomialLayout,
    },
    JoltCommittedPolynomial,
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
    let (hamming_opening_point, inc_opening_point, precommitted_finals, clear_claims) = match deps {
        Deps::Clear { stage6, stage7 } => (
            stage7
                .batch
                .hamming_weight_claim_reduction
                .opening_point
                .as_slice(),
            stage6.batch.inc_claim_reduction.opening_point.as_slice(),
            stage7.precommitted_final_openings.as_slice(),
            Some((&stage6.output_claims, &stage7.output_claims)),
        ),
        Deps::Zk { stage6, stage7 } => (
            stage7
                .hamming_weight_claim_reduction
                .opening_point
                .as_slice(),
            stage6.inc_claim_reduction.opening_point.as_slice(),
            stage7.precommitted_final_openings.as_slice(),
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
        CommitmentPayload::Akita(payload) => batch_entries(
            layout,
            committed_bytecode_chunk_count,
            &opening_point,
            hamming_opening_point,
            inc_opening_point,
            precommitted_finals,
            clear_claims,
            |_| Ok(&payload.packed_witness),
            #[cfg(feature = "field-inline")]
            Some(&payload.packed_witness),
        )?,
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
