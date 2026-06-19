use super::{
    inputs::Deps,
    outputs::{Stage8ClearOutput, Stage8OpeningId, Stage8Output, Stage8ZkOutput},
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{
        relations::OpeningClaim,
        stage6::inputs::Stage6OutputClaims,
        stage7::{inputs::Stage7OutputClaims, outputs::PrecommittedFinalOpening},
    },
    verifier::CheckedInputs,
    VerifierError,
};
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
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim, ZkOpeningScheme,
};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

struct Stage8BatchEntry<'a, F: Field, C> {
    id: Stage8OpeningId,
    commitment: &'a C,
    /// `None` in ZK mode, where opening claims stay committed.
    opening_claim: Option<F>,
    /// Lagrange factor embedding this polynomial's own opening point into the
    /// unified opening point.
    scale: F,
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
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Clone + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
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
            stage7.hamming_weight_opening_point.as_slice(),
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
    require_commitment_layout(&proof.commitments, layout)?;

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

    let entries = batch_entries(
        preprocessing,
        proof,
        layout,
        trusted_advice_commitment,
        &opening_point,
        hamming_opening_point,
        inc_opening_point,
        precommitted_finals,
        clear_claims,
    )?;
    let opening_ids: Vec<Stage8OpeningId> = entries.iter().map(|entry| entry.id).collect();

    if checked.zk {
        let gamma_powers = transcript.challenge_scalar_powers(entries.len());
        let commitments: Vec<PCS::Output> = entries
            .iter()
            .map(|entry| entry.commitment.clone())
            .collect();
        let joint_commitment = PCS::combine(&commitments, &gamma_powers);
        let constraint_coefficients = gamma_powers
            .iter()
            .zip(&entries)
            .map(|(gamma, entry)| *gamma * entry.scale)
            .collect::<Vec<_>>();

        let hiding_evaluation_commitment = PCS::verify_zk(
            &joint_commitment,
            pcs_opening_point.as_slice(),
            &proof.joint_opening_proof,
            &preprocessing.pcs_setup,
            transcript,
        )
        .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        })?;
        PCS::bind_zk_opening_inputs(
            transcript,
            pcs_opening_point.as_slice(),
            &hiding_evaluation_commitment,
        );

        return Ok(Stage8Output::Zk(Stage8ZkOutput {
            opening_ids,
            constraint_coefficients,
            pcs_opening_point,
            joint_commitment,
            hiding_evaluation_commitment,
        }));
    }

    let opening_claims = entries
        .iter()
        .map(|entry| {
            let opening_claim =
                entry
                    .opening_claim
                    .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                        reason: "missing clear opening claim in final batch".to_string(),
                    })?;
            Ok(VerifierOpeningClaim {
                commitment: entry.commitment.clone(),
                evaluation: EvaluationClaim::new(
                    pcs_opening_point.clone(),
                    opening_claim * entry.scale,
                ),
            })
        })
        .collect::<Result<Vec<_>, VerifierError>>()?;

    transcript.append(&LabelWithCount(b"rlc_claims", opening_claims.len() as u64));
    for claim in &opening_claims {
        claim.evaluation.value.append_to_transcript(transcript);
    }
    let gamma_powers = transcript.challenge_scalar_powers(opening_claims.len());

    let joint_claim = gamma_powers
        .iter()
        .zip(&opening_claims)
        .fold(PCS::Field::zero(), |claim, (gamma, opening)| {
            claim + *gamma * opening.evaluation.value
        });
    let commitments = opening_claims
        .iter()
        .map(|claim| claim.commitment.clone())
        .collect::<Vec<_>>();
    let joint_commitment = PCS::combine(&commitments, &gamma_powers);
    let constraint_coefficients = gamma_powers
        .iter()
        .zip(&entries)
        .map(|(gamma, entry)| *gamma * entry.scale)
        .collect::<Vec<_>>();

    PCS::verify(
        &joint_commitment,
        pcs_opening_point.as_slice(),
        joint_claim,
        &proof.joint_opening_proof,
        &preprocessing.pcs_setup,
        transcript,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })?;
    PCS::bind_opening_inputs(transcript, pcs_opening_point.as_slice(), &joint_claim);

    Ok(Stage8Output::Clear(Stage8ClearOutput {
        opening_claims,
        opening_ids,
        constraint_coefficients,
        pcs_opening_point,
        joint_claim,
        joint_commitment,
    }))
}

/// Builds the final PCS batch in the canonical order from
/// [`final_opening_polynomial_order`], resolving each polynomial's commitment,
/// opening claim (clear mode only), and unified-point embedding scale.
#[expect(
    clippy::too_many_arguments,
    reason = "gathers per-polynomial sources from several stages"
)]
fn batch_entries<'a, F, PCS, VC, ZkProof>(
    preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    proof: &'a JoltProof<PCS, VC, ZkProof>,
    layout: JoltRaPolynomialLayout,
    trusted_advice_commitment: Option<&'a PCS::Output>,
    opening_point: &[F],
    hamming_opening_point: &[F],
    inc_opening_point: &[F],
    precommitted_finals: &'a [PrecommittedFinalOpening<F>],
    clear_claims: Option<(&Stage6OutputClaims<F>, &Stage7OutputClaims<OpeningClaim<F>>)>,
) -> Result<Vec<Stage8BatchEntry<'a, F, PCS::Output>>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let precommitted_final = |polynomial: JoltCommittedPolynomial| {
        precommitted_finals
            .iter()
            .find(|opening| opening.polynomial == polynomial)
    };
    let include_trusted = precommitted_final(JoltCommittedPolynomial::TrustedAdvice).is_some();
    let include_untrusted = precommitted_final(JoltCommittedPolynomial::UntrustedAdvice).is_some();
    let committed_program = preprocessing.program.committed();
    let order = final_opening_polynomial_order(
        layout,
        include_trusted,
        include_untrusted,
        committed_program.map(|committed| committed.bytecode_chunk_count()),
    );

    let mut entries = Vec::with_capacity(order.len());
    // Core's final PCS batch order intentionally differs from proof payload order.
    for polynomial in order {
        let id = final_opening_id(polynomial);
        let (commitment, own_point, opening_claim): (&PCS::Output, &[F], Option<F>) =
            match polynomial {
                JoltCommittedPolynomial::RamInc => (
                    &proof.commitments.ram_inc,
                    inc_opening_point,
                    clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.ram_inc),
                ),
                JoltCommittedPolynomial::RdInc => (
                    &proof.commitments.rd_inc,
                    inc_opening_point,
                    clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.rd_inc),
                ),
                JoltCommittedPolynomial::InstructionRa(index) => (
                    proof
                        .commitments
                        .ra
                        .instruction
                        .get(index)
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?,
                    hamming_opening_point,
                    match clear_claims {
                        Some((_, stage7)) => Some(
                            stage7
                                .hamming_weight_claim_reduction
                                .instruction_ra
                                .get(index)
                                .map(|claim| claim.value)
                                .ok_or(VerifierError::MissingOpeningClaim { id })?,
                        ),
                        None => None,
                    },
                ),
                JoltCommittedPolynomial::BytecodeRa(index) => (
                    proof
                        .commitments
                        .ra
                        .bytecode
                        .get(index)
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?,
                    hamming_opening_point,
                    match clear_claims {
                        Some((_, stage7)) => Some(
                            stage7
                                .hamming_weight_claim_reduction
                                .bytecode_ra
                                .get(index)
                                .map(|claim| claim.value)
                                .ok_or(VerifierError::MissingOpeningClaim { id })?,
                        ),
                        None => None,
                    },
                ),
                JoltCommittedPolynomial::RamRa(index) => (
                    proof
                        .commitments
                        .ra
                        .ram
                        .get(index)
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?,
                    hamming_opening_point,
                    match clear_claims {
                        Some((_, stage7)) => Some(
                            stage7
                                .hamming_weight_claim_reduction
                                .ram_ra
                                .get(index)
                                .map(|claim| claim.value)
                                .ok_or(VerifierError::MissingOpeningClaim { id })?,
                        ),
                        None => None,
                    },
                ),
                JoltCommittedPolynomial::TrustedAdvice => {
                    let opening = precommitted_final(polynomial)
                        .ok_or(VerifierError::MissingOpeningClaim { id })?;
                    let commitment = trusted_advice_commitment
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    (commitment, opening.point.as_slice(), opening.opening_claim)
                }
                JoltCommittedPolynomial::UntrustedAdvice => {
                    let opening = precommitted_final(polynomial)
                        .ok_or(VerifierError::MissingOpeningClaim { id })?;
                    let commitment = proof
                        .untrusted_advice_commitment
                        .as_ref()
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    (commitment, opening.point.as_slice(), opening.opening_claim)
                }
                JoltCommittedPolynomial::BytecodeChunk(index) => {
                    let opening = precommitted_final(polynomial)
                        .ok_or(VerifierError::MissingOpeningClaim { id })?;
                    let commitment = committed_program
                        .and_then(|committed| committed.bytecode_chunk_commitments.get(index))
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    (commitment, opening.point.as_slice(), opening.opening_claim)
                }
                JoltCommittedPolynomial::ProgramImageInit => {
                    let opening = precommitted_final(polynomial)
                        .ok_or(VerifierError::MissingOpeningClaim { id })?;
                    let commitment = committed_program
                        .map(|committed| &committed.program_image_commitment)
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    (commitment, opening.point.as_slice(), opening.opening_claim)
                }
            };
        entries.push(Stage8BatchEntry {
            id: id.into(),
            commitment,
            opening_claim,
            scale: commitment_embedding_scale(opening_point, own_point),
        });
    }
    Ok(entries)
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
