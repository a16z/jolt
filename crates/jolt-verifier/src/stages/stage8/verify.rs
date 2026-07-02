use super::outputs::{Stage8ClearOutput, Stage8Output, Stage8ZkOutput};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{
        stage6b::{outputs::Stage6bOutputClaims, Stage6bOutput},
        stage7::{
            outputs::{PrecommittedFinalOpening, Stage7OutputClaims},
            Stage7Output,
        },
    },
    verifier::CheckedInputs,
    VerifierError,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        committed_openings::{
            commitment_embedding_scale, final_opening_id, final_opening_point,
            final_opening_polynomial_order, FinalOpeningPointInputs,
        },
        dimensions::JoltFormulaDimensions,
        ra::JoltRaPolynomialLayout,
    },
    JoltCommittedPolynomial, JoltOpeningId,
};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim, ZkOpeningScheme,
};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

struct Stage8BatchEntry<'a, F: Field, C> {
    id: JoltOpeningId,
    commitment: &'a C,
    /// `None` in ZK mode, where opening claims stay committed.
    opening_claim: Option<F>,
    /// Lagrange factor embedding this polynomial's own opening point into the
    /// unified opening point.
    scale: F,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 takes the shared formula dimensions, trusted-advice commitment, and the two upstream stage outputs it batches; bundling them would add indirection."
)]
pub fn verify<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    stage6: &Stage6bOutput<F, VC::Output>,
    stage7: &Stage7Output<F, VC::Output>,
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
    let log_t = formula_dimensions.trace.log_t();
    let layout = formula_dimensions.ra_layout;

    let (hamming_opening_point, inc_opening_point, precommitted_finals, clear_claims) =
        match (stage6, stage7) {
            (Stage6bOutput::Clear(stage6), Stage7Output::Clear(stage7)) => (
                stage7.hamming_weight_opening_point.as_slice(),
                stage6.output_points.inc_opening_point(),
                stage7.precommitted_final_openings.as_slice(),
                Some((&stage6.output_values, &stage7.output_values)),
            ),
            (Stage6bOutput::Zk(stage6), Stage7Output::Zk(stage7)) => (
                stage7.hamming_weight_opening_point.as_slice(),
                stage6.output_points.inc_opening_point(),
                stage7.precommitted_final_openings.as_slice(),
                None,
            ),
            (Stage6bOutput::Clear(_), Stage7Output::Zk(_)) => {
                return Err(VerifierError::ExpectedClearProof { field: "stage7" });
            }
            (Stage6bOutput::Zk(_), Stage7Output::Clear(_)) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage7" });
            }
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
    let opening_ids: Vec<JoltOpeningId> = entries.iter().map(|entry| entry.id).collect();

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
    clear_claims: Option<(&Stage6bOutputClaims<F>, &Stage7OutputClaims<F>)>,
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
    // The prover's final PCS batch order intentionally differs from proof payload order.
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
                JoltCommittedPolynomial::InstructionRa(index)
                | JoltCommittedPolynomial::BytecodeRa(index)
                | JoltCommittedPolynomial::RamRa(index) => {
                    let (commitment_list, claim_list): (&[PCS::Output], Option<&[F]>) =
                        match polynomial {
                            JoltCommittedPolynomial::InstructionRa(_) => (
                                &proof.commitments.ra.instruction,
                                clear_claims.map(|(_, stage7)| {
                                    stage7
                                        .hamming_weight_claim_reduction
                                        .instruction_ra
                                        .as_slice()
                                }),
                            ),
                            JoltCommittedPolynomial::BytecodeRa(_) => (
                                &proof.commitments.ra.bytecode,
                                clear_claims.map(|(_, stage7)| {
                                    stage7.hamming_weight_claim_reduction.bytecode_ra.as_slice()
                                }),
                            ),
                            JoltCommittedPolynomial::RamRa(_) => (
                                &proof.commitments.ra.ram,
                                clear_claims.map(|(_, stage7)| {
                                    stage7.hamming_weight_claim_reduction.ram_ra.as_slice()
                                }),
                            ),
                            _ => unreachable!("outer arm matches only the one-hot RA families"),
                        };
                    let commitment = commitment_list
                        .get(index)
                        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    let opening_claim = claim_list
                        .map(|claims| {
                            claims
                                .get(index)
                                .copied()
                                .ok_or(VerifierError::MissingOpeningClaim { id })
                        })
                        .transpose()?;
                    (commitment, hamming_opening_point, opening_claim)
                }
                JoltCommittedPolynomial::TrustedAdvice
                | JoltCommittedPolynomial::UntrustedAdvice
                | JoltCommittedPolynomial::BytecodeChunk(_)
                | JoltCommittedPolynomial::ProgramImageInit => {
                    let opening = precommitted_final(polynomial)
                        .ok_or(VerifierError::MissingOpeningClaim { id })?;
                    let commitment = match polynomial {
                        JoltCommittedPolynomial::TrustedAdvice => trusted_advice_commitment,
                        JoltCommittedPolynomial::UntrustedAdvice => {
                            proof.untrusted_advice_commitment.as_ref()
                        }
                        JoltCommittedPolynomial::BytecodeChunk(index) => committed_program
                            .and_then(|committed| committed.bytecode_chunk_commitments.get(index)),
                        JoltCommittedPolynomial::ProgramImageInit => {
                            committed_program.map(|committed| &committed.program_image_commitment)
                        }
                        _ => unreachable!("outer arm matches only precommitted polynomials"),
                    }
                    .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
                    (commitment, opening.point.as_slice(), opening.opening_claim)
                }
            };
        entries.push(Stage8BatchEntry {
            id,
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
