use super::outputs::Stage8Output;
#[cfg(not(feature = "akita"))]
use super::outputs::{Stage8ClearOutput, Stage8ZkOutput};
#[cfg(not(feature = "akita"))]
use super::precommitted::{precommitted_final_openings, PrecommittedFinalOpening};
#[cfg(not(feature = "akita"))]
use crate::proof::JoltCommitments;
#[cfg(not(feature = "akita"))]
use crate::stages::ids::VerifierOpeningId;
#[cfg(not(feature = "akita"))]
use crate::stages::{stage6b::outputs::Stage6bOutputClaims, stage7::outputs::Stage7OutputClaims};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage6b::Stage6bOutput, stage7::Stage7Output},
    verifier::CheckedInputs,
    VerifierError,
};
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltOpeningId;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::{
    geometry::{
        committed_openings::{
            commitment_embedding_scale, final_opening_id, final_opening_point,
            final_opening_polynomial_order, FinalOpeningPointInputs,
        },
        ra::JoltRaPolynomialLayout,
    },
    JoltCommittedPolynomial, JoltRelationId,
};
#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
#[cfg(not(feature = "akita"))]
use jolt_openings::{
    AdditivelyHomomorphic, EvaluationClaim, VerifierOpeningClaim, ZkEvaluationClaim,
    ZkOpeningScheme,
};
#[cfg(not(feature = "akita"))]
use jolt_poly::Point;
#[cfg(not(feature = "akita"))]
use jolt_transcript::LabelWithCount;
use jolt_transcript::{AppendToTranscript, Transcript};

#[cfg(not(feature = "akita"))]
/// One assembled final-opening batch entry. Public because the prover's
/// stage-8 recipe assembles its PCS batch statement through the same
/// [`batch_entries`] wiring. The id is the composite [`VerifierOpeningId`] so
/// the composed plan can carry the field-inline entry alongside the jolt ones
/// (under `field-inline`, spliced by the stage-8 `field_inline` seam).
pub struct Stage8BatchEntry<'a, F: JoltField, C> {
    pub id: VerifierOpeningId,
    pub commitment: &'a C,
    /// `None` in ZK mode, where opening claims stay committed.
    pub opening_claim: Option<F>,
    /// Lagrange factor embedding this polynomial's own opening point into the
    /// unified opening point.
    pub scale: F,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 takes the shared formula dimensions, trusted-advice commitment, and the two upstream stage outputs it batches; bundling them would add indirection."
)]
#[cfg(not(feature = "akita"))]
#[jolt_verifier_derive::fs_scope(Stage8)]
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
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Clone + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = formula_dimensions.trace.log_t();
    let layout = formula_dimensions.ra_layout;

    // Stage 7's produced opening points, and (clear mode) the stage-7 and stage-6b
    // output claims. The hamming-weight opening point and precommitted finals are
    // resolved off these — before any transcript operation — since the finals'
    // points anchor the unified opening point.
    let (stage7_points, clear) = match (stage6, stage7) {
        (Stage6bOutput::Clear(stage6), Stage7Output::Clear(stage7)) => (
            &stage7.output_points,
            Some((&stage7.output_values, &stage6.output_values)),
        ),
        (Stage6bOutput::Zk(_), Stage7Output::Zk(stage7)) => (&stage7.output_points, None),
        (Stage6bOutput::Clear(_), Stage7Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage7" });
        }
        (Stage6bOutput::Zk(_), Stage7Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage7" });
        }
    };
    let stage6_points = stage6.output_points();
    let inc_opening_point = stage6_points.inc_opening_point();
    // `batch_entries` reads the clear claims in (stage6, stage7) order.
    let clear_claims = clear.map(|(stage7_values, stage6_values)| (stage6_values, stage7_values));
    require_commitment_layout(&proof.commitments, layout)?;

    let hamming_opening_point = stage7_points
        .hamming_weight_opening_point()
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "stage 7 produced no hamming-weight openings".to_string(),
        })?;
    let precommitted_finals =
        precommitted_final_openings(&checked.precommitted, stage7_points, stage6_points, clear)?;

    let anchor_points: Vec<&[F]> = precommitted_finals
        .iter()
        .map(|opening| opening.point.as_slice())
        .collect();
    let opening_point = final_opening_point(FinalOpeningPointInputs {
        log_t,
        log_k_chunk: proof.one_hot_config.committed_chunk_bits(),
        trace_order: proof.trace_polynomial_order,
        hamming_weight_opening_point: hamming_opening_point.as_slice(),
        inc_claim_reduction_opening_point: inc_opening_point,
        precommitted_anchor_points: &anchor_points,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let pcs_opening_point = Point::high_to_low(opening_point.clone());

    let entries = batch_entries(
        preprocessing,
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        layout,
        trusted_advice_commitment,
        &opening_point,
        hamming_opening_point.as_slice(),
        inc_opening_point,
        &precommitted_finals,
        clear_claims,
    )?;
    #[cfg(feature = "field-inline")]
    let entries = {
        let mut entries = entries;
        super::field_inline::splice_final_opening(
            &mut entries,
            &proof.commitments,
            &opening_point,
            stage6_points.field_registers_inc_opening_point(),
            clear_claims.map(|(stage6, _)| stage6.field_registers_inc_claim_reduction.rd_inc),
        )?;
        entries
    };
    let opening_ids: Vec<VerifierOpeningId> = entries.iter().map(|entry| entry.id).collect();

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
        ZkEvaluationClaim::new(pcs_opening_point.as_slice(), &hiding_evaluation_commitment)
            .append_to_transcript(transcript);

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

    transcript.append(&LabelWithCount(
        b"rlc_claims",
        crate::num::u64_from_usize(opening_claims.len()),
    ));
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
    EvaluationClaim::new(pcs_opening_point.clone(), joint_claim).append_to_transcript(transcript);

    Ok(Stage8Output::Clear(Stage8ClearOutput {
        opening_claims,
        opening_ids,
        constraint_coefficients,
        pcs_opening_point,
        joint_claim,
        joint_commitment,
    }))
}

#[expect(
    clippy::too_many_arguments,
    reason = "gathers per-polynomial sources from several stages"
)]
#[cfg(not(feature = "akita"))]
/// Assemble the final-opening batch entries in `final_opening_polynomial_order`,
/// pairing each polynomial's commitment with its opening claim (clear mode) and
/// its Lagrange embedding scale. Public because the prover's stage-8 recipe
/// builds its PCS batch statement from the same assembly (passing its own
/// stage-0 commitments where the verifier passes the proof's).
pub fn batch_entries<'a, F, PCS, VC>(
    preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    commitments: &'a JoltCommitments<PCS::Output>,
    untrusted_advice_commitment: Option<&'a PCS::Output>,
    layout: JoltRaPolynomialLayout,
    trusted_advice_commitment: Option<&'a PCS::Output>,
    opening_point: &[F],
    hamming_opening_point: &[F],
    inc_opening_point: &[F],
    precommitted_finals: &'a [PrecommittedFinalOpening<F>],
    clear_claims: Option<(&Stage6bOutputClaims<F>, &Stage7OutputClaims<F>)>,
) -> Result<Vec<Stage8BatchEntry<'a, F, PCS::Output>>, VerifierError>
where
    F: JoltField,
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

    // Resolves one member of an indexed one-hot RA family: its commitment from
    // the family's commitment list and, in clear mode, its opening claim.
    fn ra_family_entry<'c, F: JoltField, O>(
        commitment_list: &'c [O],
        claim_list: Option<&[F]>,
        index: usize,
        polynomial: JoltCommittedPolynomial,
        id: JoltOpeningId,
    ) -> Result<(&'c O, Option<F>), VerifierError> {
        let commitment = commitment_list
            .get(index)
            .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
        let opening_claim = claim_list
            .map(|claims| {
                claims
                    .get(index)
                    .copied()
                    .ok_or(VerifierError::MissingOpeningClaim { id: id.into() })
            })
            .transpose()?;
        Ok((commitment, opening_claim))
    }

    // Pairs a precommitted polynomial's final opening with its commitment.
    fn precommitted_entry<'c, F: JoltField, O>(
        opening: Option<&'c PrecommittedFinalOpening<F>>,
        commitment: Option<&'c O>,
        polynomial: JoltCommittedPolynomial,
        id: JoltOpeningId,
    ) -> Result<(&'c O, &'c [F], Option<F>), VerifierError> {
        let opening = opening.ok_or(VerifierError::MissingOpeningClaim { id: id.into() })?;
        let commitment =
            commitment.ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
        Ok((commitment, opening.point.as_slice(), opening.opening_claim))
    }

    let mut entries = Vec::with_capacity(order.len());
    // The prover's final PCS batch order intentionally differs from proof payload order.
    for polynomial in order {
        let id = final_opening_id(polynomial);
        let (commitment, own_point, opening_claim): (&PCS::Output, &[F], Option<F>) =
            match polynomial {
                JoltCommittedPolynomial::RamInc => (
                    &commitments.ram_inc,
                    inc_opening_point,
                    clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.ram_inc),
                ),
                JoltCommittedPolynomial::RdInc => (
                    &commitments.rd_inc,
                    inc_opening_point,
                    clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.rd_inc),
                ),
                JoltCommittedPolynomial::InstructionRa(index) => {
                    let (commitment, opening_claim) = ra_family_entry(
                        &commitments.instruction_ra,
                        clear_claims.map(|(_, stage7)| {
                            stage7
                                .hamming_weight_claim_reduction
                                .instruction_ra
                                .as_slice()
                        }),
                        index,
                        polynomial,
                        id,
                    )?;
                    (commitment, hamming_opening_point, opening_claim)
                }
                JoltCommittedPolynomial::BytecodeRa(index) => {
                    let (commitment, opening_claim) = ra_family_entry(
                        &commitments.bytecode_ra,
                        clear_claims.map(|(_, stage7)| {
                            stage7.hamming_weight_claim_reduction.bytecode_ra.as_slice()
                        }),
                        index,
                        polynomial,
                        id,
                    )?;
                    (commitment, hamming_opening_point, opening_claim)
                }
                JoltCommittedPolynomial::RamRa(index) => {
                    let (commitment, opening_claim) = ra_family_entry(
                        &commitments.ram_ra,
                        clear_claims.map(|(_, stage7)| {
                            stage7.hamming_weight_claim_reduction.ram_ra.as_slice()
                        }),
                        index,
                        polynomial,
                        id,
                    )?;
                    (commitment, hamming_opening_point, opening_claim)
                }
                JoltCommittedPolynomial::TrustedAdvice => precommitted_entry(
                    precommitted_final(polynomial),
                    trusted_advice_commitment,
                    polynomial,
                    id,
                )?,
                JoltCommittedPolynomial::UntrustedAdvice => precommitted_entry(
                    precommitted_final(polynomial),
                    untrusted_advice_commitment,
                    polynomial,
                    id,
                )?,
                JoltCommittedPolynomial::BytecodeChunk(index) => precommitted_entry(
                    precommitted_final(polynomial),
                    committed_program
                        .and_then(|committed| committed.bytecode_chunk_commitments.get(index)),
                    polynomial,
                    id,
                )?,
                JoltCommittedPolynomial::ProgramImageInit => precommitted_entry(
                    precommitted_final(polynomial),
                    committed_program.map(|committed| &committed.program_image_commitment),
                    polynomial,
                    id,
                )?,
                JoltCommittedPolynomial::BalancedIncDigit(_)
                | JoltCommittedPolynomial::BalancedIncCarry
                | JoltCommittedPolynomial::BytecodeRegisterSelector { .. }
                | JoltCommittedPolynomial::BytecodeCircuitFlag { .. }
                | JoltCommittedPolynomial::BytecodeInstructionFlag { .. }
                | JoltCommittedPolynomial::BytecodeLookupSelector { .. }
                | JoltCommittedPolynomial::BytecodeRafFlag { .. }
                | JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { .. }
                | JoltCommittedPolynomial::BytecodeImmBytes { .. }
                | JoltCommittedPolynomial::ProgramImageBytes => {
                    // Lattice-mode polynomials open through the fixed-prefix
                    // path in `stage8::packed`, never the homomorphic RLC batch.
                    return Err(VerifierError::FinalOpeningBatchFailed {
                        reason: format!(
                            "polynomial {polynomial:?} is not part of the stage 8 prover order"
                        ),
                    });
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

#[cfg(not(feature = "akita"))]
fn require_commitment_layout<C>(
    commitments: &JoltCommitments<C>,
    layout: JoltRaPolynomialLayout,
) -> Result<(), VerifierError> {
    // The FR commitment payload is part of the expected layout: the composed
    // final opening cannot assemble without the `FieldRdInc` commitment.
    #[cfg(feature = "field-inline")]
    super::field_inline::require_commitment(commitments)?;
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "layout totals are small per-polynomial chunk counts; the sum cannot overflow usize"
    )]
    let expected = 2 + layout.total();
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "a sum of in-memory commitment counts and a small constant cannot overflow usize"
    )]
    let got = 2
        + commitments.instruction_ra.len()
        + commitments.bytecode_ra.len()
        + commitments.ram_ra.len();
    if got != expected {
        return Err(VerifierError::InvalidCommitmentCount { expected, got });
    }
    if commitments.instruction_ra.len() != layout.instruction()
        || commitments.bytecode_ra.len() != layout.bytecode()
        || commitments.ram_ra.len() != layout.ram()
    {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "commitment layout mismatch: expected instruction={}, bytecode={}, ram={}; got instruction={}, bytecode={}, ram={}",
                layout.instruction(),
                layout.bytecode(),
                layout.ram(),
                commitments.instruction_ra.len(),
                commitments.bytecode_ra.len(),
                commitments.ram_ra.len()
            ),
        });
    }
    Ok(())
}

#[cfg(all(test, not(feature = "akita")))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::ids::VerifierOpeningId;
    use jolt_claims::protocols::jolt::geometry::committed_openings::{
        final_opening_id, final_opening_polynomial_order,
    };
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, Ring};

    fn layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 2).unwrap()
    }

    /// Synthetic entries in the jolt final-opening order, commitment-free
    /// (`C = ()`): the id/scale mechanics under test are commitment-agnostic.
    fn base_entries(include_advice: bool) -> Vec<Stage8BatchEntry<'static, Fr, ()>> {
        final_opening_polynomial_order(layout(), include_advice, include_advice, None)
            .into_iter()
            .map(|polynomial| Stage8BatchEntry {
                id: final_opening_id(polynomial).into(),
                commitment: &(),
                opening_claim: None,
                scale: Fr::from_u64(1),
            })
            .collect()
    }

    fn jolt_id(polynomial: JoltCommittedPolynomial) -> VerifierOpeningId {
        final_opening_id(polynomial).into()
    }

    /// FR-off pin: the batch plan is exactly the jolt-typed final-opening
    /// order lifted into composite ids — no extra entries, unchanged order.
    #[test]
    fn base_final_opening_plan_is_the_jolt_order() {
        let expected: Vec<VerifierOpeningId> =
            final_opening_polynomial_order(layout(), true, true, None)
                .into_iter()
                .map(jolt_id)
                .collect();
        let ids: Vec<VerifierOpeningId> = base_entries(true)
            .into_iter()
            .map(|entry| entry.id)
            .collect();
        assert_eq!(ids, expected);
    }

    /// FR-on pin: the composed plan is exactly the spec's field-inline
    /// final-opening order — `RamInc@Inc`, `RdInc@Inc`,
    /// `FieldRdInc@FieldRegistersIncClaimReduction`, then the RA families and
    /// the advice entries (`specs/field-inline-protocol.md`, "Stage 6
    /// Composition" / the stage-8 final-opening order block).
    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_final_opening_plan_matches_the_spec_order() {
        use crate::proof::{FieldInlineCommitments, FieldRegistersCommitments};
        use jolt_claims::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;

        let commitments = JoltCommitments::new((), (), vec![(), ()], vec![(), ()], vec![()])
            .with_field_inline(FieldInlineCommitments {
                field_registers: FieldRegistersCommitments { rd_inc: () },
            });
        let opening_point = [2u64, 3, 5].map(Fr::from_u64);
        let field_point = [3u64, 5].map(Fr::from_u64);

        let mut entries = base_entries(true);
        crate::stages::stage8::field_inline::splice_final_opening(
            &mut entries,
            &commitments,
            &opening_point,
            &field_point,
            Some(Fr::from_u64(7)),
        )
        .unwrap();

        let ids: Vec<VerifierOpeningId> = entries.iter().map(|entry| entry.id).collect();
        let expected = vec![
            jolt_id(JoltCommittedPolynomial::RamInc),
            jolt_id(JoltCommittedPolynomial::RdInc),
            field_rd_inc_reduced().into(),
            jolt_id(JoltCommittedPolynomial::InstructionRa(0)),
            jolt_id(JoltCommittedPolynomial::InstructionRa(1)),
            jolt_id(JoltCommittedPolynomial::BytecodeRa(0)),
            jolt_id(JoltCommittedPolynomial::RamRa(0)),
            jolt_id(JoltCommittedPolynomial::RamRa(1)),
            jolt_id(JoltCommittedPolynomial::TrustedAdvice),
            jolt_id(JoltCommittedPolynomial::UntrustedAdvice),
        ];
        assert_eq!(ids, expected);

        // The spliced entry mirrors RdInc's embedding treatment: the same
        // dense embedding helper over the FR reduction's own point.
        let spliced = entries
            .iter()
            .find(|entry| entry.id == field_rd_inc_reduced().into())
            .unwrap();
        assert_eq!(
            spliced.scale,
            commitment_embedding_scale(&opening_point, &field_point)
        );
        assert_eq!(spliced.opening_claim, Some(Fr::from_u64(7)));

        // Advice-free batches splice at the same anchor position.
        let mut without_advice = base_entries(false);
        crate::stages::stage8::field_inline::splice_final_opening(
            &mut without_advice,
            &commitments,
            &opening_point,
            &field_point,
            None,
        )
        .unwrap();
        assert_eq!(
            without_advice.get(2).map(|entry| entry.id),
            Some(field_rd_inc_reduced().into())
        );
    }

    /// The splice fails closed on a missing FR commitment payload and on a
    /// plan without its RdInc anchor.
    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_splice_fails_closed() {
        use crate::proof::{FieldInlineCommitments, FieldRegistersCommitments};

        let opening_point = [2u64, 3].map(Fr::from_u64);
        let without_payload = JoltCommitments::new((), (), Vec::new(), Vec::new(), Vec::new());
        assert!(matches!(
            crate::stages::stage8::field_inline::splice_final_opening(
                &mut base_entries(false),
                &without_payload,
                &opening_point,
                &opening_point,
                None,
            ),
            Err(VerifierError::MissingProofPayload {
                field: "commitments.field_inline"
            })
        ));

        let commitments = JoltCommitments::new((), (), Vec::new(), Vec::new(), Vec::new())
            .with_field_inline(FieldInlineCommitments {
                field_registers: FieldRegistersCommitments { rd_inc: () },
            });
        let mut anchorless: Vec<Stage8BatchEntry<'_, Fr, ()>> = Vec::new();
        assert!(matches!(
            crate::stages::stage8::field_inline::splice_final_opening(
                &mut anchorless,
                &commitments,
                &opening_point,
                &opening_point,
                None,
            ),
            Err(VerifierError::FinalOpeningBatchFailed { .. })
        ));
    }
}

#[cfg(feature = "akita")]
#[expect(
    clippy::too_many_arguments,
    reason = "same signature as the homomorphic build's verify"
)]
#[jolt_verifier_derive::fs_scope(Stage8)]
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
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + AppendToTranscript + super::OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: super::OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    // Settle committed-program word/chunk claims against their one-hot decompositions.
    let reconstruction = super::reconstruction::verify(
        checked,
        proof.stages.reconstruction_sumcheck_proof.as_ref(),
        &proof.clear_claims()?.reconstruction,
        transcript,
        stage6.clear()?,
        stage7.clear()?,
    )?;

    // OneHotTrace then opens natively at its shared point; reconstruction leaves are
    // discharged by separate auxiliary packed openings.
    super::packed::verify(
        formula_dimensions,
        proof.one_hot_config,
        preprocessing,
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        #[cfg(feature = "field-inline")]
        proof.field_inc_limbs_commitment.as_ref(),
        #[cfg(feature = "field-inline")]
        proof.clear_claims()?.field_inc_limbs.as_ref(),
        &proof.joint_opening_proof,
        transcript,
        &checked.precommitted,
        stage6.clear()?,
        stage7.clear()?,
        &reconstruction,
    )?;

    Ok(Stage8Output::Clear)
}
