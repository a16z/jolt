use jolt_claims::protocols::jolt::{
    geometry::{booleanity::BooleanityDimensions, dimensions::JoltFormulaDimensions},
    JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::{
    booleanity::{BooleanityAddressPhase, BooleanityAddressPhaseInputClaims},
    bytecode_read_raf::{
        bytecode_read_raf_address_phase_input_values_from_upstream, BytecodeReadRafAddressPhase,
    },
    outputs::{
        Stage6aCarriedChallenges, Stage6aClearOutput, Stage6aInputClaims, Stage6aOutput,
        Stage6aSumchecks, Stage6aZkOutput,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
        stage5::Stage5Output, stage6b::batch::bytecode_stage_points, zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6a's address-phase input claim folds all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage5: &Stage5Output<PCS::Field, VC::Output>,
) -> Result<Stage6aOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = formula_dimensions.trace.log_t();

    let committed_program = checked.precommitted.bytecode.is_some();

    // The upstream cycle/register points and entry index ride on the relation
    // (full geometry at construction) for the prover's address-phase kernel;
    // the verifier itself never evaluates them here.
    let stage1_cycle_binding =
        stage1
            .cycle_binding()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    let stage_points = bytecode_stage_points(
        &stage1_cycle_binding,
        stage2.batch_output_points(),
        stage3.output_points(),
        stage4.output_points(),
        stage5.output_points(),
    )?;
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;

    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let address_sumchecks = Stage6aSumchecks {
        bytecode_read_raf: BytecodeReadRafAddressPhase::new(
            formula_dimensions.bytecode_read_raf,
            committed_program,
            stage_points,
            entry_bytecode_index,
        ),
        booleanity: BooleanityAddressPhase::new(
            booleanity_dimensions,
            stage5.instruction_r_address().to_vec(),
            stage5.output_points().instruction_r_cycle().to_vec(),
        ),
    };

    // The generated per-member draw: the bytecode member's six squeezes (the
    // fold gamma plus the five per-stage folding gammas, each formerly an
    // inline `challenge_scalar_powers(..)` whose single squeeze's degree-1
    // power equals the squeezed scalar; byte- and value-equal, test-locked in
    // `bytecode_read_raf.rs` — stage 6b's folds expand the power vectors via
    // `stage_gamma_powers`, test-locked below), then the booleanity member's
    // override (the reference-address pad draw and the gamma; schedule-locked
    // in the tests below). The booleanity draws feed 6b too: the prover's
    // booleanity subprotocol samples them before the 6a batch runs, so the
    // transcript schedule fixes them here and they ride downstream as typed
    // upstream values (the same idiom as `Stage2ZkOutput`'s `product_tau_high`).
    let address_challenges = address_sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges {
        bytecode_read_raf: address_challenges.bytecode_read_raf,
        booleanity_reference_address: address_challenges.booleanity.reference_address.clone(),
        booleanity_reference_cycle: address_challenges.booleanity.reference_cycle.clone(),
        booleanity_gamma: address_challenges.booleanity.gamma,
    };

    // Every member's input points are empty (the address phase reads only
    // opening values; produced points derive from its own sumcheck point).
    let address_input_points = address_sumchecks.empty_input_points();

    if checked.zk {
        let consistency =
            address_sumchecks.verify_zk(&proof.stages.stage6a_sumcheck_proof, transcript)?;
        let output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage6a_sumcheck_proof,
            "stage6a_sumcheck_proof",
            address_sumchecks.output_claim_count(),
            JoltRelationId::BytecodeReadRaf,
        )?;
        let output_points = address_sumchecks
            .derive_opening_points(&consistency.challenges(), &address_input_points)?;
        return Ok(Stage6aOutput::Zk(Stage6aZkOutput {
            challenges: carried,
            consistency,
            output_claims,
            output_points,
        }));
    }

    let claims = &proof.clear_claims()?.stage6a;
    // Rejects val-stage claims whose presence or count disagrees with the
    // committed-program mode (the bytecode member's wire set carries the staged
    // `BytecodeValStage` ids exactly when the program is committed).
    address_sumchecks.validate_output_claims(claims)?;

    // The bytecode address-phase input claim is the gamma-folded bind of every
    // prior clear stage opening; the relation evaluates it through its input
    // `Expr` from these wired openings + the per-stage folding gammas.
    let address_input_values = Stage6aInputClaims {
        bytecode_read_raf: bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage1.clear()?.output_values,
            &stage2.clear()?.output_values,
            &stage3.clear()?.output_values,
            &stage4.clear()?.output_values,
            &stage5.clear()?.output_values,
        ),
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };

    let batch = address_sumchecks.verify_clear(
        &address_input_values,
        &address_challenges,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
    )?;
    let output_points = address_sumchecks
        .derive_opening_points(batch.reduction.point.as_slice(), &address_input_points)?;
    let expected_final_claim = address_sumchecks.expected_final_claim(
        &batch.coefficients,
        &address_input_points,
        claims,
        &output_points,
        &address_challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 6 });
    }

    // The address-phase opening order (bytecode `intermediate`, each `val_stages`,
    // then booleanity `intermediate`) is single-sourced from the generated
    // `append_output_claims` (member declaration order = canonical Fiat-Shamir
    // order; no alias dedup in the address phase).
    address_sumchecks.append_output_claims(transcript, claims);

    Ok(Stage6aOutput::Clear(Stage6aClearOutput {
        output_values: claims.clone(),
        output_points,
        challenges: carried,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::booleanity::BooleanityAddressPhaseOutputClaims;
    use super::super::bytecode_read_raf::BytecodeReadRafAddressPhaseOutputClaims;
    use super::super::outputs::Stage6aOutputClaims;
    use super::*;
    use crate::stages::relations::append_recording::RecordingTranscript;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use crate::stages::stage6b::batch::BytecodeStagePoints;
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// A stage-6a batch whose booleanity member has committed chunk width 2, so
    /// the reference-address draw pads a 1-variable stage-5 instruction address
    /// and truncates a 3-variable one.
    #[expect(clippy::unwrap_used)]
    fn sumchecks(
        instruction_r_address: Vec<Fr>,
        instruction_r_cycle: Vec<Fr>,
    ) -> Stage6aSumchecks<Fr> {
        Stage6aSumchecks::<Fr> {
            bytecode_read_raf: BytecodeReadRafAddressPhase::new(
                BytecodeReadRafDimensions::new(3, 4, 2),
                true,
                BytecodeStagePoints {
                    stage_cycle_points: Default::default(),
                    register_read_write_point: Vec::new(),
                    register_val_evaluation_point: Vec::new(),
                },
                0,
            ),
            booleanity: BooleanityAddressPhase::new(
                BooleanityDimensions::new(JoltRaPolynomialLayout::new(2, 1, 1).unwrap(), 4, 2),
                instruction_r_address,
                instruction_r_cycle,
            ),
        }
    }

    fn sample_claims() -> Stage6aOutputClaims<Fr> {
        Stage6aOutputClaims {
            bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                intermediate: fr(901),
                val_stages: Vec::new(),
            },
            booleanity: BooleanityAddressPhaseOutputClaims {
                intermediate: fr(902),
            },
        }
    }

    /// Pins the batch's `draw_challenges` to the retired hand pre-batch draw:
    /// the bytecode member's six gammas, then the booleanity member's override —
    /// the reference-address pad draw (the reversed stage-5 instruction address
    /// is narrower than the committed chunk width here) and the booleanity
    /// gamma. The reference vectors are pure computation (reversal, pad slot)
    /// off the stage-5 point the relation carries.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn draw_challenges_matches_inline_draw_sequence() {
        let address = vec![fr(11)];
        let cycle = vec![fr(21), fr(22), fr(23), fr(24)];
        let sumchecks = sumchecks(address.clone(), cycle.clone());

        let (inline_events, (inline_gammas, inline_reference_address, inline_gamma)) =
            record(|t| {
                let gammas: Vec<Fr> = (0..6).map(|_| t.challenge_scalar()).collect();
                let mut reference_address: Vec<Fr> = address.iter().rev().copied().collect();
                reference_address.extend(t.challenge_vector(1));
                (gammas, reference_address, t.challenge())
            });
        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            (1..=8u64).map(DrawEvent::Squeeze).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                challenges.bytecode_read_raf.gamma,
                challenges.bytecode_read_raf.stage1_gamma,
                challenges.bytecode_read_raf.stage2_gamma,
                challenges.bytecode_read_raf.stage3_gamma,
                challenges.bytecode_read_raf.stage4_gamma,
                challenges.bytecode_read_raf.stage5_gamma,
            ],
            inline_gammas,
        );
        assert_eq!(
            challenges.booleanity.reference_address,
            inline_reference_address
        );
        assert_eq!(
            challenges.booleanity.reference_cycle,
            cycle.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(challenges.booleanity.gamma, inline_gamma);
    }

    /// The truncate branch: a stage-5 instruction address wider than the
    /// committed chunk width keeps its reversed tail and draws no pad — only
    /// the booleanity gamma squeeze follows the bytecode member's six.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn draw_challenges_truncates_wide_reference_address_without_pad_draws() {
        let address = vec![fr(11), fr(12), fr(13)];
        let sumchecks = sumchecks(address.clone(), vec![fr(21)]);

        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(
            draw_events,
            (1..=7u64).map(DrawEvent::Squeeze).collect::<Vec<_>>()
        );
        let reversed: Vec<Fr> = address.iter().rev().copied().collect();
        assert_eq!(challenges.booleanity.reference_address, reversed[1..]);
        assert_eq!(challenges.booleanity.reference_cycle, vec![fr(21)]);
        assert_eq!(challenges.booleanity.gamma, fr(7));
    }

    /// Locks the stage-6a address-phase Fiat-Shamir append order against silent
    /// drift: bytecode read-RAF `intermediate`, each `val_stages` entry, then
    /// booleanity `intermediate`. Single-sourced from the generated
    /// `append_output_claims`.
    #[test]
    fn stage6a_output_claims_append_follows_canonical_order() {
        let sumchecks = sumchecks(Vec::new(), Vec::new());
        let mut claims = sample_claims();
        claims.bytecode_read_raf.val_stages = (903..908).map(fr).collect();

        let mut got = RecordingTranscript::default();
        sumchecks.append_output_claims(&mut got, &claims);

        let mut want = RecordingTranscript::default();
        for value in [901, 903, 904, 905, 906, 907, 902].map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }

    /// A transcript double whose every squeeze returns the same nontrivial
    /// scalar, so `challenge_scalar_powers`' output is a genuine power vector.
    #[derive(Clone, Default)]
    struct ConstantChallengeTranscript;

    impl Transcript for ConstantChallengeTranscript {
        type Challenge = Fr;
        fn new(_label: &'static [u8]) -> Self {
            Self
        }
        fn append_bytes(&mut self, _bytes: &[u8]) {}
        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(7)
        }
        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }

    /// The `stage_gamma_powers` expansion of a drawn scalar must equal
    /// `challenge_scalar_powers`' output for the same squeezed scalar — the
    /// value-identity the stage-6a generated draw substitution relies on
    /// (the prover draws each stage's power vector inline; the verifier stores
    /// the scalar and expands at the stage-6b folds).
    #[test]
    fn stage_gamma_powers_matches_challenge_scalar_powers() {
        use jolt_claims::protocols::jolt::geometry::bytecode::BYTECODE_STAGE_GAMMA_COUNTS;
        use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;

        let gamma = Fr::from_u64(7);
        let challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma,
            stage1_gamma: gamma,
            stage2_gamma: gamma,
            stage3_gamma: gamma,
            stage4_gamma: gamma,
            stage5_gamma: gamma,
        };
        let mut transcript = ConstantChallengeTranscript;
        for (powers, len) in challenges
            .stage_gamma_powers()
            .into_iter()
            .zip(BYTECODE_STAGE_GAMMA_COUNTS)
        {
            assert_eq!(powers, transcript.challenge_scalar_powers(len));
        }
    }
}
