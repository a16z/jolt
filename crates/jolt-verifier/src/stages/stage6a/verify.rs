#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedClaims;

use jolt_claims::protocols::jolt::{geometry::dimensions::JoltFormulaDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

#[cfg(feature = "field-inline")]
use super::field_inline::field_inline_bytecode_read_raf_address_phase_input_values_from_upstream;
use super::{
    batch::Stage6aBuildParts,
    booleanity::BooleanityAddressPhaseInputClaims,
    bytecode_read_raf::bytecode_read_raf_address_phase_input_values_from_upstream,
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
        stage5::Stage5Output, zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6a's address-phase input claim folds all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
#[jolt_verifier_derive::fs_scope(Stage6a)]
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
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let address_sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
        formula_dimensions,
        committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
        committed_program: checked.precommitted.bytecode.is_some(),
        entry_bytecode_index,
        stage1_cycle_binding: &stage1_cycle_binding,
        stage2_points: stage2.batch_output_points(),
        stage3_points: stage3.output_points(),
        stage4_points: stage4.output_points(),
        stage5_points: stage5.output_points(),
    })?;

    let address_challenges = address_sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges::from(&address_challenges);

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
    address_sumchecks.validate_output_claims(claims)?;

    let base_input_values = bytecode_read_raf_address_phase_input_values_from_upstream(
        &stage1.clear()?.output_values,
        &stage2.clear()?.output_values,
        &stage3.clear()?.output_values,
        &stage4.clear()?.output_values,
        &stage5.clear()?.output_values,
    );
    #[cfg(feature = "akita")]
    let base_input_values =
        jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims {
            base: base_input_values,
            inc: crate::stages::stage6b::inc_claim_reduction::inc_claim_reduction_input_values_from_upstream(
                &stage2.clear()?.output_values,
                &stage4.clear()?.output_values,
                &stage5.clear()?.output_values,
            ),
        };
    #[cfg(feature = "field-inline")]
    let base_input_values = ComposedClaims {
        base: base_input_values,
        field_inline: field_inline_bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage4.clear()?.output_values,
            &stage5.clear()?.output_values,
        ),
    };
    let address_input_values = Stage6aInputClaims {
        bytecode_read_raf: base_input_values,
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };

    let output_points = address_sumchecks.verify_clear(
        &address_input_values,
        &address_input_points,
        &address_challenges,
        claims,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
        6,
    )?;

    address_sumchecks.append_output_claims(transcript, claims);

    Ok(Stage6aOutput::Clear(Stage6aClearOutput {
        output_values: claims.clone(),
        output_points,
        challenges: carried,
    }))
}

#[cfg(test)]
#[cfg_attr(
    not(feature = "field-inline"),
    expect(
        clippy::useless_conversion,
        reason = "field-inline selects composed claim and opening types"
    )
)]
mod tests {
    use super::super::booleanity::{BooleanityAddressPhase, BooleanityAddressPhaseOutputClaims};
    use super::super::bytecode_read_raf::{
        BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims, BytecodeStagePoints,
    };
    use super::super::outputs::Stage6aOutputClaims;
    use super::*;
    use crate::stages::relations::append_recording::RecordingTranscript;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

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
                    fused_inc_cycle_points: Vec::new(),
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
            }
            .into(),
            booleanity: BooleanityAddressPhaseOutputClaims {
                intermediate: fr(902),
            },
        }
    }

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
            sumchecks.booleanity.reference_cycle(),
            cycle.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(challenges.booleanity.gamma, inline_gamma);
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test fixture slices its own three-entry address"
    )]
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
        assert_eq!(sumchecks.booleanity.reference_cycle(), vec![fr(21)]);
        assert_eq!(challenges.booleanity.gamma, fr(7));
    }

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
