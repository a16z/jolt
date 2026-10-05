#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedClaims;

use jolt_claims::protocols::jolt::{geometry::dimensions::JoltFormulaDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::CanonicalDecode;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::sites::STAGE6A;

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
    stages::{
        stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
        stage5::Stage5Output,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6a's address-phase input claim folds all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
pub fn verify<PCS, VC, H>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut VerifierTranscript<'_, H>,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage5: &Stage5Output<PCS::Field, VC::Output>,
) -> Result<Stage6aOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: CanonicalDecode,
    H: Sponge,
{
    transcript.site(STAGE6A);
    // The upstream cycle/register points and entry index ride on the relation
    // (full geometry at construction) for the prover's address-phase kernel;
    // the verifier itself never evaluates them here.
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let address_sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
        formula_dimensions,
        committed_chunk_bits: checked.one_hot_config.committed_chunk_bits(),
        committed_program: checked.precommitted.bytecode.is_some(),
        entry_bytecode_index,
        stage1_cycle_binding: &stage1_cycle_binding,
        stage2_points: stage2.batch_output_points(),
        stage3_points: stage3.output_points(),
        stage4_points: stage4.output_points(),
        stage5_points: stage5.output_points(),
    })?;

    // The generated per-member draw: the bytecode member's six squeezes (the
    // fold gamma plus the five per-stage folding gammas, each formerly an
    // inline `challenge_powers(..)` whose single draw's degree-1
    // power equals the drawn scalar; byte- and value-equal, test-locked in
    // `bytecode_read_raf.rs` — stage 6b's folds expand the power vectors via
    // `stage_gamma_powers`, test-locked below), then the booleanity member's
    // override (the reference-address pad draw and the gamma; schedule-locked
    // in the tests below). The booleanity draws feed 6b too: the prover's
    // booleanity subprotocol samples them before the 6a batch runs, so the
    // transcript schedule fixes them here and they ride downstream as typed
    // upstream values (the same idiom as `Stage2ZkOutput`'s `product_tau_high`).
    let address_challenges = address_sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges::from(&address_challenges);

    // Every member's input points are empty (the address phase reads only
    // opening values; produced points derive from its own sumcheck point).
    let address_input_points = address_sumchecks.empty_input_points();

    if checked.zk {
        let batch = address_sumchecks.verify_zk(
            checked.committed_row_len()?,
            &address_input_points,
            transcript,
        )?;
        return Ok(Stage6aOutput::Zk(Stage6aZkOutput {
            challenges: carried,
            consistency: batch.consistency,
            output_claims: batch.output_claims,
            output_points: batch.output_points,
        }));
    }

    // The bytecode address-phase input claim is the gamma-folded bind of every
    // prior clear stage opening (plus, under akita, the four reduced `Inc`
    // claims at the fused-inc consumer stage slots); the relation evaluates it
    // through its input `Expr` from these wired openings + the per-stage
    // folding gammas.
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

    // The address-phase opening order (bytecode `intermediate`, each `val_stages`,
    // then booleanity `intermediate`) is single-sourced from the generated
    // `receive_output_claims` (member declaration order = canonical Fiat-Shamir
    // order; no alias dedup in the address phase). The bytecode member's wire set
    // carries the staged `BytecodeValClaim` ids exactly when the program is
    // committed.
    let (output_points, output_values) = address_sumchecks.verify_clear(
        &address_input_values,
        &address_input_points,
        &address_challenges,
        transcript,
        6,
    )?;

    Ok(Stage6aOutput::Clear(Stage6aClearOutput {
        output_values,
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
    use crate::stages::relations::test_transcript::{assert_same_draws, fresh};
    use crate::stages::relations::ClaimRoutes;
    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, Ring};

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

    /// The batch draws the bytecode member's six uniform gammas, then the
    /// booleanity member's reference-address pad (the reversed stage-5
    /// instruction address is narrower than the committed chunk width here, so
    /// one small challenge fills the missing slot) and its small gamma.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn draw_challenges_pads_narrow_reference_address() {
        let address = vec![fr(11)];
        let cycle = vec![fr(21), fr(22), fr(23), fr(24)];
        let sumchecks = sumchecks(address.clone(), cycle.clone());

        let (challenges, (gammas, pad, gamma)) = assert_same_draws(
            |t| sumchecks.draw_challenges(t).unwrap(),
            |t| {
                let gammas: Vec<Fr> = (0..6).map(|_| t.challenge()).collect();
                let pad: Vec<Fr> = t.challenges_small(1);
                (gammas, pad, t.challenge_small::<Fr>())
            },
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
            gammas,
        );
        assert_eq!(
            challenges.booleanity.reference_address,
            [fr(11)].into_iter().chain(pad).collect::<Vec<_>>()
        );
        assert_eq!(
            sumchecks.booleanity.reference_cycle(),
            cycle.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(challenges.booleanity.gamma, gamma);
    }

    /// The truncate branch: a stage-5 instruction address wider than the
    /// committed chunk width keeps its reversed tail and draws no pad — only
    /// the booleanity gamma follows the bytecode member's six.
    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test fixture slices its own three-entry address"
    )]
    fn draw_challenges_truncates_wide_reference_address_without_pad_draws() {
        let address = vec![fr(11), fr(12), fr(13)];
        let sumchecks = sumchecks(address.clone(), vec![fr(21)]);

        let (challenges, gamma) = assert_same_draws(
            |t| sumchecks.draw_challenges(t).unwrap(),
            |t| {
                for _ in 0..6 {
                    let _: Fr = t.challenge();
                }
                t.challenge_small::<Fr>()
            },
        );

        let reversed: Vec<Fr> = address.iter().rev().copied().collect();
        assert_eq!(challenges.booleanity.reference_address, reversed[1..]);
        assert_eq!(sumchecks.booleanity.reference_cycle(), vec![fr(21)]);
        assert_eq!(challenges.booleanity.gamma, gamma);
    }

    /// Locks the stage-6a address-phase opening order: bytecode read-RAF
    /// `intermediate`, each `val_stages` entry, then booleanity `intermediate`.
    #[test]
    fn stage6a_wire_claims_follow_canonical_order() {
        let mut claims = sample_claims();
        claims.bytecode_read_raf.val_stages = (903..908).map(fr).collect();

        assert_eq!(
            Stage6aSumchecks::wire_claim_values(&claims, &ClaimRoutes::default()),
            [901, 903, 904, 905, 906, 907, 902].map(fr).to_vec()
        );
    }

    /// `stage_gamma_powers` expands each stored stage gamma into the power
    /// vector `[1, gamma, ...]` of the stage's fold width, the vector
    /// `challenge_powers` yields from the same uniform draw.
    #[test]
    fn stage_gamma_powers_expand_each_stage_gamma() {
        use jolt_claims::protocols::jolt::geometry::bytecode::BYTECODE_STAGE_GAMMA_COUNTS;
        use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;

        let gamma: Fr = fresh().challenge();
        let challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma,
            stage1_gamma: gamma,
            stage2_gamma: gamma,
            stage3_gamma: gamma,
            stage4_gamma: gamma,
            stage5_gamma: gamma,
        };
        for (powers, len) in challenges
            .stage_gamma_powers()
            .into_iter()
            .zip(BYTECODE_STAGE_GAMMA_COUNTS)
        {
            assert_eq!(powers, fresh().challenge_powers::<Fr>(len));
        }
    }
}
