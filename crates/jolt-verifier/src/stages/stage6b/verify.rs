use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::sites::STAGE6B;

#[cfg(not(feature = "akita"))]
use super::committed_reduction_cycle_phase::{
    trusted_advice_cycle_phase_input_values_from_upstream,
    untrusted_advice_cycle_phase_input_values_from_upstream,
};
#[cfg(feature = "field-inline")]
use super::field_registers_inc_claim_reduction::{
    field_registers_inc_claim_reduction_input_points_from_upstream,
    field_registers_inc_claim_reduction_input_values_from_upstream,
};
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::{
    inc_claim_reduction_input_points_from_upstream, inc_claim_reduction_input_values_from_upstream,
};
#[cfg(not(feature = "akita"))]
use super::outputs::{Stage6bCarriedChallenges, Stage6bZkOutput};
use super::ram_hamming_booleanity::RamHammingBooleanityInputClaims;
use super::{
    batch::Stage6bDraws,
    booleanity::BooleanityInputClaims,
    bytecode_read_raf::BytecodeReadRafInputClaims,
    committed_reduction_cycle_phase::{
        program_image_reduction_cycle_phase_input_values_from_upstream,
        BytecodeReductionCyclePhaseInputClaims,
    },
    instruction_ra_virtualization::{
        instruction_ra_virtualization_input_points_from_upstream,
        instruction_ra_virtualization_input_values_from_upstream,
    },
    outputs::{
        Stage6bClearOutput, Stage6bInputClaims, Stage6bInputPoints, Stage6bOutput, Stage6bSumchecks,
    },
    ram_ra_virtualization::{
        ram_ra_virtualization_input_points_from_upstream,
        ram_ra_virtualization_input_values_from_upstream,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    stages::{
        stage1::Stage1Output,
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage3::Stage3Output,
        stage4::{Stage4ClearOutput, Stage4Output, Stage4OutputPoints},
        stage5::{Stage5Output, Stage5OutputClaims, Stage5OutputPoints},
        stage6a::{outputs::Stage6aOutputClaims, Stage6aOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6b consumes the stage-6a output plus all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
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
    stage6a: &Stage6aOutput<PCS::Field, VC::Output>,
) -> Result<Stage6bOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: CanonicalDecode,
    H: Sponge,
{
    transcript.site(STAGE6B);
    // The bytecode fold gamma shares stage 6a's squeeze; it and the booleanity
    // gamma ride on the stage-6a output as typed upstream values. The post-6a
    // draws and the challenges aggregate are the promoted two-front helpers.
    let carried = stage6a.challenges();
    let draws = Stage6bDraws::draw(transcript, checked.precommitted.bytecode.is_some());

    // The batch is built after the post-6a draws, directly from the upstream stage
    // outputs; `build` derives every mode-agnostic constructor leg internally.
    let sumchecks = Stage6bSumchecks::build(
        checked,
        preprocessing,
        formula_dimensions,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6a,
        draws.eta,
    )?;

    let cycle_challenges = sumchecks.cycle_challenges(carried, &draws);

    let input_points = stage6b_input_points_from_upstream(
        &sumchecks,
        stage2.batch_output_points(),
        stage4.output_points(),
        stage5.output_points(),
    );

    // No zk protocol exists over the packed axis, so the committed arm (and its
    // runtime point-alias dedup arithmetic) is base-only.
    #[cfg(not(feature = "akita"))]
    if checked.zk {
        let batch = sumchecks.verify_zk(checked.committed_row_len()?, &input_points, transcript)?;

        return Ok(Stage6bOutput::Zk(Stage6bZkOutput {
            challenges: Stage6bCarriedChallenges {
                instruction_ra_gamma: draws.instruction_ra_gamma,
                inc_gamma: draws.inc_gamma,
                #[cfg(feature = "field-inline")]
                field_registers_inc_gamma: draws.field_registers_inc_gamma,
                bytecode_reduction_eta: draws.eta,
            },
            batch_consistency: batch.consistency,
            batch_output_claims: batch.output_claims,
            output_points: batch.output_points,
        }));
    }

    let stage2 = stage2.clear()?;
    let stage4 = stage4.clear()?;
    let stage5 = stage5.clear()?;
    let claims_6a = &stage6a.clear()?.output_values;

    let input_values = stage6b_input_values_from_upstream(
        &sumchecks,
        claims_6a,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
    )?;
    let (cycle_points, claims) = sumchecks.verify_clear(
        &input_values,
        &input_points,
        &cycle_challenges,
        transcript,
        6,
    )?;

    Ok(Stage6bOutput::Clear(Stage6bClearOutput {
        output_values: claims,
        output_points: cycle_points,
        bytecode_reduction_weights: sumchecks
            .bytecode_reduction
            .as_ref()
            .map(|reduction| reduction.weights().clone()),
    }))
}

/// Assemble the stage-6b consumed opening *values* from the address-phase claims
/// and the upstream clear outputs into the generated `Stage6bInputClaims`
/// aggregate. The `Option` cells track member presence, so a present member always
/// has its input cell populated. Public because the prover's stage-6b recipe
/// builds its batch inputs through the same wiring.
pub fn stage6b_input_values_from_upstream<F: JoltField>(
    sumchecks: &Stage6bSumchecks<F>,
    address_claims: &Stage6aOutputClaims<F>,
    #[cfg_attr(feature = "akita", expect(unused_variables))] stage2: &Stage2BatchOutputClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5OutputClaims<F>,
) -> Result<Stage6bInputClaims<F>, VerifierError> {
    Ok(Stage6bInputClaims {
        bytecode_read_raf: BytecodeReadRafInputClaims {
            address_phase: address_claims.bytecode_read_raf.intermediate,
        },
        booleanity: BooleanityInputClaims {
            address_phase: address_claims.booleanity.intermediate,
        },
        ram_hamming_booleanity: RamHammingBooleanityInputClaims::default(),
        ram_ra_virtualization: ram_ra_virtualization_input_values_from_upstream(stage5),
        instruction_ra_virtualization: instruction_ra_virtualization_input_values_from_upstream(
            stage5,
        ),
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: inc_claim_reduction_input_values_from_upstream(
            stage2,
            &stage4.output_values,
            stage5,
        ),
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction:
            field_registers_inc_claim_reduction_input_values_from_upstream(
                &stage4.output_values,
                stage5,
            ),
        #[cfg(not(feature = "akita"))]
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| {
                trusted_advice_cycle_phase_input_values_from_upstream(&stage4.ram_val_check_init)
            })
            .transpose()?,
        #[cfg(not(feature = "akita"))]
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| {
                untrusted_advice_cycle_phase_input_values_from_upstream(&stage4.ram_val_check_init)
            })
            .transpose()?,
        bytecode_reduction: sumchecks.bytecode_reduction.as_ref().map(|_| {
            BytecodeReductionCyclePhaseInputClaims {
                val_stages: address_claims.bytecode_read_raf.val_stages.clone(),
            }
        }),
        program_image_reduction: sumchecks
            .program_image_reduction
            .as_ref()
            .map(|_| {
                program_image_reduction_cycle_phase_input_values_from_upstream(
                    &stage4.ram_val_check_init,
                )
            })
            .transpose()?,
    })
}

/// Assemble the stage-6b consumed opening *points*. ZK-agnostic: only the RA / inc
/// members read the upstream output-points aggregates (which both modes expose); the
/// remaining seven members derive their produced points from their own sumcheck point
/// and read no input point, so their cells come from the generated
/// `empty_input_points` (empty, and present for present `Option` members exactly as
/// the generated `derive_opening_points` requires).
pub fn stage6b_input_points_from_upstream<F: JoltField>(
    sumchecks: &Stage6bSumchecks<F>,
    #[cfg_attr(feature = "akita", expect(unused_variables))] stage2: &Stage2BatchOutputPoints<F>,
    #[cfg_attr(
        all(feature = "akita", not(feature = "field-inline")),
        expect(unused_variables)
    )]
    stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
) -> Stage6bInputPoints<F> {
    Stage6bInputPoints {
        ram_ra_virtualization: ram_ra_virtualization_input_points_from_upstream(stage5),
        instruction_ra_virtualization: instruction_ra_virtualization_input_points_from_upstream(
            stage5,
        ),
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: inc_claim_reduction_input_points_from_upstream(stage2, stage4, stage5),
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction:
            field_registers_inc_claim_reduction_input_points_from_upstream(stage4, stage5),
        ..sumchecks.empty_input_points()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "akita"))]
    use super::super::booleanity::BooleanityOutputClaims;
    #[cfg(not(feature = "akita"))]
    use super::super::bytecode_read_raf::BytecodeReadRafOutputClaims;
    #[cfg(feature = "akita")]
    use super::super::bytecode_read_raf::LatticeBytecodeReadRafOutputClaims;
    #[cfg(feature = "field-inline")]
    use super::super::field_registers_inc_claim_reduction::FieldRegistersIncClaimReductionOutputClaims;
    #[cfg(not(feature = "akita"))]
    use super::super::inc_claim_reduction::IncClaimReductionOutputClaims;
    use super::super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
    use super::super::outputs::Stage6bOutputClaims;
    use super::super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
    use super::super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;
    use super::*;
    use crate::stages::relations::{ClaimRoute, ClaimRoutes};
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Per-mode sample claims with sentinel values in the canonical append order: base
    /// interleaves the inc member after the RA virtualizations (and, under `field-inline`, the
    /// field-inline inc member after it); Akita carries the read-raf `FusedInc` cell and the
    /// lattice booleanity digit/carry cells instead.
    fn sample_claims() -> (Stage6bOutputClaims<Fr>, u64) {
        #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
        let last = 10;
        #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
        let last = 11;
        #[cfg(not(feature = "akita"))]
        let (bytecode_read_raf, booleanity) = (
            BytecodeReadRafOutputClaims {
                bytecode_ra: vec![fr(1), fr(2)],
            },
            BooleanityOutputClaims {
                instruction_ra: vec![fr(3)],
                bytecode_ra: vec![fr(4)],
                ram_ra: vec![fr(5)],
            },
        );
        #[cfg(all(feature = "akita", not(feature = "field-inline")))]
        let last = 11;
        #[cfg(all(feature = "akita", feature = "field-inline"))]
        let last = 12;
        #[cfg(feature = "akita")]
        let (bytecode_read_raf, booleanity) = (
            LatticeBytecodeReadRafOutputClaims {
                bytecode_ra: vec![fr(1), fr(2)],
                fused_inc: fr(3),
            },
            jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims {
                instruction_ra: vec![fr(4)],
                bytecode_ra: vec![fr(5)],
                ram_ra: vec![fr(6)],
                balanced_inc_digits: vec![fr(7)],
                balanced_inc_carry: fr(8),
            },
        );
        #[cfg(not(feature = "akita"))]
        let (hamming, ram_ra_virt, instruction_ra_virt) = (fr(6), fr(7), fr(8));
        #[cfg(feature = "akita")]
        let (hamming, ram_ra_virt, instruction_ra_virt) = (fr(9), fr(10), fr(11));
        (
            Stage6bOutputClaims {
                bytecode_read_raf,
                booleanity,
                ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                    ram_hamming_weight: hamming,
                },
                ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                    ram_ra: vec![ram_ra_virt],
                },
                instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: vec![instruction_ra_virt],
                },
                #[cfg(not(feature = "akita"))]
                inc_claim_reduction: IncClaimReductionOutputClaims {
                    ram_inc: fr(9),
                    rd_inc: fr(10),
                },
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionOutputClaims {
                    // The field-inline member appends last in canonical order on both
                    // commitment axes.
                    rd_inc: fr(last),
                },
                #[cfg(not(feature = "akita"))]
                trusted_advice: None,
                #[cfg(not(feature = "akita"))]
                untrusted_advice: None,
                bytecode_reduction: None,
                program_image_reduction: None,
            },
            last,
        )
    }

    /// Locks the stage-6b clear claim order (member declaration order, each
    /// member in its canonical order) against silent drift with distinct
    /// sentinels; the `None` reductions contribute nothing.
    #[test]
    fn wire_claims_follow_declaration_order() {
        let (claims, last) = sample_claims();
        assert_eq!(
            Stage6bSumchecks::wire_claim_values(&claims, &ClaimRoutes::default()),
            (1..=last).map(fr).collect::<Vec<_>>()
        );
    }

    /// A booleanity `bytecode_ra` cell routed as an alias of its bytecode
    /// read-RAF source is neither sent nor committed.
    #[test]
    fn aliased_bytecode_ra_is_not_sent() {
        let (claims, last) = sample_claims();
        let polynomial = JoltCommittedPolynomial::BytecodeRa(0);
        let mut routes = ClaimRoutes::default();
        routes.set(
            JoltOpeningId::committed(polynomial, JoltRelationId::Booleanity),
            ClaimRoute::Alias(
                JoltOpeningId::committed(polynomial, JoltRelationId::BytecodeReadRaf).into(),
            ),
        );
        let aliased = claims.booleanity.bytecode_ra.clone();
        let expected: Vec<Fr> = (1..=last)
            .map(fr)
            .filter(|value| !aliased.contains(value))
            .collect();
        assert_eq!(
            Stage6bSumchecks::wire_claim_values(&claims, &routes),
            expected
        );
        assert_eq!(
            Stage6bSumchecks::committed_claim_values(&claims, &routes),
            expected
        );
    }
}
