use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::{
    geometry::dimensions::JoltFormulaDimensions, JoltCommittedPolynomial, JoltOpeningId,
    JoltRelationId,
};
use jolt_claims::OutputClaims;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Sponge, VerifierTranscript};

#[cfg(not(feature = "akita"))]
use super::committed_reduction_cycle_phase::{
    trusted_advice_cycle_phase_input_values_from_upstream,
    untrusted_advice_cycle_phase_input_values_from_upstream, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
#[cfg(feature = "field-inline")]
use super::field_registers_inc_claim_reduction::{
    field_registers_inc_claim_reduction_input_points_from_upstream,
    field_registers_inc_claim_reduction_input_values_from_upstream,
    FieldRegistersIncClaimReduction,
};
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::{
    inc_claim_reduction_input_points_from_upstream, inc_claim_reduction_input_values_from_upstream,
    IncClaimReduction,
};
#[cfg(not(feature = "akita"))]
use super::outputs::{Stage6bCarriedChallenges, Stage6bZkOutput};
use super::ram_hamming_booleanity::{RamHammingBooleanity, RamHammingBooleanityInputClaims};
use super::{
    batch::Stage6bDraws,
    booleanity::{Booleanity, BooleanityInputClaims},
    bytecode_read_raf::{BytecodeReadRafCycle, BytecodeReadRafInputClaims},
    committed_reduction_cycle_phase::{
        program_image_reduction_cycle_phase_input_values_from_upstream,
        BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseInputClaims,
        ProgramImageReductionCyclePhase,
    },
    instruction_ra_virtualization::{
        instruction_ra_virtualization_input_points_from_upstream,
        instruction_ra_virtualization_input_values_from_upstream, InstructionRaVirtualization,
    },
    outputs::{
        Stage6bClearOutput, Stage6bInputClaims, Stage6bInputPoints, Stage6bOutput,
        Stage6bOutputClaims, Stage6bOutputPoints, Stage6bSumchecks,
    },
    ram_ra_virtualization::{
        ram_ra_virtualization_input_points_from_upstream,
        ram_ra_virtualization_input_values_from_upstream, RamRaVirtualization,
    },
};
#[cfg(not(feature = "akita"))]
use crate::stages::zk::{committed, outputs::CommittedOutputClaimOutput};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    stages::{
        relations::{
            assemble_member_claims, assemble_member_claims_with, receive_member_openings,
            receive_member_openings_except, ComposedOpeningId,
        },
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
#[jolt_verifier_derive::fs_scope(Stage6b)]
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
        let mut committed_shape = None;
        let mut cycle_points = None;
        let (consistency, commitments) = sumchecks.verify_zk_with(
            |consistency| {
                let points =
                    sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;
                // The committed-claim count is the output-point-cell total minus
                // the booleanity bytecode-RA openings aliased to the bytecode
                // read-RAF points, the same runtime dedup as the clear wire.
                let aliases = BytecodeRaAliases::new(&points)?;
                let shape = committed::output_claim_shape(
                    checked,
                    points.point_count().saturating_sub(aliases.len()),
                )?;
                let rows = shape.row_count();
                committed_shape = Some(shape);
                cycle_points = Some(points);
                Ok(rows)
            },
            transcript,
        )?;
        let (Some(shape), Some(cycle_points)) = (committed_shape, cycle_points) else {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 6b committed shape was not derived".to_string(),
            });
        };

        return Ok(Stage6bOutput::Zk(Stage6bZkOutput {
            challenges: Stage6bCarriedChallenges {
                instruction_ra_gamma: draws.instruction_ra_gamma,
                inc_gamma: draws.inc_gamma,
                #[cfg(feature = "field-inline")]
                field_registers_inc_gamma: draws.field_registers_inc_gamma,
                bytecode_reduction_eta: draws.eta,
            },
            batch_consistency: consistency,
            batch_output_claims: CommittedOutputClaimOutput { shape, commitments },
            output_points: cycle_points,
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
    let (cycle_points, claims) = sumchecks.verify_clear_with(
        &input_values,
        &input_points,
        &cycle_challenges,
        transcript,
        6,
        receive_output_claims,
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

/// The booleanity bytecode-RA openings that share their bytecode read-RAF
/// source's opening point, keyed to that source. Such an opening is a copy of
/// its source: it is neither sent nor committed, and the verifier fills it from
/// the source. This is a runtime point equality the output `Expr`s cannot
/// express, which is why stage 6b curates its own claim order.
struct BytecodeRaAliases(BTreeMap<ComposedOpeningId, ComposedOpeningId>);

impl BytecodeRaAliases {
    fn new<F: JoltField>(points: &Stage6bOutputPoints<F>) -> Result<Self, VerifierError> {
        let booleanity_point = points.booleanity_opening_point().ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: "Stage 6 booleanity produced no opening point".to_string(),
            }
        })?;
        Ok(Self(
            points
                .bytecode_read_raf
                .bytecode_ra
                .iter()
                .enumerate()
                .filter(|(_, point)| point.as_slice() == booleanity_point)
                .map(|(index, _)| {
                    let polynomial = JoltCommittedPolynomial::BytecodeRa(index);
                    (
                        JoltOpeningId::committed(polynomial, JoltRelationId::Booleanity).into(),
                        JoltOpeningId::committed(polynomial, JoltRelationId::BytecodeReadRaf)
                            .into(),
                    )
                })
                .collect(),
        ))
    }

    #[cfg(not(feature = "akita"))]
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Receive the stage-6b output claims in the [`stage6b_opening_values`] order:
/// member declaration order, each member in its canonical order, with the
/// aliased booleanity bytecode-RA openings filled from their sources.
fn receive_output_claims<F: JoltField, H: Sponge>(
    sumchecks: &Stage6bSumchecks<F>,
    points: &Stage6bOutputPoints<F>,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<Stage6bOutputClaims<F>, VerifierError> {
    let aliases = BytecodeRaAliases::new(points)?;
    let mut received = BTreeMap::new();
    received.extend(receive_member_openings(
        &sumchecks.bytecode_read_raf,
        transcript,
    )?);
    received.extend(receive_member_openings_except(
        &sumchecks.booleanity,
        |id| aliases.0.contains_key(id),
        transcript,
    )?);
    received.extend(receive_member_openings(
        &sumchecks.ram_hamming_booleanity,
        transcript,
    )?);
    received.extend(receive_member_openings(
        &sumchecks.ram_ra_virtualization,
        transcript,
    )?);
    received.extend(receive_member_openings(
        &sumchecks.instruction_ra_virtualization,
        transcript,
    )?);
    #[cfg(not(feature = "akita"))]
    received.extend(receive_member_openings(
        &sumchecks.inc_claim_reduction,
        transcript,
    )?);
    #[cfg(feature = "field-inline")]
    received.extend(receive_member_openings(
        &sumchecks.field_registers_inc_claim_reduction,
        transcript,
    )?);
    #[cfg(not(feature = "akita"))]
    if let Some(member) = &sumchecks.trusted_advice {
        received.extend(receive_member_openings(member, transcript)?);
    }
    #[cfg(not(feature = "akita"))]
    if let Some(member) = &sumchecks.untrusted_advice {
        received.extend(receive_member_openings(member, transcript)?);
    }
    if let Some(member) = &sumchecks.bytecode_reduction {
        received.extend(receive_member_openings(member, transcript)?);
    }
    if let Some(member) = &sumchecks.program_image_reduction {
        received.extend(receive_member_openings(member, transcript)?);
    }

    Ok(Stage6bOutputClaims {
        bytecode_read_raf: assemble_member_claims::<F, BytecodeReadRafCycle<F>>(&received)?,
        booleanity: assemble_member_claims_with::<F, Booleanity<F>>(&received, |id| {
            aliases.0.get(id).copied()
        })?,
        ram_hamming_booleanity: assemble_member_claims::<F, RamHammingBooleanity<F>>(&received)?,
        ram_ra_virtualization: assemble_member_claims::<F, RamRaVirtualization<F>>(&received)?,
        instruction_ra_virtualization: assemble_member_claims::<F, InstructionRaVirtualization<F>>(
            &received,
        )?,
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: assemble_member_claims::<F, IncClaimReduction<F>>(&received)?,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: assemble_member_claims::<
            F,
            FieldRegistersIncClaimReduction<F>,
        >(&received)?,
        #[cfg(not(feature = "akita"))]
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| assemble_member_claims::<F, TrustedAdviceCyclePhase<F>>(&received))
            .transpose()?,
        #[cfg(not(feature = "akita"))]
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| assemble_member_claims::<F, UntrustedAdviceCyclePhase<F>>(&received))
            .transpose()?,
        bytecode_reduction: sumchecks
            .bytecode_reduction
            .as_ref()
            .map(|_| assemble_member_claims::<F, BytecodeReductionCyclePhase<F>>(&received))
            .transpose()?,
        program_image_reduction: sumchecks
            .program_image_reduction
            .as_ref()
            .map(|_| assemble_member_claims::<F, ProgramImageReductionCyclePhase<F>>(&received))
            .transpose()?,
    })
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

/// The stage-6b Fiat-Shamir opening-claim values in canonical absorb order.
/// Full relations and the optional members single-source their per-field order
/// from the `OutputClaims` derive's `opening_values`; `booleanity` stays
/// explicit because its `bytecode_ra` openings are conditionally deduped
/// against the bytecode-read-RAF points (a runtime point-equality the output
/// `Expr`s cannot express). Public because the prover's recorder absorbs the
/// same curated sequence.
pub fn stage6b_opening_values<F: JoltField>(
    claims: &Stage6bOutputClaims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) -> Vec<F> {
    let mut values = claims.bytecode_read_raf.opening_values();
    values.extend(&claims.booleanity.instruction_ra);
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        values.push(*opening_claim);
    }
    values.extend(&claims.booleanity.ram_ra);
    #[cfg(feature = "akita")]
    {
        values.extend(&claims.booleanity.balanced_inc_digits);
        values.push(claims.booleanity.balanced_inc_carry);
    }
    values.extend(claims.ram_hamming_booleanity.opening_values());
    values.extend(claims.ram_ra_virtualization.opening_values());
    values.extend(claims.instruction_ra_virtualization.opening_values());
    #[cfg(not(feature = "akita"))]
    values.extend(claims.inc_claim_reduction.opening_values());
    #[cfg(feature = "field-inline")]
    super::field_inline::splice_inc_values(&mut values, claims);
    // Each advice member is a single-slot per-kind claims struct, so it
    // contributes exactly its own kind's opening.
    #[cfg(not(feature = "akita"))]
    if let Some(advice) = &claims.trusted_advice {
        values.extend(advice.opening_values());
    }
    #[cfg(not(feature = "akita"))]
    if let Some(advice) = &claims.untrusted_advice {
        values.extend(advice.opening_values());
    }
    if let Some(reduction) = &claims.bytecode_reduction {
        values.extend(reduction.opening_values());
    }
    if let Some(reduction) = &claims.program_image_reduction {
        values.extend(reduction.opening_values());
    }
    values
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::expect_used)]
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
    use super::super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
    use super::super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;
    use super::*;
    use crate::stages::relations::append_recording::RecordingTranscript;
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

    const TEST_COMMITTED_CHUNK_BITS: usize = 8;

    fn shape_formula_dimensions() -> JoltFormulaDimensions {
        JoltFormulaDimensions::try_from(
            jolt_claims::protocols::jolt::geometry::dimensions::JoltOneHotDimensions {
                log_t: 8,
                instruction_address_bits: 128,
                bytecode_k: 1024,
                ram_k: 4096,
                committed_chunk_bits: TEST_COMMITTED_CHUNK_BITS,
                lookup_virtual_chunk_bits: 32,
            },
        )
        .unwrap()
    }

    /// Claims whose every wire vector length matches `formula_dimensions` exactly.
    fn shape_matched_claims(formula_dimensions: &JoltFormulaDimensions) -> Stage6bOutputClaims<Fr> {
        let bytecode_ra_len =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf)
                .bytecode_ra
                .len();
        let ra_layout = formula_dimensions.ra_layout;
        #[cfg(not(feature = "akita"))]
        let bytecode_read_raf = BytecodeReadRafOutputClaims {
            bytecode_ra: vec![fr(1); bytecode_ra_len],
        };
        #[cfg(feature = "akita")]
        let bytecode_read_raf = LatticeBytecodeReadRafOutputClaims {
            bytecode_ra: vec![fr(1); bytecode_ra_len],
            fused_inc: fr(2),
        };
        #[cfg(not(feature = "akita"))]
        let booleanity = BooleanityOutputClaims {
            instruction_ra: vec![fr(3); ra_layout.instruction()],
            bytecode_ra: vec![fr(4); ra_layout.bytecode()],
            ram_ra: vec![fr(5); ra_layout.ram()],
        };
        #[cfg(feature = "akita")]
        let booleanity =
            jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims {
                instruction_ra: vec![fr(3); ra_layout.instruction()],
                bytecode_ra: vec![fr(4); ra_layout.bytecode()],
                ram_ra: vec![fr(5); ra_layout.ram()],
                balanced_inc_digits: vec![
                    fr(6);
                    jolt_claims::protocols::jolt::lattice::geometry::BalancedIncChunking::new(
                        TEST_COMMITTED_CHUNK_BITS
                    )
                    .unwrap()
                    .chunk_count()
                ],
                balanced_inc_carry: fr(7),
            };
        Stage6bOutputClaims {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: fr(8),
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                ram_ra: vec![
                    fr(9);
                    formula_dimensions
                        .ram_ra_virtualization
                        .num_committed_ra_polys()
                ],
            },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: vec![
                    fr(10);
                    formula_dimensions
                        .instruction_ra_virtualization
                        .num_committed_ra_polys()
                ],
            },
            #[cfg(not(feature = "akita"))]
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: fr(11),
                rd_inc: fr(12),
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionOutputClaims {
                rd_inc: fr(13),
            },
            #[cfg(not(feature = "akita"))]
            trusted_advice: None,
            #[cfg(not(feature = "akita"))]
            untrusted_advice: None,
            bytecode_reduction: None,
            program_image_reduction: None,
        }
    }

    fn validate_shape(
        formula_dimensions: &JoltFormulaDimensions,
        claims: &Stage6bOutputClaims<Fr>,
    ) -> Result<(), VerifierError> {
        #[cfg(not(feature = "akita"))]
        let result = validate_cycle_phase_claim_shape(formula_dimensions, claims, None);
        #[cfg(feature = "akita")]
        let result = validate_cycle_phase_claim_shape(
            formula_dimensions,
            claims,
            None,
            TEST_COMMITTED_CHUNK_BITS,
        );
        result
    }

    fn tamper_vec(vec: &mut Vec<Fr>, pad: bool) {
        if pad {
            vec.push(fr(99));
        } else {
            let _ = vec.pop();
        }
    }

    /// Every stage-6b wire claim vector is exact-length-pinned: padding or
    /// truncating any of them must be rejected before the claims reach the
    /// Fiat-Shamir absorb (v12 #116402 — trailing entries were absorbed,
    /// admitting non-canonical padded proofs).
    #[test]
    fn cycle_phase_claim_shape_pins_every_wire_vector_length() {
        let formula_dimensions = shape_formula_dimensions();
        let claims = shape_matched_claims(&formula_dimensions);
        validate_shape(&formula_dimensions, &claims).expect("shape-matched claims validate");

        type Tamper = fn(&mut Stage6bOutputClaims<Fr>, bool);
        #[cfg_attr(not(feature = "akita"), expect(unused_mut))]
        let mut tampers: Vec<(&str, Tamper)> = vec![
            ("bytecode_read_raf.bytecode_ra", |c, pad| {
                tamper_vec(&mut c.bytecode_read_raf.bytecode_ra, pad);
            }),
            ("booleanity.instruction_ra", |c, pad| {
                tamper_vec(&mut c.booleanity.instruction_ra, pad);
            }),
            ("booleanity.bytecode_ra", |c, pad| {
                tamper_vec(&mut c.booleanity.bytecode_ra, pad);
            }),
            ("booleanity.ram_ra", |c, pad| {
                tamper_vec(&mut c.booleanity.ram_ra, pad);
            }),
            ("ram_ra_virtualization.ram_ra", |c, pad| {
                tamper_vec(&mut c.ram_ra_virtualization.ram_ra, pad);
            }),
            (
                "instruction_ra_virtualization.committed_instruction_ra",
                |c, pad| {
                    tamper_vec(
                        &mut c.instruction_ra_virtualization.committed_instruction_ra,
                        pad,
                    );
                },
            ),
        ];
        #[cfg(feature = "akita")]
        tampers.push(("booleanity.balanced_inc_digits", |c, pad| {
            tamper_vec(&mut c.booleanity.balanced_inc_digits, pad);
        }));

        for (label, tamper) in tampers {
            for pad in [true, false] {
                let mut tampered = claims.clone();
                tamper(&mut tampered, pad);
                let verb = if pad { "padded" } else { "truncated" };
                assert!(
                    validate_shape(&formula_dimensions, &tampered).is_err(),
                    "{verb} {label} must be rejected"
                );
            }
        }
    }

    /// Locks the stage-6b cycle-phase Fiat-Shamir append order against silent drift.
    /// The full relations are single-sourced via their `OutputClaims` derive;
    /// `booleanity` (conditional `bytecode_ra` dedup) and the optional reductions
    /// stay explicit. Points are empty so no `bytecode_ra` element is deduped;
    /// the `None` reductions carry absent sentinels to prove they are not appended.
    #[test]
    fn append_opening_claims_follows_canonical_order() {
        let (claims, last) = sample_claims();

        let mut got = RecordingTranscript::default();
        append_opening_claims(&mut got, &claims, &[], &[]);

        let mut want = RecordingTranscript::default();
        for value in (1..=last).map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }

    #[test]
    fn bytecode_runtime_alias_requires_equal_claims() {
        let (mut claims, _) = sample_claims();
        let alias_point = vec![fr(41), fr(42)];
        let other_point = vec![fr(43), fr(44)];
        let bytecode_points = vec![alias_point.clone(), other_point];
        let source_claim = claims
            .bytecode_read_raf
            .bytecode_ra
            .first()
            .copied()
            .expect("sample has a bytecode read-RAF claim");
        *claims
            .booleanity
            .bytecode_ra
            .first_mut()
            .expect("sample has a bytecode booleanity claim") = fr(1) - source_claim;

        let error = validate_bytecode_ra_aliases(&claims, &bytecode_points, &alias_point)
            .expect_err("mismatched evaluations at an aliased point must be rejected");
        let polynomial = JoltCommittedPolynomial::BytecodeRa(0);
        assert!(matches!(
            error,
            VerifierError::StageClaimOpeningMismatch { stage, left, right }
                if stage == "Booleanity"
                    && left
                        == JoltOpeningId::committed(polynomial, JoltRelationId::Booleanity).into()
                    && right
                        == JoltOpeningId::committed(polynomial, JoltRelationId::BytecodeReadRaf)
                            .into()
        ));

        *claims
            .booleanity
            .bytecode_ra
            .first_mut()
            .expect("sample has a bytecode booleanity claim") = source_claim;
        validate_bytecode_ra_aliases(&claims, &bytecode_points, &alias_point)
            .expect("equal evaluations at an aliased point must validate");

        *claims
            .booleanity
            .bytecode_ra
            .first_mut()
            .expect("sample has a bytecode booleanity claim") = fr(99);
        validate_bytecode_ra_aliases(&claims, &bytecode_points, &[fr(45), fr(46)])
            .expect("different evaluations at different points are not aliases");
    }
}
