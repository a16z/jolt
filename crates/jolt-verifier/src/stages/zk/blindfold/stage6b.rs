use super::*;

use jolt_claims::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::relations::ram::{
    RamHammingBooleanityOutputClaims, RamRaVirtualizationOutputClaims,
};

pub(super) fn add_stage6b<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let bytecode_reduction_layout = input.checked.precommitted.bytecode.clone();
    let program_image_reduction_layout = input.checked.precommitted.program_image.clone();
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_claims = relations::booleanity::BooleanityCyclePhase::new(booleanity_dimensions);
    let ram_hamming_claims = relations::ram::HammingBooleanity::new(trace_dimensions);
    let ram_ra_claims =
        relations::ram::RaVirtualization::new(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = relations::instruction::RaVirtualization::new(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = relations::claim_reductions::increments::ClaimReduction::new(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_inc_claims = super::field_inline::stage6b_inc_relation(log_t);
    let trusted_layout = advice_layout(input, JoltAdviceKind::Trusted);
    let trusted_claims = trusted_layout.as_ref().map(|layout| {
        relations::claim_reductions::advice::TrustedCyclePhase::new(layout.dimensions())
    });
    let untrusted_layout = advice_layout(input, JoltAdviceKind::Untrusted);
    let untrusted_claims = untrusted_layout.as_ref().map(|layout| {
        relations::claim_reductions::advice::UntrustedCyclePhase::new(layout.dimensions())
    });
    let bytecode_reduction_claims = bytecode_reduction_layout.as_ref().map(|layout| {
        relations::claim_reductions::bytecode::CyclePhase::new((
            layout.dimensions(),
            layout.chunk_count(),
        ))
    });
    let program_image_reduction_claims = program_image_reduction_layout.as_ref().map(|layout| {
        relations::claim_reductions::program_image::CyclePhase::new(layout.dimensions())
    });

    // The committed and uncommitted cycle-phase relations are distinct types, so
    // collapse the active one into its domain and batch tuple here.
    let (bytecode_domain, bytecode_claim) = if bytecode_reduction_layout.is_some() {
        let claims = relations::bytecode::ReadRafCyclePhaseCommitted::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ));
        (claims.domain(), relation_claim(&claims))
    } else {
        let claims = relations::bytecode::ReadRafCyclePhase::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ));
        (claims.domain(), relation_claim(&claims))
    };

    let mut batch_claims = vec![
        bytecode_claim,
        relation_claim(&booleanity_claims),
        relation_claim(&ram_hamming_claims),
        relation_claim(&ram_ra_claims),
        relation_claim(&instruction_ra_claims),
        relation_claim(&inc_claims),
    ];
    // Member declaration order (= batching-coefficient draw order): the FR
    // increment reduction sits after the ordinary increment reduction and
    // before the optional advice cycle phases, exactly as in
    // `Stage6bSumchecks`.
    #[cfg(feature = "field-inline")]
    batch_claims.push(relation_claim(&field_registers_inc_claims));
    if let Some(claim) = trusted_claims {
        batch_claims.push(relation_claim(&claim));
    }
    if let Some(claim) = untrusted_claims {
        batch_claims.push(relation_claim(&claim));
    }
    if let Some(claim) = &bytecode_reduction_claims {
        batch_claims.push(relation_claim(claim));
    }
    if let Some(claim) = &program_image_reduction_claims {
        batch_claims.push(relation_claim(claim));
    }

    let booleanity_opening_point = input
        .stage6b
        .output_points
        .booleanity_opening_point()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: "Stage 6 booleanity produced no opening point".to_string(),
        })?;
    let (output_ids, aliases) = stage6b_output_ids_and_aliases::<PCS::Field>(
        formula_dimensions,
        &input.stage6b.output_points.bytecode_read_raf.bytecode_ra,
        booleanity_opening_point,
        trusted_layout.as_ref(),
        untrusted_layout.as_ref(),
        bytecode_reduction_layout.as_ref(),
        program_image_reduction_layout.as_ref(),
    );

    add_batched_stage(
        builder,
        "stage6.cycle_phase",
        bytecode_domain,
        &batch_claims,
        &input.stage6b.batch_consistency,
        &input.stage6b.batch_output_claims,
        values,
        output_ids,
        aliases,
        Vec::new(),
    )
}

/// The stage-6b committed output row order and alias rows: the bytecode-RA
/// rows with the booleanity dedup, the remaining members' canonical orders in
/// member declaration order, and — under `field-inline` — the reduced FR
/// `FieldRdInc` row after the ordinary increment-reduction outputs and before
/// the optional advice cycle phases: the clear absorb order
/// (`stage6b_opening_values`) exactly.
fn stage6b_output_ids_and_aliases<F: JoltField>(
    formula_dimensions: JoltFormulaDimensions,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
    trusted_layout: Option<&AdviceClaimReductionLayout>,
    untrusted_layout: Option<&AdviceClaimReductionLayout>,
    bytecode_reduction_layout: Option<&BytecodeClaimReductionLayout>,
    program_image_reduction_layout: Option<&ProgramImageClaimReductionLayout>,
) -> (Vec<VerifierOpeningId>, Vec<OpeningAlias<VerifierOpeningId>>) {
    let (mut output_ids, aliases) = stage6_cycle_output_openings_and_aliases(
        formula_dimensions,
        bytecode_ra_opening_points,
        booleanity_opening_point,
    );
    output_ids.extend(composite_ids(
        RamHammingBooleanityOutputClaims::<F> {
            ram_hamming_weight: F::zero(),
        }
        .canonical_order(),
    ));
    output_ids.extend(composite_ids(
        RamRaVirtualizationOutputClaims::<F> {
            ram_ra: vec![
                F::zero();
                formula_dimensions
                    .ram_ra_virtualization
                    .num_committed_ra_polys()
            ],
        }
        .canonical_order(),
    ));
    output_ids.extend(composite_ids(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    ));
    output_ids.extend(
        IncClaimReductionOutputClaims::<F> {
            ram_inc: F::zero(),
            rd_inc: F::zero(),
        }
        .canonical_order()
        .into_iter()
        .map(VerifierOpeningId::from),
    );
    // The reduced FR `FieldRdInc` row, after the ordinary increment-reduction
    // outputs and before the optional advice cycle phases — the clear absorb
    // order (`stage6b_opening_values`).
    #[cfg(feature = "field-inline")]
    output_ids.extend(super::field_inline::stage6b_inc_output_ids());
    if let Some(layout) = trusted_layout {
        output_ids.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Trusted, layout.dimensions())
                .into_iter()
                .map(VerifierOpeningId::from),
        );
    }
    if let Some(layout) = untrusted_layout {
        output_ids.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Untrusted, layout.dimensions())
                .into_iter()
                .map(VerifierOpeningId::from),
        );
    }
    if let Some(layout) = bytecode_reduction_layout {
        output_ids.extend(
            bytecode_reduction::cycle_phase_output_openings(
                layout.dimensions(),
                layout.chunk_count(),
            )
            .into_iter()
            .map(VerifierOpeningId::from),
        );
    }
    if let Some(layout) = program_image_reduction_layout {
        output_ids.extend(
            program_image::cycle_phase_output_openings(layout.dimensions())
                .into_iter()
                .map(VerifierOpeningId::from),
        );
    }
    (output_ids, aliases)
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "fail-closed: unmatched opening ids yield no alias and are reported missing below"
)]
fn stage6_cycle_output_openings_and_aliases<F: JoltField>(
    formula_dimensions: JoltFormulaDimensions,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> (Vec<VerifierOpeningId>, Vec<OpeningAlias<VerifierOpeningId>>) {
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    let booleanity_output_openings =
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout);

    let mut output_ids = composite_ids(bytecode_output_openings.bytecode_ra.clone());
    let mut aliases = Vec::new();
    for id in booleanity_output_openings {
        let source = match id {
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(index)),
                relation: JoltRelationId::Booleanity,
            } if bytecode_ra_opening_points
                .get(index)
                .is_some_and(|point| point.as_slice() == booleanity_opening_point) =>
            {
                bytecode_output_openings.bytecode_ra.get(index).copied()
            }
            _ => None,
        };
        if let Some(source) = source {
            aliases.push(OpeningAlias::new(id.into(), source.into()));
        } else {
            output_ids.push(id.into());
        }
    }

    (output_ids, aliases)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use crate::stages::stage6b::booleanity::BooleanityOutputClaims;
    use crate::stages::stage6b::bytecode_read_raf::BytecodeReadRafOutputClaims;
    use crate::stages::stage6b::inc_claim_reduction::IncClaimReductionOutputClaims;
    use crate::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
    #[cfg(feature = "field-inline")]
    use crate::stages::stage6b::outputs::FieldRegistersIncClaimReductionOutputClaims;
    use crate::stages::stage6b::outputs::Stage6bOutputClaims;
    use crate::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
    use crate::stages::stage6b::ram_ra_virtualization::RamRaVirtualizationOutputClaims;
    use crate::stages::stage6b::stage6b_opening_values;
    use jolt_claims::protocols::jolt::geometry::dimensions::JoltOneHotDimensions;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn formula_dimensions() -> JoltFormulaDimensions {
        JoltFormulaDimensions::try_from(JoltOneHotDimensions {
            log_t: 8,
            instruction_address_bits: 128,
            bytecode_k: 1024,
            ram_k: 4096,
            committed_chunk_bits: 8,
            lookup_virtual_chunk_bits: 32,
        })
        .unwrap()
    }

    /// Claims with distinct sentinel values, every wire vector sized from the
    /// formula dimensions.
    fn sentinel_claims(formula_dimensions: &JoltFormulaDimensions) -> Stage6bOutputClaims<Fr> {
        let bytecode_ra_len =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf)
                .bytecode_ra
                .len();
        let ra_layout = formula_dimensions.ra_layout;
        let mut next = {
            let mut counter = 0u64;
            move || {
                counter += 1;
                fr(counter)
            }
        };
        Stage6bOutputClaims {
            bytecode_read_raf: BytecodeReadRafOutputClaims {
                bytecode_ra: (0..bytecode_ra_len).map(|_| next()).collect(),
            },
            booleanity: BooleanityOutputClaims {
                instruction_ra: (0..ra_layout.instruction()).map(|_| next()).collect(),
                bytecode_ra: (0..ra_layout.bytecode()).map(|_| next()).collect(),
                ram_ra: (0..ra_layout.ram()).map(|_| next()).collect(),
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: next(),
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                ram_ra: (0..formula_dimensions
                    .ram_ra_virtualization
                    .num_committed_ra_polys())
                    .map(|_| next())
                    .collect(),
            },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: (0..formula_dimensions
                    .instruction_ra_virtualization
                    .num_committed_ra_polys())
                    .map(|_| next())
                    .collect(),
            },
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: next(),
                rd_inc: next(),
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionOutputClaims {
                rd_inc: next(),
            },
            trusted_advice: None,
            untrusted_advice: None,
            bytecode_reduction: None,
            program_image_reduction: None,
        }
    }

    /// The stage-6b committed row order is the clear curated absorb order
    /// (`stage6b_opening_values`), locked entry-for-entry over sentinel-valued
    /// claims — FR-on: the reduced `FieldRdInc` row after the ordinary
    /// increment-reduction outputs, before the (absent here) advice cycle
    /// phases. Empty points mean no booleanity dedup fires on either side.
    #[test]
    fn stage6b_output_ids_match_the_clear_absorb_order() {
        use jolt_claims::OutputClaims as _;

        let formula_dimensions = formula_dimensions();
        let claims = sentinel_claims(&formula_dimensions);
        let clear_values = stage6b_opening_values(&claims, &[], &[]);

        let (output_ids, aliases) = stage6b_output_ids_and_aliases::<Fr>(
            formula_dimensions,
            &[],
            &[],
            None,
            None,
            None,
            None,
        );
        assert!(aliases.is_empty());
        assert_eq!(output_ids.len(), clear_values.len());

        for (id, expected) in output_ids.iter().zip(clear_values) {
            let resolved = match id {
                VerifierOpeningId::Jolt(id) => claims
                    .bytecode_read_raf
                    .resolve_output(id)
                    .or_else(|| claims.booleanity.resolve_output(id))
                    .or_else(|| claims.ram_hamming_booleanity.resolve_output(id))
                    .or_else(|| claims.ram_ra_virtualization.resolve_output(id))
                    .or_else(|| claims.instruction_ra_virtualization.resolve_output(id))
                    .or_else(|| claims.inc_claim_reduction.resolve_output(id)),
                #[cfg(feature = "field-inline")]
                VerifierOpeningId::FieldInline(id) => claims
                    .field_registers_inc_claim_reduction
                    .resolve_output(id),
                #[cfg(not(feature = "field-inline"))]
                VerifierOpeningId::FieldInline(_) => None,
            };
            assert_eq!(
                resolved,
                Some(expected),
                "row {id:?} must sit at the clear absorb position of value {expected:?}",
            );
        }
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used, clippy::get_unwrap)]
#[expect(
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    reason = "tests use plain arithmetic on fixture data"
)]
mod field_inline_tests {
    use super::*;
    use crate::stages::field_inline_bytecode::{FieldInlineBytecodeFold, FieldInlineBytecodeTable};
    use crate::stages::relations::ConcreteSumcheck as _;
    use crate::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRaf, BytecodeReadRafCycleInputs, BytecodeReadRafInputClaims,
        BytecodeReadRafOutputClaims, BytecodeReadRafTableFoldInputs, READ_RAF_CYCLE_STAGES,
    };
    use jolt_claims::protocols::field_inline::geometry::bytecode::{
        FieldInlineBytecodeFlags, FieldInlineBytecodeOperands, FieldInlineBytecodeRow,
    };
    use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_claims::protocols::jolt::relations::bytecode::{
        BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafCyclePhaseChallenges,
        ReadRafCyclePhase,
    };
    use jolt_field::{Fr, Ring};
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn point(start: u64, len: usize) -> Vec<Fr> {
        (0..len as u64).map(|i| fr(start + i)).collect()
    }

    /// The lowered stage-6b bytecode output claim — the cycle symbolic output
    /// expression over the composed `StageValue(i)` publics
    /// (`composed_bytecode_stage_values` added onto the ordinary monolith
    /// publics) — evaluates identically to the clear composed
    /// `BytecodeReadRaf::expected_output` on a synthetic fixture.
    #[test]
    fn lowered_bytecode_output_matches_the_clear_composed_claim() {
        let log_t = 2usize;
        let log_k = 2usize;
        let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, 2);
        let r_address = point(10, log_k);
        let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] =
            core::array::from_fn(|stage| point(20 + 10 * stage as u64, log_t));
        let register_read_write_point = point(70, 4);
        let register_val_evaluation_point = point(80, 4);
        let field_read_write_point = [point(90, FIELD_REGISTERS_LOG_K), point(100, log_t)].concat();
        let field_val_evaluation_point =
            [point(110, FIELD_REGISTERS_LOG_K), point(120, log_t)].concat();
        let challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma: fr(501),
            stage1_gamma: fr(502),
            stage2_gamma: fr(503),
            stage3_gamma: fr(504),
            stage4_gamma: fr(505),
            stage5_gamma: fr(506),
        };
        let stage_gammas = challenges.stage_gamma_powers();
        let mut bytecode = vec![JoltInstructionRow::default(); 4];
        *bytecode.get_mut(0).unwrap() = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADD,
            address: 9,
            operands: NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 4,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let mut field_rows = vec![FieldInlineBytecodeRow::default(); 4];
        *field_rows.get_mut(0).unwrap() = FieldInlineBytecodeRow {
            operands: FieldInlineBytecodeOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: Some(3),
            },
            flags: FieldInlineBytecodeFlags {
                mul: true,
                ..FieldInlineBytecodeFlags::default()
            },
        };
        let table = FieldInlineBytecodeTable {
            rows: field_rows,
            field_register_log_k: FIELD_REGISTERS_LOG_K,
        };
        let entry_bytecode_index = 1usize;

        // The clear composed relation.
        let relation = BytecodeReadRaf::new(BytecodeReadRafCycleInputs {
            dimensions,
            r_address: r_address.clone(),
            stage_cycle_points: stage_cycle_points.clone(),
            entry_bytecode_index,
            committed_chunk_bits: 1,
            table_fold: Some(BytecodeReadRafTableFoldInputs {
                bytecode: &bytecode,
                register_read_write_point: &register_read_write_point,
                register_val_evaluation_point: &register_val_evaluation_point,
                stage_gammas: stage_gammas.each_ref().map(Vec::as_slice),
            }),
            field_inline: FieldInlineBytecodeFold {
                table: table.clone(),
                read_write_address: point(90, FIELD_REGISTERS_LOG_K),
                read_write_cycle: point(100, log_t),
                val_evaluation_address: point(110, FIELD_REGISTERS_LOG_K),
                val_evaluation_cycle: point(120, log_t),
                gammas: crate::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                    &challenges,
                ),
            },
        })
        .unwrap();
        let sumcheck_point = point(130, log_t);
        let input_points = BytecodeReadRafInputClaims {
            address_phase: Vec::new(),
        };
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        let output_values = BytecodeReadRafOutputClaims {
            bytecode_ra: vec![fr(601), fr(602)],
        };
        let clear = relation
            .expected_output(
                &input_points,
                &output_values,
                &output_points,
                &BytecodeReadRafCyclePhaseChallenges {
                    gamma: challenges.gamma,
                },
            )
            .unwrap();

        // The lowered path: the ordinary monolith publics plus the composed FR
        // stage values (the same helper `add_stage6_publics_and_challenges`
        // bakes), folded through the lowered cycle symbolic output expression.
        let r_cycle: Vec<Fr> = sumcheck_point.iter().rev().copied().collect();
        let mut publics = bytecode::read_raf_public_values(BytecodeReadRafEvaluationInputs {
            bytecode: &bytecode,
            r_address: &r_address,
            r_cycle: &r_cycle,
            stage_cycle_points: stage_cycle_points.each_ref().map(Vec::as_slice),
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            entry_bytecode_index,
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        })
        .unwrap();
        let composed = super::field_inline::composed_bytecode_stage_values(
            &table,
            &r_address,
            &r_cycle,
            stage_cycle_points.first().unwrap(),
            &field_read_write_point,
            &field_val_evaluation_point,
            &challenges,
        )
        .unwrap();
        // A vanishing FR contribution would make this parity vacuous.
        assert!(composed.iter().any(|value| *value != fr(0)));
        for (stage_value, field_inline_value) in publics.stage_values.iter_mut().zip(composed) {
            *stage_value += field_inline_value;
        }

        let symbolic = ReadRafCyclePhase::new((dimensions, NUM_BYTECODE_VAL_STAGES));
        let openings = bytecode::read_raf_output_openings(dimensions);
        let lowered = map_expr(symbolic.output_expression::<Fr>()).evaluate(
            |id| match id {
                VerifierOpeningId::Jolt(id) => openings
                    .bytecode_ra
                    .iter()
                    .zip(&output_values.bytecode_ra)
                    .find(|(opening_id, _)| *opening_id == id)
                    .map_or_else(|| fr(0), |(_, value)| *value),
                VerifierOpeningId::FieldInline(_) => fr(0),
            },
            |_| fr(0),
            |id| match id {
                VerifierPublicId::Challenge(id) => {
                    if *id == JoltChallengeId::from(BytecodeReadRafChallenge::Gamma) {
                        challenges.gamma
                    } else {
                        fr(0)
                    }
                }
                VerifierPublicId::Jolt(JoltDerivedId::BytecodeReadRaf(public)) => {
                    publics.value(*public).unwrap_or_else(|| fr(0))
                }
                VerifierPublicId::Jolt(_)
                | VerifierPublicId::SpartanOuter(_)
                | VerifierPublicId::FieldInline(_)
                | VerifierPublicId::FieldInlineChallenge(_) => fr(0),
            },
        );

        assert_eq!(lowered, clear);
    }
}
