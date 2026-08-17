use super::*;

pub(super) fn add_stage6a<PCS, VC, ZkProof>(
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
    let bytecode_address_claims =
        relations::bytecode::ReadRafAddressPhase::new(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_address_claims =
        relations::booleanity::BooleanityAddressPhase::new(booleanity_dimensions);
    let booleanity_claims = relations::booleanity::BooleanityCyclePhase::new(booleanity_dimensions);
    let ram_hamming_claims = relations::ram::HammingBooleanity::new(trace_dimensions);
    let ram_ra_claims =
        relations::ram::RaVirtualization::new(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = relations::instruction::RaVirtualization::new(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = relations::claim_reductions::increments::ClaimReduction::new(trace_dimensions);
    let trusted_layout = input.checked.precommitted.advice(JoltAdviceKind::Trusted);
    let untrusted_layout = input.checked.precommitted.advice(JoltAdviceKind::Untrusted);

    // The cycle bytecode round count is needed by the shared publics helper; the
    // committed and uncommitted cycle-phase relations are distinct types, so pick
    // the active one's rounds here.
    let bytecode_rounds = if bytecode_reduction_layout.is_some() {
        relations::bytecode::ReadRafCyclePhaseCommitted::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ))
        .rounds()
    } else {
        relations::bytecode::ReadRafCyclePhase::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ))
        .rounds()
    };

    add_stage6_publics_and_challenges(
        input,
        values,
        bytecode_address_claims.rounds(),
        bytecode_rounds,
        booleanity_address_claims.rounds(),
        booleanity_claims.rounds(),
        ram_hamming_claims.rounds(),
        ram_ra_claims.rounds(),
        instruction_ra_claims.rounds(),
        inc_claims.rounds(),
    )?;
    if let Some(layout) = trusted_layout {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted)?;
    }
    if let Some(layout) = untrusted_layout {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted)?;
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        let eta = input
            .stage6b
            .challenges
            .bytecode_reduction_eta
            .ok_or_else(|| VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta).into(),
            })?;
        values.public(
            VerifierPublicId::Challenge(JoltChallengeId::from(
                BytecodeClaimReductionChallenge::Eta,
            )),
            eta,
        )?;
        add_bytecode_reduction_cycle_publics(input, values, layout)?;
    }
    if let Some(layout) = program_image_reduction_layout.as_ref() {
        add_program_image_reduction_cycle_publics(input, values, layout)?;
    }

    let mut address_phase_output_ids: Vec<VerifierOpeningId> =
        vec![bytecode::bytecode_read_raf_address_phase_opening().into()];
    if bytecode_reduction_layout.is_some() {
        address_phase_output_ids.extend(
            (0..bytecode_reduction::NUM_BYTECODE_VAL_STAGES)
                .map(bytecode_reduction::bytecode_val_stage_opening)
                .map(VerifierOpeningId::from),
        );
    }
    address_phase_output_ids.push(booleanity::booleanity_address_phase_opening().into());

    // The composed bytecode address-phase input claim: the ordinary symbolic
    // gamma-folded bind plus (under `field-inline`) the FR terms at the
    // extended stage-1/4/5 power indices — the clear composed `input_claim`
    // override's algebra, over the stage-1 FR carrier rows and the stage-4/5
    // FR members' rows (referencing the SAME committed rows those stages
    // lowered).
    let bytecode_claim = relation_claim(&bytecode_address_claims);
    #[cfg(feature = "field-inline")]
    let bytecode_claim = {
        let (rounds, input_expr, output_expr) = bytecode_claim;
        (
            rounds,
            input_expr + field_inline_bytecode_input_extension_expr::<PCS::Field>(),
            output_expr,
        )
    };

    add_batched_stage(
        builder,
        "stage6.address_phase",
        bytecode_address_claims.domain(),
        &[bytecode_claim, relation_claim(&booleanity_address_claims)],
        &input.stage6a.consistency,
        &input.stage6a.output_claims,
        values,
        address_phase_output_ids,
        Vec::new(),
        Vec::new(),
    )
}

/// The FR terms the composed bytecode read-RAF address-phase input claim adds
/// to the ordinary gamma-folded bind: the eight `FieldOpFlag` openings (the
/// stage-1 FR carrier rows) at the extended Stage1Gamma powers, the stage-4 FR
/// member's `FieldRdWa`/`FieldRs1Ra`/`FieldRs2Ra` rows at the extended
/// Stage4Gamma powers under the outer γ³, and the stage-5 FR member's
/// `FieldRdWa` row at the extended Stage5Gamma power under the outer γ⁴ — each
/// stage extension riding the same outer gamma power as its ordinary stage
/// claim, with no new challenge draws (spec: `field-inline-protocol.md`,
/// "Stage 6 Composition"). Value-parity with the clear composed `input_claim`
/// override is pinned by `lowered_bytecode_input_extension_matches_the_clear_composed_claim`.
#[cfg(feature = "field-inline")]
fn field_inline_bytecode_input_extension_expr<F: Field>() -> VerifierExpr<F> {
    use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;
    use jolt_claims::protocols::field_inline::geometry::spartan::outer_opening;
    use jolt_claims::protocols::field_inline::{
        FieldInlineRelationId, FieldInlineVirtualPolynomial,
    };
    use jolt_claims::protocols::jolt::BytecodeReadRafChallenge;
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_riscv::NUM_CIRCUIT_FLAGS;

    let gamma_public = |challenge: BytecodeReadRafChallenge| -> VerifierExpr<F> {
        derived(VerifierPublicId::Challenge(JoltChallengeId::from(
            challenge,
        )))
    };
    let gamma = gamma_public(BytecodeReadRafChallenge::Gamma);
    let stage1_gamma = gamma_public(BytecodeReadRafChallenge::Stage1Gamma);
    let stage4_gamma = gamma_public(BytecodeReadRafChallenge::Stage4Gamma);
    let stage5_gamma = gamma_public(BytecodeReadRafChallenge::Stage5Gamma);

    // Stage-1 extension: the eight FieldOpFlag rows at powers
    // `stage1_gamma^(2 + NUM_CIRCUIT_FLAGS + i)` (the ordinary stage-1 power
    // count is `2 + NUM_CIRCUIT_FLAGS`), riding the outer γ⁰.
    let mut extension = VerifierExpr::zero();
    for (index, flag) in FIELD_INLINE_BYTECODE_STAGE1_FLAGS.into_iter().enumerate() {
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "2 + NUM_CIRCUIT_FLAGS + index is a small constant sum over the eight FR flags"
        )]
        let power = 2 + NUM_CIRCUIT_FLAGS + index;
        extension = extension
            + stage1_gamma.clone().pow(power)
                * opening(outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
                    flag,
                )));
    }

    // Stage-4 extension: FieldRdWa/FieldRs1Ra/FieldRs2Ra at powers
    // `stage4_gamma^(3 + j)` (the ordinary stage-4 power count is 3), riding
    // the outer γ³.
    let stage4_rows = [
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineVirtualPolynomial::FieldRs1Ra,
        FieldInlineVirtualPolynomial::FieldRs2Ra,
    ];
    let mut stage4_extension = VerifierExpr::zero();
    for (index, polynomial) in stage4_rows.into_iter().enumerate() {
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "3 + index is a small constant sum over the three FR access rows"
        )]
        let power = 3 + index;
        stage4_extension = stage4_extension
            + stage4_gamma.clone().pow(power)
                * opening(
                    jolt_claims::protocols::field_inline::FieldInlineOpeningId::virtual_polynomial(
                        polynomial,
                        FieldInlineRelationId::FieldRegistersReadWriteChecking,
                    ),
                );
    }
    extension = extension + gamma.clone().pow(3) * stage4_extension;

    // Stage-5 extension: the val-evaluation FieldRdWa at the power following
    // the ordinary stage-5 count (`2 + lookup-table count`), riding the outer
    // γ⁴.
    let stage5_power = 2 + LookupTableKind::<RISCV_XLEN>::COUNT;
    extension
        + gamma.pow(4)
            * stage5_gamma.pow(stage5_power)
            * opening(
                jolt_claims::protocols::field_inline::FieldInlineOpeningId::virtual_polynomial(
                    FieldInlineVirtualPolynomial::FieldRdWa,
                    FieldInlineRelationId::FieldRegistersValEvaluation,
                ),
            )
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod field_inline_tests {
    use super::*;
    use crate::stages::relations::ConcreteSumcheck as _;
    use crate::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
        FieldInlineBytecodeReadRafInputs,
    };
    use jolt_claims::protocols::field_inline::geometry::spartan::outer_opening;
    use jolt_claims::protocols::field_inline::{
        FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
    };
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::{InputClaims as _, SumcheckChallenges as _};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The lowered composed input expression — the jolt symbolic bind plus the
    /// FR gamma-power extension — evaluates identically to the clear composed
    /// `BytecodeReadRafAddressPhase::input_claim` on synthetic values, over
    /// the SAME committed rows the stage-1/4/5 lowerings bind (the stage-1 FR
    /// carrier flags and the stage-4/5 FR member rows).
    #[test]
    fn lowered_bytecode_input_extension_matches_the_clear_composed_claim() {
        let relation = BytecodeReadRafAddressPhase::<Fr>::new(
            BytecodeReadRafDimensions::new(3, 4, 2),
            false,
            BytecodeStagePoints {
                stage_cycle_points: Default::default(),
                register_read_write_point: Vec::new(),
                register_val_evaluation_point: Vec::new(),
            },
            0,
        );
        let mut inputs = BytecodeReadRafAddressPhaseInputClaims::<Fr> {
            lookup_table_flags: vec![fr(0); LookupTableKind::<RISCV_XLEN>::COUNT],
            ..Default::default()
        };
        inputs.outer_unexpanded_pc = fr(3);
        inputs.outer_imm = fr(5);
        inputs.outer_jump = fr(7);
        inputs.product_branch = fr(11);
        inputs.instruction_input_imm = fr(13);
        inputs.rd_wa_read_write = fr(17);
        inputs.rs1_ra = fr(19);
        inputs.rs2_ra = fr(23);
        inputs.rd_wa_val_evaluation = fr(29);
        inputs.instruction_raf_flag = fr(31);
        for (index, flag) in inputs.lookup_table_flags.iter_mut().enumerate() {
            *flag = fr(100 + index as u64);
        }
        let field_inline = FieldInlineBytecodeReadRafInputs::<Fr> {
            field_op_flags: core::array::from_fn(|index| fr(200 + index as u64)),
            rd_wa_read_write: fr(301),
            rs1_ra: fr(302),
            rs2_ra: fr(303),
            rd_wa_val_evaluation: fr(304),
        };
        relation
            .set_field_inline_inputs(field_inline.clone())
            .unwrap();
        let challenges = BytecodeReadRafAddressPhaseChallenges::<Fr> {
            gamma: fr(401),
            stage1_gamma: fr(402),
            stage2_gamma: fr(403),
            stage3_gamma: fr(404),
            stage4_gamma: fr(405),
            stage5_gamma: fr(406),
        };
        let clear = relation.input_claim(&inputs, &challenges).unwrap();

        let lowered_expr = map_expr(relation.symbolic().input_expression::<Fr>())
            + field_inline_bytecode_input_extension_expr::<Fr>();
        let resolve_field_inline = |id: &FieldInlineOpeningId| -> Fr {
            use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;

            for (flag, value) in FIELD_INLINE_BYTECODE_STAGE1_FLAGS
                .into_iter()
                .zip(field_inline.field_op_flags)
            {
                if *id == outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(flag)) {
                    return value;
                }
            }
            let read_write = |polynomial| {
                FieldInlineOpeningId::virtual_polynomial(
                    polynomial,
                    FieldInlineRelationId::FieldRegistersReadWriteChecking,
                )
            };
            if *id == read_write(FieldInlineVirtualPolynomial::FieldRdWa) {
                field_inline.rd_wa_read_write
            } else if *id == read_write(FieldInlineVirtualPolynomial::FieldRs1Ra) {
                field_inline.rs1_ra
            } else if *id == read_write(FieldInlineVirtualPolynomial::FieldRs2Ra) {
                field_inline.rs2_ra
            } else if *id
                == FieldInlineOpeningId::virtual_polynomial(
                    FieldInlineVirtualPolynomial::FieldRdWa,
                    FieldInlineRelationId::FieldRegistersValEvaluation,
                )
            {
                field_inline.rd_wa_val_evaluation
            } else {
                fr(0)
            }
        };
        let lowered = lowered_expr.evaluate(
            |id| match id {
                VerifierOpeningId::Jolt(id) => inputs.resolve_input(id).unwrap_or_else(|| fr(0)),
                VerifierOpeningId::FieldInline(id) => resolve_field_inline(id),
            },
            |_| fr(0),
            |id| match id {
                VerifierPublicId::Challenge(id) => {
                    challenges.resolve_challenge(id).unwrap_or_else(|| fr(0))
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
