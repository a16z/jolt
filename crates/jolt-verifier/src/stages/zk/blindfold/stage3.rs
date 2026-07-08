use super::*;

pub(super) fn add_stage3<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let shift = relations::spartan::Shift::new(dimensions);
    let instruction_input = relations::instruction::InputVirtualization::new(dimensions);
    let registers_reduction =
        relations::claim_reductions::registers::ClaimReduction::new(dimensions);

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(SpartanShiftChallenge::Gamma)),
        input.stage3.challenges.shift.gamma,
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(InstructionInputChallenge::Gamma)),
        input.stage3.challenges.instruction_input.gamma,
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            RegistersClaimReductionChallenge::Gamma,
        )),
        input.stage3.challenges.registers_claim_reduction.gamma,
    )?;

    let shift_point = input
        .stage3
        .batch_consistency
        .try_instance_point(shift.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::SpartanShift, error))?;
    let shift_opening_point = shift_point.iter().rev().copied().collect::<Vec<_>>();
    // Stage 1's remainder cycle point (low half), read from the stage-2 carrier's
    // `product_tau_low`.
    let product_tau_low = input.stage2.product_tau_low.clone();
    let eq_plus_one_outer =
        EqPlusOnePolynomial::new(product_tau_low.clone()).evaluate(&shift_opening_point);
    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(log_t)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_plus_one_product =
        EqPlusOnePolynomial::new(product_opening_point.clone()).evaluate(&shift_opening_point);
    values.public(
        JoltDerivedId::from(SpartanShiftPublic::EqPlusOneOuter),
        eq_plus_one_outer,
    )?;
    values.public(
        JoltDerivedId::from(SpartanShiftPublic::EqPlusOneProduct),
        eq_plus_one_product,
    )?;

    let instruction_point = input
        .stage3
        .batch_consistency
        .try_instance_point(instruction_input.rounds())
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::InstructionInputVirtualization, error)
        })?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        JoltDerivedId::from(InstructionInputPublic::EqProduct),
        try_eq_mle(&instruction_opening_point, &product_opening_point)
            .map_err(|error| public_error(JoltRelationId::InstructionInputVirtualization, error))?,
    )?;

    let registers_point = input
        .stage3
        .batch_consistency
        .try_instance_point(registers_reduction.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersClaimReduction, error))?;
    let registers_opening_point = registers_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan),
        try_eq_mle(&registers_opening_point, &product_tau_low)
            .map_err(|error| public_error(JoltRelationId::RegistersClaimReduction, error))?,
    )?;

    // Single-sourced from the relations' declared alias pairs
    // (`ConcreteSumcheck::aliased_output_openings`): the committed output rows
    // absorb each member's canonical openings minus its aliased ids, and the
    // `OpeningAlias` rows mirror the same `(aliased, source)` pairs — so
    // BlindFold's row layout cannot drift from the clear path's generated absorb
    // and `validate_aliases`.
    let alias_pairs: Vec<_> = <crate::stages::stage3::outputs::InstructionInput<PCS::Field> as
        crate::stages::relations::ConcreteSumcheck<PCS::Field>>::aliased_output_openings()
        .into_iter()
        .chain(<crate::stages::stage3::outputs::RegistersClaimReduction<PCS::Field> as
            crate::stages::relations::ConcreteSumcheck<PCS::Field>>::aliased_output_openings())
        .collect();
    let aliased_targets: std::collections::BTreeSet<_> =
        alias_pairs.iter().map(|(aliased, _)| *aliased).collect();

    let zero = PCS::Field::zero();
    let mut output_ids = relations::spartan::SpartanShiftOutputClaims::<PCS::Field> {
        unexpanded_pc: zero,
        pc: zero,
        is_virtual: zero,
        is_first_in_sequence: zero,
        is_noop: zero,
    }
    .canonical_order();
    output_ids.extend(
        relations::instruction::InstructionInputOutputClaims::<PCS::Field> {
            left_operand_is_rs1: zero,
            rs1_value: zero,
            left_operand_is_pc: zero,
            unexpanded_pc: zero,
            right_operand_is_rs2: zero,
            rs2_value: zero,
            right_operand_is_imm: zero,
            imm: zero,
        }
        .canonical_order()
        .into_iter()
        .filter(|id| !aliased_targets.contains(id)),
    );
    output_ids.extend(
        relations::claim_reductions::registers::RegistersClaimReductionOutputClaims::<PCS::Field> {
            rd_write_value: zero,
            rs1_value: zero,
            rs2_value: zero,
        }
        .canonical_order()
        .into_iter()
        .filter(|id| !aliased_targets.contains(id)),
    );
    let aliases = alias_pairs
        .into_iter()
        .map(|(aliased, source)| OpeningAlias::new(aliased, source))
        .collect::<Vec<_>>();
    add_batched_stage(
        builder,
        "stage3.batch",
        shift.domain(),
        &[
            relation_claim(&shift),
            relation_claim(&instruction_input),
            relation_claim(&registers_reduction),
        ],
        &input.stage3.batch_consistency,
        &input.stage3.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}
