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
    // Stage 1's remainder cycle point, recomputed from `stage1.remainder_consistency`
    // rather than read off the stage-2 carrier's `product_tau_low`, so the
    // BakedPublicInputs derivation stays independent of the carrier field.
    let product_tau_low = stage1_remainder_cycle(input);
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

    let output_ids = vec![
        spartan::unexpanded_pc_shift(),
        spartan::pc_shift(),
        spartan::is_virtual_shift(),
        spartan::is_first_in_sequence_shift(),
        spartan::is_noop_shift(),
        instruction::left_operand_is_rs1(),
        instruction::rs1_value(),
        instruction::left_operand_is_pc(),
        instruction::right_operand_is_rs2(),
        instruction::rs2_value(),
        instruction::right_operand_is_imm(),
        instruction::imm(),
        registers_claim_reduction::rd_write_value_reduced(),
    ];
    let aliases = vec![
        OpeningAlias::new(instruction::unexpanded_pc(), spartan::unexpanded_pc_shift()),
        OpeningAlias::new(
            registers_claim_reduction::rs1_value_reduced(),
            instruction::rs1_value(),
        ),
        OpeningAlias::new(
            registers_claim_reduction::rs2_value_reduced(),
            instruction::rs2_value(),
        ),
    ];
    add_batched_stage(
        builder,
        "stage3.batch",
        shift.domain(),
        &[
            shift.rounds(),
            instruction_input.rounds(),
            registers_reduction.rounds(),
        ],
        &[
            shift.input_expression::<PCS::Field>(),
            instruction_input.input_expression::<PCS::Field>(),
            registers_reduction.input_expression::<PCS::Field>(),
        ],
        &[
            shift.output_expression::<PCS::Field>(),
            instruction_input.output_expression::<PCS::Field>(),
            registers_reduction.output_expression::<PCS::Field>(),
        ],
        &input.stage3.batch_consistency,
        &input.stage3.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}
