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
        input.stage3.challenges.shift_gamma,
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(InstructionInputChallenge::Gamma)),
        input.stage3.challenges.instruction_gamma,
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            RegistersClaimReductionChallenge::Gamma,
        )),
        input.stage3.challenges.registers_gamma,
    )?;

    let shift_point = input
        .stage3
        .batch_consistency
        .try_instance_point(shift.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::SpartanShift, error))?;
    let shift_opening_point = shift_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_plus_one_outer = EqPlusOnePolynomial::new(input.stage2.public.product_tau_low.clone())
        .evaluate(&shift_opening_point);
    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            SpartanProductDimensions::new(log_t)
                .remainder_sumcheck()
                .rounds,
        )
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
        .try_instance_point(instruction_input.spec().rounds)
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
        .try_instance_point(registers_reduction.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersClaimReduction, error))?;
    let registers_opening_point = registers_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan),
        try_eq_mle(
            &registers_opening_point,
            &input.stage2.public.product_tau_low,
        )
        .map_err(|error| public_error(JoltRelationId::RegistersClaimReduction, error))?,
    )?;

    let instruction_outputs = instruction::input_virtualization_output_openings();
    let register_outputs =
        jolt_claims::protocols::jolt::geometry::claim_reductions::registers::claim_reduction_output_openings();
    let output_ids = vec![
        shift_output_openings()[0],
        shift_output_openings()[1],
        shift_output_openings()[2],
        shift_output_openings()[3],
        shift_output_openings()[4],
        instruction_outputs[4],
        instruction_outputs[5],
        instruction_outputs[6],
        instruction_outputs[0],
        instruction_outputs[1],
        instruction_outputs[2],
        instruction_outputs[3],
        register_outputs[0],
    ];
    let aliases = vec![
        OpeningAlias::new(instruction_outputs[7], shift_output_openings()[0]),
        OpeningAlias::new(register_outputs[1], instruction_outputs[5]),
        OpeningAlias::new(register_outputs[2], instruction_outputs[1]),
    ];
    add_batched_stage(
        builder,
        "stage3.batch",
        &[
            shift.spec(),
            instruction_input.spec(),
            registers_reduction.spec(),
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
