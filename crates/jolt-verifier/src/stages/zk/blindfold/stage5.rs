use super::*;

pub(super) fn add_stage5<PCS, VC, ZkProof>(
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
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let instruction_claims =
        relations::instruction::ReadRaf::new(formula_dimensions.instruction_read_raf);
    let ram_claims = relations::ram::RaClaimReduction::new(trace_dimensions);
    let registers_claims = relations::registers::ValEvaluation::new(trace_dimensions);

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(InstructionReadRafChallenge::Gamma)),
        input.stage5.challenges.instruction_read_raf.gamma,
    )?;
    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let instruction_point = input
        .stage5
        .batch_consistency
        .try_instance_point(instruction_claims.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionReadRaf, error))?;
    let instruction_opening = formula_dimensions
        .instruction_read_raf
        .opening_point(&instruction_point)
        .map_err(|error| public_error(JoltRelationId::InstructionReadRaf, error))?;
    let stage2_instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(log_t)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionClaimReduction, error))?;
    let stage2_instruction_opening = stage2_instruction_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let eq_reduction = try_eq_mle(&stage2_instruction_opening, &instruction_opening.r_cycle)
        .map_err(|error| public_error(JoltRelationId::InstructionReadRaf, error))?;
    let left_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Left)
        .evaluate(&instruction_opening.r_address);
    let right_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Right)
        .evaluate(&instruction_opening.r_address);
    let identity_eval =
        IdentityPolynomial::new(2 * RISCV_XLEN).evaluate(&instruction_opening.r_address);
    let instruction_gamma_squared = input.stage5.challenges.instruction_read_raf.gamma
        * input.stage5.challenges.instruction_read_raf.gamma;
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        values.public(
            JoltDerivedId::from(InstructionReadRafPublic::EqTableValue(table.index())),
            eq_reduction
                * table.evaluate_mle::<PCS::Field, PCS::Field>(&instruction_opening.r_address),
        )?;
    }
    values.public(
        JoltDerivedId::from(InstructionReadRafPublic::EqRafConstant),
        eq_reduction
            * (input.stage5.challenges.instruction_read_raf.gamma * left_operand_eval
                + instruction_gamma_squared * right_operand_eval),
    )?;
    values.public(
        JoltDerivedId::from(InstructionReadRafPublic::EqRafFlag),
        eq_reduction
            * (instruction_gamma_squared * identity_eval
                - input.stage5.challenges.instruction_read_raf.gamma * left_operand_eval
                - instruction_gamma_squared * right_operand_eval),
    )?;

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)),
        input.stage5.challenges.ram_ra_claim_reduction.gamma,
    )?;
    let ram_point = input
        .stage5
        .batch_consistency
        .try_instance_point(ram_claims.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRaClaimReduction, error))?;
    let ram_cycle = trace_dimensions
        .cycle_opening_point(&ram_point)
        .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?;
    let ram_raf_cycle = &input.stage2.output_points.ram_raf_evaluation_point()[log_k..];
    let ram_read_write_cycle = &input.stage2.output_points.ram_read_write_point()[log_k..];
    let ram_val_cycle = &input.stage4.output_points.ram_val_check_point()[log_k..];
    values.public(
        JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleRaf),
        try_eq_mle(&ram_cycle, ram_raf_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
        try_eq_mle(&ram_cycle, ram_read_write_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
        try_eq_mle(&ram_cycle, ram_val_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;

    let registers_point = input
        .stage5
        .batch_consistency
        .try_instance_point(registers_claims.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersValEvaluation, error))?;
    let registers_cycle = trace_dimensions
        .cycle_opening_point(&registers_point)
        .map_err(|error| public_error(JoltRelationId::RegistersValEvaluation, error))?;
    let registers_read_write_cycle =
        &input.stage4.output_points.registers_read_write_point()[REGISTER_ADDRESS_BITS..];
    values.public(
        JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
        LtPolynomial::evaluate(&registers_cycle, registers_read_write_cycle),
    )?;

    let mut output_ids = Vec::new();
    output_ids.extend(instruction_output_openings.lookup_table_flags);
    output_ids.extend(instruction_output_openings.instruction_ra);
    output_ids.push(instruction_output_openings.instruction_raf_flag);
    output_ids.extend(
        relations::ram::RamRaClaimReductionOutputClaims::<PCS::Field> {
            ram_ra: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    output_ids.extend(
        relations::registers::RegistersValEvaluationOutputClaims::<PCS::Field> {
            rd_inc: PCS::Field::zero(),
            rd_wa: PCS::Field::zero(),
        }
        .canonical_order(),
    );

    let batch_claims = [
        relation_claim(&instruction_claims),
        relation_claim(&ram_claims),
        relation_claim(&registers_claims),
    ];

    add_batched_stage(
        builder,
        "stage5.batch",
        instruction_claims.domain(),
        &batch_claims,
        &input.stage5.batch_consistency,
        &input.stage5.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}
