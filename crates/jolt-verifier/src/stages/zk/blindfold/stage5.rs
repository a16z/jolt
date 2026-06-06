use super::*;

pub(super) fn add_stage5<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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
        instruction::read_raf::<PCS::Field>(formula_dimensions.instruction_read_raf);
    let ram_claims = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let registers_claims = registers::val_evaluation::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_claims =
        field_registers::val_evaluation::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));

    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::Gamma),
        input.stage5.public.instruction_gamma,
    )?;
    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let instruction_point = input
        .stage5
        .batch_consistency
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionReadRaf, error))?;
    let instruction_opening = formula_dimensions
        .instruction_read_raf
        .opening_point(&instruction_point)
        .map_err(|error| public_error(JoltRelationId::InstructionReadRaf, error))?;
    let stage2_instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(2)
                .rounds,
        )
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
    let instruction_gamma_squared =
        input.stage5.public.instruction_gamma * input.stage5.public.instruction_gamma;
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        values.challenge(
            JoltChallengeId::from(InstructionReadRafChallenge::EqTableValue(table.index())),
            eq_reduction
                * table.evaluate_mle::<PCS::Field, PCS::Field>(&instruction_opening.r_address),
        )?;
    }
    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::EqRafConstant),
        eq_reduction
            * (input.stage5.public.instruction_gamma * left_operand_eval
                + instruction_gamma_squared * right_operand_eval),
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::EqRafFlag),
        eq_reduction
            * (instruction_gamma_squared * identity_eval
                - input.stage5.public.instruction_gamma * left_operand_eval
                - instruction_gamma_squared * right_operand_eval),
    )?;

    values.challenge(
        JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma),
        input.stage5.public.ram_gamma,
    )?;
    let ram_point = input
        .stage5
        .batch_consistency
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRaClaimReduction, error))?;
    let ram_cycle = trace_dimensions
        .cycle_opening_point(&ram_point)
        .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?;
    let ram_raf_cycle = &input
        .stage2
        .ram_ra_claim_reduction_inputs
        .ram_raf_evaluation_opening_point[log_k..];
    let ram_read_write_cycle = &input
        .stage2
        .ram_ra_claim_reduction_inputs
        .ram_read_write_opening_point[log_k..];
    let ram_val_cycle = &input.stage4.ram_val_check_opening_point[log_k..];
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleRaf),
        try_eq_mle(&ram_cycle, ram_raf_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
        try_eq_mle(&ram_cycle, ram_read_write_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleValCheck),
        try_eq_mle(&ram_cycle, ram_val_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaClaimReduction, error))?,
    )?;

    let registers_point = input
        .stage5
        .batch_consistency
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersValEvaluation, error))?;
    let registers_cycle = trace_dimensions
        .cycle_opening_point(&registers_point)
        .map_err(|error| public_error(JoltRelationId::RegistersValEvaluation, error))?;
    let registers_read_write_cycle =
        &input.stage4.registers_read_write_opening_point[REGISTER_ADDRESS_BITS..];
    values.challenge(
        JoltChallengeId::from(RegistersValEvaluationChallenge::LtCycle),
        LtPolynomial::evaluate(&registers_cycle, registers_read_write_cycle),
    )?;

    #[cfg(feature = "field-inline")]
    {
        let field_registers_point = input
            .stage5
            .batch_consistency
            .try_instance_point(field_registers_claims.sumcheck.rounds)
            .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersValEvaluation, error))?;
        let field_registers_cycle = trace_dimensions
            .cycle_opening_point(&field_registers_point)
            .map_err(|error| public_error(JoltRelationId::RegistersValEvaluation, error))?;
        let field_log_k = input.context.protocol.field_inline.field_register_log_k;
        let field_registers_read_write_cycle = input
            .stage4
            .field_registers_read_write_opening_point
            .get(field_log_k..)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: format!(
                    "field-register read-write opening point is shorter than the field-register address: expected at least {field_log_k}, got {}",
                    input.stage4.field_registers_read_write_opening_point.len()
                ),
            })?;
        values.challenge(
            FieldInlineChallengeId::from(FieldRegistersValEvaluationChallenge::LtCycle),
            LtPolynomial::evaluate(&field_registers_cycle, field_registers_read_write_cycle),
        )?;
    }

    let mut output_ids = Vec::new();
    output_ids.extend(map_jolt_opening_ids(
        instruction_output_openings.lookup_table_flags,
    ));
    output_ids.extend(map_jolt_opening_ids(
        instruction_output_openings.instruction_ra,
    ));
    output_ids.push(VerifierOpeningId::Jolt(
        instruction_output_openings.instruction_raf_flag,
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::ra_claim_reduction_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        registers::val_evaluation_output_openings().to_vec(),
    ));
    #[cfg(feature = "field-inline")]
    output_ids.extend(map_field_inline_opening_ids(
        field_registers::val_evaluation_output_openings().to_vec(),
    ));

    let batch_claims = [
        (
            instruction_claims.sumcheck.rounds,
            map_jolt_expr(instruction_claims.input.expression().clone()),
            map_jolt_expr(instruction_claims.output.expression().clone()),
        ),
        (
            ram_claims.sumcheck.rounds,
            map_jolt_expr(ram_claims.input.expression().clone()),
            map_jolt_expr(ram_claims.output.expression().clone()),
        ),
        (
            registers_claims.sumcheck.rounds,
            map_jolt_expr(registers_claims.input.expression().clone()),
            map_jolt_expr(registers_claims.output.expression().clone()),
        ),
    ];
    #[cfg(feature = "field-inline")]
    let mut batch_claims = batch_claims.to_vec();
    #[cfg(feature = "field-inline")]
    batch_claims.push((
        field_registers_claims.sumcheck.rounds,
        map_field_inline_expr(field_registers_claims.input.expression().clone()),
        map_field_inline_expr(field_registers_claims.output.expression().clone()),
    ));

    let coefficients = &input.stage5.batch_consistency.batching_coefficients;
    if batch_claims.len() != coefficients.len() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage5.batch: expected {} batching coefficients, got {}",
                batch_claims.len(),
                coefficients.len()
            ),
        });
    }
    let input_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((rounds, input_expr, _), coefficient)| {
            let scale = *coefficient
                * PCS::Field::pow2(input.stage5.batch_consistency.max_num_vars - *rounds);
            acc + scale_expr(input_expr.clone(), scale)
        },
    );
    let output_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((_, _, output_expr), coefficient)| {
            acc + scale_expr(output_expr.clone(), *coefficient)
        },
    );

    add_stage(
        builder,
        "stage5.batch",
        SumcheckStatement::new(
            input.stage5.batch_consistency.max_num_vars,
            input.stage5.batch_consistency.max_degree,
        ),
        domain_spec(instruction_claims.sumcheck),
        input.stage5.batch_consistency.consistency.clone(),
        &input.stage5.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
        input_claim,
        output_claim,
    )
}
