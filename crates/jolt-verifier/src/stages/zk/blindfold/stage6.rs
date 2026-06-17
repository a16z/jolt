use super::*;

pub(super) fn add_stage6<PCS, VC, ZkProof, PcsAssist>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof, PcsAssist>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
    PcsAssist: PcsProofAssist<PCS>,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let bytecode_claims = bytecode::read_raf::<PCS::Field>(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_claims = booleanity::booleanity::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_inc_claims =
        field_increments::claim_reduction::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));
    let (trusted_layout, trusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Untrusted);

    add_stage6_publics_and_challenges(
        input,
        values,
        &bytecode_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    )?;
    #[cfg(feature = "field-inline")]
    {
        values.challenge(
            FieldInlineChallengeId::from(FieldRegistersIncClaimReductionChallenge::Gamma),
            input.stage6.public.field_inline.field_inc_gamma,
        )?;
        let field_inc_point = input
            .stage6
            .batch_consistency
            .try_instance_point(field_inc_claims.sumcheck.rounds)
            .map_err(|error| stage_sumcheck_error(JoltRelationId::IncClaimReduction, error))?;
        let field_inc_opening_point = trace_dimensions
            .cycle_opening_point(&field_inc_point)
            .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?;
        let field_log_k = input.proof.protocol.field_inline.field_register_log_k;
        let field_read_write_cycle = input
            .stage4
            .field_registers_read_write_opening_point
            .get(field_log_k..)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register read-write opening point is shorter than the field-register address: expected at least {field_log_k}, got {}",
                    input.stage4.field_registers_read_write_opening_point.len()
                ),
            })?;
        let field_val_evaluation_cycle = input
            .stage5
            .field_inline
            .field_registers_val_evaluation
            .opening_point
            .get(field_log_k..)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register val-evaluation opening point is shorter than the field-register address: expected at least {field_log_k}, got {}",
                    input.stage5
                        .field_inline
                        .field_registers_val_evaluation
                        .opening_point
                        .len()
                ),
            })?;
        values.public(
            FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqReadWrite),
            try_eq_mle(&field_inc_opening_point, field_read_write_cycle)
                .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
        )?;
        values.public(
            FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqValEvaluation),
            try_eq_mle(&field_inc_opening_point, field_val_evaluation_cycle)
                .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
        )?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        trusted_layout.as_ref(),
        trusted_claims.as_ref(),
        input.stage6.trusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        untrusted_layout.as_ref(),
        untrusted_claims.as_ref(),
        input.stage6.untrusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted, public)?;
    }

    let bytecode_input_expr = map_jolt_expr(bytecode_claims.input.expression().clone());
    #[cfg(feature = "field-inline")]
    let bytecode_input_expr = bytecode_input_expr
        + map_field_inline_bytecode_expr(field_bytecode::read_raf_input_extension::<PCS::Field>());

    let mut batch_claims = vec![
        (
            bytecode_claims.sumcheck.rounds,
            bytecode_input_expr,
            map_jolt_expr(bytecode_claims.output.expression().clone()),
        ),
        (
            booleanity_claims.sumcheck.rounds,
            map_jolt_expr(booleanity_claims.input.expression().clone()),
            map_jolt_expr(booleanity_claims.output.expression().clone()),
        ),
        (
            ram_hamming_claims.sumcheck.rounds,
            map_jolt_expr(ram_hamming_claims.input.expression().clone()),
            map_jolt_expr(ram_hamming_claims.output.expression().clone()),
        ),
        (
            ram_ra_claims.sumcheck.rounds,
            map_jolt_expr(ram_ra_claims.input.expression().clone()),
            map_jolt_expr(ram_ra_claims.output.expression().clone()),
        ),
        (
            instruction_ra_claims.sumcheck.rounds,
            map_jolt_expr(instruction_ra_claims.input.expression().clone()),
            map_jolt_expr(instruction_ra_claims.output.expression().clone()),
        ),
        (
            inc_claims.sumcheck.rounds,
            map_jolt_expr(inc_claims.input.expression().clone()),
            map_jolt_expr(inc_claims.output.expression().clone()),
        ),
    ];
    #[cfg(feature = "field-inline")]
    batch_claims.push((
        field_inc_claims.sumcheck.rounds,
        map_field_inline_expr(field_inc_claims.input.expression().clone()),
        map_field_inline_expr(field_inc_claims.output.expression().clone()),
    ));
    if let Some(claim) = trusted_claims {
        batch_claims.push((
            claim.sumcheck.rounds,
            map_jolt_expr(claim.input.expression().clone()),
            map_jolt_expr(claim.output.expression().clone()),
        ));
    }
    if let Some(claim) = untrusted_claims {
        batch_claims.push((
            claim.sumcheck.rounds,
            map_jolt_expr(claim.input.expression().clone()),
            map_jolt_expr(claim.output.expression().clone()),
        ));
    }

    let mut output_ids = Vec::new();
    output_ids.extend(map_jolt_opening_ids(
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf).bytecode_ra,
    ));
    output_ids.extend(map_jolt_opening_ids(
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout),
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::hamming_booleanity_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization),
    ));
    output_ids.extend(map_jolt_opening_ids(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        increments::claim_reduction_output_openings().to_vec(),
    ));
    #[cfg(feature = "field-inline")]
    output_ids.extend(map_field_inline_opening_ids(
        field_increments::claim_reduction_output_openings().to_vec(),
    ));
    if let Some(layout) = trusted_layout {
        output_ids.extend(map_jolt_opening_ids(advice::cycle_phase_output_openings(
            JoltAdviceKind::Trusted,
            layout.dimensions(),
        )));
    }
    if let Some(layout) = untrusted_layout {
        output_ids.extend(map_jolt_opening_ids(advice::cycle_phase_output_openings(
            JoltAdviceKind::Untrusted,
            layout.dimensions(),
        )));
    }

    let coefficients = &input.stage6.batch_consistency.batching_coefficients;
    if batch_claims.len() != coefficients.len() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage6.batch: expected {} batching coefficients, got {}",
                batch_claims.len(),
                coefficients.len()
            ),
        });
    }
    let input_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((rounds, input_expr, _), coefficient)| {
            let scale = *coefficient
                * PCS::Field::pow2(input.stage6.batch_consistency.max_num_vars - *rounds);
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
        "stage6.batch",
        SumcheckStatement::new(
            input.stage6.batch_consistency.max_num_vars,
            input.stage6.batch_consistency.max_degree,
        ),
        domain_spec(bytecode_claims.sumcheck),
        input.stage6.batch_consistency.consistency.clone(),
        &input.stage6.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
        input_claim,
        output_claim,
    )
}
