use super::*;

pub(super) fn add_stage2<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<PCS::Field, VC::Output>,
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
    let read_write_dimensions = input.proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        ram::RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let product_uniskip = JoltSumcheckSpec::centered_integer(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        1,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
    );
    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.public.product_tau_high,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: error.to_string(),
    })?;
    let product_uniskip_input =
        selected_product_uniskip_input_expr::<PCS::Field>(&product_weights)?;
    builder = add_stage(
        builder,
        "stage2.product_uniskip",
        SumcheckStatement::new(product_uniskip.rounds, product_uniskip.degree),
        domain_spec(product_uniskip),
        input.stage2.product_uniskip_consistency.clone(),
        &input.stage2.product_uniskip_output_claims,
        values,
        vec![VerifierOpeningId::Jolt(product_uniskip_opening())],
        Vec::new(),
        product_uniskip_input,
        opening(VerifierOpeningId::Jolt(product_uniskip_opening())),
    )?;

    let ram_read_write = relations::ram::ReadWriteChecking::new(read_write_dimensions);
    let product_remainder = relations::spartan::ProductRemainder::new(product_dimensions);
    let instruction_reduction =
        relations::claim_reductions::instruction::ClaimReduction::new(trace_dimensions);
    let ram_raf = relations::ram::RafEvaluation::new(raf_dimensions);
    let ram_output = relations::ram::OutputCheck::new(read_write_dimensions);

    let ram_read_write_point = input
        .stage2
        .batch_consistency
        .try_instance_point(ram_read_write.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamReadWriteChecking, error))?;
    let ram_read_write_opening = read_write_dimensions
        .read_write_opening_point(&ram_read_write_point)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    let eq_cycle = try_eq_mle(
        &input.stage2.public.product_tau_low,
        &ram_read_write_opening.r_cycle,
    )
    .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
        input.stage2.public.ram_read_write_gamma,
    )?;
    values.public(JoltPublicId::from(RamReadWritePublic::EqCycle), eq_cycle)?;

    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(product_remainder.spec().rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let product_tau_high_bound = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.public.product_tau_high,
        input.stage2.public.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_low_eq =
        try_eq_mle(&input.stage2.public.product_tau_low, &product_opening_point)
            .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_lagrange_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.public.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_kernel = product_tau_high_bound * product_tau_low_eq;
    let product_remainder_output = selected_product_remainder_output_expr::<PCS::Field>(
        &product_lagrange_weights,
        product_tau_kernel,
    )?;

    let instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(instruction_reduction.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionClaimReduction, error))?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(
        &instruction_opening_point,
        &input.stage2.public.product_tau_low,
    )
    .map_err(|error| public_error(JoltRelationId::InstructionClaimReduction, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            InstructionClaimReductionChallenge::Gamma,
        )),
        input.stage2.public.instruction_gamma,
    )?;
    values.public(
        JoltPublicId::from(InstructionClaimReductionPublic::EqSpartan),
        eq_spartan,
    )?;

    let active_stage2_rounds = log_t + log_k;
    let phase1_offset = input
        .stage2
        .batch_consistency
        .try_round_offset(active_stage2_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?
        + read_write_dimensions.phase1_num_rounds();
    let ram_raf_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_raf.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_address = read_write_dimensions
        .address_opening_point(&ram_raf_point)
        .map_err(|error| public_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_unmap_address = IdentityPolynomial::new(log_k).evaluate(&ram_raf_address)
        * PCS::Field::from_u64(8)
        + PCS::Field::from_u64(input.checked.public_io.memory_layout.get_lowest_address());
    values.public(
        JoltPublicId::from(RamRafEvaluationPublic::UnmapAddress),
        ram_raf_unmap_address,
    )?;

    let ram_output_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_output.spec().rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamOutputCheck, error))?;
    let ram_output_address = read_write_dimensions
        .address_opening_point(&ram_output_point)
        .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let output_publics = ram_output_publics(
        input,
        &input.stage2.public.output_address_challenges,
        &ram_output_address,
    )?;
    values.public(
        JoltPublicId::from(RamOutputCheckPublic::EqIoMask),
        output_publics.0,
    )?;
    values.public(
        JoltPublicId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
        output_publics.1,
    )?;

    let mut output_ids = Vec::new();
    output_ids.extend(map_jolt_opening_ids(
        ram::read_write_checking_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        product_remainder_output_openings().to_vec(),
    ));
    let instruction_outputs =
        jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::claim_reduction_output_openings();
    output_ids.push(VerifierOpeningId::Jolt(instruction_outputs[1]));
    output_ids.push(VerifierOpeningId::Jolt(instruction_outputs[2]));
    output_ids.extend(map_jolt_opening_ids(
        ram::raf_evaluation_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::output_check_output_openings().to_vec(),
    ));
    let aliases = vec![
        OpeningAlias::new(
            VerifierOpeningId::Jolt(instruction_outputs[0]),
            VerifierOpeningId::Jolt(product_remainder_output_openings()[4]),
        ),
        OpeningAlias::new(
            VerifierOpeningId::Jolt(instruction_outputs[3]),
            VerifierOpeningId::Jolt(product_remainder_output_openings()[0]),
        ),
        OpeningAlias::new(
            VerifierOpeningId::Jolt(instruction_outputs[4]),
            VerifierOpeningId::Jolt(product_remainder_output_openings()[1]),
        ),
    ];

    let mut batch_claims = vec![
        (
            ram_read_write.spec().rounds,
            map_jolt_expr(ram_read_write.input_expression::<PCS::Field>()),
            map_jolt_expr(ram_read_write.output_expression::<PCS::Field>()),
        ),
        (
            product_remainder.spec().rounds,
            map_jolt_expr(product_remainder.input_expression::<PCS::Field>()),
            product_remainder_output,
        ),
        (
            instruction_reduction.spec().rounds,
            map_jolt_expr(instruction_reduction.input_expression::<PCS::Field>()),
            map_jolt_expr(instruction_reduction.output_expression::<PCS::Field>()),
        ),
    ];
    batch_claims.extend([
        (
            ram_raf.spec().rounds,
            map_jolt_expr(ram_raf.input_expression::<PCS::Field>()),
            map_jolt_expr(ram_raf.output_expression::<PCS::Field>()),
        ),
        (
            ram_output.spec().rounds,
            map_jolt_expr(ram_output.input_expression::<PCS::Field>()),
            map_jolt_expr(ram_output.output_expression::<PCS::Field>()),
        ),
    ]);
    let coefficients = &input.stage2.batch_consistency.batching_coefficients;
    if batch_claims.len() != coefficients.len() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.batch: expected {} batching coefficients, got {}",
                batch_claims.len(),
                coefficients.len()
            ),
        });
    }
    let input_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((rounds, input_expr, _), coefficient)| {
            let scale = *coefficient
                * PCS::Field::pow2(input.stage2.batch_consistency.max_num_vars - *rounds);
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
        "stage2.batch",
        SumcheckStatement::new(
            input.stage2.batch_consistency.max_num_vars,
            input.stage2.batch_consistency.max_degree,
        ),
        domain_spec(ram_read_write.spec()),
        input.stage2.batch_consistency.consistency.clone(),
        &input.stage2.batch_output_claims,
        values,
        output_ids,
        aliases,
        input_claim,
        output_claim,
    )
}

fn selected_product_uniskip_input_expr<F: Field>(
    weights: &[F],
) -> Result<VerifierExpr<F>, VerifierError> {
    let [product_weight, should_branch_weight, should_jump_weight, rest @ ..] = weights else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.product_uniskip: expected {} weights, got {}",
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                weights.len()
            ),
        });
    };
    let expr = scale_expr(
        opening(VerifierOpeningId::Jolt(product_outer_opening())),
        *product_weight,
    ) + scale_expr(
        opening(VerifierOpeningId::Jolt(
            product_should_branch_outer_opening(),
        )),
        *should_branch_weight,
    ) + scale_expr(
        opening(VerifierOpeningId::Jolt(product_should_jump_outer_opening())),
        *should_jump_weight,
    );

    if !rest.is_empty() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.product_uniskip: expected no field weights, got {}",
                rest.len()
            ),
        });
    }
    Ok(expr)
}

fn selected_product_remainder_output_expr<F: Field>(
    weights: &[F],
    tau_kernel: F,
) -> Result<VerifierExpr<F>, VerifierError> {
    let [instruction_product_weight, should_branch_weight, should_jump_weight, rest @ ..] = weights
    else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.batch: expected {} product weights, got {}",
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                weights.len()
            ),
        });
    };
    let [left_instruction_input, right_instruction_input, jump_flag, _write_lookup_output_to_rd, lookup_output, branch_flag, next_is_noop, _virtual_instruction] =
        product_remainder_output_openings();

    let left_base = scale_expr(
        opening(VerifierOpeningId::Jolt(left_instruction_input)),
        *instruction_product_weight,
    ) + scale_expr(
        opening(VerifierOpeningId::Jolt(lookup_output)),
        *should_branch_weight,
    ) + scale_expr(
        opening(VerifierOpeningId::Jolt(jump_flag)),
        *should_jump_weight,
    );
    let right_base = scale_expr(
        opening(VerifierOpeningId::Jolt(right_instruction_input)),
        *instruction_product_weight,
    ) + scale_expr(
        opening(VerifierOpeningId::Jolt(branch_flag)),
        *should_branch_weight,
    ) + scale_expr(VerifierExpr::one(), *should_jump_weight)
        + scale_expr(
            opening(VerifierOpeningId::Jolt(next_is_noop)),
            -*should_jump_weight,
        );

    let (left, right) = {
        if !rest.is_empty() {
            return Err(VerifierError::BlindFoldConstructionFailed {
                reason: format!(
                    "stage2.batch: expected no field product weights, got {}",
                    rest.len()
                ),
            });
        }
        (left_base, right_base)
    };

    Ok(scale_expr(left * right, tau_kernel))
}
