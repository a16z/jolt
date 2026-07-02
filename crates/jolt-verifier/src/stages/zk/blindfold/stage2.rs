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

    // `product_tau_low` is stage 1's remainder cycle point (low half), computed once
    // by the stage-2 verifier and carried on the ZK output.
    let product_tau_low = input.stage2.product_tau_low.clone();

    let product_uniskip_rounds = 1;
    let product_uniskip_degree = SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE;
    let product_uniskip_domain =
        JoltSumcheckDomain::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE);
    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_tau_high,
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
        SumcheckStatement::new(product_uniskip_rounds, product_uniskip_degree),
        domain_spec(product_uniskip_domain),
        input.stage2.product_uniskip_consistency.clone(),
        &input.stage2.product_uniskip_output_claims,
        values,
        vec![product_uniskip_opening()],
        Vec::new(),
        product_uniskip_input,
        opening(product_uniskip_opening()),
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
        .try_instance_point(ram_read_write.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamReadWriteChecking, error))?;
    let ram_read_write_opening = read_write_dimensions
        .read_write_opening_point(&ram_read_write_point)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    let eq_cycle = try_eq_mle(&product_tau_low, &ram_read_write_opening.r_cycle)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
        input.stage2.challenges.ram_read_write.gamma,
    )?;
    values.public(JoltDerivedId::from(RamReadWritePublic::EqCycle), eq_cycle)?;

    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(product_remainder.rounds())
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let product_tau_high_bound = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_tau_high,
        input.stage2.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_low_eq = try_eq_mle(&product_tau_low, &product_opening_point)
        .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_lagrange_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_uniskip_challenge,
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
        .try_instance_point(instruction_reduction.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionClaimReduction, error))?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(&instruction_opening_point, &product_tau_low)
        .map_err(|error| public_error(JoltRelationId::InstructionClaimReduction, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            InstructionClaimReductionChallenge::Gamma,
        )),
        input.stage2.challenges.instruction_claim_reduction.gamma,
    )?;
    values.public(
        JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
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
        .try_instance_point_at(phase1_offset, ram_raf.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_address = read_write_dimensions
        .address_opening_point(&ram_raf_point)
        .map_err(|error| public_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_unmap_address = IdentityPolynomial::new(log_k).evaluate(&ram_raf_address)
        * PCS::Field::from_u64(8)
        + PCS::Field::from_u64(input.checked.public_io.memory_layout.get_lowest_address());
    values.public(
        JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
        ram_raf_unmap_address,
    )?;

    let ram_output_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_output.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamOutputCheck, error))?;
    let ram_output_address = read_write_dimensions
        .address_opening_point(&ram_output_point)
        .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let output_publics = ram_output_publics(
        input,
        &input.stage2.output_address_challenges,
        &ram_output_address,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::EqIoMask),
        output_publics.0,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
        output_publics.1,
    )?;

    let product_order = relations::spartan::ProductRemainderOutputClaims::<PCS::Field> {
        left_instruction_input: PCS::Field::zero(),
        right_instruction_input: PCS::Field::zero(),
        jump_flag: PCS::Field::zero(),
        write_lookup_output_to_rd: PCS::Field::zero(),
        lookup_output: PCS::Field::zero(),
        branch_flag: PCS::Field::zero(),
        next_is_noop: PCS::Field::zero(),
        virtual_instruction: PCS::Field::zero(),
    }
    .canonical_order();
    let instruction_outputs =
        relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims::<
            PCS::Field,
        > {
            lookup_output: Some(PCS::Field::zero()),
            left_lookup_operand: PCS::Field::zero(),
            right_lookup_operand: PCS::Field::zero(),
            left_instruction_input: Some(PCS::Field::zero()),
            right_instruction_input: Some(PCS::Field::zero()),
        }
        .canonical_order();

    let mut output_ids = Vec::new();
    output_ids.extend(
        relations::ram::RamReadWriteOutputClaims::<PCS::Field> {
            val: PCS::Field::zero(),
            ra: PCS::Field::zero(),
            inc: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    output_ids.extend(product_order.clone());
    output_ids.push(instruction_outputs[1]);
    output_ids.push(instruction_outputs[2]);
    output_ids.extend(
        relations::ram::RamRafEvaluationOutputClaims::<PCS::Field> {
            ram_ra: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    output_ids.extend(
        relations::ram::RamOutputCheckOutputClaims::<PCS::Field> {
            val_final: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    let aliases = vec![
        OpeningAlias::new(instruction_outputs[0], product_order[4]),
        OpeningAlias::new(instruction_outputs[3], product_order[0]),
        OpeningAlias::new(instruction_outputs[4], product_order[1]),
    ];

    let batch_claims = [
        relation_claim(&ram_read_write),
        (
            product_remainder.rounds(),
            map_jolt_expr(product_remainder.input_expression::<PCS::Field>()),
            product_remainder_output,
        ),
        relation_claim(&instruction_reduction),
        relation_claim(&ram_raf),
        relation_claim(&ram_output),
    ];

    add_batched_stage(
        builder,
        "stage2.batch",
        ram_read_write.domain(),
        &batch_claims,
        &input.stage2.batch_consistency,
        &input.stage2.batch_output_claims,
        values,
        output_ids,
        aliases,
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
    let expr = scale_expr(opening(product_outer_opening()), *product_weight)
        + scale_expr(
            opening(product_should_branch_outer_opening()),
            *should_branch_weight,
        )
        + scale_expr(
            opening(product_should_jump_outer_opening()),
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
    let left_base = scale_expr(
        opening(left_instruction_input_product()),
        *instruction_product_weight,
    ) + scale_expr(opening(lookup_output_product()), *should_branch_weight)
        + scale_expr(opening(jump_flag_product()), *should_jump_weight);
    let right_base = scale_expr(
        opening(right_instruction_input_product()),
        *instruction_product_weight,
    ) + scale_expr(opening(branch_flag_product()), *should_branch_weight)
        + scale_expr(VerifierExpr::one(), *should_jump_weight)
        + scale_expr(opening(next_is_noop_product()), -*should_jump_weight);

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
