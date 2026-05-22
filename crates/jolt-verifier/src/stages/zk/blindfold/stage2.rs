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

    let product_uniskip = spartan::product_uniskip::<PCS::Field>(product_dimensions);
    let product_weights = centered_lagrange_evals_array::<
        PCS::Field,
        { PRODUCT_UNISKIP_DOMAIN_SIZE },
    >(input.stage2.public.product_tau_high)
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: error.to_string(),
    })?;
    for (index, value) in product_weights.into_iter().enumerate() {
        values.public(
            JoltPublicId::from(SpartanProductVirtualizationPublic::UniskipLagrangeWeight(
                index,
            )),
            value,
        )?;
    }
    builder = add_single_stage(
        builder,
        "stage2.product_uniskip",
        &product_uniskip,
        &input.stage2.product_uniskip_consistency,
        &input.stage2.product_uniskip_output_claims,
        values,
        vec![product_uniskip_opening()],
        Vec::new(),
    )?;

    let ram_read_write = ram::read_write_checking::<PCS::Field>(read_write_dimensions);
    let product_remainder = spartan::product_remainder::<PCS::Field>(product_dimensions);
    let instruction_reduction =
        jolt_claims::protocols::jolt::formulas::claim_reductions::instruction::claim_reduction::<
            PCS::Field,
        >(trace_dimensions);
    let ram_raf = ram::raf_evaluation::<PCS::Field>(raf_dimensions);
    let ram_output = ram::output_check::<PCS::Field>(read_write_dimensions);

    let ram_read_write_point = input
        .stage2
        .batch_consistency
        .try_instance_point(ram_read_write.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamReadWriteChecking, error))?;
    let ram_read_write_opening = read_write_dimensions
        .read_write_opening_point(&ram_read_write_point)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    let eq_cycle = try_eq_mle(
        &input.stage2.public.product_tau_low,
        &ram_read_write_opening.r_cycle,
    )
    .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    values.challenge(
        JoltChallengeId::from(RamReadWriteChallenge::Gamma),
        input.stage2.public.ram_read_write_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(RamReadWriteChallenge::EqCycle),
        eq_cycle,
    )?;

    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(product_remainder.sumcheck.rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let product_tau_high_bound = centered_lagrange_kernel(
        PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.public.product_tau_high,
        input.stage2.public.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_low_eq =
        try_eq_mle(&input.stage2.public.product_tau_low, &product_opening_point)
            .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_publics = SpartanProductPublicValues {
        lagrange_weights: centered_lagrange_evals_array::<
            PCS::Field,
            { PRODUCT_UNISKIP_DOMAIN_SIZE },
        >(input.stage2.public.product_uniskip_challenge)
        .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?,
        tau_kernel: product_tau_high_bound * product_tau_low_eq,
    };
    values.public(
        JoltPublicId::from(SpartanProductVirtualizationPublic::TauKernel),
        product_publics
            .value(SpartanProductVirtualizationPublic::TauKernel)
            .unwrap_or_else(PCS::Field::zero),
    )?;
    for index in 0..PRODUCT_UNISKIP_DOMAIN_SIZE {
        values.public(
            JoltPublicId::from(SpartanProductVirtualizationPublic::LagrangeWeight(index)),
            product_publics
                .value(SpartanProductVirtualizationPublic::LagrangeWeight(index))
                .unwrap_or_else(PCS::Field::zero),
        )?;
    }

    let instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(instruction_reduction.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionClaimReduction, error))?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(
        &instruction_opening_point,
        &input.stage2.public.product_tau_low,
    )
    .map_err(|error| public_error(JoltRelationId::InstructionClaimReduction, error))?;
    values.challenge(
        JoltChallengeId::from(InstructionClaimReductionChallenge::Gamma),
        input.stage2.public.instruction_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionClaimReductionChallenge::EqSpartan),
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
        .try_instance_point_at(phase1_offset, ram_raf.sumcheck.rounds)
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
        .try_instance_point_at(phase1_offset, ram_output.sumcheck.rounds)
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
    output_ids.extend(ram::read_write_checking_output_openings());
    output_ids.extend(product_remainder_output_openings());
    let instruction_outputs =
        jolt_claims::protocols::jolt::formulas::claim_reductions::instruction::claim_reduction_output_openings();
    output_ids.push(instruction_outputs[1]);
    output_ids.push(instruction_outputs[2]);
    output_ids.extend(ram::raf_evaluation_output_openings());
    output_ids.extend(ram::output_check_output_openings());
    let aliases = vec![
        OpeningAlias::new(
            instruction_outputs[0],
            product_remainder_output_openings()[4],
        ),
        OpeningAlias::new(
            instruction_outputs[3],
            product_remainder_output_openings()[0],
        ),
        OpeningAlias::new(
            instruction_outputs[4],
            product_remainder_output_openings()[1],
        ),
    ];
    add_batched_stage(
        builder,
        "stage2.batch",
        &[
            ram_read_write,
            product_remainder,
            instruction_reduction,
            ram_raf,
            ram_output,
        ],
        &input.stage2.batch_consistency,
        &input.stage2.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}
