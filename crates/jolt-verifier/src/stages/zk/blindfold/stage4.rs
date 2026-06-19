use super::*;

pub(super) fn add_stage4<PCS, VC, ZkProof>(
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
    let register_dimensions = input
        .proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let registers_claims = registers::read_write_checking::<PCS::Field>(register_dimensions);
    let ram_init = ram_val_check_init(input)?;
    let ram_val_claims = ram::val_check::<PCS::Field>(trace_dimensions, ram_init);

    values.public(
        JoltPublicId::from(RegistersReadWritePublic::Gamma),
        input.stage4.public.registers_gamma,
    )?;
    let registers_point = input
        .stage4
        .batch_consistency
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersReadWriteChecking, error))?;
    let registers_opening = register_dimensions
        .read_write_opening_point(&registers_point)
        .map_err(|error| public_error(JoltRelationId::RegistersReadWriteChecking, error))?;
    let registers_reduction_point = input
        .stage3
        .batch_consistency
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(3)
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersClaimReduction, error))?;
    let registers_reduction_opening = registers_reduction_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    values.public(
        JoltPublicId::from(RegistersReadWritePublic::EqCycle),
        try_eq_mle(&registers_reduction_opening, &registers_opening.r_cycle)
            .map_err(|error| public_error(JoltRelationId::RegistersReadWriteChecking, error))?,
    )?;

    values.public(
        JoltPublicId::from(RamValCheckPublic::Gamma),
        input.stage4.public.ram_val_check_gamma,
    )?;
    let ram_val_point = input
        .stage4
        .batch_consistency
        .try_instance_point(ram_val_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamValCheck, error))?;
    let ram_val_cycle = ram_val_point.iter().rev().copied().collect::<Vec<_>>();
    let r_cycle = input
        .stage2
        .ram_val_check_inputs
        .ram_read_write_opening_point
        .get(log_k..)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: "RAM read-write opening point is shorter than the RAM address".to_string(),
        })?;
    values.public(
        JoltPublicId::from(RamValCheckPublic::LtCyclePlusGamma),
        LtPolynomial::evaluate(&ram_val_cycle, r_cycle) + input.stage4.public.ram_val_check_gamma,
    )?;

    let mut output_ids = Vec::new();
    if input.proof.untrusted_advice_commitment.is_some() {
        output_ids.push(VerifierOpeningId::Jolt(ram::val_check_advice_opening(
            JoltAdviceKind::Untrusted,
        )));
    }
    if input.checked.trusted_advice_commitment_present {
        output_ids.push(VerifierOpeningId::Jolt(ram::val_check_advice_opening(
            JoltAdviceKind::Trusted,
        )));
    }
    if input.checked.precommitted.program_image.is_some() {
        output_ids.push(VerifierOpeningId::Jolt(
            program_image::ram_val_check_contribution_opening(),
        ));
    }
    output_ids.extend(map_jolt_opening_ids(
        registers::read_write_checking_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::val_check_output_openings().to_vec(),
    ));

    let mut batch_claims = vec![(
        registers_claims.sumcheck.rounds,
        map_jolt_expr(registers_claims.input.expression().clone()),
        map_jolt_expr(registers_claims.output.expression().clone()),
    )];
    batch_claims.push((
        ram_val_claims.sumcheck.rounds,
        map_jolt_expr(ram_val_claims.input.expression().clone()),
        map_jolt_expr(ram_val_claims.output.expression().clone()),
    ));

    let coefficients = &input.stage4.batch_consistency.batching_coefficients;
    if batch_claims.len() != coefficients.len() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage4.batch: expected {} batching coefficients, got {}",
                batch_claims.len(),
                coefficients.len()
            ),
        });
    }
    let input_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((rounds, input_expr, _), coefficient)| {
            let scale = *coefficient
                * PCS::Field::pow2(input.stage4.batch_consistency.max_num_vars - *rounds);
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
        "stage4.batch",
        SumcheckStatement::new(
            input.stage4.batch_consistency.max_num_vars,
            input.stage4.batch_consistency.max_degree,
        ),
        domain_spec(registers_claims.sumcheck),
        input.stage4.batch_consistency.consistency.clone(),
        &input.stage4.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
        input_claim,
        output_claim,
    )
}
