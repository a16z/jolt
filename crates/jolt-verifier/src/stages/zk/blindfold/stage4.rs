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

    values.challenge(
        JoltChallengeId::from(RegistersReadWriteChallenge::Gamma),
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
    values.challenge(
        JoltChallengeId::from(RegistersReadWriteChallenge::EqCycle),
        try_eq_mle(&registers_reduction_opening, &registers_opening.r_cycle)
            .map_err(|error| public_error(JoltRelationId::RegistersReadWriteChecking, error))?,
    )?;

    values.challenge(
        JoltChallengeId::from(RamValCheckChallenge::Gamma),
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
    values.challenge(
        JoltChallengeId::from(RamValCheckChallenge::LtCyclePlusGamma),
        LtPolynomial::evaluate(&ram_val_cycle, r_cycle) + input.stage4.public.ram_val_check_gamma,
    )?;

    let mut output_ids = Vec::new();
    if input.proof.untrusted_advice_commitment.is_some() {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Untrusted));
    }
    if input.checked.trusted_advice_commitment_present {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Trusted));
    }
    output_ids.extend(registers::read_write_checking_output_openings());
    output_ids.extend(ram::val_check_output_openings());
    add_batched_stage(
        builder,
        "stage4.batch",
        &[registers_claims, ram_val_claims],
        &input.stage4.batch_consistency,
        &input.stage4.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}
