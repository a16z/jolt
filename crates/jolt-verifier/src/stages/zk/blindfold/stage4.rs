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
    let registers_claims = relations::registers::ReadWriteChecking::new(register_dimensions);
    let ram_init = ram_val_check_init(input)?;
    // Supply the `Val_init` decomposition scalars as `Public` values (formerly
    // baked as `Term` constants in the expression); the advice / program-image
    // openings they weight remain hidden witnesses.
    values.public(
        JoltDerivedId::from(RamValCheckPublic::InitEval),
        ram_init.public_eval,
    )?;
    for contribution in &ram_init.contributions {
        values.public(
            JoltDerivedId::from(contribution.selector),
            contribution.neg_selector,
        )?;
    }
    let ram_val_claims = relations::ram::RamValCheck::new(relations::ram::RamValCheckShape {
        dimensions: trace_dimensions,
        contributions: ram_init
            .contributions
            .iter()
            .map(|contribution| relations::ram::RamValContribution {
                selector: contribution.selector,
                opening: contribution.opening,
            })
            .collect(),
    });

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)),
        input.stage4.challenges.registers_read_write.gamma,
    )?;
    let registers_point = input
        .stage4
        .batch_consistency
        .try_instance_point(registers_claims.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersReadWriteChecking, error))?;
    let registers_opening = register_dimensions
        .read_write_opening_point(&registers_point)
        .map_err(|error| public_error(JoltRelationId::RegistersReadWriteChecking, error))?;
    let registers_reduction_point = input
        .stage3
        .batch_consistency
        .try_instance_point(log_t)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RegistersClaimReduction, error))?;
    let registers_reduction_opening = registers_reduction_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    values.public(
        JoltDerivedId::from(RegistersReadWritePublic::EqCycle),
        try_eq_mle(&registers_reduction_opening, &registers_opening.r_cycle)
            .map_err(|error| public_error(JoltRelationId::RegistersReadWriteChecking, error))?,
    )?;

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RamValCheckChallenge::Gamma)),
        input.stage4.challenges.ram_val_check.gamma,
    )?;
    let ram_val_point = input
        .stage4
        .batch_consistency
        .try_instance_point(ram_val_claims.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamValCheck, error))?;
    let ram_val_cycle = ram_val_point.iter().rev().copied().collect::<Vec<_>>();
    let r_cycle = input
        .stage2
        .output_points
        .ram_read_write_point()
        .get(log_k..)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: "RAM read-write opening point is shorter than the RAM address".to_string(),
        })?;
    values.public(
        JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma),
        LtPolynomial::evaluate(&ram_val_cycle, r_cycle)
            + input.stage4.challenges.ram_val_check.gamma,
    )?;

    let mut output_ids = Vec::new();
    if input.proof.untrusted_advice_commitment.is_some() {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Untrusted));
    }
    if input.checked.trusted_advice_commitment_present {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Trusted));
    }
    if input.checked.precommitted.program_image.is_some() {
        output_ids.push(program_image::ram_val_check_contribution_opening());
    }
    output_ids.extend(
        relations::registers::RegistersReadWriteOutputClaims::<PCS::Field> {
            registers_val: PCS::Field::zero(),
            rs1_ra: PCS::Field::zero(),
            rs2_ra: PCS::Field::zero(),
            rd_wa: PCS::Field::zero(),
            rd_inc: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    // The advice / program-image openings are produced by the RAM value-check
    // instance, but the stage-4 commit (flush) order appends them *first* (above),
    // before the registers; so here, at the tail, only the main `ram_ra`/`ram_inc`
    // canonical order is emitted (advice / program-image leaves left `None`),
    // preserving the prover's per-stage opening-id block order.
    output_ids.extend(
        relations::ram::RamValCheckOutputClaims::<PCS::Field> {
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
            ram_ra: PCS::Field::zero(),
            ram_inc: PCS::Field::zero(),
        }
        .canonical_order(),
    );

    let batch_claims = [
        relation_claim(&registers_claims),
        relation_claim(&ram_val_claims),
    ];

    add_batched_stage(
        builder,
        "stage4.batch",
        registers_claims.domain(),
        &batch_claims,
        &input.stage4.batch_consistency,
        &input.stage4.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}
