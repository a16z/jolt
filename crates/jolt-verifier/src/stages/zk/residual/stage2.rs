use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::ClaimReduction as InstructionClaimReduction;
use jolt_claims::protocols::jolt::relations::ram::{OutputCheck, RafEvaluation, ReadWriteChecking};
use jolt_claims::protocols::jolt::{SpartanProductVirtualizationPublic, TraceDimensions};

use super::*;
use crate::stages::relations::{ConcreteSumcheck, SymbolicOf};
use crate::stages::stage2::{product_remainder::ProductRemainder, product_uniskip::ProductUniskip};

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage2<F, PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    mut builder: ResidualBuilder<F>,
    values: &mut SourceValues<F>,
) -> Result<ResidualBuilder<F>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let log_k = crate::num::ilog2(input.checked.ram_K);
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = input.checked.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    // `product_tau_low` is stage 1's remainder cycle point (low half), computed once
    // by the stage-2 verifier and carried on the ZK output.
    let product_tau_low = input.stage2.product_tau_low.clone();

    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_tau_high,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: error.to_string(),
    })?;
    for (index, weight) in product_weights.into_iter().enumerate() {
        values.public(
            JoltDerivedId::from(SpartanProductVirtualizationPublic::UniskipLagrangeWeight(
                index,
            )),
            weight,
        )?;
    }
    builder = add_uniskip_stage(
        builder,
        "stage2.product_uniskip",
        UniskipRound {
            degree: SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
            domain: SumcheckDomain::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE),
            rounds: &input.stage2.product_uniskip_rounds,
            output_claims: &input.stage2.product_uniskip_output_claims,
        },
        values,
        selected_product_claim::<F, ProductUniskip<F>>(product_dimensions),
    )?;

    let ram_read_write = ReadWriteChecking::new(read_write_dimensions);
    let product_remainder = SymbolicOf::<F, ProductRemainder<F>>::new(product_dimensions);
    let instruction_reduction = InstructionClaimReduction::new(trace_dimensions);
    let ram_raf = RafEvaluation::new(raf_dimensions);
    let ram_output = OutputCheck::new(read_write_dimensions);

    let ram_read_write_point = input
        .stage2
        .batch_rounds
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
        .batch_rounds
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
    for (index, weight) in product_lagrange_weights.into_iter().enumerate() {
        values.public(
            JoltDerivedId::from(SpartanProductVirtualizationPublic::LagrangeWeight(index)),
            weight,
        )?;
    }
    values.public(
        JoltDerivedId::from(SpartanProductVirtualizationPublic::TauKernel),
        product_tau_high_bound * product_tau_low_eq,
    )?;

    let instruction_point = input
        .stage2
        .batch_rounds
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

    #[cfg(feature = "field-inline")]
    let field_registers_reduction = super::field_inline::stage2_claim_reduction(
        values,
        log_t,
        &input.stage2.batch_rounds,
        input
            .stage2
            .challenges
            .field_registers_claim_reduction
            .gamma,
        &product_tau_low,
    )?;

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "log_t and log_k are ilog2 results (< 64); the sum cannot overflow usize"
    )]
    let active_stage2_rounds = log_t + log_k;
    let phase1_offset = input
        .stage2
        .batch_rounds
        .try_round_offset(active_stage2_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?
        .checked_add(read_write_dimensions.phase1_num_rounds())
        .ok_or_else(|| VerifierError::ResidualConstructionFailed {
            reason: "stage2: RAM RAF phase-1 round offset overflows usize".to_string(),
        })?;
    let ram_raf_point = input
        .stage2
        .batch_rounds
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
        .batch_rounds
        .try_instance_point_at(phase1_offset, ram_output.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamOutputCheck, error))?;
    let ram_output_address = read_write_dimensions
        .address_opening_point(&ram_output_point)
        .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let output_publics = ram_output_publics(
        input,
        &input.stage2.challenges.ram_output_check.output_address,
        &ram_output_address,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::EqAddress),
        output_publics.0,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::IoMask),
        output_publics.1,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::ValIo),
        output_publics.2,
    )?;

    // Member declaration order (= batching-coefficient draw order): the field-inline claim
    // reduction sits between the instruction reduction and RAM RAF evaluation, exactly as in
    // `Stage2BatchSumchecks`.
    let mut batch_claims = vec![
        relation_claim(&ram_read_write),
        selected_product_claim::<F, ProductRemainder<F>>(product_dimensions),
        relation_claim(&instruction_reduction),
    ];
    #[cfg(feature = "field-inline")]
    batch_claims.push(relation_claim(&field_registers_reduction));
    batch_claims.push(relation_claim(&ram_raf));
    batch_claims.push(relation_claim(&ram_output));

    add_batched_stage(
        builder,
        "stage2.batch",
        ram_read_write.domain(),
        &batch_claims,
        &input.stage2.batch_rounds,
        &input.stage2.batch_output_claims,
        values,
    )
}

/// The verifier's selected (ordinary or composed) product relation's claim.
fn selected_product_claim<F, I>(dimensions: SpartanProductDimensions) -> MemberClaim<F>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SymbolicOf<F, I>: SymbolicSumcheck<Shape = SpartanProductDimensions>,
    <SymbolicOf<F, I> as SymbolicSumcheck>::OpeningId: Into<ComposedOpeningId>,
    <SymbolicOf<F, I> as SymbolicSumcheck>::DerivedId: Into<VerifierPublicId>,
    <SymbolicOf<F, I> as SymbolicSumcheck>::ChallengeId: Into<VerifierPublicId>,
{
    relation_claim(&SymbolicOf::<F, I>::new(dimensions))
}
