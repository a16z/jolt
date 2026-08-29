use super::*;

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafOutputOpenings;
use jolt_claims::protocols::jolt::relations::ram::RamRaClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationOutputClaims;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage5<F, PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<F, VC::Output>,
    values: &mut SourceValues<F>,
) -> Result<Builder<F, VC::Output>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let log_k = crate::num::ilog2(input.checked.ram_K);
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
    let ram_raf_cycle = point_suffix(
        input.stage2.output_points.ram_raf_evaluation_point(),
        log_k,
        JoltRelationId::RamRaClaimReduction,
    )?;
    let ram_read_write_cycle = point_suffix(
        input.stage2.output_points.ram_read_write_point(),
        log_k,
        JoltRelationId::RamRaClaimReduction,
    )?;
    let ram_val_cycle = point_suffix(
        input.stage4.output_points.ram_val_check_point(),
        log_k,
        JoltRelationId::RamRaClaimReduction,
    )?;
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
    let registers_read_write_cycle = point_suffix(
        input.stage4.output_points.registers_read_write_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::RegistersValEvaluation,
    )?;
    values.public(
        JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
        LtPolynomial::evaluate(&registers_cycle, registers_read_write_cycle),
    )?;

    // The FR val-evaluation member (declared last, no instance challenge) and
    // its baked `LtCycle` public, at the same source-values position as
    // before.
    #[cfg(feature = "field-inline")]
    let field_registers_claims = super::field_inline::stage5_val_evaluation(
        values,
        log_t,
        input
            .stage5
            .output_points
            .field_registers_val_evaluation_point(),
        input
            .stage4
            .output_points
            .field_registers_read_write_point(),
    )?;

    let output_ids = stage5_output_ids::<PCS::Field>(instruction_output_openings);

    // Member declaration order (= batching-coefficient draw order): the FR
    // val-evaluation member is declared last, exactly as in `Stage5Sumchecks`.
    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut batch_claims = vec![
        relation_claim(&instruction_claims),
        relation_claim(&ram_claims),
        relation_claim(&registers_claims),
    ];
    #[cfg(feature = "field-inline")]
    batch_claims.push(relation_claim(&field_registers_claims));

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
        Vec::new(),
    )
}

/// The stage-5 committed output row order: the instruction read-RAF openings,
/// the reduced RAM RA, the register value-evaluation openings, then (under
/// `field-inline`) the two FR val-evaluation rows at the tail — the clear
/// absorb order (the generated `Stage5Sumchecks` member-declaration absorb).
fn stage5_output_ids<F: JoltField>(
    instruction_output_openings: InstructionReadRafOutputOpenings,
) -> Vec<VerifierOpeningId> {
    let mut output_ids: Vec<VerifierOpeningId> =
        composite_ids(instruction_output_openings.lookup_table_flags);
    output_ids.extend(composite_ids(instruction_output_openings.instruction_ra));
    output_ids.push(instruction_output_openings.instruction_raf_flag.into());
    output_ids.extend(composite_ids(
        RamRaClaimReductionOutputClaims::<F> { ram_ra: F::zero() }.canonical_order(),
    ));
    output_ids.extend(composite_ids(
        RegistersValEvaluationOutputClaims::<F> {
            rd_inc: F::zero(),
            rd_wa: F::zero(),
        }
        .canonical_order(),
    ));
    // The two FR val-evaluation rows, after the ordinary register
    // value-evaluation outputs — the clear absorb order (the FR member is
    // declared last, so the generated absorb appends them at the tail).
    #[cfg(feature = "field-inline")]
    output_ids.extend(super::field_inline::stage5_output_ids());
    output_ids
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use crate::stages::stage5::outputs::{Stage5OutputClaims, Stage5Sumchecks};
    use crate::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
    use crate::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
    use crate::stages::stage5::InstructionReadRaf;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
    use jolt_claims::protocols::jolt::relations::ram::RamRaClaimReductionOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationOutputClaims;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The stage-5 committed row order is the clear absorb order (the
    /// generated member-declaration `opening_values`), locked entry-for-entry
    /// over sentinel-valued claims (FR-on: the two FR val-evaluation rows at
    /// the tail).
    #[test]
    fn stage5_output_ids_match_the_clear_absorb_order() {
        let log_t = 3usize;
        let dimensions = InstructionReadRafDimensions::try_from((log_t, 128, 3)).unwrap();
        let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
        let sumchecks = Stage5Sumchecks::<Fr> {
            instruction_read_raf: InstructionReadRaf::new(dimensions),
            ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, 3),
            registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation:
                crate::stages::stage5::outputs::FieldRegistersValEvaluation::new(
                    jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions::new(log_t),
                ),
        };
        let openings = instruction::read_raf_output_openings(dimensions);
        let claims = Stage5OutputClaims::<Fr> {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: (0..openings.lookup_table_flags.len() as u64)
                    .map(|index| fr(100 + index))
                    .collect(),
                instruction_ra: (0..openings.instruction_ra.len() as u64)
                    .map(|index| fr(200 + index))
                    .collect(),
                instruction_raf_flag: fr(300),
            },
            ram_ra_claim_reduction: RamRaClaimReductionOutputClaims { ram_ra: fr(301) },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: fr(302),
                rd_wa: fr(303),
            },
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation:
                crate::stages::stage5::outputs::FieldRegistersValEvaluationOutputClaims {
                    rd_inc: fr(401),
                    rd_wa: fr(402),
                },
        };
        let clear_values = sumchecks.opening_values(&claims);

        let output_ids = stage5_output_ids::<Fr>(instruction::read_raf_output_openings(dimensions));
        assert_eq!(output_ids.len(), clear_values.len());
        for (id, expected) in output_ids.iter().zip(clear_values) {
            let resolved = match id {
                VerifierOpeningId::Jolt(id) => claims
                    .instruction_read_raf
                    .resolve_output(id)
                    .or_else(|| claims.ram_ra_claim_reduction.resolve_output(id))
                    .or_else(|| claims.registers_val_evaluation.resolve_output(id)),
                #[cfg(feature = "field-inline")]
                VerifierOpeningId::FieldInline(id) => {
                    claims.field_registers_val_evaluation.resolve_output(id)
                }
                #[cfg(not(feature = "field-inline"))]
                VerifierOpeningId::FieldInline(_) => None,
            };
            assert_eq!(
                resolved,
                Some(expected),
                "row {id:?} must sit at the clear absorb position of value {expected:?}",
            );
        }
    }
}
