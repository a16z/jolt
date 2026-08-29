use super::*;

use jolt_claims::protocols::jolt::relations::ram::RamValCheckOutputClaims;
use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteOutputClaims;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage4<F, PCS, VC, ZkProof>(
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
    let register_dimensions = input
        .proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    // Eager: the proof-supplied phase split feeds round-count subtractions
    // (`phase3_cycle_rounds` etc.) before any lazy point-derivation check.
    register_dimensions
        .validate_phase_split()
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
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

    // The FR read/write member and its baked publics (relation + gamma +
    // EqCycle), at the same source-values position as before. The upstream
    // reduced point (`r_prod`) — the FR claim reduction's stage-2 opening
    // point — is the fixed cycle.
    #[cfg(feature = "field-inline")]
    let field_registers_claims = super::field_inline::stage4_read_write(
        values,
        log_t,
        input.stage4.challenges.field_registers_read_write.gamma,
        input
            .stage2
            .output_points
            .field_registers_claim_reduction
            .rd_value(),
        input
            .stage4
            .output_points
            .field_registers_read_write_point(),
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

    let output_ids = stage4_output_ids::<PCS::Field>(
        input.proof.untrusted_advice_commitment.is_some(),
        input.checked.trusted_advice_commitment_present,
        input.checked.precommitted.program_image.is_some(),
    );

    // Member declaration order (= batching-coefficient draw order): the FR
    // read/write member sits between the registers and the RAM value-check,
    // exactly as in `Stage4Sumchecks`.
    let mut batch_claims = vec![relation_claim(&registers_claims)];
    #[cfg(feature = "field-inline")]
    batch_claims.push(relation_claim(&field_registers_claims));
    batch_claims.push(relation_claim(&ram_val_claims));

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
        Vec::new(),
    )
}

/// The stage-4 committed output row order: the staged `Val_init` advice /
/// program-image openings first, the register read/write openings, then
/// (under `field-inline`) the five FR read/write rows, then `ram_ra`/`ram_inc`
/// — the clear absorb order (`Stage4OutputClaims::opening_values`) exactly.
fn stage4_output_ids<F: JoltField>(
    untrusted_advice: bool,
    trusted_advice: bool,
    program_image: bool,
) -> Vec<VerifierOpeningId> {
    let mut output_ids: Vec<VerifierOpeningId> = Vec::new();
    if untrusted_advice {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Untrusted).into());
    }
    if trusted_advice {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Trusted).into());
    }
    if program_image {
        output_ids.push(program_image::ram_val_check_contribution_opening().into());
    }
    output_ids.extend(composite_ids(
        RegistersReadWriteOutputClaims::<F> {
            registers_val: F::zero(),
            rs1_ra: F::zero(),
            rs2_ra: F::zero(),
            rd_wa: F::zero(),
            rd_inc: F::zero(),
        }
        .canonical_order(),
    ));
    // The five FR read/write rows, spliced after the register openings and
    // before `ram_ra`/`ram_inc` — the clear absorb order.
    #[cfg(feature = "field-inline")]
    output_ids.extend(super::field_inline::stage4_output_ids());
    // The advice / program-image openings are produced by the RAM value-check
    // instance, but the stage-4 commit (flush) order appends them *first* (above),
    // before the registers; so here, at the tail, only the main `ram_ra`/`ram_inc`
    // canonical order is emitted (advice / program-image leaves left `None`),
    // preserving the prover's per-stage opening-id block order.
    output_ids.extend(composite_ids(
        RamValCheckOutputClaims::<F> {
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
            ram_ra: F::zero(),
            ram_inc: F::zero(),
        }
        .canonical_order(),
    ));
    output_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The stage-4 committed row order is the clear curated absorb order
    /// (`Stage4OutputClaims::opening_values`), locked entry-for-entry over
    /// sentinel-valued claims: every lowered id resolves to the value at its
    /// row position (FR-on: the five FR rows spliced after the registers,
    /// before `ram_ra`/`ram_inc`).
    #[test]
    fn stage4_output_ids_match_the_clear_absorb_order() {
        use crate::stages::stage4::outputs::Stage4OutputClaims;
        use jolt_claims::protocols::jolt::relations::ram::RamValCheckOutputClaims;
        use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteOutputClaims;

        let claims = Stage4OutputClaims::<Fr> {
            registers_read_write: RegistersReadWriteOutputClaims {
                registers_val: fr(1),
                rs1_ra: fr(2),
                rs2_ra: fr(3),
                rd_wa: fr(4),
                rd_inc: fr(5),
            },
            #[cfg(feature = "field-inline")]
            field_registers_read_write:
                crate::stages::stage4::outputs::FieldRegistersReadWriteOutputClaims {
                    registers_val: fr(11),
                    rs1_ra: fr(12),
                    rs2_ra: fr(13),
                    rd_wa: fr(14),
                    rd_inc: fr(15),
                },
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
                ram_ra: fr(6),
                ram_inc: fr(7),
            },
        };
        let clear_values = claims.opening_values();
        let output_ids = stage4_output_ids::<Fr>(false, false, false);
        assert_eq!(output_ids.len(), clear_values.len());
        #[cfg(not(feature = "field-inline"))]
        assert_eq!(output_ids.len(), 7);
        #[cfg(feature = "field-inline")]
        assert_eq!(output_ids.len(), 12);

        for (id, expected) in output_ids.iter().zip(clear_values) {
            let resolved = match id {
                VerifierOpeningId::Jolt(id) => claims
                    .registers_read_write
                    .resolve_output(id)
                    .or_else(|| claims.ram_val_check.resolve_output(id)),
                #[cfg(feature = "field-inline")]
                VerifierOpeningId::FieldInline(id) => {
                    claims.field_registers_read_write.resolve_output(id)
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
