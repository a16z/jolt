//! The BlindFold lowering's field-inline seam: every field-inline-specific piece of the ZK
//! verifier's R1CS build in one place — the field-inline members' symbolic relations and baked
//! publics per stage, the composed bytecode public extension, the field-inline output-row
//! splices, and the field-inline-lane expression terms. Each blindfold stage file keeps
//! exactly one contiguous, flagged region per interaction point, calling into here.

use jolt_claims::protocols::field_inline::geometry::claim_reductions as field_claim_reductions;
use jolt_claims::protocols::field_inline::geometry::registers as field_registers_geometry;
use jolt_claims::protocols::field_inline::geometry::spartan as field_spartan_geometry;
use jolt_claims::protocols::field_inline::relations::claim_reductions::increments as field_increments;
use jolt_claims::protocols::field_inline::relations::claim_reductions::registers as field_registers_reduction;
use jolt_claims::protocols::field_inline::relations::registers as field_registers;
use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineRelationId,
    FieldRegistersClaimReductionChallenge, FieldRegistersClaimReductionPublic,
    FieldRegistersIncClaimReductionChallenge, FieldRegistersIncClaimReductionPublic,
    FieldRegistersReadWriteChallenge, FieldRegistersReadWritePublic, FieldRegistersTraceDimensions,
    FieldRegistersValEvaluationPublic, FIELD_REGISTERS_LOG_K,
};
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::SymbolicSumcheck as _;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_poly::{try_eq_mle, LtPolynomial};
use jolt_riscv::JoltInstructionRow;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use super::{SourceValues, VerifierOpeningId};
use crate::config::JOLT_VERIFIER_CONFIG;
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::field_inline_bytecode::field_inline_stage_gamma_powers;
use crate::stages::stage4::Stage4OutputPoints;
use crate::stages::stage5::Stage5OutputPoints;
use crate::VerifierError;

pub(super) fn public_error(stage: FieldInlineRelationId, error: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{stage:?}"),
        reason: error.to_string(),
    }
}

/// The variables past the first `prefix_len` of a field-inline `address ++ cycle` opening
/// point (the field-inline cycle sub-point).
pub(super) fn point_suffix<F: JoltField>(
    point: &[F],
    prefix_len: usize,
    stage: FieldInlineRelationId,
) -> Result<&[F], VerifierError> {
    point.get(prefix_len..).ok_or_else(|| {
        public_error(
            stage,
            format!(
                "opening point is too short: expected at least {prefix_len} variables, got {}",
                point.len()
            ),
        )
    })
}

/// The five field value/product openings following the common stage-1 columns.
pub(super) fn stage1_appended_opening_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_spartan_geometry::outer_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-2 field-inline claim-reduction member and its baked publics: its `EqSpartan` is
/// the same `Eq(reduced point, tau_low)` derivation as the instruction reduction (same rounds,
/// same batch suffix, same reversed opening point — pinned in stage2's clear tests); its gamma
/// is the drawn batch challenge.
pub(super) fn stage2_claim_reduction<F: JoltField, C>(
    values: &mut SourceValues<F>,
    log_t: usize,
    batch_consistency: &BatchedCommittedSumcheckConsistency<F, C>,
    gamma: F,
    product_tau_low: &[F],
) -> Result<field_registers_reduction::ClaimReduction, VerifierError> {
    let reduction =
        field_registers_reduction::ClaimReduction::new(FieldRegistersTraceDimensions::new(log_t));
    let reduction_point = batch_consistency
        .try_instance_point(reduction.rounds())
        .map_err(|error| {
            public_error(FieldInlineRelationId::FieldRegistersClaimReduction, error)
        })?;
    let reduction_opening_point = reduction_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        FieldInlineChallengeId::from(FieldRegistersClaimReductionChallenge::Gamma),
        gamma,
    )?;
    values.public(
        FieldInlineDerivedId::from(FieldRegistersClaimReductionPublic::EqSpartan),
        try_eq_mle(&reduction_opening_point, product_tau_low).map_err(|error| {
            public_error(FieldInlineRelationId::FieldRegistersClaimReduction, error)
        })?,
    )?;
    Ok(reduction)
}

/// The field-inline portion of the product member's canonical output rows.
pub(super) fn stage2_product_opening_ids() -> impl Iterator<Item = VerifierOpeningId> {
    jolt_claims::protocols::field_inline::geometry::product::selected_product_remainder_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-4 field-register read/write member and its baked publics: shape from the
/// compile-time protocol config, gamma from the drawn batch, `EqCycle` mirroring the ordinary
/// registers derivation — `Eq(upstream field-inline reduced cycle point, own cycle sub-point
/// past the field-register address prefix)`.
pub(super) fn stage4_read_write<F: JoltField>(
    values: &mut SourceValues<F>,
    log_t: usize,
    gamma: F,
    fixed_cycle: &[F],
    read_write_point: &[F],
) -> Result<field_registers::ReadWriteChecking, VerifierError> {
    let field_inline_dimensions = JOLT_VERIFIER_CONFIG
        .field_inline
        .read_write_dimensions(log_t);
    let claims = field_registers::ReadWriteChecking::new(field_inline_dimensions);
    values.public(
        FieldInlineChallengeId::from(FieldRegistersReadWriteChallenge::Gamma),
        gamma,
    )?;
    let own_cycle = point_suffix(
        read_write_point,
        field_inline_dimensions.log_k(),
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )?;
    values.public(
        FieldInlineDerivedId::from(FieldRegistersReadWritePublic::EqCycle),
        try_eq_mle(fixed_cycle, own_cycle).map_err(|error| {
            public_error(
                FieldInlineRelationId::FieldRegistersReadWriteChecking,
                error,
            )
        })?,
    )?;
    Ok(claims)
}

/// The five field-register read/write rows, spliced after the register openings and before
/// `ram_ra`/`ram_inc` — the clear absorb order.
pub(super) fn stage4_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_registers_geometry::read_write_checking_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-5 field-register value-evaluation member (declared last, no instance challenge)
/// and its baked `LtCycle` public: `Lt(own cycle sub-point, upstream field-register read/write
/// cycle sub-point)` over the field-register address prefix.
pub(super) fn stage5_val_evaluation<F: JoltField>(
    values: &mut SourceValues<F>,
    log_t: usize,
    val_evaluation_point: &[F],
    read_write_point: &[F],
) -> Result<field_registers::ValEvaluation, VerifierError> {
    let claims = field_registers::ValEvaluation::new(FieldRegistersTraceDimensions::new(log_t));
    let own_cycle = point_suffix(
        val_evaluation_point,
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )?;
    let upstream_cycle = point_suffix(
        read_write_point,
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )?;
    values.public(
        FieldInlineDerivedId::from(FieldRegistersValEvaluationPublic::LtCycle),
        LtPolynomial::evaluate(own_cycle, upstream_cycle),
    )?;
    Ok(claims)
}

/// The two field-register value-evaluation rows, after the ordinary register value-evaluation
/// outputs — the clear absorb order (the field-inline member is declared last, so the
/// generated absorb appends them at the tail).
pub(super) fn stage5_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_registers_geometry::val_evaluation_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// Derive field-register accesses from the bytecode and add their stage-value
/// contributions onto the ordinary staged bytecode publics BEFORE they bake, so the same
/// `StageValue(i)` publics the symbolic output expression references carry both families —
/// exactly the clear composed relation's public composition.
pub(super) fn extend_bytecode_stage_values<F: JoltField, PCS: CommitmentScheme>(
    stage_values: &mut [F; 5],
    program: &ProgramPreprocessing<PCS>,
    r_address: &[F],
    r_cycle: &[F],
    read_write_point: &[F],
    val_evaluation_point: &[F],
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> Result<(), VerifierError> {
    let bytecode = &program
        .as_full()
        .ok_or_else(crate::stages::stage6b::field_inline::committed_program_rejection)?
        .bytecode
        .bytecode;
    let field_inline_stage_values = composed_bytecode_stage_values(
        bytecode,
        r_address,
        r_cycle,
        read_write_point,
        val_evaluation_point,
        challenges,
    )?;
    for (stage_value, field_inline_value) in stage_values.iter_mut().zip(field_inline_stage_values)
    {
        *stage_value += field_inline_value;
    }
    Ok(())
}

/// Evaluate the field-register access contributions with the same bytecode and
/// opening-point geometry used by the clear verifier.
pub(super) fn composed_bytecode_stage_values<F: JoltField>(
    bytecode: &[JoltInstructionRow],
    r_address: &[F],
    r_cycle: &[F],
    field_read_write_point: &[F],
    field_val_evaluation_point: &[F],
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> Result<[F; 5], VerifierError> {
    use crate::stages::field_inline_bytecode::field_inline_checked_split;
    use jolt_claims::protocols::field_inline::geometry::bytecode as field_inline_bytecode;

    let (read_write_address, read_write_cycle) = field_inline_checked_split(
        "BlindFold stage4 field-register read-write opening",
        field_read_write_point,
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )?;
    let (val_evaluation_address, val_evaluation_cycle) = field_inline_checked_split(
        "BlindFold stage5 field-register val-evaluation opening",
        field_val_evaluation_point,
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )?;
    let gammas = field_inline_stage_gamma_powers(challenges);
    let public_values = field_inline_bytecode::read_raf_public_values(
        field_inline_bytecode::FieldInlineBytecodeReadRafEvaluationInputs {
            bytecode,
            r_address,
            r_cycle,
            field_register_read_write_point: read_write_address,
            field_register_read_write_cycle_point: read_write_cycle,
            field_register_val_evaluation_point: val_evaluation_address,
            field_register_val_evaluation_cycle_point: val_evaluation_cycle,
            stage4_gammas: &gammas.stage4,
            stage5_gammas: &gammas.stage5,
        },
    )
    .map_err(|error| public_error(FieldInlineRelationId::FieldRegistersSpartanOuter, error))?;
    Ok(public_values.stage_values)
}

/// The stage-6b field-register increment-reduction member's symbolic relation.
pub(super) fn stage6b_inc_relation(log_t: usize) -> field_increments::ClaimReduction {
    field_increments::ClaimReduction::new(FieldRegistersTraceDimensions::new(log_t))
}

/// The field-register increment reduction's publics and challenge. It is trace-domain with the
/// same suffix window as the ordinary increment reduction, so its reduced opening point is the
/// SAME `inc_opening_point`; the Eq publics mirror the ordinary member's derivations over the
/// stage-4/5 field-inline cycle sub-points (past the field-register address prefix).
pub(super) fn stage6b_inc_publics<F: JoltField>(
    values: &mut SourceValues<F>,
    inc_opening_point: &[F],
    gamma: F,
    stage4_points: &Stage4OutputPoints<F>,
    stage5_points: &Stage5OutputPoints<F>,
) -> Result<(), VerifierError> {
    values.public(
        FieldInlineChallengeId::from(FieldRegistersIncClaimReductionChallenge::Gamma),
        gamma,
    )?;
    let read_write_cycle = point_suffix(
        stage4_points.field_registers_read_write_point(),
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersIncClaimReduction,
    )?;
    let val_evaluation_cycle = point_suffix(
        stage5_points.field_registers_val_evaluation_point(),
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersIncClaimReduction,
    )?;
    values.public(
        FieldInlineDerivedId::from(FieldRegistersIncClaimReductionPublic::EqReadWrite),
        try_eq_mle(inc_opening_point, read_write_cycle).map_err(|error| {
            public_error(
                FieldInlineRelationId::FieldRegistersIncClaimReduction,
                error,
            )
        })?,
    )?;
    values.public(
        FieldInlineDerivedId::from(FieldRegistersIncClaimReductionPublic::EqValEvaluation),
        try_eq_mle(inc_opening_point, val_evaluation_cycle).map_err(|error| {
            public_error(
                FieldInlineRelationId::FieldRegistersIncClaimReduction,
                error,
            )
        })?,
    )?;
    Ok(())
}

/// The reduced field-inline `FieldRdInc` row, after the ordinary increment-reduction outputs
/// and before the optional advice cycle phases — the clear absorb order
/// (`stage6b_opening_values`).
pub(super) fn stage6b_inc_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_claim_reductions::increments::claim_reduction_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}
