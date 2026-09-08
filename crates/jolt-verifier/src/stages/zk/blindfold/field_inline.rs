//! The BlindFold lowering's field-inline seam: every FR-specific piece of the
//! ZK verifier's R1CS build in one place — the FR members' symbolic relations
//! and baked publics per stage, the composed bytecode public extension, the FR
//! output-row splices, and the FR-lane expression terms. Each blindfold stage
//! file keeps exactly one contiguous, flagged region per interaction point,
//! calling into here.

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
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use super::{SourceValues, VerifierOpeningId};
use crate::config::JOLT_VERIFIER_CONFIG;
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::field_inline_bytecode::{
    field_inline_stage_gamma_powers, FieldInlineBytecodeTable,
};
use crate::stages::stage4::Stage4OutputPoints;
use crate::stages::stage5::Stage5OutputPoints;
use crate::VerifierError;

pub(super) fn public_error(stage: FieldInlineRelationId, error: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{stage:?}"),
        reason: error.to_string(),
    }
}

/// The variables past the first `prefix_len` of an FR `address ++ cycle`
/// opening point (the FR cycle sub-point).
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

/// The 13 FR-local Spartan-outer rows appended after the 35 ordinary stage-1
/// columns, in appended-column order — the clear absorb/commit order.
pub(super) fn stage1_appended_opening_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_spartan_geometry::outer_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-2 FR claim-reduction member and its baked publics: its `EqSpartan`
/// is the same `Eq(reduced point, tau_low)` derivation as the instruction
/// reduction (same rounds, same batch suffix, same reversed opening point —
/// pinned in stage2's clear tests); its gamma is the drawn batch challenge.
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

/// The FR portion of the product member's canonical output rows.
pub(super) fn stage2_product_opening_ids() -> impl Iterator<Item = VerifierOpeningId> {
    jolt_claims::protocols::field_inline::geometry::product::selected_product_remainder_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-4 FR read/write member and its baked publics: shape from the
/// compile-time protocol config, gamma from the drawn batch, `EqCycle`
/// mirroring the ordinary registers derivation — `Eq(upstream FR reduced cycle
/// point, own cycle sub-point past the FR address prefix)`.
pub(super) fn stage4_read_write<F: JoltField>(
    values: &mut SourceValues<F>,
    log_t: usize,
    gamma: F,
    fixed_cycle: &[F],
    read_write_point: &[F],
) -> Result<field_registers::ReadWriteChecking, VerifierError> {
    let fr_dimensions = JOLT_VERIFIER_CONFIG
        .field_inline
        .read_write_dimensions(log_t);
    let claims = field_registers::ReadWriteChecking::new(fr_dimensions);
    values.public(
        FieldInlineChallengeId::from(FieldRegistersReadWriteChallenge::Gamma),
        gamma,
    )?;
    let own_cycle = point_suffix(
        read_write_point,
        fr_dimensions.log_k(),
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

/// The five FR read/write rows, spliced after the register openings and
/// before `ram_ra`/`ram_inc` — the clear absorb order.
pub(super) fn stage4_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_registers_geometry::read_write_checking_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// The stage-5 FR val-evaluation member (declared last, no instance
/// challenge) and its baked `LtCycle` public: `Lt(own cycle sub-point,
/// upstream FR read/write cycle sub-point)` over the FR address prefix.
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

/// The two FR val-evaluation rows, after the ordinary register
/// value-evaluation outputs — the clear absorb order (the FR member is
/// declared last, so the generated absorb appends them at the tail).
pub(super) fn stage5_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_registers_geometry::val_evaluation_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}

/// Load the preprocessed FR side table and add its composed stage-value
/// contributions onto the ordinary staged bytecode publics BEFORE they bake,
/// so the same `StageValue(i)` publics the symbolic output expression
/// references carry both families — exactly the clear composed relation's
/// public composition.
#[expect(
    clippy::too_many_arguments,
    reason = "the extension is a pure function of the bytecode bind points and the stage-4/5 FR opening points; bundling them would only rename the seam"
)]
pub(super) fn extend_bytecode_stage_values<F: JoltField, PCS: CommitmentScheme>(
    stage_values: &mut [F; 5],
    program: &ProgramPreprocessing<PCS>,
    r_address: &[F],
    r_cycle: &[F],
    stage1_cycle_point: &[F],
    read_write_point: &[F],
    val_evaluation_point: &[F],
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> Result<(), VerifierError> {
    let table = crate::stages::field_inline_bytecode::convert_field_inline_bytecode(
        crate::stages::field_inline_bytecode::required_field_inline_bytecode(program)?,
    )?;
    let field_inline_stage_values = composed_bytecode_stage_values(
        &table,
        r_address,
        r_cycle,
        stage1_cycle_point,
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

/// The FR side-table public stage-value contributions at
/// `(r_address, r_cycle)`: the converted rows folded under the FR-extended
/// stage-1/4/5 gamma powers, each stage weighted by its own cycle-eq factor —
/// the same `read_raf_public_values` evaluation (over the same point splits)
/// the clear composed relation performs, so the composed `StageValue(i)`
/// publics cannot drift from the clear check.
pub(super) fn composed_bytecode_stage_values<F: JoltField>(
    table: &FieldInlineBytecodeTable,
    r_address: &[F],
    r_cycle: &[F],
    stage1_cycle_point: &[F],
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
            bytecode: &table.rows,
            field_register_log_k: table.field_register_log_k,
            r_address,
            r_cycle,
            stage1_cycle_point,
            field_register_read_write_point: read_write_address,
            field_register_read_write_cycle_point: read_write_cycle,
            field_register_val_evaluation_point: val_evaluation_address,
            field_register_val_evaluation_cycle_point: val_evaluation_cycle,
            stage1_gammas: &gammas.stage1,
            stage4_gammas: &gammas.stage4,
            stage5_gammas: &gammas.stage5,
        },
    )
    .map_err(|error| public_error(FieldInlineRelationId::FieldRegistersSpartanOuter, error))?;
    Ok(public_values.stage_values)
}

/// The stage-6b FR increment-reduction member's symbolic relation.
pub(super) fn stage6b_inc_relation(log_t: usize) -> field_increments::ClaimReduction {
    field_increments::ClaimReduction::new(FieldRegistersTraceDimensions::new(log_t))
}

/// The FR increment reduction's publics and challenge. It is trace-domain with
/// the same suffix window as the ordinary increment reduction, so its reduced
/// opening point is the SAME `inc_opening_point`; the Eq publics mirror the
/// ordinary member's derivations over the stage-4/5 FR cycle sub-points (past
/// the FR address prefix).
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

/// The reduced FR `FieldRdInc` row, after the ordinary increment-reduction
/// outputs and before the optional advice cycle phases — the clear absorb
/// order (`stage6b_opening_values`).
pub(super) fn stage6b_inc_output_ids() -> impl Iterator<Item = VerifierOpeningId> {
    field_claim_reductions::increments::claim_reduction_output_openings()
        .into_iter()
        .map(VerifierOpeningId::from)
}
