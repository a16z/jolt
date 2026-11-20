use jolt_claims::protocols::field_inline::geometry::bytecode::{
    FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT, FIELD_INLINE_BYTECODE_STAGE5_GAMMA_COUNT,
};
use jolt_claims::protocols::field_inline::FieldInlineRelationId;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_field::Field;

use crate::VerifierError;

/// The field-inline-extended per-stage gamma power vectors for the bytecode read-RAF folds.
/// Extends the ordinary stage-4/5 power sequences (the same drawn scalars, more powers — no
/// new Fiat-Shamir draws) to the field-inline counts; field op flags use the ordinary stage-1 fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeStageGammas<F> {
    pub stage4: [F; FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT],
    pub stage5: [F; FIELD_INLINE_BYTECODE_STAGE5_GAMMA_COUNT],
}

/// Expand the carried stage-4/5 scalars into the field-inline-extended power vectors (`[1,
/// γ, γ², …]`, sized by the field-inline gamma counts).
pub fn field_inline_stage_gamma_powers<F: Field>(
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> FieldInlineBytecodeStageGammas<F> {
    FieldInlineBytecodeStageGammas {
        stage4: gamma_powers(challenges.stage4_gamma),
        stage5: gamma_powers(challenges.stage5_gamma),
    }
}

fn gamma_powers<F: Field, const N: usize>(gamma: F) -> [F; N] {
    let mut power = F::one();
    std::array::from_fn(|_| {
        let current = power;
        power *= gamma;
        current
    })
}

/// Field-register points and batching powers for bytecode read-RAF. Full mode
/// evaluates the decoded rows. Committed Akita obtains the corresponding address
/// evaluations from the two staged openings. Each uses its own cycle equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeFold<F> {
    /// The `FIELD_REGISTERS_LOG_K`-variable address prefix of the stage-4 field-inline
    /// read-write opening point.
    pub read_write_address: Vec<F>,
    /// The cycle suffix of the stage-4 field-register read-write opening point.
    pub read_write_cycle: Vec<F>,
    /// The `FIELD_REGISTERS_LOG_K`-variable address prefix of the stage-5 field-inline
    /// val-evaluation opening point.
    pub val_evaluation_address: Vec<F>,
    /// The cycle suffix of the stage-5 field-register value-evaluation opening point.
    pub val_evaluation_cycle: Vec<F>,
    pub gammas: FieldInlineBytecodeStageGammas<F>,
}

/// [`crate::stages::stage6_checked_split`] for field-inline opening points, attributing the
/// failure to the field-inline relation consuming the split.
pub(crate) fn field_inline_checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    stage: FieldInlineRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() < split_at {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: format!("{stage:?}"),
            reason: format!(
                "{label} has {} variables, expected at least {split_at}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}
