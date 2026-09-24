//! Stage 5's field-inline seam: every field-inline-specific divergence of the stage-5 verifier
//! in one place — the field-register value-evaluation batch member and its input wiring from
//! stage 4's field-register read/write checking. `verify.rs` interacts with the field-inline
//! protocol only through the functions here (plus the field-inline carrier fields, which are
//! proof shape).

use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
use jolt_field::JoltField;

use super::field_registers_val_evaluation::{
    FieldRegistersValEvaluation, FieldRegistersValEvaluationInputClaims,
};
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};

/// The stage-5 field-inline batch member (declared last in the batch; it draws no instance
/// challenge, so composing it changes no stage-5 draw).
pub fn val_evaluation_member<F: JoltField>(log_t: usize) -> FieldRegistersValEvaluation<F> {
    FieldRegistersValEvaluation::new(FieldRegistersTraceDimensions::new(log_t))
}

/// Wire the consumed `FieldRegistersVal` opening *value* from the upstream field-inline
/// read-write checking (stage 4). The upstream cell is a plain (non-optional) field of the
/// field-inline stage-4 claims, so presence is a compile-time fact.
pub fn val_evaluation_inputs<F: JoltField>(
    stage4: &Stage4OutputClaims<F>,
) -> FieldRegistersValEvaluationInputClaims<F> {
    FieldRegistersValEvaluationInputClaims {
        registers_val: stage4.field_registers_read_write.registers_val,
    }
}

/// Wire the consumed `FieldRegistersVal` opening *point* from the upstream field-inline
/// read-write checking (stage 4).
pub fn val_evaluation_input_points<F: JoltField>(
    stage4: &Stage4OutputPoints<F>,
) -> FieldRegistersValEvaluationInputClaims<Vec<F>> {
    FieldRegistersValEvaluationInputClaims {
        registers_val: stage4.field_registers_read_write_point().to_vec(),
    }
}
