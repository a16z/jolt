//! Field-register claim-reduction wiring for the stage-2 batch.
use super::field_registers_claim_reduction::{
    FieldRegistersClaimReduction, FieldRegistersClaimReductionInputClaims,
};
use crate::stages::stage1::Stage1ClearOutput;
use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
use jolt_field::JoltField;

/// The stage-2 FR batch member. The FR claim reduction shares the trace domain
/// (`log_T` rounds) with the product remainder, so both bind the same batch
/// suffix — the spec's `r_prod` sharing.
pub fn claim_reduction_member<F: JoltField>(
    log_t: usize,
    tau_low: Vec<F>,
) -> FieldRegistersClaimReduction<F> {
    FieldRegistersClaimReduction::new(FieldRegistersTraceDimensions::new(log_t), tau_low)
}

/// Wire the consumed FR value opening *values* from stage 1's composed outer
/// sumcheck. The composed carrier requires these values structurally.
pub fn claim_reduction_inputs<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
) -> FieldRegistersClaimReductionInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder.field_inline;
    FieldRegistersClaimReductionInputClaims {
        rd_value: outer.rd_value,
        rs1_value: outer.rs1_value,
        rs2_value: outer.rs2_value,
    }
}
