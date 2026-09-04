//! Stage 4's field-inline seam: every FR-specific divergence of the stage-4
//! verifier in one place — the FR read/write batch member, its input wiring
//! from stage 2's FR claim reduction, and the curated absorb splice.
//! `verify.rs`/`outputs.rs` interact with the FR protocol only through the
//! functions here (plus the FR carrier fields, which are proof shape).

use jolt_field::JoltField;

use super::field_registers_read_write_checking::{
    FieldRegistersReadWriteChecking, FieldRegistersReadWriteInputClaims,
};
use super::outputs::Stage4OutputClaims;
use crate::config::JOLT_VERIFIER_CONFIG;
use crate::stages::relations::OutputClaims as _;
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};

/// The stage-4 FR batch member. FR dimensions are pinned by the compile-time
/// protocol config (phase1 = log_t, phase2 = log_k), not the proof's
/// rw_config, so no eager phase-split validation is needed.
pub fn read_write_member<F: JoltField>(log_t: usize) -> FieldRegistersReadWriteChecking<F> {
    FieldRegistersReadWriteChecking::new(
        JOLT_VERIFIER_CONFIG
            .field_inline
            .read_write_dimensions(log_t),
    )
}

/// Wire the consumed FR value opening *values* from stage 2's FR claim
/// reduction. The upstream cells are plain (non-optional) fields of the FR-on
/// stage-2 batch claims, so presence is a compile-time fact — an FR-on proof
/// without them fails proof deserialization / shape validation upstream.
pub fn read_write_inputs<F: JoltField>(
    stage2: &Stage2BatchOutputClaims<F>,
) -> FieldRegistersReadWriteInputClaims<F> {
    let reduction = &stage2.field_registers_claim_reduction;
    FieldRegistersReadWriteInputClaims {
        rd_value: reduction.rd_value,
        rs1_value: reduction.rs1_value,
        rs2_value: reduction.rs2_value,
    }
}

/// Wire the consumed FR opening *points* from stage 2's FR claim reduction,
/// all sharing that relation's reduced opening point (`r_prod`).
pub fn read_write_input_points<F: JoltField>(
    stage2: &Stage2BatchOutputPoints<F>,
) -> FieldRegistersReadWriteInputClaims<Vec<F>> {
    let reduction = &stage2.field_registers_claim_reduction;
    FieldRegistersReadWriteInputClaims {
        rd_value: reduction.rd_value().to_vec(),
        rs1_value: reduction.rs1_value().to_vec(),
        rs2_value: reduction.rs2_value().to_vec(),
    }
}

/// Splice the five FR read/write openings into the stage-4 Fiat-Shamir value
/// order: after the ordinary register openings, before the RAM value-check
/// ones (the spec's committed row order, `specs/field-inline-protocol.md`,
/// "Stage 4 Composition").
pub(super) fn splice_read_write_values<F: JoltField>(
    values: &mut Vec<F>,
    claims: &Stage4OutputClaims<F>,
) {
    values.extend(claims.field_registers_read_write.opening_values());
}
