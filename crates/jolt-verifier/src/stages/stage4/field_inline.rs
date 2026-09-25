//! Field-inline openings in the stage-4 Fiat-Shamir absorption order.

use jolt_field::JoltField;

use super::outputs::Stage4OutputClaims;
use crate::stages::relations::OutputClaims as _;

/// Splice the five field-register read/write openings into the stage-4 Fiat-Shamir value
/// order: after the ordinary register openings, before the RAM value-check ones (the spec's
/// committed row order, `specs/field-inline-protocol.md`, "Stage 4 Composition").
pub(super) fn splice_read_write_values<F: JoltField>(
    values: &mut Vec<F>,
    claims: &Stage4OutputClaims<F>,
) {
    values.extend(claims.field_registers_read_write.opening_values());
}
