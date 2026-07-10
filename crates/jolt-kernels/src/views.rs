//! Shared witness-view and table helpers for the per-relation kernels.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_witness::protocols::jolt_vm::{jolt_opening_oracle_ref, JoltVmNamespace};
use jolt_witness::{
    MaterializationPolicy, PolynomialEncoding, RetentionHint, ViewRequirement, WitnessProvider,
};

use crate::KernelError;

/// Materialize a dense field-element table of the oracle behind `opening`.
pub(crate) fn dense_view<F: Field>(
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    opening: JoltOpeningId,
) -> Result<Vec<F>, KernelError<F>> {
    let oracle = jolt_opening_oracle_ref(opening)?;
    let view = witness.oracle_view(ViewRequirement {
        oracle,
        encoding: PolynomialEncoding::Dense,
        materialization: MaterializationPolicy::BackendChoice,
        retention: RetentionHint::Ephemeral,
    })?;
    Ok(view
        .as_slice()
        .ok_or(KernelError::Unsupported {
            reason: "oracle view was not materialized as a dense slice",
        })?
        .to_vec())
}

/// `eq(point, ·)` evaluations, big-endian (`point[0]` pairs the index MSB).
pub(crate) fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.to_vec()).evaluations()
}

/// Tile `base` `copies` times: the `(address ‖ cycle)`-indexed replication of a
/// cycle-indexed table across the address dimension (address bits are the high
/// bits of the joint index).
pub(crate) fn tile<F: Field>(base: &[F], copies: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(base.len() * copies);
    for _ in 0..copies {
        out.extend_from_slice(base);
    }
    out
}
