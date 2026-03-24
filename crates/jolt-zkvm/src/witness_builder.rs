//! Witness utilities for sumcheck vertices.

use jolt_field::Field;

/// Evaluates a multilinear polynomial at a point.
pub fn eval_poly<F: Field>(table: &[F], point: &[F]) -> F {
    jolt_poly::Polynomial::new(table.to_vec()).evaluate(point)
}
