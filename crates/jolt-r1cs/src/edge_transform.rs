//! Combined partial evaluation for the Spartan inner sumcheck.
//!
//! This is the "edge transform" between the outer and inner sumcheck:
//! after the outer sumcheck produces challenge $r_x$, the prover materializes
//! the combined row polynomial $M(r_x, \cdot)$ as a dense polynomial over
//! column indices. The inner sumcheck then proves $\sum_y M(r_x, y) \cdot z(y)$.

use jolt_field::Field;
use jolt_poly::{EqPolynomial, Polynomial};

use crate::uniform_key::UniformR1csKey;

/// Materializes the combined row polynomial $M(r_x, \cdot)$ for the inner sumcheck
/// using the uniform key's sparse structure.
///
/// $$M(r_x, y) = \rho_A \cdot A(r_x, y) + \rho_B \cdot B(r_x, y) + \rho_C \cdot C(r_x, y)$$
///
/// For a uniform R1CS, $r_x = (r_{\text{cycle}}, r_{\text{constraint}})$ and the matrix factors
/// as $M(r_x, y) = \widetilde{eq}(r_{\text{cycle}}, y_{\text{cycle}}) \cdot M_{\text{local}}(r_{\text{constraint}}, y_{\text{var}})$.
///
/// The result is a dense polynomial over all `total_cols_padded` column indices.
pub fn combined_partial_evaluate<F: Field>(
    key: &UniformR1csKey<F>,
    r_x: &[F],
    rho_a: F,
    rho_b: F,
    rho_c: F,
    total_cols_padded: usize,
) -> Polynomial<F> {
    let cycle_vars = key.num_cycle_vars();
    let (r_x_cycle, r_x_constraint) = r_x.split_at(cycle_vars);

    let eq_constraint = EqPolynomial::new(r_x_constraint.to_vec()).evaluations();
    let eq_cycle = EqPolynomial::new(r_x_cycle.to_vec()).evaluations();

    // Build the combined local row: M_local(r_constraint, v) for each variable v
    let mut local_row = vec![F::zero(); key.num_vars_padded];
    for (k, (a_row, (b_row, c_row))) in key
        .a_sparse
        .iter()
        .zip(key.b_sparse.iter().zip(key.c_sparse.iter()))
        .enumerate()
    {
        let w = eq_constraint[k];
        if w.is_zero() {
            continue;
        }

        for &(j, coeff) in a_row {
            local_row[j] += w * rho_a * coeff;
        }
        for &(j, coeff) in b_row {
            local_row[j] += w * rho_b * coeff;
        }
        for &(j, coeff) in c_row {
            local_row[j] += w * rho_c * coeff;
        }
    }

    // Expand to full column space: M(r_x, y) = eq(r_cycle, y_cycle) * local_row[y_var]
    let mut combined = vec![F::zero(); total_cols_padded];
    for (c, &eq_c) in eq_cycle.iter().enumerate() {
        if eq_c.is_zero() {
            continue;
        }
        let base = c * key.num_vars_padded;
        for (v, &local_val) in local_row.iter().enumerate() {
            if !local_val.is_zero() {
                combined[base + v] = eq_c * local_val;
            }
        }
    }

    Polynomial::new(combined)
}
