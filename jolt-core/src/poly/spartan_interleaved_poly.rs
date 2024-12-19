use super::{
    dense_interleaved_poly::DenseInterleavedPolynomial, dense_mlpoly::DensePolynomial,
    multilinear_polynomial::MultilinearPolynomial, sparse_interleaved_poly::SparseCoefficient,
    split_eq_poly::SplitEqPolynomial, unipoly::UniPoly,
};
use crate::{
    field::{JoltField, OptimizedMul},
    r1cs::builder::{eval_offset_lc, CombinedUniformBuilder, Constraint, OffsetEqConstraint},
    subprotocols::{
        grand_product::BatchedGrandProductLayer,
        sumcheck::{BatchedCubicSumcheck, Bindable},
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_ff::Zero;
use rayon::prelude::*;

#[derive(Default, Debug, Clone)]
pub struct SpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck.
    pub(crate) unbound_coeffs: Vec<SparseCoefficient<i128>>,
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    /// A reused buffer where bound values are written to during `bind`.
    /// With every bind, `coeffs` and `binding_scratch_space` are swapped.
    binding_scratch_space: Vec<SparseCoefficient<F>>,
    /// The length of the layer if it were represented by a single dense vector.
    pub(crate) dense_len: usize,
}

impl<F: JoltField> SpartanInterleavedPolynomial<F> {
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::new")]
    pub fn new(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>], // N variables of (S steps)
        padded_num_constraints: usize,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();
        let dense_len = padded_num_constraints * num_steps;

        let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
        let chunk_size = num_steps.div_ceil(num_chunks);

        let unbound_coeffs: Vec<SparseCoefficient<i128>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut coeffs = Vec::with_capacity(chunk_size * padded_num_constraints * 3);
                for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                    // Uniform constraints
                    for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                        let global_index = step_index * padded_num_constraints + constraint_index;

                        // Az
                        let mut az_coeff = 0;
                        if !constraint.a.terms().is_empty() {
                            az_coeff = constraint
                                .a
                                .evaluate_row_i128(flattened_polynomials, step_index);
                            if !az_coeff.is_zero() {
                                coeffs.push((global_index, az_coeff).into());
                            }
                        }
                        // Bz
                        let mut bz_coeff = 0;
                        if !constraint.b.terms().is_empty() {
                            bz_coeff = constraint
                                .b
                                .evaluate_row_i128(flattened_polynomials, step_index);
                            if !bz_coeff.is_zero() {
                                coeffs.push((global_index, bz_coeff).into());
                            }
                        }
                        // Cz = Az ⊙ Cz
                        if !az_coeff.is_zero() && !bz_coeff.is_zero() {
                            let cz_coeff = az_coeff * bz_coeff;
                            coeffs.push((global_index, cz_coeff).into());
                        }
                    }

                    // Cross-step constraints
                    for (constraint_index, constraint) in cross_step_constraints.iter().enumerate()
                    {
                        let global_index = step_index * padded_num_constraints
                            + uniform_constraints.len()
                            + constraint_index;
                        // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
                        let next_step_index = if step_index + 1 < num_steps {
                            Some(step_index + 1)
                        } else {
                            None
                        };

                        // Az
                        let eq_a_eval = eval_offset_lc(
                            &constraint.a,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let eq_b_eval = eval_offset_lc(
                            &constraint.b,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let coeff = eq_a_eval - eq_b_eval;
                        if !coeff.is_zero() {
                            coeffs.push((global_index, coeff).into());
                            // If Az != 0, then the condition must be false (i.e. Bz = 0)
                        } else {
                            // Bz
                            let condition_eval = eval_offset_lc(
                                &constraint.cond,
                                flattened_polynomials,
                                step_index,
                                next_step_index,
                            );
                            if !condition_eval.is_zero() {
                                coeffs.push((global_index, condition_eval).into());
                            }
                        }
                        // Cz is always 0 for cross-step constraints
                    }
                }

                coeffs
            })
            .collect();

        Self {
            unbound_coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            dense_len,
        }
    }
}
