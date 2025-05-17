use super::{
    multilinear_polynomial::MultilinearPolynomial, sparse_interleaved_poly::SparseCoefficient,
    split_eq_poly::GruenSplitEqPolynomial, unipoly::CompressedUniPoly,
};
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
#[cfg(test)]
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
use crate::{
    field::{JoltField, OptimizedMul},
    r1cs::builder::{eval_offset_lc, Constraint, OffsetEqConstraint},
    utils::{math::Math, transcript::Transcript},
};
use ark_ff::Zero;
use rayon::prelude::*;

#[derive(Default, Debug, Clone)]
pub struct SpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Before the polynomial is bound
    /// the first time, all the coefficients can be represented by `i128`s.
    pub(crate) unbound_coeffs: Vec<SparseCoefficient<i128>>,
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Once the polynomial has been
    /// bound, we switch to using `bound_coeffs` instead of `unbound_coeffs`, because
    /// coefficients will be full-width field elements rather than `i128`s.
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    /// A reused buffer where bound values are written to during `bind`.
    /// With every bind, `coeffs` and `binding_scratch_space` are swapped.
    binding_scratch_space: Vec<SparseCoefficient<F>>,
    /// The length of one of the Az, Bz, or Cz polynomials if it were represented by
    /// a single dense vector.
    dense_len: usize,
}

impl<F: JoltField> SpartanInterleavedPolynomial<F> {
    /// Computes the matrix-vector products Az, Bz, and Cz as a single interleaved sparse vector
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::new")]
    pub fn new(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>], // N variables of (S steps)
        padded_num_constraints: usize,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
        let chunk_size = num_steps.div_ceil(num_chunks);

        let unbound_coeffs: Vec<SparseCoefficient<i128>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut coeffs = Vec::with_capacity(chunk_size * padded_num_constraints * 3);
                for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                    // Uniform constraints
                    for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                        let global_index =
                            3 * (step_index * padded_num_constraints + constraint_index);

                        // Az
                        let mut az_coeff = 0;
                        if !constraint.a.terms().is_empty() {
                            az_coeff = constraint
                                .a
                                .evaluate_row(flattened_polynomials, step_index);
                            if !az_coeff.is_zero() {
                                coeffs.push((global_index, az_coeff).into());
                            }
                        }
                        // Bz
                        let mut bz_coeff = 0;
                        if !constraint.b.terms().is_empty() {
                            bz_coeff = constraint
                                .b
                                .evaluate_row(flattened_polynomials, step_index);
                            if !bz_coeff.is_zero() {
                                coeffs.push((global_index + 1, bz_coeff).into());
                            }
                        }
                        // Cz = Az ⊙ Cz
                        if !az_coeff.is_zero() && !bz_coeff.is_zero() {
                            let cz_coeff = az_coeff * bz_coeff;
                            #[cfg(test)]
                            {
                                if cz_coeff != constraint
                                    .c
                                    .evaluate_row(flattened_polynomials, step_index) {
                                        let mut constraint_string = String::new();
                                        let _ = constraint
                                            .pretty_fmt::<4, JoltR1CSInputs, F>(
                                                &mut constraint_string,
                                                flattened_polynomials,
                                                step_index,
                                            );
                                        println!("{constraint_string}");
                                        panic!(
                                            "Uniform constraint {constraint_index} violated at step {step_index}",
                                        );
                                    }
                            }
                            coeffs.push((global_index + 2, cz_coeff).into());
                        }
                    }

                    // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
                    let next_step_index = if step_index + 1 < num_steps {
                        Some(step_index + 1)
                    } else {
                        None
                    };

                    // Cross-step constraints
                    for (constraint_index, constraint) in cross_step_constraints.iter().enumerate()
                    {
                        let global_index = 3
                            * (step_index * padded_num_constraints
                                + uniform_constraints.len()
                                + constraint_index);

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
                        let az_coeff = eq_a_eval - eq_b_eval;
                        if !az_coeff.is_zero() {
                            coeffs.push((global_index, az_coeff).into());
                            // If Az != 0, then the condition must be false (i.e. Bz = 0)
                            #[cfg(test)]
                            {
                                let bz_coeff = eval_offset_lc(
                                    &constraint.cond,
                                    flattened_polynomials,
                                    step_index,
                                    next_step_index,
                                );
                                assert_eq!(bz_coeff, 0, "Cross-step constraint {constraint_index} violated at step {step_index}");
                            }
                        } else {
                            // Bz
                            let bz_coeff = eval_offset_lc(
                                &constraint.cond,
                                flattened_polynomials,
                                step_index,
                                next_step_index,
                            );
                            if !bz_coeff.is_zero() {
                                coeffs.push((global_index + 1, bz_coeff).into());
                            }
                        }
                        // Cz is always 0 for cross-step constraints
                    }
                }

                coeffs
            })
            .collect();

        #[cfg(test)]
        {
            // Check that indices are monotonically increasing
            let mut prev_index = unbound_coeffs[0].index;
            for coeff in unbound_coeffs[1..].iter() {
                assert!(coeff.index > prev_index);
                prev_index = coeff.index;
            }
        }

        Self {
            unbound_coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            dense_len: num_steps * padded_num_constraints,
        }
    }

    #[cfg(test)]
    fn uninterleave(&self) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>) {
        let mut az = vec![F::zero(); self.dense_len];
        let mut bz = vec![F::zero(); self.dense_len];
        let mut cz = vec![F::zero(); self.dense_len];

        if !self.is_bound() {
            for coeff in &self.unbound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = F::from_i128(coeff.value),
                    1 => bz[coeff.index / 3] = F::from_i128(coeff.value),
                    2 => cz[coeff.index / 3] = F::from_i128(coeff.value),
                    _ => unreachable!(),
                }
            }
        } else {
            for coeff in &self.bound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = coeff.value,
                    1 => bz[coeff.index / 3] = coeff.value,
                    2 => cz[coeff.index / 3] = coeff.value,
                    _ => unreachable!(),
                }
            }
        }
        (
            DensePolynomial::new(az),
            DensePolynomial::new(bz),
            DensePolynomial::new(cz),
        )
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    /// The first round of the first Spartan sumcheck. Since the polynomials
    /// are still unbound at the beginning of this round, we can replace some
    /// of the field arithmetic with `i128` arithmetic.
    ///
    /// Note that we implement the extra optimization of only computing the quadratic
    /// evaluation at infinity, since the eval at zero is always zero.
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::first_sumcheck_round")]
    pub fn first_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        assert!(!self.is_bound());

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo 6 cannot
        // be processed independently.
        let block_size = self
            .unbound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .unbound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let num_x_in_bits = eq_poly.E_in_current_len().log_2();
        let x_in_bitmask = (1 << num_x_in_bits) - 1;

        // In the first round, we only need to compute the quadratic evaluation at infinity,
        // since the eval at zero is always zero.
        let quadratic_eval_at_infty = chunks
            .par_iter()
            .map(|chunk| {
                let mut eval_point_infty = F::zero();

                let mut inner_sums = F::zero();
                let mut prev_x_out = 0;

                for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = sparse_block[0].index / 6;
                    let x_in = block_index & x_in_bitmask;
                    let E_in_evals = eq_poly.E_in_current()[x_in];
                    let x_out = block_index >> num_x_in_bits;

                    if x_out != prev_x_out {
                        eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums;
                        inner_sums = F::zero();
                        prev_x_out = x_out;
                    }

                    // This holds the az0, az1, bz0, bz1 evals. No need for cz0, cz1 since we only need
                    // the eval at infinity.
                    let mut az0 = 0i128;
                    let mut az1 = 0i128;
                    let mut bz0 = 0i128;
                    let mut bz1 = 0i128;
                    for coeff in sparse_block {
                        let local_idx = coeff.index % 6;
                        if local_idx == 0 {
                            az0 = coeff.value;
                        } else if local_idx == 1 {
                            bz0 = coeff.value;
                        } else if local_idx == 3 {
                            az1 = coeff.value;
                        } else if local_idx == 4 {
                            bz1 = coeff.value;
                        }
                    }
                    let az_infty = az1 - az0;
                    let bz_infty = bz1 - bz0;
                    if az_infty != 0 && bz_infty != 0 {
                        inner_sums += E_in_evals.mul_i128(
                            az_infty.checked_mul(bz_infty).unwrap_or_else(|| {
                                panic!("az_infty * bz_infty overflow");
                            }),
                        );
                    }
                }
                eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums;
                eval_point_infty
            })
            .reduce(|| F::zero(), |sum, evals| sum + evals);

        let r_i = process_eq_sumcheck_round(
            (F::zero(), quadratic_eval_at_infty),
            eq_poly,
            polys,
            r,
            claim,
            transcript,
        );

        #[cfg(test)]
        let (mut az, mut bz, mut cz) = self.uninterleave();

        // Compute the number of non-zero bound coefficients that will be produced
        // per chunk.
        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        self.bound_coeffs = Vec::with_capacity(total_output_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.bound_coeffs.set_len(total_output_len);
        }
        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.bound_coeffs.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(unbound_coeffs, output_slice)| {
                let mut output_index = 0;
                for block in unbound_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<i128>, Option<i128>) = (None, None);
                    let mut bz_coeff: (Option<i128>, Option<i128>) = (None, None);
                    let mut cz_coeff: (Option<i128>, Option<i128>) = (None, None);

                    for coeff in block {
                        match coeff.index % 6 {
                            0 => az_coeff.0 = Some(coeff.value),
                            1 => bz_coeff.0 = Some(coeff.value),
                            2 => cz_coeff.0 = Some(coeff.value),
                            3 => az_coeff.1 = Some(coeff.value),
                            4 => bz_coeff.1 = Some(coeff.value),
                            5 => cz_coeff.1 = Some(coeff.value),
                            _ => unreachable!(),
                        }
                    }
                    if az_coeff != (None, None) {
                        let (low, high) = (az_coeff.0.unwrap_or(0), az_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 1,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 2,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len())
            });
        self.dense_len /= 2;

        #[cfg(test)]
        {
            // Check that the binding is consistent with binding Az, Bz, Cz individually
            let (az_bound, bz_bound, cz_bound) = self.uninterleave();
            az.bound_poly_var_bot(&r_i);
            bz.bound_poly_var_bot(&r_i);
            cz.bound_poly_var_bot(&r_i);
            for i in 0..az.len() {
                if az_bound[i] != az[i] {
                    println!("{i} {} != {}", az_bound[i], az[i]);
                }
            }
            assert!(az_bound.Z[..az_bound.len()] == az.Z[..az.len()]);
            assert!(bz_bound.Z[..bz_bound.len()] == bz.Z[..bz.len()]);
            assert!(cz_bound.Z[..cz_bound.len()] == cz.Z[..cz.len()]);
        }
    }

    /// All subsequent rounds of the first Spartan sumcheck.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::subsequent_sumcheck_round"
    )]
    pub fn subsequent_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        assert!(self.is_bound());

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo 6 cannot
        // be processed independently.
        let block_size = self
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let quadratic_evals = if eq_poly.E_in_current_len() == 1 {
            let evals = chunks
                .par_iter()
                .flat_map_iter(|chunk| {
                    chunk
                        .chunk_by(|x, y| x.index / 6 == y.index / 6)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 6;
                            let mut block = [F::zero(); 6];
                            for coeff in sparse_block {
                                block[coeff.index % 6] = coeff.value;
                            }

                            let az = (block[0], block[3]);
                            let bz = (block[1], block[4]);
                            let cz0 = block[2];

                            let az_eval_infty = az.1 - az.0;
                            let bz_eval_infty = bz.1 - bz.0;

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                                eq_evals
                                    .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        } else {
            let num_x_in_bits = eq_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            let evals = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x_out = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x_in = block_index & x_bitmask;
                        let E_in_eval = eq_poly.E_in_current()[x_in];
                        let x_out = block_index >> num_x_in_bits;

                        if x_out != prev_x_out {
                            let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                            eval_point_0 += E_out_eval * inner_sums.0;
                            eval_point_infty += E_out_eval * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x_out = x_out;
                        }

                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1 - az.0;
                        let bz_eval_infty = bz.1 - bz.0;

                        inner_sums.0 += E_in_eval.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 +=
                            E_in_eval.mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x_out] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        };

        let r_i = process_eq_sumcheck_round(quadratic_evals, eq_poly, polys, r, claim, transcript);

        #[cfg(test)]
        let (mut az, mut bz, mut cz) = self.uninterleave();

        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        if self.binding_scratch_space.is_empty() {
            self.binding_scratch_space = Vec::with_capacity(total_output_len);
        }
        unsafe {
            self.binding_scratch_space.set_len(total_output_len);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(coeffs, output_slice)| {
                let mut output_index = 0;
                for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                    let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                    let mut cz_coeff: (Option<F>, Option<F>) = (None, None);

                    for coeff in block {
                        match coeff.index % 6 {
                            0 => az_coeff.0 = Some(coeff.value),
                            1 => bz_coeff.0 = Some(coeff.value),
                            2 => cz_coeff.0 = Some(coeff.value),
                            3 => az_coeff.1 = Some(coeff.value),
                            4 => bz_coeff.1 = Some(coeff.value),
                            5 => cz_coeff.1 = Some(coeff.value),
                            _ => unreachable!(),
                        }
                    }
                    if az_coeff != (None, None) {
                        let (low, high) = (
                            az_coeff.0.unwrap_or(F::zero()),
                            az_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (
                            bz_coeff.0.unwrap_or(F::zero()),
                            bz_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index + 1, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (
                            cz_coeff.0.unwrap_or(F::zero()),
                            cz_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index + 2, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len())
            });

        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        self.dense_len /= 2;

        #[cfg(test)]
        {
            // Check that the binding is consistent with binding Az, Bz, Cz individually
            let (az_bound, bz_bound, cz_bound) = self.uninterleave();
            az.bound_poly_var_bot(&r_i);
            bz.bound_poly_var_bot(&r_i);
            cz.bound_poly_var_bot(&r_i);
            assert!(az_bound.Z[..az_bound.len()] == az.Z[..az.len()]);
            assert!(bz_bound.Z[..bz_bound.len()] == bz.Z[..bz.len()]);
            assert!(cz_bound.Z[..cz_bound.len()] == cz.Z[..cz.len()]);
        }
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients.
    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
            let mut Az_coeff_found = false;
            let mut Bz_coeff_found = false;
            let mut Cz_coeff_found = false;
            for coeff in block {
                match coeff.index % 3 {
                    0 => {
                        if !Az_coeff_found {
                            Az_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    1 => {
                        if !Bz_coeff_found {
                            Bz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    2 => {
                        if !Cz_coeff_found {
                            Cz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        output_size
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        let mut final_cz_eval = F::zero();
        for i in 0..3 {
            if let Some(coeff) = self.bound_coeffs.get(i) {
                match coeff.index {
                    0 => final_az_eval = coeff.value,
                    1 => final_bz_eval = coeff.value,
                    2 => final_cz_eval = coeff.value,
                    _ => {}
                }
            }
        }

        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}
