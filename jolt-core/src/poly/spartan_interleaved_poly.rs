#![allow(clippy::too_many_arguments)]
use super::{
    eq_poly::EqPolynomial, split_eq_poly::GruenSplitEqPolynomial, unipoly::CompressedUniPoly,
};
use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
use crate::{
    field::{JoltField, OptimizedMul},
    transcripts::Transcript,
    utils::small_value::accum::{s160_to_field, SignedUnreducedAccum, UnreducedProduct},
    utils::{math::Math, small_scalar::SmallScalar, small_value::svo_helpers},
    zkvm::r1cs::{
        constraints::{eval_az_bz_batch_from_row, CzKind, UNIFORM_R1CS},
        inputs::R1CSCycleInputs,
    },
    zkvm::JoltSharedPreprocessing,
};
use allocative::Allocative;
use ark_ff::biginteger::{I8OrI96, S160};
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Number of rounds to use for small value optimization.
/// Testing & estimation shows that 3 rounds is the best tradeoff
/// It may be 4 rounds when we switch to streaming / GPU proving
pub const NUM_SVO_ROUNDS: usize = 3;

pub const TOTAL_NUM_ACCUMS: usize = svo_helpers::total_num_accums(NUM_SVO_ROUNDS);
pub const NUM_NONTRIVIAL_TERNARY_POINTS: usize =
    svo_helpers::num_non_trivial_ternary_points(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_ZERO: usize = svo_helpers::num_accums_eval_zero(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_INFTY: usize = svo_helpers::num_accums_eval_infty(NUM_SVO_ROUNDS);

/// Number of Y-assignments per SVO block. Equal to 2^NUM_SVO_ROUNDS.
/// This is the size of the subspace over the prefix Y used in small-value optimization.
pub const Y_SVO_SPACE_SIZE: usize = 1 << NUM_SVO_ROUNDS;

/// Number of interleaved coefficients per logical block for a fixed (x_out, x_in):
///  - 2 polynomials (Az, Bz)
///  - 2 evaluations at x_next ∈ {0, 1}
///  - Y_SVO_SPACE_SIZE assignments of Y
///    So total = 4 * Y_SVO_SPACE_SIZE.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE: usize = 4 * Y_SVO_SPACE_SIZE;

/// Bit-width of a logical block. Computed as log2(4) + NUM_SVO_ROUNDS = 2 + NUM_SVO_ROUNDS.
/// Use this for fast block id calculation via shifts
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT: usize = 2 + NUM_SVO_ROUNDS;

/// Bitmask for the local offset within a block. Equals (1 << SHIFT) - 1.
/// Use this to extract local offsets via bit-and, instead of modulo.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_MASK: usize =
    (1usize << Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT) - 1;

// Modifications for streaming version:
// 1. Do not have the `unbound_coeffs` in the struct
// 2. Do streaming rounds until we get to a small enough cached size that we can store `bound_coeffs`

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient<T> {
    pub(crate) index: usize,
    pub(crate) value: T,
}

impl<T> Allocative for SparseCoefficient<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

#[derive(Clone, Debug, Allocative)]
pub struct SpartanInterleavedPolynomial<const NUM_SVO_ROUNDS: usize, F: JoltField> {
    /// A list of sparse vectors representing the (interleaved) coefficients for the Az, Bz polynomials
    /// Generated from binary evaluations. Each inner Vec is sorted by index.
    ///
    /// (note: **no** Cz coefficients are stored here, since they are not needed for small value
    /// precomputation, and can be computed on the fly in streaming round)
    // pub(crate) ab_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<F>>>,
    pub(crate) az_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<I8OrI96>>>,
    pub(crate) bz_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<S160>>>,

    /// The bound coefficients for the Az, Bz, Cz polynomials. Will be populated in the streaming round
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,

    padded_num_constraints: usize,
}

impl<const NUM_SVO_ROUNDS: usize, F: JoltField> SpartanInterleavedPolynomial<NUM_SVO_ROUNDS, F> {
    /// Compute the unbound coefficients for the Az and Bz polynomials (no Cz coefficients are
    /// needed), along with the accumulators for the small value optimization (SVO) rounds.
    ///
    /// Recall that the accumulators are of the form: accum_i[v_0, ..., v_{i-1}, u] = \sum_{y_rest}
    /// \sum_{x_out} E_out(x_out || y_rest) * \sum_{x_in} E_in(x_in) * P(x_out, x_in, y_rest, u,
    /// v_0, ..., v_{i-1}),
    ///
    /// for all i < NUM_SVO_ROUNDS, v_0,..., v_{i-1} \in {0,1,∞}, u \in {0,∞}, and P(X) = Az(X) *
    /// Bz(X) - Cz(X).
    ///
    /// Note that we have reverse the order of variables from the paper, since in this codebase the
    /// indexing is MSB to LSB (as we go from 0 to N-1, i.e. left to right).
    ///
    /// Note that only the accumulators with at least one infinity among v_j and u are non-zero, so
    /// the fully binary ones do not need to be computed. Plus, the ones with at least one infinity
    /// will NOT have any Cz contributions.
    ///
    /// This is why we do not need to compute the Cz terms in the unbound coefficients.
    ///
    /// The output of the accumulators is ([F; NUM_ACCUMS_EVAL_ZERO]; [F; NUM_ACCUMS_EVAL_INFTY]),
    /// where the outer array is for evals at u = 0 and u = ∞. The inner array contains all non-zero
    /// accumulators across all rounds, concatenated in order.
    ///
    /// For 1 round of small value optimization, this is:
    /// - Eval at zero: empty
    /// - Eval at infty: acc_1(infty)
    ///
    /// For 2 rounds of small value optimization, this is same as 1 round, with addition of:
    /// (recall: we do MSB => LSB, so 0/infty refers to the leftmost variable)
    /// - Eval at zero: acc_2(0, infty)
    /// - Eval at infty: acc_2(infty,0), acc_2(infty,1), acc_2(infty, infty)
    ///
    /// Total = 5 accumulators
    ///
    /// For 3 rounds of small value optimization, this is same as 2 rounds, with addition of:
    /// - Eval at zero: acc_3(0, 0, infty), acc_3(0, 1, infty),
    ///   acc_3(0, infty, 0), acc_3(0, infty, 1), acc_3(0, infty, infty)
    /// - Eval at infty: acc_3(infty, v_1, v_2), where v_1, v_2 \in {0, 1, infty}
    ///
    /// Total = 19 accumulators
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::new_with_precompute")]
    pub fn new_with_precompute(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        tau: &[F],
    ) -> ([F; NUM_ACCUMS_EVAL_ZERO], [F; NUM_ACCUMS_EVAL_INFTY], Self) {
        // Variable layout and binding order (MSB -> LSB):
        // 0 ... (N/2 - l) ... (n_s) ... (N - l) ... (N - i - 1) ... (N - 1)
        // where:
        //   n_s = number of step vars (log2(steps))
        //   N   = total num vars (step + constraint)
        //   l   = NUM_SVO_ROUNDS (number of Y-prefix variables used by SVO)
        // Partition:
        //   - 0 .. (N/2 - l)              => x_out (MSB block shared with y_rest)
        //   - (N/2 - l) .. (n_s)          => x_in_step
        //   - (n_s) .. (N - l)            => x_in_constraint (non-SVO constraint vars)
        //   - (N - l) .. (N - i - 1)      => y_rest (suffix of Y for current round)
        //   - (N - i - 1) .. (N - 1)      => (u || v_0..v_{i-1}) SVO variables
        // We use MSB->LSB indexing throughout the codebase.

        let padded_num_constraints = UNIFORM_R1CS.len().next_power_of_two();
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        assert_eq!(
            tau.len(),
            total_num_vars,
            "tau length ({}) mismatch with R1CS variable count (step_vars {} + constraint_vars {})",
            tau.len(),
            num_step_vars,
            num_constraint_vars
        );
        assert!(
            NUM_SVO_ROUNDS <= num_constraint_vars,
            "NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) cannot exceed total constraint variables ({num_constraint_vars})"
        );

        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
        let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;
        assert_eq!(
            num_non_svo_z_vars,
            total_num_vars - NUM_SVO_ROUNDS,
            "num_non_svo_z_vars ({num_non_svo_z_vars}) + NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) must be == total_num_vars ({total_num_vars})"
        );

        let potential_x_out_vars = total_num_vars / 2 - NUM_SVO_ROUNDS;
        let iter_num_x_out_vars = std::cmp::min(potential_x_out_vars, num_step_vars);
        let iter_num_x_in_vars = num_non_svo_z_vars - iter_num_x_out_vars;
        let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
        let iter_num_x_in_constraint_vars = num_non_svo_constraint_vars;
        assert_eq!(
            iter_num_x_in_vars,
            iter_num_x_in_step_vars + iter_num_x_in_constraint_vars
        );
        assert_eq!(num_non_svo_z_vars, iter_num_x_out_vars + iter_num_x_in_vars);

        let num_uniform_r1cs_constraints = UNIFORM_R1CS.len();
        let rem_num_uniform_r1cs_constraints = num_uniform_r1cs_constraints % Y_SVO_SPACE_SIZE;

        // Build split-eq helper and precompute E_in (over x_in) and E_out (over x_out)
        let eq_poly = GruenSplitEqPolynomial::new_for_small_value(
            tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
            // Scale E_in by R^2 so typed accumulators reduce to field semantics
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let E_in_evals = eq_poly.E_in_current();
        let E_out_vec = &eq_poly.E_out_vec;

        assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        let num_x_out_vals = 1usize << iter_num_x_out_vars;
        let num_x_in_step_vals = 1usize << iter_num_x_in_step_vars;

        struct PrecomputeTaskOutput<F: JoltField> {
            az_coeffs_local: Vec<SparseCoefficient<I8OrI96>>,
            bz_coeffs_local: Vec<SparseCoefficient<S160>>,
            // New-typed accumulators (small-value path)
            svo_accums_zero_local: [F; NUM_ACCUMS_EVAL_ZERO],
            svo_accums_infty_local: [F; NUM_ACCUMS_EVAL_INFTY],
        }

        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        assert!(
            num_parallel_chunks > 0 || num_x_out_vals == 0,
            "num_parallel_chunks must be positive if there are x_out_vals to process"
        );

        let x_out_chunk_size = if num_x_out_vals > 0 {
            std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };

        // Parallel over chunks of x_out values. For each (x_out, x_in_step):
        //   - Evaluate each constraint-row's A and B LC at step index to obtain Az/Bz blocks
        //   - Fold Az/Bz with E_in(x_in) into tA contributions
        //   - Distribute tA to SVO accumulators via E_out(x_out)
        // We also collect sparse AB coefficients interleaved by (x_next ∈ {0,1}).
        let collected_chunk_outputs: Vec<PrecomputeTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let cycles_per_chunk = (x_out_end - x_out_start) * num_x_in_step_vals;

                let max_coeffs_capacity = cycles_per_chunk * num_uniform_r1cs_constraints;

                let mut chunk_az_coeffs: Vec<SparseCoefficient<I8OrI96>> =
                    Vec::with_capacity(max_coeffs_capacity);
                let mut chunk_bz_coeffs: Vec<SparseCoefficient<S160>> =
                    Vec::with_capacity(max_coeffs_capacity);

                let mut chunk_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut chunk_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                for x_out_val in x_out_start..x_out_end {
                    let mut tA_pos_acc_for_current_x_out =
                        [UnreducedProduct::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut tA_neg_acc_for_current_x_out =
                        [UnreducedProduct::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut current_x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                    let mut current_x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                    for x_in_step_val in 0..num_x_in_step_vals {
                        let current_step_idx =
                            (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;
                        let mut current_x_in_constraint_val = 0;
                        // Materialize row once for this step; reused for all chunks
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        let mut binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                        let mut binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];

                        // Iterate constraints in Y_SVO_SPACE_SIZE blocks so we can call the
                        // small-value kernels on full Az/Bz blocks when available.
                        for (uniform_chunk_iter_idx, uniform_svo_chunk) in
                            UNIFORM_R1CS.chunks(Y_SVO_SPACE_SIZE).enumerate()
                        {
                            let chunk_size = uniform_svo_chunk.len();
                            // Fill Az/Bz values for this chunk using materialized row
                            eval_az_bz_batch_from_row::<F>(
                                uniform_svo_chunk,
                                &row_inputs,
                                &mut binary_az_block[..chunk_size],
                                &mut binary_bz_block[..chunk_size],
                            );

                            // Check per-constraint A/B agree between typed named-eval path
                            // and legacy field LC evaluation.
                            #[cfg(test)]
                            {
                                for idx_in_svo_block in 0..chunk_size {
                                    let named = &uniform_svo_chunk[idx_in_svo_block];

                                    let az_field_from_typed =
                                        binary_az_block[idx_in_svo_block].to_field::<F>();
                                    let bz_field_from_typed =
                                        s160_to_field::<F>(&binary_bz_block[idx_in_svo_block]);

                                    let az_field_baseline =
                                        named.cons.a.evaluate_row_with(&row_inputs);
                                    let bz_field_baseline =
                                        named.cons.b.evaluate_row_with(&row_inputs);

                                    if az_field_from_typed != az_field_baseline {
                                        panic!(
                                            "[SVO A mismatch] step={} cons={:?} idx={}",
                                            current_step_idx, named.name, idx_in_svo_block
                                        );
                                    }
                                    if bz_field_from_typed != bz_field_baseline {
                                        panic!(
                                            "[SVO B mismatch] step={} cons={:?} idx={}",
                                            current_step_idx, named.name, idx_in_svo_block
                                        );
                                    }
                                }
                            }

                            // Process the batch results and populate coefficient vectors
                            for idx_in_svo_block in 0..chunk_size {
                                let global_constraint_idx_in_step =
                                    (uniform_chunk_iter_idx << NUM_SVO_ROUNDS) + idx_in_svo_block;

                                let az = binary_az_block[idx_in_svo_block];
                                let bz = binary_bz_block[idx_in_svo_block];

                                if !az.is_zero() {
                                    let az_index = 2
                                        * (current_step_idx * padded_num_constraints
                                            + global_constraint_idx_in_step);
                                    chunk_az_coeffs.push((az_index, az).into());
                                }
                                if !bz.is_zero() {
                                    let bz_index = 2
                                        * (current_step_idx * padded_num_constraints
                                            + global_constraint_idx_in_step)
                                        + 1;
                                    chunk_bz_coeffs.push((bz_index, bz).into());
                                }
                            }

                            // If this is a full block, compute and update tA, then reset Az, Bz blocks
                            // (the last block may not be full, in which case we need to delay
                            // computation of tA until after processing all constraints in the block)
                            if uniform_svo_chunk.len() == Y_SVO_SPACE_SIZE {
                                let x_in_val = (x_in_step_val << iter_num_x_in_constraint_vars)
                                    | current_x_in_constraint_val;
                                let E_in_val = &E_in_evals[x_in_val];

                                // New typed path
                                svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                                    &binary_az_block,
                                    &binary_bz_block,
                                    E_in_val,
                                    &mut tA_pos_acc_for_current_x_out,
                                    &mut tA_neg_acc_for_current_x_out,
                                );

                                current_x_in_constraint_val += 1;
                                // Reset local blocks for the next iteration
                                binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                                binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];
                            }
                        }

                        // Process final partial block, if any
                        if rem_num_uniform_r1cs_constraints > 0 {
                            let x_in_val_last = (x_in_step_val << iter_num_x_in_constraint_vars)
                                | current_x_in_constraint_val;
                            let E_in_val_last = &E_in_evals[x_in_val_last];

                            // New typed path on partial block currently in binary_* blocks
                            svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                                &binary_az_block,
                                &binary_bz_block,
                                E_in_val_last,
                                &mut tA_pos_acc_for_current_x_out,
                                &mut tA_neg_acc_for_current_x_out,
                            );
                        }
                    }

                    // finalize: reduce unreduced accumulators and combine pos/neg into field
                    let mut tA_sum_for_current_x_out = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    for i in 0..NUM_NONTRIVIAL_TERNARY_POINTS {
                        let pos_f = F::from_montgomery_reduce_2n(tA_pos_acc_for_current_x_out[i]);
                        let neg_f = F::from_montgomery_reduce_2n(tA_neg_acc_for_current_x_out[i]);
                        // E_in was pre-scaled by R^2, so reduction already matches field semantics
                        tA_sum_for_current_x_out[i] = pos_f - neg_f;
                    }

                    // Distribute accumulated tA for this x_out into the SVO accumulators
                    // (both zero and infty evaluations) using precomputed E_out tables.
                    svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                        &tA_sum_for_current_x_out,
                        x_out_val,
                        E_out_vec,
                        &mut current_x_out_svo_zero,
                        &mut current_x_out_svo_infty,
                    );

                    for i in 0..NUM_ACCUMS_EVAL_ZERO {
                        chunk_svo_accums_zero[i] += current_x_out_svo_zero[i];
                    }
                    for i in 0..NUM_ACCUMS_EVAL_INFTY {
                        chunk_svo_accums_infty[i] += current_x_out_svo_infty[i];
                    }
                }

                PrecomputeTaskOutput {
                    az_coeffs_local: chunk_az_coeffs,
                    bz_coeffs_local: chunk_bz_coeffs,
                    svo_accums_zero_local: chunk_svo_accums_zero,
                    svo_accums_infty_local: chunk_svo_accums_infty,
                }
            })
            .collect();

        let mut final_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut final_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];
        let mut final_az_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<I8OrI96>>> =
            Vec::with_capacity(collected_chunk_outputs.len());
        let mut final_bz_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<S160>>> =
            Vec::with_capacity(collected_chunk_outputs.len());

        for task_output in collected_chunk_outputs {
            final_az_unbound_coeffs_shards.push(task_output.az_coeffs_local);
            final_bz_unbound_coeffs_shards.push(task_output.bz_coeffs_local);
            if NUM_ACCUMS_EVAL_ZERO > 0 {
                for idx in 0..NUM_ACCUMS_EVAL_ZERO {
                    final_svo_accums_zero[idx] += task_output.svo_accums_zero_local[idx];
                }
            }
            if NUM_ACCUMS_EVAL_INFTY > 0 {
                for idx in 0..NUM_ACCUMS_EVAL_INFTY {
                    final_svo_accums_infty[idx] += task_output.svo_accums_infty_local[idx];
                }
            }
        }

        (
            final_svo_accums_zero,  // Use new baseline results
            final_svo_accums_infty, // Use new baseline results
            Self {
                az_unbound_coeffs_shards: final_az_unbound_coeffs_shards,
                bz_unbound_coeffs_shards: final_bz_unbound_coeffs_shards,
                bound_coeffs: vec![],
                binding_scratch_space: vec![],
                padded_num_constraints,
            },
        )
    }

    /// This function uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the small value precomputed rounds.
    ///
    /// At this point, we have the `ab_unbound_coeffs` generated from `new_with_precompute`. We will
    /// use these to compute the evals {Az/Bz/Cz}(r, u, x') needed for later linear-time sumcheck
    /// rounds (storing them in `bound_coeffs`), and compute the polynomial for this
    /// round at the same time.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out, x_in,
    /// 0, r) * unbound_coeffs_b(x_out, x_in, 0, r) - unbound_coeffs_c(x_out, x_in, 0, r))`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b,c" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az, Bz, Cz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r) = \sum_{binary y} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    ///
    /// Finally, as we compute each `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs`. which is still in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then derive the next challenge from the transcript, and bind these
    /// bound coeffs for the next round.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::streaming_sumcheck_round"
    )]
    pub fn streaming_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        let num_y_svo_vars = r_challenges.len();
        debug_assert_eq!(
            num_y_svo_vars, NUM_SVO_ROUNDS,
            "r_challenges length mismatch with NUM_SVO_ROUNDS"
        );
        let mut r_rev = r_challenges.clone();
        r_rev.reverse();
        // Scale eq(r, y) by R^2 so unreduced fmadd reductions match field semantics
        let eq_r_evals = EqPolynomial::evals_with_scaling(&r_rev, Some(F::MONTGOMERY_R_SQUARE));

        struct StreamingTaskOutput<F: JoltField> {
            bound_coeffs_local: Vec<SparseCoefficient<F>>,
            sumcheck_eval_at_0_local: F,
            sumcheck_eval_at_infty_local: F,
            // Old baseline (field) contributions for A/B compare
            sumcheck_eval_at_0_old_local: F,
            sumcheck_eval_at_infty_old_local: F,
        }

        // These are needed to derive x_out_val_stream and x_in_val_stream from a block_id
        let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();

        // Take ownership of shards and merge per pair
        let az_shards_to_process = std::mem::take(&mut self.az_unbound_coeffs_shards);
        let bz_shards_to_process = std::mem::take(&mut self.bz_unbound_coeffs_shards);

        let collected_chunk_outputs: Vec<StreamingTaskOutput<F>> = az_shards_to_process
            .into_par_iter()
            .zip_eq(bz_shards_to_process.into_par_iter())
            .map(|(az_shard_data, bz_shard_data)| {
                let estimated_num_bound_coeffs = az_shard_data.len() + bz_shard_data.len();
                let mut task_bound_coeffs = Vec::with_capacity(estimated_num_bound_coeffs);
                let mut task_sum_contrib_0 = F::zero();
                let mut task_sum_contrib_infty = F::zero();
                let mut task_sum_contrib_0_old = F::zero();
                let mut task_sum_contrib_infty_old = F::zero();

                let mut az_iter = az_shard_data.iter().peekable();
                let mut bz_iter = bz_shard_data.iter().peekable();

                while az_iter.peek().is_some() || bz_iter.peek().is_some() {
                    let next_az_index = az_iter.peek().map_or(usize::MAX, |c| c.index);
                    let next_bz_index = bz_iter.peek().map_or(usize::MAX, |c| c.index);
                    // should never be both usize::MAX
                    debug_assert!(next_az_index != usize::MAX || next_bz_index != usize::MAX);
                    let current_block_id = core::cmp::min(next_az_index, next_bz_index)
                        >> Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT;

                    let x_out_val_stream = current_block_id >> num_streaming_x_in_vars;
                    let x_in_val_stream = current_block_id & ((1 << num_streaming_x_in_vars) - 1);

                    // Be permissive during debugging: clamp out-of-range to zero contribution
                    let e_out_val = if x_out_val_stream < eq_poly.E_out_current_len() {
                        eq_poly.E_out_current()[x_out_val_stream]
                    } else {
                        F::zero()
                    };
                    let e_in_val = if eq_poly.E_in_current_len() > 1 {
                        if x_in_val_stream < eq_poly.E_in_current_len() {
                            eq_poly.E_in_current()[x_in_val_stream]
                        } else {
                            F::zero()
                        }
                    } else if eq_poly.E_in_current_len() == 1 {
                        eq_poly.E_in_current()[0]
                    } else {
                        // E_in_current_len() == 0, meaning no x_in variables for eq_poly
                        F::one() // Effective contribution of E_in is 1
                    };
                    let e_block = e_out_val * e_in_val;

                    let mut az_acc = [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];
                    let mut bz_acc = [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];
                    let mut cz_acc = [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];
                    // Old baseline per-block field accumulators
                    let mut az0_field = F::zero();
                    let mut az1_field = F::zero();
                    let mut bz0_field = F::zero();
                    let mut bz1_field = F::zero();
                    let mut cz0_field = F::zero();
                    let mut cz1_field = F::zero();

                    loop {
                        let az_in_block = az_iter.peek().is_some_and(|c| {
                            (c.index >> Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT) == current_block_id
                        });
                        let bz_in_block = bz_iter.peek().is_some_and(|c| {
                            (c.index >> Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT) == current_block_id
                        });

                        if !az_in_block && !bz_in_block {
                            break;
                        }

                        let next_az_index = az_iter.peek().map_or(usize::MAX, |c| c.index);
                        let next_bz_index = bz_iter.peek().map_or(usize::MAX, |c| c.index);

                        if az_in_block && next_az_index <= next_bz_index {
                            let az_coeff = az_iter.next().unwrap();
                            let az_orig_val = az_coeff.value;
                            let mut paired_bz_opt: Option<S160> = None;

                            let local_offset = az_coeff.index & Y_SVO_RELATED_COEFF_BLOCK_SIZE_MASK;
                            let y_val_idx = (local_offset >> 1) % Y_SVO_SPACE_SIZE;
                            let x_next_val = (local_offset >> 1) / Y_SVO_SPACE_SIZE; // 0 or 1
                            let eq_r_y = eq_r_evals[y_val_idx];

                            az_acc[x_next_val].fmadd_az::<F>(&eq_r_y, az_orig_val);
                            // Baseline field accumulation for Az
                            let az_field_val = az_orig_val.to_field::<F>();
                            if x_next_val == 0 {
                                az0_field += eq_r_y * az_field_val;
                            } else {
                                az1_field += eq_r_y * az_field_val;
                            }

                            if let Some(bz_peek) = bz_iter.peek() {
                                if bz_peek.index == az_coeff.index + 1 {
                                    let bz_coeff = bz_iter.next().unwrap();
                                    paired_bz_opt = Some(bz_coeff.value);
                                    bz_acc[x_next_val].fmadd_bz::<F>(&eq_r_y, bz_coeff.value);
                                    // Baseline field accumulation for paired Bz and Cz
                                    let bz_field_val = s160_to_field::<F>(&bz_coeff.value);
                                    if x_next_val == 0 {
                                        bz0_field += eq_r_y * bz_field_val;
                                    } else {
                                        bz1_field += eq_r_y * bz_field_val;
                                    }
                                    // Baseline Cz: gate by per-row CzKind
                                    // Recover in-step row id from global index: drop parity and mod padded
                                    let row_in_step =
                                        (az_coeff.index >> 1) % self.padded_num_constraints;
                                    if row_in_step < UNIFORM_R1CS.len()
                                        && matches!(UNIFORM_R1CS[row_in_step].cz, CzKind::NonZero)
                                    {
                                        let prod_field = az_field_val * bz_field_val;
                                        if x_next_val == 0 {
                                            cz0_field += eq_r_y * prod_field;
                                        } else {
                                            cz1_field += eq_r_y * prod_field;
                                        }
                                    }
                                }
                            }

                            if let Some(bz_for_az) = paired_bz_opt {
                                // Typed Cz: gate by per-row CzKind using recovered in-step row id
                                let row_in_step =
                                    (az_coeff.index >> 1) % self.padded_num_constraints;
                                if row_in_step < UNIFORM_R1CS.len()
                                    && matches!(UNIFORM_R1CS[row_in_step].cz, CzKind::NonZero)
                                {
                                    let prod = az_orig_val * bz_for_az;
                                    cz_acc[x_next_val].fmadd_prod::<F>(&eq_r_y, prod);
                                }
                            }
                        } else if bz_in_block {
                            let bz_coeff = bz_iter.next().unwrap();
                            let local_offset = bz_coeff.index & Y_SVO_RELATED_COEFF_BLOCK_SIZE_MASK;
                            let y_val_idx = (local_offset >> 1) % Y_SVO_SPACE_SIZE;
                            let x_next_val = (local_offset >> 1) / Y_SVO_SPACE_SIZE; // 0 or 1
                            let eq_r_y = eq_r_evals[y_val_idx];

                            bz_acc[x_next_val].fmadd_bz::<F>(&eq_r_y, bz_coeff.value);
                            // Baseline field accumulation for unpaired Bz
                            let bz_field_val = s160_to_field::<F>(&bz_coeff.value);
                            if x_next_val == 0 {
                                bz0_field += eq_r_y * bz_field_val;
                            } else {
                                bz1_field += eq_r_y * bz_field_val;
                            }
                        }
                    }

                    let az0_at_r = az_acc[0].reduce_to_field::<F>();
                    let bz0_at_r = bz_acc[0].reduce_to_field::<F>();
                    let cz0_at_r = cz_acc[0].reduce_to_field::<F>();
                    let az1_at_r = az_acc[1].reduce_to_field::<F>();
                    let bz1_at_r = bz_acc[1].reduce_to_field::<F>();
                    let cz1_at_r = cz_acc[1].reduce_to_field::<F>();

                    if !az0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id, az0_at_r).into());
                    }
                    if !bz0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 1, bz0_at_r).into());
                    }
                    if !cz0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 2, cz0_at_r).into());
                    }
                    if !az1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 3, az1_at_r).into());
                    }
                    if !bz1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 4, bz1_at_r).into());
                    }
                    if !cz1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 5, cz1_at_r).into());
                    }

                    let p_at_xk0 = az0_at_r * bz0_at_r - cz0_at_r;
                    let az_eval_infty = az1_at_r - az0_at_r;
                    let bz_eval_infty = bz1_at_r - bz0_at_r;
                    let p_slope_term = az_eval_infty * bz_eval_infty;

                    task_sum_contrib_0 += e_block * p_at_xk0;
                    task_sum_contrib_infty += e_block * p_slope_term;

                    // Baseline equivalents from field accumulators
                    let p_at_xk0_old = az0_field * bz0_field - cz0_field;
                    let p_slope_term_old = (az1_field - az0_field) * (bz1_field - bz0_field);

                    task_sum_contrib_0_old += e_block * p_at_xk0_old;
                    task_sum_contrib_infty_old += e_block * p_slope_term_old;
                }

                StreamingTaskOutput {
                    bound_coeffs_local: task_bound_coeffs,
                    sumcheck_eval_at_0_local: task_sum_contrib_0,
                    sumcheck_eval_at_infty_local: task_sum_contrib_infty,
                    sumcheck_eval_at_0_old_local: task_sum_contrib_0_old,
                    sumcheck_eval_at_infty_old_local: task_sum_contrib_infty_old,
                }
            })
            .collect();

        // Aggregate sumcheck contributions directly from collected_chunk_outputs
        let mut total_sumcheck_eval_at_0 = F::zero();
        let mut total_sumcheck_eval_at_infty = F::zero();
        let mut total_sumcheck_eval_at_0_old = F::zero();
        let mut total_sumcheck_eval_at_infty_old = F::zero();
        for task_output in &collected_chunk_outputs {
            // Iterate by reference before consuming collected_chunk_outputs
            total_sumcheck_eval_at_0 += task_output.sumcheck_eval_at_0_local;
            total_sumcheck_eval_at_infty += task_output.sumcheck_eval_at_infty_local;
            total_sumcheck_eval_at_0_old += task_output.sumcheck_eval_at_0_old_local;
            total_sumcheck_eval_at_infty_old += task_output.sumcheck_eval_at_infty_old_local;
        }

        // Compute r_i challenge using aggregated sumcheck values (use new baseline)
        let r_i = process_eq_sumcheck_round(
            (total_sumcheck_eval_at_0, total_sumcheck_eval_at_infty),
            eq_poly,
            round_polys,
            r_challenges,
            claim,
            transcript,
        );

        // Bind coefficients directly from task outputs into scratch space
        let per_task_output_sizes: Vec<usize> = collected_chunk_outputs
            .par_iter() // Iterate over collected_chunk_outputs by reference for calculating sizes
            .map(|task_output| {
                let coeffs_from_task = &task_output.bound_coeffs_local;
                let mut current_task_total_output_size = 0;
                for sub_block_of_6 in
                    coeffs_from_task.chunk_by(|sc1, sc2| sc1.index / 6 == sc2.index / 6)
                {
                    // Self::binding_output_length expects a slice representing one such sub_block_of_6
                    current_task_total_output_size += Self::binding_output_length(sub_block_of_6);
                }
                current_task_total_output_size
            })
            .collect();

        let total_binding_output_len: usize = per_task_output_sizes.iter().sum();

        // Prepare binding_scratch_space
        if self.binding_scratch_space.capacity() < total_binding_output_len {
            self.binding_scratch_space
                .reserve_exact(total_binding_output_len - self.binding_scratch_space.capacity());
        }
        unsafe {
            self.binding_scratch_space.set_len(total_binding_output_len);
        }

        // Create mutable slices into binding_scratch_space, one for each task's output
        let mut output_slices_for_tasks: Vec<&mut [SparseCoefficient<F>]> =
            Vec::with_capacity(collected_chunk_outputs.len());
        let mut scratch_remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in per_task_output_sizes {
            let (first, second) = scratch_remainder.split_at_mut(slice_len);
            output_slices_for_tasks.push(first);
            scratch_remainder = second;
        }
        debug_assert_eq!(scratch_remainder.len(), 0);

        collected_chunk_outputs // Now consume collected_chunk_outputs
            .into_par_iter()
            .zip_eq(output_slices_for_tasks.into_par_iter())
            .for_each(|(task_output, output_slice_for_task)| {
                let coeffs_from_task = &task_output.bound_coeffs_local;

                let mut current_output_idx_in_slice = 0;
                // Iterate through sub-blocks of 6 within this task's coeffs_from_task
                for sub_block_of_6 in
                    coeffs_from_task.chunk_by(|sc1, sc2| sc1.index / 6 == sc2.index / 6)
                {
                    if sub_block_of_6.is_empty() {
                        continue;
                    }
                    let block_idx_for_6_coeffs = sub_block_of_6[0].index / 6;

                    let mut az0 = F::zero();
                    let mut bz0 = F::zero();
                    let mut cz0 = F::zero();
                    let mut az1 = F::zero();
                    let mut bz1 = F::zero();
                    let mut cz1 = F::zero();

                    for coeff in sub_block_of_6 {
                        match coeff.index % 6 {
                            0 => az0 = coeff.value,
                            1 => bz0 = coeff.value,
                            2 => cz0 = coeff.value,
                            3 => az1 = coeff.value,
                            4 => bz1 = coeff.value,
                            5 => cz1 = coeff.value,
                            _ => unreachable!(),
                        }
                    }

                    let new_block_idx = block_idx_for_6_coeffs;

                    let bound_az = az0 + r_i * (az1 - az0);
                    if !bound_az.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx, bound_az).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                    let bound_bz = bz0 + r_i * (bz1 - bz0);
                    if !bound_bz.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx + 1, bound_bz).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                    if !cz0.is_zero() || !cz1.is_zero() {
                        let bound_cz = cz0 + r_i * (cz1 - cz0);
                        output_slice_for_task[current_output_idx_in_slice] =
                            (3 * new_block_idx + 2, bound_cz).into();
                        current_output_idx_in_slice += 1;
                    }
                }
            });

        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
    }

    /// Streaming variant that recomputes A/B per cycle instead of consuming cached
    /// unbound coefficient shards. This reduces memory by not materializing
    /// Az/Bz shards at all. It computes {Az,Bz,Cz}(r, x_next∈{0,1}) per (x_out, x_in),
    /// aggregates t_i(0), t_i(∞), derives r_i, then binds into 3 coeffs per block.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::streaming_sumcheck_round_new"
    )]
    pub fn streaming_sumcheck_round_new<ProofTranscript: Transcript>(
        &mut self,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        let num_y_svo_vars = r_challenges.len();
        debug_assert_eq!(
            num_y_svo_vars, NUM_SVO_ROUNDS,
            "r_challenges length mismatch with NUM_SVO_ROUNDS"
        );
        let mut r_rev = r_challenges.clone();
        r_rev.reverse();
        // Scale eq(r, y) by R^2 so unreduced fmadd reductions match field semantics
        let eq_r_evals = EqPolynomial::evals_with_scaling(&r_rev, Some(F::MONTGOMERY_R_SQUARE));

        // Derive partition from current eq_poly lengths and static SVO params
        let padded_num_constraints = self.padded_num_constraints;
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        debug_assert!(NUM_SVO_ROUNDS <= num_constraint_vars);

        let num_x_out_vals = eq_poly.E_out_current_len();
        let iter_num_x_out_vars = if num_x_out_vals > 0 {
            num_x_out_vals.log_2()
        } else {
            0
        };

        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        debug_assert!(iter_num_x_out_vars <= num_step_vars);
        let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
        let num_x_in_step_vals = if iter_num_x_in_step_vars > 0 {
            1usize << iter_num_x_in_step_vars
        } else {
            1
        };

        let num_uniform_r1cs_constraints = UNIFORM_R1CS.len();
        let y_blocks_in_constraints = if num_uniform_r1cs_constraints > 0 {
            num_uniform_r1cs_constraints.div_ceil(Y_SVO_SPACE_SIZE)
        } else {
            0
        };
        let num_block_pairs_per_step = self.padded_num_constraints >> (NUM_SVO_ROUNDS + 1);

        struct TaskOut<F: JoltField> {
            bound6_at_r: Vec<SparseCoefficient<F>>,
            sum0: F,
            sumInf: F,
        }

        // Parallel chunking across x_out
        let num_parallel_chunks = if num_x_out_vals > 0 {
            core::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        let x_out_chunk_size = if num_x_out_vals > 0 {
            core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };

        let results: Vec<TaskOut<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);

                let mut task_sum0 = F::zero();
                let mut task_sumInf = F::zero();
                let mut task_bound6_at_r: Vec<SparseCoefficient<F>> = Vec::new();

                for x_out_val in x_out_start..x_out_end {
                    for x_in_step_val in 0..num_x_in_step_vals {
                        let current_step_idx =
                            (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;

                        // Materialize row once per step
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        // Iterate block-pairs (each pair groups two Y-blocks: x_next=0 and x_next=1)
                        for block_pair_idx in 0..num_block_pairs_per_step {
                            // typed accumulators for x_next ∈ {0,1} within this block-pair
                            let mut az_acc =
                                [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];
                            let mut bz_acc =
                                [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];
                            let mut cz_acc =
                                [SignedUnreducedAccum::new(), SignedUnreducedAccum::new()];

                            // process up to two Y-blocks for this pair
                            for k in 0..2 {
                                let chunk_index = (block_pair_idx << 1) | k;
                                if chunk_index >= y_blocks_in_constraints {
                                    continue;
                                }

                                let start = chunk_index * Y_SVO_SPACE_SIZE;
                                let end = core::cmp::min(
                                    start + Y_SVO_SPACE_SIZE,
                                    num_uniform_r1cs_constraints,
                                );
                                let uniform_svo_chunk = &UNIFORM_R1CS[start..end];
                                let chunk_size = uniform_svo_chunk.len();

                                let mut binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                                let mut binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];

                                eval_az_bz_batch_from_row::<F>(
                                    uniform_svo_chunk,
                                    &row_inputs,
                                    &mut binary_az_block[..chunk_size],
                                    &mut binary_bz_block[..chunk_size],
                                );

                                let x_next_val = k; // 0 or 1 within the pair
                                for idx_in_svo_block in 0..chunk_size {
                                    let eq = eq_r_evals[idx_in_svo_block];
                                    let az = binary_az_block[idx_in_svo_block];
                                    let bz = binary_bz_block[idx_in_svo_block];

                                    az_acc[x_next_val].fmadd_az::<F>(&eq, az);
                                    bz_acc[x_next_val].fmadd_bz::<F>(&eq, bz);

                                    // Cz gate by row kind using recovered row id
                                    let row_in_step =
                                        (chunk_index << NUM_SVO_ROUNDS) + idx_in_svo_block;
                                    if row_in_step < UNIFORM_R1CS.len()
                                        && matches!(UNIFORM_R1CS[row_in_step].cz, CzKind::NonZero)
                                    {
                                        let prod = az * bz;
                                        cz_acc[x_next_val].fmadd_prod::<F>(&eq, prod);
                                    }
                                }
                            }

                            // reduce to field values at y=r for both x_next
                            let az0 = az_acc[0].reduce_to_field::<F>();
                            let bz0 = bz_acc[0].reduce_to_field::<F>();
                            let cz0 = cz_acc[0].reduce_to_field::<F>();
                            let az1 = az_acc[1].reduce_to_field::<F>();
                            let bz1 = bz_acc[1].reduce_to_field::<F>();
                            let cz1 = cz_acc[1].reduce_to_field::<F>();

                            // sumcheck contributions
                            let p0 = az0 * bz0 - cz0;
                            let slope = (az1 - az0) * (bz1 - bz0);

                            // Compute block_id consistent with shard-based indexing
                            let current_block_id =
                                current_step_idx * num_block_pairs_per_step + block_pair_idx;

                            let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();
                            let x_out_idx = current_block_id >> num_streaming_x_in_vars;
                            let x_in_idx = current_block_id & ((1 << num_streaming_x_in_vars) - 1);

                            let e_out = if x_out_idx < eq_poly.E_out_current_len() {
                                eq_poly.E_out_current()[x_out_idx]
                            } else {
                                F::zero()
                            };
                            let e_in = if eq_poly.E_in_current_len() == 0 {
                                F::one()
                            } else if eq_poly.E_in_current_len() == 1 {
                                eq_poly.E_in_current()[0]
                            } else if x_in_idx < eq_poly.E_in_current_len() {
                                eq_poly.E_in_current()[x_in_idx]
                            } else {
                                F::zero()
                            };
                            let e_block = e_out * e_in;

                            task_sum0 += e_block * p0;
                            task_sumInf += e_block * slope;

                            // record six-at-r values
                            let block_id = current_block_id;
                            if !az0.is_zero() {
                                task_bound6_at_r.push((6 * block_id, az0).into());
                            }
                            if !bz0.is_zero() {
                                task_bound6_at_r.push((6 * block_id + 1, bz0).into());
                            }
                            if !cz0.is_zero() {
                                task_bound6_at_r.push((6 * block_id + 2, cz0).into());
                            }
                            if !az1.is_zero() {
                                task_bound6_at_r.push((6 * block_id + 3, az1).into());
                            }
                            if !bz1.is_zero() {
                                task_bound6_at_r.push((6 * block_id + 4, bz1).into());
                            }
                            if !cz1.is_zero() {
                                task_bound6_at_r.push((6 * block_id + 5, cz1).into());
                            }
                        }
                    }
                }

                TaskOut {
                    bound6_at_r: task_bound6_at_r,
                    sum0: task_sum0,
                    sumInf: task_sumInf,
                }
            })
            .collect();

        // Aggregate totals and derive r_i
        let totals = results.iter().fold((F::zero(), F::zero()), |acc, t| {
            (acc.0 + t.sum0, acc.1 + t.sumInf)
        });
        let r_i = process_eq_sumcheck_round(
            totals,
            eq_poly,
            round_polys,
            r_challenges,
            claim,
            transcript,
        );

        // Pre-size binding_scratch_space using same helper
        let per_task_sizes: Vec<usize> = results
            .par_iter()
            .map(|t| {
                let mut size = 0usize;
                for block6 in t.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                    size += Self::binding_output_length(block6);
                }
                size
            })
            .collect();
        let total_len: usize = per_task_sizes.iter().sum();
        if self.binding_scratch_space.capacity() < total_len {
            self.binding_scratch_space
                .reserve_exact(total_len - self.binding_scratch_space.capacity());
        }
        unsafe {
            self.binding_scratch_space.set_len(total_len);
        }

        // Partition scratch and bind in parallel
        let mut slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(results.len());
        let mut rem = self.binding_scratch_space.as_mut_slice();
        for len in per_task_sizes {
            let (a, b) = rem.split_at_mut(len);
            slices.push(a);
            rem = b;
        }

        results
            .into_par_iter()
            .zip_eq(slices.into_par_iter())
            .for_each(|(t, out)| {
                let mut i = 0usize;
                for block6 in t.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                    if block6.is_empty() {
                        continue;
                    }
                    let blk = block6[0].index / 6;

                    let mut az0 = F::zero();
                    let mut bz0 = F::zero();
                    let mut cz0 = F::zero();
                    let mut az1 = F::zero();
                    let mut bz1 = F::zero();
                    let mut cz1 = F::zero();
                    for c in block6 {
                        match c.index % 6 {
                            0 => az0 = c.value,
                            1 => bz0 = c.value,
                            2 => cz0 = c.value,
                            3 => az1 = c.value,
                            4 => bz1 = c.value,
                            5 => cz1 = c.value,
                            _ => {}
                        }
                    }

                    let azb = az0 + r_i * (az1 - az0);
                    if !azb.is_zero() {
                        out[i] = (3 * blk, azb).into();
                        i += 1;
                    }
                    let bzb = bz0 + r_i * (bz1 - bz0);
                    if !bzb.is_zero() {
                        out[i] = (3 * blk + 1, bzb).into();
                        i += 1;
                    }
                    if !cz0.is_zero() || !cz1.is_zero() {
                        let czb = cz0 + r_i * (cz1 - cz0);
                        out[i] = (3 * blk + 2, czb).into();
                        i += 1;
                    }
                }
            });

        core::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
    }

    /// This function computes the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    ///
    /// We then process this to form `s_i(X) = l_i(X) * t_i(X)`, append `s_i.compress()` to the transcript,
    /// derive next challenge `r_i`, then bind both `eq_poly` and `bound_coeffs` with `r_i`.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::remaining_sumcheck_round"
    )]
    pub fn remaining_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        current_claim: &mut F,
    ) {
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

        // If `E_in` is fully bound, then we simply sum over `E_out`
        let quadratic_evals = if eq_poly.E_in_current_len() == 1 {
            let evals: (F, F) = chunks
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
            // If `E_in` is not fully bound, then we have to collect the sum over `E_out` as well
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_bitmask = (1 << num_x1_bits) - 1;

            let evals: (F, F) = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x2 = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x1 = block_index & x1_bitmask;
                        let E_in_evals = eq_poly.E_in_current()[x1];
                        let x2 = block_index >> num_x1_bits;

                        if x2 != prev_x2 {
                            eval_point_0 += eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                            eval_point_infty += eq_poly.E_out_current()[prev_x2] * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x2 = x2;
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

                        inner_sums.0 +=
                            E_in_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 += E_in_evals
                            .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x2] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        };

        // Use the helper function to process the rest of the sumcheck round
        let r_i = process_eq_sumcheck_round(
            quadratic_evals, // (t_i(0), t_i(infty))
            eq_poly,         // Helper will bind this
            round_polys,
            r_challenges,
            current_claim,
            transcript,
        );

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
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients. Only invoked on `bound_coeffs` which holds
    /// Az/Bz/Cz bound evaluations.
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
