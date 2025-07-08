use super::{
    eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial,
    sparse_interleaved_poly::SparseCoefficient, split_eq_poly::GruenSplitEqPolynomial,
    unipoly::CompressedUniPoly,
};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::R1CSInputsOracle;
use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
use crate::{
    field::{JoltField, OptimizedMul, OptimizedMulI128},
    r1cs::builder::Constraint,
    utils::{
        math::Math,
        small_value::{svo_helpers, NUM_SVO_ROUNDS},
        transcript::Transcript,
    },
};
use ark_ff::Zero;
use rayon::prelude::*;

pub const TOTAL_NUM_ACCUMS: usize = svo_helpers::total_num_accums(NUM_SVO_ROUNDS);
pub const NUM_NONTRIVIAL_TERNARY_POINTS: usize =
    svo_helpers::num_non_trivial_ternary_points(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_ZERO: usize = svo_helpers::num_accums_eval_zero(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_INFTY: usize = svo_helpers::num_accums_eval_infty(NUM_SVO_ROUNDS);

pub const Y_SVO_SPACE_SIZE: usize = 1 << NUM_SVO_ROUNDS;
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE: usize = 4 * Y_SVO_SPACE_SIZE; // Az/Bz * Xk=0/1 * Y_SVO_SPACE_SIZE

// Modifications for streaming version:
// 1. Do not have the `unbound_coeffs` in the struct
// 2. Do streaming rounds until we get to a small enough cached size that we can store `bound_coeffs`

#[derive(Clone, Debug)]
pub struct SpartanInterleavedPolynomial<const NUM_SVO_ROUNDS: usize, F: JoltField> {
    /// A list of sparse vectors representing the (interleaved) coefficients for the Az, Bz polynomials
    /// Generated from binary evaluations. Each inner Vec is sorted by index.
    ///
    /// (note: **no** Cz coefficients are stored here, since they are not needed for small value
    /// precomputation, and can be computed on the fly in streaming round)
    pub(crate) ab_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<i128>>>,

    /// The bound coefficients for the Az, Bz, Cz polynomials. Will be populated in the streaming round
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,
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
    #[tracing::instrument(
        skip_all,
        name = "NewSpartanInterleavedPolynomial::new_with_precompute"
    )]
    pub fn new_with_precompute(
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        flattened_polynomials: &[MultilinearPolynomial<F>],
        tau: &[F],
    ) -> ([F; NUM_ACCUMS_EVAL_ZERO], [F; NUM_ACCUMS_EVAL_INFTY], Self) {
        // The variable layout looks as follows:
        // 0 ... (N/2 - l) ... (n_s) ... (N - l) ... (N - i - 1) ... (N - 1)
        // where n_s = num_step_vars, n_c = num_constraint_vars, N = n_s + n_c, l = NUM_SVO_ROUNDS
        // and i is an iterator over 0..l (for the SVO rounds)

        // Within this layout, we have the partition:
        // - 0 ... (N/2 - l) is x_out
        // - (N/2 - l) ... (n_s) is x_in_step
        // - (n_s) ... (N - l) is x_in_constraint (i.e. non_svo_constraint)
        // - (N/2 - l) ... (N - l) in total is x_in
        // - (N - l) ... (N - i - 1) is y_suffix_svo
        // - (N - i - 1) ... (N - 1) is u || v_config

        // --- Variable Definitions ---
        let num_steps = flattened_polynomials[0].len();
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

        // Number of constraint variables that are NOT part of the SVO prefix Y.
        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
        let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;
        assert_eq!(
            num_non_svo_z_vars,
            total_num_vars - NUM_SVO_ROUNDS,
            "num_non_svo_z_vars ({num_non_svo_z_vars}) + NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) must be == total_num_vars ({total_num_vars})"
        );

        // --- Define Iteration Spaces for Non-SVO Z variables (x_out_val, x_in_val) ---
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

        // Assertions about the layout of uniform constraints
        let num_uniform_r1cs_constraints = uniform_constraints.len();
        let rem_num_uniform_r1cs_constraints = num_uniform_r1cs_constraints % Y_SVO_SPACE_SIZE;

        // --- Setup: E_in and E_out tables ---
        // Call GruenSplitEqPolynomial::new_for_small_value with the determined variable splits.
        let eq_poly = GruenSplitEqPolynomial::new_for_small_value(
            tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
        );
        let E_in_evals = eq_poly.E_in_current();
        let E_out_vec = &eq_poly.E_out_vec;

        assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        let num_x_out_vals = 1usize << iter_num_x_out_vars;
        let num_x_in_step_vals = 1usize << iter_num_x_in_step_vars;
        let _num_x_in_non_svo_constraint_vals: usize = 1usize << iter_num_x_in_constraint_vars;

        assert_eq!(
            (1usize << iter_num_x_in_vars),
            E_in_evals.len(),
            "num_x_in_vals ({}) != E_in_evals.len ({})",
            (1usize << iter_num_x_in_vars),
            E_in_evals.len()
        );

        // Define the structure returned by each parallel map task
        struct PrecomputeTaskOutput<F: JoltField> {
            ab_coeffs_local: Vec<SparseCoefficient<i128>>,
            svo_accums_zero_local: [F; NUM_ACCUMS_EVAL_ZERO],
            svo_accums_infty_local: [F; NUM_ACCUMS_EVAL_INFTY],
        }

        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                // Setting number of chunks for more even work distribution
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1 // Avoid 0 chunks if num_x_out_vals is 0
        };
        assert!(
            num_parallel_chunks > 0 || num_x_out_vals == 0,
            "num_parallel_chunks must be positive if there are x_out_vals to process"
        );

        let x_out_chunk_size = if num_x_out_vals > 0 {
            std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0 // No work per chunk if no x_out_vals
        };

        let collected_chunk_outputs: Vec<PrecomputeTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let cycles_per_chunk = (x_out_end - x_out_start) * num_x_in_step_vals;

                // We will be pushing at most 2 values (corresponding to Az and Bz) to `chunk_ab_coeffs`
                // for each constraint in the chunk.
                let max_ab_coeffs_capacity = 2 * cycles_per_chunk * num_uniform_r1cs_constraints;
                let mut chunk_ab_coeffs = Vec::with_capacity(max_ab_coeffs_capacity);

                let mut chunk_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut chunk_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                // Iterate over x_out_vals in this chunk
                for x_out_val in x_out_start..x_out_end {
                    // Accumulator for SUM_{x_in} E_in * P_ext for this specific x_out_val.
                    let mut tA_sum_for_current_x_out = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut current_x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                    let mut current_x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                    // Iterate over x_in_step_vals in this chunk
                    for x_in_step_val in 0..num_x_in_step_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;

                        let mut current_x_in_constraint_val = 0;

                        let mut binary_az_block = [0i128; Y_SVO_SPACE_SIZE];
                        let mut binary_bz_block = [0i128; Y_SVO_SPACE_SIZE];

                        // Process Uniform Constraints
                        for (uniform_chunk_iter_idx, uniform_svo_chunk) in uniform_constraints.chunks(Y_SVO_SPACE_SIZE).enumerate() {
                            for (idx_in_svo_block, constraint) in uniform_svo_chunk.iter().enumerate() {
                                let constraint_idx_in_step = (uniform_chunk_iter_idx << NUM_SVO_ROUNDS) + idx_in_svo_block;

                                let global_r1cs_idx = 2 * (current_step_idx * padded_num_constraints + constraint_idx_in_step);

                                if !constraint.a.terms().is_empty() {
                                    let az = constraint
                                        .a
                                        .evaluate_row(flattened_polynomials, current_step_idx);
                                    if !az.is_zero() {
                                        binary_az_block[idx_in_svo_block] = az;
                                        chunk_ab_coeffs.push((global_r1cs_idx, az).into());
                                    }
                                }

                                if !constraint.b.terms().is_empty() {
                                    let bz = constraint
                                        .b
                                        .evaluate_row(flattened_polynomials, current_step_idx);
                                    if !bz.is_zero() {
                                        binary_bz_block[idx_in_svo_block] = bz;
                                        chunk_ab_coeffs.push((global_r1cs_idx + 1, bz).into());
                                    }
                                }
                            }

                            // If this is a full block, compute and update tA, then reset Az, Bz blocks
                            // (the last block may not be full, in which case we need to delay
                            // computation of tA until after processing all constraints in the block)
                            if uniform_svo_chunk.len() == Y_SVO_SPACE_SIZE {
                                let x_in_val = (x_in_step_val << iter_num_x_in_constraint_vars) | current_x_in_constraint_val;
                                let E_in_val = &E_in_evals[x_in_val];

                                svo_helpers::compute_and_update_tA_inplace_generic::<NUM_SVO_ROUNDS, F>(
                                    &binary_az_block,
                                    &binary_bz_block,
                                    E_in_val,
                                    &mut tA_sum_for_current_x_out,
                                );

                                current_x_in_constraint_val += 1;
                                binary_az_block = [0i128; Y_SVO_SPACE_SIZE];
                                binary_bz_block = [0i128; Y_SVO_SPACE_SIZE];
                            }
                        }

                        // Process the last block if it wasn't full
                        if rem_num_uniform_r1cs_constraints > 0 {
                            let x_in_val_last = (x_in_step_val << iter_num_x_in_constraint_vars) | current_x_in_constraint_val;
                            let E_in_val_last = &E_in_evals[x_in_val_last];

                            svo_helpers::compute_and_update_tA_inplace_generic::<NUM_SVO_ROUNDS, F>(
                                &binary_az_block,
                                &binary_bz_block,
                                E_in_val_last,
                                &mut tA_sum_for_current_x_out,
                            );
                        }
                    } // End x_in_step_val loop

                    // Distribute the accumulated tA values to the SVO accumulators
                    svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                        &tA_sum_for_current_x_out,
                        x_out_val,
                        E_out_vec,
                        &mut current_x_out_svo_zero,
                        &mut current_x_out_svo_infty,
                    );

                    // Accumulate SVO contributions for this x_out_val into chunk accumulators
                    for i in 0..NUM_ACCUMS_EVAL_ZERO {
                        chunk_svo_accums_zero[i] += current_x_out_svo_zero[i];
                    }
                    for i in 0..NUM_ACCUMS_EVAL_INFTY {
                        chunk_svo_accums_infty[i] += current_x_out_svo_infty[i];
                    }
                } // End loop over x_out_val in chunk

                PrecomputeTaskOutput {
                    ab_coeffs_local: chunk_ab_coeffs,
                    svo_accums_zero_local: chunk_svo_accums_zero,
                    svo_accums_infty_local: chunk_svo_accums_infty,
                }
            }) // End .map() over chunks
            .collect(); // Collect all chunk outputs

        // --- Finalization ---
        let mut final_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut final_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];
        let mut final_ab_unbound_coeffs_shards: Vec<Vec<SparseCoefficient<i128>>> =
            Vec::with_capacity(collected_chunk_outputs.len());

        for task_output in collected_chunk_outputs {
            final_ab_unbound_coeffs_shards.push(task_output.ab_coeffs_local); // Move Vec directly

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

        // final_ab_unbound_coeffs_shards is now fully populated and SVO accumulators are summed.

        // Debug check for sortedness
        #[cfg(test)]
        {
            if NUM_SVO_ROUNDS > 0 {
                for shard in &final_ab_unbound_coeffs_shards {
                    // Iterate over &Vec directly
                    if !shard.is_empty() {
                        let mut prev_index = shard[0].index;
                        for coeff in shard.iter().skip(1) {
                            assert!(
                                coeff.index > prev_index,
                                "Indices not monotonically increasing in shard: prev {}, current {}",
                                prev_index, coeff.index
                            );
                            prev_index = coeff.index;
                        }
                    }
                }
            }
        }

        // Return final SVO accumulators and Self struct.
        (
            final_svo_accums_zero,
            final_svo_accums_infty,
            Self {
                ab_unbound_coeffs_shards: final_ab_unbound_coeffs_shards,
                bound_coeffs: vec![],
                binding_scratch_space: vec![],
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
        name = "NewSpartanInterleavedPolynomial::streaming_sumcheck_round"
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
        assert_eq!(
            num_y_svo_vars, NUM_SVO_ROUNDS,
            "r_challenges length mismatch with NUM_SVO_ROUNDS"
        );
        let mut r_rev = r_challenges.clone();
        r_rev.reverse();
        let eq_r_evals = EqPolynomial::evals(&r_rev);

        struct StreamingTaskOutput<F: JoltField> {
            bound_coeffs_local: Vec<SparseCoefficient<F>>,
            sumcheck_eval_at_0_local: F,
            sumcheck_eval_at_infty_local: F,
        }

        // These are needed to derive x_out_val_stream and x_in_val_stream from a block_id
        let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();

        // Take ownership
        let shards_to_process = std::mem::take(&mut self.ab_unbound_coeffs_shards);

        let collected_chunk_outputs: Vec<StreamingTaskOutput<F>> = shards_to_process // Use the taken vec
            .into_par_iter() // Consumes and gives ownership to closures
            .map(|shard_data: Vec<SparseCoefficient<i128>>| {
                // shard_data is now owned Vec
                // Estimate the number of bound coefficients to preallocate
                // Quick math: the shard data has Az + Bz unbound coeffs. Worst case is that each such coeff
                // is in its own `Y_SVO_SPACE_SIZE`-sized block, thus giving a 1-1 correspondence between
                // unbound and bound coeffs for Az and Bz. We also need to account for the same number of Cz coeffs.
                // So the most conservative estimate is `3 * shard_data.len() / 2`, but in practice we see fewer bound coeffs.
                let estimated_num_bound_coeffs = shard_data.len();
                let mut task_bound_coeffs = Vec::with_capacity(estimated_num_bound_coeffs);
                let mut task_sum_contrib_0 = F::zero();
                let mut task_sum_contrib_infty = F::zero();

                for logical_block_coeffs in shard_data.chunk_by(|c1, c2| {
                    // Use owned shard_data directly
                    c1.index / Y_SVO_RELATED_COEFF_BLOCK_SIZE ==
                        c2.index / Y_SVO_RELATED_COEFF_BLOCK_SIZE
                }) {
                    if logical_block_coeffs.is_empty() {
                        continue;
                    }

                    let current_block_id =
                        logical_block_coeffs[0].index / Y_SVO_RELATED_COEFF_BLOCK_SIZE;

                    let x_out_val_stream = current_block_id >> num_streaming_x_in_vars;
                    let x_in_val_stream = current_block_id & ((1 << num_streaming_x_in_vars) - 1);

                    let e_out_val = eq_poly.E_out_current()[x_out_val_stream];
                    let e_in_val = if eq_poly.E_in_current_len() > 1 {
                        eq_poly.E_in_current()[x_in_val_stream]
                    } else if eq_poly.E_in_current_len() == 1 {
                        eq_poly.E_in_current()[0]
                    } else {
                        // E_in_current_len() == 0, meaning no x_in variables for eq_poly
                        F::one() // Effective contribution of E_in is 1
                    };

                    let mut az0_at_r = F::zero();
                    let mut az1_at_r = F::zero();
                    let mut bz0_at_r = F::zero();
                    let mut bz1_at_r = F::zero();
                    let mut cz0_at_r = F::zero();
                    let mut cz1_at_r = F::zero();

                    let mut coeff_idx_in_block = 0;
                    while coeff_idx_in_block < logical_block_coeffs.len() {
                        let current_coeff = &logical_block_coeffs[coeff_idx_in_block];
                        let local_offset = current_coeff.index % Y_SVO_RELATED_COEFF_BLOCK_SIZE;
                        let current_is_B = local_offset % 2 == 1;
                        let y_val_idx = (local_offset / 2) % Y_SVO_SPACE_SIZE;
                        let x_next_val = local_offset / 2 / Y_SVO_SPACE_SIZE; // 0 or 1
                        let eq_r_y = eq_r_evals[y_val_idx];

                        if current_is_B {
                            // Current coefficient is Bz
                            let bz_orig_val = current_coeff.value;
                            match x_next_val {
                                0 => {
                                    bz0_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_val);
                                }
                                1 => {
                                    bz1_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_val);
                                }
                                _ => unreachable!(),
                            }
                            coeff_idx_in_block += 1;
                        } else {
                            // Current coefficient is Az
                            let az_orig_val = current_coeff.value;
                            let mut bz_orig_for_this_az = 0i128;

                            match x_next_val {
                                0 => {
                                    az0_at_r += eq_r_y.mul_i128_1_optimized(az_orig_val);
                                }
                                1 => {
                                    az1_at_r += eq_r_y.mul_i128_1_optimized(az_orig_val);
                                }
                                _ => unreachable!(),
                            }

                            if coeff_idx_in_block + 1 < logical_block_coeffs.len() {
                                let next_coeff = &logical_block_coeffs[coeff_idx_in_block + 1];
                                if next_coeff.index == current_coeff.index + 1 {
                                    bz_orig_for_this_az = next_coeff.value;
                                    let next_local_offset =
                                        next_coeff.index % Y_SVO_RELATED_COEFF_BLOCK_SIZE;
                                    let next_x_next_val = next_local_offset / 2 / Y_SVO_SPACE_SIZE;
                                    debug_assert_eq!(
                                        x_next_val,
                                        next_x_next_val,
                                        "Paired Az/Bz should share x_next_val. Current idx {}, next idx {}, current x_next {}, next x_next {}",
                                        current_coeff.index,
                                        next_coeff.index,
                                        x_next_val,
                                        next_x_next_val
                                    );

                                    match x_next_val {
                                        // x_next_val of the current Az
                                        0 => {
                                            bz0_at_r +=
                                                eq_r_y.mul_i128_1_optimized(bz_orig_for_this_az);
                                        }
                                        1 => {
                                            bz1_at_r +=
                                                eq_r_y.mul_i128_1_optimized(bz_orig_for_this_az);
                                        }
                                        _ => unreachable!(),
                                    }
                                    coeff_idx_in_block += 1; // Consumed the Bz coefficient as well
                                }
                            }
                            coeff_idx_in_block += 1; // Consumed the Az coefficient

                            if !az_orig_val.is_zero() && !bz_orig_for_this_az.is_zero() {
                                let cz_orig_val = az_orig_val.wrapping_mul(bz_orig_for_this_az);
                                match x_next_val {
                                    // x_next_val of the current Az
                                    0 => {
                                        cz0_at_r += eq_r_y.mul_i128(cz_orig_val);
                                    }
                                    1 => {
                                        cz1_at_r += eq_r_y.mul_i128(cz_orig_val);
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }

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

                    task_sum_contrib_0 += e_out_val * e_in_val * p_at_xk0;
                    task_sum_contrib_infty += e_out_val * e_in_val * p_slope_term;
                }

                StreamingTaskOutput {
                    bound_coeffs_local: task_bound_coeffs,
                    sumcheck_eval_at_0_local: task_sum_contrib_0,
                    sumcheck_eval_at_infty_local: task_sum_contrib_infty,
                }
            })
            .collect();

        // Aggregate sumcheck contributions directly from collected_chunk_outputs
        let mut total_sumcheck_eval_at_0 = F::zero();
        let mut total_sumcheck_eval_at_infty = F::zero();
        for task_output in &collected_chunk_outputs {
            // Iterate by reference before consuming collected_chunk_outputs
            total_sumcheck_eval_at_0 += task_output.sumcheck_eval_at_0_local;
            total_sumcheck_eval_at_infty += task_output.sumcheck_eval_at_infty_local;
        }

        // Compute r_i challenge using aggregated sumcheck values
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
                for sub_block_of_6 in coeffs_from_task.chunk_by(
                    |sc1, sc2| sc1.index / 6 == sc2.index / 6
                ) {
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
                            0 => {
                                az0 = coeff.value;
                            }
                            1 => {
                                bz0 = coeff.value;
                            }
                            2 => {
                                cz0 = coeff.value;
                            }
                            3 => {
                                az1 = coeff.value;
                            }
                            4 => {
                                bz1 = coeff.value;
                            }
                            5 => {
                                cz1 = coeff.value;
                            }
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
                    let bound_cz = cz0 + r_i * (cz1 - cz0);
                    if !bound_cz.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx + 2, bound_cz).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                }
                debug_assert_eq!(
                    current_output_idx_in_slice,
                    output_slice_for_task.len(),
                    "Mismatch in written elements vs pre-calculated slice length for task output"
                );
            });

        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
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
        name = "NewSpartanInterleavedPolynomial::remaining_sumcheck_round"
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
                            0 => {
                                az_coeff.0 = Some(coeff.value);
                            }
                            1 => {
                                bz_coeff.0 = Some(coeff.value);
                            }
                            2 => {
                                cz_coeff.0 = Some(coeff.value);
                            }
                            3 => {
                                az_coeff.1 = Some(coeff.value);
                            }
                            4 => {
                                bz_coeff.1 = Some(coeff.value);
                            }
                            5 => {
                                cz_coeff.1 = Some(coeff.value);
                            }
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
                    0 => {
                        final_az_eval = coeff.value;
                    }
                    1 => {
                        final_bz_eval = coeff.value;
                    }
                    2 => {
                        final_cz_eval = coeff.value;
                    }
                    _ => {}
                }
            }
        }
        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}

pub struct SpartanInterleavedPolynomialOracle<
    'a,
    const NUM_SVO_ROUNDS: usize,
    F: JoltField,
    PCS,
    ProofTranscript,
> where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub padded_num_constraints: usize,
    pub uniform_constraints: &'a [Constraint],
    pub input_polys_oracle: R1CSInputsOracle<'a, F, PCS, ProofTranscript>,
    pub num_pieces: usize,
    pub bound_coeffs: Vec<SparseCoefficient<F>>,
    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<'a, F: JoltField, PCS, ProofTranscript>
    SpartanInterleavedPolynomialOracle<'a, NUM_SVO_ROUNDS, F, PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn new(
        padded_num_constraints: usize,
        uniform_constraints: &'a [Constraint],
        input_polys_oracle: R1CSInputsOracle<'a, F, PCS, ProofTranscript>,
    ) -> Self
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_pieces = std::cmp::min(
            input_polys_oracle.shard_length,
            rayon::current_num_threads().next_power_of_two() * 8,
        );
        SpartanInterleavedPolynomialOracle {
            padded_num_constraints,
            uniform_constraints,
            input_polys_oracle,
            num_pieces,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
        }
    }

    pub fn generate_shard(&mut self) -> Vec<Vec<SparseCoefficient<i128>>>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let shard_length = self.input_polys_oracle.shard_length;
        let shard_idx = self.input_polys_oracle.step / shard_length;
        let input_polys_shard = self.input_polys_oracle.next().unwrap();

        let num_uniform_constraints = self.uniform_constraints.len();

        let piece_size = shard_length / self.num_pieces;

        (0..self.num_pieces)
            .into_par_iter()
            .map(|chunk_index| {
                let mut local_az_bz_coeffs =
                    Vec::with_capacity(2 * piece_size * num_uniform_constraints);

                for step_index in piece_size * chunk_index..piece_size * (chunk_index + 1) {
                    // Uniform constraints
                    for (constraint_index, constraint) in
                        self.uniform_constraints.iter().enumerate()
                    {
                        let global_index = 2
                            * ((step_index + shard_idx * shard_length)
                                * self.padded_num_constraints
                                + constraint_index);

                        if !constraint.a.terms().is_empty() {
                            let az = constraint.a.evaluate_row(&input_polys_shard, step_index);
                            if !az.is_zero() {
                                local_az_bz_coeffs.push((global_index, az).into());
                            }
                        }

                        if !constraint.b.terms().is_empty() {
                            let bz = constraint.b.evaluate_row(&input_polys_shard, step_index);
                            if !bz.is_zero() {
                                local_az_bz_coeffs.push((global_index + 1, bz).into());
                            }
                        }
                    }
                }
                local_az_bz_coeffs
            })
            .collect()
    }

    pub fn compute_accumulators(
        &mut self,
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        tau: &[F],
        shard_length: usize,
    ) -> ([F; NUM_ACCUMS_EVAL_ZERO], [F; NUM_ACCUMS_EVAL_INFTY]) {
        assert!(shard_length.is_power_of_two());

        let total_num_vars =
            (self.input_polys_oracle.trace.len() * padded_num_constraints).ilog2() as usize;
        let num_step_vars = self.input_polys_oracle.trace.len().ilog2() as usize;
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };

        assert_eq!(total_num_vars, num_constraint_vars + num_step_vars);
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
            "NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) cannot exceed total constraint variables ({num_constraint_vars})",
        );

        // Number of constraint variables that are NOT part of the SVO prefix Y.
        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
        let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;
        assert_eq!(
            num_non_svo_z_vars,
            total_num_vars - NUM_SVO_ROUNDS,
            "num_non_svo_z_vars ({num_non_svo_z_vars}) + NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) must be == total_num_vars ({total_num_vars})"
        );

        // --- Define Iteration Spaces for Non-SVO Z variables (x_out_val, x_in_val) ---
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

        let num_uniform_r1cs_constraints = uniform_constraints.len();
        let _rem_num_uniform_r1cs_constraints = num_uniform_r1cs_constraints % Y_SVO_SPACE_SIZE;

        // --- Setup: E_in and E_out tables ---
        // Call GruenSplitEqPolynomial::new_for_small_value with the determined variable splits.
        let eq_poly = GruenSplitEqPolynomial::new_for_small_value(
            tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
        );
        let E_in_evals = eq_poly.E_in_current();
        let E_out_vec = &eq_poly.E_out_vec;

        assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        let num_x_out_vals = 1usize << iter_num_x_out_vars;

        assert_eq!(
            1usize << iter_num_x_in_vars,
            E_in_evals.len(),
            "num_x_in_vals ({}) != E_in_evals.len ({})",
            1usize << iter_num_x_in_vars,
            E_in_evals.len()
        );

        let num_shards = self.input_polys_oracle.trace.len() / shard_length;
        assert!(num_shards > 0);
        let num_shard_vars = (shard_length.ilog2() + padded_num_constraints.ilog2()) as usize;
        let mut svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

        if num_shard_vars <= NUM_SVO_ROUNDS + iter_num_x_in_vars {
            // There are multiple shards for every value of x_out_vars. So we iterate over every value of x_out_vars
            // and stream all shards corresponding to that value of x_out_vars.
            let shards_per_x_out_val = num_shards / num_x_out_vals;
            for x_out_val in 0..num_x_out_vals {
                // Accumulator for SUM_{x_in} E_in * P_ext for this specific x_out_val.
                let mut tA_sum_for_current_x_out = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                let mut current_x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut current_x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                for _ in 0..shards_per_x_out_val {
                    let shard = self.next().unwrap();
                    let tA_sum_for_current_shard = shard
                        .par_iter()
                        .map(|piece| {
                            let mut tA_sum_for_current_piece =
                                [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];

                            piece
                                .chunk_by(|u, v| {
                                    u.index >> (NUM_SVO_ROUNDS + 1)
                                        == v.index >> (NUM_SVO_ROUNDS + 1)
                                })
                                .for_each(|svo_block| {
                                    let svo_block_idx = svo_block[0].index >> (NUM_SVO_ROUNDS + 1);
                                    let x_in_val = svo_block_idx & ((1 << iter_num_x_in_vars) - 1);
                                    let E_in_val = E_in_evals[x_in_val];

                                    svo_helpers::process_svo_block(
                                        svo_block,
                                        &mut tA_sum_for_current_piece,
                                        E_in_val,
                                    );
                                });

                            tA_sum_for_current_piece
                        })
                        .reduce(
                            || [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS],
                            |acc, val| {
                                let mut updated_acc = acc;
                                for i in 0..NUM_NONTRIVIAL_TERNARY_POINTS {
                                    updated_acc[i] += val[i];
                                }
                                updated_acc
                            },
                        );

                    for i in 0..NUM_NONTRIVIAL_TERNARY_POINTS {
                        tA_sum_for_current_x_out[i] += tA_sum_for_current_shard[i];
                    }
                }

                // All shards corresponding to x_out_val have been processed.
                // Distribute the accumulated tA values to the SVO accumulators
                svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                    &tA_sum_for_current_x_out,
                    x_out_val,
                    E_out_vec,
                    &mut current_x_out_svo_zero,
                    &mut current_x_out_svo_infty,
                );

                // Add current_x_out_svo_zero and current_x_out_svo_infty to svo_accums_zero and svo_accums_infty
                for i in 0..NUM_ACCUMS_EVAL_ZERO {
                    svo_accums_zero[i] += current_x_out_svo_zero[i];
                }
                for i in 0..NUM_ACCUMS_EVAL_INFTY {
                    svo_accums_infty[i] += current_x_out_svo_infty[i];
                }
            }
        } else {
            // There are multiple values of x_out_vars in the same shard. So we stream a shard and divide it into blocks
            // based on the value of x_out_vars.
            for _ in 0..num_shards {
                let shard = self.next().unwrap();

                let (current_shard_svo_zero, current_shard_svo_infty) = shard
                    .par_iter()
                    .map(|piece| {
                        // let mut tA_sum_for_current_piece = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                        let mut current_piece_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                        let mut current_piece_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];
                        piece
                            .chunk_by(|u, v| {
                                u.index >> (NUM_SVO_ROUNDS + iter_num_x_in_vars + 1)
                                    == v.index >> (NUM_SVO_ROUNDS + iter_num_x_in_vars + 1)
                            })
                            .for_each(|x_out_val_block| {
                                let mut tA_sum_for_current_x_out =
                                    [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                                x_out_val_block
                                    .chunk_by(|u, v| {
                                        u.index >> (NUM_SVO_ROUNDS + 1)
                                            == v.index >> (NUM_SVO_ROUNDS + 1)
                                    })
                                    .for_each(|svo_block| {
                                        let svo_block_idx =
                                            svo_block[0].index >> (NUM_SVO_ROUNDS + 1);
                                        let x_in_val =
                                            svo_block_idx & ((1 << iter_num_x_in_vars) - 1);
                                        let E_in_val = E_in_evals[x_in_val];

                                        svo_helpers::process_svo_block(
                                            svo_block,
                                            &mut tA_sum_for_current_x_out,
                                            E_in_val,
                                        );
                                    });

                                let x_out_val = x_out_val_block[0].index
                                    >> (NUM_SVO_ROUNDS + iter_num_x_in_vars + 1);
                                svo_helpers::distribute_tA_to_svo_accumulators_generic::<
                                    NUM_SVO_ROUNDS,
                                    F,
                                >(
                                    &tA_sum_for_current_x_out,
                                    x_out_val,
                                    E_out_vec,
                                    &mut current_piece_svo_zero,
                                    &mut current_piece_svo_infty,
                                );
                            });
                        (current_piece_svo_zero, current_piece_svo_infty)
                    })
                    .reduce(
                        || {
                            (
                                [F::zero(); NUM_ACCUMS_EVAL_ZERO],
                                [F::zero(); NUM_ACCUMS_EVAL_INFTY],
                            )
                        },
                        |acc, val| {
                            let mut updated_acc = acc;
                            for i in 0..NUM_ACCUMS_EVAL_ZERO {
                                updated_acc.0[i] += val.0[i];
                            }
                            for i in 0..NUM_ACCUMS_EVAL_INFTY {
                                updated_acc.1[i] += val.1[i];
                            }
                            updated_acc
                        },
                    );

                for i in 0..NUM_ACCUMS_EVAL_ZERO {
                    svo_accums_zero[i] += current_shard_svo_zero[i];
                }
                for i in 0..NUM_ACCUMS_EVAL_INFTY {
                    svo_accums_infty[i] += current_shard_svo_infty[i];
                }
            }
        }
        (svo_accums_zero, svo_accums_infty)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn streaming_rounds(
        &mut self,
        num_shards: usize,
        streaming_rounds_start: usize,
        streaming_rounds_end: usize,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        r: &mut Vec<F>,
        eq_r_evals: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
        transcript: &mut ProofTranscript,
    ) {
        let mut partially_bound_coeffs = Vec::<Vec<SparseCoefficient<F>>>::new();
        let mut r_i = F::zero();

        let piece_size =
            (self.input_polys_oracle.shard_length / self.num_pieces) * self.padded_num_constraints;

        assert!(piece_size.is_power_of_two());
        let split_index = std::cmp::min(streaming_rounds_end, piece_size.ilog2() as usize - 1);

        // All blocks start and end in the same piece
        for round in streaming_rounds_start..=split_index {
            let block_size = 1 << (round + 1);

            let mut eval_at_zero = F::zero();
            let mut eval_at_infinity = F::zero();

            assert_eq!(self.input_polys_oracle.step, 0);
            for _ in 0..num_shards {
                let shard = self.next().unwrap();

                let num_x_in_vars = eq_poly.E_in_current_len().log_2();

                let collected_outputs: Vec<(F, F, Vec<SparseCoefficient<F>>)> = shard.par_iter().map(|piece| {
                        let mut eval_at_zero_current_piece = F::zero();
                        let mut eval_at_infinity_current_piece = F::zero();
                        let mut partially_bound_coeffs_current_piece =
                            Vec::<SparseCoefficient<F>>::new();

                        piece
                            .chunk_by(|u, v| u.index / (2 * block_size) == v.index / (2 * block_size))
                            .for_each(|block| {
                                let current_block_id = block[0].index / (2 * block_size);
                                let x_in_val =
                                    current_block_id & ((1 << num_x_in_vars) - 1);
                                let x_out_val = current_block_id >> num_x_in_vars;
                                let e_out_val = eq_poly.E_out_current()[x_out_val];
                                let e_in_val = if eq_poly.E_in_current_len() > 1 {
                                    eq_poly.E_in_current()[x_in_val]
                                } else if eq_poly.E_in_current_len() == 1 {
                                    eq_poly.E_in_current()[0]
                                } else {
                                    // E_in_current_len() == 0, meaning no x_in variables for eq_poly
                                    F::one() // Effective contribution of E_in is 1
                                };

                                let mut az0_at_r = F::zero();
                                let mut az1_at_r = F::zero();
                                let mut bz0_at_r = F::zero();
                                let mut bz1_at_r = F::zero();
                                let mut cz0_at_r = F::zero();
                                let mut cz1_at_r = F::zero();

                                let mut coeff_idx_in_block = 0;
                                while coeff_idx_in_block < block.len() {
                                    let current_coeff = &block[coeff_idx_in_block];
                                    let local_offset =
                                        current_coeff.index % (2 * block_size);
                                    let current_is_B = local_offset % 2 == 1;
                                    let y_val_idx = (local_offset / 2) % (1 << round);
                                    let x_next_val = local_offset / block_size; // 0 or 1
                                    let eq_r_y = eq_r_evals[y_val_idx];

                                    if current_is_B {
                                        // Current coefficient is Bz
                                        let bz_orig_val = current_coeff.value;
                                        match x_next_val {
                                            0 => {
                                                bz0_at_r +=
                                                    eq_r_y.mul_i128_1_optimized(
                                                        bz_orig_val
                                                    );
                                            }
                                            1 => {
                                                bz1_at_r +=
                                                    eq_r_y.mul_i128_1_optimized(
                                                        bz_orig_val
                                                    );
                                            }
                                            _ => unreachable!(),
                                        }
                                        coeff_idx_in_block += 1;
                                    } else {
                                        // Current coefficient is Az
                                        let az_orig_val = current_coeff.value;
                                        let mut bz_orig_for_this_az = 0i128;

                                        match x_next_val {
                                            0 => {
                                                az0_at_r +=
                                                    eq_r_y.mul_i128_1_optimized(
                                                        az_orig_val
                                                    );
                                            }
                                            1 => {
                                                az1_at_r +=
                                                    eq_r_y.mul_i128_1_optimized(
                                                        az_orig_val
                                                    );
                                            }
                                            _ => unreachable!(),
                                        }

                                        if coeff_idx_in_block + 1 < block.len() {
                                            let next_coeff = &block[coeff_idx_in_block + 1];
                                            if next_coeff.index == current_coeff.index + 1 {
                                                bz_orig_for_this_az = next_coeff.value;
                                                let next_local_offset =
                                                    next_coeff.index % (2 * block_size);
                                                let next_x_next_val =
                                                    next_local_offset / block_size;
                                                debug_assert_eq!(
                                                    x_next_val,
                                                    next_x_next_val,
                                                    "Paired Az/Bz should share x_next_val. Current idx {}, next idx {}, current x_next {}, next x_next {}",
                                                    current_coeff.index,
                                                    next_coeff.index,
                                                    x_next_val,
                                                    next_x_next_val
                                                );

                                                match x_next_val {
                                                    // x_next_val of the current Az
                                                    0 => {
                                                        bz0_at_r +=
                                                            eq_r_y.mul_i128_1_optimized(
                                                                bz_orig_for_this_az
                                                            );
                                                    }
                                                    1 => {
                                                        bz1_at_r +=
                                                            eq_r_y.mul_i128_1_optimized(
                                                                bz_orig_for_this_az
                                                            );
                                                    }
                                                    _ => unreachable!(),
                                                }
                                                coeff_idx_in_block += 1; // Consumed the Bz coefficient as well
                                            }
                                        }
                                        coeff_idx_in_block += 1; // Consumed the Az coefficient

                                        if
                                        !az_orig_val.is_zero() &&
                                            !bz_orig_for_this_az.is_zero()
                                        {
                                            let cz_orig_val =
                                                az_orig_val.wrapping_mul(
                                                    bz_orig_for_this_az
                                                );
                                            match x_next_val {
                                                // x_next_val of the current Az
                                                0 => {
                                                    cz0_at_r +=
                                                        eq_r_y.mul_i128(cz_orig_val);
                                                }
                                                1 => {
                                                    cz1_at_r +=
                                                        eq_r_y.mul_i128(cz_orig_val);
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    }
                                }

                                let p_at_xk0 = az0_at_r * bz0_at_r - cz0_at_r;
                                let az_eval_infinity = az1_at_r - az0_at_r;
                                let bz_eval_infinity = bz1_at_r - bz0_at_r;
                                let p_slope_term = az_eval_infinity * bz_eval_infinity;

                                eval_at_zero_current_piece += e_out_val * e_in_val * p_at_xk0;
                                eval_at_infinity_current_piece +=
                                    e_out_val * e_in_val * p_slope_term;

                                if round == streaming_rounds_end {
                                    if !az0_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id, az0_at_r).into()
                                        );
                                    }
                                    if !bz0_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id + 1, bz0_at_r).into()
                                        );
                                    }
                                    if !cz0_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id + 2, cz0_at_r).into()
                                        );
                                    }
                                    if !az1_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id + 3, az1_at_r).into()
                                        );
                                    }
                                    if !bz1_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id + 4, bz1_at_r).into()
                                        );
                                    }
                                    if !cz1_at_r.is_zero() {
                                        partially_bound_coeffs_current_piece.push(
                                            (6 * current_block_id + 5, cz1_at_r).into()
                                        );
                                    }
                                }
                            });

                        (eval_at_zero_current_piece, eval_at_infinity_current_piece, partially_bound_coeffs_current_piece)
                    }).collect();

                for (
                    eval_at_zero_current_piece,
                    eval_at_infinity_current_piece,
                    partially_bound_coeffs_current_piece,
                ) in collected_outputs
                {
                    eval_at_zero += eval_at_zero_current_piece;
                    eval_at_infinity += eval_at_infinity_current_piece;
                    partially_bound_coeffs.push(partially_bound_coeffs_current_piece);
                }
            }

            r_i = process_eq_sumcheck_round(
                (eval_at_zero, eval_at_infinity),
                eq_poly,
                polys,
                r,
                claim,
                transcript,
            );
            let eq_r_evals_mid = eq_r_evals.len();
            for i in 0..eq_r_evals_mid {
                let temp = r_i * eq_r_evals[i];
                eq_r_evals.push(temp);
                eq_r_evals[i] -= temp;
            }
        }

        // Block size larger than the size of a piece.
        for round in (split_index + 1)..=streaming_rounds_end {
            assert_eq!(self.input_polys_oracle.step, 0);
            let block_size = 1 << (round + 1);

            let mut eval_at_zero = F::zero();
            let mut eval_at_infinity = F::zero();

            for _ in 0..num_shards {
                let shard = self.next().unwrap();

                let num_x_in_vars = eq_poly.E_in_current_len().log_2();

                let collected_outputs: Vec<(F, F, Vec<SparseCoefficient<F>>)> = shard
                    .par_chunk_by(|p_1, p_2| {
                        p_1[0].index / (2 * block_size) == p_2[0].index / (2 * block_size)
                    })
                    .map(|block| {
                        let current_block_id = block[0][0].index / (2 * block_size);
                        let x_in_val =
                            current_block_id & ((1 << num_x_in_vars) - 1);
                        let x_out_val = current_block_id >> num_x_in_vars;
                        let e_out_val = eq_poly.E_out_current()[x_out_val];
                        let e_in_val = if eq_poly.E_in_current_len() > 1 {
                            eq_poly.E_in_current()[x_in_val]
                        } else if eq_poly.E_in_current_len() == 1 {
                            eq_poly.E_in_current()[0]
                        } else {
                            // E_in_current_len() == 0, meaning no x_in variables for eq_poly
                            F::one() // Effective contribution of E_in is 1
                        };

                        let mut eval_at_zero_current_block = F::zero();
                        let mut eval_at_infinity_current_block = F::zero();
                        let mut partially_bound_coeff_current_block = Vec::<SparseCoefficient<F>>::new();

                        let (
                            az0_at_r, az1_at_r, bz0_at_r, bz1_at_r, cz0_at_r, cz1_at_r,
                        ) = block.par_iter().map(|piece| {
                            let mut az0_at_r_current_piece = F::zero();
                            let mut az1_at_r_current_piece = F::zero();
                            let mut bz0_at_r_current_piece = F::zero();
                            let mut bz1_at_r_current_piece = F::zero();
                            let mut cz0_at_r_current_piece = F::zero();
                            let mut cz1_at_r_current_piece = F::zero();

                            let mut idx = 0;
                            while idx < piece.len() {
                                let local_offset =
                                    piece[idx].index % (2 * block_size);
                                let current_is_B = local_offset % 2 == 1;
                                let y_val_idx = (local_offset / 2) % (1 << round);
                                let x_next_val = local_offset / block_size; // 0 or 1
                                let eq_r_y = eq_r_evals[y_val_idx];

                                if current_is_B {
                                    // Current coefficient is Bz
                                    let bz_orig_val = piece[idx].value;
                                    match x_next_val {
                                        0 => {
                                            bz0_at_r_current_piece +=
                                                eq_r_y.mul_i128_1_optimized(
                                                    bz_orig_val
                                                );
                                        }
                                        1 => {
                                            bz1_at_r_current_piece +=
                                                eq_r_y.mul_i128_1_optimized(
                                                    bz_orig_val
                                                );
                                        }
                                        _ => unreachable!(),
                                    }
                                    idx += 1;
                                } else {
                                    // Current coefficient is Az
                                    let az_orig_val = piece[idx].value;
                                    let mut bz_orig_for_this_az = 0i128;

                                    match x_next_val {
                                        0 => {
                                            az0_at_r_current_piece +=
                                                eq_r_y.mul_i128_1_optimized(
                                                    az_orig_val
                                                );
                                        }
                                        1 => {
                                            az1_at_r_current_piece +=
                                                eq_r_y.mul_i128_1_optimized(
                                                    az_orig_val
                                                );
                                        }
                                        _ => unreachable!(),
                                    }

                                    if idx + 1 < piece.len() {
                                        let next_coeff = &piece[idx + 1];
                                        if next_coeff.index == piece[idx].index + 1 {
                                            bz_orig_for_this_az = next_coeff.value;
                                            let next_local_offset =
                                                next_coeff.index % (2 * block_size);
                                            let next_x_next_val =
                                                next_local_offset / block_size;
                                            debug_assert_eq!(
                                                x_next_val,
                                                next_x_next_val,
                                                "Paired Az/Bz should share x_next_val. Current idx {}, next idx {}, current x_next {}, next x_next {}",
                                                piece[idx].index,
                                                next_coeff.index,
                                                x_next_val,
                                                next_x_next_val
                                            );

                                            match x_next_val {
                                                // x_next_val of the current Az
                                                0 => {
                                                    bz0_at_r_current_piece +=
                                                        eq_r_y.mul_i128_1_optimized(
                                                            bz_orig_for_this_az
                                                        );
                                                }
                                                1 => {
                                                    bz1_at_r_current_piece +=
                                                        eq_r_y.mul_i128_1_optimized(
                                                            bz_orig_for_this_az
                                                        );
                                                }
                                                _ => unreachable!(),
                                            }
                                            idx += 1; // Consumed the Bz coefficient as well
                                        }
                                    }
                                    idx += 1; // Consumed the Az coefficient

                                    if
                                    !az_orig_val.is_zero() &&
                                        !bz_orig_for_this_az.is_zero()
                                    {
                                        let cz_orig_val =
                                            az_orig_val.wrapping_mul(
                                                bz_orig_for_this_az
                                            );
                                        match x_next_val {
                                            // x_next_val of the current Az
                                            0 => {
                                                cz0_at_r_current_piece +=
                                                    eq_r_y.mul_i128(cz_orig_val);
                                            }
                                            1 => {
                                                cz1_at_r_current_piece +=
                                                    eq_r_y.mul_i128(cz_orig_val);
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                            }
                            (az0_at_r_current_piece, az1_at_r_current_piece, bz0_at_r_current_piece, bz1_at_r_current_piece, cz0_at_r_current_piece, cz1_at_r_current_piece)
                        }).reduce(|| (F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero()), |acc, val| {
                            let mut updated_acc = acc;
                            updated_acc.0 += val.0;
                            updated_acc.1 += val.1;
                            updated_acc.2 += val.2;
                            updated_acc.3 += val.3;
                            updated_acc.4 += val.4;
                            updated_acc.5 += val.5;
                            updated_acc
                        });

                        let p_at_xk0 = az0_at_r * bz0_at_r - cz0_at_r;
                        let az_eval_infinity = az1_at_r - az0_at_r;
                        let bz_eval_infinity = bz1_at_r - bz0_at_r;
                        let p_slope_term = az_eval_infinity * bz_eval_infinity;

                        eval_at_zero_current_block += e_out_val * e_in_val * p_at_xk0;

                        eval_at_infinity_current_block +=
                            e_out_val * e_in_val * p_slope_term;


                        if round == streaming_rounds_end {
                            if !az0_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id, az0_at_r).into());
                            }
                            if !bz0_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id + 1, bz0_at_r).into());
                            }
                            if !cz0_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id + 2, cz0_at_r).into());
                            }
                            if !az1_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id + 3, az1_at_r).into());
                            }
                            if !bz1_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id + 4, bz1_at_r).into());
                            }
                            if !cz1_at_r.is_zero() {
                                partially_bound_coeff_current_block.push((6 * current_block_id + 5, cz1_at_r).into());
                            }
                        }
                        (eval_at_zero_current_block, eval_at_infinity_current_block, partially_bound_coeff_current_block)
                    }).collect();

                for (
                    eval_at_zero_current_piece,
                    eval_at_infinity_current_piece,
                    partially_bound_coeffs_current_piece,
                ) in collected_outputs
                {
                    eval_at_zero += eval_at_zero_current_piece;
                    eval_at_infinity += eval_at_infinity_current_piece;
                    partially_bound_coeffs.push(partially_bound_coeffs_current_piece);
                }
            }

            r_i = process_eq_sumcheck_round(
                (eval_at_zero, eval_at_infinity),
                eq_poly,
                polys,
                r,
                claim,
                transcript,
            );
            if round != streaming_rounds_end {
                let eq_r_evals_mid = eq_r_evals.len();
                for i in 0..eq_r_evals_mid {
                    let temp = r_i * eq_r_evals[i];
                    eq_r_evals.push(temp);
                    eq_r_evals[i] -= temp;
                }
            }
        }

        let num_chunks = rayon::current_num_threads() * 8;
        let chunk_size = partially_bound_coeffs.len() / num_chunks;

        let binding_output_lengths: Vec<usize> = (0..num_chunks)
            .map(|chunk_idx| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = if chunk_idx == num_chunks - 1 {
                    partially_bound_coeffs.len()
                } else {
                    start_idx + chunk_size
                };

                let mut size = 0;
                for idx in start_idx..end_idx {
                    partially_bound_coeffs[idx]
                        .chunk_by(|u, v| u.index / 6 == v.index / 6)
                        .for_each(|block| size += Self::binding_output_length(block));
                }
                size
            })
            .collect();

        let binding_output_length = binding_output_lengths.iter().sum();

        // Prepare binding_scratch_space
        if self.binding_scratch_space.capacity() < binding_output_length {
            self.binding_scratch_space
                .reserve_exact(binding_output_length - self.binding_scratch_space.capacity());
        }
        unsafe {
            self.binding_scratch_space.set_len(binding_output_length);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(num_chunks);
        let mut scratch_remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in binding_output_lengths {
            let (first, second) = scratch_remainder.split_at_mut(slice_len);
            output_slices.push(first);
            scratch_remainder = second;
        }
        assert_eq!(scratch_remainder.len(), 0);

        (0..num_chunks)
            .into_par_iter()
            .zip(output_slices.into_par_iter())
            .for_each(|(chunk_idx, output_slice)| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = if chunk_idx == num_chunks - 1 {
                    partially_bound_coeffs.len()
                } else {
                    start_idx + chunk_size
                };

                let mut current_output_idx_in_slice = 0;
                for chunk_idx in start_idx..end_idx {
                    let coeffs_to_bind = &partially_bound_coeffs[chunk_idx];
                    for block in coeffs_to_bind.chunk_by(|sc1, sc2| sc1.index / 6 == sc2.index / 6)
                    {
                        if block.is_empty() {
                            continue;
                        }
                        let block_idx_for_6_coeffs = block[0].index / 6;

                        let mut az0 = F::zero();
                        let mut bz0 = F::zero();
                        let mut cz0 = F::zero();
                        let mut az1 = F::zero();
                        let mut bz1 = F::zero();
                        let mut cz1 = F::zero();

                        for coeff in block {
                            match coeff.index % 6 {
                                0 => {
                                    az0 = coeff.value;
                                }
                                1 => {
                                    bz0 = coeff.value;
                                }
                                2 => {
                                    cz0 = coeff.value;
                                }
                                3 => {
                                    az1 = coeff.value;
                                }
                                4 => {
                                    bz1 = coeff.value;
                                }
                                5 => {
                                    cz1 = coeff.value;
                                }
                                _ => unreachable!(),
                            }
                        }

                        let new_block_idx = block_idx_for_6_coeffs;

                        let bound_az = az0 + r_i * (az1 - az0);
                        if !bound_az.is_zero() {
                            if current_output_idx_in_slice < output_slice.len() {
                                output_slice[current_output_idx_in_slice] =
                                    (3 * new_block_idx, bound_az).into();
                            }
                            current_output_idx_in_slice += 1;
                        }
                        let bound_bz = bz0 + r_i * (bz1 - bz0);
                        if !bound_bz.is_zero() {
                            if current_output_idx_in_slice < output_slice.len() {
                                output_slice[current_output_idx_in_slice] =
                                    (3 * new_block_idx + 1, bound_bz).into();
                            }
                            current_output_idx_in_slice += 1;
                        }
                        let bound_cz = cz0 + r_i * (cz1 - cz0);
                        if !bound_cz.is_zero() {
                            if current_output_idx_in_slice < output_slice.len() {
                                output_slice[current_output_idx_in_slice] =
                                    (3 * new_block_idx + 2, bound_cz).into();
                            }
                            current_output_idx_in_slice += 1;
                        }
                    }
                }
                assert_eq!(
                    current_output_idx_in_slice,
                    output_slice.len(),
                    "Mismatch in written elements vs pre-calculated slice length for task output"
                );
            });
        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
    }

    pub fn remaining_sumcheck_rounds(
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
                            0 => {
                                az_coeff.0 = Some(coeff.value);
                            }
                            1 => {
                                bz_coeff.0 = Some(coeff.value);
                            }
                            2 => {
                                cz_coeff.0 = Some(coeff.value);
                            }
                            3 => {
                                az_coeff.1 = Some(coeff.value);
                            }
                            4 => {
                                bz_coeff.1 = Some(coeff.value);
                            }
                            5 => {
                                cz_coeff.1 = Some(coeff.value);
                            }
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
                    0 => {
                        final_az_eval = coeff.value;
                    }
                    1 => {
                        final_bz_eval = coeff.value;
                    }
                    2 => {
                        final_cz_eval = coeff.value;
                    }
                    _ => {}
                }
            }
        }
        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}

impl<'a, F: JoltField, PCS, ProofTranscript> Iterator
    for SpartanInterleavedPolynomialOracle<'a, NUM_SVO_ROUNDS, F, PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Item = Vec<Vec<SparseCoefficient<i128>>>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generate_shard())
    }
}
