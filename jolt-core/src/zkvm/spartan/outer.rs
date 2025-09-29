#![allow(clippy::too_many_arguments)]
use crate::field::{JoltField, OptimizedMul};
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::univariate_skip::{
    compute_az_r_group0, compute_az_r_group1, compute_bz_r_group0, compute_bz_r_group1,
    compute_cz_r_group1,
};
// use crate::utils::errors::ProofVerifyError;
// #[cfg(not(target_arch = "wasm32"))]
// use crate::utils::profiling::print_current_memory_usage;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
// use crate::utils::thread::drop_in_background_thread;
use crate::zkvm::JoltSharedPreprocessing;
use crate::zkvm::ProofVerifyError;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_serialize::*;
use std::marker::PhantomData;
use tracer::instruction::Cycle;

use crate::{
    utils::math::Math,
    utils::univariate_skip::accum::s160_to_field,
    zkvm::r1cs::{
        constraints::{
            eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
            eval_cz_second_group, UNIFORM_R1CS, UNIVARIATE_SKIP_DEGREE,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        },
        inputs::R1CSCycleInputs,
    },
};
use allocative::Allocative;
use ark_ff::biginteger::S160;
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
/// The proof format for Spartan's outer sumcheck.
/// This is different from the generic `SumcheckInstanceProof` since we apply univariate skip to the
/// outer sumcheck, which means the first round needs to be handled differently.
pub struct OuterSumcheckProof<F: JoltField, ProofTranscript: Transcript> {
    /// The first polynomial (of high degree) from the univariate skip.
    /// We send the whole polynomial for easier verification (only one extra field element)
    pub first_poly: UniPoly<F>,
    /// The remaining polynomials from the second round onwards.
    pub remaining_proof: SumcheckInstanceProof<F, ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> OuterSumcheckProof<F, ProofTranscript> {
    pub fn new(
        first_poly: UniPoly<F>,
        remaining_polys: Vec<CompressedUniPoly<F>>,
    ) -> OuterSumcheckProof<F, ProofTranscript> {
        OuterSumcheckProof {
            first_poly,
            remaining_proof: SumcheckInstanceProof::new(remaining_polys),
        }
    }

    /// Verify this sumcheck proof given that the first variable has higher degree than the rest.
    /// (which happens during Spartan's outer sumcheck with univariate skip)
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `const N`: the first degree plus one (e.g. the size of the first evaluation domain)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound_first`: Maximum allowed degree of the first univariate polynomial
    /// - `degree_bound_rest`: Maximum allowed degree of the rest of the univariate polynomials
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify<const N: usize>(
        &self,
        num_rounds: usize,
        degree_bound_first: usize,
        degree_bound_rest: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.remaining_proof.compressed_polys.len() + 1, num_rounds);

        // verification for the first round
        if self.first_poly.degree() > degree_bound_first {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound_first,
                self.first_poly.degree(),
            ));
        }
        self.first_poly.append_to_transcript(transcript);
        let r_0 = transcript.challenge_scalar();
        r.push(r_0);
        // First round: send all coeffs; check symmetric-domain sum is zero (initial claim),
        // then set next claim to s1(r_0)
        let (ok, next_claim) = self
            .first_poly
            .check_sum_evals_and_set_new_claim::<N>(&F::zero(), &r_0);
        if !ok {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        let (e, r_rest) = self.remaining_proof.verify(
            next_claim,
            num_rounds - 1,
            degree_bound_rest,
            transcript,
        )?;

        r.extend(r_rest);

        Ok((e, r))
    }
}

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

#[derive(Clone, Debug, Allocative, Default)]
pub struct SpartanInterleavedPoly<F: JoltField> {
    /// The bound coefficients for the Az, Bz, Cz polynomials.
    /// Will be populated in the streaming round (after SVO rounds)
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> SpartanInterleavedPoly<F> {
    pub fn new() -> Self {
        Self {
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
        }
    }
}

#[derive(Allocative)]
pub struct OuterSumcheck<F: JoltField, ProofTranscript: Transcript> {
    // The claim to be used through the sumcheck rounds, initialized to zero
    claim: F,
    // The interleaved polynomial of Az/Bz/Cz bound evaluations, to be used through the sumcheck rounds
    interleaved_poly: SpartanInterleavedPoly<F>,
    // The split-eq polynomial to be used through the sumcheck rounds
    split_eq_poly: GruenSplitEqPolynomial<F>,
    // The last of tau vector, used for univariate skip
    tau_high: F,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> OuterSumcheck<F, ProofTranscript> {
    // Initialize a Spartan outer sumcheck instance given the tau vector
    pub fn initialize(tau: &[F]) -> Self {
        let tau_low = &tau[0..tau.len() - 1];
        Self {
            claim: F::zero(),
            interleaved_poly: SpartanInterleavedPoly::new(),
            split_eq_poly: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            tau_high: tau[tau.len() - 1],
            _marker: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "OuterSumcheck::prove")]
    pub fn prove(
        preprocessing: &JoltSharedPreprocessing,
        trace: &[Cycle],
        num_rounds: usize,
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (OuterSumcheckProof<F, ProofTranscript>, Vec<F>, [F; 3]) {
        let mut r = Vec::new();

        let mut outer_sumcheck = Self::initialize(tau);

        let extended_evals = outer_sumcheck.compute_univariate_skip_evals(preprocessing, trace);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("SpartanInterleavedPoly", &outer_sumcheck);

        // First round (univariate skip): build s1(Z) = lagrange_poly(Z) * t1(Z)
        let first_poly = outer_sumcheck.process_first_round_from_extended_evals(
            &extended_evals,
            transcript,
            &mut r,
        );

        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();

        outer_sumcheck.streaming_sumcheck_round(
            preprocessing,
            trace,
            transcript,
            &mut r,
            &mut polys,
        );

        for _ in 2..num_rounds {
            outer_sumcheck.remaining_sumcheck_round(transcript, &mut r, &mut polys);
        }

        (
            OuterSumcheckProof::new(first_poly, polys),
            r,
            outer_sumcheck.final_sumcheck_evals(),
        )
    }

    /// NEW! Doing small-value optimization with univariate skip instead of round batching / compression
    /// We hard-code one univariate skip of degree \ceil{NUM_CONSTRAINTS / 2} - 1 (currently 13)
    ///
    /// Returns the 13 accumulators:
    /// t_1(z) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    ///             \sum_{y = 0,1} E[y] * (Az(x_out, x_in, y, z) * Bz(x_out, x_in, y, z) - Cz(x_out, x_in, y, z))
    ///
    /// for all z in {-12, ... , 12} \ {-6, ..., 7}
    #[tracing::instrument(skip_all, name = "OuterSumcheck::compute_univariate_skip_evals")]
    pub fn compute_univariate_skip_evals(
        &mut self,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
    ) -> [F; UNIVARIATE_SKIP_DEGREE] {
        // Precompute 13 Lagrange coefficient vectors (length 14) for target Z values
        let start: i64 = -((UNIVARIATE_SKIP_DEGREE as i64) / 2); // -6
        let mut target_shifts: [i64; UNIVARIATE_SKIP_DEGREE] = [0; UNIVARIATE_SKIP_DEGREE];
        let mut tix = 0;
        for k in 0..6 {
            target_shifts[tix] = (-7 - k) - start;
            tix += 1; // negatives
            target_shifts[tix] = (8 + k) - start;
            tix += 1; // positives
        }
        target_shifts[12] = (-13) - start;
        let coeffs_per_j: [[i32; UNIVARIATE_SKIP_DOMAIN_SIZE]; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| {
                LagrangeHelper::shift_coeffs_i32::<{ UNIVARIATE_SKIP_DOMAIN_SIZE }>(
                    target_shifts[j],
                )
            });

        let num_x_out_vals = self.split_eq_poly.E_out_current_len();
        let num_x_in_vals = self.split_eq_poly.E_in_current_len();
        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        let x_out_chunk_size = if num_x_out_vals > 0 {
            std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_field: [F; UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        // Eq weights
                        let e_out = self.split_eq_poly.E_out_current()[x_out_val];
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else {
                            self.split_eq_poly.E_in_current()[x_in_val]
                        };
                        let e_block = e_out * e_in;

                        // First group (14 eq-conditional, Cz=0): Az bool, Bz S160
                        let az1_bool = eval_az_first_group(&row_inputs);
                        let bz1_s160 = eval_bz_first_group(&row_inputs);

                        // Second group (13 + pad): Az i128, Bz/Cz S160
                        let az2_i96 = eval_az_second_group(&row_inputs);
                        let bz2 = eval_bz_second_group(&row_inputs);
                        let cz2 = eval_cz_second_group(&row_inputs);

                        let mut az2_i128_padded: [i128; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [0; UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut bz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut cz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        for i in 0..(UNIVARIATE_SKIP_DOMAIN_SIZE - 1) {
                            az2_i128_padded[i] = az2_i96[i].to_i128();
                            bz2_s160_padded[i] = bz2[i];
                            cz2_s160_padded[i] = cz2[i];
                        }

                        // Accumulate per target j using product-of-sums form
                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];

                            // Group 1: (Σ c_i * Az1[i]) * (Σ c_i * Bz1[i])
                            let mut az1_ext = F::zero();
                            let mut bz1_ext = F::zero();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                if az1_bool[i] {
                                    az1_ext += F::one().mul_i64(c);
                                }
                                let bz_f = s160_to_field::<F>(&bz1_s160[i]);
                                bz1_ext += bz_f.mul_i64(c);
                            }
                            acc_field[j] += e_block * (az1_ext * bz1_ext);

                            // Group 2: (Σ c_i * Az2[i])*(Σ c_i * Bz2[i]) - (Σ c_i * Cz2[i])
                            let mut az2_ext = F::zero();
                            let mut bz2_ext = F::zero();
                            let mut cz2_ext = F::zero();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                let az_f = F::from_i128(az2_i128_padded[i]);
                                az2_ext += az_f.mul_i64(c);
                                let bz_f = s160_to_field::<F>(&bz2_s160_padded[i]);
                                bz2_ext += bz_f.mul_i64(c);
                                let cz_f = s160_to_field::<F>(&cz2_s160_padded[i]);
                                cz2_ext += cz_f.mul_i64(c);
                            }
                            acc_field[j] += e_block * (az2_ext * bz2_ext - cz2_ext);
                        }
                    }
                }

                let mut out = [F::zero(); UNIVARIATE_SKIP_DEGREE];
                for j in 0..UNIVARIATE_SKIP_DEGREE {
                    out[j] = acc_field[j];
                }
                out
            })
            .reduce(
                || [F::zero(); UNIVARIATE_SKIP_DEGREE],
                |mut a, b| {
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
    }

    #[inline]
    fn process_first_round_from_extended_evals(
        &mut self,
        extended_evals: &[F; UNIVARIATE_SKIP_DEGREE],
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
    ) -> UniPoly<F> {
        // Map the 13 interleaved extended evals into the full symmetric domain of size 27: Z in [-13..13]
        let mut t1_vals: [F; 2 * UNIVARIATE_SKIP_DEGREE + 1] =
            [F::zero(); 2 * UNIVARIATE_SKIP_DEGREE + 1];
        for (idx, &val) in extended_evals.iter().enumerate() {
            let z: i64 = if (idx & 1) == 0 {
                // even: negatives starting at -7
                -7 - ((idx / 2) as i64)
            } else {
                // odd: positives starting at 8
                8 + ((idx / 2) as i64)
            };
            let pos = (z + (UNIVARIATE_SKIP_DEGREE as i64)) as usize + UNIVARIATE_SKIP_DEGREE;
            t1_vals[pos] = val;
        }

        // Interpolate degree-26 coefficients of t1 from its 27 values
        let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<
            { 2 * UNIVARIATE_SKIP_DEGREE + 1 },
        >(&t1_vals);

        // Build lagrange_poly(Z) coefficients of degree 13 from basis values at tau_high over base window [-6..7]
        let lagrange_poly_values =
            LagrangePolynomial::evals::<UNIVARIATE_SKIP_DOMAIN_SIZE>(&self.tau_high);
        let lagrange_poly_coeffs = LagrangePolynomial::interpolate_coeffs::<
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&lagrange_poly_values);

        // Convolve lagrange_poly (len 14) with t1 (len 27) to get s1 (len 40), degree 39
        let mut s1_coeffs: Vec<F> =
            vec![F::zero(); UNIVARIATE_SKIP_DOMAIN_SIZE + (2 * UNIVARIATE_SKIP_DEGREE + 1) - 1];
        for (i, &a) in lagrange_poly_coeffs.iter().enumerate() {
            for (j, &b) in t1_coeffs.iter().enumerate() {
                s1_coeffs[i + j] += a * b;
            }
        }

        // Append full s1 poly (send all coeffs), derive r0, set claim (do NOT bind eq_poly yet)
        let s1_poly = UniPoly::from_coeff(s1_coeffs.clone());
        s1_poly.append_to_transcript(transcript);
        let r0: F = transcript.challenge_scalar();
        r.push(r0);
        self.claim = UniPoly::eval_with_coeffs(&s1_coeffs[..], &r0);
        s1_poly
    }

    /// This function uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the small value precomputed rounds.
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
    /// `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r) = \sum_{y in D} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    ///
    /// Finally, as we compute each `{a/b/c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs`. which is still in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then derive the next challenge from the transcript, and bind these
    /// bound coeffs for the next round.
    #[tracing::instrument(skip_all, name = "OuterSumcheck::streaming_sumcheck_round")]
    pub fn streaming_sumcheck_round(
        &mut self,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        transcript: &mut ProofTranscript,
        r_challenge: &mut Vec<F>, // Only one challenge right now
        round_polys: &mut Vec<CompressedUniPoly<F>>,
    ) {
        // Lagrange basis over the univariate-skip domain (size 14)
        let lagrange_evals_r =
            LagrangePolynomial::evals::<UNIVARIATE_SKIP_DOMAIN_SIZE>(&r_challenge[0]);

        let eq_poly = &mut self.split_eq_poly;

        // Derive partition from current eq_poly lengths and static SVO params
        let _padded_num_constraints = UNIFORM_R1CS.len().next_power_of_two();

        let num_x_out_vals = eq_poly.E_out_current_len();
        let _iter_num_x_out_vars = if num_x_out_vals > 0 {
            num_x_out_vals.log_2()
        } else {
            0
        };

        let num_x_in_vals = eq_poly.E_in_current_len();
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        let _num_uniform_r1cs_constraints = UNIFORM_R1CS.len();

        struct StreamingTaskOutput<F: JoltField> {
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

        let results: Vec<StreamingTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);

                let mut task_sum0 = F::zero();
                let mut task_sumInf = F::zero();
                let mut task_bound6_at_r: Vec<SparseCoefficient<F>> = Vec::new();

                for x_out_val in x_out_start..x_out_end {
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        // Materialize row once per step
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        // Compute {Az/Bz/Cz}(x_out, x_in, {0,1}, r)
                        // Note: should have specialized functions for each group (0 or 1)
                        // Az is always binary for first group
                        // Cz is always zero for first group for instance (so we don't even need to?)

                        // Then compute az_r * bz_r - cz_r, multiply by e_in (w/ delayed reduction)

                        // reduce to field values at y=r for both x_next
                        let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        // cz0 is always zero for first group
                        let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                        // sumcheck contributions
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);

                        // Compute block_id consistent with shard-based indexing
                        let current_block_id = current_step_idx;

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
                        // cz0 is now always zero per our grouping
                        // if !cz0.is_zero() {
                        //     task_bound6_at_r.push((6 * block_id + 2, cz0).into());
                        // }
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

                StreamingTaskOutput {
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
        let r_i = self.process_eq_sumcheck_round(totals, round_polys, r_challenge, transcript);

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
        if self.interleaved_poly.binding_scratch_space.capacity() < total_len {
            self.interleaved_poly
                .binding_scratch_space
                .reserve_exact(total_len - self.interleaved_poly.binding_scratch_space.capacity());
        }
        unsafe {
            self.interleaved_poly
                .binding_scratch_space
                .set_len(total_len);
        }

        // Partition scratch and bind in parallel
        let mut slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(results.len());
        let mut rem = self.interleaved_poly.binding_scratch_space.as_mut_slice();
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

        core::mem::swap(
            &mut self.interleaved_poly.bound_coeffs,
            &mut self.interleaved_poly.binding_scratch_space,
        );
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
    #[tracing::instrument(skip_all, name = "OuterSumcheck::remaining_sumcheck_round")]
    pub fn remaining_sumcheck_round(
        &mut self,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
    ) {
        // Compute quadratic evals in a limited scope to avoid holding an immutable borrow during the mutable self call
        let quadratic_evals = {
            let block_size = self
                .interleaved_poly
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(6);
            let chunks: Vec<_> = self
                .interleaved_poly
                .bound_coeffs
                .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
                .collect();

            if self.split_eq_poly.E_in_current_len() == 1 {
                chunks
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

                                let eq_evals = self.split_eq_poly.E_out_current()[block_index];

                                (
                                    eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                                    eq_evals.mul_0_optimized(
                                        az_eval_infty.mul_0_optimized(bz_eval_infty),
                                    ),
                                )
                            })
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                    )
            } else {
                let num_x1_bits = self.split_eq_poly.E_in_current_len().log_2();
                let x1_bitmask = (1 << num_x1_bits) - 1;

                chunks
                    .par_iter()
                    .map(|chunk| {
                        let mut eval_point_0 = F::zero();
                        let mut eval_point_infty = F::zero();

                        let mut inner_sums = (F::zero(), F::zero());
                        let mut prev_x2 = 0;

                        for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                            let block_index = sparse_block[0].index / 6;
                            let x1 = block_index & x1_bitmask;
                            let E_in_evals = self.split_eq_poly.E_in_current()[x1];
                            let x2 = block_index >> num_x1_bits;

                            if x2 != prev_x2 {
                                eval_point_0 +=
                                    self.split_eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                                eval_point_infty +=
                                    self.split_eq_poly.E_out_current()[prev_x2] * inner_sums.1;

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

                        eval_point_0 += self.split_eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                        eval_point_infty +=
                            self.split_eq_poly.E_out_current()[prev_x2] * inner_sums.1;

                        (eval_point_0, eval_point_infty)
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                    )
            }
        };

        // Use the helper function to process the rest of the sumcheck round
        let r_i = self.process_eq_sumcheck_round(
            quadratic_evals, // (t_i(0), t_i(infty))
            round_polys,
            r_challenges,
            transcript,
        );

        // Rebuild chunks after mutable borrow above
        let block_size = self
            .interleaved_poly
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .interleaved_poly
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        if self.interleaved_poly.binding_scratch_space.is_empty() {
            self.interleaved_poly.binding_scratch_space = Vec::with_capacity(total_output_len);
        }
        unsafe {
            self.interleaved_poly
                .binding_scratch_space
                .set_len(total_output_len);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.interleaved_poly.binding_scratch_space.as_mut_slice();
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

        std::mem::swap(
            &mut self.interleaved_poly.bound_coeffs,
            &mut self.interleaved_poly.binding_scratch_space,
        );
    }

    /// Helper function to encapsulate the common subroutine for sumcheck with eq poly factor:
    /// - Compute the linear factor E_i(X) from the current eq-poly
    /// - Reconstruct the cubic polynomial s_i(X) = E_i(X) * t_i(X) for the i-th round
    /// - Compress the cubic polynomial
    /// - Append the compressed polynomial to the transcript
    /// - Derive the challenge for the next round
    /// - Bind the cubic polynomial to the challenge
    /// - Update the claim as the evaluation of the cubic polynomial at the challenge
    ///
    /// Returns the derived challenge
    #[inline]
    pub fn process_eq_sumcheck_round(
        &mut self,
        quadratic_evals: (F, F), // (t_i(0), t_i(infty))
        polys: &mut Vec<CompressedUniPoly<F>>,
        r: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> F {
        let eq_poly = &mut self.split_eq_poly;
        let claim = &mut self.claim;
        let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

        let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
            // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
            [
                eq_poly.current_scalar - scalar_times_w_i,
                scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
            ],
            quadratic_evals.0,
            quadratic_evals.1,
            *claim,
        );

        // Compress and add to transcript
        let compressed_poly = cubic_poly.compress();
        compressed_poly.append_to_transcript(transcript);

        // Derive challenge
        let r_i: F = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Evaluate for next round's claim
        *claim = cubic_poly.evaluate(&r_i);

        // Bind eq_poly for next round
        eq_poly.bind(r_i);

        r_i
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
            if let Some(coeff) = self.interleaved_poly.bound_coeffs.get(i) {
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
