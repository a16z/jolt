//! Stage 3b: Jagged Assist - Batch MLE Verification
//!
//! This module implements the Jagged Assist optimization from Theorem 1.5 of the
//! "Jagged Polynomial Commitments" paper (Hemo et al., May 2025).
//!
//! The Jagged Assist reduces verifier costs when computing:
//!   f̂_jagged(r_s, r_x, r_dense) = Σ_{y ∈ [K]} eq(r_s, y) · ĝ(r_x, r_dense, t_{y-1}, t_y)
//!
//! Without the assist, the verifier must do K branching program evaluations.
//! With the assist, the verifier:
//! 1. Receives K claimed evaluations v_y from the prover
//! 2. Verifies them via a batch sumcheck (random linear combination)
//! 3. Does ONE branching program evaluation at a random point
//!
//! **Key insight**: K = number of POLYNOMIALS in the jagged collection, NOT the number
//! of virtualization rows. The cumulative sizes t_k come from the jagged bijection.
//!
//! This is a pure sumcheck protocol - no PCS commitment required.

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::marker::PhantomData;

use super::branching_program::{
    bit_to_field, get_coordinate_info, CoordType, JaggedBranchingProgram, Point,
};
use crate::zkvm::recursion::bijection::VarCountJaggedBijection;

/// Proof for the Jagged Assist batch MLE verification
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JaggedAssistProof<F: JoltField, T: Transcript> {
    /// Claimed evaluations v_k = ĝ(r_x, r_dense, t_{k-1}, t_k) for each polynomial k
    pub claimed_evaluations: Vec<F>,
    /// Sumcheck proof for the batch verification
    pub sumcheck_proof: crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
}

/// Parameters for the Jagged Assist sumcheck
#[derive(Clone, Debug)]
pub struct JaggedAssistParams<F: JoltField> {
    /// Number of bits for the branching program (log of dense size)
    pub num_bits: usize,
    /// Number of polynomials K
    pub num_polynomials: usize,
    /// The evaluation points x_k = (r_x, r_dense, t_{k-1}, t_k) for each polynomial
    pub evaluation_points: Vec<JaggedAssistEvalPoint<F>>,
    /// Batching randomness r (powers r^k used as coefficients)
    pub batching_base: F,
}

/// A single evaluation point for the branching program
#[derive(Clone, Debug)]
pub struct JaggedAssistEvalPoint<F: JoltField> {
    /// r_x (constraint challenge point)
    pub r_x: Vec<F>,
    /// r_dense (dense sumcheck challenge point)
    pub r_dense: Vec<F>,
    /// t_{k-1} (cumulative size before this polynomial)
    pub t_prev: usize,
    /// t_k (cumulative size including this polynomial)
    pub t_curr: usize,
}

/// Prover for the Jagged Assist batch MLE verification
///
/// Uses the forward-backward decomposition (Lemma 4.6) for efficient sumcheck.
/// Variables are processed in **interleaved order**: (a₀,b₀,c₀,d₀, a₁,b₁,c₁,d₁, ...)
/// which aligns with ROBP layers for O(K × n × w²) complexity.
pub struct JaggedAssistProver<F: JoltField, T: Transcript> {
    /// Parameters
    pub params: JaggedAssistParams<F>,
    /// Claimed evaluations (computed by prover)
    pub claimed_evaluations: Vec<F>,
    /// Powers of batching randomness: r^0, r^1, ..., r^{K-1}
    pub r_powers: Vec<F>,
    /// Current sumcheck round (0 to 4*num_bits - 1)
    pub round: usize,
    /// Number of sumcheck variables (4 * num_bits for a, b, c, d)
    pub num_sumcheck_vars: usize,
    /// Cached eq polynomial values for efficient sumcheck
    /// eq_cache[k] = eq(bound_challenges, x_k[0..round]) for all k
    eq_cache: Vec<F>,

    // Forward-backward decomposition state (Lemma 4.6)
    /// Forward states for each polynomial: forward_states[k][s] = prob of reaching state s
    /// after processing completed ROBP layers
    forward_states: Vec<[F; 4]>,
    /// Precomputed backward vectors for each polynomial
    /// backward_states[k][layer][s] = prob of reaching accept from state s at layer
    backward_states: Vec<Vec<[F; 4]>>,
    /// Accumulated challenges for the current (incomplete) ROBP layer
    /// Order: [a_challenge, b_challenge, c_challenge, d_challenge]
    current_layer_challenges: Vec<F>,

    /// The branching program (cached for efficiency)
    branching_program: JaggedBranchingProgram,
    /// Phantom
    _marker: PhantomData<T>,
}

impl<F: JoltField, T: Transcript> JaggedAssistProver<F, T> {
    /// Create a new Jagged Assist prover
    ///
    /// # Arguments
    /// * `r_x` - Constraint challenge point from Stage 1
    /// * `r_dense` - Dense challenge point from Stage 3
    /// * `bijection` - The jagged bijection (provides K and cumulative sizes)
    /// * `num_bits` - Number of bits for branching program
    /// * `transcript` - For sampling batching randomness
    pub fn new(
        r_x: Vec<F>,
        r_dense: Vec<F>,
        bijection: &VarCountJaggedBijection,
        num_bits: usize,
        transcript: &mut T,
    ) -> Self {
        let num_polynomials = bijection.num_polynomials();
        let branching_program = JaggedBranchingProgram::new(num_bits);

        // Step 1: Compute claimed evaluations v_k = ĝ(r_x, r_dense, t_{k-1}, t_k)
        // Parallelized over all K polynomials
        let za = Point::from_slice(&r_x);
        let zb = Point::from_slice(&r_dense);

        // Build evaluation points (lightweight, serial)
        let evaluation_points: Vec<JaggedAssistEvalPoint<F>> = (0..num_polynomials)
            .map(|poly_idx| {
                let t_prev = bijection.cumulative_size_before(poly_idx);
                let t_curr = bijection.cumulative_size(poly_idx);
                JaggedAssistEvalPoint {
                    r_x: r_x.clone(),
                    r_dense: r_dense.clone(),
                    t_prev,
                    t_curr,
                }
            })
            .collect();

        // Parallel computation of claimed evaluations and backward states
        let (claimed_evaluations, backward_states): (Vec<F>, Vec<Vec<[F; 4]>>) = evaluation_points
            .par_iter()
            .map(|eval_point| {
                let zc = Point::from_usize(eval_point.t_prev, num_bits);
                let zd = Point::from_usize(eval_point.t_curr, num_bits);

                let v_k = branching_program.eval_multilinear(&za, &zb, &zc, &zd);
                let backward =
                    branching_program.precompute_backward(&r_x, &r_dense, eval_point.t_prev, eval_point.t_curr);

                (v_k, backward)
            })
            .unzip();

        // Step 2: Append claimed evaluations to transcript
        for v in &claimed_evaluations {
            transcript.append_scalar(v);
        }

        // Step 3: Sample batching randomness
        let batching_base: F = transcript.challenge_scalar();

        // Compute powers: r^0, r^1, ..., r^{K-1}
        let mut r_powers = Vec::with_capacity(num_polynomials);
        let mut r_pow = F::one();
        for _ in 0..num_polynomials {
            r_powers.push(r_pow);
            r_pow *= batching_base;
        }

        let params = JaggedAssistParams {
            num_bits,
            num_polynomials,
            evaluation_points,
            batching_base,
        };

        // The sumcheck is over 4 * num_bits variables (a, b, c, d)
        let num_sumcheck_vars = 4 * num_bits;

        // Initialize eq_cache - starts with all ones (eq at empty prefix)
        let eq_cache = vec![F::one(); num_polynomials];

        // Initialize forward states - all mass in initial state
        let forward_states: Vec<[F; 4]> = (0..num_polynomials)
            .map(|_| JaggedBranchingProgram::initial_forward_state())
            .collect();

        Self {
            params,
            claimed_evaluations,
            r_powers,
            round: 0,
            num_sumcheck_vars,
            eq_cache,
            forward_states,
            backward_states,
            current_layer_challenges: Vec::with_capacity(4),
            branching_program,
            _marker: PhantomData,
        }
    }

    /// Get the input claim: Σ_k r^k · v_k
    pub fn get_input_claim(&self) -> F {
        self.r_powers
            .iter()
            .zip(&self.claimed_evaluations)
            .map(|(r_k, v_k)| *r_k * *v_k)
            .sum()
    }

    /// Extract the claimed evaluations (for inclusion in proof)
    pub fn get_claimed_evaluations(&self) -> &[F] {
        &self.claimed_evaluations
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for JaggedAssistProver<F, T> {
    fn degree(&self) -> usize {
        // P(b) = g(b) · Σ_k r^k · eq(b, x_k)
        // g(b) is 0/1 on hypercube, eq is multilinear
        // Product is degree 2 in each variable
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_sumcheck_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.get_input_claim()
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let num_bits = self.params.num_bits;
        let num_polynomials = self.params.num_polynomials;

        // Determine which coordinate (a, b, c, d) and layer this round corresponds to
        let (coord_type, layer) = get_coordinate_info(self.round, num_bits);
        let coord_idx = self.round % 4; // 0=a, 1=b, 2=c, 3=d

        // ============ OPTIMIZATION: Precompute transition matrices ============
        // The T matrix only depends on (za, zb, zc, zd).
        // - Some coordinates are challenges (same for all polynomials)
        // - Some are bits from t_prev/t_curr (only 0 or 1)
        // - One is λ (only 0, 1, or 2)
        // So there are at most 3 × 2 × 2 = 12 unique T matrices per round.

        // Get the fixed challenge values for coordinates already bound
        let za_fixed = if coord_idx > 0 {
            Some(self.current_layer_challenges[0])
        } else {
            None
        };
        let zb_fixed = if coord_idx > 1 {
            Some(self.current_layer_challenges[1])
        } else {
            None
        };
        let zc_fixed = if coord_idx > 2 {
            Some(self.current_layer_challenges[2])
        } else {
            None
        };

        // Precompute T matrices for all (lambda, zc_bit, zd_bit) combinations
        // Index: t_matrices[lambda_idx][zc_bit][zd_bit]
        // But some indices may not be used depending on coord_idx
        let t_matrices_init = [[[[F::zero(); 4]; 4]; 2]; 2]; // [zc_bit][zd_bit][from][to]
        let mut t_matrices_by_lambda = [t_matrices_init, t_matrices_init, t_matrices_init]; // [lambda][zc][zd]

        for lambda_idx in 0..3 {
            let lambda = F::from_u64(lambda_idx as u64);

            // Determine which (zc_bit, zd_bit) combinations are possible
            let zc_values: Vec<F> = if coord_idx == 2 {
                vec![lambda] // zc = λ, same for all
            } else if let Some(zc) = zc_fixed {
                vec![zc] // zc is a fixed challenge
            } else {
                vec![F::zero(), F::one()] // zc is a bit (0 or 1)
            };

            let zd_values: Vec<F> = if coord_idx == 3 {
                vec![lambda] // zd = λ, same for all
            } else {
                vec![F::zero(), F::one()] // zd is a bit (0 or 1)
            };

            for (zc_idx, &zc) in zc_values.iter().enumerate() {
                for (zd_idx, &zd) in zd_values.iter().enumerate() {
                    // Compute (za, zb) for this combination
                    let za = if coord_idx == 0 {
                        lambda
                    } else {
                        za_fixed.unwrap()
                    };

                    let zb = if coord_idx == 1 {
                        lambda
                    } else if let Some(zb) = zb_fixed {
                        zb
                    } else {
                        // zb comes from r_dense[layer], same for all polynomials
                        self.params.evaluation_points[0]
                            .r_dense
                            .get(layer)
                            .cloned()
                            .unwrap_or(F::zero())
                    };

                    let t = self.branching_program.transition_matrix_full(za, zb, zc, zd);
                    t_matrices_by_lambda[lambda_idx][zc_idx][zd_idx] = t;
                }
            }
        }

        // ============ Main loop: apply precomputed T matrices (PARALLELIZED) ============
        // Parallel sum over all K polynomials for each λ value
        let evals: [F; 3] = std::array::from_fn(|lambda_idx| {
            let lambda = F::from_u64(lambda_idx as u64);

            (0..num_polynomials)
                .into_par_iter()
                .map(|k| {
                    let eval_point = &self.params.evaluation_points[k];

                    // Get the x_k coordinate value for eq computation
                    let x_k_coord = match coord_type {
                        CoordType::A => eval_point.r_x.get(layer).cloned().unwrap_or(F::zero()),
                        CoordType::B => eval_point.r_dense.get(layer).cloned().unwrap_or(F::zero()),
                        CoordType::C => bit_to_field((eval_point.t_prev >> layer) & 1),
                        CoordType::D => bit_to_field((eval_point.t_curr >> layer) & 1),
                    };

                    // Compute eq contribution
                    let eq_lambda =
                        lambda * x_k_coord + (F::one() - lambda) * (F::one() - x_k_coord);
                    let eq_contrib = self.r_powers[k] * self.eq_cache[k] * eq_lambda;

                    // Look up the correct T matrix based on (zc_bit, zd_bit)
                    let zc_idx = if coord_idx >= 2 {
                        0 // zc is λ or a fixed challenge, use index 0
                    } else {
                        (eval_point.t_prev >> layer) & 1
                    };

                    let zd_idx = if coord_idx == 3 {
                        0 // zd is λ, use index 0
                    } else {
                        (eval_point.t_curr >> layer) & 1
                    };

                    let t_matrix = &t_matrices_by_lambda[lambda_idx][zc_idx][zd_idx];

                    // Apply F · T · B using precomputed T
                    let g_contrib = JaggedBranchingProgram::apply_transition_matrix(
                        &self.forward_states[k],
                        t_matrix,
                        &self.backward_states[k][layer + 1],
                    );

                    g_contrib * eq_contrib
                })
                .sum()
        });

        UniPoly::from_evals(&evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        let r: F = r_j.into();
        let num_bits = self.params.num_bits;

        // Determine which coordinate and layer
        let (coord_type, layer) = get_coordinate_info(self.round, num_bits);
        let coord_idx = self.round % 4; // 0=a, 1=b, 2=c, 3=d

        // Update eq_cache with the new challenge (parallelized)
        self.eq_cache
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, eq_val)| {
                let x_k_coord = match coord_type {
                    CoordType::A => self.params.evaluation_points[k]
                        .r_x
                        .get(layer)
                        .cloned()
                        .unwrap_or(F::zero()),
                    CoordType::B => self.params.evaluation_points[k]
                        .r_dense
                        .get(layer)
                        .cloned()
                        .unwrap_or(F::zero()),
                    CoordType::C => bit_to_field((self.params.evaluation_points[k].t_prev >> layer) & 1),
                    CoordType::D => bit_to_field((self.params.evaluation_points[k].t_curr >> layer) & 1),
                };
                let eq_factor = r * x_k_coord + (F::one() - r) * (F::one() - x_k_coord);
                *eq_val *= eq_factor;
            });

        // Track the challenge for the current layer
        self.current_layer_challenges.push(r);

        // If we've completed a full ROBP layer (4 variables: a, b, c, d)
        if coord_idx == 3 {
            // Update forward states for all polynomials (parallelized)
            let za = self.current_layer_challenges[0];
            let zb = self.current_layer_challenges[1];
            let zc = self.current_layer_challenges[2];
            let zd = self.current_layer_challenges[3];

            self.forward_states.par_iter_mut().for_each(|fwd| {
                *fwd = self.branching_program.update_forward(fwd, za, zb, zc, zd);
            });

            // Clear for next layer
            self.current_layer_challenges.clear();
        }

        self.round += 1;
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // No polynomial commitments for Jagged Assist - it's a pure sumcheck
    }
}

/// Verifier for the Jagged Assist batch MLE verification
pub struct JaggedAssistVerifier<F: JoltField, T: Transcript> {
    /// Claimed evaluations from prover
    pub claimed_evaluations: Vec<F>,
    /// Parameters (computed by verifier)
    pub params: JaggedAssistParams<F>,
    /// Powers of batching randomness
    pub r_powers: Vec<F>,
    /// Number of sumcheck variables
    pub num_sumcheck_vars: usize,
    /// Phantom
    _marker: PhantomData<T>,
}

impl<F: JoltField, T: Transcript> JaggedAssistVerifier<F, T> {
    /// Create a new Jagged Assist verifier
    #[tracing::instrument(skip_all, name = "JaggedAssistVerifier::new")]
    pub fn new(
        claimed_evaluations: Vec<F>,
        r_x: Vec<F>,
        r_dense: Vec<F>,
        bijection: &VarCountJaggedBijection,
        num_bits: usize,
        transcript: &mut T,
    ) -> Self {
        let num_polynomials = bijection.num_polynomials();

        // Build evaluation points (same as prover)
        let mut evaluation_points = Vec::with_capacity(num_polynomials);
        for poly_idx in 0..num_polynomials {
            let t_prev = bijection.cumulative_size_before(poly_idx);
            let t_curr = bijection.cumulative_size(poly_idx);
            evaluation_points.push(JaggedAssistEvalPoint {
                r_x: r_x.clone(),
                r_dense: r_dense.clone(),
                t_prev,
                t_curr,
            });
        }

        // Append claimed evaluations to transcript (must match prover)
        for v in &claimed_evaluations {
            transcript.append_scalar(v);
        }

        // Sample batching randomness (must match prover)
        let batching_base: F = transcript.challenge_scalar();

        // Compute powers
        let mut r_powers = Vec::with_capacity(num_polynomials);
        let mut r_pow = F::one();
        for _ in 0..num_polynomials {
            r_powers.push(r_pow);
            r_pow *= batching_base;
        }

        let params = JaggedAssistParams {
            num_bits,
            num_polynomials,
            evaluation_points,
            batching_base,
        };

        let num_sumcheck_vars = 4 * num_bits;

        Self {
            claimed_evaluations,
            params,
            r_powers,
            num_sumcheck_vars,
            _marker: PhantomData,
        }
    }

    /// Compute f̂_jagged using the verified claimed evaluations
    ///
    /// Call this AFTER the sumcheck has verified the claimed evaluations.
    /// f̂_jagged = Σ_k eq(r_s, matrix_row[k]) · v_k
    ///
    /// Note: This requires the matrix_row mapping from polynomial index to virtualization row.
    pub fn compute_f_jagged_with_mapping(&self, eq_r_s: &[F], matrix_rows: &[usize]) -> F {
        self.claimed_evaluations
            .iter()
            .enumerate()
            .map(|(k, v_k)| {
                let row = matrix_rows[k];
                eq_r_s.get(row).cloned().unwrap_or(F::zero()) * *v_k
            })
            .sum()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for JaggedAssistVerifier<F, T> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_sumcheck_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        // Input claim: Σ_k r^k · v_k
        self.r_powers
            .iter()
            .zip(&self.claimed_evaluations)
            .map(|(r_k, v_k)| *r_k * *v_k)
            .sum()
    }

    #[tracing::instrument(skip_all, name = "JaggedAssistVerifier::expected_output_claim")]
    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Final verification:
        // 1. Compute ĝ(ρ) via ONE branching program evaluation
        // 2. Compute Σ_k r^k · eq(ρ, x_k) efficiently by factoring out common terms
        // 3. Return ĝ(ρ) · eq_sum

        let num_bits = self.params.num_bits;
        let branching_program = JaggedBranchingProgram::new(num_bits);

        // Extract ρ = (ρ_a, ρ_b, ρ_c, ρ_d) from sumcheck challenges
        // Challenges are in INTERLEAVED order: (a₀,b₀,c₀,d₀, a₁,b₁,c₁,d₁, ...)
        let rho: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();

        let mut rho_a = Vec::with_capacity(num_bits);
        let mut rho_b = Vec::with_capacity(num_bits);
        let mut rho_c = Vec::with_capacity(num_bits);
        let mut rho_d = Vec::with_capacity(num_bits);

        for layer in 0..num_bits {
            let base = layer * 4;
            rho_a.push(rho[base]);
            rho_b.push(rho[base + 1]);
            rho_c.push(rho[base + 2]);
            rho_d.push(rho[base + 3]);
        }

        // ONE branching program evaluation
        let g_at_rho = branching_program.eval_multilinear(
            &Point::from(rho_a.clone()),
            &Point::from(rho_b.clone()),
            &Point::from(rho_c.clone()),
            &Point::from(rho_d.clone()),
        );

        // Compute eq_sum = Σ_k r^k · eq(ρ, x_k) efficiently
        // Key insight: r_x and r_dense are the SAME for all K polynomials
        // So: eq(ρ, x_k) = eq(ρ_a, r_x) · eq(ρ_b, r_dense) · eq(ρ_c, t_{k-1}) · eq(ρ_d, t_k)
        //                = eq_ab · eq_c[t_{k-1}] · eq_d[t_k]
        // where eq_ab is constant for all k

        // Get r_x and r_dense from first eval point (same for all)
        // Pad to num_bits with zeros to match rho_a/rho_b length
        let r_x = &self.params.evaluation_points[0].r_x;
        let r_dense = &self.params.evaluation_points[0].r_dense;

        let mut r_x_padded = r_x.clone();
        r_x_padded.resize(num_bits, F::zero());
        let mut r_dense_padded = r_dense.clone();
        r_dense_padded.resize(num_bits, F::zero());

        // Compute eq_ab = eq(ρ_a, r_x) · eq(ρ_b, r_dense) - O(num_bits)
        let eq_a = EqPolynomial::mle(&rho_a, &r_x_padded);
        let eq_b = EqPolynomial::mle(&rho_b, &r_dense_padded);
        let eq_ab = eq_a * eq_b;

        // Precompute eq_c and eq_d tables - O(2^num_bits)
        // Note: EqPolynomial::evals uses big-endian, but t values use LSB-first,
        // so we reverse rho_c and rho_d to match index ordering
        let rho_c_rev: Vec<F> = rho_c.iter().rev().cloned().collect();
        let rho_d_rev: Vec<F> = rho_d.iter().rev().cloned().collect();
        let eq_c_evals = EqPolynomial::<F>::evals(&rho_c_rev);
        let eq_d_evals = EqPolynomial::<F>::evals(&rho_d_rev);

        // Compute Σ_k r^k · eq_c[t_{k-1}] · eq_d[t_k] - O(K)
        let eq_cd_sum: F = self
            .r_powers
            .iter()
            .zip(&self.params.evaluation_points)
            .map(|(r_k, eval_point)| {
                *r_k * eq_c_evals[eval_point.t_prev] * eq_d_evals[eval_point.t_curr]
            })
            .sum();

        g_at_rho * eq_ab * eq_cd_sum
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // No polynomial commitments for Jagged Assist - it's a pure sumcheck
    }
}
