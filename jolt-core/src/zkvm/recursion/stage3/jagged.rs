//! Stage 3: Jagged Transform Sumcheck
//!
//! This stage performs a sumcheck to verify the jagged-to-dense polynomial transform
//! following the technique from "Jagged Polynomial Commitments" paper.
//!
//! The jagged transform maps a sparse representation (with padding) to a dense
//! representation that excludes redundant values. In our case:
//! - GT operations use 4-variable MLEs (16 evaluations)
//! - G1 operations use 8-variable MLEs (256 evaluations)
//! - In the sparse matrix, 4-var MLEs are padded to 8 vars by repeating each value 16x
//! - The dense representation removes this redundancy
//!
//! Protocol:
//! 1. Input: Opening claim from Stage 2: M(r_s_final, r_x_prev) = v_sparse
//! 2. Sumcheck proves: v_sparse = Σ_i q(i) · f_jagged(r_s_final, r_x_prev, i)
//! 3. Output: Dense polynomial opening claim q(r_dense) = v_dense
//!
//! Where:
//! - M(s,x) is the sparse constraint matrix (virtualized in Stage 2)
//! - q(i) is the dense polynomial containing only non-redundant entries
//! - f_jagged(r_s, r_x, i) = eq(row(i), r_s) · eq(col(i), r_x) for boolean i
//!
//! For field element evaluation (verifier side), we use Claim 3.2.1 from the paper:
//!   f̂_jagged(r_s, r_x, r_dense) = Σ_{y∈{0,1}^k} eq(r_x, y) · ĝ(r_s, r_dense, t_{y-1}, t_y)
//! where g(a, b, c, d) = 1 iff b < d and b = a + c
//! - f_jagged is an indicator function implementing the sparse-to-dense bijection
//! - r_s_final, r_x_prev are the evaluation points from Stage 2
//! - r_dense is the final sumcheck challenge point

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};
use rayon::prelude::*;

use crate::zkvm::recursion::bijection::{ConstraintMapping, VarCountJaggedBijection};

/// Parameters for Stage 3 jagged sumcheck
#[derive(Clone)]
pub struct JaggedSumcheckParams {
    /// Number of s variables (from sparse matrix)
    pub num_s_vars: usize,
    /// Number of constraint variables (x)
    pub num_constraint_vars: usize,
    /// Number of dense variables (for the dense polynomial)
    pub num_dense_vars: usize,
    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
    /// Committed polynomial for dense matrix
    pub polynomial: CommittedPolynomial,
}

impl JaggedSumcheckParams {
    pub fn new(num_s_vars: usize, num_constraint_vars: usize, num_dense_vars: usize) -> Self {
        Self {
            num_s_vars,
            num_constraint_vars,
            num_dense_vars,
            sumcheck_id: SumcheckId::RecursionJagged,
            polynomial: CommittedPolynomial::DoryDenseMatrix,
        }
    }

    /// Total sumcheck rounds: all dense variables
    pub fn num_rounds(&self) -> usize {
        self.num_dense_vars
    }
}

/// Stage 3 prover that reduces sparse matrix claims to dense polynomial claims
pub struct JaggedSumcheckProver<F: JoltField, T: Transcript> {
    /// Parameters
    pub params: JaggedSumcheckParams,

    /// Opening claim from Stage 2: M(r_s_final, r_x_prev) = v
    pub sparse_opening_point_s: Vec<F>,
    pub sparse_opening_point_x: Vec<F>,
    pub sparse_claim_value: F,

    /// Dense polynomial q(i) containing only non-zero entries
    pub dense_poly: MultilinearPolynomial<F>,

    /// Jagged indicator polynomial \hat f_jagged(r_s, r_x, i) as an MLE over the dense index `i`.
    ///
    /// This is the multilinear extension of the boolean function:
    /// `f(i) = eq(row(i), r_s) * eq(col(i), r_x)` for `i < dense_size`, and `0` for padded indices.
    pub jagged_indicator_poly: MultilinearPolynomial<F>,

    /// Equality polynomial eq(r_s, s) where s comes from i_sparse(i)
    pub eq_r_s: MultilinearPolynomial<F>,

    /// Equality polynomial eq(r_x, x) where x comes from i_sparse(i)
    pub eq_r_x: MultilinearPolynomial<F>,

    /// Bijection for sparse-to-dense mapping
    pub bijection: VarCountJaggedBijection,

    /// Mapping for decoding polynomial indices to matrix rows
    pub mapping: ConstraintMapping,

    /// Precomputed matrix row indices for each polynomial index
    pub matrix_rows: Vec<usize>,

    /// Current round number
    pub round: usize,

    /// Phantom data
    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> JaggedSumcheckProver<F, T> {
    pub fn new(
        sparse_opening_point: (Vec<F>, Vec<F>),
        sparse_claim_value: F,
        dense_poly: DensePolynomial<F>,
        bijection: VarCountJaggedBijection,
        mapping: ConstraintMapping,
        matrix_rows: Vec<usize>,
        _transcript: &mut T,
        num_s_vars: usize,
        num_constraint_vars: usize,
    ) -> Self {
        let (r_s_final, r_x_prev) = sparse_opening_point;
        let num_dense_vars = dense_poly.get_num_vars();

        // --------------------------------------------------------------------
        // Build the jagged indicator polynomial over the dense index `i`.
        //
        // IMPORTANT: The Stage 3 sumcheck operates over the multilinear extension of this
        // indicator, so the prover must be able to evaluate it at non-boolean points as the
        // verifier challenges are absorbed. We accomplish this by materializing the boolean
        // evaluation table once and then binding it in `ingest_challenge` (just like `dense_poly`).
        // --------------------------------------------------------------------

        // We intentionally compute `eq` values using `EqPolynomial::mle` with LSB-first bit vectors,
        // matching the conventions used elsewhere in Stage 3 (and in the verifier's "old method"
        // checks). This avoids subtle endianness issues with `EqPolynomial::evals`.

        let num_polynomials = bijection.num_polynomials();
        let mut eq_s_cache = vec![F::zero(); num_polynomials];
        for poly_idx in 0..num_polynomials {
            let matrix_row = matrix_rows[poly_idx];
            let row_bits = index_to_binary_vec::<F>(matrix_row, num_s_vars);
            eq_s_cache[poly_idx] = EqPolynomial::mle(&row_bits, &r_s_final);
        }

        let max_cols = 1usize << num_constraint_vars;
        let mut eq_x_cache = vec![F::zero(); max_cols];
        for x_idx in 0..max_cols {
            let x_bits = index_to_binary_vec::<F>(x_idx, num_constraint_vars);
            eq_x_cache[x_idx] = EqPolynomial::mle(&x_bits, &r_x_prev);
        }

        let padded_dense_size = 1usize << num_dense_vars;
        let mut jagged_indicator_evals = vec![F::zero(); padded_dense_size];
        for poly_idx in 0..num_polynomials {
            let t_prev = bijection.cumulative_size_before(poly_idx);
            let t_curr = bijection.cumulative_size(poly_idx);
            let native_size = t_curr - t_prev;
            let eq_s = eq_s_cache[poly_idx];
            for col in 0..native_size {
                jagged_indicator_evals[t_prev + col] = eq_s * eq_x_cache[col];
            }
        }

        // Initialize eq(r_s, s) polynomial - starts as delta function at r_s
        let eq_r_s = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_s_final));

        // Initialize eq(r_x, x) polynomial - starts as delta function at r_x
        let eq_r_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x_prev));

        Self {
            params: JaggedSumcheckParams::new(num_s_vars, num_constraint_vars, num_dense_vars),
            sparse_opening_point_s: r_s_final,
            sparse_opening_point_x: r_x_prev,
            sparse_claim_value,
            dense_poly: MultilinearPolynomial::from(dense_poly.Z),
            jagged_indicator_poly: MultilinearPolynomial::from(jagged_indicator_evals),
            eq_r_s,
            eq_r_x,
            bijection,
            mapping,
            matrix_rows,
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Helper function to convert index to binary representation
fn index_to_binary_vec<F: JoltField>(index: usize, num_vars: usize) -> Vec<F> {
    let mut binary = Vec::with_capacity(num_vars);
    let mut idx = index;

    for _ in 0..num_vars {
        binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
        idx >>= 1;
    }

    binary
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for JaggedSumcheckProver<F, T> {
    fn degree(&self) -> usize {
        2 // Degree from q(i) * f_jagged(i) where f_jagged is degree 1
    }

    fn num_rounds(&self) -> usize {
        self.params.num_dense_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.sparse_claim_value
    }

    #[tracing::instrument(skip_all, name = "Jagged::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 2;
        let num_vars_remaining = self.dense_poly.get_num_vars();

        if num_vars_remaining == 0 {
            return UniPoly::from_coeff(vec![self.dense_poly.get_bound_coeff(0)]);
        }

        let half = 1 << (num_vars_remaining - 1);

        // Compute the Stage 3 sumcheck message for:
        //   g(i) = q(i) * \hat f_jagged(r_s, r_x, i)
        //
        // Both `q` and `\hat f_jagged` are multilinear in `i`, so we can obtain their
        // needed univariate evaluations via `sumcheck_evals_array` and multiply pointwise.
        let total_evals = (0..half)
            .into_par_iter()
            .map(|suffix_idx| {
                let q_evals = self
                    .dense_poly
                    .sumcheck_evals_array::<DEGREE>(suffix_idx, BindingOrder::LowToHigh);
                let f_evals = self
                    .jagged_indicator_poly
                    .sumcheck_evals_array::<DEGREE>(suffix_idx, BindingOrder::LowToHigh);

                let mut evals = [F::zero(); DEGREE];
                for t in 0..DEGREE {
                    evals[t] = q_evals[t] * f_evals[t];
                }
                evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    #[tracing::instrument(skip_all, name = "Jagged::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.dense_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.jagged_indicator_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // After all sumcheck rounds, we have the opening claim on the dense polynomial
        let dense_claim = self.dense_poly.get_bound_coeff(0);

        // The opening point must be in BIG_ENDIAN order for PCS verification.
        // Sumcheck challenges arrive low-to-high (LSB-first), so reverse before storing.
        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            sumcheck_challenges.iter().rev().cloned().collect(),
            dense_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Stage 3 verifier
pub struct JaggedSumcheckVerifier<F: JoltField> {
    /// Parameters
    pub params: JaggedSumcheckParams,

    /// Opening claim from Stage 2
    pub sparse_opening_point_s: Vec<F>,
    pub sparse_opening_point_x: Vec<F>,
    pub sparse_claim_value: F,

    /// Bijection metadata
    pub bijection: VarCountJaggedBijection,

    /// Mapping for decoding polynomial indices to matrix rows
    pub mapping: ConstraintMapping,

    /// Precomputed matrix row indices for each polynomial index
    pub matrix_rows: Vec<usize>,

    /// Precomputed cumulative sizes for each matrix row
    /// row_cumulative_sizes[i] = total size of all polynomials in rows 0..i
    pub row_cumulative_sizes: Vec<usize>,

    /// Claimed evaluations from Jagged Assist (Stage 3b)
    /// Used for O(K) f̂_jagged computation instead of O(K × branching_program)
    /// claimed_evaluations[y] = v_y = ĝ(r_x, r_dense, t_{y-1}, t_y)
    pub claimed_evaluations: Vec<F>,
}

impl<F: JoltField> JaggedSumcheckVerifier<F> {
    pub fn new(
        sparse_opening_point: (Vec<F>, Vec<F>),
        sparse_claim_value: F,
        bijection: VarCountJaggedBijection,
        mapping: ConstraintMapping,
        matrix_rows: Vec<usize>,
        params: JaggedSumcheckParams,
        claimed_evaluations: Vec<F>,
    ) -> Self {
        let _new_span = tracing::info_span!(
            "JaggedSumcheckVerifier::new",
            num_polys = bijection.num_polynomials(),
            num_s_vars = params.num_s_vars
        )
        .entered();

        let (r_s_final, r_x_prev) = sparse_opening_point;

        let _precompute_span = tracing::info_span!("precompute_row_cumulative_sizes").entered();
        // Build mapping from matrix row to cumulative size
        // This is needed to implement the formula correctly
        let num_rows = 1 << params.num_s_vars; // 2^num_s_vars
        let mut row_cumulative_sizes = vec![0usize; num_rows + 1]; // +1 for easier indexing

        // For each polynomial, add its size to its row's total
        for poly_idx in 0..bijection.num_polynomials() {
            let matrix_row = matrix_rows[poly_idx];
            let poly_size = if poly_idx == 0 {
                bijection.cumulative_size(0)
            } else {
                bijection.cumulative_size(poly_idx) - bijection.cumulative_size(poly_idx - 1)
            };
            row_cumulative_sizes[matrix_row + 1] += poly_size;
        }

        // Convert to cumulative sums
        for i in 1..=num_rows {
            row_cumulative_sizes[i] += row_cumulative_sizes[i - 1];
        }
        drop(_precompute_span);

        Self {
            params,
            sparse_opening_point_s: r_s_final,
            sparse_opening_point_x: r_x_prev,
            sparse_claim_value,
            bijection,
            mapping,
            matrix_rows,
            row_cumulative_sizes,
            claimed_evaluations,
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.params.num_dense_vars
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for JaggedSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        2 // Degree from q(i) * f_jagged(i) where f_jagged is degree 1
    }

    fn num_rounds(&self) -> usize {
        self.params.num_dense_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.sparse_claim_value
    }

    #[tracing::instrument(skip_all, name = "JaggedSumcheckVerifier::expected_output_claim")]
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the dense polynomial opening claim from the accumulator
        let (_, dense_claim) = accumulator
            .get_committed_polynomial_opening(self.params.polynomial, self.params.sumcheck_id);

        // f̂_jagged = Σ_y eq(r_s, y) · v_y = v̂(r_s)
        // This is just evaluating the MLE of claimed_evaluations at sparse_opening_point_s
        // Note: EqPolynomial::evals uses big-endian bit ordering, but index_to_binary_vec
        // uses little-endian, so we reverse the point to match.
        let eq_evals = {
            let _span = tracing::info_span!(
                "jagged_eq_evals",
                num_vars = self.sparse_opening_point_s.len()
            )
            .entered();
            let r_s_reversed: Vec<F> = self.sparse_opening_point_s.iter().rev().cloned().collect();
            EqPolynomial::<F>::evals(&r_s_reversed)
        };

        let f_jagged_at_r_dense: F = {
            let _span =
                tracing::info_span!("jagged_inner_product", k = self.claimed_evaluations.len())
                    .entered();
            eq_evals
                .iter()
                .zip(self.claimed_evaluations.iter())
                .map(|(eq, v)| *eq * *v)
                .sum()
        };

        dense_claim * f_jagged_at_r_dense
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Register opening claim on dense polynomial
        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            // Stored in BIG_ENDIAN order for PCS verification.
            sumcheck_challenges.iter().rev().cloned().collect(),
        );
    }
}
