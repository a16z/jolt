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
    utils::errors::ProofVerifyError,
    zkvm::witness::CommittedPolynomial,
};
use rayon::prelude::*;

use crate::zkvm::recursion::bijection::{
    ConstraintMapping, JaggedTransform, VarCountJaggedBijection,
};
use ark_bn254::Fq;
use ark_ff::Zero;

use super::branching_program::{JaggedBranchingProgram, Point};

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
        transcript: &mut T,
        num_s_vars: usize,
        num_constraint_vars: usize,
    ) -> Self {
        let (r_s_final, r_x_prev) = sparse_opening_point;
        let num_dense_vars = dense_poly.get_num_vars();

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
            eq_r_s,
            eq_r_x,
            bijection,
            mapping,
            matrix_rows,
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn index_to_binary(index: usize, num_vars: usize) -> Vec<F> {
        let mut binary = Vec::with_capacity(num_vars);
        let mut idx = index;

        for _ in 0..num_vars {
            binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
            idx >>= 1;
        }

        binary
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

        // Following the paper: for boolean i, we use ft(zr, zc, i) = eq(rowt(i), zr) · eq(colt(i), zc)
        // This is equation (4) from the paper

        // Compute the sumcheck polynomial:
        // g(X) = Σ_{b ∈ {0,1}^{m-round-1}} q(X, b) * f_jagged(r_s, r_x, (X, b))
        let total_evals = (0..half)
            .into_par_iter()
            .map(|suffix_idx| {
                // Get q(X, suffix) evaluations
                let q_evals = self
                    .dense_poly
                    .sumcheck_evals_array::<DEGREE>(suffix_idx, BindingOrder::LowToHigh);

                let mut evals = [F::zero(); DEGREE];

                for t in 0..DEGREE {
                    // Compute the dense index for (t, suffix)
                    let dense_idx = (t << (num_vars_remaining - 1)) | suffix_idx;

                    // Check if this dense index is within the actual dense size
                    if dense_idx
                        < <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(
                            &self.bijection,
                        )
                    {
                        // For boolean i, use the direct formula from equation (4)
                        // ft(zr, zc, i) = eq(rowt(i), zr) · eq(colt(i), zc)

                        // Get the polynomial index and col (evaluation index)
                        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(
                            &self.bijection,
                            dense_idx,
                        );
                        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(
                            &self.bijection,
                            dense_idx,
                        );

                        // Get the actual matrix row from precomputed values
                        let matrix_row = self.matrix_rows[poly_idx];

                        // Convert matrix_row to binary representation (this is the s-index)
                        let s_binary = Self::index_to_binary(matrix_row, self.params.num_s_vars);
                        let eq_s = EqPolynomial::mle(&s_binary, &self.sparse_opening_point_s);

                        // Convert col to binary representation (this is the x-index)
                        let x_binary = Self::index_to_binary(col, self.params.num_constraint_vars);
                        let eq_x = EqPolynomial::mle(&x_binary, &self.sparse_opening_point_x);

                        #[cfg(test)]
                        if self.round == self.params.num_dense_vars - 1 && suffix_idx == 0 && t < 3
                        {
                            println!("\n  PROVER: Computing for dense_idx {}:", dense_idx);
                            println!("    poly_idx: {}, eval_idx: {}", poly_idx, col);
                            println!("    matrix_row: {}", matrix_row);
                            println!("    q_evals[{}]: {:?}", t, q_evals[t]);
                            println!("    eq_s: {:?}", eq_s);
                            println!("    eq_x: {:?}", eq_x);
                            println!(
                                "    f_jagged = q * eq_s * eq_x = {:?}",
                                q_evals[t] * eq_s * eq_x
                            );
                        }

                        // f_jagged(r_s, r_x, i) = eq(row(i), r_s) * eq(col(i), r_x)
                        evals[t] = q_evals[t] * eq_s * eq_x;
                    }
                    // If dense_idx >= dense_size, contribution is 0 (padding in dense polynomial)
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

        // The opening point is r_dense (the sumcheck challenges)
        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            sumcheck_challenges.to_vec(),
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
}

impl<F: JoltField> JaggedSumcheckVerifier<F> {
    pub fn new(
        sparse_opening_point: (Vec<F>, Vec<F>),
        sparse_claim_value: F,
        bijection: VarCountJaggedBijection,
        mapping: ConstraintMapping,
        matrix_rows: Vec<usize>,
        params: JaggedSumcheckParams,
    ) -> Self {
        let (r_s_final, r_x_prev) = sparse_opening_point;

        Self {
            params,
            sparse_opening_point_s: r_s_final,
            sparse_opening_point_x: r_x_prev,
            sparse_claim_value,
            bijection,
            mapping,
            matrix_rows,
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

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the dense polynomial opening claim from the accumulator
        let (_, dense_claim) = accumulator
            .get_committed_polynomial_opening(self.params.polynomial, self.params.sumcheck_id);

        // Convert challenges to field elements, reversing to match high-to-low order
        // (sumcheck produces low-to-high, but we need high-to-low for polynomial evaluation)
        let r_dense: Vec<F> = sumcheck_challenges
            .iter()
            // .rev()
            .map(|c| (*c).into())
            .collect();

        #[cfg(test)]
        {
            println!("\n=== VERIFIER EXPECTED_OUTPUT_CLAIM ===");
            println!("Dense claim from accumulator: {:?}", dense_claim);
            println!("r_dense (sumcheck challenges): {:?}", r_dense);
            println!("sparse_opening_point_s: {:?}", self.sparse_opening_point_s);
            println!("sparse_opening_point_x: {:?}", self.sparse_opening_point_x);
        }

        // Following the paper's Claim 3.2.1, adapted for row-based jaggedness:
        // Original formula: ˆft(zr, zc, i) = Σ_{y∈{0,1}^k} eq(zc, y) · ĝ(zr, i, t_{y-1}, t_y)
        // where g(a, b, c, d) = 1 if and only if b < d and b = a + c

        // Key adaptation: The paper assumes column-based jaggedness (different columns have
        // different heights), but our system has row-based jaggedness (different rows have
        // different widths). So we adapt the formula by:
        // - Iterating y over rows (polynomials) instead of columns
        // - Using row cumulative sizes instead of column cumulative heights
        // - Swapping the role of row/column indices in the eq polynomials
        //
        // Our adapted formula: ˆft(zr, zc, i) = Σ_{y∈rows} eq(zr, y) · ĝ(zc, i, t_{y-1}, t_y)
        // where:
        // - zr = self.sparse_opening_point_s (polynomial/row challenge)
        // - zc = self.sparse_opening_point_x (evaluation/column challenge)
        // - i = r_dense (the field element from sumcheck)
        // - y iterates over all polynomials (rows)

        let mut f_jagged_at_r_dense = F::zero();

        // Sum over all y ∈ {0,1}^k (all polynomials/rows)
        for poly_idx in 0..self.bijection.num_polynomials() {
            // Get t_{y-1} and t_y
            let t_prev = self.bijection.cumulative_size_before(poly_idx);
            let t_curr = self.bijection.cumulative_size(poly_idx);

            // Get the actual matrix row from precomputed values
            let matrix_row = self.matrix_rows[poly_idx];

            // Convert matrix_row to binary for eq evaluation
            // In our case: rows are polynomials (s-indexed), cols are evaluations (x-indexed)
            // The paper iterates y over columns, but our jaggedness is per-row
            // So we adapt: iterate over rows, and y represents the matrix row index
            let row_binary = index_to_binary_vec::<F>(matrix_row, self.params.num_s_vars);
            let eq_zr_y = EqPolynomial::mle(&row_binary, &self.sparse_opening_point_s);

            #[cfg(test)]
            if poly_idx < 3 {
                println!("\n  VERIFIER: Processing poly_idx {}:", poly_idx);
                println!("    matrix_row: {}", matrix_row);
                println!("    t_prev: {}, t_curr: {}", t_prev, t_curr);
                println!("    eq_zr_y: {:?}", eq_zr_y);
            }

            // Now we need to compute ĝ(zc, i, t_{y-1}, t_y) for the multilinear extension of g
            // g(a, b, c, d) = 1 iff b < d and b = a + c
            // In our adapted formula (with swapped row/col roles):
            // - a = zc (sparse_opening_point_x - the column/evaluation index)
            // - b = i (r_dense - the dense index)
            // - c = t_{y-1} (cumulative size before this row)
            // - d = t_y (cumulative size including this row)

            // Use the branching program optimization
            // The branching program computes g(a,b,c,d) = 1[b < d ∧ b = a + c]
            //
            // We need: Σ_{a,b} g(a,b,t_prev,t_curr) * eq(a, sparse_x) * eq(b, r_dense)
            //
            // But note that g(a,b,t_prev,t_curr) = 1 only when:
            // 1) b < t_curr
            // 2) b = a + t_prev
            //
            // So this sum equals: Σ_{a} 1[a + t_prev < t_curr] * eq(a, sparse_x) * eq(a + t_prev, r_dense)
            // Which is exactly what the original code computed!

            let num_bits =
                std::cmp::max(self.params.num_constraint_vars, self.params.num_dense_vars);

            // Create the branching program
            let prog = JaggedBranchingProgram::new(num_bits);

            // Compute ĝ(sparse_x, r_dense, t_prev, t_curr) directly
            // This gives us the MLE of g evaluated at our challenge points
            let za = Point::from_slice(&self.sparse_opening_point_x);
            let zb = Point::from_slice(&r_dense);
            let zc = Point::from_usize(t_prev, num_bits);
            let zd = Point::from_usize(t_curr, num_bits);

            let g_mle = prog.eval_multilinear(&za, &zb, &zc, &zd);

            #[cfg(test)]
            if poly_idx < 3 {
                println!("      Using branching program optimization:");
                println!("        sparse_x: {:?}", self.sparse_opening_point_x);
                println!("        r_dense: {:?}", r_dense);
                println!("        t_prev: {}, t_curr: {}", t_prev, t_curr);
                println!("        g_mle from branching program: {:?}", g_mle);

                // Let's also compute it the old way to compare
                let mut old_g_sum = F::zero();
                for x_idx in 0..(t_curr - t_prev) {
                    let dense_idx = t_prev + x_idx;
                    let x_binary = index_to_binary_vec::<F>(x_idx, self.params.num_constraint_vars);
                    let eq_a_zc = EqPolynomial::mle(&x_binary, &self.sparse_opening_point_x);
                    let dense_idx_binary =
                        index_to_binary_vec::<F>(dense_idx, self.params.num_dense_vars);
                    let eq_b_i = EqPolynomial::mle(&dense_idx_binary, &r_dense);
                    old_g_sum += eq_a_zc * eq_b_i;
                }
                println!("        g_sum from old method: {:?}", old_g_sum);
                println!("        Match: {}", g_mle == old_g_sum);
                println!("        total contribution: {:?}", eq_zr_y * g_mle);
            }

            // Add eq(zr, y) * ĝ contribution
            f_jagged_at_r_dense += eq_zr_y * g_mle;
        }

        #[cfg(test)]
        {
            println!("\n  VERIFIER: Final computation:");
            println!("    f_jagged_at_r_dense: {:?}", f_jagged_at_r_dense);
            println!("    dense_claim: {:?}", dense_claim);
            println!("    final result: {:?}", dense_claim * f_jagged_at_r_dense);
        }

        // Return q(r_dense) * f̂_jagged(r_s, r_x, r_dense)
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
            sumcheck_challenges.to_vec(),
        );
    }
}
