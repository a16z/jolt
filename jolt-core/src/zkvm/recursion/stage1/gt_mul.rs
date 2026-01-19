//! GT multiplication sumcheck for proving GT multiplication constraints
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * C_i(x)
//! Where C_i(x) = a_i(x) × b_i(x) - c_i(x) - Q_i(x) × g(x)
//!
//! This is a separate sumcheck protocol for GT multiplication constraints.
//! Output: Virtual polynomial claims for each polynomial in each constraint

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{recursion::utils::virtual_polynomial_utils::*, witness::VirtualPolynomial},
    virtual_claims,
};
use rayon::prelude::*;

/// Helper to append all virtual claims for a GT mul constraint
fn append_gt_mul_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    lhs_claim: F,
    rhs_claim: F,
    result_claim: F,
    quotient_claim: F,
) {
    let claims = virtual_claims![
        VirtualPolynomial::RecursionMulLhs(constraint_idx) => lhs_claim,
        VirtualPolynomial::RecursionMulRhs(constraint_idx) => rhs_claim,
        VirtualPolynomial::RecursionMulResult(constraint_idx) => result_claim,
        VirtualPolynomial::RecursionMulQuotient(constraint_idx) => quotient_claim,
    ];
    append_virtual_claims(accumulator, transcript, sumcheck_id, opening_point, &claims);
}

/// Helper to retrieve all virtual claims for a GT mul constraint
fn get_gt_mul_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (F, F, F, F) {
    let polynomials = vec![
        VirtualPolynomial::RecursionMulLhs(constraint_idx),
        VirtualPolynomial::RecursionMulRhs(constraint_idx),
        VirtualPolynomial::RecursionMulResult(constraint_idx),
        VirtualPolynomial::RecursionMulQuotient(constraint_idx),
    ];
    let claims = get_virtual_claims(accumulator, sumcheck_id, &polynomials);
    (claims[0], claims[1], claims[2], claims[3])
}

/// Helper to append virtual opening points for a GT mul constraint (verifier side)
fn append_gt_mul_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    let polynomials = vec![
        VirtualPolynomial::RecursionMulLhs(constraint_idx),
        VirtualPolynomial::RecursionMulRhs(constraint_idx),
        VirtualPolynomial::RecursionMulResult(constraint_idx),
        VirtualPolynomial::RecursionMulQuotient(constraint_idx),
    ];
    append_virtual_openings(accumulator, transcript, sumcheck_id, opening_point, &polynomials);
}

/// Individual polynomial data for a single GT mul constraint
#[derive(Clone)]
pub struct GtMulConstraintPolynomials<F: JoltField> {
    pub lhs: Vec<F>,
    pub rhs: Vec<F>,
    pub result: Vec<F>,
    pub quotient: Vec<F>,
    pub constraint_index: usize,
}

/// Parameters for GT mul sumcheck
#[derive(Clone)]
pub struct GtMulParams {
    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraints
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl GtMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // 11 vars for uniform matrix (4 element + 7 padding)
            num_constraints,
            sumcheck_id: SumcheckId::GtMul,
        }
    }
}

/// Prover for GT mul sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct GtMulProver<F: JoltField, T: Transcript> {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: GtMulParams,

    /// g(x) polynomial for constraint evaluation
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub g_poly: MultilinearPolynomial<F>,

    /// Equality polynomial for constraint variables x
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x: MultilinearPolynomial<F>,

    /// Random challenge for eq(r_x, x)
    pub r_x: Vec<F::Challenge>,

    /// Gamma coefficient for batching constraints
    pub gamma: F,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

    /// LHS polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub lhs_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// RHS polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rhs_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Result polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub result_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Quotient polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub quotient_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Current round in the protocol
    pub round: usize,

    /// Cached evaluation claims for each polynomial
    pub lhs_claims: Vec<F>,
    pub rhs_claims: Vec<F>,
    pub result_claims: Vec<F>,
    pub quotient_claims: Vec<F>,

    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> GtMulProver<F, T> {
    pub fn new(
        params: GtMulParams,
        constraint_polys: Vec<GtMulConstraintPolynomials<F>>,
        g_poly: DensePolynomial<F>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x[..]));

        let mut constraint_indices = Vec::new();
        let mut lhs_mlpoly = Vec::new();
        let mut rhs_mlpoly = Vec::new();
        let mut result_mlpoly = Vec::new();
        let mut quotient_mlpoly = Vec::new();

        for poly in constraint_polys {
            constraint_indices.push(poly.constraint_index);
            lhs_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.lhs,
            )));
            rhs_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.rhs,
            )));
            result_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.result,
            )));
            quotient_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.quotient,
            )));
        }

        let g_mlpoly = MultilinearPolynomial::LargeScalars(g_poly);

        Self {
            params,
            g_poly: g_mlpoly,
            eq_x,
            r_x,
            gamma: gamma.into(),
            constraint_indices,
            lhs_mlpoly,
            rhs_mlpoly,
            result_mlpoly,
            quotient_mlpoly,
            round: 0,
            lhs_claims: vec![],
            rhs_claims: vec![],
            result_claims: vec![],
            quotient_claims: vec![],
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for GtMulProver<F, T> {
    fn degree(&self) -> usize {
        3 // Degree from constraint: lhs * rhs
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // For GT mul constraints, the sumcheck proves 0 = Σ_x eq(r_x, x) * Σ_i γ^i * C_i(x)
        // So the initial claim is 0
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3; // Max degree for GT mul constraints
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                let g_evals = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [F::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                for i in 0..self.lhs_mlpoly.len() {
                    let lhs_evals = self.lhs_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let rhs_evals = self.rhs_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let result_evals = self.result_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let quotient_evals = self.quotient_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        // Constraint: lhs * rhs - result - quotient * g
                        let constraint_val = lhs_evals[t] * rhs_evals[t]
                            - result_evals[t]
                            - quotient_evals[t] * g_evals[t];

                        x_evals[t] += eq_x_evals[t] * gamma_power * constraint_val;
                    }

                    gamma_power *= self.gamma;
                }
                x_evals
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

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.lhs_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rhs_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.result_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.quotient_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        if self.round == self.params.num_constraint_vars {
            self.lhs_claims.clear();
            self.rhs_claims.clear();
            self.result_claims.clear();
            self.quotient_claims.clear();

            for i in 0..self.lhs_mlpoly.len() {
                self.lhs_claims.push(self.lhs_mlpoly[i].get_bound_coeff(0));
                self.rhs_claims.push(self.rhs_mlpoly[i].get_bound_coeff(0));
                self.result_claims
                    .push(self.result_mlpoly[i].get_bound_coeff(0));
                self.quotient_claims
                    .push(self.quotient_mlpoly[i].get_bound_coeff(0));
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        for i in 0..self.lhs_mlpoly.len() {
            append_gt_mul_virtual_claims(
                accumulator,
                transcript,
                i,  // Use local index, not global constraint index
                self.params.sumcheck_id,
                &opening_point,
                self.lhs_claims[i],
                self.rhs_claims[i],
                self.result_claims[i],
                self.quotient_claims[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for GT mul sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct GtMulVerifier<F: JoltField> {
    pub params: GtMulParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub num_constraints: usize,
    pub constraint_indices: Vec<usize>,
}

impl<F: JoltField> GtMulVerifier<F> {
    pub fn new<T: Transcript>(
        params: GtMulParams,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>();
        let num_constraints = params.num_constraints;

        Self {
            params,
            r_x,
            gamma: gamma.into(),
            num_constraints,
            constraint_indices,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for GtMulVerifier<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_f: Vec<F> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_f, &r_star_f);
        let g_eval: F = {
            use crate::poly::dense_mlpoly::DensePolynomial;
            use crate::poly::multilinear_polynomial::MultilinearPolynomial;
            use crate::zkvm::recursion::constraints_sys::DoryMatrixBuilder;
            use ark_bn254::Fq;
            use jolt_optimizations::get_g_mle;
            use std::any::TypeId;

            // Runtime check that F = Fq
            if TypeId::of::<F>() != TypeId::of::<Fq>() {
                panic!("g polynomial evaluation requires F = Fq for recursion SNARK");
            }

            // Get 4-var g polynomial and pad to match constraint vars
            let g_mle_4var = get_g_mle();
            let g_mle_padded = if r_star_f.len() == 11 {
                DoryMatrixBuilder::pad_4var_to_11var_zero_padding(&g_mle_4var)
            } else if r_star_f.len() == 8 {
                DoryMatrixBuilder::pad_4var_to_8var_zero_padding(&g_mle_4var)
            } else {
                g_mle_4var
            };
            // SAFETY: We checked F = Fq above, so this transmute is safe
            let g_poly_fq =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_padded));
            let r_star_fq: &Vec<Fq> = unsafe { std::mem::transmute(&r_star_f) };
            let g_eval_fq = g_poly_fq.evaluate_dot_product(r_star_fq);
            unsafe { std::mem::transmute_copy(&g_eval_fq) }
        };

        let mut total = F::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (lhs_claim, rhs_claim, result_claim, quotient_claim) = get_gt_mul_virtual_claims(
                accumulator,
                i,  // Use local index, not global constraint index
                self.params.sumcheck_id,
            );

            // Compute the GT mul constraint: lhs * rhs - result - quotient * g(x)
            let constraint_value = lhs_claim * rhs_claim - result_claim - quotient_claim * g_eval;

            total += gamma_power * constraint_value;
            gamma_power *= self.gamma;
        }

        eq_eval * total
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        for i in 0..self.num_constraints {
            append_gt_mul_virtual_openings(
                accumulator,
                transcript,
                i,  // Use local index, not global constraint index
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
