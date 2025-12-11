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
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;
use ark_ff::Zero;
use rayon::prelude::*;

/// Helper to append all virtual claims for a GT mul constraint
fn append_gt_mul_virtual_claims<T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
    lhs_claim: Fq,
    rhs_claim: Fq,
    result_claim: Fq,
    quotient_claim: Fq,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulLhs(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        lhs_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulRhs(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        rhs_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulResult(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        result_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        quotient_claim,
    );
}

/// Helper to retrieve all virtual claims for a GT mul constraint
fn get_gt_mul_virtual_claims(
    accumulator: &VerifierOpeningAccumulator<Fq>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (Fq, Fq, Fq, Fq) {
    let (_, lhs_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionMulLhs(constraint_idx),
        sumcheck_id,
    );
    let (_, rhs_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionMulRhs(constraint_idx),
        sumcheck_id,
    );
    let (_, result_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionMulResult(constraint_idx),
        sumcheck_id,
    );
    let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionMulQuotient(constraint_idx),
        sumcheck_id,
    );

    (lhs_claim, rhs_claim, result_claim, quotient_claim)
}

/// Helper to append virtual opening points for a GT mul constraint (verifier side)
fn append_gt_mul_virtual_openings<T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulLhs(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulRhs(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulResult(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionMulQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
}

/// Individual polynomial data for a single GT mul constraint
#[derive(Clone)]
pub struct GtMulConstraintPolynomials {
    pub lhs: Vec<Fq>,
    pub rhs: Vec<Fq>,
    pub result: Vec<Fq>,
    pub quotient: Vec<Fq>,
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
            num_constraint_vars: 8, // Fixed for Fq12
            num_constraints,
            sumcheck_id: SumcheckId::GtMul,
        }
    }
}

/// Prover for GT mul sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct GtMulProver<T: Transcript> {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: GtMulParams,

    /// g(x) polynomial for constraint evaluation
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub g_poly: MultilinearPolynomial<Fq>,

    /// Equality polynomial for constraint variables x
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x: MultilinearPolynomial<Fq>,

    /// Random challenge for eq(r_x, x)
    pub r_x: Vec<<Fq as JoltField>::Challenge>,

    /// Gamma coefficient for batching constraints
    pub gamma: Fq,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

    /// LHS polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub lhs_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// RHS polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rhs_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Result polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub result_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Quotient polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub quotient_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Current round in the protocol
    pub round: usize,

    /// Cached evaluation claims for each polynomial
    pub lhs_claims: Vec<Fq>,
    pub rhs_claims: Vec<Fq>,
    pub result_claims: Vec<Fq>,
    pub quotient_claims: Vec<Fq>,

    pub _marker: std::marker::PhantomData<T>,
}

impl<T: Transcript> GtMulProver<T> {
    pub fn new(
        params: GtMulParams,
        constraint_polys: Vec<GtMulConstraintPolynomials>,
        g_poly: DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x[..]));

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

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for GtMulProver<T> {
    fn degree(&self) -> usize {
        3 // Degree from constraint: lhs * rhs
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        // For GT mul constraints, the sumcheck proves 0 = Σ_x eq(r_x, x) * Σ_i γ^i * C_i(x)
        // So the initial claim is 0
        Fq::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
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

                let mut x_evals = [Fq::zero(); DEGREE];
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
                || [Fq::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
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
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        for i in 0..self.lhs_mlpoly.len() {
            append_gt_mul_virtual_claims(
                accumulator,
                transcript,
                self.constraint_indices[i],
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
pub struct GtMulVerifier {
    pub params: GtMulParams,
    pub r_x: Vec<<Fq as JoltField>::Challenge>,
    pub gamma: Fq,
    pub num_constraints: usize,
    pub constraint_indices: Vec<usize>,
}

impl GtMulVerifier {
    pub fn new<T: Transcript>(
        params: GtMulParams,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for GtMulVerifier {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_fq: Vec<Fq> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_fq: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_fq, &r_star_fq);
        let g_eval = {
            use crate::poly::dense_mlpoly::DensePolynomial;
            use crate::poly::multilinear_polynomial::MultilinearPolynomial;
            use jolt_optimizations::get_g_mle;
            use crate::subprotocols::recursion_constraints::DoryMatrixBuilder;

            // Get 4-var g polynomial and pad to 8 vars
            let g_mle_4var = get_g_mle();
            let g_mle_8var = if r_star_fq.len() == 8 {
                DoryMatrixBuilder::pad_4var_to_8var(&g_mle_4var)
            } else {
                g_mle_4var
            };
            let g_poly =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_8var));
            g_poly.evaluate_dot_product(&r_star_fq)
        };

        let mut total = Fq::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (lhs_claim, rhs_claim, result_claim, quotient_claim) = get_gt_mul_virtual_claims(
                accumulator,
                self.constraint_indices[i],
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
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        for i in 0..self.num_constraints {
            append_gt_mul_virtual_openings(
                accumulator,
                transcript,
                self.constraint_indices[i],
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
