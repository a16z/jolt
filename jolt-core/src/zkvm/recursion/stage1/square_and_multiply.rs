//! Square-and-multiply sumcheck for proving GT exponentiation constraints
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * C_i(x)
//! Where C_i(x) = ρ_{i+1}(x) - ρ_i(x)² × a(x)^{b_i} - Q_i(x) × g(x)
//!
//! This is Stage 1 of the new two-stage recursion protocol.
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
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use rayon::prelude::*;

/// Helper to append all virtual claims for a constraint
fn append_constraint_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    base_claim: F,
    rho_prev_claim: F,
    rho_curr_claim: F,
    quotient_claim: F,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        base_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        rho_prev_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        rho_curr_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        quotient_claim,
    );
}

/// Helper to retrieve all virtual claims for a constraint
fn get_constraint_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (F, F, F, F) {
    let (_, base_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
    );
    let (_, rho_prev_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
    );
    let (_, rho_curr_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
    );
    let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
    );

    (base_claim, rho_prev_claim, rho_curr_claim, quotient_claim)
}

/// Helper to append virtual opening points for a constraint (verifier side)
fn append_constraint_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
}

/// Individual polynomial data for a single constraint
#[derive(Clone)]
pub struct ConstraintPolynomials<F: JoltField> {
    pub base: Vec<F>,
    pub rho_prev: Vec<F>,
    pub rho_curr: Vec<F>,
    pub quotient: Vec<F>,
    pub bit: bool,
    pub constraint_index: usize,
}

/// Parameters for square-and-multiply sumcheck
#[derive(Clone)]
pub struct SquareAndMultiplyParams {
    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraints
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl SquareAndMultiplyParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 8, // Fixed for Fq12
            num_constraints,
            sumcheck_id: SumcheckId::SquareAndMultiply,
        }
    }
}

/// Prover for square-and-multiply sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SquareAndMultiplyProver<F: JoltField> {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: SquareAndMultiplyParams,

    /// Constraint bits for base^{b_i} evaluation
    pub constraint_bits: Vec<bool>,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

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

    /// Base polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub base_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Rho_prev polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_prev_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Rho_curr polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_curr_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Quotient polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub quotient_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Individual claims for each constraint (not batched)
    pub base_claims: Vec<F>,
    pub rho_prev_claims: Vec<F>,
    pub rho_curr_claims: Vec<F>,
    pub quotient_claims: Vec<F>,

    /// Current round
    pub round: usize,
}

impl<F: JoltField> SquareAndMultiplyProver<F> {
    pub fn new<T: Transcript>(
        params: SquareAndMultiplyParams,
        constraint_polys: Vec<ConstraintPolynomials<F>>,
        g_poly: DensePolynomial<F>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));
        let mut constraint_bits = Vec::new();
        let mut constraint_indices = Vec::new();
        let mut base_mlpoly = Vec::new();
        let mut rho_prev_mlpoly = Vec::new();
        let mut rho_curr_mlpoly = Vec::new();
        let mut quotient_mlpoly = Vec::new();

        for poly in constraint_polys {
            constraint_bits.push(poly.bit);
            constraint_indices.push(poly.constraint_index);
            base_mlpoly.push(MultilinearPolynomial::from(poly.base));
            rho_prev_mlpoly.push(MultilinearPolynomial::from(poly.rho_prev));
            rho_curr_mlpoly.push(MultilinearPolynomial::from(poly.rho_curr));
            quotient_mlpoly.push(MultilinearPolynomial::from(poly.quotient));
        }

        Self {
            params,
            constraint_bits,
            constraint_indices,
            g_poly: MultilinearPolynomial::LargeScalars(g_poly),
            eq_x,
            r_x,
            gamma: gamma.into(),
            base_mlpoly,
            rho_prev_mlpoly,
            rho_curr_mlpoly,
            quotient_mlpoly,
            base_claims: vec![],
            rho_prev_claims: vec![],
            rho_curr_claims: vec![],
            quotient_claims: vec![],
            round: 0,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SquareAndMultiplyProver<F> {
    fn degree(&self) -> usize {
        4 // Degree from constraint: rho_prev^2 * base
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "SquareAndMultiply::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 4;
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

                for i in 0..self.constraint_bits.len() {
                    let base_evals_hint = self.base_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let rho_prev_evals_hint = self.rho_prev_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let rho_curr_evals_hint = self.rho_curr_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let quotient_evals_hint = self.quotient_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        // base^{b_i}: if bit is true, use base; else use 1
                        let base_power = if self.constraint_bits[i] {
                            base_evals_hint[t]
                        } else {
                            F::one()
                        };
                        let constraint_val = rho_curr_evals_hint[t]
                            - rho_prev_evals_hint[t] * rho_prev_evals_hint[t] * base_power
                            - quotient_evals_hint[t] * g_evals[t];

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

    #[tracing::instrument(skip_all, name = "SquareAndMultiply::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.base_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rho_prev_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rho_curr_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.quotient_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        if self.round == self.params.num_constraint_vars {
            self.base_claims.clear();
            self.rho_prev_claims.clear();
            self.rho_curr_claims.clear();
            self.quotient_claims.clear();

            for i in 0..self.constraint_bits.len() {
                self.base_claims
                    .push(self.base_mlpoly[i].get_bound_coeff(0));
                self.rho_prev_claims
                    .push(self.rho_prev_mlpoly[i].get_bound_coeff(0));
                self.rho_curr_claims
                    .push(self.rho_curr_mlpoly[i].get_bound_coeff(0));
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

        for i in 0..self.constraint_bits.len() {
            append_constraint_virtual_claims(
                accumulator,
                transcript,
                self.constraint_indices[i], // Use global constraint index
                self.params.sumcheck_id,
                &opening_point,
                self.base_claims[i],
                self.rho_prev_claims[i],
                self.rho_curr_claims[i],
                self.quotient_claims[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for square-and-multiply sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SquareAndMultiplyVerifier<F: JoltField> {
    pub params: SquareAndMultiplyParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub num_constraints: usize,
    pub constraint_bits: Vec<bool>,
    pub constraint_indices: Vec<usize>,
}

impl<F: JoltField> SquareAndMultiplyVerifier<F> {
    pub fn new<T: Transcript>(
        params: SquareAndMultiplyParams,
        constraint_bits: Vec<bool>,
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
            constraint_bits,
            constraint_indices,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SquareAndMultiplyVerifier<F> {
    fn degree(&self) -> usize {
        4
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
            use jolt_optimizations::get_g_mle;

            // The g polynomial is specific to Fq12 arithmetic
            // For the recursion SNARK, F must be Fq
            // TODO: Abstract this for true genericity
            use ark_bn254::Fq;
            use std::any::TypeId;

            // Runtime check that F = Fq
            if TypeId::of::<F>() != TypeId::of::<Fq>() {
                panic!("g polynomial evaluation requires F = Fq for recursion SNARK");
            }

            // Get 4-var g polynomial and pad to 8 vars
            let g_mle_4var = get_g_mle();
            let g_mle_8var = if r_star_f.len() == 8 {
                DoryMatrixBuilder::pad_4var_to_8var_zero_padding(&g_mle_4var)
            } else {
                g_mle_4var
            };

            // Create polynomial and evaluate
            // SAFETY: We checked F = Fq above, so this transmute is safe
            let g_poly_fq =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_8var));
            let r_star_fq: &Vec<Fq> = unsafe { std::mem::transmute(&r_star_f) };
            let g_eval_fq = g_poly_fq.evaluate_dot_product(r_star_fq);
            unsafe { std::mem::transmute_copy(&g_eval_fq) }
        };

        let mut total = F::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (base_claim, rho_prev_claim, rho_curr_claim, quotient_claim) =
                get_constraint_virtual_claims(
                    accumulator,
                    self.constraint_indices[i],
                    self.params.sumcheck_id,
                );

            // Compute the constraint: ρ_{i+1} - ρ_i^2 * base^{b_i} - q_i * g(x)
            let base_power = if self.constraint_bits[i] {
                base_claim
            } else {
                F::one()
            };

            let constraint_value = rho_curr_claim
                - rho_prev_claim * rho_prev_claim * base_power
                - quotient_claim * g_eval;

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
            append_constraint_virtual_openings(
                accumulator,
                transcript,
                self.constraint_indices[i],
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
