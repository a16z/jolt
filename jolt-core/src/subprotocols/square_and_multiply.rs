use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::RecursionCommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_bn254;
use jolt_optimizations::ExponentiationSteps;
use rayon::prelude::*;

/// Shared parameters for ExpSumcheck prover and verifier
#[derive(Clone)]
pub struct ExpSumcheckParams<F: JoltField> {
    /// Index of the exponentiation this instance belongs to
    exponentiation_index: usize,
    num_constraints: usize,
    gamma_powers: Vec<F>,
    r: Vec<F>,
    /// Number of variables (expecting 4 for exp polys)
    num_vars: usize,
    /// Exponent bits b_1, b_2, ..., b_t (public to both prover and verifier)
    bits: Vec<bool>,
}

#[derive(Allocative)]
struct ExpProverState<F: JoltField> {
    /// MLE of accumulator polynomials, ρ: ρ_0, ρ_1, ..., ρ_t
    rho_polys: Vec<MultilinearPolynomial<F>>,
    /// MLE of quotient polynomials q_1, q_2, ..., q_t
    quotient_polys: Vec<MultilinearPolynomial<F>>,
    /// MLE of base polynomial a(x)
    base_poly: MultilinearPolynomial<F>,
    /// MLE of g(x) = X^12 - 18X^6 + 82
    g_poly: MultilinearPolynomial<F>,
    /// eq(r, x)
    eq_poly: MultilinearPolynomial<F>,
}

/// Prover for ExpSumcheck
#[derive(Allocative)]
pub struct ExpSumcheckProver<F: JoltField> {
    params: ExpSumcheckParams<F>,
    prover_state: ExpProverState<F>,
}

/// Verifier for ExpSumcheck
pub struct ExpSumcheckVerifier<F: JoltField> {
    params: ExpSumcheckParams<F>,
}

impl<F: JoltField> ExpSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ExpSumcheckProver::new")]
    pub fn new(
        exponentiation_index: usize,
        steps: &ExponentiationSteps,
        r: Vec<F>,
        gamma: F,
    ) -> Self
    where
        F: From<ark_bn254::Fq>,
    {
        assert_eq!(r.len(), 4, "Expected 4 variables for Exp sumcheck");

        let rho_polys: Vec<MultilinearPolynomial<F>> = steps
            .rho_mles
            .iter()
            .map(|mle| MultilinearPolynomial::from_fq_vec(mle.clone()))
            .collect();

        let quotient_polys: Vec<MultilinearPolynomial<F>> = steps
            .quotient_mles
            .iter()
            .map(|mle| MultilinearPolynomial::from_fq_vec(mle.clone()))
            .collect();

        let base_poly = MultilinearPolynomial::from_fq_vec(
            jolt_optimizations::fq12_to_multilinear_evals(&steps.base),
        );
        let g_poly =
            MultilinearPolynomial::from_fq_vec(jolt_optimizations::witness_gen::get_g_mle());

        let eq_poly =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(EqPolynomial::evals(&r)));

        let num_constraints = steps.quotient_mles.len();
        let gamma_powers: Vec<F> = (0..num_constraints)
            .scan(F::one(), |acc, _| {
                let current = *acc;
                *acc *= gamma;
                Some(current * gamma)
            })
            .collect();

        let params = ExpSumcheckParams {
            exponentiation_index,
            num_constraints,
            gamma_powers,
            r: r.clone(),
            num_vars: 4,
            bits: steps.bits.clone(),
        };

        let prover_state = ExpProverState {
            rho_polys,
            quotient_polys,
            base_poly,
            g_poly,
            eq_poly,
        };

        Self {
            params,
            prover_state,
        }
    }
}

impl<F: JoltField> ExpSumcheckVerifier<F> {
    pub fn new(
        exponentiation_index: usize,
        num_constraints: usize,
        bits: Vec<bool>,
        r: Vec<F>,
        gamma: F,
    ) -> Self {
        assert_eq!(r.len(), 4, "Expected 4 variables for Exp sumcheck");

        let gamma_powers: Vec<F> = (0..num_constraints)
            .scan(F::one(), |acc, _| {
                let current = *acc;
                *acc *= gamma;
                Some(current * gamma)
            })
            .collect();

        let params = ExpSumcheckParams {
            exponentiation_index,
            num_constraints,
            gamma_powers,
            r,
            num_vars: 4,
            bits,
        };

        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ExpSumcheckProver<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let prover_state = &self.prover_state;
        const DEGREE: usize = 4;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = prover_state
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut constraint_evals = [F::zero(); DEGREE];

                for eval_idx in 0..DEGREE {
                    let mut batched_sum = F::zero();

                    for constraint_idx in 0..self.params.num_constraints {
                        let rho_prev_eval = prover_state.rho_polys[constraint_idx]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)[eval_idx];
                        let rho_curr_eval = prover_state.rho_polys[constraint_idx + 1]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)[eval_idx];
                        let q_eval = prover_state.quotient_polys[constraint_idx]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)[eval_idx];
                        let base_eval = prover_state
                            .base_poly
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)[eval_idx];
                        let g_eval = prover_state
                            .g_poly
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)[eval_idx];

                        let base_power = if self.params.bits[constraint_idx] {
                            base_eval
                        } else {
                            F::one()
                        };

                        let constraint_i = rho_curr_eval
                            - rho_prev_eval * rho_prev_eval * base_power
                            - q_eval * g_eval;

                        batched_sum += self.params.gamma_powers[constraint_idx] * constraint_i;
                    }

                    constraint_evals[eval_idx] = batched_sum;
                }
                [
                    eq_evals[0] * constraint_evals[0],
                    eq_evals[1] * constraint_evals[1],
                    eq_evals[2] * constraint_evals[2],
                    eq_evals[3] * constraint_evals[3],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..(DEGREE) {
                        running[i] += new[i];
                    }
                    running
                },
            );

        UniPoly::from_coeff(univariate_poly_evals.to_vec())
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = &mut self.prover_state;
        use rayon::prelude::*;

        rayon::scope(|s| {
                s.spawn(|_| {
                    prover_state
                        .eq_poly
                        .bind_parallel(r_j, BindingOrder::LowToHigh);
                });

                s.spawn(|_| {
                    prover_state
                        .base_poly
                        .bind_parallel(r_j, BindingOrder::LowToHigh);
                });
                s.spawn(|_| {
                    prover_state
                        .g_poly
                        .bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });

            prover_state
                .rho_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));

        prover_state
            .quotient_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = sumcheck_challenges.to_vec();
        let prover_state = &self.prover_state;

        let mut rho_polynomials = Vec::new();
        let mut rho_claims = Vec::new();
        for (i, rho_poly) in prover_state.rho_polys.iter().enumerate() {
            rho_polynomials.push(RecursionCommittedPolynomial::RecursionRho(
                self.params.exponentiation_index,
                i,
            ));
            rho_claims.push(rho_poly.final_sumcheck_claim());
        }

        accumulator.append_dense_recursion(
            rho_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.clone(),
            &rho_claims,
        );

        let mut quotient_polynomials = Vec::new();
        let mut quotient_claims = Vec::new();
        for (i, q_poly) in prover_state.quotient_polys.iter().enumerate() {
            quotient_polynomials.push(RecursionCommittedPolynomial::RecursionQuotient(
                self.params.exponentiation_index,
                i,
            ));
            quotient_claims.push(q_poly.final_sumcheck_claim());
        }

        accumulator.append_dense_recursion(
            quotient_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.clone(),
            &quotient_claims,
        );

        accumulator.append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::RecursionBase(self.params.exponentiation_index),
                RecursionCommittedPolynomial::RecursionG(self.params.exponentiation_index),
            ],
            SumcheckId::RecursionCheck,
            opening_point,
            &[
                prover_state.base_poly.final_sumcheck_claim(),
                prover_state.g_poly.final_sumcheck_claim(),
            ],
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ExpSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        r_sumcheck: &[F::Challenge],
    ) -> F {
        // this sumcheck is LowToHigh hence we need to reverse here.
        let r = self.params.r.iter().cloned().rev().collect::<Vec<_>>();

        let rho_evals: Vec<F> = (0..=self.params.num_constraints)
            .map(|i| {
                accumulator
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::RecursionRho(self.params.exponentiation_index, i),
                        SumcheckId::RecursionCheck,
                    )
                    .1
            })
            .collect();

        let base_eval = accumulator
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::RecursionBase(self.params.exponentiation_index),
                SumcheckId::RecursionCheck,
            )
            .1;

        let g_eval = accumulator
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::RecursionG(self.params.exponentiation_index),
                SumcheckId::RecursionCheck,
            )
            .1;

        let q_evals: Vec<F> = (0..self.params.num_constraints)
            .map(|i| {
                accumulator
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::RecursionQuotient(
                            self.params.exponentiation_index,
                            i,
                        ),
                        SumcheckId::RecursionCheck,
                    )
                    .1
            })
            .collect();

        let eq_eval = EqPolynomial::mle(&r, r_sumcheck);

        let batched_constraint = (0..self.params.num_constraints)
            .map(|i| {
                let base_power = if self.params.bits[i] { base_eval } else { F::one() };
                let constraint =
                    rho_evals[i + 1] - rho_evals[i].square() * base_power - q_evals[i] * g_eval;
                self.params.gamma_powers[i] * constraint
            })
            .sum::<F>();

        eq_eval * batched_constraint
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = sumcheck_challenges.to_vec();

        let mut rho_polynomials = Vec::new();
        for i in 0..=self.params.num_constraints {
            rho_polynomials.push(RecursionCommittedPolynomial::RecursionRho(
                self.params.exponentiation_index,
                i,
            ));
        }

        accumulator.append_dense_recursion(
            rho_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.clone(),
        );

        let mut quotient_polynomials = Vec::new();
        for i in 0..self.params.num_constraints {
            quotient_polynomials.push(RecursionCommittedPolynomial::RecursionQuotient(
                self.params.exponentiation_index,
                i,
            ));
        }

        accumulator.append_dense_recursion(
            quotient_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.clone(),
        );

        accumulator.append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::RecursionBase(self.params.exponentiation_index),
                RecursionCommittedPolynomial::RecursionG(self.params.exponentiation_index),
            ],
            SumcheckId::RecursionCheck,
            opening_point,
        );
    }
}
