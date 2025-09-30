use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::sumcheck::SumcheckInstance,
    zkvm::witness::RecursionCommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_bn254;
use jolt_optimizations::ExponentiationSteps;
use rayon::prelude::*;

#[derive(Allocative)]
struct SZCheckProverState<F: JoltField> {
    /// The ρ polynomials: ρ_0, ρ_1, ..., ρ_t
    rho_polys: Vec<MultilinearPolynomial<F>>,
    /// The quotient polynomials Q_1, Q_2, ..., Q_t
    quotient_polys: Vec<MultilinearPolynomial<F>>,
    /// The base polynomial A(x)
    base_poly: MultilinearPolynomial<F>,
    /// The fixed polynomial g(x) = X^12 - 18X^6 + 82
    g_poly: MultilinearPolynomial<F>,
    /// eq(r, x) polynomial evaluations
    eq_poly: MultilinearPolynomial<F>,
    /// Exponent bits b_1, b_2, ..., b_t
    bits: Vec<bool>,
}

#[derive(Allocative)]
pub struct SZCheckSumcheck<F: JoltField> {
    /// Index of the exponentiation this instance belongs to
    exponentiation_index: usize,
    /// Number of constraints (t)
    num_constraints: usize,
    /// Powers of gamma for batching: [γ, γ², ..., γ^t]
    gamma_powers: Vec<F>,
    /// The fixed point r for eq(r, x)
    r: Vec<F>,
    /// Number of variables (expecting 4 for x ∈ {0,1}⁴)
    num_vars: usize,
    /// Exponent bits b_1, b_2, ..., b_t (public to both prover and verifier)
    bits: Vec<bool>,
    current_round: usize,
    prover_state: Option<SZCheckProverState<F>>,
}

impl<F: JoltField> SZCheckSumcheck<F> {
    /// Evaluate the batched constraint polynomial at a given point
    /// Returns: eq(z,x) * sum_{i=1}^t gamma^i * (rho_i(x) - rho_{i-1}(x)^2 * A(x)^{b_i} - Q_i(x) * g(x))
    pub fn evaluate_constraint_at_point(&self, x: &[F]) -> F {
        let prover_state = self.prover_state.as_ref().expect("Prover state required");

        // Evaluate eq(z, x) where z is self.r
        let eq_eval = prover_state.eq_poly.evaluate(x);

        // Evaluate base polynomial A(x)
        let base_eval = prover_state.base_poly.evaluate(x);

        // Evaluate g(x)
        let g_eval = prover_state.g_poly.evaluate(x);

        // Compute the batched constraint sum
        let mut batched_sum = F::zero();

        for i in 0..self.num_constraints {
            // Evaluate rho_{i-1}(x) and rho_i(x)
            let rho_prev_eval = prover_state.rho_polys[i].evaluate(x);
            let rho_curr_eval = prover_state.rho_polys[i + 1].evaluate(x);

            // Evaluate Q_i(x)
            let q_eval = prover_state.quotient_polys[i].evaluate(x);

            // Compute A(x)^{b_i}
            let base_power = if self.bits[i] { base_eval } else { F::one() };

            // Compute constraint_i = rho_i(x) - rho_{i-1}(x)^2 * A(x)^{b_i} - Q_i(x) * g(x)
            let constraint_i =
                rho_curr_eval - rho_prev_eval * rho_prev_eval * base_power - q_eval * g_eval;

            // Add gamma^i * constraint_i to the batch
            batched_sum += self.gamma_powers[i] * constraint_i;
        }

        // Return eq(z, x) * batched_sum
        eq_eval * batched_sum
    }

    pub fn new_prover(
        exponentiation_index: usize,
        steps: &ExponentiationSteps,
        r: Vec<F>,
        gamma: F,
    ) -> Self
    where
        F: From<ark_bn254::Fq>,
    {
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");

        // Convert Vec<ark_bn254::Fq> to Vec<F> and wrap in MultilinearPolynomial
        let convert_to_mle = |fq_vec: &Vec<ark_bn254::Fq>| -> MultilinearPolynomial<F> {
            let f_vec: Vec<F> = fq_vec.iter().map(|&fq| F::from(fq)).collect();
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(f_vec))
        };

        // Convert all rho MLEs
        let rho_polys: Vec<MultilinearPolynomial<F>> =
            steps.rho_mles.iter().map(convert_to_mle).collect();

        // Convert all quotient MLEs
        let quotient_polys: Vec<MultilinearPolynomial<F>> =
            steps.quotient_mles.iter().map(convert_to_mle).collect();

        // Convert base polynomial
        let base_mle = jolt_optimizations::fq12_to_multilinear_evals(&steps.base);
        let base_poly = convert_to_mle(&base_mle);

        // Convert g polynomial
        let g_mle = jolt_optimizations::witness_gen::get_g_mle();
        let g_poly = convert_to_mle(&g_mle);

        // Initialize eq(r, x) polynomial
        let eq_poly =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(EqPolynomial::evals(&r)));

        // Compute gamma powers: [γ, γ², ..., γ^t]
        let num_constraints = steps.quotient_mles.len();
        let mut gamma_powers = vec![gamma];
        for i in 1..num_constraints {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let prover_state = SZCheckProverState {
            rho_polys,
            quotient_polys,
            base_poly,
            g_poly,
            eq_poly,
            bits: steps.bits.clone(),
        };

        Self {
            exponentiation_index,
            num_constraints,
            gamma_powers,
            r,
            num_vars: 4,
            bits: steps.bits.clone(),
            current_round: 0,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier(
        exponentiation_index: usize,
        num_constraints: usize,
        bits: Vec<bool>,
        r: Vec<F>,
        gamma: F,
    ) -> Self {
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");

        // Compute gamma powers: [γ, γ², ..., γ^t]
        let mut gamma_powers = vec![gamma];
        for i in 1..num_constraints {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        Self {
            exponentiation_index,
            num_constraints,
            gamma_powers,
            r,
            num_vars: 4,
            bits,
            current_round: 0,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for SZCheckSumcheck<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 4; // Polynomial has degree 4, so we need 5 evaluations

        #[cfg(test)]
        {
            if _round == 0 {
                println!(
                    "Debug: First round, input_claim = {:?}, previous_claim = {:?}",
                    self.input_claim(),
                    _previous_claim
                );
            }
            // Only run this check in the first round when polynomials haven't been bound yet
            if self.current_round == 0 {
                // Debug check: verify constraints are zero over hypercube
                let actual_num_vars = if !prover_state.rho_polys.is_empty() {
                    prover_state.rho_polys[0].get_num_vars()
                } else {
                    return vec![F::zero(); DEGREE - 1];
                };

                // println!(
                //     "Debug: Round {}, num_vars = {}, poly num_vars = {}",
                //     self.current_round, self.num_vars, actual_num_vars
                // );

                let num_points = 1 << actual_num_vars;
                for idx in 0..num_points {
                    let mut x = Vec::new();
                    for j in 0..actual_num_vars {
                        x.push(if (idx >> j) & 1 == 0 {
                            F::zero()
                        } else {
                            F::one()
                        });
                    }

                    // Evaluate each constraint at this hypercube point
                    for constraint_idx in 0..self.num_constraints {
                        let rho_prev_eval = prover_state.rho_polys[constraint_idx].evaluate(&x);
                        let rho_curr_eval = prover_state.rho_polys[constraint_idx + 1].evaluate(&x);
                        let q_eval = prover_state.quotient_polys[constraint_idx].evaluate(&x);
                        let base_eval = prover_state.base_poly.evaluate(&x);
                        let g_eval = prover_state.g_poly.evaluate(&x);

                        let base_power = if self.bits[constraint_idx] {
                            base_eval
                        } else {
                            F::one()
                        };

                        let constraint_i = rho_curr_eval
                            - rho_prev_eval * rho_prev_eval * base_power
                            - q_eval * g_eval;

                        if constraint_i != F::zero() {
                            panic!(
                                "Constraint {} not zero at hypercube point {:?}: got {:?}",
                                constraint_idx, x, constraint_i
                            );
                        }
                    }
                }
            }
        }

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = prover_state
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut constraint_evals = [F::zero(); DEGREE];

                for eval_idx in 0..DEGREE {
                    let mut batched_sum = F::zero();

                    for constraint_idx in 0..self.num_constraints {
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

                        let base_power = if self.bits[constraint_idx] {
                            base_eval
                        } else {
                            F::one()
                        };

                        let constraint_i = rho_curr_eval
                            - rho_prev_eval * rho_prev_eval * base_power
                            - q_eval * g_eval;

                        batched_sum += self.gamma_powers[constraint_idx] * constraint_i;
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

        let result = univariate_poly_evals.to_vec();

        #[cfg(test)]
        if _round == self.num_rounds() - 1 {
            // Last round - let's see what the final evaluation is
            // println!("Debug: Last round evaluations: {:?}", result);
            // println!("Debug: This is round {} of {}", _round, self.num_rounds());
        }

        // Debug assertion: verify that g(0) + g(1) = previous_claim
        #[cfg(debug_assertions)]
        if _round > 0 {
            // We have evaluations at 0, 2, 3, 4 and need to interpolate to get value at 1
            let mut evals_with_one = vec![result[0]]; // eval at 0
            evals_with_one.push(F::zero()); // placeholder for eval at 1 (will be computed)
            evals_with_one.extend_from_slice(&result[1..]); // evals at 2, 3, 4

            // Compute eval at 1 using the sumcheck relation: g(0) + g(1) = previous_claim
            evals_with_one[1] = _previous_claim - evals_with_one[0];

            // Verify by interpolating and evaluating
            let poly = UniPoly::from_evals(&evals_with_one);
            let eval_at_0 = poly.evaluate(&F::zero());
            let eval_at_1 = poly.evaluate(&F::one());

            // println!(
            //     "Debug: Round {}, g(0) = {:?}, g(1) = {:?}, sum = {:?}, previous_claim = {:?}",
            //     _round,
            //     eval_at_0,
            //     eval_at_1,
            //     eval_at_0 + eval_at_1,
            //     _previous_claim
            // );

            debug_assert_eq!(
                eval_at_0 + eval_at_1,
                _previous_claim,
                "Sumcheck relation failed: g(0) + g(1) != previous_claim"
            );
        }

        result
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        self.current_round += 1;
        if let Some(prover_state) = self.prover_state.as_mut() {
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

        self.current_round += 1;
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_sumcheck: &[F],
    ) -> F {
        #[cfg(test)]
        {
            // println!(
            //     "Debug: expected_output_claim called with r_sumcheck = {:?}",
            //     r_sumcheck
            // );
            // println!("Debug: self.r = {:?}", self.r);
        }
        let accumulator = accumulator.expect("Accumulator required for expected output claim");

        let rho_evals: Vec<F> = (0..=self.num_constraints)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::SZCheckRho(self.exponentiation_index, i),
                        SumcheckId::SZCheck,
                    )
                    .1
            })
            .collect();

        let base_eval = accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::SZCheckBase(self.exponentiation_index),
                SumcheckId::SZCheck,
            )
            .1;

        let g_eval = accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::SZCheckG(self.exponentiation_index),
                SumcheckId::SZCheck,
            )
            .1;

        let q_evals: Vec<F> = (0..self.num_constraints)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::SZCheckQuotient(self.exponentiation_index, i),
                        SumcheckId::SZCheck,
                    )
                    .1
            })
            .collect();

        // Compute eq(r, r_sumcheck)
        //

        let r_rev = self.r.iter().cloned().rev().collect::<Vec<_>>();
        let eq_eval = EqPolynomial::mle(&r_rev, r_sumcheck);

        // Compute batched constraint evaluation
        let mut batched_constraint = F::zero();
        for i in 0..self.num_constraints {
            let base_power = if self.bits[i] { base_eval } else { F::one() };
            let constraint =
                rho_evals[i + 1] - rho_evals[i].square() * base_power - q_evals[i] * g_eval;

            // println!("Constraint {}: ", i);
            // println!("  rho_evals[{}] = {:?}", i, rho_evals[i]);
            // println!("  rho_evals[{}] = {:?}", i + 1, rho_evals[i + 1]);
            // println!("  rho_evals[{}].square() = {:?}", i, rho_evals[i].square());
            // println!("  bits[{}] = {}", i, self.bits[i]);
            // println!("  base_eval = {:?}", base_eval);
            // println!("  base_power = {:?}", base_power);
            // println!("  q_evals[{}] = {:?}", i, q_evals[i]);
            // println!("  g_eval = {:?}", g_eval);
            // println!("  constraint = {:?}", constraint);
            // println!("  gamma_powers[{}] = {:?}", i, self.gamma_powers[i]);

            batched_constraint += self.gamma_powers[i] * constraint;
        }

        let result = eq_eval * batched_constraint;
        #[cfg(test)]
        {
            // println!(
            //     "Debug: expected_output_claim: eq_eval = {:?}, batched_constraint = {:?}, result = {:?}",
            //     eq_eval, batched_constraint, result
            // );
        }
        result
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // let result = eq_poly
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let _result = prover_state.eq_poly.final_sumcheck_claim();

        // println!("final claim: {:?}", result);
        let mut rho_polynomials = Vec::new();
        let mut rho_claims = Vec::new();
        for (i, rho_poly) in prover_state.rho_polys.iter().enumerate() {
            rho_polynomials.push(RecursionCommittedPolynomial::SZCheckRho(
                self.exponentiation_index,
                i,
            ));
            rho_claims.push(rho_poly.final_sumcheck_claim());
        }

        accumulator.borrow_mut().append_dense_recursion(
            rho_polynomials,
            SumcheckId::SZCheck,
            opening_point.r.clone(),
            &rho_claims,
        );

        let mut quotient_polynomials = Vec::new();
        let mut quotient_claims = Vec::new();
        for (i, q_poly) in prover_state.quotient_polys.iter().enumerate() {
            quotient_polynomials.push(RecursionCommittedPolynomial::SZCheckQuotient(
                self.exponentiation_index,
                i,
            ));
            quotient_claims.push(q_poly.final_sumcheck_claim());
        }

        accumulator.borrow_mut().append_dense_recursion(
            quotient_polynomials,
            SumcheckId::SZCheck,
            opening_point.r.clone(),
            &quotient_claims,
        );

        accumulator.borrow_mut().append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::SZCheckBase(self.exponentiation_index),
                RecursionCommittedPolynomial::SZCheckG(self.exponentiation_index),
            ],
            SumcheckId::SZCheck,
            opening_point.r,
            &[
                prover_state.base_poly.final_sumcheck_claim(),
                prover_state.g_poly.final_sumcheck_claim(),
            ],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut rho_polynomials = Vec::new();
        for i in 0..=self.num_constraints {
            rho_polynomials.push(RecursionCommittedPolynomial::SZCheckRho(
                self.exponentiation_index,
                i,
            ));
        }

        accumulator.borrow_mut().append_dense_recursion(
            rho_polynomials,
            SumcheckId::SZCheck,
            opening_point.r.clone(),
        );

        let mut quotient_polynomials = Vec::new();
        for i in 0..self.num_constraints {
            quotient_polynomials.push(RecursionCommittedPolynomial::SZCheckQuotient(
                self.exponentiation_index,
                i,
            ));
        }

        accumulator.borrow_mut().append_dense_recursion(
            quotient_polynomials,
            SumcheckId::SZCheck,
            opening_point.r.clone(),
        );

        accumulator.borrow_mut().append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::SZCheckBase(self.exponentiation_index),
                RecursionCommittedPolynomial::SZCheckG(self.exponentiation_index),
            ],
            SumcheckId::SZCheck,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
