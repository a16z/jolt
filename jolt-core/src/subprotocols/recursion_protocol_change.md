Hey, check out recursion_sumcheck.rs and recursion_constraints.rs.

This protocol essentially proves that dory verifier values (in this case, supporting GT exps) are well formed.

The protocol is a little difficult to reason about and extend for other computations, because there is bespoke indexing and it is a complicated sumcheck involving 4 final claims, which is very unique for sumchecks in this codebase. We want to refactor to make use of virtualization and make things more idiomatic.

Essentially what we want to do is refactor the protocol from what it does right now:

(1) Commit to M(s, x) (2) run the GT exp constraint check over all constraints (3) run a batching sumcheck to reduce 4 claims of M into 1 (this is done in reduce and prove) and (4) use hyrax to prove the openings (also handled in reduce and prove)

What we want to do is modify the sumchecks and the flow. That is, we still want to commit to some giant M (because we only want to have a single opening) but instead of the first sumcheck idexing into M directly, we will do something like the old protocl that I ripped up (it uses an old API but you can retrofit it to the new traits similar to recursion_sumcheck.rs):
```rust
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
    /// Exponent bits b_1, b_2, ..., b_t
    bits: Vec<bool>,
}

#[derive(Allocative)]
pub struct ExpSumcheck<F: JoltField> {
    /// Index of the exponentiation this instance belongs to
    exponentiation_index: usize,
    num_constraints: usize,
    gamma_powers: Vec<F>,
    r: Vec<F>,
    /// Number of variables (expecting 4 for exp polys)
    num_vars: usize,
    /// Exponent bits b_1, b_2, ..., b_t (public to both prover and verifier)
    bits: Vec<bool>,
    prover_state: Option<ExpProverState<F>>,
}

impl<F: JoltField> ExpSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ExpSumcheck::new_prover")]
    pub fn new_prover(
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

        let prover_state = ExpProverState {
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
        assert_eq!(r.len(), 4, "Expected 4 variables for Exp sumcheck");

        let gamma_powers: Vec<F> = (0..num_constraints)
            .scan(F::one(), |acc, _| {
                let current = *acc;
                *acc *= gamma;
                Some(current * gamma)
            })
            .collect();

        Self {
            exponentiation_index,
            num_constraints,
            gamma_powers,
            r,
            num_vars: 4,
            bits,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ExpSumcheck<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
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

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
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
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheck::expected_output_claim")]
    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_sumcheck: &[F],
    ) -> F {
        let accumulator = accumulator.expect("Accumulator required for expected output claim");

        // this sumcheck is LowToHigh hence we need to reverse here.
        let r = self.r.iter().cloned().rev().collect::<Vec<_>>();

        let rho_evals: Vec<F> = (0..=self.num_constraints)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::RecursionRho(self.exponentiation_index, i),
                        SumcheckId::RecursionCheck,
                    )
                    .1
            })
            .collect();

        let base_eval = accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::RecursionBase(self.exponentiation_index),
                SumcheckId::RecursionCheck,
            )
            .1;

        let g_eval = accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::RecursionG(self.exponentiation_index),
                SumcheckId::RecursionCheck,
            )
            .1;

        let q_evals: Vec<F> = (0..self.num_constraints)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_recursion_polynomial_opening(
                        RecursionCommittedPolynomial::RecursionQuotient(
                            self.exponentiation_index,
                            i,
                        ),
                        SumcheckId::RecursionCheck,
                    )
                    .1
            })
            .collect();

        let eq_eval = EqPolynomial::mle(&r, r_sumcheck);

        let batched_constraint = (0..self.num_constraints)
            .map(|i| {
                let base_power = if self.bits[i] { base_eval } else { F::one() };
                let constraint =
                    rho_evals[i + 1] - rho_evals[i].square() * base_power - q_evals[i] * g_eval;
                self.gamma_powers[i] * constraint
            })
            .sum::<F>();

        eq_eval * batched_constraint
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheck::cache_openings_prover")]
    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let mut rho_polynomials = Vec::new();
        let mut rho_claims = Vec::new();
        for (i, rho_poly) in prover_state.rho_polys.iter().enumerate() {
            rho_polynomials.push(RecursionCommittedPolynomial::RecursionRho(
                self.exponentiation_index,
                i,
            ));
            rho_claims.push(rho_poly.final_sumcheck_claim());
        }

        accumulator.borrow_mut().append_dense_recursion(
            rho_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.r.clone(),
            &rho_claims,
        );

        let mut quotient_polynomials = Vec::new();
        let mut quotient_claims = Vec::new();
        for (i, q_poly) in prover_state.quotient_polys.iter().enumerate() {
            quotient_polynomials.push(RecursionCommittedPolynomial::RecursionQuotient(
                self.exponentiation_index,
                i,
            ));
            quotient_claims.push(q_poly.final_sumcheck_claim());
        }

        accumulator.borrow_mut().append_dense_recursion(
            quotient_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.r.clone(),
            &quotient_claims,
        );

        accumulator.borrow_mut().append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::RecursionBase(self.exponentiation_index),
                RecursionCommittedPolynomial::RecursionG(self.exponentiation_index),
            ],
            SumcheckId::RecursionCheck,
            opening_point.r,
            &[
                prover_state.base_poly.final_sumcheck_claim(),
                prover_state.g_poly.final_sumcheck_claim(),
            ],
        );
    }

    #[tracing::instrument(skip_all, name = "ExpSumcheck::cache_openings_verifier")]
    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut rho_polynomials = Vec::new();
        for i in 0..=self.num_constraints {
            rho_polynomials.push(RecursionCommittedPolynomial::RecursionRho(
                self.exponentiation_index,
                i,
            ));
        }

        accumulator.borrow_mut().append_dense_recursion(
            rho_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.r.clone(),
        );

        let mut quotient_polynomials = Vec::new();
        for i in 0..self.num_constraints {
            quotient_polynomials.push(RecursionCommittedPolynomial::RecursionQuotient(
                self.exponentiation_index,
                i,
            ));
        }

        accumulator.borrow_mut().append_dense_recursion(
            quotient_polynomials,
            SumcheckId::RecursionCheck,
            opening_point.r.clone(),
        );

        accumulator.borrow_mut().append_dense_recursion(
            vec![
                RecursionCommittedPolynomial::RecursionBase(self.exponentiation_index),
                RecursionCommittedPolynomial::RecursionG(self.exponentiation_index),
            ],
            SumcheckId::RecursionCheck,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}````


Now, there are some differences we want. Namely, we do not want to commit to each rho, a(x), etc. We want to treat those as **virtual polynomials**. You should read other sumchecks and the opening_proof.rs to understand how we can virtualze things when appending claims about virtual polys to the accumulators.

The point is that, after this sumcheck, we will now virtualize the claims about each rho(r_x) a(r_x) etc via M. That is, you will write another file that performs a sum-check: ∑​eq(rs​,s)(M(s,x∗)−μ(s))=0. Mu here are the values arising from each indiviual ith poly in the first square_and_multiply sumcheck, and we can batch all together with gammas well. Essentially: this second sumcheck enforces that the rows of this M(s, x) correspond to each of the ith constraints. 

You will need to adjust the recursion_constraints.rs infra so that we can perform a square_and_multiply sumcheck directly (That old method seemed to work well with gammas) and then we will need to construct the big M poly. But, this virtualization sumcheck is nicer because: (a) now each sumcheck only is over their respecive variables (x or s), there are no offset bits in the second sumcheck (hence only one final claim) and now the reduce and prove will not need to run a batching sumcheck (since it is only a single claim!)

This is much cleaner and will make it easier to add other types of constraints with a generic virtualization of all the constraint polys into a single big M.

We can rename recursion_sumcheck.rs to square_and_multiply.rs, and then have a recursuon_virtualization.rs which has the virtualization sumcheck, and then we can in either file have the full e2e snark test (creates the witness from dory, the one that commits to M, runs the sumchecks, runs reduce and prove, and verifies everything)

These sumchecks are dependent. That is, we need to run the recursion one first using the batched api, then the second, and then reduce and prove. This is basically a full SNARK, you can check prover.rs in zkvm folder for example of how this flow may work. in general we want to reuse infra and utils wherever we can but here we are obviously using F and hyrax, which is fine because they implement the appropiate traits.


some more thing we need to fix:

1. we don't need backwards compatability, the new scheme is fine only to supporting
2. for square and multiply, we should essentially be left with claims of the type p(r_x) FOR EACH poly in EACH constraint (so we are appending many virtual polys)
3. Make sure you are batching the same way that the old square_and_multiply does (and same in the expected output claim). See in this file again if you need reminder of that file
4. the virtualization sumcheck i guess can fix the r_x for mu since it comes from that sumcheck
5. i want to eliminae the notion of offset bits here since it is not needed (we can just have generic s / index vars)
6. the virtual polys hence need to be indexed by usize (like base(usize)) because there are many of them!
