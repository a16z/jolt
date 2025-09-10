use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    zkvm::witness::VirtualPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

const DEGREE: usize = 3;
const NUM_A_POLYS: usize = 256;

#[derive(Allocative)]
struct SquareAndMultiplyProverState<F: JoltField> {
    /// The sequence of a polynomials: a_0, a_1, ..., a_127
    a_polys: Vec<MultilinearPolynomial<F>>,
    /// The fixed polynomial g(x)
    g: MultilinearPolynomial<F>,
    /// eq(r, x) polynomial evaluations
    eq_poly: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct SquareAndMultiplySumcheck<F: JoltField> {
    /// Powers of gamma for batching: [1, γ, γ², ..., γ^126]
    gamma_powers: Vec<F>,
    /// The fixed point r for eq(r, x)
    r: Vec<F>,
    /// Number of variables (expecting 4 for x ∈ {0,1}⁴)
    num_vars: usize,
    prover_state: Option<SquareAndMultiplyProverState<F>>,
}

impl<F: JoltField> SquareAndMultiplySumcheck<F> {
    pub fn new_prover(
        a_polys: Vec<MultilinearPolynomial<F>>,
        g: MultilinearPolynomial<F>,
        r: Vec<F>,
        gamma: F,
    ) -> Self {
        assert_eq!(a_polys.len(), NUM_A_POLYS);
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");
        assert_eq!(g.len(), 16, "g(x) should have 2^4 = 16 coefficients");

        // Compute gamma powers
        let mut gamma_powers = vec![F::one(); NUM_A_POLYS - 1];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // Compute eq(r, x) evaluations for all x ∈ {0,1}⁴
        let eq_poly = EqPolynomial::evals(&r).into();

        let prover_state = SquareAndMultiplyProverState {
            a_polys,
            g,
            eq_poly,
        };

        Self {
            gamma_powers,
            r,
            num_vars: 4,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier(r: Vec<F>, gamma: F) -> Self {
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");

        // Compute gamma powers
        let mut gamma_powers = vec![F::one(); NUM_A_POLYS - 1];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            gamma_powers,
            r,
            num_vars: 4,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for SquareAndMultiplySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();

        let n = 1 << (self.num_vars - round - 1);

        let univariate_evals: [F; 3] = (0..n)
            .into_par_iter()
            .map(|k| {
                // Get evaluations of eq polynomial
                let eq_evals = prover_state
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                // Get evaluations of g polynomial
                let g_evals = prover_state
                    .g
                    .sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                // Compute the batched constraint evaluations
                let mut constraint_evals = [F::zero(); DEGREE];

                for j in 0..DEGREE {
                    let mut sum = F::zero();

                    // Sum over i from 1 to 127
                    for i in 1..NUM_A_POLYS {
                        let a_i_eval = prover_state.a_polys[i]
                            .sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh)[j];

                        let a_i_minus_1_eval = prover_state.a_polys[i - 1]
                            .sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh)[j];

                        let a_i_minus_2_eval = if i >= 2 {
                            prover_state.a_polys[i - 2]
                                .sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh)[j]
                        } else {
                            F::zero() // a_{-1} = 0
                        };

                        // Compute: a_i - (a_{i-1} * g + a_{i-2}^2)
                        let constraint = a_i_eval
                            - (a_i_minus_1_eval * g_evals[j] + a_i_minus_2_eval * a_i_minus_2_eval);

                        // Multiply by gamma^{i-1} (since we're summing from i=1)
                        sum += self.gamma_powers[i - 1] * constraint;
                    }

                    constraint_evals[j] = eq_evals[j] * sum;
                }

                constraint_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for i in 0..DEGREE {
                        acc[i] += evals[i];
                    }
                    acc
                },
            );

        univariate_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = self.prover_state.as_mut() {
            // Bind all polynomials with the challenge
            prover_state
                .eq_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            prover_state.g.bind_parallel(r_j, BindingOrder::LowToHigh);

            for a_poly in prover_state.a_polys.iter_mut() {
                a_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_sumcheck: &[F],
    ) -> F {
        let accumulator = accumulator.unwrap();

        // Compute eq(r, r_sumcheck)
        let eq_eval = EqPolynomial::mle(&self.r, r_sumcheck);

        // Get claimed evaluations of a_i polynomials at r_sumcheck
        let mut sum = F::zero();

        for i in 1..NUM_A_POLYS {
            let a_i_claim = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::SquareMultiplyA(i),
                    SumcheckId::SquareAndMultiply,
                )
                .1;

            let a_i_minus_1_claim = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::SquareMultiplyA(i - 1),
                    SumcheckId::SquareAndMultiply,
                )
                .1;

            let a_i_minus_2_claim = if i >= 2 {
                accumulator
                    .borrow()
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::SquareMultiplyA(i - 2),
                        SumcheckId::SquareAndMultiply,
                    )
                    .1
            } else {
                F::zero()
            };

            let g_claim = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::SquareMultiplyG,
                    SumcheckId::SquareAndMultiply,
                )
                .1;

            // Compute: a_i - (a_{i-1} * g + a_{i-2}^2)
            let constraint =
                a_i_claim - (a_i_minus_1_claim * g_claim + a_i_minus_2_claim * a_i_minus_2_claim);

            sum += self.gamma_powers[i - 1] * constraint;
        }

        eq_eval * sum
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self.prover_state.as_ref().unwrap();

        // Cache g(r_sumcheck)
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SquareMultiplyG,
            SumcheckId::SquareAndMultiply,
            opening_point.clone(),
            prover_state.g.final_sumcheck_claim(),
        );

        // Cache all a_i(r_sumcheck) evaluations
        for i in 0..NUM_A_POLYS {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::SquareMultiplyA(i),
                SumcheckId::SquareAndMultiply,
                opening_point.clone(),
                prover_state.a_polys[i].final_sumcheck_claim(),
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Cache g(r_sumcheck)
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SquareMultiplyG,
            SumcheckId::SquareAndMultiply,
            opening_point.clone(),
        );

        // Cache all a_i(r_sumcheck) evaluations
        for i in 0..NUM_A_POLYS {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::SquareMultiplyA(i),
                SumcheckId::SquareAndMultiply,
                opening_point.clone(),
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// Multiplication constraint sumcheck

#[derive(Allocative, Clone)]
struct AccumulatorMultiplyProverState<F: JoltField> {
    /// accumulator polynomials rho_0, rho_1, ..., rho_127
    rho_polys: Vec<MultilinearPolynomial<F>>,
    /// a(x) derived from a^e
    a: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    exp_result: F,
}

/// Σ eq(r, x) * Σᵢ γⁱ * ρᵢ(x) * a(x)^{eᵢ}
#[derive(Allocative)]
pub struct AccumulatorMultiplySumcheck<F: JoltField> {
    gamma_powers: Vec<F>,
    r: Vec<F>,
    exponent_bits: Vec<u8>,
    num_vars: usize,
    prover_state: Option<AccumulatorMultiplyProverState<F>>,
}

impl<F: JoltField> AccumulatorMultiplySumcheck<F> {
    pub fn new_prover(
        rho_polys: Vec<MultilinearPolynomial<F>>,
        a: MultilinearPolynomial<F>,
        r: Vec<F>,
        gamma: F,
        exponent_bits: Vec<u8>,
        exp_result: F,
    ) -> Self {
        assert_eq!(rho_polys.len(), NUM_A_POLYS);
        assert_eq!(exponent_bits.len(), NUM_A_POLYS);
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");
        assert_eq!(a.len(), 16, "a(x) should have 2^4 = 16 coefficients");

        // Verify all exponent bits are 0 or 1
        // TODO(markosg04) this is ok? doesn't need to be a sumcheck
        for &bit in &exponent_bits {
            assert!(bit == 0 || bit == 1, "Exponent bits must be 0 or 1");
        }

        // Compute gamma powers
        let mut gamma_powers = vec![F::one(); NUM_A_POLYS];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // Compute eq(r, x) evaluations for all x ∈ {0,1}⁴
        let eq_poly = EqPolynomial::evals(&r).into();

        let prover_state = AccumulatorMultiplyProverState {
            rho_polys,
            a,
            eq_poly,
            exp_result,
        };

        Self {
            gamma_powers,
            r,
            exponent_bits,
            num_vars: 4,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier(r: Vec<F>, gamma: F, exponent_bits: Vec<u8>) -> Self {
        assert_eq!(r.len(), 4, "Expected 4 variables for x ∈ {{0,1}}⁴");
        assert_eq!(exponent_bits.len(), NUM_A_POLYS);

        // Verify all exponent bits are 0 or 1
        for &bit in &exponent_bits {
            assert!(bit == 0 || bit == 1, "Exponent bits must be 0 or 1");
        }

        // Compute gamma powers
        let mut gamma_powers = vec![F::one(); NUM_A_POLYS];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            gamma_powers,
            r,
            exponent_bits,
            num_vars: 4,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for AccumulatorMultiplySumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        self.prover_state.as_ref().unwrap().exp_result
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();

        // Number of elements in current round
        let n = 1 << (self.num_vars - round - 1);

        const ACCUMULATOR_DEGREE: usize = 2;

        // Compute univariate evaluations at 0 and 2
        // (evaluation at 1 will be interpolated from previous_claim)
        let univariate_evals: [F; ACCUMULATOR_DEGREE] = (0..n)
            .into_par_iter()
            .map(|k| {
                // Get evaluations of eq polynomial
                let eq_evals = prover_state
                    .eq_poly
                    .sumcheck_evals_array::<ACCUMULATOR_DEGREE>(k, BindingOrder::LowToHigh);

                // Get evaluations of a polynomial
                let a_evals = prover_state
                    .a
                    .sumcheck_evals_array::<ACCUMULATOR_DEGREE>(k, BindingOrder::LowToHigh);

                // Compute the batched evaluations
                let mut result_evals = [F::zero(); ACCUMULATOR_DEGREE];

                for j in 0..ACCUMULATOR_DEGREE {
                    let mut sum = F::zero();

                    // Sum over i from 0 to 127
                    for i in 0..NUM_A_POLYS {
                        let rho_i_eval = prover_state.rho_polys[i]
                            .sumcheck_evals_array::<ACCUMULATOR_DEGREE>(k, BindingOrder::LowToHigh)
                            [j];

                        // Compute a(x)^{e_i}: if e_i = 0, this is 1; if e_i = 1, this is a(x)
                        let a_power = if self.exponent_bits[i] == 0 {
                            F::one()
                        } else {
                            a_evals[j]
                        };

                        // Add γⁱ * ρᵢ(x) * a(x)^{eᵢ}
                        sum += self.gamma_powers[i] * rho_i_eval * a_power;
                    }

                    result_evals[j] = eq_evals[j] * sum;
                }

                result_evals
            })
            .reduce(
                || [F::zero(); ACCUMULATOR_DEGREE],
                |mut acc, evals| {
                    for i in 0..ACCUMULATOR_DEGREE {
                        acc[i] += evals[i];
                    }
                    acc
                },
            );

        univariate_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = self.prover_state.as_mut() {
            // Bind all polynomials with the challenge
            prover_state
                .eq_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            prover_state.a.bind_parallel(r_j, BindingOrder::LowToHigh);

            for rho_poly in prover_state.rho_polys.iter_mut() {
                rho_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_sumcheck: &[F],
    ) -> F {
        let accumulator = accumulator.unwrap();

        // Compute eq(r, r_sumcheck)
        let eq_eval = EqPolynomial::mle(&self.r, r_sumcheck);

        // Get claimed evaluation of a at r_sumcheck
        let a_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SquareMultiplyBase,
                SumcheckId::SquareAndMultiply,
            )
            .1;

        // Compute the sum
        let mut sum = F::zero();

        for i in 0..NUM_A_POLYS {
            let rho_i_claim = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::SquareMultiplyRho(i),
                    SumcheckId::SquareAndMultiply,
                )
                .1;

            // Compute a^{e_i}: if e_i = 0, this is 1; if e_i = 1, this is a
            let a_power = if self.exponent_bits[i] == 0 {
                F::one()
            } else {
                a_claim
            };

            sum += self.gamma_powers[i] * rho_i_claim * a_power;
        }

        eq_eval * sum
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self.prover_state.as_ref().unwrap();

        // Cache a(r_sumcheck)
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SquareMultiplyBase,
            SumcheckId::SquareAndMultiply,
            opening_point.clone(),
            prover_state.a.final_sumcheck_claim(),
        );

        // Cache all rho_i(r_sumcheck) evaluations
        for i in 0..NUM_A_POLYS {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::SquareMultiplyRho(i),
                SumcheckId::SquareAndMultiply,
                opening_point.clone(),
                prover_state.rho_polys[i].final_sumcheck_claim(),
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Cache a(r_sumcheck)
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SquareMultiplyBase,
            SumcheckId::SquareAndMultiply,
            opening_point.clone(),
        );

        // Cache all rho_i(r_sumcheck) evaluations
        for i in 0..NUM_A_POLYS {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::SquareMultiplyRho(i),
                SumcheckId::SquareAndMultiply,
                opening_point.clone(),
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
