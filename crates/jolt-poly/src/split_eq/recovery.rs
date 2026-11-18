use crate::UnivariatePoly;
use jolt_field::Field;
use std::marker::PhantomData;

/// Recovery for one round, borrowing its split equality state so it cannot advance.
/// Private fields certify an invertible `l(1)`. The caller remains responsible for
/// the consistency of the supplied coefficients and claim.
///
/// Every method returns `s = l*q` with one more coefficient than `q` and keeps
/// trailing zeros, so a degree-3 message always has four coefficients, as
/// [`GruenRound::gruen_poly_deg_3`](super::GruenRound::gruen_poly_deg_3) does
/// for every round. The sumcheck prover trims the batched round polynomial
/// before it reaches the transcript.
#[derive(Clone, Copy)]
pub struct PreparedLinearRecovery<'a, F: Field> {
    at_zero: F,
    slope: F,
    at_one_inverse: F,
    round: PhantomData<&'a ()>,
}

impl<F: Field> PreparedLinearRecovery<'_, F> {
    pub(super) fn new(at_zero: F, at_one: F) -> Option<Self> {
        Some(Self {
            at_zero,
            slope: at_one - at_zero,
            at_one_inverse: at_one.inverse()?,
            round: PhantomData,
        })
    }

    fn linear_term(&self, constant: F, sum_except_linear: F, claim: F) -> F {
        (claim - self.at_zero * constant) * self.at_one_inverse - sum_except_linear
    }

    /// Recovers `q`'s linear coefficient from `s(0) + s(1) = claim`, then returns
    /// `s = l*q`. The input is `[q_0, q_2, ..., q_d]`; empty input gives `[0]`.
    pub fn gruen_poly_from_coeffs_except_linear(
        self,
        coefficients: &[F],
        claim: F,
    ) -> UnivariatePoly<F> {
        let Some((&constant, rest)) = coefficients.split_first() else {
            return UnivariatePoly::new(vec![F::zero()]);
        };
        let sum = coefficients
            .iter()
            .copied()
            .fold(F::zero(), |sum, c| sum + c);
        let mut full = Vec::with_capacity(coefficients.len() + 1);
        full.push(constant);
        full.push(self.linear_term(constant, sum, claim));
        full.extend_from_slice(rest);
        mul_linear(&full, self.at_zero, self.slope)
    }

    /// Recovers the linear coefficient of `q(X) = constant + linear*X + quadratic*X²`
    /// from `s(0) + s(1) = claim`, then returns `s = l*q`.
    pub fn gruen_poly_deg_3(self, constant: F, quadratic: F, claim: F) -> UnivariatePoly<F> {
        let linear = self.linear_term(constant, constant + quadratic, claim);
        mul_linear(&[constant, linear, quadratic], self.at_zero, self.slope)
    }
}

#[inline(always)]
pub(super) fn mul_linear<F: Field>(q_coeffs: &[F], at_zero: F, slope: F) -> UnivariatePoly<F> {
    let mut coefficients = vec![F::zero(); q_coeffs.len() + 1];
    for (index, q_coeff) in q_coeffs.iter().copied().enumerate() {
        coefficients[index] += q_coeff * at_zero;
        coefficients[index + 1] += q_coeff * slope;
    }
    UnivariatePoly::new(coefficients)
}
