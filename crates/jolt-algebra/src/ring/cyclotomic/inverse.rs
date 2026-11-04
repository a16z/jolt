//! Inversion in `Z_q[X]/(X^D + 1)` through the tower of even-power subrings.

use super::*;

impl<F: Field, const D: usize> CyclotomicRing<F, D> {
    /// Multiplicative inverse, or `None` when `self` is not a unit or `D` is
    /// not a power of two.
    ///
    /// The conjugate `self(-X)` makes `self * self(-X)` invariant under
    /// `X -> -X`, so the product has only even powers and lives in
    /// `Z_q[Y]/(Y^{D/2} + 1)` with `Y = X^2`. Recursing down to degree one
    /// leaves a single field inversion, and
    /// `self^{-1} = self(-X) * (self * self(-X))^{-1}`.
    pub fn inverse(&self) -> Option<Self> {
        if !D.is_power_of_two() {
            return None;
        }
        negacyclic_inverse(&self.coeffs).map(|inverse| Self::from_slice(&inverse))
    }
}

/// Inverse in `Z_q[X]/(X^n + 1)` for a power-of-two `n = value.len()`.
fn negacyclic_inverse<F: Field>(value: &[F]) -> Option<Vec<F>> {
    if let [constant] = value {
        return constant.inverse().map(|inverse| vec![inverse]);
    }
    let conjugate: Vec<F> = value
        .iter()
        .enumerate()
        .map(|(i, coeff)| if i % 2 == 0 { *coeff } else { -*coeff })
        .collect();
    let norm: Vec<F> = negacyclic_mul(value, &conjugate)
        .into_iter()
        .step_by(2)
        .collect();
    let norm_inverse = negacyclic_inverse(&norm)?;
    let mut lifted = vec![F::zero(); value.len()];
    for (dst, src) in lifted.iter_mut().step_by(2).zip(norm_inverse) {
        *dst = src;
    }
    Some(negacyclic_mul(&lifted, &conjugate))
}

/// Product in `Z_q[X]/(X^n + 1)` for operands of equal length `n`.
fn negacyclic_mul<F: Field>(lhs: &[F], rhs: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); lhs.len()];
    negacyclic_mul_accumulate(lhs, rhs, &mut out);
    out
}
