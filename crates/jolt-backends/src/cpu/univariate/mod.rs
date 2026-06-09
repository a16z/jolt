use jolt_field::Field;
use jolt_poly::{CompressedPoly, UnivariatePoly};

#[inline(always)]
pub fn compress<F: Field>(poly: &UnivariatePoly<F>) -> CompressedPoly<F> {
    let coefficients = poly.coefficients();
    debug_assert!(
        coefficients.len() >= 2,
        "cannot compress a polynomial of degree < 1"
    );
    let mut coeffs_except_linear_term = Vec::with_capacity(coefficients.len() - 1);
    coeffs_except_linear_term.push(coefficients[0]);
    coeffs_except_linear_term.extend_from_slice(&coefficients[2..]);
    debug_assert_eq!(coeffs_except_linear_term.len() + 1, coefficients.len());
    CompressedPoly::new(coeffs_except_linear_term)
}

#[inline(always)]
pub fn decompress<F: Field>(poly: &CompressedPoly<F>, hint: F) -> UnivariatePoly<F> {
    poly.decompress(hint)
}

#[inline(always)]
pub fn eval_from_hint<F: Field>(poly: &CompressedPoly<F>, hint: F, point: F) -> F {
    poly.eval_from_hint(&hint, &point)
}

#[inline(always)]
pub fn from_evals<F: Field>(evals: &[F]) -> UnivariatePoly<F> {
    UnivariatePoly::from_evals(evals)
}

#[inline(always)]
pub fn from_evals_and_hint<F: Field>(hint: F, evals: &[F]) -> UnivariatePoly<F> {
    UnivariatePoly::from_evals_and_hint(hint, evals)
}

#[inline(always)]
pub fn from_evals_toom<F: Field>(evals: &[F]) -> UnivariatePoly<F> {
    UnivariatePoly::from_evals_toom(evals)
}
