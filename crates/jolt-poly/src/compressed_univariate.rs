//! Compressed univariate polynomial with the linear term omitted.
//!
//! Used in sumcheck proofs to save one field element per round polynomial.
//! The linear term is recoverable from the sumcheck claim `f(0) + f(1)`.

use jolt_field::JoltField;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::univariate::{UnivariatePoly, UnivariatePolynomial};

/// Compressed univariate polynomial: stores `[c0, c2, c3, ...]` with the
/// linear coefficient `c1` omitted.
///
/// Given the hint `h = f(0) + f(1)`, the linear term is recovered as:
/// `c1 = h - 2*c0 - c2 - c3 - ...`
///
/// This saves one field element per sumcheck round polynomial in proof
/// serialization (32 bytes for BN254).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct CompressedPoly<F: JoltField> {
    coeffs_except_linear_term: Vec<F>,
}

impl<F: JoltField> UnivariatePolynomial<F> for CompressedPoly<F> {
    /// Degree of the polynomial.
    ///
    /// A degree-d polynomial has d+1 coefficients; the compressed form stores
    /// d of them (all except the linear term), so `stored_len == degree`.
    fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

impl<F: JoltField> CompressedPoly<F> {
    /// Creates a compressed polynomial from the stored coefficients `[c0, c2, c3, ...]`.
    pub fn new(coeffs_except_linear_term: Vec<F>) -> Self {
        Self {
            coeffs_except_linear_term,
        }
    }

    /// The stored coefficients `[c0, c2, c3, ...]` (linear term omitted).
    pub fn coeffs_except_linear_term(&self) -> &[F] {
        &self.coeffs_except_linear_term
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs_except_linear_term.is_empty()
    }

    /// Recovers the omitted linear term from the hint `h = f(0) + f(1)`.
    ///
    /// `c1 = h - 2*c0 - c2 - c3 - ...`
    #[inline]
    fn recover_linear_term(&self, hint: F) -> F {
        // Deserialized proofs can carry an empty coefficient vector; fail with
        // a clear contract violation instead of an index panic. Callers on
        // untrusted data must reject empty polynomials first (`is_empty`).
        assert!(
            !self.coeffs_except_linear_term.is_empty(),
            "cannot evaluate an empty compressed polynomial"
        );
        let c0 = self.coeffs_except_linear_term[0];
        let mut linear_term = hint - c0 - c0;
        for &c in &self.coeffs_except_linear_term[1..] {
            linear_term -= c;
        }
        linear_term
    }

    /// Evaluates the polynomial at `point` using the hint `h = f(0) + f(1)`.
    ///
    /// Recovers the linear term, then evaluates via ascending-power accumulation
    /// in O(d) multiplications.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn evaluate_with_hint(&self, hint: F, point: F) -> F {
        self.eval_from_hint_observed(&hint, &point, || {})
    }

    /// Like [`evaluate_with_hint`](Self::evaluate_with_hint) but takes hint and
    /// point by reference, avoiding a copy at call sites that already hold references.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn eval_from_hint(&self, hint: &F, point: &F) -> F {
        self.eval_from_hint_observed(hint, point, || {})
    }

    /// Evaluates while calling `observe_mul` for each field multiplication.
    #[inline]
    pub fn eval_from_hint_observed(&self, hint: &F, point: &F, mut observe_mul: impl FnMut()) -> F {
        let linear_term = self.recover_linear_term(*hint);

        let mut x_pow = *point;
        observe_mul();
        let mut sum = self.coeffs_except_linear_term[0] + *point * linear_term;
        for &c in &self.coeffs_except_linear_term[1..] {
            observe_mul();
            x_pow *= *point;
            observe_mul();
            sum += c * x_pow;
        }
        sum
    }

    /// Recovers the full polynomial given the hint `h = f(0) + f(1)`.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    pub fn decompress(&self, hint: F) -> UnivariatePoly<F> {
        let linear_term = self.recover_linear_term(hint);

        let mut coeffs = Vec::with_capacity(self.coeffs_except_linear_term.len() + 1);
        coeffs.push(self.coeffs_except_linear_term[0]);
        coeffs.push(linear_term);
        coeffs.extend_from_slice(&self.coeffs_except_linear_term[1..]);
        UnivariatePoly::new(coeffs)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use num_traits::{One, Zero};

    /// Helper: build a standard polynomial p(x) = c0 + c1*x + c2*x^2 + ...
    /// and compute the sumcheck hint h = p(0) + p(1).
    fn poly_and_hint(coeffs: Vec<Fr>) -> (UnivariatePoly<Fr>, Fr) {
        let p = UnivariatePoly::new(coeffs);
        let hint = p.evaluate(Fr::zero()) + p.evaluate(Fr::one());
        (p, hint)
    }

    #[test]
    fn compress_decompress_round_trip() {
        // p(x) = 1 + 3x + 2x^2
        let (p, hint) = poly_and_hint(vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(2)]);
        let compressed = p.compress();
        let recovered = compressed.decompress(hint);
        assert_eq!(recovered, p);
    }

    #[test]
    fn evaluate_with_hint_matches_standard() {
        // p(x) = 1 + 3x + 2x^2
        let (p, hint) = poly_and_hint(vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(2)]);
        let compressed = p.compress();

        for i in 0..10 {
            let x = Fr::from_u64(i);
            assert_eq!(
                compressed.evaluate_with_hint(hint, x),
                p.evaluate(x),
                "mismatch at x={i}"
            );
        }
    }

    #[test]
    fn compress_linear_polynomial() {
        // p(x) = 5 + 7x  (degree 1)
        let (p, hint) = poly_and_hint(vec![Fr::from_u64(5), Fr::from_u64(7)]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 1);
        // Stored coefficients: [c0] = [5]
        assert_eq!(compressed.coeffs_except_linear_term().len(), 1);

        let recovered = compressed.decompress(hint);
        assert_eq!(recovered, p);

        let x = Fr::from_u64(3);
        assert_eq!(compressed.evaluate_with_hint(hint, x), p.evaluate(x));
    }

    #[test]
    fn compress_cubic_polynomial() {
        // p(x) = 1 + 3x + 2x^2 + x^3  (typical sumcheck degree)
        let (p, hint) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 3);
        // Stored: [c0, c2, c3] = [1, 2, 1]
        assert_eq!(compressed.coeffs_except_linear_term().len(), 3);

        let recovered = compressed.decompress(hint);
        assert_eq!(recovered, p);

        for i in 0..10 {
            let x = Fr::from_u64(i);
            assert_eq!(
                compressed.evaluate_with_hint(hint, x),
                p.evaluate(x),
                "mismatch at x={i}"
            );
        }
    }

    #[test]
    fn serde_round_trip() {
        let (p, _) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();
        let bytes =
            bincode::serde::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
        let recovered: CompressedPoly<Fr> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        assert_eq!(compressed, recovered);
    }

    #[test]
    #[should_panic(expected = "cannot evaluate an empty compressed polynomial")]
    fn empty_compressed_poly_evaluation_rejected() {
        let empty = CompressedPoly::<Fr>::new(vec![]);
        let _ = empty.evaluate_with_hint(Fr::one(), Fr::one());
    }

    #[test]
    #[should_panic(expected = "cannot evaluate an empty compressed polynomial")]
    fn empty_compressed_poly_decompress_rejected() {
        let empty = CompressedPoly::<Fr>::new(vec![]);
        let _ = empty.decompress(Fr::one());
    }

    #[test]
    fn degree_matches_standard() {
        for deg in 1..=5 {
            let coeffs: Vec<Fr> = (0..=deg).map(|i| Fr::from_u64(i as u64 + 1)).collect();
            let p = UnivariatePoly::new(coeffs);
            let compressed = p.compress();
            assert_eq!(
                compressed.degree(),
                p.degree(),
                "degree mismatch for deg={deg}"
            );
        }
    }
}
