//! Compressed univariate polynomial with the linear term omitted.
//!
//! Used in sumcheck proofs to save one field element per round polynomial.
//! The linear term is recoverable from the sumcheck claim `f(0) + f(1)`.

use jolt_field::Field;
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
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct CompressedPoly<F: Field> {
    coeffs_except_linear_term: Vec<F>,
}

impl<F: Field> UnivariatePolynomial<F> for CompressedPoly<F> {
    /// Degree of the polynomial.
    ///
    /// A degree-d polynomial has d+1 coefficients; the compressed form stores
    /// d of them (all except the linear term), so `stored_len == degree`.
    fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

impl<F: Field> CompressedPoly<F> {
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

    /// Evaluates the polynomial at `point` using the hint `h = f(0) + f(1)`.
    ///
    /// Recovers the linear term, then evaluates via ascending-power accumulation
    /// in O(d) multiplications.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn evaluate_with_hint(&self, hint: F, point: F) -> F {
        self.eval_from_hint(&hint, &point)
    }

    /// Like [`evaluate_with_hint`](Self::evaluate_with_hint) but takes hint and
    /// point by reference, avoiding a copy at call sites that already hold references.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn eval_from_hint(&self, hint: &F, point: &F) -> F {
        eval_round_wire(
            WireFormat::Compressed,
            &self.coeffs_except_linear_term,
            *hint,
            *point,
        )
    }

    /// Recovers the full polynomial given the hint `h = f(0) + f(1)`.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    pub fn decompress(&self, hint: F) -> UnivariatePoly<F> {
        let (constant, linear_term, higher) =
            recover_compressed_terms(&self.coeffs_except_linear_term, hint);

        let mut coeffs = Vec::with_capacity(self.coeffs_except_linear_term.len() + 1);
        coeffs.push(constant);
        coeffs.push(linear_term);
        coeffs.extend_from_slice(higher);
        UnivariatePoly::new(coeffs)
    }
}

/// The omitted coefficient and the claim relation used to recover it.
#[derive(Clone, Copy, Debug)]
pub enum WireFormat<F> {
    Compressed,
    EqFactored { tau: F },
}

/// Evaluates a borrowed round without allocating or copying coefficients.
/// Empty eq-factored wires represent the constant `claim`.
///
/// # Panics
/// A compressed wire must contain its constant coefficient.
#[inline]
pub fn eval_round_wire<F: Field>(format: WireFormat<F>, wire: &[F], claim: F, point: F) -> F {
    match format {
        WireFormat::Compressed => {
            let (constant, linear, higher) = recover_compressed_terms(wire, claim);
            let mut power = point;
            let mut sum = constant + point * linear;
            for &coefficient in higher {
                power *= point;
                sum += coefficient * power;
            }
            sum
        }
        WireFormat::EqFactored { tau } => {
            let constant = claim - tau * wire.iter().copied().sum::<F>();
            constant
                + wire
                    .iter()
                    .rev()
                    .copied()
                    .fold(F::zero(), |acc, coefficient| acc * point + coefficient)
                    * point
        }
    }
}

#[inline]
#[expect(
    clippy::expect_used,
    reason = "preserves the owned compressed polynomial empty-input contract"
)]
fn recover_compressed_terms<F: Field>(wire: &[F], claim: F) -> (F, F, &[F]) {
    let (&constant, higher) = wire
        .split_first()
        .expect("cannot evaluate an empty compressed polynomial");
    let mut linear = claim - constant - constant;
    for &coefficient in higher {
        linear -= coefficient;
    }
    (constant, linear, higher)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use num_traits::{One, Zero};

    fn poly_and_hint(coeffs: Vec<Fr>) -> (UnivariatePoly<Fr>, Fr) {
        let p = UnivariatePoly::new(coeffs);
        let hint = p.evaluate(Fr::zero()) + p.evaluate(Fr::one());
        (p, hint)
    }

    #[test]
    fn compress_linear_polynomial() {
        let (p, hint) = poly_and_hint(vec![Fr::from_u64(5), Fr::from_u64(7)]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 1);
        assert_eq!(compressed.coeffs_except_linear_term().len(), 1);

        let recovered = compressed.decompress(hint);
        assert_eq!(recovered, p);

        let x = Fr::from_u64(3);
        assert_eq!(compressed.evaluate_with_hint(hint, x), p.evaluate(x));
    }

    #[test]
    fn compress_cubic_polynomial() {
        let (p, hint) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 3);
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
