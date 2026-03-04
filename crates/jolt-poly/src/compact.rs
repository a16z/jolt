//! Compact multilinear polynomial storing small scalars with on-demand field promotion.

use std::borrow::Cow;
use std::marker::PhantomData;

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::dense::DensePolynomial;
use crate::eq::EqPolynomial;
use crate::traits::MultilinearPolynomial;

/// Trait for scalar types that can be stored compactly and promoted to a field element.
///
/// Implementations exist for common integer types (`bool`, `u8`..`u128`, `i64`, `i128`),
/// enabling polynomials that use as little as 1 bit per evaluation instead of 32 bytes.
pub trait SmallScalar:
    Copy + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>
{
    /// Converts this scalar to the corresponding field element.
    fn to_field<F: Field>(self) -> F;
}

impl SmallScalar for bool {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_bool(self)
    }
}

impl SmallScalar for u8 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u8(self)
    }
}

impl SmallScalar for u16 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u16(self)
    }
}

impl SmallScalar for u32 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u32(self)
    }
}

impl SmallScalar for u64 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self)
    }
}

impl SmallScalar for i64 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_i64(self)
    }
}

impl SmallScalar for i128 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self)
    }
}

impl SmallScalar for u128 {
    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u128(self)
    }
}

/// Compact multilinear polynomial: stores small scalars and converts to `F` on demand.
///
/// Reduces memory by up to 32x for Boolean polynomials (1 byte per evaluation
/// instead of 32 bytes for a BN254 field element). The trade-off is that
/// field promotion happens during `evaluate` and `bind`, adding conversion cost
/// to those operations.
///
/// # Type Parameters
/// - `S`: The compact scalar type (e.g., `u8`, `bool`)
/// - `F`: The target field type
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompactPolynomial<S: SmallScalar, F: Field> {
    scalars: Vec<S>,
    num_vars: usize,
    #[serde(skip)]
    _marker: PhantomData<F>,
}

impl<S: SmallScalar, F: Field> CompactPolynomial<S, F> {
    pub fn new(scalars: Vec<S>) -> Self {
        Self::from_scalars(scalars)
    }

    /// Creates a compact polynomial from a vector of small scalars.
    ///
    /// # Panics
    /// Panics if `scalars.len()` is not a power of two (or zero).
    pub fn from_scalars(scalars: Vec<S>) -> Self {
        let len = scalars.len();
        if len == 0 {
            return Self {
                scalars,
                num_vars: 0,
                _marker: PhantomData,
            };
        }
        assert!(
            len.is_power_of_two(),
            "scalar count must be a power of two, got {len}"
        );
        let num_vars = len.trailing_zeros() as usize;
        Self {
            scalars,
            num_vars,
            _marker: PhantomData,
        }
    }
}

impl<S: SmallScalar, F: Field> MultilinearPolynomial<F> for CompactPolynomial<S, F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn len(&self) -> usize {
        self.scalars.len()
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();
        self.scalars
            .iter()
            .zip(eq_evals.iter())
            .map(|(&s, &e)| {
                let f: F = s.to_field();
                f * e
            })
            .sum()
    }

    fn bind(&self, scalar: F) -> DensePolynomial<F> {
        let half = self.scalars.len() / 2;
        let mut result = Vec::with_capacity(half);

        for i in 0..half {
            let lo: F = self.scalars[i].to_field();
            let hi: F = self.scalars[i + half].to_field();
            result.push(lo + scalar * (hi - lo));
        }

        DensePolynomial::new(result)
    }

    fn evaluations(&self) -> Cow<'_, [F]> {
        Cow::Owned(self.scalars.iter().map(|&s| s.to_field()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn compact_u8_matches_dense() {
        let scalars: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let compact = CompactPolynomial::<u8, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_u8(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }

    #[test]
    fn compact_bool_matches_dense() {
        let scalars: Vec<bool> = vec![true, false, false, true];
        let compact = CompactPolynomial::<bool, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_bool(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(20);
        let point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }

    #[test]
    fn compact_evaluations_match() {
        let scalars: Vec<u32> = vec![10, 20, 30, 40];
        let compact = CompactPolynomial::<u32, Fr>::new(scalars.clone());
        let evals = compact.evaluations();
        for (i, &s) in scalars.iter().enumerate() {
            assert_eq!(evals[i], Fr::from_u32(s));
        }
    }

    #[test]
    fn compact_bind_matches_dense_bind() {
        let scalars: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let compact = CompactPolynomial::<u16, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_u16(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(30);
        let scalar = Fr::random(&mut rng);

        let compact_bound = compact.bind(scalar);
        let dense_bound = dense.bind(scalar);

        assert_eq!(compact_bound, dense_bound);
    }

    #[test]
    fn serde_round_trip_compact_u8() {
        let scalars: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let compact = CompactPolynomial::<u8, Fr>::new(scalars);
        let bytes = bincode::serialize(&compact).unwrap();
        let recovered: CompactPolynomial<u8, Fr> = bincode::deserialize(&bytes).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(40);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        assert_eq!(compact.evaluate(&point), recovered.evaluate(&point));
    }

    #[test]
    fn serde_round_trip_compact_bool() {
        let scalars: Vec<bool> = vec![true, false, true, false];
        let compact = CompactPolynomial::<bool, Fr>::new(scalars);
        let bytes = bincode::serialize(&compact).unwrap();
        let recovered: CompactPolynomial<bool, Fr> = bincode::deserialize(&bytes).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(41);
        let point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
        assert_eq!(compact.evaluate(&point), recovered.evaluate(&point));
    }

    #[test]
    fn compact_i64_negative_values() {
        let scalars: Vec<i64> = vec![-1, 0, 1, -100, i64::MIN, i64::MAX, -42, 42];
        let compact = CompactPolynomial::<i64, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_i64(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(50);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }

    #[test]
    fn compact_i64_bind_matches_dense() {
        let scalars: Vec<i64> = vec![i64::MIN, i64::MAX, -1, 1];
        let compact = CompactPolynomial::<i64, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_i64(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(51);
        let scalar = Fr::random(&mut rng);

        assert_eq!(compact.bind(scalar), dense.bind(scalar));
    }

    #[test]
    fn compact_i128_negative_values() {
        let scalars: Vec<i128> = vec![-1, 0, 1, -999, i128::MIN, i128::MAX, -7, 7];
        let compact = CompactPolynomial::<i128, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_i128(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(60);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }

    #[test]
    fn compact_i128_bind_matches_dense() {
        let scalars: Vec<i128> = vec![i128::MIN, i128::MAX, -1, 1];
        let compact = CompactPolynomial::<i128, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_i128(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(61);
        let scalar = Fr::random(&mut rng);

        assert_eq!(compact.bind(scalar), dense.bind(scalar));
    }

    #[test]
    fn compact_u128_large_values() {
        let scalars: Vec<u128> = vec![u128::MAX, u128::MAX - 1, 0, 1];
        let compact = CompactPolynomial::<u128, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_u128(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(70);
        let point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }

    #[test]
    fn compact_u128_bind_matches_dense() {
        let scalars: Vec<u128> = vec![u128::MAX, 0, 1 << 127, 1];
        let compact = CompactPolynomial::<u128, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_u128(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(71);
        let scalar = Fr::random(&mut rng);

        assert_eq!(compact.bind(scalar), dense.bind(scalar));
    }

    #[test]
    fn compact_bind_bind_consistency() {
        let scalars: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let compact = CompactPolynomial::<u32, Fr>::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from_u32(s)).collect();
        let dense = DensePolynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(80);
        let r1 = Fr::random(&mut rng);
        let r2 = Fr::random(&mut rng);
        let remaining: Vec<Fr> = (0..1).map(|_| Fr::random(&mut rng)).collect();

        // bind(r1).bind(r2) should produce same result as dense path
        let bound1 = compact.bind(r1);
        let bound2 = bound1.bind(r2);
        let result = bound2.evaluate(&remaining);

        let mut full_point = vec![r1, r2];
        full_point.extend_from_slice(&remaining);
        assert_eq!(result, dense.evaluate(&full_point));
    }

    #[test]
    fn compact_empty_polynomial() {
        let compact = CompactPolynomial::<u8, Fr>::new(vec![]);
        assert_eq!(compact.num_vars(), 0);
        assert!(compact.is_empty());
    }

    #[test]
    fn compact_single_element() {
        let compact = CompactPolynomial::<u64, Fr>::new(vec![42]);
        assert_eq!(compact.num_vars(), 0);
        assert_eq!(compact.evaluate(&[]), Fr::from_u64(42));
    }
}
