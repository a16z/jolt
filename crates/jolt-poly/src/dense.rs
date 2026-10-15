//! Polynomial stored as evaluations over the Boolean hypercube.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use jolt_field::JoltField;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::eq::EqPolynomial;
use crate::BindingOrder;

#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 1024;

/// Multilinear polynomial stored as evaluations over the Boolean hypercube $\{0,1\}^n$.
///
/// Generic over the scalar type `T`:
/// - When `T` is a [`JoltField`] type: full polynomial with in-place [`bind`](Polynomial::bind),
///   [`evaluate`](Polynomial::evaluate), and arithmetic operators.
/// - When `T` is a small type (`u8`, `bool`, `i64`, etc.): compact storage with
///   [`bind_to_field`](Polynomial::bind_to_field) for on-demand field promotion.
#[cfg_attr(
    feature = "parallel",
    expect(
        clippy::unsafe_derive_deserialize,
        reason = "deserialization goes through PolynomialRaw and validates the polynomial dimensions"
    )
)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    bound(serialize = "T: Serialize", deserialize = "T: for<'a> Deserialize<'a>"),
    try_from = "PolynomialRaw<T>"
)]
pub struct Polynomial<T> {
    evals: Vec<T>,
    num_vars: usize,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "T: for<'a> Deserialize<'a>"))]
struct PolynomialRaw<T> {
    evals: Vec<T>,
    num_vars: usize,
}

impl<T> TryFrom<PolynomialRaw<T>> for Polynomial<T> {
    type Error = String;

    fn try_from(raw: PolynomialRaw<T>) -> Result<Self, Self::Error> {
        let len = raw.evals.len();
        let expected = if len == 0 {
            0
        } else if len.is_power_of_two() {
            len.trailing_zeros() as usize
        } else {
            return Err(format!(
                "evaluation count must be a power of two, got {len}"
            ));
        };
        if raw.num_vars != expected {
            return Err(format!(
                "num_vars mismatch: expected {expected}, got {}",
                raw.num_vars
            ));
        }
        Ok(Self {
            evals: raw.evals,
            num_vars: raw.num_vars,
        })
    }
}

impl<T> Polynomial<T> {
    /// Creates a polynomial from its evaluations over the Boolean hypercube.
    ///
    /// # Panics
    /// Panics if `evals.len()` is not a power of two (or zero).
    pub fn new(evals: Vec<T>) -> Self {
        let len = evals.len();
        if len == 0 {
            return Self { evals, num_vars: 0 };
        }
        assert!(
            len.is_power_of_two(),
            "evaluation count must be a power of two, got {len}"
        );
        let num_vars = len.trailing_zeros() as usize;
        Self { evals, num_vars }
    }

    /// Number of variables `n`. The polynomial has `2^n` evaluations.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    /// The raw evaluation table over the Boolean hypercube.
    #[inline]
    pub fn evals(&self) -> &[T] {
        &self.evals
    }

    pub fn into_evals(self) -> Vec<T> {
        self.evals
    }
}

impl<T: Copy> Polynomial<T> {
    /// Fixes the first variable to `scalar`, promoting all evaluations to field elements.
    ///
    /// Produces a `Polynomial<F>` with `n - 1` variables:
    /// $$g(x_2, \ldots, x_n) = (1 - s) \cdot f(0, x_2, \ldots) + s \cdot f(1, x_2, \ldots)$$
    ///
    /// When `T = F`, the `From` conversion is the identity and the compiler
    /// eliminates it, making this equivalent to an allocating bind.
    pub fn bind_to_field<F: JoltField + From<T>>(&self, scalar: F) -> Polynomial<F> {
        assert!(self.num_vars > 0, "cannot bind a zero-variable polynomial");
        let half = self.evals.len() / 2;
        let mut result = Vec::with_capacity(half);
        for i in 0..half {
            let lo: F = self.evals[i].into();
            let hi: F = self.evals[i + half].into();
            result.push(lo + scalar * (hi - lo));
        }
        Polynomial {
            evals: result,
            num_vars: self.num_vars - 1,
        }
    }
}

impl<F: JoltField> Polynomial<F> {
    /// Creates a polynomial with random evaluations.
    pub fn random(num_vars: usize, rng: &mut impl RngCore) -> Self {
        let evals = (0..(1 << num_vars)).map(|_| F::random(rng)).collect();
        Self { evals, num_vars }
    }

    /// Fixes the first (MSB) variable to `scalar` in place, halving the evaluations.
    ///
    /// The evaluations table is laid out so that the first variable controls the
    /// upper/lower half split: indices `0..half` have $x_1 = 0$ and indices
    /// `half..2*half` have $x_1 = 1$. The result is:
    /// $$g(x_2, \ldots, x_n) = f(0, x_2, \ldots) + s \cdot (f(1, x_2, \ldots) - f(0, x_2, \ldots))$$
    ///
    /// Equivalent to `bind_with_order(scalar, BindingOrder::HighToLow)`.
    #[inline]
    pub fn bind(&mut self, scalar: F) {
        self.bind_high_to_low(scalar);
    }

    /// Binds with the specified variable ordering.
    ///
    /// - `BindingOrder::HighToLow`: binds the MSB (first variable, index `0`).
    ///   Pairs `evals[i]` with `evals[i + half]`.
    /// - `BindingOrder::LowToHigh`: binds the LSB (last variable, index `n-1`).
    ///   Pairs `evals[2*i]` with `evals[2*i + 1]`.
    #[inline]
    pub fn bind_with_order(&mut self, scalar: F, order: crate::BindingOrder) {
        match order {
            BindingOrder::HighToLow => self.bind_high_to_low(scalar),
            BindingOrder::LowToHigh => self.bind_low_to_high(scalar),
        }
    }

    #[inline]
    fn bind_high_to_low(&mut self, scalar: F) {
        assert!(self.num_vars > 0, "cannot bind a zero-variable polynomial");
        let half = self.evals.len() / 2;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                let (lo, hi) = self.evals.split_at_mut(half);
                lo.par_iter_mut().zip(hi.par_iter()).for_each(|(a, b)| {
                    *a = *a + scalar * (*b - *a);
                });
            } else {
                for i in 0..half {
                    let lo = self.evals[i];
                    let hi = self.evals[i + half];
                    self.evals[i] = lo + scalar * (hi - lo);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.evals[i];
                let hi = self.evals[i + half];
                self.evals[i] = lo + scalar * (hi - lo);
            }
        }

        self.evals.truncate(half);
        self.num_vars -= 1;
    }

    #[inline]
    fn bind_low_to_high(&mut self, scalar: F) {
        assert!(self.num_vars > 0, "cannot bind a zero-variable polynomial");
        let half = self.evals.len() / 2;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                // Parallel: write into a new buffer to avoid aliasing
                let coeffs = &self.evals;
                let new: Vec<F> = (0..half)
                    .into_par_iter()
                    .map(|i| {
                        let lo = coeffs[2 * i];
                        let hi = coeffs[2 * i + 1];
                        lo + scalar * (hi - lo)
                    })
                    .collect();
                self.evals = new;
            } else {
                for i in 0..half {
                    let lo = self.evals[2 * i];
                    let hi = self.evals[2 * i + 1];
                    self.evals[i] = lo + scalar * (hi - lo);
                }
                self.evals.truncate(half);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.evals[2 * i];
                let hi = self.evals[2 * i + 1];
                self.evals[i] = lo + scalar * (hi - lo);
            }
            self.evals.truncate(half);
        }

        self.num_vars -= 1;
    }

    /// Binds the LSB variable in place without another buffer:
    /// `v[j] = v[2j] + r·(v[2j+1] − v[2j])`. Once the backing allocation
    /// reaches 8x the live length it is released; the return value reports
    /// that release so callers can purge after it.
    ///
    /// Each power-of-two level writes below its read window. Earlier outputs
    /// lie below later reads, and each level splits into disjoint `dst` and
    /// `src`.
    #[inline]
    pub fn bind_low_to_high_in_place(&mut self, scalar: F) -> bool {
        assert!(self.num_vars > 0, "cannot bind a zero-variable polynomial");
        debug_assert!(self.evals.len().is_power_of_two());
        let half = self.evals.len() / 2;

        let (lo, hi) = (self.evals[0], self.evals[1]);
        self.evals[0] = lo + scalar * (hi - lo);
        let mut e = 2;
        while e <= half {
            let (head, tail) = self.evals.split_at_mut(e);
            let dst = &mut head[e / 2..];
            let src = &tail[..e];
            let write = |(k, out): (usize, &mut F)| {
                let lo = src[2 * k];
                *out = lo + scalar * (src[2 * k + 1] - lo);
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                dst.par_iter_mut()
                    .enumerate()
                    .with_min_len(1 << 10)
                    .for_each(write);
            }
            #[cfg(not(feature = "parallel"))]
            dst.iter_mut().enumerate().for_each(write);
            e *= 2;
        }
        self.evals.truncate(half);

        self.num_vars -= 1;
        let shrink = self.evals.capacity() >= 8 * self.evals.len().max(1);
        if shrink {
            self.evals.shrink_to_fit();
        }
        shrink
    }

    /// Binds the LSB variable, writing the result into a caller-provided scratch buffer.
    ///
    /// This has the same semantics as `bind_with_order(scalar, BindingOrder::LowToHigh)`,
    /// but avoids allocating a fresh output vector on every large parallel bind.
    #[inline]
    pub fn bind_low_to_high_reusing_scratch(&mut self, scalar: F, scratch: &mut Vec<F>) {
        assert!(self.num_vars > 0, "cannot bind a zero-variable polynomial");
        let half = self.evals.len() / 2;
        scratch.clear();
        // `reserve` is relative to `len` (0 after `clear`), so this guarantees
        // `capacity >= half`. Reserving only `half - capacity` would no-op
        // whenever the shortfall fits in the existing capacity, leaving the
        // spare-capacity slice below `half` and panicking in the parallel path.
        scratch.reserve(half);

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                let coeffs = &self.evals;
                let spare = &mut scratch.spare_capacity_mut()[..half];
                (spare, coeffs.par_chunks_exact(2))
                    .into_par_iter()
                    .with_min_len(PAR_THRESHOLD)
                    .for_each(|(dest, pair)| {
                        let _ = dest.write(pair[0] + scalar * (pair[1] - pair[0]));
                    });
                // SAFETY: every spare slot in `0..half` is written exactly once above.
                unsafe { scratch.set_len(half) };
                std::mem::swap(&mut self.evals, scratch);
                self.num_vars -= 1;
                return;
            }
        }

        for pair in self.evals.chunks_exact(2) {
            scratch.push(pair[0] + scalar * (pair[1] - pair[0]));
        }
        std::mem::swap(&mut self.evals, scratch);
        self.num_vars -= 1;
    }

    /// Returns the `(lo, hi)` pair for the given index and binding order.
    ///
    /// For sumcheck round polynomial evaluation at index `j`:
    /// - `HighToLow`: `lo = evals[j]`, `hi = evals[j + half]`
    /// - `LowToHigh`: `lo = evals[2*j]`, `hi = evals[2*j + 1]`
    #[inline]
    pub fn sumcheck_eval_pair(&self, index: usize, order: crate::BindingOrder) -> (F, F) {
        match order {
            BindingOrder::HighToLow => {
                let half = self.evals.len() / 2;
                (self.evals[index], self.evals[index + half])
            }
            BindingOrder::LowToHigh => (self.evals[2 * index], self.evals[2 * index + 1]),
        }
    }

    #[inline]
    pub fn sumcheck_round_eval(&self, index: usize, point: F) -> F {
        let (lo, hi) = self.sumcheck_eval_pair(index, BindingOrder::HighToLow);
        lo + point * (hi - lo)
    }

    #[inline]
    pub fn sumcheck_round_eval_with_order(
        &self,
        index: usize,
        point: F,
        order: crate::BindingOrder,
    ) -> F {
        let (lo, hi) = self.sumcheck_eval_pair(index, order);
        lo + point * (hi - lo)
    }

    /// Evaluates the polynomial at `point` using the multilinear extension formula:
    /// $$f(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, r)$$
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();

        #[cfg(feature = "parallel")]
        {
            if self.evals.len() >= PAR_THRESHOLD {
                use rayon::prelude::*;
                return self
                    .evals
                    .par_iter()
                    .zip(eq_evals.par_iter())
                    .map(|(&f, &e)| f * e)
                    .sum();
            }
        }

        self.evals
            .iter()
            .zip(eq_evals.iter())
            .map(|(&f, &e)| f * e)
            .sum()
    }

    /// Evaluates by sequentially binding each variable, consuming `self`.
    ///
    /// More memory-efficient than `evaluate` when the polynomial is no longer needed,
    /// as it avoids materializing the full eq table.
    pub fn evaluate_and_consume(mut self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        for &r in point {
            self.bind(r);
        }
        debug_assert_eq!(self.evals.len(), 1);
        self.evals[0]
    }

    #[inline]
    pub fn evaluations(&self) -> &[F] {
        &self.evals
    }
}

impl<F: JoltField> From<Vec<F>> for Polynomial<F> {
    fn from(evaluations: Vec<F>) -> Self {
        Self::new(evaluations)
    }
}

impl<F: JoltField> crate::MultilinearEvaluation<F> for Polynomial<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn len(&self) -> usize {
        self.evals.len()
    }

    fn evaluate(&self, point: &[F]) -> F {
        Polynomial::evaluate(self, point)
    }
}

impl<F: JoltField> crate::MultilinearBinding<F> for Polynomial<F> {
    fn bind(&mut self, scalar: F) {
        Polynomial::bind(self, scalar);
    }
}

#[inline]
fn assert_matching_dims<F: JoltField>(a: &Polynomial<F>, b: &Polynomial<F>) -> (usize, usize) {
    assert_eq!(
        a.num_vars, b.num_vars,
        "num_vars mismatch: {} vs {}",
        a.num_vars, b.num_vars
    );
    (a.num_vars, a.evals.len())
}

impl<F: JoltField> Add for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

impl<F: JoltField> Add<&Self> for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self {
        self += rhs;
        self
    }
}

impl<F: JoltField> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<F: JoltField> AddAssign<&Self> for Polynomial<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let (_nv, len) = assert_matching_dims(self, rhs);

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.evals
                    .par_iter_mut()
                    .zip(rhs.evals.par_iter())
                    .for_each(|(a, b)| *a += *b);
                return;
            }
        }

        for i in 0..len {
            self.evals[i] += rhs.evals[i];
        }
    }
}

impl<F: JoltField> Sub for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;
        self
    }
}

impl<F: JoltField> Sub<&Self> for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self {
        self -= rhs;
        self
    }
}

impl<F: JoltField> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<F: JoltField> SubAssign<&Self> for Polynomial<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        let (_nv, len) = assert_matching_dims(self, rhs);

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.evals
                    .par_iter_mut()
                    .zip(rhs.evals.par_iter())
                    .for_each(|(a, b)| *a -= *b);
                return;
            }
        }

        for i in 0..len {
            self.evals[i] -= rhs.evals[i];
        }
    }
}

impl<F: JoltField> Mul<F> for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        let len = self.evals.len();

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.evals.par_iter_mut().for_each(|a| *a *= rhs);
                return self;
            }
        }

        for i in 0..len {
            self.evals[i] *= rhs;
        }
        self
    }
}

impl<F: JoltField> Mul<F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, rhs: F) -> Polynomial<F> {
        self.clone() * rhs
    }
}

impl<F: JoltField> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(mut self) -> Self {
        let len = self.evals.len();

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.evals.par_iter_mut().for_each(|a| *a = -*a);
                return self;
            }
        }

        for i in 0..len {
            self.evals[i] = -self.evals[i];
        }
        self
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_field::{Field, Ring};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn bind_to_field_then_evaluate_equals_direct_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let n = 5;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let direct = poly.evaluate(&point);

        let bound = poly.bind_to_field(point[0]);
        let via_bind = bound.evaluate(&point[1..]);

        assert_eq!(direct, via_bind);
    }

    #[test]
    fn empty_polynomial() {
        let poly = Polynomial::<Fr>::new(vec![]);
        assert_eq!(poly.num_vars(), 0);
        assert!(poly.is_empty());
    }

    #[test]
    fn single_evaluation() {
        let val = Fr::from_u64(42);
        let poly = Polynomial::new(vec![val]);
        assert_eq!(poly.num_vars(), 0);
        assert_eq!(poly.evaluate(&[]), val);
    }

    #[test]
    fn serde_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let bytes = bincode::serde::encode_to_vec(&poly, bincode::config::standard()).unwrap();
        let recovered: Polynomial<Fr> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        assert_eq!(poly, recovered);
    }

    #[test]
    fn serde_round_trip_empty() {
        let poly = Polynomial::<Fr>::new(vec![]);
        let bytes = bincode::serde::encode_to_vec(&poly, bincode::config::standard()).unwrap();
        let recovered: Polynomial<Fr> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        assert_eq!(poly, recovered);
    }

    #[test]
    fn parallel_bind_matches_bind_to_field() {
        // n=11 -> 2048 evaluations, above PAR_THRESHOLD=1024
        let mut rng = ChaCha20Rng::seed_from_u64(201);
        let n = 11;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let scalar = Fr::random(&mut rng);

        let bound = poly.bind_to_field(scalar);

        let mut poly_mut = poly;
        poly_mut.bind(scalar);

        assert_eq!(bound.evaluations(), poly_mut.evaluations());
    }

    #[test]
    fn negation() {
        let mut rng = ChaCha20Rng::seed_from_u64(503);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);

        let neg = -poly.clone();
        for i in 0..neg.evaluations().len() {
            assert_eq!(neg.evaluations()[i], -poly.evaluations()[i]);
        }
    }

    #[test]
    fn add_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(504);
        let n = 5;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let sum = a.clone() + &b;
        assert_eq!(
            sum.evaluate(&point),
            a.evaluate(&point) + b.evaluate(&point)
        );
    }

    #[test]
    fn sub_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(505);
        let n = 5;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let diff = a.clone() - &b;
        assert_eq!(
            diff.evaluate(&point),
            a.evaluate(&point) - b.evaluate(&point)
        );
    }

    #[test]
    fn scalar_mul_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(506);
        let n = 5;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let s = Fr::random(&mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let scaled = poly.clone() * s;
        assert_eq!(scaled.evaluate(&point), poly.evaluate(&point) * s);
    }

    #[test]
    #[should_panic(expected = "num_vars mismatch")]
    fn add_mismatched_num_vars_panics() {
        let mut rng = ChaCha20Rng::seed_from_u64(510);
        let a = Polynomial::<Fr>::random(3, &mut rng);
        let b = Polynomial::<Fr>::random(4, &mut rng);
        let _ = a + b;
    }

    #[test]
    fn add_assign_accumulation() {
        let mut rng = ChaCha20Rng::seed_from_u64(511);
        let n = 4;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let c = Polynomial::<Fr>::random(n, &mut rng);

        let mut acc = a.clone();
        acc += &b;
        acc += &c;

        let expected = a.clone() + &b + &c;
        assert_eq!(acc, expected);
    }

    #[test]
    fn ref_scalar_mul() {
        let mut rng = ChaCha20Rng::seed_from_u64(514);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let s = Fr::random(&mut rng);

        let owned_result = poly.clone() * s;
        let ref_result = &poly * s;
        assert_eq!(owned_result, ref_result);
    }

    #[test]
    fn compact_i128_bind_to_field_matches_dense() {
        let scalars: Vec<i128> = vec![-1, 0, 1, -999, i128::MIN, i128::MAX, -7, 7];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(60);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_bind_chain_consistency() {
        let scalars: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(80);
        let r1 = Fr::random(&mut rng);
        let r2 = Fr::random(&mut rng);
        let remaining: Vec<Fr> = (0..1).map(|_| Fr::random(&mut rng)).collect();

        let mut bound = compact.bind_to_field::<Fr>(r1);
        bound.bind(r2);
        let result = bound.evaluate(&remaining);

        let mut full_point = vec![r1, r2];
        full_point.extend_from_slice(&remaining);
        assert_eq!(result, dense.evaluate(&full_point));
    }

    #[test]
    fn bind_low_to_high_reusing_scratch_matches_plain_bind_across_rounds() {
        let mut rng = ChaCha20Rng::seed_from_u64(600);
        // One scratch buffer shared across every polynomial and round,
        // pre-seeded with junk to prove stale contents cannot leak through.
        let mut scratch: Vec<Fr> = vec![Fr::from_u64(0xbad); 7];
        // n = 12 crosses PAR_THRESHOLD on the first bind, then successive
        // rounds shrink below it, covering both the parallel and serial paths.
        for n in [1usize, 2, 5, 12] {
            let poly = Polynomial::<Fr>::random(n, &mut rng);
            let mut with_scratch = poly.clone();
            let mut reference = poly;
            for round in 0..n {
                let challenge = Fr::random(&mut rng);
                reference.bind_with_order(challenge, BindingOrder::LowToHigh);
                with_scratch.bind_low_to_high_reusing_scratch(challenge, &mut scratch);
                assert_eq!(with_scratch, reference, "n={n} round={round}");
            }
            assert_eq!(with_scratch.len(), 1, "n={n}");
        }
    }

    #[test]
    fn sumcheck_round_eval_with_order_equals_bound_evaluations_for_both_orders() {
        let mut rng = ChaCha20Rng::seed_from_u64(602);
        let poly = Polynomial::<Fr>::random(6, &mut rng);
        let point = Fr::random(&mut rng);

        for order in [BindingOrder::HighToLow, BindingOrder::LowToHigh] {
            let mut bound = poly.clone();
            bound.bind_with_order(point, order);
            for index in 0..bound.len() {
                assert_eq!(
                    poly.sumcheck_round_eval_with_order(index, point, order),
                    bound.evaluations()[index],
                    "{order:?} index {index}"
                );
            }
        }
    }

    #[test]
    fn low_to_high_binding_produces_correct_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(900);
        let n = 5;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        // HighToLow binds point[0] first (MSB), so binding sequentially
        // with point[0], point[1], ... should yield evaluate(point).
        let mut hi_to_lo = poly.clone();
        for &r in &point {
            hi_to_lo.bind_with_order(r, BindingOrder::HighToLow);
        }
        assert_eq!(hi_to_lo.len(), 1);
        assert_eq!(hi_to_lo.evaluations()[0], poly.evaluate(&point));

        // LowToHigh binds point[n-1] first (LSB), so to get the same
        // evaluation we must reverse the order of challenges.
        let mut lo_to_hi = poly.clone();
        for &r in point.iter().rev() {
            lo_to_hi.bind_with_order(r, BindingOrder::LowToHigh);
        }
        assert_eq!(lo_to_hi.len(), 1);
        assert_eq!(lo_to_hi.evaluations()[0], poly.evaluate(&point));
    }
}
