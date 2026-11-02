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

/// Binds adjacent evaluation pairs in place, with `boundary` supplying the
/// missing high entry of an odd-length prefix. The implicit suffix itself
/// is the caller's responsibility. Returns whether backing storage shrank.
pub fn bind_prefix_in_place<F: JoltField>(values: &mut Vec<F>, scalar: F, boundary: F) -> bool {
    bind_prefix_waves(values, scalar, &[boundary], |dst, src| {
        let write = |(k, out): (usize, &mut F)| {
            let lo = src[2 * k];
            *out = scalar.mul_add(src[2 * k + 1] - lo, lo);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            dst.par_iter_mut()
                .enumerate()
                .with_min_len(min_parallel_binds::<F>())
                .for_each(write);
        }
        #[cfg(not(feature = "parallel"))]
        dst.iter_mut().enumerate().for_each(write);
    })
}

/// Row-major [`bind_prefix_in_place`]: `values` holds whole rows of
/// `boundary.len()` columns, adjacent rows bind column-wise, and `boundary`
/// supplies the missing high row of an odd-length prefix.
pub fn bind_prefix_rows_in_place<F: JoltField>(
    values: &mut Vec<F>,
    scalar: F,
    boundary: &[F],
) -> bool {
    let columns = boundary.len();
    bind_prefix_waves(values, scalar, boundary, |dst, src| {
        let write = |(out, pair): (&mut [F], &[F])| {
            let (lo, hi) = pair.split_at(columns);
            bind_row(scalar, out, lo, hi);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            dst.par_chunks_exact_mut(columns)
                .zip(src.par_chunks_exact(2 * columns))
                .with_min_len(min_parallel_binds::<F>().div_ceil(columns))
                .for_each(write);
        }
        #[cfg(not(feature = "parallel"))]
        dst.chunks_exact_mut(columns)
            .zip(src.chunks_exact(2 * columns))
            .for_each(write);
    })
}

#[cfg(feature = "parallel")]
fn min_parallel_binds<F: JoltField>() -> usize {
    if F::NUM_BYTES <= 16 {
        1 << 14
    } else {
        1 << 10
    }
}

#[inline]
fn bind_row<F: JoltField>(scalar: F, out: &mut [F], lo: &[F], hi: &[F]) {
    for ((out, &lo), &hi) in out.iter_mut().zip(lo).zip(hi) {
        *out = scalar.mul_add(hi - lo, lo);
    }
}

/// The in-place dyadic wave schedule over rows of `boundary.len()` values.
/// `bind_wave(dst, src)` binds the complete row pairs of `src` into `dst`.
///
/// Each wave writes below its source window, so parallel writes never
/// overwrite an unread entry. No second field buffer is allocated.
fn bind_prefix_waves<F: JoltField>(
    values: &mut Vec<F>,
    scalar: F,
    boundary: &[F],
    bind_wave: impl Fn(&mut [F], &[F]),
) -> bool {
    let columns = boundary.len();
    let rows = values.len().checked_div(columns).unwrap_or(0);
    if rows == 0 {
        return false;
    }
    let (first, rest) = values.split_at_mut(columns);
    let hi = rest.get(..columns).unwrap_or(boundary);
    for (lo, &hi) in first.iter_mut().zip(hi) {
        *lo = scalar.mul_add(hi - *lo, *lo);
    }
    let mut start = 2;
    while start < rows {
        let count = (rows - start).min(start);
        let (head, tail) = values.split_at_mut(start * columns);
        let dst = &mut head[start / 2 * columns..(start / 2 + count.div_ceil(2)) * columns];
        let (complete, odd) = dst.split_at_mut(count / 2 * columns);
        let src = &tail[..count * columns];
        bind_wave(complete, &src[..count / 2 * 2 * columns]);
        if count % 2 != 0 {
            bind_row(scalar, odd, &src[(count - 1) * columns..], boundary);
        }
        start *= 2;
    }
    values.truncate(rows.div_ceil(2) * columns);
    let shrink = values.capacity() >= 8 * values.len().max(1);
    if shrink {
        values.shrink_to_fit();
    }
    shrink
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
            result.push(scalar.mul_add(hi - lo, lo));
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
                lo.par_iter_mut()
                    .zip(hi.par_iter())
                    .with_min_len(PAR_THRESHOLD)
                    .for_each(|(a, b)| {
                        *a = scalar.mul_add(*b - *a, *a);
                    });
            } else {
                for i in 0..half {
                    let lo = self.evals[i];
                    let hi = self.evals[i + half];
                    self.evals[i] = scalar.mul_add(hi - lo, lo);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.evals[i];
                let hi = self.evals[i + half];
                self.evals[i] = scalar.mul_add(hi - lo, lo);
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
                    .with_min_len(PAR_THRESHOLD)
                    .map(|i| {
                        let lo = coeffs[2 * i];
                        let hi = coeffs[2 * i + 1];
                        scalar.mul_add(hi - lo, lo)
                    })
                    .collect();
                self.evals = new;
            } else {
                for i in 0..half {
                    let lo = self.evals[2 * i];
                    let hi = self.evals[2 * i + 1];
                    self.evals[i] = scalar.mul_add(hi - lo, lo);
                }
                self.evals.truncate(half);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.evals[2 * i];
                let hi = self.evals[2 * i + 1];
                self.evals[i] = scalar.mul_add(hi - lo, lo);
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
        let shrunk = bind_prefix_in_place(&mut self.evals, scalar, F::zero());
        self.num_vars -= 1;
        shrunk
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
                        let _ = dest.write(scalar.mul_add(pair[1] - pair[0], pair[0]));
                    });
                // SAFETY: every spare slot in `0..half` is written exactly once above.
                unsafe { scratch.set_len(half) };
                std::mem::swap(&mut self.evals, scratch);
                self.num_vars -= 1;
                return;
            }
        }

        for pair in self.evals.chunks_exact(2) {
            scratch.push(scalar.mul_add(pair[1] - pair[0], pair[0]));
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
    #[test]
    fn prefix_rows_binding_matches_each_column() {
        let columns = 3;
        for live in [0usize, 1, 2, 3, 7, 9, 4095, 4097] {
            let tail: Vec<Fr> = (0..columns).map(|c| Fr::from_u64(97 + c as u64)).collect();
            let mut rows: Vec<_> = (0..live * columns)
                .map(|i| Fr::from_u64(13 * i as u64 + 5))
                .collect();
            let len = live.max(2).next_power_of_two();
            let mut full: Vec<_> = (0..columns)
                .map(|c| {
                    let mut column: Vec<_> = (0..live).map(|r| rows[r * columns + c]).collect();
                    column.resize(len, tail[c]);
                    Polynomial::new(column)
                })
                .collect();
            for round in 0..len.ilog2() {
                let challenge = Fr::from_u64(u64::from(round) + 19);
                let _ = bind_prefix_rows_in_place(&mut rows, challenge, &tail);
                for (c, full) in full.iter_mut().enumerate() {
                    full.bind_with_order(challenge, BindingOrder::LowToHigh);
                    let live = rows.len() / columns;
                    let column: Vec<_> = (0..live).map(|r| rows[r * columns + c]).collect();
                    assert_eq!(&full.evals()[..live], column.as_slice());
                    assert!(full.evals()[live..].iter().all(|value| *value == tail[c]));
                }
            }
        }
    }

    #[test]
    fn prefix_in_place_binding_preserves_constant_extension() {
        for live in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 4095, 4096, 4097] {
            for tail in [Fr::from_u64(0), Fr::from_u64(97)] {
                let mut prefix: Vec<_> =
                    (0..live).map(|i| Fr::from_u64(13 * i as u64 + 5)).collect();
                let len = live.max(2).next_power_of_two();
                let mut full = prefix.clone();
                full.resize(len, tail);
                let mut full = Polynomial::new(full);
                for round in 0..len.ilog2() {
                    let challenge = Fr::from_u64(u64::from(round) + 19);
                    let _ = bind_prefix_in_place(&mut prefix, challenge, tail);
                    full.bind_with_order(challenge, BindingOrder::LowToHigh);
                    assert_eq!(&full.evals()[..prefix.len()], prefix.as_slice());
                    assert!(full.evals()[prefix.len()..]
                        .iter()
                        .all(|value| *value == tail));
                }
            }
        }
    }
}
