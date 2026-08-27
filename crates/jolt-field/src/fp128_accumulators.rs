use core::marker::PhantomData;

use crate::signed::S256;
use crate::{Accumulator, Fp128, Fp128MulU64Accum, Fp128ProductAccum, Ring, Unreduced};

/// Deferred-reduction accumulator for fp128 products.
#[derive(Clone, Copy)]
pub struct Fp128Accumulator<const P: u128>(Fp128ProductAccum, PhantomData<Fp128<P>>);

impl<const P: u128> Default for Fp128Accumulator<P> {
    #[inline(always)]
    fn default() -> Self {
        Self(Fp128ProductAccum([0; 4]), PhantomData)
    }
}

impl<const P: u128> Accumulator for Fp128Accumulator<P> {
    type Element = Fp128<P>;

    #[inline(always)]
    fn add(&mut self, value: Fp128<P>) {
        self.0 += Fp128ProductAccum::from(value);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline(always)]
    fn reduce(self) -> Fp128<P> {
        Fp128::<P>::reduce_product(self.0)
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fp128<P>, b: Fp128<P>) {
        self.0 += a.mul_unreduced(b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, a: Fp128<P>, b: u64) {
        let product = a.mul_u64_unreduced(b);
        self.0 .0[0] += product.0[0];
        self.0 .0[1] += product.0[1];
        self.0 .0[2] += product.0[2];
    }

    #[inline(always)]
    fn fmadd_u128(&mut self, a: Fp128<P>, b: u128) {
        self.0 += a.mul_unreduced(Fp128::<P>::from_u128(b));
    }
}

/// Deferred-reduction accumulator for signed scalar products.
#[derive(Clone, Copy)]
pub struct Fp128SignedAccumulator<const P: u128> {
    pos: Fp128MulU64Accum,
    neg: Fp128MulU64Accum,
    marker: PhantomData<Fp128<P>>,
}

impl<const P: u128> Default for Fp128SignedAccumulator<P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: Fp128MulU64Accum([0; 3]),
            neg: Fp128MulU64Accum([0; 3]),
            marker: PhantomData,
        }
    }
}

impl<const P: u128> Fp128SignedAccumulator<P> {
    #[inline(always)]
    fn accumulate(&mut self, term: Fp128MulU64Accum, is_positive: bool) {
        if is_positive {
            self.pos += term;
        } else {
            self.neg += term;
        }
    }

    #[inline(always)]
    fn reduce_signed(self) -> Fp128<P> {
        Fp128::<P>::reduce_small_product(self.pos) - Fp128::<P>::reduce_small_product(self.neg)
    }

    #[inline(always)]
    fn product(value: Fp128<P>, magnitude: [u64; 4]) -> Fp128<P> {
        if magnitude[3] == 0 {
            Fp128::<P>::solinas_reduce(&value.mul_wide_limbs::<3, 5>([
                magnitude[0],
                magnitude[1],
                magnitude[2],
            ]))
        } else {
            Fp128::<P>::solinas_reduce(&value.mul_wide_limbs::<4, 6>(magnitude))
        }
    }
}

impl<const P: u128> Accumulator for Fp128SignedAccumulator<P> {
    type Element = Fp128<P>;

    #[inline(always)]
    fn add(&mut self, value: Fp128<P>) {
        self.pos += Fp128MulU64Accum::from(value);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }

    #[inline(always)]
    fn reduce(self) -> Fp128<P> {
        self.reduce_signed()
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fp128<P>, b: Fp128<P>) {
        self.add(a * b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, value: Fp128<P>, scalar: u64) {
        if scalar != 0 {
            self.accumulate(value.mul_u64_unreduced(scalar), true);
        }
    }

    #[inline(always)]
    fn fmadd_i64(&mut self, value: Fp128<P>, scalar: i64) {
        if scalar != 0 {
            self.accumulate(value.mul_u64_unreduced(scalar.unsigned_abs()), scalar > 0);
        }
    }

    #[inline(always)]
    fn fmadd_signed_u64(&mut self, value: Fp128<P>, magnitude: u64, is_positive: bool) {
        if magnitude != 0 {
            self.accumulate(value.mul_u64_unreduced(magnitude), is_positive);
        }
    }

    #[inline(always)]
    fn fmadd_s256(&mut self, value: Fp128<P>, scalar: &S256) {
        let magnitude = scalar.magnitude_limbs();
        if magnitude == [0; 4] {
            return;
        }
        self.accumulate(
            Fp128MulU64Accum::from(Self::product(value, magnitude)),
            scalar.is_positive,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, NaiveAccumulator};
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    const TEST_MODULUS: u128 = u128::MAX - 274;
    type TestField = Fp128<TEST_MODULUS>;

    #[test]
    fn ring_accumulator_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut candidate = Fp128Accumulator::<TEST_MODULUS>::default();
        let mut expected = NaiveAccumulator::<TestField>::default();
        for _ in 0..16_384 {
            let a = <TestField as Field>::random(&mut rng);
            let b = <TestField as Field>::random(&mut rng);
            candidate.fmadd(a, b);
            expected.fmadd(a, b);
        }
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn small_scalar_accumulator_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let mut candidate = Fp128SignedAccumulator::<TEST_MODULUS>::default();
        let mut expected = NaiveAccumulator::<TestField>::default();
        for index in 0..16_384 {
            let value = <TestField as Field>::random(&mut rng);
            let scalar = rng.next_u64() as i64;
            let scalar = if index % 2 == 0 { scalar } else { -scalar };
            candidate.fmadd_i64(value, scalar);
            expected.fmadd_i64(value, scalar);
        }
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn signed_product_accumulator_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let mut candidate = Fp128SignedAccumulator::<TEST_MODULUS>::default();
        let mut expected = NaiveAccumulator::<TestField>::default();
        for index in 0..16_384 {
            let value = <TestField as Field>::random(&mut rng);
            let scalar = S256::new(
                [
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    if index % 3 == 0 { rng.next_u64() } else { 0 },
                ],
                index % 2 == 0,
            );
            candidate.fmadd_s256(value, &scalar);
            expected.fmadd_s256(value, &scalar);
        }
        let value = <TestField as Field>::random(&mut rng);
        candidate.fmadd_signed_u64(value, u64::MAX, false);
        expected.fmadd_signed_u64(value, u64::MAX, false);
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn accumulators_handle_add_and_zero() {
        let value = TestField::from_u64(7);
        let mut ring = Fp128Accumulator::<TEST_MODULUS>::default();
        ring.add(value);
        ring.fmadd_u64(value, 5);
        assert_eq!(ring.reduce(), TestField::from_u64(42));

        let mut small = Fp128SignedAccumulator::<TEST_MODULUS>::default();
        small.add(value);
        small.fmadd_i64(value, -1);
        assert_eq!(small.reduce(), TestField::from_u64(0));
    }
}
