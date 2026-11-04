use crate::signed::S256;
use crate::{Accumulator, Fp128, Fp128MulU64Accum, Fp128ProductAccum, Ring, Unreduced};

/// Deferred-reduction accumulator for fp128 products.
#[derive(Clone, Copy)]
pub struct Fp128Accumulator<const P: u128>(Fp128ProductAccum);

impl<const P: u128> Default for Fp128Accumulator<P> {
    #[inline(always)]
    fn default() -> Self {
        Self(Fp128ProductAccum([0; 4]))
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
}

impl<const P: u128> Default for Fp128SignedAccumulator<P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: Fp128MulU64Accum([0; 3]),
            neg: Fp128MulU64Accum([0; 3]),
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

    /// Fold the high limbs into three deferred slots. With B = 2^64,
    /// B² ≡ C, so B³*r3 + B⁴*r4 + B⁵*r5 ≡ B*(C*r3 + C²*r5) + C²*r4.
    /// C < 2^32 implies C² + C < B; both intermediate products fit u128.
    #[inline(always)]
    fn product(value: Fp128<P>, magnitude: [u64; 4]) -> Fp128MulU64Accum {
        let [r0, r1, r2, r3, r4, r5] = if magnitude[3] == 0 {
            let [r0, r1, r2, r3, r4] =
                value.mul_wide_limbs::<3, 5>([magnitude[0], magnitude[1], magnitude[2]]);
            [r0, r1, r2, r3, r4, 0]
        } else {
            value.mul_wide_limbs::<4, 6>(magnitude)
        };
        let c = Fp128::<P>::C;
        let low = u128::from(r4) * (c * c);
        let high = u128::from(r3) * c + u128::from(r5) * (c * c);
        Fp128MulU64Accum([
            u128::from(r0) + u128::from(low as u64),
            u128::from(r1) + (low >> 64) + u128::from(high as u64),
            u128::from(r2) + (high >> 64),
        ])
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

    /// The sign folds into the multiplicand, so every term lands in `pos`
    /// without a data-dependent branch. The product's top limb folds through
    /// `2^192 ≡ C·2^64 (mod p)` into the upper slots, which then grow by
    /// `< 2^65` per term — headroom `2^62` terms for the reduction's carry
    /// chain.
    #[inline(always)]
    fn fmadd_i128(&mut self, value: Fp128<P>, scalar: i128) {
        let value = if scalar < 0 { -value } else { value };
        let [r0, r1, r2, r3] = value.mul_wide_u128(scalar.unsigned_abs());
        let folded = u128::from(r3) * u128::from(Fp128::<P>::C_LO);
        self.pos += Fp128MulU64Accum([
            u128::from(r0),
            u128::from(r1) + u128::from(folded as u64),
            u128::from(r2) + (folded >> 64),
        ]);
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
        self.accumulate(Self::product(value, magnitude), scalar.is_positive);
    }
}

/// Deferred field sums retain only the two nonzero base-2^64 slots.
/// Each slot has headroom for at least 2^64 - 1 field additions, including
/// additions represented by merged partial sums.
#[derive(Clone, Copy, Default)]
pub struct Fp128SumAccumulator<const P: u128>([u128; 2]);

impl<const P: u128> Accumulator for Fp128SumAccumulator<P> {
    type Element = Fp128<P>;

    #[inline(always)]
    fn add(&mut self, value: Fp128<P>) {
        for (slot, limb) in self.0.iter_mut().zip(value.to_limbs()) {
            *slot += u128::from(limb);
        }
    }
    #[inline(always)]
    fn merge(&mut self, other: Self) {
        for (slot, other) in self.0.iter_mut().zip(other.0) {
            *slot += other;
        }
    }
    #[inline(always)]
    fn reduce(self) -> Fp128<P> {
        Fp128::<P>::reduce_product(Fp128ProductAccum([self.0[0], self.0[1], 0, 0]))
    }
    #[inline(always)]
    fn fmadd(&mut self, a: Fp128<P>, b: Fp128<P>) {
        self.add(a * b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, NaiveAccumulator};
    use num_bigint::BigUint;
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
            let wide = (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) as i128;
            candidate.fmadd_i128(value, wide);
            expected.fmadd(value, TestField::from_i128(wide));
        }
        let value = <TestField as Field>::random(&mut rng);
        candidate.fmadd_signed_u64(value, u64::MAX, false);
        expected.fmadd_signed_u64(value, u64::MAX, false);
        for extreme in [i128::MIN, i128::MAX, -1] {
            candidate.fmadd_i128(value, extreme);
            expected.fmadd(value, TestField::from_i128(extreme));
        }
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn signed_product_boundaries_and_merge_headroom() {
        fn check<const P: u128>() {
            let modulus = BigUint::from(P);
            for value in [0, 1, P - 1] {
                for magnitude in [
                    [0; 4],
                    [u64::MAX, 0, 0, 0],
                    [u64::MAX, u64::MAX, 0, 0],
                    [u64::MAX, u64::MAX, u64::MAX, 0],
                    [0, 0, 0, 1],
                    [u64::MAX; 4],
                ] {
                    let integer = magnitude.iter().rev().fold(BigUint::from(0u8), |n, &limb| {
                        (n << 64) + BigUint::from(limb)
                    });
                    let product = (BigUint::from(value) * integer) % &modulus;
                    for is_positive in [true, false] {
                        let mut expected = if is_positive || product == BigUint::from(0u8) {
                            product.clone()
                        } else {
                            &modulus - &product
                        };
                        let mut accumulator = Fp128SignedAccumulator::<P>::default();
                        accumulator.fmadd_s256(
                            Fp128::<P>::from_u128(value),
                            &S256::new(magnitude, is_positive),
                        );
                        for log_terms in 0..=62 {
                            let [lo, hi] = accumulator.reduce().to_limbs();
                            let actual = (BigUint::from(hi) << 64) + BigUint::from(lo);
                            assert_eq!(actual, expected, "log_terms={log_terms}");
                            if log_terms < 62 {
                                accumulator.merge(accumulator);
                                expected = (expected << 1) % &modulus;
                            }
                        }
                    }
                }
            }
        }
        check::<TEST_MODULUS>();
        // Largest supported offset; the modular identity does not require primality.
        check::<{ 0u128.wrapping_sub(u32::MAX as u128) }>();
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
