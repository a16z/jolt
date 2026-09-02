use akita_config::proof_optimized::fp128::Field as AkitaField;
use akita_field::unreduced::{Fp128MulU64Accum, Fp128ProductAccum, HasUnreducedOps};

use crate::signed::S256;
use crate::Accumulator;

/// Deferred-reduction accumulator for fp128 products.
#[derive(Clone, Copy)]
pub struct AkitaAccumulator(Fp128ProductAccum);

impl Default for AkitaAccumulator {
    #[inline(always)]
    fn default() -> Self {
        Self(Fp128ProductAccum::ZERO)
    }
}

impl Accumulator for AkitaAccumulator {
    type Element = AkitaField;

    #[inline(always)]
    fn add(&mut self, value: AkitaField) {
        self.0 += Fp128ProductAccum::from(value);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline(always)]
    fn reduce(self) -> AkitaField {
        AkitaField::reduce_product_accum(self.0)
    }

    #[inline(always)]
    fn fmadd(&mut self, a: AkitaField, b: AkitaField) {
        self.0 += a.mul_to_product_accum(b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, a: AkitaField, b: u64) {
        let product = a.mul_u64_unreduced(b);
        self.0 .0[0] += product.0[0];
        self.0 .0[1] += product.0[1];
        self.0 .0[2] += product.0[2];
    }
}

/// Deferred-reduction accumulator for signed scalar products.
#[derive(Clone, Copy)]
pub struct AkitaSignedAccumulator {
    pos: Fp128MulU64Accum,
    neg: Fp128MulU64Accum,
}

impl Default for AkitaSignedAccumulator {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: Fp128MulU64Accum::ZERO,
            neg: Fp128MulU64Accum::ZERO,
        }
    }
}

impl AkitaSignedAccumulator {
    #[inline(always)]
    fn accumulate(&mut self, term: Fp128MulU64Accum, is_positive: bool) {
        if is_positive {
            self.pos += term;
        } else {
            self.neg += term;
        }
    }

    #[inline(always)]
    fn reduce_signed(self) -> AkitaField {
        AkitaField::reduce_mul_u64_accum(self.pos) - AkitaField::reduce_mul_u64_accum(self.neg)
    }

    #[inline(always)]
    fn product(value: AkitaField, magnitude: [u64; 4]) -> AkitaField {
        if magnitude[3] == 0 {
            AkitaField::solinas_reduce(&value.mul_wide_limbs::<3, 5>([
                magnitude[0],
                magnitude[1],
                magnitude[2],
            ]))
        } else {
            AkitaField::solinas_reduce(&value.mul_wide_limbs::<4, 6>(magnitude))
        }
    }
}

impl Accumulator for AkitaSignedAccumulator {
    type Element = AkitaField;

    #[inline(always)]
    fn add(&mut self, value: AkitaField) {
        self.pos += Fp128MulU64Accum::from(value);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }

    #[inline(always)]
    fn reduce(self) -> AkitaField {
        self.reduce_signed()
    }

    #[inline(always)]
    fn fmadd(&mut self, a: AkitaField, b: AkitaField) {
        self.add(a * b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, value: AkitaField, scalar: u64) {
        if scalar != 0 {
            self.accumulate(value.mul_u64_unreduced(scalar), true);
        }
    }

    #[inline(always)]
    fn fmadd_i64(&mut self, value: AkitaField, scalar: i64) {
        if scalar != 0 {
            self.accumulate(value.mul_u64_unreduced(scalar.unsigned_abs()), scalar > 0);
        }
    }

    #[inline(always)]
    fn fmadd_signed_u64(&mut self, value: AkitaField, magnitude: u64, is_positive: bool) {
        if magnitude != 0 {
            self.accumulate(value.mul_u64_unreduced(magnitude), is_positive);
        }
    }

    #[inline(always)]
    fn fmadd_s256(&mut self, value: AkitaField, scalar: &S256) {
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

    #[test]
    fn ring_accumulator_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut candidate = AkitaAccumulator::default();
        let mut expected = NaiveAccumulator::<AkitaField>::default();
        for _ in 0..16_384 {
            let a = <AkitaField as Field>::random(&mut rng);
            let b = <AkitaField as Field>::random(&mut rng);
            candidate.fmadd(a, b);
            expected.fmadd(a, b);
        }
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn small_scalar_accumulator_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let mut candidate = AkitaSignedAccumulator::default();
        let mut expected = NaiveAccumulator::<AkitaField>::default();
        for index in 0..16_384 {
            let value = <AkitaField as Field>::random(&mut rng);
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
        let mut candidate = AkitaSignedAccumulator::default();
        let mut expected = NaiveAccumulator::<AkitaField>::default();
        for index in 0..16_384 {
            let value = <AkitaField as Field>::random(&mut rng);
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
        let value = <AkitaField as Field>::random(&mut rng);
        candidate.fmadd_signed_u64(value, u64::MAX, false);
        expected.fmadd_signed_u64(value, u64::MAX, false);
        assert_eq!(candidate.reduce(), expected.reduce());
    }

    #[test]
    fn accumulators_handle_add_and_zero() {
        let value = AkitaField::from_u64(7);
        let mut ring = AkitaAccumulator::default();
        ring.add(value);
        ring.fmadd_u64(value, 5);
        assert_eq!(ring.reduce(), AkitaField::from_u64(42));

        let mut small = AkitaSignedAccumulator::default();
        small.add(value);
        small.fmadd_i64(value, -1);
        assert_eq!(small.reduce(), AkitaField::from_u64(0));
    }
}
