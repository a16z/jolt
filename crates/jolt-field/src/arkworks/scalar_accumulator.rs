//! Deferred-reduction accumulator for BN254 Fr times raw integer scalars.

use crate::accumulator::FieldScalarAccumulator;
use crate::arkworks::bn254::Fr;
use crate::Limbs;

/// Accumulates sums of `Fr * u64/u128` in Montgomery form.
#[derive(Clone, Copy)]
pub struct ScalarAccumulator {
    /// A field element has 4 limbs; multiplying by a u128 gives 6 limbs, and
    /// the extra limb provides carry headroom for many bucket additions.
    limbs: Limbs<7>,
}

impl Default for ScalarAccumulator {
    #[inline]
    fn default() -> Self {
        Self {
            limbs: Limbs::zero(),
        }
    }
}

impl FieldScalarAccumulator for ScalarAccumulator {
    type Field = Fr;

    #[inline(always)]
    fn add(&mut self, value: Fr) {
        self.add_mul_u64(value, 1);
    }

    #[inline(always)]
    fn add_mul_u64(&mut self, value: Fr, scalar: u64) {
        if scalar != 0 {
            self.limbs
                .fmadd::<4, 1>(&value.inner_limbs(), &Limbs::<1>::from_u64(scalar));
        }
    }

    #[inline(always)]
    fn add_mul_u128(&mut self, value: Fr, scalar: u128) {
        if scalar >> 64 == 0 {
            self.add_mul_u64(value, scalar as u64);
        } else {
            let scalar_limbs = Limbs::<2>::new([scalar as u64, (scalar >> 64) as u64]);
            self.limbs
                .fmadd::<4, 2>(&value.inner_limbs(), &scalar_limbs);
        }
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.limbs.add_assign_trunc::<7>(&other.limbs);
    }

    #[inline(always)]
    fn reduce(self) -> Fr {
        Fr::from_barrett_reduced_limbs(self.limbs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn accumulates_raw_scalars() {
        let mut acc = ScalarAccumulator::default();
        acc.add(Fr::from_u64(3));
        acc.add_mul_u64(Fr::from_u64(5), 7);
        acc.add_mul_u128(Fr::from_u64(11), 1u128 << 80);

        let expected =
            Fr::from_u64(3) + Fr::from_u64(5).mul_u64(7) + Fr::from_u64(11).mul_u128(1u128 << 80);
        assert_eq!(acc.reduce(), expected);
    }
}
