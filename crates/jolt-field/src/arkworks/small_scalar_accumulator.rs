use crate::{Limbs, SignedScalarAccumulator};

use super::{bn254::Fr, bn254_ops};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrSmallScalarAccumulator {
    pos: Limbs<5>,
    neg: Limbs<5>,
}

impl Default for FrSmallScalarAccumulator {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: Limbs::zero(),
            neg: Limbs::zero(),
        }
    }
}

impl FrSmallScalarAccumulator {
    #[inline(always)]
    fn add_to_pos(&mut self, value: Fr) {
        self.pos.add_assign_trunc::<4>(&value.inner_limbs());
    }

    #[inline(always)]
    fn add_to_neg(&mut self, value: Fr) {
        self.neg.add_assign_trunc::<4>(&value.inner_limbs());
    }

    #[inline(always)]
    fn fmadd_magnitude_to_pos(&mut self, value: Fr, scalar: u64) {
        if scalar == 0 {
            return;
        }
        if scalar == 1 {
            self.add_to_pos(value);
            return;
        }
        self.pos
            .add_assign_trunc::<5>(&bn254_ops::mul_u64_unreduced(value.0, scalar));
    }

    #[inline(always)]
    fn fmadd_magnitude_to_neg(&mut self, value: Fr, scalar: u64) {
        if scalar == 0 {
            return;
        }
        if scalar == 1 {
            self.add_to_neg(value);
            return;
        }
        self.neg
            .add_assign_trunc::<5>(&bn254_ops::mul_u64_unreduced(value.0, scalar));
    }
}

impl SignedScalarAccumulator for FrSmallScalarAccumulator {
    type Element = Fr;

    #[inline(always)]
    fn add(&mut self, value: Fr) {
        self.add_to_pos(value);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, value: Fr, scalar: u64) {
        self.fmadd_magnitude_to_pos(value, scalar);
    }

    #[inline(always)]
    fn fmadd_i64(&mut self, value: Fr, scalar: i64) {
        let magnitude = scalar.unsigned_abs();
        if scalar >= 0 {
            self.fmadd_magnitude_to_pos(value, magnitude);
        } else {
            self.fmadd_magnitude_to_neg(value, magnitude);
        }
    }

    #[inline(always)]
    fn reduce(self) -> Fr {
        if self.pos >= self.neg {
            Fr::from_inner(bn254_ops::reduce_nplus1(
                self.pos.sub_trunc::<5, 5>(&self.neg),
            ))
        } else {
            -Fr::from_inner(bn254_ops::reduce_nplus1(
                self.neg.sub_trunc::<5, 5>(&self.pos),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::FromPrimitiveInt;

    use super::*;

    #[test]
    fn signed_small_scalar_accumulator_reduces_mixed_terms() {
        let mut acc = FrSmallScalarAccumulator::default();
        acc.fmadd_u64(Fr::from_u64(3), 16);
        acc.fmadd_i64(Fr::from_u64(5), -7);
        acc.add(Fr::from_u64(11));

        assert_eq!(acc.reduce(), Fr::from_u64(24));
    }

    #[test]
    fn signed_small_scalar_accumulator_handles_negative_result() {
        let mut acc = FrSmallScalarAccumulator::default();
        acc.fmadd_i64(Fr::from_u64(9), -13);
        acc.fmadd_u64(Fr::from_u64(2), 7);

        assert_eq!(acc.reduce(), -Fr::from_u64(103));
    }
}
