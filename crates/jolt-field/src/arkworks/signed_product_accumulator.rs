use crate::{signed::S256, Limbs, SignedProductAccumulator};
use ark_ff::{BigInt, MontConfig};
use num_traits::Zero;

use super::{bn254::Fr, bn254_ops};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrSignedProductAccumulator {
    pos: [u128; 8],
    neg: [u128; 8],
}

impl Default for FrSignedProductAccumulator {
    #[inline]
    fn default() -> Self {
        Self {
            pos: [0; 8],
            neg: [0; 8],
        }
    }
}

impl FrSignedProductAccumulator {
    #[inline(always)]
    fn fmadd_magnitude(slots: &mut [u128; 8], value: Fr, magnitude: Limbs<4>) {
        let value = value.inner_limbs();
        for i in 0..4 {
            for j in 0..4 {
                let product = (value.0[i] as u128) * (magnitude.0[j] as u128);
                slots[i + j] += (product as u64) as u128;
                slots[i + j + 1] += ((product >> 64) as u64) as u128;
            }
        }
    }

    #[inline]
    fn normalize(slots: [u128; 8]) -> Limbs<9> {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (index, slot) in slots.into_iter().enumerate() {
            let (sum, overflow) = slot.overflowing_add(carry);
            out[index] = sum as u64;
            carry = (sum >> 64) + ((overflow as u128) << 64);
        }
        out[8] = carry as u64;
        Limbs(out)
    }
}

impl SignedProductAccumulator for FrSignedProductAccumulator {
    type Element = Fr;

    #[inline(always)]
    fn fmadd_s256(&mut self, value: Fr, scalar: &S256) {
        if scalar.is_zero() {
            return;
        }
        if scalar.is_positive {
            Self::fmadd_magnitude(&mut self.pos, value, scalar.magnitude);
        } else {
            Self::fmadd_magnitude(&mut self.neg, value, scalar.magnitude);
        }
    }

    #[inline]
    fn reduce(self) -> Fr {
        let pos = Self::normalize(self.pos);
        let neg = Self::normalize(self.neg);
        let montgomery_r_value =
            Fr::from_bigint_unchecked(Limbs(<ark_bn254::FrConfig as MontConfig<4>>::R2.0));
        let reduced = if pos >= neg {
            Fr::from_inner(bn254_ops::from_montgomery_reduce(BigInt::from(
                pos.sub_trunc::<9, 9>(&neg),
            )))
        } else {
            -Fr::from_inner(bn254_ops::from_montgomery_reduce(BigInt::from(
                neg.sub_trunc::<9, 9>(&pos),
            )))
        };
        reduced * montgomery_r_value
    }
}

#[cfg(test)]
mod tests {
    use crate::FromPrimitiveInt;

    use super::*;

    fn s256_to_fr(value: &S256) -> Fr {
        let mut bytes = [0u8; 32];
        for (index, limb) in value.magnitude_limbs().iter().copied().enumerate() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        let magnitude = Fr::from_le_bytes_mod_order(&bytes);
        if value.is_positive {
            magnitude
        } else {
            -magnitude
        }
    }

    #[test]
    fn signed_product_accumulator_reduces_mixed_terms() {
        let terms = [
            (Fr::from_u64(3), S256::from_i128(17)),
            (Fr::from_u64(11), S256::from_i128(-9)),
            (Fr::from_u64(42), S256::new([7, 5, 3, 1], true)),
            (Fr::from_u64(6), S256::new([u64::MAX, 19, 0, 0], false)),
        ];

        let mut acc = FrSignedProductAccumulator::default();
        let mut expected = Fr::from_u64(0);
        for (field, scalar) in terms {
            acc.fmadd_s256(field, &scalar);
            expected += field * s256_to_fr(&scalar);
        }

        assert_eq!(acc.reduce(), expected);
    }
}
