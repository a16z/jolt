#[cfg(feature = "challenge-254-bit")]
use crate::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::challenge::MontU128Challenge;
use crate::{Field, ReductionOps, UnreducedField, UnreducedOps, WithChallenge};
use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rand_core::RngCore;

type Fr = ark_bn254::Fr;
type FrConfig = ark_bn254::FrConfig;

impl Field for Fr {
    const NUM_BYTES: usize = 32;

    fn random<R: RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = <Self as ark_ff::PrimeField>::into_bigint(*self);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as Field>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn num_bits(&self) -> u32 {
        <Self as ark_ff::PrimeField>::into_bigint(*self).num_bits()
    }

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bool(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
    }

    fn from_u8(n: u8) -> Self {
        Self::from(n as u64)
    }

    fn from_u16(n: u16) -> Self {
        Self::from(n as u64)
    }

    fn from_u32(n: u32) -> Self {
        Self::from(n as u64)
    }

    fn from_u64(n: u64) -> Self {
        Self::from(n)
    }

    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            -Self::from(val.unsigned_abs())
        } else {
            Self::from(val as u64)
        }
    }

    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            -Self::from_u128(val.unsigned_abs())
        } else {
            Self::from_u128(val as u128)
        }
    }

    fn from_u128(val: u128) -> Self {
        if val <= u64::MAX as u128 {
            Self::from(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
        }
    }
}

impl<const N: usize> UnreducedField<Fr> for BigInt<N> {
    fn mul_trunc<const M: usize>(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let len = std::cmp::min(N, N + N);

        for i in 0..std::cmp::min(N, len) {
            let mut carry = 0u64;
            for j in 0..std::cmp::min(N, len - i) {
                if i + j < N {
                    let product = (self.0[i] as u128) * (other.0[j] as u128) + (result.0[i + j] as u128) + (carry as u128);
                    result.0[i + j] = product as u64;
                    carry = (product >> 64) as u64;
                }
            }
            if i + N < N && i + N < len {
                result.0[i + N] = carry;
            }
        }

        result
    }

    fn add_assign_mixed<const M: usize>(&mut self, other: &Self) {
        *self += *other;
    }
}

impl UnreducedOps for Fr {
    type UnreducedType = BigInt<8>;

    fn as_unreduced_ref(&self) -> &Self::UnreducedType {
        // SAFETY: We're casting a reference to Fr (which contains BigInt<4>)
        // to a reference to BigInt<8>. The first 4 limbs are valid.
        unsafe {
            let ptr = self as *const Self as *const BigInt<4>;
            let extended_ptr = ptr as *const BigInt<8>;
            &*extended_ptr
        }
    }

    fn mul_unreduced(self, other: Self) -> Self::UnreducedType {
        let mut result = Self::UnreducedType::zero();
        let a = self.into_bigint();
        let b = other.into_bigint();

        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..4 {
                if i + j < 8 {
                    let product = (a.0[i] as u128) * (b.0[j] as u128) + (result.0[i + j] as u128) + (carry as u128);
                    result.0[i + j] = product as u64;
                    carry = (product >> 64) as u64;
                }
            }
            if i + 4 < 8 {
                result.0[i + 4] = carry;
            }
        }

        result
    }

    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedType {
        let mut result = Self::UnreducedType::zero();
        let a = self.into_bigint();
        let mut carry = 0u64;

        for i in 0..4 {
            let product = (a.0[i] as u128) * (other as u128) + (carry as u128);
            result.0[i] = product as u64;
            carry = (product >> 64) as u64;
        }
        result.0[4] = carry;

        result
    }

    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedType {
        // Split the u128 into two u64s: low and high
        let low = other as u64;
        let high = (other >> 64) as u64;

        // Multiply by the low part
        let mut result = self.mul_u64_unreduced(low);

        // If high part is non-zero, multiply and add shifted
        if high != 0 {
            let high_result = self.mul_u64_unreduced(high);
            let mut carry = 0u64;
            for i in 0..7 {
                let sum = (result.0[i + 1] as u128) + (high_result.0[i] as u128) + (carry as u128);
                result.0[i + 1] = sum as u64;
                carry = (sum >> 64) as u64;
            }
        }

        result
    }
}

impl ReductionOps for Fr {
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R)
    };
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R2)
    };

    fn from_montgomery_reduce(unreduced: Self::UnreducedType) -> Self {
        use crate::{MontgomeryReduce, FMAdd};
        let mut result = unreduced;
        result.fmadd(&<Self as ReductionOps>::MONTGOMERY_R_SQUARE.into_bigint(), &BigInt::<4>::zero());
        result.montgomery_reduce()
    }

    fn from_barrett_reduce(unreduced: Self::UnreducedType) -> Self {
        use crate::BarrettReduce;
        unreduced.barrett_reduce()
    }
}

impl WithChallenge for Fr {
    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<Fr>;

    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<Fr>;
}

impl<const N: usize, const M: usize> crate::FMAdd<BigInt<4>, BigInt<M>> for BigInt<N> {
    fn fmadd(&mut self, left: &BigInt<4>, right: &BigInt<M>) {
        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..M {
                if i + j < N {
                    let product = (left.0[i] as u128) * (right.0[j] as u128) + (self.0[i + j] as u128) + (carry as u128);
                    self.0[i + j] = product as u64;
                    carry = (product >> 64) as u64;
                } else {
                    break;
                }
            }
            if i + M < N {
                self.0[i + M] = self.0[i + M].wrapping_add(carry);
            }
        }
    }
}

impl crate::MontgomeryReduce<Fr> for BigInt<8> {
    fn montgomery_reduce(&self) -> Fr {
        // Convert to bytes and use mod_order reduction
        let mut bytes = [0u8; 64];
        for i in 0..8 {
            let limb_bytes = self.0[i].to_le_bytes();
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb_bytes);
        }
        Fr::from_le_bytes_mod_order(&bytes)
    }
}

impl crate::BarrettReduce<Fr> for BigInt<8> {
    fn barrett_reduce(&self) -> Fr {
        // Convert to bytes and use mod_order reduction
        let mut bytes = [0u8; 64];
        for i in 0..8 {
            let limb_bytes = self.0[i].to_le_bytes();
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb_bytes);
        }
        Fr::from_le_bytes_mod_order(&bytes)
    }
}