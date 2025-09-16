use super::{FieldOps, JoltField};
use crate::utils::thread::unsafe_allocate_zero_vec;
use allocative::Allocative;
use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use rayon::prelude::*;
use std::io::{Read, Write};

impl FieldOps for ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for &ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

lazy_static::lazy_static! {
    static ref SMALL_VALUE_LOOKUP_TABLES: [Vec<ark_bn254::Fr>; 2] = ark_bn254::Fr::compute_lookup_tables();
}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;
    type SmallValueLookupTables = [Vec<Self>; 2];

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        // These two lookup tables correspond to the two 16-bit limbs of a u64
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..2 {
            let bitshift = 16 * i;
            let unit = <Self as ark_ff::PrimeField>::from_u64(1 << bitshift).unwrap();
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as ark_ff::PrimeField>::from_u64(j).unwrap())
                .collect();
        }

        lookup_tables
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        // The new `from_u64` is faster than doing 4 lookups & adding them together
        // but it's slower than doing <=2 lookups & adding them together (if n fits in u16 or u32)
        if n <= u16::MAX as u64 {
            <Self as JoltField>::from_u16(n as u16)
        } else if n <= u32::MAX as u64 {
            <Self as JoltField>::from_u32(n as u32)
        } else {
            <Self as ark_ff::PrimeField>::from_u64(n).unwrap()
        }
    }

    // We assume that MontU128 is a Big Into with the least significant digits set to 0.
    // In Arkworks 0 index is the least significant digit.
    #[inline(always)]
    fn from_u128_mont(n: MontU128) -> Self {
        let n_val = n.0;

        <Self as ark_ff::PrimeField>::from_bigint_unchecked(BigInt::new(n_val)).unwrap()
    }

    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u64 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                -<Self as JoltField>::from_u32(val as u32)
            } else {
                -<Self as JoltField>::from_u64(val)
            }
        } else {
            let val = val as u64;
            if val <= u16::MAX as u64 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                <Self as JoltField>::from_u32(val as u32)
            } else {
                <Self as JoltField>::from_u64(val)
            }
        }
    }

    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u128 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                -<Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                -<Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                -<Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        } else {
            let val = val as u128;
            if val <= u16::MAX as u128 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                <Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                <Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        }
    }

    fn from_u128(val: u128) -> Self {
        if val <= u16::MAX as u128 {
            <Self as JoltField>::from_u16(val as u16)
        } else if val <= u32::MAX as u128 {
            <Self as JoltField>::from_u32(val as u32)
        } else if val <= u64::MAX as u128 {
            <Self as JoltField>::from_u64(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
        }
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = self.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as JoltField>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }

    fn num_bits(&self) -> u32 {
        self.into_bigint().num_bits()
    }

    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else if self.is_one() {
            <Self as JoltField>::from_u64(n)
        } else {
            ark_ff::Fp::mul_u64(*self, n)
        }
    }

    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else if self.is_one() {
            <Self as JoltField>::from_i128(n)
        } else {
            ark_ff::Fp::mul_i128(*self, n)
        }
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        ark_ff::Fp::mul_u128(*self, n)
    }

    #[inline(always)]
    fn mul_u128_mont_form(&self, n: MontU128) -> Self {
        let n_val = n.0;
        ark_ff::Fp::mul_hi_u128(*self, n_val)
    }

    //#[inline(always)]
    //fn mul_two_u128s(&self, x: MontU128, y: MontU128) -> Self {
    //    let x_val = x.0;
    //    let y_val = y.0;
    //    if x_val == 0 || y_val == 0 {
    //        Self::zero()
    //    } else if x_val == 1 {
    //        Self::from_u128(y_val)
    //    } else if y_val == 1 {
    //        Self::from_u128(x_val)
    //    } else {
    //        // here you need a low-level method from ark_ff for 128x128 multiplication
    //        ark_ff::Fp::mul_two_u128s(self, x_val, y_val)
    //    }
    //}
}

// Provide ergonomic operators for multiplying by a u128 already in Montgomery form.
impl core::ops::Mul<MontU128> for ark_bn254::Fr {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: MontU128) -> Self::Output {
        Self::mul_u128_mont_form(&self, rhs)
    }
}

impl core::ops::Mul<MontU128> for &ark_bn254::Fr {
    type Output = ark_bn254::Fr;
    #[inline(always)]
    fn mul(self, rhs: MontU128) -> Self::Output {
        <ark_bn254::Fr as JoltField>::mul_u128_mont_form(self, rhs)
    }
}

impl core::ops::MulAssign<MontU128> for ark_bn254::Fr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: MontU128) {
        *self = Self::mul_u128_mont_form(self, rhs);
    }
}

// --- Add ---
impl core::ops::Add<MontU128> for ark_bn254::Fr {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: MontU128) -> Self::Output {
        self + Self::from_u128_mont(rhs)
    }
}

impl core::ops::Add<MontU128> for &ark_bn254::Fr {
    type Output = ark_bn254::Fr;
    #[inline(always)]
    fn add(self, rhs: MontU128) -> Self::Output {
        // Here Self is &ark_bn254::Fr, not ark_bn254::Fr.
        // so we need the explicit type and not Self::from_u128_form
        *self + ark_bn254::Fr::from_u128_mont(rhs)
    }
}

impl core::ops::AddAssign<MontU128> for ark_bn254::Fr {
    #[inline(always)]
    fn add_assign(&mut self, rhs: MontU128) {
        *self += Self::from_u128_mont(rhs);
    }
}

// --- Sub ---
impl core::ops::Sub<MontU128> for ark_bn254::Fr {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: MontU128) -> Self::Output {
        self - Self::from_u128_mont(rhs)
    }
}

impl core::ops::Sub<MontU128> for &ark_bn254::Fr {
    type Output = ark_bn254::Fr;
    #[inline(always)]
    fn sub(self, rhs: MontU128) -> Self::Output {
        *self - <ark_bn254::Fr as JoltField>::from_u128_mont(rhs)
    }
}

impl core::ops::SubAssign<MontU128> for ark_bn254::Fr {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: MontU128) {
        *self = *self - Self::from_u128_mont(rhs);
    }
}

/// A zero cost-wrapper around `u128` indicating that the value is already
/// in Montgomery form for the target field
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Allocative)]
pub struct MontU128([u64; 4]);

impl From<u128> for MontU128 {
    fn from(val: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        let val_masked = val & (u128::MAX >> 3);

        let low = val_masked as u64;
        let high = (val_masked >> 64) as u64;
        // let bigint = BigInt::new([0, 0, low, high]);

        MontU128([0, 0, low, high])
    }
}

// impl From<MontU128> for u128 {
//     fn from(val: MontU128) -> u128 {
//         val.0
//     }
// }

impl From<u64> for MontU128 {
    fn from(val: u64) -> Self {
        MontU128::from(val as u128)
    }
}

// === Serialization impls ===

impl CanonicalSerialize for MontU128 {
    // fn serialize<W: Write>(&self, writer: W) -> Result<(), SerializationError> {
    //     self.serialize_uncompressed(writer)
    // }
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.serialize_uncompressed(writer)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.uncompressed_size()
    }

    fn serialize_compressed<W: Write>(&self, writer: W) -> Result<(), SerializationError> {
        // u128 doesn’t really compress, so just write as 16 bytes
        self.serialize_uncompressed(writer)
    }

    fn compressed_size(&self) -> usize {
        16
    }

    // fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
    //     writer.write_all(&self.0.to_le_bytes())?;
    //     Ok(())
    // }
    fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        for limb in &self.0 {
            writer.write_all(&limb.to_le_bytes())?;
        }
        Ok(())
    }

    fn uncompressed_size(&self) -> usize {
        16
    }
}

impl Valid for MontU128 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for MontU128 {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_uncompressed(reader)
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_uncompressed(reader)
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_uncompressed_unchecked(reader)
    }

    // fn deserialize_uncompressed<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
    //     let mut buf = [0u8; 16];
    //     reader.read_exact(&mut buf)?;
    //     Ok(MontU128(u128::from_le_bytes(buf)))
    // }
    fn deserialize_uncompressed<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let mut buf = [0u8; 32]; // 4 limbs × 8 bytes each
        reader.read_exact(&mut buf)?;

        let mut limbs = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            let end = start + 8;
            limbs[i] = u64::from_le_bytes(buf[start..end].try_into().unwrap());
        }

        Ok(MontU128(limbs))
    }

    // fn deserialize_uncompressed_unchecked<R: Read>(
    //     mut reader: R,
    // ) -> Result<Self, SerializationError> {
    //     let mut buf = [0u8; 16];
    //     reader.read_exact(&mut buf)?;
    //     Ok(MontU128(u128::from_le_bytes(buf)))
    // }
    fn deserialize_uncompressed_unchecked<R: Read>(
        mut reader: R,
    ) -> Result<Self, SerializationError> {
        let mut buf = [0u8; 32]; // 4 * 8 = 32 bytes
        reader.read_exact(&mut buf)?;

        let mut limbs = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            let end = start + 8;
            limbs[i] = u64::from_le_bytes(buf[start..end].try_into().unwrap());
        }

        Ok(MontU128(limbs))
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;
    use ark_std::test_rng;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_chacha::rand_core::RngCore;
    #[test]
    fn implicit_montgomery_conversion() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = rng.next_u64();
            assert_eq!(<Fr as JoltField>::from_u64(x), Fr::one().mul_u64(x));
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = Fr::random(&mut rng);
            assert_eq!(y * <Fr as JoltField>::from_u64(x), y.mul_u64(x));
        }
    }

    #[test]
    fn test_small_binding_multiplications() {
        // SANITY CHECK
        let b_1 = Fr::new_unchecked(BigInt([0, 0, 1, 0]));
        // as z is small enough the masking should not matter
        // and it should be stored as the BigInt above
        let z = MontU128::from(1_u128);
        let c_1 = Fr::from_u128_mont(z);
        assert_eq!(b_1, c_1);

        let lhs = Fr::ZERO;
        let rhs = Fr::zero();
        let rhs2 = Fr::from_u128_mont(MontU128::from(0_u128));
        assert_eq!(lhs, rhs);
        assert_eq!(rhs, rhs2);

        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..10000 {
            let a = Fr::random(&mut rng);

            let x = MontU128::from(rng.gen::<u128>());
            let _ = Fr::from_u128_mont(x).mul_u128_mont_form(x);
            let b = Fr::from_u128_mont(x);
            let lhs = a * b;
            let rhs = a.mul_u128_mont_form(x);
            let rhs_two = a * x;
            assert_eq!(lhs, rhs);
            assert_eq!(lhs, rhs_two);

            let x = MontU128::from(rng.gen::<u128>());
            let y = MontU128::from(rng.gen::<u128>());
            let ans1 = (a - y) * x;
            let ans2 = (a - Fr::from_u128_mont(y)).mul_u128_mont_form(x);
            assert_eq!(ans1, ans2);
        }
    }
}
