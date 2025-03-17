use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{One, Zero};
use constantine_sys::*;
use rayon::prelude::*;

use super::{FieldOps, JoltField};
use crate::utils::thread::unsafe_allocate_zero_vec;

// Create a wrapper type around bn254_snarks_fr to implement traits for
#[derive(Copy, Clone)]
pub struct ConstantineFr(pub bn254_snarks_fr);

// Implement FieldOps for our wrapper type
impl FieldOps for ConstantineFr {}
impl<'a, 'b> FieldOps<&'b ConstantineFr, ConstantineFr> for &'a ConstantineFr {}
impl<'b> FieldOps<&'b ConstantineFr, ConstantineFr> for ConstantineFr {}

// Static lookup tables for small values
static mut SMALL_VALUE_LOOKUP_TABLES: [Vec<ConstantineFr>; 4] = [vec![], vec![], vec![], vec![]];

// Required for CanonicalDeserialize_with_mode
impl Valid for ConstantineFr {
    fn check(&self) -> Result<(), SerializationError> {
        // Since the bn254 field elements are valid by construction, we just return Ok
        Ok(())
    }
}

impl ark_serialize::CanonicalSerialize for ConstantineFr {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        // Create a buffer to hold the marshaled bytes
        let mut bytes = [0u8; 32]; // 32 bytes for BN254 field element

        // Marshal the field element to big-endian bytes
        let success = unsafe { ctt_bn254_snarks_fr_marshalBE(bytes.as_mut_ptr(), 32, &self.0) };

        if !success {
            return Err(SerializationError::InvalidData);
        }

        // Write the bytes to the provided writer
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        32
    }
}

impl ark_serialize::CanonicalDeserialize for ConstantineFr {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        // Read the appropriate number of bytes
        let mut bytes = [0u8; 32];
        reader.read_exact(&mut bytes)?;

        // Create an uninitialized field element
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };

        // Unmarshal the bytes to a field element
        let success =
            unsafe { ctt_bn254_snarks_fr_unmarshalBE(&mut result, bytes.as_ptr(), bytes.len()) };

        if !success {
            return Err(SerializationError::InvalidData);
        }

        Ok(Self(result))
    }
}

impl bn254_snarks_fr {}

impl ConstantineFr {
    /// Creates a ConstantineFr from a big-endian byte array
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        // Create an uninitialized bn254_snarks_fr
        let mut fr = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };

        // Call the unmarshal function
        let success = unsafe {
            ctt_bn254_snarks_fr_unmarshalBE(
                &mut fr,                       // destination pointer
                bytes.as_ptr() as *const byte, // source pointer (cast to byte*)
                bytes.len(),                   // length of the byte array
            )
        };

        // Check if the operation was successful
        if !success {
            // Handle error - you might want to panic or return a Result instead
            panic!("Failed to unmarshal bytes to bn254_snarks_fr");
        }

        // Return the new ConstantineFr
        Self(fr)
    }

    /// Returns an array [u64; 4] in little-endian form from this ConstantineFr
    pub fn to_limbs_le(&self) -> [u64; 4] {
        // Get the big-endian representation first
        let mut be_bytes = [0u8; 32];

        let success = unsafe {
            ctt_bn254_snarks_fr_marshalBE(be_bytes.as_mut_ptr(), be_bytes.len(), &self.0)
        };

        if !success {
            panic!("Failed to marshal field element to bytes");
        }

        // Convert big-endian bytes to little-endian u64 array
        let mut result = [0u64; 4];

        // Convert each 8-byte chunk to a u64, reversing byte order
        for i in 0..4 {
            let start = i * 8;
            let end = start + 8;
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&be_bytes[start..end]);

            // Reverse the bytes to go from BE to LE and convert to u64
            chunk.reverse();
            result[i] = u64::from_le_bytes(chunk);
        }

        result
    }

    /// Creates a ConstantineFr from a little-endian u64 array
    pub fn from_limbs_le(limbs: [u64; 4]) -> Self {
        // Convert little-endian u64 array to big-endian bytes
        let mut be_bytes = [0u8; 32];

        for i in 0..4 {
            let mut chunk = limbs[i].to_le_bytes();
            // Reverse the bytes to go from LE to BE
            chunk.reverse();
            let start = i * 8;
            let end = start + 8;
            be_bytes[start..end].copy_from_slice(&chunk);
        }

        // Use the from_be_bytes method
        Self::from_be_bytes(&be_bytes)
    }
}

// Implement required traits for our wrapper type
impl Zero for ConstantineFr {
    fn zero() -> Self {
        let result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        Self(result)
    }

    fn is_zero(&self) -> bool {
        unsafe { ctt_bn254_snarks_fr_is_zero(&self.0) != 0 }
    }
}

impl One for ConstantineFr {
    fn one() -> Self {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_set_one(&mut result);
        }
        Self(result)
    }

    fn is_one(&self) -> bool {
        unsafe { ctt_bn254_snarks_fr_is_one(&self.0) != 0 }
    }
}

impl Default for ConstantineFr {
    fn default() -> Self {
        Self::zero()
    }
}

// Basic arithmetic operators
impl Neg for ConstantineFr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_neg(&mut result, &self.0);
        }
        Self(result)
    }
}

impl JoltField for ConstantineFr {
    const NUM_BYTES: usize = 32;

    type SmallValueLookupTables = [Vec<Self>; 4];

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        // Generate random bytes and convert to field element
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Self::from_bytes(&bytes)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        // These four lookup tables correspond to the four 16-bit limbs of a u64
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..4 {
            let bitshift = 16 * i;
            // Create a field element representing 2^(16*i)
            let unit = Self::from_u64(1u64 << bitshift);

            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * Self::from_u64(j as u64))
                .collect();
        }

        lookup_tables
    }

    fn initialize_lookup_tables(init: Self::SmallValueLookupTables) {
        unsafe {
            SMALL_VALUE_LOOKUP_TABLES = init;
        }
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        #[cfg(test)]
        {
            Self::from_u64(n as u64)
        }
        #[cfg(not(test))]
        {
            unsafe { SMALL_VALUE_LOOKUP_TABLES[0][n as usize] }
        }
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        #[cfg(test)]
        {
            Self::from_u64(n as u64)
        }
        #[cfg(not(test))]
        {
            unsafe { SMALL_VALUE_LOOKUP_TABLES[0][n as usize] }
        }
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        #[cfg(test)]
        {
            Self::from_u64(n as u64)
        }
        #[cfg(not(test))]
        {
            const BITMASK: u32 = (1 << 16) - 1;
            unsafe {
                SMALL_VALUE_LOOKUP_TABLES[0][(n & BITMASK) as usize]
                    + SMALL_VALUE_LOOKUP_TABLES[1][((n >> 16) & BITMASK) as usize]
            }
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        // Convert u64 to big-endian bytes
        let bytes = n.to_be_bytes();

        // Create an uninitialized bn254_snarks_fr
        let mut fr = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };

        // Call the unmarshal function
        let success = unsafe {
            ctt_bn254_snarks_fr_unmarshalBE(&mut fr, bytes.as_ptr() as *const byte, bytes.len())
        };

        // Check if the operation was successful
        if !success {
            // For a u64 value, this should never fail, but handle it anyway
            panic!("Failed to convert u64 to bn254_snarks_fr");
        }

        // Wrap the result in ConstantineFr
        Self(fr)
    }

    fn from_i64(val: i64) -> Self {
        if val < 0 {
            let abs_val = (-val) as u64;
            if abs_val <= u16::MAX as u64 {
                -Self::from_u16(abs_val as u16)
            } else if abs_val <= u32::MAX as u64 {
                -Self::from_u32(abs_val as u32)
            } else {
                -Self::from_u64(abs_val)
            }
        } else {
            let val = val as u64;
            if val <= u16::MAX as u64 {
                Self::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                Self::from_u32(val as u32)
            } else {
                Self::from_u64(val)
            }
        }
    }

    fn from_i128(val: i128) -> Self {
        if val < 0 {
            let abs_val = (-val) as u128;
            if abs_val <= u16::MAX as u128 {
                -Self::from_u16(abs_val as u16)
            } else if abs_val <= u32::MAX as u128 {
                -Self::from_u32(abs_val as u32)
            } else if abs_val <= u64::MAX as u128 {
                -Self::from_u64(abs_val as u64)
            } else {
                // If the value is larger than u64::MAX, we need to truncate it
                -Self::from_u64(abs_val as u64)
            }
        } else {
            let val = val as u128;
            if val <= u16::MAX as u128 {
                Self::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                Self::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                Self::from_u64(val as u64)
            } else {
                // If the value is larger than u64::MAX, we need to truncate it
                Self::from_u64(val as u64)
            }
        }
    }

    fn to_u64(&self) -> Option<u64> {
        // Create a buffer to hold the marshalled bytes
        let mut bytes = [0u8; 32];

        // Marshal the field element to big-endian bytes
        let success =
            unsafe { ctt_bn254_snarks_fr_marshalBE(bytes.as_mut_ptr(), bytes.len(), &self.0) };

        if !success {
            return None;
        }

        // Check if only the lowest 8 bytes are non-zero (u64 fits)
        let is_small = bytes[0..24].iter().all(|&b| b == 0);

        if is_small {
            // Convert the last 8 bytes to u64
            let mut u64_bytes = [0u8; 8];
            u64_bytes.copy_from_slice(&bytes[24..32]);
            Some(u64::from_be_bytes(u64_bytes))
        } else {
            None
        }
    }

    fn square(&self) -> Self {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_square(&mut result, &self.0);
        }
        Self(result)
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
            unsafe {
                ctt_bn254_snarks_fr_inv(&mut result, &self.0);
            }
            Some(Self(result))
        }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);

        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            let success = ctt_bn254_snarks_fr_unmarshalBE(&mut result, bytes.as_ptr(), bytes.len());
            if !success {
                // If unmarshalling fails, return zero
                ctt_bn254_snarks_fr_set_zero(&mut result);
            }
        }
        Self(result)
    }

    fn num_bits(&self) -> u32 {
        todo!("...")
    }

    fn montgomery_r2() -> Option<Self> {
        todo!("...")
    }

    #[inline(always)]
    fn mul_u64_unchecked(&self, n: u64) -> Self {
        self.mul(Self::from_u64(n))
    }
}

impl Add for ConstantineFr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_sum(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Add<&ConstantineFr> for ConstantineFr {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_sum(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Add<ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn add(self, rhs: ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_sum(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Add<&ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn add(self, rhs: &ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_sum(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Sub for ConstantineFr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_diff(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Sub<&ConstantineFr> for ConstantineFr {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_diff(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Sub<ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn sub(self, rhs: ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_diff(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Sub<&ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn sub(self, rhs: &ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_diff(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Mul for ConstantineFr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Mul<&ConstantineFr> for ConstantineFr {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &rhs.0);
        }
        Self(result)
    }
}

impl Mul<ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn mul(self, rhs: ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Mul<&ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn mul(self, rhs: &ConstantineFr) -> Self::Output {
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &rhs.0);
        }
        ConstantineFr(result)
    }
}

impl Div for ConstantineFr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Check if divisor is zero
        if rhs.is_zero() {
            panic!("Division by zero");
        }

        // Compute inverse of rhs and multiply
        let mut inv_rhs = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_inv(&mut inv_rhs, &rhs.0);
        }

        // Multiply by inverse
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &inv_rhs);
        }

        Self(result)
    }
}

impl Div<&ConstantineFr> for ConstantineFr {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        // Check if divisor is zero
        if rhs.is_zero() {
            panic!("Division by zero");
        }

        // Compute inverse of rhs and multiply
        let mut inv_rhs = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_inv(&mut inv_rhs, &rhs.0);
        }

        // Multiply by inverse
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &inv_rhs);
        }

        Self(result)
    }
}

impl Div<ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn div(self, rhs: ConstantineFr) -> Self::Output {
        // Check if divisor is zero
        if rhs.is_zero() {
            panic!("Division by zero");
        }

        // Compute inverse of rhs and multiply
        let mut inv_rhs = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_inv(&mut inv_rhs, &rhs.0);
        }

        // Multiply by inverse
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &inv_rhs);
        }

        ConstantineFr(result)
    }
}

impl Div<&ConstantineFr> for &ConstantineFr {
    type Output = ConstantineFr;

    fn div(self, rhs: &ConstantineFr) -> Self::Output {
        // Check if divisor is zero
        if rhs.is_zero() {
            panic!("Division by zero");
        }

        // Compute inverse of rhs and multiply
        let mut inv_rhs = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_inv(&mut inv_rhs, &rhs.0);
        }

        // Multiply by inverse
        let mut result = unsafe { std::mem::zeroed::<bn254_snarks_fr>() };
        unsafe {
            ctt_bn254_snarks_fr_prod(&mut result, &self.0, &inv_rhs);
        }

        ConstantineFr(result)
    }
}

// Assignment operators
impl AddAssign for ConstantineFr {
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            ctt_bn254_snarks_fr_add_in_place(&mut self.0, &rhs.0);
        }
    }
}

impl AddAssign<&ConstantineFr> for ConstantineFr {
    fn add_assign(&mut self, rhs: &Self) {
        unsafe {
            ctt_bn254_snarks_fr_add_in_place(&mut self.0, &rhs.0);
        }
    }
}

impl SubAssign for ConstantineFr {
    fn sub_assign(&mut self, rhs: Self) {
        unsafe {
            ctt_bn254_snarks_fr_sub_in_place(&mut self.0, &rhs.0);
        }
    }
}

impl SubAssign<&ConstantineFr> for ConstantineFr {
    fn sub_assign(&mut self, rhs: &Self) {
        unsafe {
            ctt_bn254_snarks_fr_sub_in_place(&mut self.0, &rhs.0);
        }
    }
}

impl MulAssign for ConstantineFr {
    fn mul_assign(&mut self, rhs: Self) {
        unsafe {
            ctt_bn254_snarks_fr_mul_in_place(&mut self.0, &rhs.0);
        }
    }
}

impl MulAssign<&ConstantineFr> for ConstantineFr {
    fn mul_assign(&mut self, rhs: &Self) {
        unsafe {
            ctt_bn254_snarks_fr_mul_in_place(&mut self.0, &rhs.0);
        }
    }
}

// Display and Debug formatting
impl Display for ConstantineFr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limbs = self.to_limbs_le();
        write!(
            f,
            "ConstantineFr([{}, {}, {}, {}])",
            limbs[0], limbs[1], limbs[2], limbs[3]
        )
    }
}

impl Debug for ConstantineFr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

// Hash implementation
impl Hash for ConstantineFr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut bytes = [0u8; 32];
        unsafe {
            let success = ctt_bn254_snarks_fr_marshalBE(bytes.as_mut_ptr(), bytes.len(), &self.0);
            if success {
                bytes.hash(state);
            } else {
                // If serialization fails, hash a consistent value
                0u64.hash(state);
            }
        }
    }
}

// Equality implementations
impl PartialEq for ConstantineFr {
    fn eq(&self, other: &Self) -> bool {
        unsafe { ctt_bn254_snarks_fr_is_eq(&self.0, &other.0) != 0 }
    }
}

impl Eq for ConstantineFr {}

// Sum and Product implementations
impl std::iter::Sum for ConstantineFr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> std::iter::Sum<&'a ConstantineFr> for ConstantineFr {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl std::iter::Product for ConstantineFr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> std::iter::Product<&'a ConstantineFr> for ConstantineFr {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_zero_one() {
        let zero = ConstantineFr::zero();
        let one = ConstantineFr::one();

        assert!(zero.is_zero());
        assert!(!one.is_zero());
        assert!(one.is_one());
        assert!(!zero.is_one());
    }

    #[test]
    fn test_arithmetic() {
        let a = ConstantineFr::from_u64(10);
        let b = ConstantineFr::from_u64(5);

        // Addition
        let c = a + b;
        assert_eq!(c, ConstantineFr::from_u64(15));

        // Subtraction
        let d = a - b;
        assert_eq!(d, ConstantineFr::from_u64(5));

        // Multiplication
        let e = a * b;
        assert_eq!(e, ConstantineFr::from_u64(50));

        // Division
        let f = a / b;
        assert_eq!(f, ConstantineFr::from_u64(2));
    }

    // #[test]
    // fn test_random() {
    //     let mut rng = thread_rng();
    //     let a = ConstantineFr::random(&mut rng);
    //     let b = ConstantineFr::random(&mut rng);

    //     // Two random elements should almost certainly be different
    //     assert_ne!(a, b);
    // }

    #[test]
    fn test_serialization() {
        let a = ConstantineFr::from_u64(12345);

        // Serialize to bytes
        let mut serialize_with_moded = Vec::new();
        a.serialize_with_mode(&mut serialize_with_moded, Compress::No)
            .unwrap();

        // Deserialize_with_mode back to field element
        let b = ConstantineFr::deserialize_with_mode(
            &mut &serialize_with_moded[..],
            Compress::No,
            Validate::No,
        )
        .unwrap();

        // Check equality
        assert_eq!(a, b);
    }

    #[test]
    fn test_inverse() {
        let a = ConstantineFr::from_u64(5);
        let a_inv = a.inverse().unwrap();

        // a * a^(-1) should equal 1
        let product = a * a_inv;
        assert_eq!(product, ConstantineFr::one());
    }
}
