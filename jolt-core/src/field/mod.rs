use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::Read;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, Valid, Validate};
use ark_std::{One, Zero};

pub trait FieldOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
{
}

/// A zero cost-wrapper around `u128` indicating that the value is already
/// in Montgomery form for the target field
#[cfg_attr(feature = "allocative", derive(Allocative))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MontU128(u128);

impl From<u128> for MontU128 {
    fn from(val: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        let val_masked = val & (u128::MAX >> 3);
        MontU128(val_masked)
    }
}

impl From<MontU128> for u128 {
    fn from(val: MontU128) -> u128 {
        val.0
    }
}
impl From<u64> for MontU128 {
    fn from(val: u64) -> Self {
        MontU128::from(val as u128)
    }
}
// === Serialization impls ===

use ark_serialize::{CanonicalSerialize, Compress, SerializationError};
use std::io::Write;

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

    fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        writer.write_all(&self.0.to_le_bytes())?;
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

    fn deserialize_uncompressed<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let mut buf = [0u8; 16];
        reader.read_exact(&mut buf)?;
        Ok(MontU128(u128::from_le_bytes(buf)))
    }

    fn deserialize_uncompressed_unchecked<R: Read>(
        mut reader: R,
    ) -> Result<Self, SerializationError> {
        let mut buf = [0u8; 16];
        reader.read_exact(&mut buf)?;
        Ok(MontU128(u128::from_le_bytes(buf)))
    }
}

pub trait JoltField:
    'static
    + Sized
    + Zero
    + One
    + Neg<Output = Self>
    + FieldOps<Self, Self>
    + for<'a> FieldOps<&'a Self, Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + core::iter::Sum<Self>
    + for<'a> core::iter::Sum<&'a Self>
    + core::iter::Product<Self>
    + for<'a> core::iter::Product<&'a Self>
    + Eq
    + Copy
    + Sync
    + Send
    + Display
    + Debug
    + Default
    + CanonicalSerialize
    + CanonicalDeserialize
    + Hash
    + MaybeAllocative
{
    /// Number of bytes occupied by a single field element.
    const NUM_BYTES: usize;
    /// An implementation of `JoltField` may use some precomputed lookup tables to speed up the
    /// conversion of small primitive integers (e.g. `u16` values) into field elements. For example,
    /// the arkworks BN254 scalar field requires a conversion into the Montgomery form, which naively
    /// requires a field multiplication, but can instead be looked up.
    type SmallValueLookupTables: Clone + Default + CanonicalSerialize + CanonicalDeserialize;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
    /// Computes the small-value lookup tables.
    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    /// Conversion from primitive integers to field elements in Montgomery form.
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn from_u128(n: u128) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i128(val: i128) -> Self;
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn to_u64(&self) -> Option<u64> {
        unimplemented!("conversion to u64 not implemented");
    }
    fn num_bits(&self) -> u32 {
        unimplemented!("num_bits is not implemented");
    }

    /// Does a field multiplication with a `u64`.
    /// The result will be in Montgomery form (if BN254)
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    // Here n is already in Montgomery form.
    // This MUST be overridden by a concrete field implementation that knows how
    // to multiply by an u128 already in Montgomery form (e.g., using mul_hi_u128).
    // Providing a generic fallback (like converting via from_u128) would be WRONG,
    // because it would interpret n as a canonical integer and re-encode it,
    // effectively multiplying by R and corrupting the result.
    fn mul_u128_mont_form(&self, _n: MontU128) -> Self {
        unimplemented!("mul_u128_mont_form must be implemented by the concrete field type")
    }

    //fn mul_two_u128s(&self, x: MontU128, y: MontU128) -> Self {
    //    unimplemented!("Must be implemented by the conrete field type")
    //}
    //
    fn from_u128_mont(_n: MontU128) -> Self {
        unimplemented!("Must be implemented by the concrete field type")
    }

    fn mul_pow_2(&self, mut pow: usize) -> Self {
        if pow > 255 {
            panic!("pow > 255");
        }
        let mut res = *self;
        while pow >= 64 {
            res = res.mul_u64(1 << 63);
            pow -= 63;
        }
        res.mul_u64(1 << pow)
    }
}

#[cfg(feature = "allocative")]
pub trait MaybeAllocative: Allocative {}
#[cfg(feature = "allocative")]
impl<T: Allocative> MaybeAllocative for T {}
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative {}
#[cfg(not(feature = "allocative"))]
impl<T> MaybeAllocative for T {}

pub trait OptimizedMul<Rhs, Output>: Sized + Mul<Rhs, Output = Output> {
    fn mul_0_optimized(self, other: Rhs) -> Self::Output;
    fn mul_1_optimized(self, other: Rhs) -> Self::Output;
    fn mul_01_optimized(self, other: Rhs) -> Self::Output;
}

impl<T> OptimizedMul<T, T> for T
where
    T: JoltField,
{
    #[inline(always)]
    fn mul_0_optimized(self, other: T) -> T {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: T) -> T {
        if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: T) -> T {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self.mul_1_optimized(other)
        }
    }
}

pub trait OptimizedMulI128<Output>: Sized {
    fn mul_i128_0_optimized(self, other: i128) -> Output;
    fn mul_i128_1_optimized(self, other: i128) -> Output;
    fn mul_i128_01_optimized(self, other: i128) -> Output;
}

/// Implement `OptimizedMul` for `JoltField` with `i128`
impl<T> OptimizedMulI128<T> for T
where
    T: JoltField,
{
    #[inline(always)]
    fn mul_i128_0_optimized(self, other: i128) -> T {
        if other.is_zero() {
            Self::zero()
        } else {
            self.mul_i128(other)
        }
    }

    #[inline(always)]
    fn mul_i128_1_optimized(self, other: i128) -> T {
        if other.is_one() {
            self
        } else {
            self.mul_i128(other)
        }
    }

    #[inline(always)]
    fn mul_i128_01_optimized(self, other: i128) -> T {
        if other.is_zero() {
            Self::zero()
        } else {
            self.mul_i128_1_optimized(other)
        }
    }
}

pub mod ark;
pub mod tracked_ark;
