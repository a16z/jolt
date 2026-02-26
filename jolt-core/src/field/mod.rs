use allocative::Allocative;
use ark_ff::BigInt;
use ark_ff::UniformRand;
use num_traits::{One, Zero};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub trait FieldOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
{
}

pub trait ChallengeFieldOps<F>:
    Copy
    + Send
    + Sync
    + Into<F>
    + Add<F, Output = F>
    + for<'a> Add<&'a F, Output = F>
    + Sub<F, Output = F>
    + for<'a> Sub<&'a F, Output = F>
    + Mul<F, Output = F>
    + for<'a> Mul<&'a F, Output = F>
    + Add<Self, Output = F>
    + for<'a> Add<&'a Self, Output = F>
    + Sub<Self, Output = F>
    + for<'a> Sub<&'a Self, Output = F>
    + Mul<Self, Output = F>
    + for<'a> Mul<&'a Self, Output = F>
{
}

pub trait FieldChallengeOps<C>:
    Add<C, Output = Self>
    + for<'a> Add<&'a C, Output = Self>
    + Sub<C, Output = Self>
    + for<'a> Sub<&'a C, Output = Self>
    + Mul<C, Output = Self>
    + for<'a> Mul<&'a C, Output = Self>
{
}

impl<F, C> ChallengeFieldOps<F> for C where
    C: Copy
        + Send
        + Sync
        + Into<F>
        + Add<F, Output = F>
        + for<'a> Add<&'a F, Output = F>
        + Sub<F, Output = F>
        + for<'a> Sub<&'a F, Output = F>
        + Mul<F, Output = F>
        + for<'a> Mul<&'a F, Output = F>
        + Add<C, Output = F>
        + for<'a> Add<&'a C, Output = F>
        + Sub<C, Output = F>
        + for<'a> Sub<&'a C, Output = F>
        + Mul<C, Output = F>
        + for<'a> Mul<&'a C, Output = F>
{
}

impl<F, C> FieldChallengeOps<C> for F where
    F: JoltField
        + Add<C, Output = F>
        + for<'a> Add<&'a C, Output = F>
        + Sub<C, Output = F>
        + for<'a> Sub<&'a C, Output = F>
        + Mul<C, Output = F>
        + for<'a> Mul<&'a C, Output = F>
{
}

/// Common bounds shared by all unreduced integer types.
pub trait UnreducedInteger:
    'static
    + Clone
    + Copy
    + Debug
    + Display
    + Send
    + Sync
    + Default
    + Eq
    + PartialEq
    + Ord
    + Zero
    + From<u128>
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + for<'a> AddAssign<&'a Self>
    + SubAssign
    + for<'a> SubAssign<&'a Self>
{
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
    + FieldChallengeOps<Self::Challenge>
{
    const NUM_BYTES: usize;

    /// Number of 64-bit limbs in the canonical field element representation.
    /// BN254: 4, Fp128: 2.
    const NUM_LIMBS: usize;

    /// Montgomery factor R = 2^(64*NUM_LIMBS) mod p.
    /// For non-Montgomery fields (e.g. Solinas), R = 1.
    const MONTGOMERY_R: Self;

    /// Squared Montgomery factor R^2.
    /// For non-Montgomery fields, R^2 = 1.
    const MONTGOMERY_R_SQUARE: Self;

    // Unreduced types (ladder from narrowest to widest)
    //
    // Each level is both:
    //   (a) the result type of a specific widening operation, AND
    //   (b) an accumulator for values from the level below (1 extra limb = 64 bits headroom).
    //
    // Levels 0-2 are Barrett-reduced; levels 3-4 are Montgomery-reduced.

    /// Level 0: field element reinterpreted as an integer (NUM_LIMBS wide).
    /// Produced by `to_unreduced()`.
    type UnreducedElem: UnreducedInteger;

    /// Level 1: field × u64 product (NUM_LIMBS + 1 wide).
    /// Also serves as the accumulator for UnreducedElem values.
    type UnreducedMulU64: UnreducedInteger
        + Add<Self::UnreducedElem, Output = Self::UnreducedMulU64>
        + AddAssign<Self::UnreducedElem>;

    /// Level 2: field × u128 product (NUM_LIMBS + 2 wide).
    /// Also serves as the accumulator for UnreducedMulU64 values.
    type UnreducedMulU128: UnreducedInteger
        + AddAssign<Self::UnreducedElem>
        + AddAssign<Self::UnreducedMulU64>;

    /// Level 3: accumulator for UnreducedMulU128 values (2*NUM_LIMBS - 1 wide).
    /// Also holds truncated products (e.g. S160/S192 × elem).
    type UnreducedMulU128Accum: UnreducedInteger
        + AddAssign<Self::UnreducedElem>
        + AddAssign<Self::UnreducedMulU64>
        + AddAssign<Self::UnreducedMulU128>;

    /// Level 4: field × field full product (2*NUM_LIMBS wide).
    /// Also serves as accumulator for lower levels.
    type UnreducedProduct: UnreducedInteger
        + AddAssign<Self::UnreducedElem>
        + AddAssign<Self::UnreducedMulU64>
        + AddAssign<Self::UnreducedMulU128>;

    /// Level 5: product accumulator with carry headroom (2*NUM_LIMBS + 1 wide).
    /// Widest type — accumulates field × field products before Montgomery reduction.
    type UnreducedProductAccum: UnreducedInteger
        + AddAssign<Self::UnreducedElem>
        + AddAssign<Self::UnreducedProduct>;

    type SmallValueLookupTables: Clone + Default + CanonicalSerialize + CanonicalDeserialize;
    type Challenge: 'static
        + Sized
        + Copy
        + Clone
        + Send
        + Sync
        + Debug
        + Display
        + Default
        + Eq
        + Hash
        + CanonicalSerialize
        + CanonicalDeserialize
        + Allocative
        + From<u128>
        + Into<Self>
        + ChallengeFieldOps<Self>
        + UniformRand
        + OptimizedMul<Self, Self>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        unimplemented!("Small-value lookup tables are unimplemented")
    }

    fn from_bool(val: bool) -> Self;
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i128(val: i128) -> Self;
    fn from_u128(val: u128) -> Self;
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn to_u64(&self) -> Option<u64> {
        unimplemented!("conversion to u64 not implemented");
    }
    fn num_bits(&self) -> u32 {
        unimplemented!("num_bits is not implemented");
    }

    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
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

    /// Reinterpret field element as its unreduced integer representation.
    fn to_unreduced(&self) -> Self::UnreducedElem;

    /// Widening multiply: field × u64 → UnreducedMulU64.
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64;

    /// Widening multiply: field × u128 → UnreducedMulU128.
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128;

    /// Widening multiply: field × field → UnreducedProduct (tight, no headroom).
    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct;

    /// Widening multiply: field × field → UnreducedProductAccum (with carry headroom).
    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum;

    /// Widening multiply on unreduced elem: UnreducedElem × u64 → UnreducedMulU64.
    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64;

    /// Truncated multiply: UnreducedElem × UnreducedElem → UnreducedProductAccum.
    /// Used when accumulating field × field products in sumcheck inner loops.
    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum;

    /// Truncated multiply: field × M-limb magnitude → UnreducedMulU128Accum.
    /// Used by `WideAccumS` for accumulating field × {S160, S192} products.
    /// For BN254 this calls `mul_trunc`; for smaller fields this may eagerly reduce.
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedMulU128Accum;

    /// Truncated multiply: field × M-limb magnitude → UnreducedProduct.
    /// Used by `FullAccumS` for accumulating field × {S192, S256} products.
    /// For BN254 this calls `mul_trunc`; for smaller fields this may eagerly reduce.
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedProduct;

    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self;
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self;
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self;
    fn reduce_product(x: Self::UnreducedProduct) -> Self;
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self;
}

/// Unified fused-multiply-add trait for accumulators.
/// Perform: acc += left * right.
pub trait FMAdd<Left, Right>: Sized {
    fn fmadd(&mut self, left: &Left, right: &Right);
}

/// Trait for accumulators that finish with Barrett reduction to a field element
pub trait BarrettReduce<F: JoltField> {
    fn barrett_reduce(&self) -> F;
}

/// Trait for accumulators that finish with Montgomery reduction to a field element
pub trait MontgomeryReduce<F: JoltField> {
    fn montgomery_reduce(&self) -> F;
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

impl<F> OptimizedMul<F, F> for F
where
    F: JoltField,
{
    #[inline(always)]
    fn mul_0_optimized(self, other: F) -> F {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: F) -> F {
        if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: F) -> F {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self.mul_1_optimized(other)
        }
    }
}

pub mod ark;
pub mod challenge;
pub mod tracked_ark;
