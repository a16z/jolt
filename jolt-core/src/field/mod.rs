use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};

pub trait FieldOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
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
{
    /// Number of bytes occupied by a single field element.
    const NUM_BYTES: usize;
    /// An implementation of `JoltField` may use some precomputed lookup tables to speed up the
    /// conversion of small primitive integers (e.g. `u16` values) into field elements. For example,
    /// the arkworks BN254 scalar field requires a conversion into Montgomery form, which naively
    /// requires a field multiplication, but can instead be looked up.
    type SmallValueLookupTables: Clone + Default + CanonicalSerialize + CanonicalDeserialize = ();

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
    /// Computes the small-value lookup tables.
    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    /// Initializes the static lookup tables using the provided values.
    fn initialize_lookup_tables(_init: Self::SmallValueLookupTables) {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
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

    /// The R^2 value used in Montgomery arithmetic for some prime fields.
    /// Returns `None` if the field doesn't use Montgomery arithmetic.
    fn montgomery_r2() -> Option<Self> {
        None
    }

    /// Does an "unchecked" field multiplication with a `u64`.
    /// WARNING: For `x.mul_u64_unchecked(y)` to be equal to `x * F::from_u64(y)`,
    /// which is presumably what you want, we need to correct for the fact that `y` is
    /// not in Montgomery form. This is typically accomplished by multiplying the left
    /// operand by an additional R^2 factor (see `JoltField::montgomery_r2`).
    #[inline(always)]
    fn mul_u64_unchecked(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }
}

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

pub mod ark;
pub mod binius;
pub mod constantine;

// Tests to check that the field operations are correct between `ark` and `constantine`
#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::constantine::ConstantineFr;
    use ark_bn254::Fr as ArkFr;
    use ark_serialize::Compress;
    use ark_std::test_rng;
    use ark_std::UniformRand;
    use rand::Rng;
    use rand_chacha::rand_core::RngCore;

    fn compare_field_ops(a: ArkFr, b: ConstantineFr) {
        // Compare basic properties
        assert_eq!(a.is_zero(), b.is_zero());
        assert_eq!(a.is_one(), b.is_one());

        // Compare arithmetic operations
        println!("Comparing sums of {} and {}", a, b);
        let ark_sum = a + a;
        let const_sum = b + b;
        assert_eq!(ark_sum.to_u64().unwrap(), const_sum.to_limbs_le()[3]);

        // println!("Comparing products of {} and {}", a, b);
        // let ark_prod = a * a;
        // let const_prod = b * b;
        // assert_eq!(ark_prod.0 .0, const_prod.to_limbs_le());

        // println!("Comparing squares of {} and {}", a, b);
        // let ark_square = a.square();
        // let const_square = b.square();
        // assert_eq!(ark_square.0 .0, const_square.to_limbs_le());

        // println!("Comparing inverses of {} and {}", a, b);
        // if !a.is_zero() {
        //     let ark_inv = a.inverse().unwrap();
        //     let const_inv = b.inverse().unwrap();
        //     assert_eq!(ark_inv.0 .0, const_inv.to_limbs_le());
        // }
    }

    #[test]
    fn test_field_ops_consistency() {
        let mut plain_rng = rand::thread_rng();
        // let mut rng = test_rng();

        // Test with small values
        for i in 0..100 {
            let ark_val = ArkFr::from_u64(i);
            let const_val = ConstantineFr::from_u64(i);
            compare_field_ops(ark_val, const_val);
        }

        // // Test with random values
        // for _ in 0..100 {
        //     let val = plain_rng.gen::<u64>();
        //     // Generate a random u64 value, then convert it to a field element
        //     let ark_val = ArkFr::from_u64(val);
        //     let const_val = ConstantineFr::from_u64(val);
        //     compare_field_ops(ark_val, const_val);
        // }
    }

    #[test]
    fn test_field_serialization_consistency() {
        let mut rng = test_rng();

        // Test serialization consistency
        for _ in 0..100 {
            let ark_val = ark_bn254::Fr::rand(&mut rng);
            let const_val = ConstantineFr::from_u64(ark_val.0 .0[0]);

            // Serialize both values
            let mut ark_bytes = Vec::new();
            ark_val
                .serialize_with_mode(&mut ark_bytes, Compress::No)
                .unwrap();

            let mut const_bytes = Vec::new();
            const_val
                .serialize_with_mode(&mut const_bytes, Compress::No)
                .unwrap();

            // Compare serialized bytes
            assert_eq!(ark_bytes, const_bytes);
        }
    }

    // #[test]
    // fn test_field_conversion_consistency() {
    //     let mut rng = thread_rng();

    //     // Test conversion between different integer types
    //     for _ in 0..100 {
    //         let i64_val = rng.gen::<i64>();
    //         let i128_val = rng.gen::<i128>();

    //         // Test i64 conversion
    //         let ark_i64 = ArkFr::from_i64(i64_val);
    //         let const_i64 = ConstantineFr::from_i64(i64_val);
    //         assert_eq!(ark_i64.0, const_i64.to_i64().unwrap());

    //         // Test i128 conversion
    //         let ark_i128 = ArkFr::from_i128(i128_val);
    //         let const_i128 = ConstantineFr::from_i128(i128_val);
    //         assert_eq!(ark_i128.0, const_i128.to_i64().unwrap());
    //     }
    // }
}
