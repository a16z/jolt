//! Quadratic, quartic, and octic extension fields.
//!
//! Akita supports one concrete degree-4 and degree-8 extension over each prime
//! base field (`FpExt4`, `FpExt8`): the cyclotomic ring-subfield basis used by
//! trace reduction and production fp32 presets. There is no alternate power- or
//! tower-basis quartic implementation.

mod fp_ext2;
mod fp_ext4;
mod fp_ext8;
pub(crate) mod lift;
mod native_algebra;
#[cfg(test)]
mod tests;

use super::prime::{Fp128, Fp32, Fp64};
use super::unreduced::{
    AccumPair, FoldMatrixFp32, FoldMatrixFp64, FpExt2Fp64ProductAccum, FpExt4Fp32ProductAccum,
    HasOptimizedFold, HasUnreducedOps,
};
use crate::{
    CanonicalField, FieldCore, FromPrimitiveInt, HalvingField, Invertible,
    MulBaseUnreduced, RandomSampling, RingCore,
};
use rand_core::RngCore;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use fp_ext2::{Ext2, FpExt2, FpExt2Config, NegOneNr, TwoNr};
pub use fp_ext4::{FpExt4, FpExt4MulBackend};
pub(crate) use fp_ext8::{fp_ext8_mul_schedule, fp_ext8_square_schedule};
pub use fp_ext8::{FpExt8, FpExt8MulBackend};

