//! Packed field abstractions and architecture-specific SIMD backends.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub(crate) mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub(crate) mod avx512;
pub(crate) mod ext;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;

pub use ext::{PackedFpExt2, PackedFpExt4, PackedFpExt8};

use crate::ext::{
    fp_ext4_mul_coeffs, fp_ext4_square_coeffs, fp_ext8_mul_schedule, fp_ext8_square_schedule,
    FpExt2Config,
};
use crate::{FieldCore, Fp128, Fp32, Fp64};
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use num_traits::Zero;

/// Packed arithmetic over a scalar field.
pub trait PackedField:
    'static + Copy + Send + Sync + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// Scalar field type.
    type Scalar: FieldCore;

    /// Number of scalar lanes.
    const WIDTH: usize;

    /// Build from a lane generator.
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar;

    /// Extract one lane.
    fn extract(&self, lane: usize) -> Self::Scalar;

    /// Pack a scalar slice into packed values.
    ///
    /// # Panics
    ///
    /// Panics if the length is not divisible by `WIDTH`.
    #[inline]
    fn pack_slice(buf: &[Self::Scalar]) -> Vec<Self> {
        assert!(
            buf.len() % Self::WIDTH == 0,
            "slice length {} must be divisible by WIDTH {}",
            buf.len(),
            Self::WIDTH
        );
        buf.chunks_exact(Self::WIDTH)
            .map(|chunk| Self::from_fn(|i| chunk[i]))
            .collect()
    }

    /// Packed prefix + scalar suffix split.
    #[inline]
    fn pack_slice_with_suffix(buf: &[Self::Scalar]) -> (Vec<Self>, &[Self::Scalar]) {
        let split = buf.len() - (buf.len() % Self::WIDTH);
        let (packed, suffix) = buf.split_at(split);
        (Self::pack_slice(packed), suffix)
    }

    /// Unpack packed values into a flat scalar vector.
    #[inline]
    fn unpack_slice(buf: &[Self]) -> Vec<Self::Scalar> {
        let mut out = Vec::with_capacity(buf.len() * Self::WIDTH);
        for packed in buf {
            for lane in 0..Self::WIDTH {
                out.push(packed.extract(lane));
            }
        }
        out
    }

    /// Broadcast one scalar across all lanes.
    fn broadcast(value: Self::Scalar) -> Self;

    /// Square one packed value.
    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }

    /// Invert one packed value lane-wise.
    #[inline]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: FieldCore,
    {
        let mut inverses = Vec::with_capacity(Self::WIDTH);
        for lane in 0..Self::WIDTH {
            inverses.push(self.extract(lane).inverse()?);
        }
        Some(Self::from_fn(|i| inverses[i]))
    }

    /// Backend hook for multiplying two packed `FpExt2` values in coefficient form.
    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }

    /// Backend hook for multiplying packed ring-subfield quartics.
    #[inline(always)]
    fn fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        fp_ext4_mul_coeffs::<Self>(a, b)
    }

    /// Backend hook for squaring packed ring-subfield quartics.
    #[inline(always)]
    fn fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        fp_ext4_square_coeffs::<Self>(a)
    }

    /// Backend hook for inverting packed ring-subfield quartics.
    #[inline(always)]
    fn fp_ext4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: FieldCore,
    {
        let zero = Self::broadcast(Self::Scalar::zero());
        let [a0, a1, a2, a3] = a;
        let x0 = a0;
        let x1 = a2;
        let y0 = a1 - a3;
        let y1 = a3;

        let x0x1 = x0 * x1;
        let y0y1 = y0 * y1;
        let x1_square = x1 * x1;
        let y1_square = y1 * y1;
        let aa = (x0 * x0 + x1_square + x1_square, x0x1 + x0x1);
        let bb = (y0 * y0 + y1_square + y1_square, y0y1 + y0y1);
        let nr_bb = (bb.0 + bb.0 + bb.1 + bb.1, bb.0 + bb.1 + bb.1);
        let norm = (aa.0 - nr_bb.0, aa.1 - nr_bb.1);
        let inv_norm_base = (norm.0 * norm.0 - (norm.1 * norm.1 + norm.1 * norm.1)).inverse()?;
        let inv_norm = (norm.0 * inv_norm_base, (zero - norm.1) * inv_norm_base);

        let v0 = x0 * inv_norm.0;
        let v1 = x1 * inv_norm.1;
        let constant = (
            v0 + v1 + v1,
            (x0 + x1) * (inv_norm.0 + inv_norm.1) - v0 - v1,
        );
        let neg_y0 = zero - y0;
        let neg_y1 = zero - y1;
        let w0 = neg_y0 * inv_norm.0;
        let w1 = neg_y1 * inv_norm.1;
        let e1_coeff = (
            w0 + w1 + w1,
            (neg_y0 + neg_y1) * (inv_norm.0 + inv_norm.1) - w0 - w1,
        );

        Some([constant.0, e1_coeff.0 + e1_coeff.1, constant.1, e1_coeff.1])
    }

    /// Backend hook for multiplying packed ring-subfield degree-8 elements.
    #[inline(always)]
    fn fp_ext8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        fp_ext8_mul_schedule(
            a,
            b,
            Self::broadcast(Self::Scalar::zero()),
            |x, y| x + y,
            |x, y| x - y,
            |x, y| x * y,
        )
    }

    /// Backend hook for squaring packed ring-subfield degree-8 elements.
    #[inline(always)]
    fn fp_ext8_square(a: [Self; 8]) -> [Self; 8] {
        fp_ext8_square_schedule(
            a,
            Self::broadcast(Self::Scalar::zero()),
            |x, y| x + y,
            |x, y| x - y,
            |x, y| x * y,
        )
    }
}

/// Scalar fallback packed type with one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct NoPacking<T>(pub [T; 1]);

impl<T: FieldCore> Add for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0]])
    }
}

impl<T: FieldCore> Sub for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0]])
    }
}

impl<T: FieldCore> Mul for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([self.0[0] * rhs.0[0]])
    }
}

impl<T: FieldCore> AddAssign for NoPacking<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: FieldCore> SubAssign for NoPacking<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: FieldCore> MulAssign for NoPacking<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: FieldCore + 'static> PackedField for NoPacking<T> {
    const WIDTH: usize = 1;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar,
    {
        Self([f(0)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        debug_assert_eq!(lane, 0);
        self.0[0]
    }
    type Scalar = T;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value])
    }
}

/// Scalar field -> packed field association.
pub trait HasPacking: FieldCore {
    /// Packed representation for this scalar field.
    type Packing: PackedField<Scalar = Self>;
}

/// Selects the packed backend for a Solinas prime at compile time:
/// NEON on aarch64, AVX-512 then AVX2 on x86_64, scalar `NoPacking` otherwise.
macro_rules! select_packing {
    ($alias:ident<$p:ident: $p_ty:ty>, $scalar:ident, $neon:ident, $avx512:ident, $avx2:ident) => {
        /// Selected packed backend for this prime width.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        pub type $alias<const $p: $p_ty> = neon::$neon<$p>;

        /// Selected packed backend for this prime width.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq"
        ))]
        pub type $alias<const $p: $p_ty> = avx512::$avx512<$p>;

        /// Selected packed backend for this prime width.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512dq"))
        ))]
        pub type $alias<const $p: $p_ty> = avx2::$avx2<$p>;

        /// Selected packed backend for this prime width.
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "avx2")
        )))]
        pub type $alias<const $p: $p_ty> = NoPacking<$scalar<$p>>;

        impl<const $p: $p_ty> HasPacking for $scalar<$p> {
            type Packing = $alias<$p>;
        }
    };
}

select_packing!(Fp32Packing<P: u32>, Fp32, PackedFp32Neon, PackedFp32Avx512, PackedFp32Avx2);
select_packing!(Fp64Packing<P: u64>, Fp64, PackedFp64Neon, PackedFp64Avx512, PackedFp64Avx2);
select_packing!(
    Fp128Packing<P: u128>,
    Fp128,
    PackedFp128Neon,
    PackedFp128Avx512,
    PackedFp128Avx2
);

#[cfg(test)]
mod tests;
