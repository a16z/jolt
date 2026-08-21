//! Packed-lane contracts: [`Packed`] (`WIDTH` parallel scalar lanes),
//! [`WithPacking`] (scalar → packed association), and the [`NoPacking`]
//! one-lane fallback used on targets without a SIMD backend.
//!
//! The extension kernel hooks default to the shared coefficient schedules
//! (`crate::schedules`), so every backend computes the same field values;
//! SIMD backends override the degree-4 hooks with fused deferred-reduction
//! dot products.

use crate::{Ext2Config, Field};
use num_traits::Zero;
use std::ops::{Add, Mul, Sub};

/// `WIDTH` scalar field lanes with element-wise arithmetic.
///
/// # Invariants
///
/// - Every lane of every value is a canonical scalar; `extract` after any
///   operation equals the same operation on the extracted inputs.
/// - `from_fn`/`extract`/`broadcast` are mutually consistent:
///   `Self::from_fn(f).extract(i) == f(i)` for `i < WIDTH`.
pub trait Packed:
    'static + Copy + Send + Sync + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// Scalar field type of one lane.
    type Scalar: Field;

    /// Number of scalar lanes.
    const WIDTH: usize;

    /// Builds a packed value from a lane generator.
    fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self;

    /// Extracts one lane.
    fn extract(&self, lane: usize) -> Self::Scalar;

    /// Broadcasts one scalar across all lanes.
    fn broadcast(value: Self::Scalar) -> Self;

    /// Packs a scalar slice into packed values.
    ///
    /// # Panics
    ///
    /// Panics if the length is not divisible by [`WIDTH`](Self::WIDTH).
    #[inline]
    fn pack_slice(buf: &[Self::Scalar]) -> Vec<Self> {
        assert_eq!(buf.len() % Self::WIDTH, 0, "length not divisible by width");
        buf.chunks_exact(Self::WIDTH)
            .map(|chunk| Self::from_fn(|i| chunk[i]))
            .collect()
    }

    /// Splits into a packed prefix and a scalar suffix shorter than `WIDTH`.
    #[inline]
    fn pack_slice_with_suffix(buf: &[Self::Scalar]) -> (Vec<Self>, &[Self::Scalar]) {
        let (packed, suffix) = buf.split_at(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_slice(packed), suffix)
    }

    /// Unpacks packed values into a flat scalar vector.
    #[inline]
    fn unpack_slice(buf: &[Self]) -> Vec<Self::Scalar> {
        buf.iter()
            .flat_map(|p| (0..Self::WIDTH).map(move |i| p.extract(i)))
            .collect()
    }

    /// Squares one packed value.
    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }

    /// Lane-wise inversion; `None` if any lane is zero.
    #[inline]
    fn inverse(self) -> Option<Self> {
        let lanes: Option<Vec<_>> = (0..Self::WIDTH)
            .map(|i| self.extract(i).inverse())
            .collect();
        lanes.map(|lanes| Self::from_fn(|i| lanes[i]))
    }

    /// Kernel hook: packed quadratic-extension multiply in coefficient form
    /// (Karatsuba; the non-residue fast paths come from [`Ext2Config`]).
    #[inline(always)]
    fn ext2_mul<C: Ext2Config<Self::Scalar>>(
        a0: Self,
        a1: Self,
        b0: Self,
        b1: Self,
    ) -> (Self, Self) {
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }

    /// Kernel hook: packed degree-4 extension multiply in the `[1, e1, e2,
    /// e3]` basis.
    #[inline(always)]
    fn ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        crate::schedules::ext4_mul_coeffs(a, b)
    }

    /// Kernel hook: packed degree-4 extension squaring.
    #[inline(always)]
    fn ext4_square(a: [Self; 4]) -> [Self; 4] {
        crate::schedules::ext4_square_coeffs(a)
    }

    /// Kernel hook: packed degree-8 extension multiply in the `[1, e1, ...,
    /// e7]` basis.
    #[inline(always)]
    fn ext8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        let zero = Self::broadcast(Self::Scalar::zero());
        crate::schedules::ext8_mul_schedule(a, b, zero, |x, y| x + y, |x, y| x - y, |x, y| x * y)
    }

    /// Kernel hook: packed degree-8 extension squaring.
    #[inline(always)]
    fn ext8_square(a: [Self; 8]) -> [Self; 8] {
        let zero = Self::broadcast(Self::Scalar::zero());
        crate::schedules::ext8_square_schedule(a, zero, |x, y| x + y, |x, y| x - y, |x, y| x * y)
    }
}

/// Associates a packed representation with a scalar field.
pub trait WithPacking: Field {
    /// Packed representation (the target's widest available backend).
    type Packing: Packed<Scalar = Self>;
}

/// One-lane fallback with no SIMD path: plain scalar arithmetic per "lane".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct NoPacking<T>(pub [T; 1]);

crate::impl_ring_ops!(impl[T: Field] NoPacking<T> {
    add(a, b): NoPacking([a.0[0] + b.0[0]]),
    sub(a, b): NoPacking([a.0[0] - b.0[0]]),
    mul(a, b): NoPacking([a.0[0] * b.0[0]]),
    neg(a): NoPacking([-a.0[0]]),
    zero: NoPacking([T::zero()]),
    one: NoPacking([T::one()]),
});

impl<T: Field + 'static> Packed for NoPacking<T> {
    type Scalar = T;
    const WIDTH: usize = 1;

    #[inline]
    fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        Self([f(0)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> T {
        debug_assert_eq!(lane, 0);
        self.0[0]
    }

    #[inline]
    fn broadcast(value: T) -> Self {
        Self([value])
    }
}
