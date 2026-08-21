//! Packed extension towers in transpose layout: coefficient `j` of every
//! lane lives in packed base vector `j`, so `WIDTH` extension values
//! multiply in parallel through the [`Packed`] kernel hooks — which consume
//! the shared coefficient schedules (`crate::schedules`) unless a SIMD
//! engine overrides them with fused kernels.
//!
//! Inversion is lane-wise scalar throughout (the [`Packed::inverse`]
//! default): the base-field Fermat inversion is lane-serial in any
//! formulation and dominates the cost.

use crate::solinas::{FpExt2, FpExt4, FpExt8};
use crate::{Ext2Config, Field, Packed, PseudoMersenne, WithPacking};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

/// Packed [`FpExt2`]: `WIDTH` quadratic-extension lanes as two packed
/// coefficient vectors.
pub struct PackedFpExt2<PF: Packed, C: Ext2Config<PF::Scalar>> {
    /// Packed degree-0 coefficients.
    pub c0: PF,
    /// Packed degree-1 coefficients.
    pub c1: PF,
    _cfg: PhantomData<fn() -> C>,
}

impl<PF: Packed, C: Ext2Config<PF::Scalar>> PackedFpExt2<PF, C> {
    /// Constructs from packed coefficient vectors.
    #[inline]
    pub fn new(c0: PF, c1: PF) -> Self {
        Self {
            c0,
            c1,
            _cfg: PhantomData,
        }
    }
}

// Manual std impls: derives would demand `C: Clone` etc. on the config ZST.
impl<PF: Packed, C: Ext2Config<PF::Scalar>> Clone for PackedFpExt2<PF, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<PF: Packed, C: Ext2Config<PF::Scalar>> Copy for PackedFpExt2<PF, C> {}

impl<PF: Packed, C: Ext2Config<PF::Scalar>> Add for PackedFpExt2<PF, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}
impl<PF: Packed, C: Ext2Config<PF::Scalar>> Sub for PackedFpExt2<PF, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}
impl<PF: Packed, C: Ext2Config<PF::Scalar>> Mul for PackedFpExt2<PF, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (c0, c1) = PF::ext2_mul::<C>(self.c0, self.c1, rhs.c0, rhs.c1);
        Self::new(c0, c1)
    }
}

impl<PF: Packed, C: Ext2Config<PF::Scalar> + 'static> Packed for PackedFpExt2<PF, C> {
    type Scalar = FpExt2<PF::Scalar, C>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
        let vals: Vec<Self::Scalar> = (0..PF::WIDTH).map(f).collect();
        Self::new(
            PF::from_fn(|i| vals[i].coeffs[0]),
            PF::from_fn(|i| vals[i].coeffs[1]),
        )
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        FpExt2::new(self.c0.extract(lane), self.c1.extract(lane))
    }

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(
            PF::broadcast(value.coeffs[0]),
            PF::broadcast(value.coeffs[1]),
        )
    }
}

impl<F, C> WithPacking for FpExt2<F, C>
where
    F: Field + WithPacking,
    C: Ext2Config<F> + 'static,
{
    type Packing = PackedFpExt2<F::Packing, C>;
}

/// Packed [`FpExt4`]: `WIDTH` quartic lanes as four packed coefficient
/// vectors in the `[1, e1, e2, e3]` basis.
#[derive(Clone, Copy)]
pub struct PackedFpExt4<PF: Packed> {
    /// Packed coefficients in basis order.
    pub coeffs: [PF; 4],
}

impl<PF: Packed> PackedFpExt4<PF> {
    /// Constructs from packed coefficient vectors.
    #[inline]
    pub fn new(coeffs: [PF; 4]) -> Self {
        Self { coeffs }
    }
}

impl<PF: Packed> Add for PackedFpExt4<PF> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}
impl<PF: Packed> Sub for PackedFpExt4<PF> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}
impl<PF: Packed> Mul for PackedFpExt4<PF> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::ext4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<PF: Packed> Packed for PackedFpExt4<PF>
where
    PF::Scalar: PseudoMersenne,
{
    type Scalar = FpExt4<PF::Scalar>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
        let vals: Vec<Self::Scalar> = (0..PF::WIDTH).map(f).collect();
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| vals[i].coeffs[j])))
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        FpExt4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(value.coeffs.map(PF::broadcast))
    }

    /// Squaring via the dedicated kernel hook (fewer base multiplies).
    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::ext4_square(self.coeffs))
    }
}

impl<F: PseudoMersenne + WithPacking> WithPacking for FpExt4<F> {
    type Packing = PackedFpExt4<F::Packing>;
}

/// Packed [`FpExt8`]: `WIDTH` octic lanes as eight packed coefficient
/// vectors in the `[1, e1, ..., e7]` basis.
#[derive(Clone, Copy)]
pub struct PackedFpExt8<PF: Packed> {
    /// Packed coefficients in basis order.
    pub coeffs: [PF; 8],
}

impl<PF: Packed> PackedFpExt8<PF> {
    /// Constructs from packed coefficient vectors.
    #[inline]
    pub fn new(coeffs: [PF; 8]) -> Self {
        Self { coeffs }
    }
}

impl<PF: Packed> Add for PackedFpExt8<PF> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}
impl<PF: Packed> Sub for PackedFpExt8<PF> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}
impl<PF: Packed> Mul for PackedFpExt8<PF> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::ext8_mul(self.coeffs, rhs.coeffs))
    }
}

impl<PF: Packed> Packed for PackedFpExt8<PF>
where
    PF::Scalar: PseudoMersenne,
{
    type Scalar = FpExt8<PF::Scalar>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
        let vals: Vec<Self::Scalar> = (0..PF::WIDTH).map(f).collect();
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| vals[i].coeffs[j])))
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        FpExt8::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(value.coeffs.map(PF::broadcast))
    }

    /// Squaring via the dedicated kernel hook.
    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::ext8_square(self.coeffs))
    }
}

impl<F: PseudoMersenne + WithPacking> WithPacking for FpExt8<F> {
    type Packing = PackedFpExt8<F::Packing>;
}
