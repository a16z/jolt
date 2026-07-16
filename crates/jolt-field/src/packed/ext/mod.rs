//! Packed extension field types using transpose-based packing.
//!
//! A `PackedFpExt2` stores `[PF; 2]` where `PF` is the packed base field.
//! Each `PF` lane contains the corresponding coefficient of an `FpExt2` element.
//! This enables WIDTH-fold parallel arithmetic over `FpExt2` using existing SIMD
//! base-field operations.

#![expect(
    clippy::expl_impl_clone_on_copy,
    reason = "manual Clone avoids adding irrelevant generic Clone bounds"
)]

use crate::ext::{FpExt2, FpExt2Config, FpExt4, FpExt4MulBackend, FpExt8, FpExt8MulBackend};
use crate::packed::{HasPacking, PackedField, PackedValue};
use crate::{FieldCore, Invertible};
use core::ops::{Add, Mul, Sub};

/// Packed `FpExt2` elements stored in transpose layout: `[PF; 2]`.
///
/// If `PF` has width `W`, this represents `W` parallel `FpExt2` values.
pub struct PackedFpExt2<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> {
    /// Degree-0 coefficient (packed across SIMD lanes).
    pub c0: PF,
    /// Degree-1 coefficient (packed across SIMD lanes).
    pub c1: PF,
    _marker: std::marker::PhantomData<fn() -> (F, C)>,
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> Clone
    for PackedFpExt2<F, C, PF>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> Copy
    for PackedFpExt2<F, C, PF>
{
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> std::fmt::Debug
    for PackedFpExt2<F, C, PF>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFpExt2").finish_non_exhaustive()
    }
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> PackedFpExt2<F, C, PF> {
    /// Create a `PackedFpExt2` from its two packed coefficients.
    #[inline]
    pub fn new(c0: PF, c1: PF) -> Self {
        Self {
            c0,
            c1,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C, PF> PackedValue for PackedFpExt2<F, C, PF>
where
    F: FieldCore + 'static,
    C: FpExt2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = FpExt2<F, C>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut c0s = Vec::with_capacity(PF::WIDTH);
        let mut c1s = Vec::with_capacity(PF::WIDTH);
        for i in 0..PF::WIDTH {
            let val = f(i);
            c0s.push(val.coeffs[0]);
            c1s.push(val.coeffs[1]);
        }
        Self::new(PF::from_fn(|i| c0s[i]), PF::from_fn(|i| c1s[i]))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        FpExt2::new(self.c0.extract(lane), self.c1.extract(lane))
    }
}

impl<F, C, PF> Add for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}

impl<F, C, PF> Sub for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}

impl<F, C, PF> Mul for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (c0, c1) = PF::fp_ext2_mul::<C>(self.c0, self.c1, rhs.c0, rhs.c1);
        Self::new(c0, c1)
    }
}

impl<F, C, PF> PackedField for PackedFpExt2<F, C, PF>
where
    F: FieldCore + 'static,
    C: FpExt2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = FpExt2<F, C>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(
            PF::broadcast(value.coeffs[0]),
            PF::broadcast(value.coeffs[1]),
        )
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        let norm = self.c0 * self.c0 - C::mul_non_residue(self.c1 * self.c1, PF::broadcast);
        let inv_norm = norm.inverse()?;
        let zero = PF::broadcast(F::zero());
        Some(Self::new(self.c0 * inv_norm, (zero - self.c1) * inv_norm))
    }
}

impl<F, C> HasPacking for FpExt2<F, C>
where
    F: FieldCore + HasPacking + 'static,
    C: FpExt2Config<F> + 'static,
{
    type Packing = PackedFpExt2<F, C, F::Packing>;
}

/// Packed `FpExt4` elements stored as `[PF; 4]`.
pub struct PackedFpExt4<F: FieldCore, PF: PackedField<Scalar = F>> {
    /// Packed coefficients in `[1, e1, e2, e3]` order.
    pub coeffs: [PF; 4],
    _marker: std::marker::PhantomData<fn() -> F>,
}

impl<F, PF> Clone for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PF> Copy for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF> std::fmt::Debug for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFpExt4").finish_non_exhaustive()
    }
}

impl<F, PF> PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    /// Create a packed value from packed ring-subfield coefficients.
    #[inline]
    pub fn new(coeffs: [PF; 4]) -> Self {
        Self {
            coeffs,
            _marker: std::marker::PhantomData,
        }
    }

    /// Square using the packed ring-subfield backend hook.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self::new(PF::fp_ext4_square(self.coeffs))
    }
}

impl<F, PF> PackedValue for PackedFpExt4<F, PF>
where
    F: FieldCore + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = FpExt4<F>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut coeffs: [Vec<F>; 4] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for i in 0..PF::WIDTH {
            let val = f(i);
            for (j, coeff) in val.coeffs.into_iter().enumerate() {
                coeffs[j].push(coeff);
            }
        }
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| coeffs[j][i])))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        FpExt4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, PF> Add for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let [b0, b1, b2, b3] = rhs.coeffs;
        Self::new([a0 + b0, a1 + b1, a2 + b2, a3 + b3])
    }
}

impl<F, PF> Sub for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let [b0, b1, b2, b3] = rhs.coeffs;
        Self::new([a0 - b0, a1 - b1, a2 - b2, a3 - b3])
    }
}

impl<F, PF> Mul for PackedFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::fp_ext4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F, PF> PackedField for PackedFpExt4<F, PF>
where
    F: FieldCore + FpExt4MulBackend + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = FpExt4<F>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::fp_ext4_square(self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        Some(Self::new(PF::fp_ext4_inverse(self.coeffs)?))
    }
}

impl<F> HasPacking for FpExt4<F>
where
    F: FieldCore + HasPacking + FpExt4MulBackend + 'static,
{
    type Packing = PackedFpExt4<F, F::Packing>;
}

/// Packed `FpExt8` elements stored in transpose layout: `[PF; 8]`.
///
/// Each `PF` lane contains one coefficient of a degree-8 Chebyshev-basis element.
pub struct PackedFpExt8<F: FieldCore, PF: PackedField<Scalar = F>> {
    /// Packed coefficients in `[1, e1, ..., e7]` order.
    pub coeffs: [PF; 8],
    _marker: std::marker::PhantomData<fn() -> F>,
}

impl<F, PF> Clone for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PF> Copy for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF> std::fmt::Debug for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFpExt8").finish_non_exhaustive()
    }
}

impl<F, PF> PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    /// Create a packed value from packed ring-subfield coefficients.
    #[inline]
    pub fn new(coeffs: [PF; 8]) -> Self {
        Self {
            coeffs,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, PF> PackedValue for PackedFpExt8<F, PF>
where
    F: FieldCore + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = FpExt8<F>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut coeffs: [Vec<F>; 8] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for i in 0..PF::WIDTH {
            let val = f(i);
            for (j, coeff) in val.coeffs.into_iter().enumerate() {
                coeffs[j].push(coeff);
            }
        }
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| coeffs[j][i])))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        FpExt8::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, PF> Add for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F, PF> Sub for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F, PF> Mul for PackedFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::fp_ext8_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F, PF> PackedField for PackedFpExt8<F, PF>
where
    F: FieldCore + FpExt8MulBackend + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = FpExt8<F>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::fp_ext8_square(self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        // FpExt8 inversion uses Gaussian elimination — delegate lane by lane.
        let mut coeffs: [Vec<F>; 8] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for lane in 0..PF::WIDTH {
            let scalar = self.extract(lane);
            let inv = scalar.inverse()?;
            for (j, c) in inv.coeffs.into_iter().enumerate() {
                coeffs[j].push(c);
            }
        }
        Some(Self::new(std::array::from_fn(|j| {
            PF::from_fn(|i| coeffs[j][i])
        })))
    }
}

impl<F> HasPacking for FpExt8<F>
where
    F: FieldCore + HasPacking + FpExt8MulBackend + 'static,
{
    type Packing = PackedFpExt8<F, F::Packing>;
}

#[cfg(test)]
mod tests;
