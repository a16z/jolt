//! Delayed-reduction product accumulators.
//!
//! Each accumulator widens field products into `u128` limbs so a batch of
//! products can be summed without intermediate modular reduction, then
//! reduced once via the owning field's `HasUnreducedOps` impl.

use super::*;

/// Accumulator for `Fp32 × u64` and `Fp32 × Fp32` products.
///
/// Products are split into two 64-bit limbs stored as u128 slots. The second
/// limb is zero for `Fp32 × Fp32` products.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp32ProductAccum(pub [u128; 2]);

impl Fp32ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Reduce accumulated products to a canonical `Fp32<P>`.
    #[inline]
    pub fn reduce<const P: u32>(self) -> Fp32<P> {
        let [s0, s1] = self.0;
        let a = Fp32::<P>::from_canonical_u128_reduced(s0);
        let b = Fp32::<P>::from_canonical_u128_reduced(s1);
        let shift = Fp32::<P>::from_canonical_u32(Fp32::<P>::SHIFT64_MOD_P);
        a + b * shift
    }
}

impl<const P: u32> From<Fp32<P>> for Fp32ProductAccum {
    #[inline]
    fn from(x: Fp32<P>) -> Self {
        Self([x.to_limbs() as u128, 0])
    }
}

impl Add for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
        ])
    }
}
impl AddAssign for Fp32ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_add(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_add(rhs.0[1]);
    }
}
impl Sub for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
        ])
    }
}
impl SubAssign for Fp32ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
    }
}
impl Neg for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([self.0[0].wrapping_neg(), self.0[1].wrapping_neg()])
    }
}

/// Accumulator for `FpExt4<Fp32>` products with delayed reduction.
///
/// Each slot holds the unreduced u128 sum for one of the 4 ring-subfield
/// coefficients. The fused polynomial-multiply + φ(X)-reduction is already
/// applied in the formulas — only the per-coefficient Solinas reduction
/// (`from_canonical_u128_reduced`) is deferred.
///
/// Headroom: each single product contributes at most 7 × P² ≈ 2^65 per
/// slot (slot 0 is the worst case). The u128 capacity of 2^128 allows up
/// to 2^63 accumulations before overflow.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpExt4Fp32ProductAccum(pub [u128; 4]);

impl FpExt4Fp32ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Reduce accumulated unreduced coefficients to a canonical
    /// `FpExt4<Fp32<P>>`.
    #[inline]
    pub fn reduce<const P: u32>(self) -> [Fp32<P>; 4] {
        [
            Fp32::<P>::from_canonical_u128_reduced(self.0[0]),
            Fp32::<P>::from_canonical_u128_reduced(self.0[1]),
            Fp32::<P>::from_canonical_u128_reduced(self.0[2]),
            Fp32::<P>::from_canonical_u128_reduced(self.0[3]),
        ]
    }
}

impl Add for FpExt4Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}
impl AddAssign for FpExt4Fp32ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_add(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_add(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_add(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_add(rhs.0[3]);
    }
}
impl Sub for FpExt4Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}
impl SubAssign for FpExt4Fp32ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_sub(rhs.0[3]);
    }
}
impl Neg for FpExt4Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

/// Accumulator for `Fp64 × u64` products (also used for `Fp64 × Fp64`).
///
/// Each product is ≤ 128 bits, split into two u64 halves stored as u128 slots.
/// Headroom: 2^64 additions per slot before overflow.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp64ProductAccum(pub [u128; 2]);

impl Fp64ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Reduce accumulated products to a canonical `Fp64<P>`.
    #[inline]
    pub fn reduce<const P: u64>(self) -> Fp64<P> {
        let [s0, s1] = self.0;
        // s0 = Σ lo_i, s1 = Σ hi_i; value = s0 + s1 * 2^64
        let a = Fp64::<P>::solinas_reduce(s0);
        let b = Fp64::<P>::solinas_reduce(s1);
        let shift = Fp64::<P>::solinas_reduce(1u128 << 64);
        let b_shifted = Fp64::<P>::solinas_reduce(b.mul_wide_u64(shift.to_limbs()));
        a + b_shifted
    }
}

impl<const P: u64> From<Fp64<P>> for Fp64ProductAccum {
    #[inline]
    fn from(x: Fp64<P>) -> Self {
        Self([x.to_limbs() as u128, 0])
    }
}

impl Add for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
        ])
    }
}
impl AddAssign for Fp64ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_add(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_add(rhs.0[1]);
    }
}
impl Sub for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
        ])
    }
}
impl SubAssign for Fp64ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
    }
}
impl Neg for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([self.0[0].wrapping_neg(), self.0[1].wrapping_neg()])
    }
}

/// Accumulator for `FpExt2<Fp64>` products with delayed reduction.
///
/// Each coefficient is stored as an `Fp64ProductAccum` (lo64/hi64 limb-split).
/// This avoids carry-chain arithmetic -- addition is `wrapping_add` per slot.
/// Reduction delegates to `Fp64ProductAccum::reduce` per coefficient.
///
/// Headroom: each `Fp64ProductAccum` slot holds u64 halves in u128,
/// so 2^64 accumulations before overflow.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpExt2Fp64ProductAccum(pub [u128; 4]);

impl FpExt2Fp64ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Reduce accumulated products to a canonical `[Fp64<P>; 2]`.
    #[inline]
    pub fn reduce<const P: u64>(self) -> [Fp64<P>; 2] {
        [
            Fp64ProductAccum([self.0[0], self.0[1]]).reduce::<P>(),
            Fp64ProductAccum([self.0[2], self.0[3]]).reduce::<P>(),
        ]
    }
}

impl Add for FpExt2Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}
impl AddAssign for FpExt2Fp64ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_add(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_add(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_add(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_add(rhs.0[3]);
    }
}
impl Sub for FpExt2Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}
impl SubAssign for FpExt2Fp64ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_sub(rhs.0[3]);
    }
}
impl Neg for FpExt2Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

/// Accumulator for `Fp128 × u64` products.
///
/// Each `mul_wide_u64` produces 3 u64 limbs; stored as `[u128; 3]`.
/// Headroom: 2^64 additions per slot.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128MulU64Accum(pub [u128; 3]);

impl Fp128MulU64Accum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 3]);

    /// Reduce to canonical `Fp128<P>`.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let [s0, s1, s2] = self.0;
        let c0 = s0 >> 64;
        let r0 = s0 as u64;
        let t1 = s1 + c0;
        let r1 = t1 as u64;
        let c1 = t1 >> 64;
        let t2 = s2 + c1;
        let r2 = t2 as u64;
        let r3 = (t2 >> 64) as u64;
        Fp128::<P>::solinas_reduce(&[r0, r1, r2, r3])
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128MulU64Accum {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let [lo, hi] = x.to_limbs();
        Self([lo as u128, hi as u128, 0])
    }
}

impl Add for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}
impl AddAssign for Fp128MulU64Accum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}
impl Sub for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
        ])
    }
}
impl SubAssign for Fp128MulU64Accum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
    }
}
impl Neg for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
        ])
    }
}

/// Accumulator for `Fp128 × Fp128` products.
///
/// Each `mul_wide` produces 4 u64 limbs; stored as `[u128; 4]`.
/// Headroom: 2^64 additions per slot.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128ProductAccum(pub [u128; 4]);

impl Fp128ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Reduce to canonical `Fp128<P>`.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let [s0, s1, s2, s3] = self.0;
        let c0 = s0 >> 64;
        let r0 = s0 as u64;
        let t1 = s1 + c0;
        let r1 = t1 as u64;
        let c1 = t1 >> 64;
        let t2 = s2 + c1;
        let r2 = t2 as u64;
        let c2 = t2 >> 64;
        let t3 = s3 + c2;
        let r3 = t3 as u64;
        let r4 = (t3 >> 64) as u64;
        Fp128::<P>::solinas_reduce(&[r0, r1, r2, r3, r4])
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128ProductAccum {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let [lo, hi] = x.to_limbs();
        Self([lo as u128, hi as u128, 0, 0])
    }
}

impl Add for Fp128ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}
impl AddAssign for Fp128ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_add(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_add(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_add(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_add(rhs.0[3]);
    }
}
impl Sub for Fp128ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}
impl SubAssign for Fp128ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_sub(rhs.0[3]);
    }
}
impl Neg for Fp128ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

/// Pair accumulator for extension fields.
///
/// Wraps two base-field accumulators `(c0, c1)` component-wise.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumPair<A>(pub A, pub A);

impl<A: AdditiveGroup> Add for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<A: AdditiveGroup> AddAssign for AccumPair<A> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl<A: AdditiveGroup> Sub for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl<A: AdditiveGroup> SubAssign for AccumPair<A> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}
impl<A: AdditiveGroup> Neg for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}
