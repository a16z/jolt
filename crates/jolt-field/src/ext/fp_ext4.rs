//! Akita's only degree-4 extension field (cyclotomic ring-subfield basis).
//!
//! Coefficients are stored in the `[1, e1, e2, e3]` basis used by trace reduction
//! and production fp32 presets.

#![expect(
    clippy::expl_impl_clone_on_copy,
    reason = "manual Clone avoids adding irrelevant generic Clone bounds"
)]

use super::*;

/// Multiply ring-subfield quartic coefficient arrays in `[1, e1, e2, e3]` basis.
#[inline]
pub(crate) fn fp_ext4_mul_coeffs<F, A>(a: [A; 4], b: [A; 4]) -> [A; 4]
where
    F: FieldCore,
    A: ExtensionCoeff<F>,
{
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    let tail0 = a1 * b1 + a2 * b2 + a3 * b3;
    [
        a0 * b0 + tail0 + tail0,
        a0 * b1 + a1 * b0 + a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2,
        a0 * b2 + a2 * b0 + a1 * b1 + a1 * b3 + a3 * b1 - a3 * b3,
        a0 * b3 + a3 * b0 + a1 * b2 + a2 * b1 - a2 * b3 - a3 * b2,
    ]
}

/// Square ring-subfield quartic coefficient arrays in `[1, e1, e2, e3]` basis.
#[inline]
pub(crate) fn fp_ext4_square_coeffs<F, A>(a: [A; 4]) -> [A; 4]
where
    F: FieldCore,
    A: ExtensionCoeff<F>,
{
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

    let v0 = x0 * y0;
    let v1 = x1 * y1;
    let ab = (v0 + v1 + v1, (x0 + x1) * (y0 + y1) - v0 - v1);
    let constant = (bb.0 + bb.0 + bb.1 + bb.1, bb.0 + bb.1 + bb.1);
    let coeff_e1 = (ab.0 + ab.0, ab.1 + ab.1);

    [
        aa.0 + constant.0,
        coeff_e1.0 + coeff_e1.1,
        aa.1 + constant.1,
        coeff_e1.1,
    ]
}

#[inline(always)]
fn fp32_product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
    ((a.to_limbs() as u64) * (b.to_limbs() as u64)) as u128
}

#[inline(always)]
fn fp32_square_product<const P: u32>(a: Fp32<P>) -> u128 {
    fp32_product(a, a)
}

#[inline(always)]
fn fp32_reduce_accum<const P: u32>(x: u128) -> Fp32<P> {
    Fp32::<P>::from_canonical_u128_reduced(x)
}

#[inline(always)]
fn fp32_modulus_square<const P: u32>() -> u128 {
    (P as u128) * (P as u128)
}

#[inline(always)]
fn fp32_modulus_bits<const P: u32>() -> u32 {
    32 - P.leading_zeros()
}

/// Backend hook for scalar ring-subfield quartic multiplication.
///
/// The default is the generic coefficient formula. Concrete base fields can
/// override this when their representation supports fusing product sums before
/// reduction.
pub trait FpExt4MulBackend: FieldCore {
    /// Multiply two ring-subfield coefficient arrays in `[1, e1, e2, e3]` basis.
    #[inline(always)]
    fn fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        fp_ext4_mul_coeffs::<Self, Self>(a, b)
    }

    /// Square one ring-subfield coefficient array in `[1, e1, e2, e3]` basis.
    #[inline(always)]
    fn fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        fp_ext4_square_coeffs::<Self, Self>(a)
    }
}

impl<const P: u64> FpExt4MulBackend for Fp64<P> {}
impl<const P: u128> FpExt4MulBackend for Fp128<P> {}

impl<const P: u32> FpExt4MulBackend for Fp32<P> {
    #[inline(always)]
    fn fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        let modulus_square = fp32_modulus_square::<P>();
        [
            fp32_reduce_accum(
                fp32_product(a0, b0)
                    + 2 * (fp32_product(a1, b1) + fp32_product(a2, b2) + fp32_product(a3, b3)),
            ),
            fp32_reduce_accum(
                fp32_product(a0, b1)
                    + fp32_product(a1, b0)
                    + fp32_product(a1, b2)
                    + fp32_product(a2, b1)
                    + fp32_product(a2, b3)
                    + fp32_product(a3, b2),
            ),
            fp32_reduce_accum(
                fp32_product(a0, b2)
                    + fp32_product(a2, b0)
                    + fp32_product(a1, b1)
                    + fp32_product(a1, b3)
                    + fp32_product(a3, b1)
                    + modulus_square
                    - fp32_product(a3, b3),
            ),
            fp32_reduce_accum(
                fp32_product(a0, b3)
                    + fp32_product(a3, b0)
                    + fp32_product(a1, b2)
                    + fp32_product(a2, b1)
                    + 2 * modulus_square
                    - fp32_product(a2, b3)
                    - fp32_product(a3, b2),
            ),
        ]
    }

    #[inline(always)]
    fn fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        if fp32_modulus_bits::<P>() != 32 {
            return Self::fp_ext4_mul(a, a);
        }

        let [a0, a1, a2, a3] = a;
        let modulus_square = fp32_modulus_square::<P>();
        let a0_square = fp32_square_product(a0);
        let a1_square = fp32_square_product(a1);
        let a2_square = fp32_square_product(a2);
        let a3_square = fp32_square_product(a3);
        let a0a1 = fp32_product(a0, a1);
        let a0a2 = fp32_product(a0, a2);
        let a0a3 = fp32_product(a0, a3);
        let a1a2 = fp32_product(a1, a2);
        let a1a3 = fp32_product(a1, a3);
        let a2a3 = fp32_product(a2, a3);

        [
            fp32_reduce_accum(a0_square + 2 * (a1_square + a2_square + a3_square)),
            fp32_reduce_accum(2 * (a0a1 + a1a2 + a2a3)),
            fp32_reduce_accum(2 * a0a2 + a1_square + 2 * a1a3 + modulus_square - a3_square),
            fp32_reduce_accum(2 * (a0a3 + a1a2 + modulus_square - a2a3)),
        ]
    }
}

/// Widening `FpExt4<Fp32<P>>` multiplication that skips per-coefficient
/// Solinas reduction, returning `FpExt4Fp32ProductAccum` instead.
///
/// The φ(X) ring reduction is already fused into the formulas — only the
/// base-field modular reduction is deferred.
#[inline(always)]
pub(crate) fn fp_ext4_mul_to_accum_fp32<const P: u32>(
    a: [Fp32<P>; 4],
    b: [Fp32<P>; 4],
) -> FpExt4Fp32ProductAccum {
    #[inline(always)]
    fn product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
        (a.to_limbs() as u128) * (b.to_limbs() as u128)
    }

    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    let modulus_square = (P as u128) * (P as u128);
    FpExt4Fp32ProductAccum([
        product(a0, b0) + 2 * (product(a1, b1) + product(a2, b2) + product(a3, b3)),
        product(a0, b1)
            + product(a1, b0)
            + product(a1, b2)
            + product(a2, b1)
            + product(a2, b3)
            + product(a3, b2),
        product(a0, b2)
            + product(a2, b0)
            + product(a1, b1)
            + product(a1, b3)
            + product(a3, b1)
            + modulus_square
            - product(a3, b3),
        product(a0, b3) + product(a3, b0) + product(a1, b2) + product(a2, b1) + 2 * modulus_square
            - product(a2, b3)
            - product(a3, b2),
    ])
}

/// Quartic fixed-subfield element in the Akita cyclotomic basis.
///
/// Coordinates are `[c0, c1, c2, c3]` in basis `[1, e1, e2, e3]`, where
/// `e_j = zeta^(jm) + zeta^(-jm)` for `m = D / 8` inside a compatible
/// cyclotomic ring. The scalar arithmetic is independent of the concrete ring
/// dimension `D`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: FieldCore + allocative::Allocative")
)]
#[repr(transparent)]
pub struct FpExt4<F: FieldCore> {
    /// Coefficients in basis `[1, e1, e2, e3]`.
    pub coeffs: [F; 4],
}

impl<F: FieldCore> FpExt4<F> {
    /// Construct from ring-subfield basis coefficients `[c0, c1, c2, c3]`.
    #[inline]
    pub fn new(coeffs: [F; 4]) -> Self {
        Self { coeffs }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new([F::zero(); 4])
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new([F::one(), F::zero(), F::zero(), F::zero()])
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }

    /// Construct from a `u64` embedded in the base field.
    #[inline]
    pub fn from_u64(val: u64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new([F::from_u64(val), F::zero(), F::zero(), F::zero()])
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new([F::from_i64(val), F::zero(), F::zero(), F::zero()])
    }

    #[inline(always)]
    fn fp_ext2_mul_by_e2_nr(lhs: (F, F), rhs: (F, F)) -> (F, F) {
        let (a0, a1) = lhs;
        let (b0, b1) = rhs;
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let c1 = (a0 + a1) * (b0 + b1) - v0 - v1;
        let c0 = v0 + v1 + v1;
        (c0, c1)
    }

    #[inline(always)]
    fn fp_ext2_square_by_e2_nr(x: (F, F)) -> (F, F) {
        let (a0, a1) = x;
        let a0a1 = a0 * a1;
        (a0.square() + a1.square() + a1.square(), a0a1 + a0a1)
    }

    #[inline(always)]
    fn fp_ext2_mul_by_e1_nr(x: (F, F)) -> (F, F) {
        let (x0, x1) = x;
        (x0 + x0 + x1 + x1, x0 + x1 + x1)
    }

    #[inline(always)]
    fn fp_ext2_inverse_by_e2_nr(x: (F, F)) -> Option<(F, F)> {
        let (x0, x1) = x;
        let inv_norm = (x0.square() - (x1.square() + x1.square())).inverse()?;
        Some((x0 * inv_norm, -x1 * inv_norm))
    }
}

impl<F: FieldCore + std::fmt::Debug> std::fmt::Debug for FpExt4<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FpExt4")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore> Clone for FpExt4<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore> Copy for FpExt4<F> {}

impl<F: FieldCore> Default for FpExt4<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldCore> PartialEq for FpExt4<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FieldCore> Eq for FpExt4<F> {}

impl<F: FieldCore> Add for FpExt4<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
            self.coeffs[2] + rhs.coeffs[2],
            self.coeffs[3] + rhs.coeffs[3],
        ])
    }
}

impl<F: FieldCore> Sub for FpExt4<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
            self.coeffs[2] - rhs.coeffs[2],
            self.coeffs[3] - rhs.coeffs[3],
        ])
    }
}

impl<F: FieldCore> Neg for FpExt4<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new([
            -self.coeffs[0],
            -self.coeffs[1],
            -self.coeffs[2],
            -self.coeffs[3],
        ])
    }
}

impl<F: FieldCore> AddAssign for FpExt4<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] + rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] + rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] + rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] + rhs.coeffs[3];
    }
}

impl<F: FieldCore> SubAssign for FpExt4<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] - rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] - rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] - rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] - rhs.coeffs[3];
    }
}

impl<F: FpExt4MulBackend> Mul for FpExt4<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(F::fp_ext4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F: FpExt4MulBackend> MulAssign for FpExt4<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore> Add<&'a Self> for FpExt4<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, F: FieldCore> Sub<&'a Self> for FpExt4<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, F: FpExt4MulBackend> Mul<&'a Self> for FpExt4<F> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + FpExt4MulBackend> RingCore for FpExt4<F> {
    #[inline(always)]
    fn square(&self) -> Self {
        Self::new(F::fp_ext4_square(self.coeffs))
    }
}

impl<F: FieldCore + FpExt4MulBackend> Invertible for FpExt4<F> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let [a0, a1, a2, a3] = self.coeffs;
        let a = (a0, a2);
        let b = (a1 - a3, a3);

        let aa = Self::fp_ext2_square_by_e2_nr(a);
        let bb = Self::fp_ext2_square_by_e2_nr(b);
        let norm = {
            let nr_bb = Self::fp_ext2_mul_by_e1_nr(bb);
            (aa.0 - nr_bb.0, aa.1 - nr_bb.1)
        };
        let inv_norm = Self::fp_ext2_inverse_by_e2_nr(norm)?;
        let constant = Self::fp_ext2_mul_by_e2_nr(a, inv_norm);
        let e1_coeff = Self::fp_ext2_mul_by_e2_nr((-b.0, -b.1), inv_norm);

        Some(Self::new([
            constant.0,
            e1_coeff.0 + e1_coeff.1,
            constant.1,
            e1_coeff.1,
        ]))
    }
}

impl<F: HalvingField + FpExt4MulBackend> HalvingField for FpExt4<F> {
    #[inline]
    fn half(self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i].half()))
    }
}

impl<F: FieldCore + RandomSampling> RandomSampling for FpExt4<F> {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new([
            F::random(rng),
            F::random(rng),
            F::random(rng),
            F::random(rng),
        ])
    }
}

impl<F: FieldCore + FromPrimitiveInt> FromPrimitiveInt for FpExt4<F> {
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new([F::from_u128(val), F::zero(), F::zero(), F::zero()])
    }

    fn from_i128(val: i128) -> Self {
        Self::new([F::from_i128(val), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore + BalancedDigitLookup> BalancedDigitLookup for FpExt4<F> {}

impl<const P: u32> HasUnreducedOps for FpExt4<Fp32<P>> {
    type MulU64Accum = Self;
    type ProductAccum = FpExt4Fp32ProductAccum;

    // `fp_ext4_mul_to_accum_fp32` widens each Fp32 limb product
    // (< 7·p² ≈ 2^65) into a u128 slot with no `mod 2^128` wrap, so summing a
    // batch and reducing once matches per-limb reduce-then-add exactly. Covered
    // by `fp_ext4_fp32_accum_summation`.
    const DELAYED_PRODUCT_SUM_IS_EXACT: bool = true;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum {
        let small = Fp32::<P>::from_u64(small);
        Self::new(self.coeffs.map(|coeff| coeff * small))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::ProductAccum {
        fp_ext4_mul_to_accum_fp32(self.coeffs, other.coeffs)
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self {
        accum
    }

    #[inline]
    fn reduce_product_accum(accum: Self::ProductAccum) -> Self {
        Self::new(accum.reduce::<P>())
    }
}

impl<const P: u32> MulBaseUnreduced<Fp32<P>> for FpExt4<Fp32<P>> {
    #[inline]
    fn mul_base_to_product_accum(self, x: Fp32<P>) -> Self::ProductAccum {
        // E × F has no cross terms: scale each base coordinate into its own
        // u128 slot. Each product is `< p² < 2^62`, so a summed batch reduces
        // exactly (see `DELAYED_PRODUCT_SUM_IS_EXACT`).
        let x = x.to_limbs() as u128;
        let [a0, a1, a2, a3] = self.coeffs;
        FpExt4Fp32ProductAccum([
            (a0.to_limbs() as u128) * x,
            (a1.to_limbs() as u128) * x,
            (a2.to_limbs() as u128) * x,
            (a3.to_limbs() as u128) * x,
        ])
    }
}

impl<const P: u32> HasOptimizedFold for FpExt4<Fp32<P>> {
    type FoldCtx = FoldMatrixFp32;

    #[inline]
    fn precompute_fold(r: Self) -> FoldMatrixFp32 {
        let [r0, r1, r2, r3] = r.coeffs;
        let two = Fp32::<P>::from_u64(2);
        FoldMatrixFp32([
            [
                r0.to_limbs(),
                (two * r1).to_limbs(),
                (two * r2).to_limbs(),
                (two * r3).to_limbs(),
            ],
            [
                r1.to_limbs(),
                (r0 + r2).to_limbs(),
                (r1 + r3).to_limbs(),
                r2.to_limbs(),
            ],
            [
                r2.to_limbs(),
                (r1 + r3).to_limbs(),
                r0.to_limbs(),
                (r1 - r3).to_limbs(),
            ],
            [
                r3.to_limbs(),
                r2.to_limbs(),
                (r1 - r3).to_limbs(),
                (r0 - r2).to_limbs(),
            ],
        ])
    }

    #[inline]
    fn fold_one(ctx: &FoldMatrixFp32, even: Self, odd: Self) -> Self {
        let m = &ctx.0;
        let d: [u32; 4] = std::array::from_fn(|j| (odd.coeffs[j] - even.coeffs[j]).to_limbs());
        let folded: [Fp32<P>; 4] = if P < (1u32 << 31) {
            // P < 2^31: each product < 2^62, sum of 4 < 2^64, fits in u64.
            std::array::from_fn(|row| {
                let acc: u64 = (m[row][0] as u64) * (d[0] as u64)
                    + (m[row][1] as u64) * (d[1] as u64)
                    + (m[row][2] as u64) * (d[2] as u64)
                    + (m[row][3] as u64) * (d[3] as u64);
                Fp32::<P>::from_u64(acc) + even.coeffs[row]
            })
        } else {
            std::array::from_fn(|row| {
                let acc: u128 = (m[row][0] as u128) * (d[0] as u128)
                    + (m[row][1] as u128) * (d[1] as u128)
                    + (m[row][2] as u128) * (d[2] as u128)
                    + (m[row][3] as u128) * (d[3] as u128);
                Fp32::<P>::from_canonical_u128_reduced(acc) + even.coeffs[row]
            })
        };
        FpExt4::new(folded)
    }
}

macro_rules! impl_fp_ext4_unreduced_identity {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasUnreducedOps for FpExt4<$base<$p>> {
            type MulU64Accum = Self;
            type ProductAccum = Self;

            #[inline]
            fn mul_u64_unreduced(self, small: u64) -> Self {
                let small = $base::<$p>::from_u64(small);
                Self::new(self.coeffs.map(|coeff| coeff * small))
            }
            #[inline]
            fn mul_to_product_accum(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn reduce_mul_u64_accum(accum: Self) -> Self {
                accum
            }
            #[inline]
            fn reduce_product_accum(accum: Self) -> Self {
                accum
            }
        }

        impl<const $p: $pty> MulBaseUnreduced<$base<$p>> for FpExt4<$base<$p>> {}
    };
}

impl_fp_ext4_unreduced_identity!(Fp64<P: u64>);
impl_fp_ext4_unreduced_identity!(Fp128<P: u128>);

macro_rules! impl_fp_ext4_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasOptimizedFold for FpExt4<$base<$p>> {
            type FoldCtx = Self;
            #[inline]
            fn precompute_fold(r: Self) -> Self {
                r
            }
            #[inline]
            fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
                even + *r * (odd - even)
            }
        }
    };
}

impl_fp_ext4_default_optimized_fold!(Fp64<P: u64>);
impl_fp_ext4_default_optimized_fold!(Fp128<P: u128>);
