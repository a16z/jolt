use super::*;

/// `FpExt2Config` with non-residue = -1.
///
/// Valid when `p ≡ 3 (mod 4)`, i.e. -1 is a quadratic non-residue.
pub struct NegOneNr;

impl<F: FieldCore> FpExt2Config<F> for NegOneNr {
    const IS_NEG_ONE: bool = true;

    fn non_residue() -> F {
        -F::one()
    }
}

/// `FpExt2Config` with non-residue = 2.
///
/// Valid when `p ≡ 5 (mod 8)`, i.e. 2 is a quadratic non-residue.
/// All Akita pseudo-Mersenne primes (`2^k - c` with `c ≡ 3 mod 8`)
/// satisfy this.
pub struct TwoNr;

impl<F: FieldCore + FromPrimitiveInt> FpExt2Config<F> for TwoNr {
    fn non_residue() -> F {
        F::from_u64(2)
    }

    #[inline]
    fn mul_non_residue<A, B>(x: A, _from_base: B) -> A
    where
        A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
        B: FnOnce(F) -> A,
    {
        x + x
    }
}

/// Parameters for an `FpExt2` quadratic extension over base field `F`.
pub trait FpExt2Config<F: FieldCore> {
    /// Whether the non-residue is -1.
    ///
    /// When `true`, multiplication by the non-residue is a free negation and
    /// the Karatsuba/squaring routines can avoid a base-field multiply.
    const IS_NEG_ONE: bool = false;

    /// Non-residue `NR` such that `u^2 = NR`.
    fn non_residue() -> F;

    /// Multiply a coefficient by the quadratic non-residue.
    #[inline]
    fn mul_non_residue<A, B>(x: A, from_base: B) -> A
    where
        A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
        B: FnOnce(F) -> A,
    {
        if Self::IS_NEG_ONE {
            from_base(F::zero()) - x
        } else {
            from_base(Self::non_residue()) * x
        }
    }
}

/// Quadratic extension element `c0 + c1 * u` with `u^2 = NR`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: FieldCore + allocative::Allocative, C: FpExt2Config<F>")
)]
#[repr(transparent)]
pub struct FpExt2<F: FieldCore, C: FpExt2Config<F>> {
    /// Coefficients `[c0, c1]` in basis `[1, u]`.
    pub coeffs: [F; 2],
    _cfg: PhantomData<fn() -> C>,
}

impl<F: FieldCore, C: FpExt2Config<F>> FpExt2<F, C> {
    /// Construct `c0 + c1 * u`.
    #[inline]
    pub fn new(c0: F, c1: F) -> Self {
        Self {
            coeffs: [c0, c1],
            _cfg: PhantomData,
        }
    }

    /// Degree-0 coefficient.
    #[inline]
    pub fn c0(&self) -> F {
        self.coeffs[0]
    }

    /// Degree-1 coefficient.
    #[inline]
    pub fn c1(&self) -> F {
        self.coeffs[1]
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }

    /// Construct from a `u64` embedded in the base field.
    #[inline]
    pub fn from_u64(val: u64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(F::from_u64(val), F::zero())
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(F::from_i64(val), F::zero())
    }

    /// Multiply a base-field element by the non-residue.
    ///
    /// When `IS_NEG_ONE` is true this is just a negation (no multiply).
    #[inline(always)]
    fn mul_nr(x: F) -> F {
        C::mul_non_residue(x, |base| base)
    }

    /// Return the conjugate `c0 - c1 * u`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.coeffs[0], -self.coeffs[1])
    }

    /// Return the norm in the base field: `c0^2 - NR * c1^2`.
    #[inline]
    pub fn norm(self) -> F {
        (self.coeffs[0] * self.coeffs[0]) - Self::mul_nr(self.coeffs[1] * self.coeffs[1])
    }
}

impl<F: FieldCore + std::fmt::Debug, C: FpExt2Config<F>> std::fmt::Debug for FpExt2<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FpExt2")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Clone for FpExt2<F, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Copy for FpExt2<F, C> {}

impl<F: FieldCore, C: FpExt2Config<F>> Default for FpExt2<F, C> {
    fn default() -> Self {
        Self::new(F::zero(), F::zero())
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> PartialEq for FpExt2<F, C> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs[0] == other.coeffs[0] && self.coeffs[1] == other.coeffs[1]
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Eq for FpExt2<F, C> {}

impl<F: FieldCore, C: FpExt2Config<F>> Add for FpExt2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> Sub for FpExt2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> Neg for FpExt2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.coeffs[0], -self.coeffs[1])
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> AddAssign for FpExt2<F, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] + rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] + rhs.coeffs[1];
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> SubAssign for FpExt2<F, C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] - rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] - rhs.coeffs[1];
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> Mul for FpExt2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let v0 = self.coeffs[0] * rhs.coeffs[0];
        let v1 = self.coeffs[1] * rhs.coeffs[1];
        let cross = (self.coeffs[0] + self.coeffs[1]) * (rhs.coeffs[0] + rhs.coeffs[1]);
        Self::new(v0 + Self::mul_nr(v1), cross - v0 - v1)
    }
}
impl<F: FieldCore, C: FpExt2Config<F>> MulAssign for FpExt2<F, C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore, C: FpExt2Config<F>> Add<&'a Self> for FpExt2<F, C> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C: FpExt2Config<F>> Sub<&'a Self> for FpExt2<F, C> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F: FieldCore, C: FpExt2Config<F>> Mul<&'a Self> for FpExt2<F, C> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> RingCore for FpExt2<F, C> {
    /// Specialized squaring: 2 base-field multiplications instead of 3.
    ///
    /// `(c0 + c1·u)^2 = (c0^2 + NR·c1^2) + (2·c0·c1)·u`
    #[inline(always)]
    fn square(&self) -> Self {
        let v0 = self.coeffs[0] * self.coeffs[0];
        let v1 = self.coeffs[1] * self.coeffs[1];
        Self::new(
            v0 + Self::mul_nr(v1),
            (self.coeffs[0] + self.coeffs[0]) * self.coeffs[1],
        )
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> FieldCore for FpExt2<F, C> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inverse()?;
        Some(Self::new(self.coeffs[0] * inv_n, (-self.coeffs[1]) * inv_n))
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(F::random(rng), F::random(rng))
    }
}

impl<F: HalvingField, C: FpExt2Config<F>> HalvingField for FpExt2<F, C> {
    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs[0].half(), self.coeffs[1].half())
    }
}

impl<F: FieldCore + FromPrimitiveInt, C: FpExt2Config<F>> FromPrimitiveInt for FpExt2<F, C> {
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(F::from_u128(val), F::zero())
    }

    fn from_i128(val: i128) -> Self {
        Self::new(F::from_i128(val), F::zero())
    }
}

/// Identity-stub `HasUnreducedOps` for `FpExt2` variants without a dedicated
/// delayed-reduction accumulator. `ProductAccum = Self`, so every multiply
/// reduces immediately. Same pattern as `FpExt4<Fp64/Fp128>` and
/// `FpExt8<*>`.
macro_rules! impl_fp_ext2_unreduced_identity {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty, C: FpExt2Config<$base<$p>>> HasUnreducedOps for FpExt2<$base<$p>, C> {
            type MulU64Accum = Self;
            type ProductAccum = Self;

            #[inline]
            fn mul_u64_unreduced(self, small: u64) -> Self {
                self * Self::from_u64(small)
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

        impl<const $p: $pty, C: FpExt2Config<$base<$p>>> MulBaseUnreduced<$base<$p>>
            for FpExt2<$base<$p>, C>
        {
        }
    };
}

impl_fp_ext2_unreduced_identity!(Fp32<P: u32>);
impl_fp_ext2_unreduced_identity!(Fp128<P: u128>);

macro_rules! impl_fp_ext2_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty, C: FpExt2Config<$base<$p>>> HasOptimizedFold for FpExt2<$base<$p>, C> {
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

impl_fp_ext2_default_optimized_fold!(Fp32<P: u32>);
impl_fp_ext2_default_optimized_fold!(Fp128<P: u128>);

/// Specialized EOR fold for `FpExt2<Fp64<P>, C>`.
///
/// Mirrors `FpExt4<Fp32>`: precompute the "multiply by `r`" matrix
/// once per round, then fold each pair as `even + r·(odd − even)` using
/// base-field (`u64`) products with a single delayed reduction per output
/// coordinate. Only `Fp64` bases are specialized; other bases keep the generic
/// `FpExt2` fold via `impl_fp_ext2_default_optimized_fold`.
impl<const P: u64, C: FpExt2Config<Fp64<P>>> HasOptimizedFold for FpExt2<Fp64<P>, C> {
    type FoldCtx = FoldMatrixFp64;

    /// Build the 2×2 "multiply by `r`" matrix in the `[1, u]` basis.
    ///
    /// For `r = r0 + r1·u` and `u² = NR`, multiplying `(a0, a1)` by `r` yields
    /// `(r0·a0 + NR·r1·a1, r1·a0 + r0·a1)`, i.e. the matrix
    /// `[[r0, NR·r1], [r1, r0]]`. `NR·r1` is materialized once via `mul_nr`
    /// (a free negation for `IS_NEG_ONE`, a doubling for the `NR = 2` preset).
    #[inline]
    fn precompute_fold(r: Self) -> FoldMatrixFp64 {
        let r0 = r.coeffs[0];
        let r1 = r.coeffs[1];
        let nr_r1 = Self::mul_nr(r1);
        FoldMatrixFp64([
            [r0.to_limbs(), nr_r1.to_limbs()],
            [r1.to_limbs(), r0.to_limbs()],
        ])
    }

    /// Fold one pair: `even + r·(odd − even)`.
    ///
    /// Each output coordinate is the sum of two `u64×u64 → u128` base products,
    /// reduced once by `Fp64::reduce_sum_of_two_products`. This is the
    /// schoolbook product (4 base multiplies, 2 reductions) with delayed
    /// reduction, versus the generic Karatsuba multiply (3 multiplies, 3
    /// reductions). The reduced coordinates are canonical, so the result is
    /// byte-identical to the generic fold.
    #[inline]
    fn fold_one(ctx: &FoldMatrixFp64, even: Self, odd: Self) -> Self {
        let m = &ctx.0;
        let d0 = (odd.coeffs[0] - even.coeffs[0]).to_limbs() as u128;
        let d1 = (odd.coeffs[1] - even.coeffs[1]).to_limbs() as u128;
        let c0 =
            Fp64::<P>::reduce_sum_of_two_products((m[0][0] as u128) * d0, (m[0][1] as u128) * d1);
        let c1 =
            Fp64::<P>::reduce_sum_of_two_products((m[1][0] as u128) * d0, (m[1][1] as u128) * d1);
        Self::new(even.coeffs[0] + c0, even.coeffs[1] + c1)
    }
}

/// Split `value = lo128 + hi_carry * 2^128` into base-2^64 limbs
/// `[bits 0..64, bits 64..]` for a `Fp64ProductAccum` slot pair.
///
/// The high limb may exceed 64 bits (it carries `hi_carry` in bits 64.., which
/// is small — at most 2 here), and the accumulator's `reduce` reconstructs
/// `lo + hi * 2^64` exactly, so the full (>128-bit) coefficient survives without
/// the wrap-mod-2^128 that a single-`u128` intermediate would incur.
#[inline(always)]
fn fp64_accum_limbs(lo128: u128, hi_carry: u128) -> [u128; 2] {
    [lo128 as u64 as u128, (lo128 >> 64) | (hi_carry << 64)]
}

/// Widening `FpExt2<Fp64<P>, C>` multiplication with delayed reduction.
///
/// Each coefficient is a combination of base products that can exceed 128 bits
/// — `c0` reaches `p00 + p^2` (IS_NEG_ONE) or `p00 + 2*p11` (just under 2^130),
/// and `c1 = p01 + p10` reaches ~2^129. Forming them in a single `u128` would
/// drop the carry into bit 128 (wrap mod 2^128), which is *not* congruent mod
/// `p` and corrupts the delayed sum. We instead track the carry explicitly and
/// store base-2^64 limbs via [`fp64_accum_limbs`], so summing a batch and
/// reducing once is exact. For `IS_NEG_ONE` configs the `p^2` bias keeps `c0`
/// non-negative (and `p^2 == 0 (mod p)`, so it is invisible after reduction).
#[inline(always)]
pub(crate) fn fp_ext2_mul_to_accum_fp64<const P: u64, C: FpExt2Config<Fp64<P>>>(
    a: [Fp64<P>; 2],
    b: [Fp64<P>; 2],
) -> FpExt2Fp64ProductAccum {
    let p00: u128 = a[0].mul_wide(b[0]);
    let p11 = a[1].mul_wide(b[1]);
    let p01 = a[0].mul_wide(b[1]);
    let p10 = a[1].mul_wide(b[0]);

    let [c0_lo, c0_hi] = if C::IS_NEG_ONE {
        // c0 = p00 + p^2 - p11, non-negative and < 2^129.
        let modulus_sq = (P as u128) * (P as u128);
        let (sum, carry_add) = p00.overflowing_add(modulus_sq);
        let (diff, borrow) = sum.overflowing_sub(p11);
        // c0 >= 0 guarantees carry_add >= borrow, so this stays in {0, 1}.
        let hi_carry = (carry_add as u128) - (borrow as u128);
        fp64_accum_limbs(diff, hi_carry)
    } else {
        // c0 = p00 + 2*p11, < 3*p^2 < 2^130 (carry in {0, 1, 2}).
        let (sum1, carry1) = p00.overflowing_add(p11);
        let (sum2, carry2) = sum1.overflowing_add(p11);
        let hi_carry = (carry1 as u128) + (carry2 as u128);
        fp64_accum_limbs(sum2, hi_carry)
    };
    // c1 = p01 + p10, < 2*p^2 < 2^129 (carry in {0, 1}).
    let (c1_sum, c1_carry) = p01.overflowing_add(p10);
    let [c1_lo, c1_hi] = fp64_accum_limbs(c1_sum, c1_carry as u128);

    FpExt2Fp64ProductAccum([c0_lo, c0_hi, c1_lo, c1_hi])
}

impl<const P: u64, C: FpExt2Config<Fp64<P>>> HasUnreducedOps for FpExt2<Fp64<P>, C> {
    type MulU64Accum = AccumPair<<Fp64<P> as HasUnreducedOps>::MulU64Accum>;
    type ProductAccum = FpExt2Fp64ProductAccum;

    // `fp_ext2_mul_to_accum_fp64` keeps the full >128-bit coefficient via carry-aware
    // base-2^64 limbs, so summing a batch and reducing once equals per-term `Mul`.
    // Covered by the `Ext2<Prime64Offset59>` rounds in
    // `sparse_tensor_factor_matches_dense_factor_rounds`.
    const DELAYED_PRODUCT_SUM_IS_EXACT: bool = true;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum {
        AccumPair(
            self.coeffs[0].mul_u64_unreduced(small),
            self.coeffs[1].mul_u64_unreduced(small),
        )
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> FpExt2Fp64ProductAccum {
        fp_ext2_mul_to_accum_fp64::<P, C>(self.coeffs, other.coeffs)
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self {
        Self::new(
            Fp64::<P>::reduce_mul_u64_accum(accum.0),
            Fp64::<P>::reduce_mul_u64_accum(accum.1),
        )
    }

    #[inline]
    fn reduce_product_accum(accum: FpExt2Fp64ProductAccum) -> Self {
        let [c0, c1] = accum.reduce::<P>();
        Self::new(c0, c1)
    }
}

impl<const P: u64, C: FpExt2Config<Fp64<P>>> MulBaseUnreduced<Fp64<P>> for FpExt2<Fp64<P>, C> {}

/// Default quadratic extension used by the Solinas backend tests and helpers.
pub type Ext2<F> = FpExt2<F, TwoNr>;

impl<F: FieldCore + serde::Serialize, C: FpExt2Config<F>> serde::Serialize for FpExt2<F, C> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.coeffs.serialize(serializer)
    }
}

impl<'de, F, C> serde::Deserialize<'de> for FpExt2<F, C>
where
    F: FieldCore + serde::Deserialize<'de>,
    C: FpExt2Config<F>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let [c0, c1] = <[F; 2]>::deserialize(deserializer)?;
        Ok(Self::new(c0, c1))
    }
}
