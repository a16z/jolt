//! Akita's only degree-8 extension field (cyclotomic ring-subfield basis).
//!
//! Coefficients are stored in the Chebyshev basis `[1, e1, ..., e7]`.

#![expect(
    clippy::expl_impl_clone_on_copy,
    reason = "manual Clone avoids adding irrelevant generic Clone bounds"
)]

use super::*;

/// Chebyshev `φ` fold-back for a degree-8 accumulator, using caller-supplied
/// add/sub so the same routine serves scalar, `i64`, and SIMD lane types.
///
/// `φ(k)` maps a product onto the `[1, e1, ..., e7]` basis:
/// `k = 0 → 2·constant`, `1 ≤ k ≤ 7 → +e_k`, `k = 8 → 0`,
/// `9 ≤ k ≤ 15 → −e_{16−k}`.
#[inline(always)]
fn fp_ext8_add_phi<V: Copy>(
    out: &mut [V; 8],
    idx: usize,
    value: V,
    add: &impl Fn(V, V) -> V,
    sub: &impl Fn(V, V) -> V,
) {
    match idx {
        0 => out[0] = add(out[0], add(value, value)),
        1..=7 => out[idx] = add(out[idx], value),
        8 => {}
        9..=15 => out[16 - idx] = sub(out[16 - idx], value),
        _ => unreachable!("fp_ext8 Chebyshev index out of range"),
    }
}

/// Karatsuba schedule for `FpExt8` multiplication in the Chebyshev
/// basis, generic over a lane type `V` and its add/sub/mul.
///
/// One schedule serves every backend: the scalar field default and the NEON /
/// AVX2 / AVX-512 SIMD kernels. The schedule is purely an additive combination
/// of products, so callers that reduce per operation (field or intrinsic ops)
/// and callers that defer reduction to the end are both correct, provided the
/// accumulator does not overflow.
#[inline(always)]
pub(crate) fn fp_ext8_mul_schedule<V, A, S, M>(
    a: [V; 8],
    b: [V; 8],
    zero: V,
    add: A,
    sub: S,
    mul: M,
) -> [V; 8]
where
    V: Copy,
    A: Fn(V, V) -> V,
    S: Fn(V, V) -> V,
    M: Fn(V, V) -> V,
{
    let diag: [V; 8] = std::array::from_fn(|i| mul(a[i], b[i]));
    let mut out = [zero; 8];
    out[0] = diag[0];

    for k in 1..8 {
        let mixed = sub(sub(mul(add(a[0], a[k]), add(b[0], b[k])), diag[0]), diag[k]);
        out[k] = add(out[k], mixed);
    }

    for (i, &diag_i) in diag.iter().enumerate().skip(1) {
        out[0] = add(out[0], add(diag_i, diag_i));
        fp_ext8_add_phi(&mut out, i + i, diag_i, &add, &sub);
    }

    for i in 1..8 {
        for j in (i + 1)..8 {
            let mixed = sub(sub(mul(add(a[i], a[j]), add(b[i], b[j])), diag[i]), diag[j]);
            fp_ext8_add_phi(&mut out, i + j, mixed, &add, &sub);
            fp_ext8_add_phi(&mut out, j - i, mixed, &add, &sub);
        }
    }

    out
}

/// Squaring schedule for `FpExt8`, generic over a lane type `V`.
///
/// Uses `(a_i + a_j)² − a_i² − a_j² = 2·a_i·a_j` to compute `a_i·a_j` directly
/// and double, saving one add and two subs per cross-term versus the Karatsuba
/// form. Shares `fp_ext8_add_phi` with [`fp_ext8_mul_schedule`].
#[inline(always)]
pub(crate) fn fp_ext8_square_schedule<V, A, S, M>(
    a: [V; 8],
    zero: V,
    add: A,
    sub: S,
    mul: M,
) -> [V; 8]
where
    V: Copy,
    A: Fn(V, V) -> V,
    S: Fn(V, V) -> V,
    M: Fn(V, V) -> V,
{
    let sq: [V; 8] = std::array::from_fn(|i| mul(a[i], a[i]));
    let mut out = [zero; 8];
    out[0] = sq[0];

    for k in 1..8 {
        let cross = mul(a[0], a[k]);
        out[k] = add(out[k], add(cross, cross));
    }

    for (i, &sq_i) in sq.iter().enumerate().skip(1) {
        out[0] = add(out[0], add(sq_i, sq_i));
        fp_ext8_add_phi(&mut out, i + i, sq_i, &add, &sub);
    }

    for i in 1..8 {
        for j in (i + 1)..8 {
            let cross = mul(a[i], a[j]);
            let doubled = add(cross, cross);
            fp_ext8_add_phi(&mut out, i + j, doubled, &add, &sub);
            fp_ext8_add_phi(&mut out, j - i, doubled, &add, &sub);
        }
    }

    out
}

#[inline(always)]
fn fp_ext8_mul_coeffs<F: FieldCore>(a: [F; 8], b: [F; 8]) -> [F; 8] {
    fp_ext8_mul_schedule(a, b, F::zero(), |x, y| x + y, |x, y| x - y, |x, y| x * y)
}

/// Backend hook for scalar ring-subfield degree-8 multiplication.
pub trait FpExt8MulBackend: FieldCore {
    /// Multiply coefficient arrays in `[1, e1, ..., e7]` basis.
    #[inline(always)]
    fn fp_ext8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        fp_ext8_mul_coeffs::<Self>(a, b)
    }
}

impl<const P: u32> FpExt8MulBackend for Fp32<P> {}
impl<const P: u64> FpExt8MulBackend for Fp64<P> {}
impl<const P: u128> FpExt8MulBackend for Fp128<P> {}

/// Degree-8 ring subfield element in canonical basis `[1, e1, ..., e7]`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: FieldCore + allocative::Allocative")
)]
#[repr(transparent)]
pub struct FpExt8<F: FieldCore> {
    /// Coefficients in basis `[1, e1, ..., e7]`.
    pub coeffs: [F; 8],
}

impl<F: FieldCore> FpExt8<F> {
    /// Construct from canonical ring-subfield basis coefficients.
    #[inline]
    pub fn new(coeffs: [F; 8]) -> Self {
        Self { coeffs }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new([F::zero(); 8])
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::one()
            } else {
                F::zero()
            }
        }))
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
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_u64(val)
            } else {
                F::zero()
            }
        }))
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i64(val)
            } else {
                F::zero()
            }
        }))
    }
}

impl<F: FieldCore + std::fmt::Debug> std::fmt::Debug for FpExt8<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FpExt8")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore> Clone for FpExt8<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore> Copy for FpExt8<F> {}

impl<F: FieldCore> Default for FpExt8<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldCore> PartialEq for FpExt8<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FieldCore> Eq for FpExt8<F> {}

impl<F: FieldCore> Add for FpExt8<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F: FieldCore> Sub for FpExt8<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F: FieldCore> Neg for FpExt8<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(std::array::from_fn(|i| -self.coeffs[i]))
    }
}

impl<F: FieldCore> AddAssign for FpExt8<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<F: FieldCore> SubAssign for FpExt8<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<F: FpExt8MulBackend> Mul for FpExt8<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(F::fp_ext8_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F: FpExt8MulBackend> MulAssign for FpExt8<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore> Add<&'a Self> for FpExt8<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, F: FieldCore> Sub<&'a Self> for FpExt8<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, F: FpExt8MulBackend> Mul<&'a Self> for FpExt8<F> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + FpExt8MulBackend> RingCore for FpExt8<F> {
    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }
}

impl<F: FieldCore + FpExt8MulBackend> Invertible for FpExt8<F> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let mut aug = [[F::zero(); 9]; 8];
        for col in 0..8 {
            let mut basis = [F::zero(); 8];
            basis[col] = F::one();
            let product = *self * Self::new(basis);
            for (row, coeff) in product.coeffs.iter().copied().enumerate() {
                aug[row][col] = coeff;
            }
        }
        aug[0][8] = F::one();

        for col in 0..8 {
            let pivot = (col..8).find(|&row| !aug[row][col].is_zero())?;
            if pivot != col {
                aug.swap(col, pivot);
            }
            let inv = aug[col][col].inverse()?;
            for entry in &mut aug[col][col..=8] {
                *entry *= inv;
            }
            for row in 0..8 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                if factor.is_zero() {
                    continue;
                }
                let pivot_row = aug[col];
                for (target, pivot) in aug[row][col..=8]
                    .iter_mut()
                    .zip(pivot_row[col..=8].iter().copied())
                {
                    *target -= factor * pivot;
                }
            }
        }

        Some(Self::new(std::array::from_fn(|i| aug[i][8])))
    }
}

impl<F: HalvingField + FpExt8MulBackend> HalvingField for FpExt8<F> {
    #[inline]
    fn half(self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i].half()))
    }
}

impl<F: FieldCore + RandomSampling> RandomSampling for FpExt8<F> {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(std::array::from_fn(|_| F::random(rng)))
    }
}

impl<F: FieldCore + FromPrimitiveInt> FromPrimitiveInt for FpExt8<F> {
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_u128(val)
            } else {
                F::zero()
            }
        }))
    }

    fn from_i128(val: i128) -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i128(val)
            } else {
                F::zero()
            }
        }))
    }
}

macro_rules! impl_fp_ext8_unreduced_identity {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasUnreducedOps for FpExt8<$base<$p>> {
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

        impl<const $p: $pty> MulBaseUnreduced<$base<$p>> for FpExt8<$base<$p>> {}
    };
}

impl_fp_ext8_unreduced_identity!(Fp32<P: u32>);
impl_fp_ext8_unreduced_identity!(Fp64<P: u64>);
impl_fp_ext8_unreduced_identity!(Fp128<P: u128>);

macro_rules! impl_fp_ext8_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasOptimizedFold for FpExt8<$base<$p>> {
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

impl_fp_ext8_default_optimized_fold!(Fp32<P: u32>);
impl_fp_ext8_default_optimized_fold!(Fp64<P: u64>);
impl_fp_ext8_default_optimized_fold!(Fp128<P: u128>);
