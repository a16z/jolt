//! Extension towers over the Solinas prime fields: quadratic ([`FpExt2`]),
//! quartic ([`FpExt4`]), and octic ([`FpExt8`]), plus the Frobenius/Moore
//! machinery.
//!
//! `FpExt4`/`FpExt8` use the cyclotomic ring-subfield basis `[1, e1, ...]`
//! (`e_j = zeta^(jm) + zeta^(-jm)`) aligned with trace reduction; there is no
//! alternate power- or tower-basis quartic implementation. Their multiply
//! dispatches through the [`PseudoMersenne`] kernel hooks; every base field
//! keeps the generic-schedule defaults (`crate::schedules`) — the baseline's
//! fused u128-accumulation `Fp32` override lost the checkpoint-6 bench gate
//! (see specs/jolt-field-rebuild.md and `benches/ext4_kernels.rs`).
//!
//! Frobenius powers are intentionally algebraic (raise to powers of the base
//! modulus) rather than basis-specific: one auditable contract first;
//! cheaper specializations can come later.

#![expect(
    clippy::expect_used,
    reason = "registered pseudo-Mersenne parameters are a field-type invariant"
)]

use crate::solinas::pseudo_mersenne_modulus;
use crate::{Ext2Config, ExtField, Field, FieldError, PseudoMersenne, Ring};
use num_traits::Zero;
use rand_core::RngCore;
use std::marker::PhantomData;

/// Quadratic extension element `c0 + c1·u` with `u^2 = NR` given by the
/// config `C`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: Field + allocative::Allocative, C: Ext2Config<F>")
)]
#[repr(transparent)]
pub struct FpExt2<F: Field, C: Ext2Config<F>> {
    /// Coefficients `[c0, c1]` in basis `[1, u]`.
    pub coeffs: [F; 2],
    _cfg: PhantomData<fn() -> C>,
}

/// Default quadratic extension used by the Solinas backend.
pub type Ext2<F> = FpExt2<F, crate::TwoNr>;

impl<F: Field, C: Ext2Config<F>> FpExt2<F, C> {
    /// Constructs `c0 + c1·u`.
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

    /// Multiplies a base-field element by the non-residue (a free negation
    /// when `C` declares a non-residue of `-1`).
    #[inline(always)]
    fn mul_nr(x: F) -> F {
        C::mul_non_residue(x, |base| base)
    }

    /// Returns the conjugate `c0 − c1·u`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.coeffs[0], -self.coeffs[1])
    }

    /// Returns the norm in the base field: `c0² − NR·c1²`.
    #[inline]
    pub fn norm(self) -> F {
        (self.coeffs[0] * self.coeffs[0]) - Self::mul_nr(self.coeffs[1] * self.coeffs[1])
    }
}

// Manual std impls: derives would impose their bounds on the config ZST `C`.
impl<F: Field, C: Ext2Config<F>> Clone for FpExt2<F, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<F: Field, C: Ext2Config<F>> Copy for FpExt2<F, C> {}
impl<F: Field, C: Ext2Config<F>> Default for FpExt2<F, C> {
    fn default() -> Self {
        Self::new(F::zero(), F::zero())
    }
}
impl<F: Field, C: Ext2Config<F>> PartialEq for FpExt2<F, C> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}
impl<F: Field, C: Ext2Config<F>> Eq for FpExt2<F, C> {}
impl<F: Field, C: Ext2Config<F>> std::hash::Hash for FpExt2<F, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}
impl<F: Field, C: Ext2Config<F>> std::fmt::Debug for FpExt2<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FpExt2")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}
impl<F: Field, C: Ext2Config<F>> std::fmt::Display for FpExt2<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.coeffs[0], self.coeffs[1])
    }
}

crate::impl_ring_ops!(impl[F: Field, C: Ext2Config<F>] FpExt2<F, C> {
    add(a, b): FpExt2::new(a.coeffs[0] + b.coeffs[0], a.coeffs[1] + b.coeffs[1]),
    sub(a, b): FpExt2::new(a.coeffs[0] - b.coeffs[0], a.coeffs[1] - b.coeffs[1]),
    // Karatsuba: 3 base multiplies (2 when NR = −1 makes mul_nr free).
    mul(a, b): {
        let v0 = a.coeffs[0] * b.coeffs[0];
        let v1 = a.coeffs[1] * b.coeffs[1];
        let cross = (a.coeffs[0] + a.coeffs[1]) * (b.coeffs[0] + b.coeffs[1]);
        FpExt2::new(v0 + Self::mul_nr(v1), cross - v0 - v1)
    },
    neg(a): FpExt2::new(-a.coeffs[0], -a.coeffs[1]),
    zero: FpExt2::new(F::zero(), F::zero()),
    one: FpExt2::new(F::one(), F::zero()),
});

impl<F: Field, C: Ext2Config<F>> Ring for FpExt2<F, C> {
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self::new(F::from_u64(v), F::zero())
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        Self::new(F::from_i64(v), F::zero())
    }
    #[inline]
    fn from_u128(v: u128) -> Self {
        Self::new(F::from_u128(v), F::zero())
    }
    #[inline]
    fn from_i128(v: i128) -> Self {
        Self::new(F::from_i128(v), F::zero())
    }

    /// Specialized squaring, 2 base multiplies instead of 3:
    /// `(c0 + c1·u)² = (c0² + NR·c1²) + (2·c0·c1)·u`.
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

impl<F: Field, C: Ext2Config<F>> Field for FpExt2<F, C> {
    /// Inversion via the norm: `x^{-1} = conjugate(x) / norm(x)`.
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

    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs[0].half(), self.coeffs[1].half())
    }

    #[inline]
    fn two_inv() -> Self {
        Self::new(F::two_inv(), F::zero())
    }
}

impl<F: Field + serde::Serialize, C: Ext2Config<F>> serde::Serialize for FpExt2<F, C> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.coeffs.serialize(serializer)
    }
}
impl<'de, F, C> serde::Deserialize<'de> for FpExt2<F, C>
where
    F: Field + serde::Deserialize<'de>,
    C: Ext2Config<F>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let [c0, c1] = <[F; 2]>::deserialize(deserializer)?;
        Ok(Self::new(c0, c1))
    }
}

/// Quartic extension element in the cyclotomic ring-subfield basis
/// `[1, e1, e2, e3]`. Multiplication dispatches through
/// [`PseudoMersenne::ext4_mul`].
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: Field + allocative::Allocative")
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(transparent)]
pub struct FpExt4<F: Field> {
    /// Coefficients in basis `[1, e1, e2, e3]`.
    pub coeffs: [F; 4],
}

impl<F: Field> FpExt4<F> {
    /// Constructs from basis coefficients `[c0, c1, c2, c3]`.
    #[inline]
    pub fn new(coeffs: [F; 4]) -> Self {
        Self { coeffs }
    }

    // Arithmetic in the degree-2 subfield generated by e2 (`e2² = 2`),
    // used by `inverse` to reduce quartic inversion to base inversion.
    #[inline(always)]
    fn ext2_mul_by_e2_nr(lhs: (F, F), rhs: (F, F)) -> (F, F) {
        let (a0, a1) = lhs;
        let (b0, b1) = rhs;
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let c1 = (a0 + a1) * (b0 + b1) - v0 - v1;
        (v0 + v1 + v1, c1)
    }

    #[inline(always)]
    fn ext2_square_by_e2_nr(x: (F, F)) -> (F, F) {
        let (a0, a1) = x;
        let a0a1 = a0 * a1;
        (a0.square() + a1.square() + a1.square(), a0a1 + a0a1)
    }

    #[inline(always)]
    fn ext2_mul_by_e1_nr(x: (F, F)) -> (F, F) {
        let (x0, x1) = x;
        (x0 + x0 + x1 + x1, x0 + x1 + x1)
    }

    #[inline(always)]
    fn ext2_inverse_by_e2_nr(x: (F, F)) -> Option<(F, F)> {
        let (x0, x1) = x;
        let inv_norm = (x0.square() - (x1.square() + x1.square())).inverse()?;
        Some((x0 * inv_norm, -x1 * inv_norm))
    }
}

impl<F: Field> std::fmt::Display for FpExt4<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [c0, c1, c2, c3] = self.coeffs;
        write!(f, "({c0}, {c1}, {c2}, {c3})")
    }
}

crate::impl_ring_ops!(impl[F: PseudoMersenne] FpExt4<F> {
    add(a, b): FpExt4::new(std::array::from_fn(|i| a.coeffs[i] + b.coeffs[i])),
    sub(a, b): FpExt4::new(std::array::from_fn(|i| a.coeffs[i] - b.coeffs[i])),
    mul(a, b): FpExt4::new(F::ext4_mul(a.coeffs, b.coeffs)),
    neg(a): FpExt4::new(std::array::from_fn(|i| -a.coeffs[i])),
    zero: FpExt4::new([F::zero(); 4]),
    one: FpExt4::new([F::one(), F::zero(), F::zero(), F::zero()]),
});

impl<F: PseudoMersenne> Ring for FpExt4<F> {
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self::new([F::from_u64(v), F::zero(), F::zero(), F::zero()])
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        Self::new([F::from_i64(v), F::zero(), F::zero(), F::zero()])
    }
    #[inline]
    fn from_u128(v: u128) -> Self {
        Self::new([F::from_u128(v), F::zero(), F::zero(), F::zero()])
    }
    #[inline]
    fn from_i128(v: i128) -> Self {
        Self::new([F::from_i128(v), F::zero(), F::zero(), F::zero()])
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self::new(F::ext4_square(self.coeffs))
    }
}

impl<F: PseudoMersenne> Field for FpExt4<F> {
    /// Inversion via the subfield tower: write `self = a + b·e1` over the
    /// e2-subfield, invert the norm `a² − e1²·b²` there, then one base
    /// inversion.
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let [a0, a1, a2, a3] = self.coeffs;
        let a = (a0, a2);
        let b = (a1 - a3, a3);

        let aa = Self::ext2_square_by_e2_nr(a);
        let bb = Self::ext2_square_by_e2_nr(b);
        let norm = {
            let nr_bb = Self::ext2_mul_by_e1_nr(bb);
            (aa.0 - nr_bb.0, aa.1 - nr_bb.1)
        };
        let inv_norm = Self::ext2_inverse_by_e2_nr(norm)?;
        let constant = Self::ext2_mul_by_e2_nr(a, inv_norm);
        let e1_coeff = Self::ext2_mul_by_e2_nr((-b.0, -b.1), inv_norm);

        Some(Self::new([
            constant.0,
            e1_coeff.0 + e1_coeff.1,
            constant.1,
            e1_coeff.1,
        ]))
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(std::array::from_fn(|_| F::random(rng)))
    }

    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs.map(F::half))
    }

    #[inline]
    fn two_inv() -> Self {
        Self::new([F::two_inv(), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: Field + serde::Serialize> serde::Serialize for FpExt4<F> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.coeffs.serialize(serializer)
    }
}
impl<'de, F: Field + serde::Deserialize<'de>> serde::Deserialize<'de> for FpExt4<F> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(<[F; 4]>::deserialize(deserializer)?))
    }
}

/// Octic extension element in the Chebyshev basis `[1, e1, ..., e7]`.
/// Multiplication dispatches through [`PseudoMersenne::ext8_mul`].
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[cfg_attr(
    feature = "allocative",
    allocative(bound = "F: Field + allocative::Allocative")
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(transparent)]
pub struct FpExt8<F: Field> {
    /// Coefficients in basis `[1, e1, ..., e7]`.
    pub coeffs: [F; 8],
}

impl<F: Field> FpExt8<F> {
    /// Constructs from canonical basis coefficients.
    #[inline]
    pub fn new(coeffs: [F; 8]) -> Self {
        Self { coeffs }
    }

    #[inline]
    fn from_constant(c: F) -> Self {
        let mut coeffs = [F::zero(); 8];
        coeffs[0] = c;
        Self::new(coeffs)
    }
}

impl<F: Field> std::fmt::Display for FpExt8<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [c0, c1, c2, c3, c4, c5, c6, c7] = self.coeffs;
        write!(f, "({c0}, {c1}, {c2}, {c3}, {c4}, {c5}, {c6}, {c7})")
    }
}

crate::impl_ring_ops!(impl[F: PseudoMersenne] FpExt8<F> {
    add(a, b): FpExt8::new(std::array::from_fn(|i| a.coeffs[i] + b.coeffs[i])),
    sub(a, b): FpExt8::new(std::array::from_fn(|i| a.coeffs[i] - b.coeffs[i])),
    mul(a, b): FpExt8::new(F::ext8_mul(a.coeffs, b.coeffs)),
    neg(a): FpExt8::new(std::array::from_fn(|i| -a.coeffs[i])),
    zero: FpExt8::new([F::zero(); 8]),
    one: FpExt8::from_constant(F::one()),
});

impl<F: PseudoMersenne> Ring for FpExt8<F> {
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self::from_constant(F::from_u64(v))
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        Self::from_constant(F::from_i64(v))
    }
    #[inline]
    fn from_u128(v: u128) -> Self {
        Self::from_constant(F::from_u128(v))
    }
    #[inline]
    fn from_i128(v: i128) -> Self {
        Self::from_constant(F::from_i128(v))
    }

    /// Squaring via the dedicated schedule (fewer base ops than the mul
    /// schedule; identical field result).
    #[inline(always)]
    fn square(&self) -> Self {
        Self::new(F::ext8_square(self.coeffs))
    }
}

impl<F: PseudoMersenne> Field for FpExt8<F> {
    /// Inversion by dense Gaussian elimination on the 8×8 multiplication
    /// matrix — explicit and auditable; octic inversion is not a hot path.
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

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(std::array::from_fn(|_| F::random(rng)))
    }

    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs.map(F::half))
    }

    #[inline]
    fn two_inv() -> Self {
        Self::from_constant(F::two_inv())
    }
}

impl<F: Field + serde::Serialize> serde::Serialize for FpExt8<F> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.coeffs.serialize(serializer)
    }
}
impl<'de, F: Field + serde::Deserialize<'de>> serde::Deserialize<'de> for FpExt8<F> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(<[F; 8]>::deserialize(deserializer)?))
    }
}

#[inline]
fn field_pow_u128<E: Field>(mut base: E, mut exp: u128) -> E {
    let mut acc = E::one();
    while exp > 0 {
        if (exp & 1) == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

#[inline]
fn base_modulus<F: PseudoMersenne>() -> u128 {
    pseudo_mersenne_modulus(F::MODULUS_BITS, F::OFFSET)
        .expect("pseudo-Mersenne modulus parameters must be valid")
}

fn frobenius_pow_via_base_modulus<F, E>(value: E, power: usize) -> E
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    let q = base_modulus::<F>();
    let mut out = value;
    for _ in 0..(power % E::DEGREE.max(1)) {
        out = field_pow_u128(out, q);
    }
    out
}

/// A pseudo-Mersenne base field is its own degree-1 extension.
impl<F: PseudoMersenne> ExtField<F> for F {
    const DEGREE: usize = 1;

    #[inline]
    fn lift_base(x: F) -> Self {
        x
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        self * x
    }

    #[inline]
    fn from_base_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> F,
    {
        f(0)
    }

    #[inline]
    fn base_coefficient(&self, index: usize) -> F {
        assert_eq!(index, 0);
        *self
    }

    /// Frobenius is the identity on the prime field.
    #[inline]
    fn frobenius_pow(self, _power: usize) -> Self {
        self
    }
}

impl<F: PseudoMersenne, C: Ext2Config<F>> ExtField<F> for FpExt2<F, C> {
    const DEGREE: usize = 2;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(x, F::zero())
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }

    #[inline]
    fn from_base_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> F,
    {
        Self::new(f(0), f(1))
    }

    #[inline]
    fn base_coefficient(&self, index: usize) -> F {
        self.coeffs[index]
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F: PseudoMersenne> ExtField<F> for FpExt4<F> {
    const DEGREE: usize = 4;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs.map(|c| c * x))
    }

    #[inline]
    fn from_base_fn<G>(f: G) -> Self
    where
        G: FnMut(usize) -> F,
    {
        Self::new(std::array::from_fn(f))
    }

    #[inline]
    fn base_coefficient(&self, index: usize) -> F {
        self.coeffs[index]
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F: PseudoMersenne> ExtField<F> for FpExt8<F> {
    const DEGREE: usize = 8;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::from_constant(x)
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs.map(|c| c * x))
    }

    #[inline]
    fn from_base_fn<G>(f: G) -> Self
    where
        G: FnMut(usize) -> F,
    {
        Self::new(std::array::from_fn(f))
    }

    #[inline]
    fn base_coefficient(&self, index: usize) -> F {
        self.coeffs[index]
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

/// Returns the first `width` elements of the canonical extension basis.
///
/// For [`FpExt4`]/[`FpExt8`] this is the ring-subfield basis `[1, e1, ...]`,
/// so the chosen Moore-type theta family is aligned with the coefficient
/// packing basis.
///
/// # Errors
///
/// Returns an error if `width > E::DEGREE`.
pub fn canonical_extension_basis<F, E>(width: usize) -> Result<Vec<E>, FieldError>
where
    F: Field,
    E: ExtField<F>,
{
    if width > E::DEGREE {
        return Err(FieldError::InvalidInput(format!(
            "Frobenius theta width {width} exceeds extension degree {}",
            E::DEGREE
        )));
    }
    Ok((0..width)
        .map(|idx| {
            let mut coeffs = vec![F::zero(); E::DEGREE];
            coeffs[idx] = F::one();
            E::from_base_slice(&coeffs)
        })
        .collect())
}

/// Solves `M_t(theta) z = r`, where `M_t(theta)_{j,h} = theta_h^(q^-j)`.
///
/// Dense elimination on purpose: supported Frobenius widths are tiny
/// (`≤ [E:F]`) and explicit validation beats a clever specialized solver.
///
/// # Errors
///
/// Returns an error on a dimension mismatch or a singular Moore-type matrix.
pub fn solve_frobenius_moore<F, E>(thetas: &[E], rhs: &[E]) -> Result<Vec<E>, FieldError>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    let n = thetas.len();
    if rhs.len() != n {
        return Err(FieldError::InvalidSize {
            expected: n,
            actual: rhs.len(),
        });
    }
    let mut matrix = (0..n)
        .map(|row| {
            thetas
                .iter()
                .map(|&theta| theta.frobenius_inv_pow(row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut values = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .find(|&row| !matrix[row][col].is_zero())
            .ok_or_else(|| {
                FieldError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
            })?;
        if pivot != col {
            matrix.swap(col, pivot);
            values.swap(col, pivot);
        }
        let inv = matrix[col][col].inverse().ok_or_else(|| {
            FieldError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
        })?;
        for entry in &mut matrix[col][col..] {
            *entry *= inv;
        }
        values[col] *= inv;

        let pivot_tail = matrix[col][col..].to_vec();
        let pivot_value = values[col];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            if factor.is_zero() {
                continue;
            }
            for (entry, &pivot_entry) in matrix[row][col..].iter_mut().zip(pivot_tail.iter()) {
                *entry -= factor * pivot_entry;
            }
            values[row] -= factor * pivot_value;
        }
    }
    Ok(values)
}

/// Validates that the canonical theta family gives a nonsingular Moore-type
/// matrix for `width`.
///
/// # Errors
///
/// Returns an error if theta construction fails or the Moore solve rejects.
pub fn validate_canonical_frobenius_thetas<F, E>(width: usize) -> Result<(), FieldError>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    let thetas = canonical_extension_basis::<F, E>(width)?;
    let rhs = (0..width)
        .map(|idx| E::lift_base(F::from_u64((idx + 1) as u64)))
        .collect::<Vec<_>>();
    solve_frobenius_moore::<F, E>(&thetas, &rhs).map(|_| ())
}
