//! Extension-field contracts: the tower surface over a base field.
//!
//! [`ExtField`] is the degree-`d` extension contract (embedding, coefficient
//! access in the canonical basis, Frobenius); [`Ext2Config`] configures a
//! quadratic extension `F[u]/(u^2 − NR)` through a zero-sized type, with the
//! [`NegOneNr`] and [`TwoNr`] presets.
//!
//! [`MulBaseUnreduced`] is the deferred ext×base multiply, stated in terms
//! of [`Unreduced::Product`].

use crate::{Field, PseudoMersenne, Ring, Unreduced};
use std::ops::{Add, Mul, Sub};

/// An algebraic extension of the base field `F`.
///
/// Provides the extension degree, embedding of and multiplication by base
/// elements, coefficient access in the canonical basis `{1, e1, ...}`, and
/// Frobenius powers `x -> x^(q^power)` for `q = |F|`.
pub trait ExtField<F: Field>: Field {
    /// Extension degree `[Self : F]`.
    const DEGREE: usize;

    /// Embeds `x ∈ F` as the constant coefficient.
    fn lift_base(x: F) -> Self;

    /// Returns `self * x` where `x` is a base-field scalar, scaling each
    /// base coordinate directly (no full extension multiply).
    fn mul_base(self, x: F) -> Self;

    /// Constructs from a base-coefficient generator.
    ///
    /// Calls `f` exactly once for each index in `0..Self::DEGREE`, in ascending
    /// order, and uses the result as that canonical-basis coefficient.
    fn from_base_fn<G>(f: G) -> Self
    where
        G: FnMut(usize) -> F;

    /// Returns coefficient `index` in the canonical basis.
    ///
    /// # Panics
    ///
    /// Panics if `index >= Self::DEGREE`.
    fn base_coefficient(&self, index: usize) -> F;

    /// Constructs from a coefficient slice `[c0, c1, ..., c_{d−1}]`.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != Self::DEGREE`.
    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), Self::DEGREE);
        Self::from_base_fn(|index| coeffs[index])
    }

    /// Returns the base-field coefficients in the canonical basis.
    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        (0..Self::DEGREE)
            .map(|index| self.base_coefficient(index))
            .collect()
    }

    /// Applies `x -> x^(q^power)`, where `q = |F|`.
    fn frobenius_pow(self, power: usize) -> Self;

    /// Applies the inverse Frobenius power: since `x -> x^q` has order
    /// `DEGREE` on `Self`, this is `frobenius_pow(DEGREE − power)`.
    #[inline]
    fn frobenius_inv_pow(self, power: usize) -> Self {
        self.frobenius_pow((Self::DEGREE - (power % Self::DEGREE)) % Self::DEGREE)
    }
}

/// Deferred-reduction extension-times-base multiply.
///
/// Scales `self` by a base scalar `x` into [`Unreduced::Product`] without
/// reducing, so a batch of `E × F` products can be summed and reduced once.
/// When [`Unreduced::SUM_IS_EXACT`] holds, the reduced sum equals the
/// per-term [`ExtField::mul_base`] sum within the accumulator's headroom.
///
/// `E × F` has no cross terms, so the default body (lift `x` and reuse
/// [`Unreduced::mul_unreduced`]) is correct everywhere; extensions whose
/// product-accumulator layout admits cheaper coordinate scaling override it.
pub trait MulBaseUnreduced<F: Field>: ExtField<F> + Unreduced {
    /// Accumulates `self · x` (extension times base scalar) unreduced.
    #[inline]
    fn mul_base_unreduced(self, x: F) -> Self::Product {
        self.mul_unreduced(Self::lift_base(x))
    }
}

/// A base field is its own degree-1 extension; the default body is exact.
impl<F: PseudoMersenne + Unreduced + ExtField<F>> MulBaseUnreduced<F> for F {}

/// Arithmetic form of an [`Ext2Config`] non-residue.
///
/// Configurations must use [`Self::Generic`] unless [`Ext2Config::non_residue`]
/// returns exactly `-1` or `2`. Packed and delayed-reduction kernels use this
/// value to select formulas that are valid only for those two constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ext2NonResidueKind {
    /// Any non-residue without a dedicated arithmetic formula.
    Generic,
    /// The non-residue is exactly `-1`.
    NegOne,
    /// The non-residue is exactly `2`.
    Two,
}

/// Parameters for a quadratic extension `F[u]/(u^2 − NR)` over `F`.
///
/// Implemented by zero-sized config types so the non-residue choice is a
/// compile-time property of the extension type.
pub trait Ext2Config<F: Ring> {
    /// Arithmetic form of the non-residue.
    ///
    /// The default keeps the generic formula. Implementations may select a
    /// specialized kind only when [`Self::non_residue`] returns that value.
    const NON_RESIDUE_KIND: Ext2NonResidueKind = Ext2NonResidueKind::Generic;

    /// The quadratic non-residue `NR` with `u^2 = NR`.
    fn non_residue() -> F;

    /// Multiplies a coefficient by the non-residue, generic over the lane
    /// type `A` (field elements or unreduced accumulator lanes);
    /// `from_base` embeds a base constant into `A`.
    #[inline]
    fn mul_non_residue<A, B>(x: A, from_base: B) -> A
    where
        A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
        B: FnOnce(F) -> A,
    {
        if Self::NON_RESIDUE_KIND == Ext2NonResidueKind::NegOne {
            from_base(F::zero()) - x
        } else {
            from_base(Self::non_residue()) * x
        }
    }
}

/// [`Ext2Config`] with non-residue −1; valid when `p ≡ 3 (mod 4)`.
pub struct NegOneNr;

impl<F: Ring> Ext2Config<F> for NegOneNr {
    const NON_RESIDUE_KIND: Ext2NonResidueKind = Ext2NonResidueKind::NegOne;

    #[inline]
    fn non_residue() -> F {
        -F::one()
    }
}

/// [`Ext2Config`] with non-residue 2; valid when `p ≡ 5 (mod 8)`, which
/// holds for every registered pseudo-Mersenne prime (`2^k − c`, `c ≡ 3 mod 8`).
pub struct TwoNr;

impl<F: Ring> Ext2Config<F> for TwoNr {
    const NON_RESIDUE_KIND: Ext2NonResidueKind = Ext2NonResidueKind::Two;

    #[inline]
    fn non_residue() -> F {
        F::from_u64(2)
    }

    /// Multiplication by 2 is a doubling: one add, no multiply.
    #[inline]
    fn mul_non_residue<A, B>(x: A, _from_base: B) -> A
    where
        A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
        B: FnOnce(F) -> A,
    {
        x + x
    }
}
