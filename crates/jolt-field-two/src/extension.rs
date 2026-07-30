//! Extension-field contracts: the tower surface over a base field.
//!
//! [`ExtField`] is the degree-`d` extension contract (embedding, coefficient
//! access in the canonical basis, Frobenius); [`Ext2Config`] configures a
//! quadratic extension `F[u]/(u^2 − NR)` through a zero-sized type, with the
//! [`NegOneNr`] and [`TwoNr`] presets.
//!
//! `MulBaseUnreduced` (deferred ext×base multiply) is deferred to the
//! `Unreduced` checkpoint: its contract is stated in terms of
//! `Unreduced::Product`.

use crate::{Field, Ring};
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

    /// Constructs from a coefficient slice `[c0, c1, ..., c_{d−1}]`.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != Self::DEGREE`.
    fn from_base_slice(coeffs: &[F]) -> Self;

    /// Returns the base-field coefficients in the canonical basis.
    fn to_base_vec(&self) -> Vec<F>;

    /// Applies `x -> x^(q^power)`, where `q = |F|`.
    fn frobenius_pow(self, power: usize) -> Self;

    /// Applies the inverse Frobenius power: since `x -> x^q` has order
    /// `DEGREE` on `Self`, this is `frobenius_pow(DEGREE − power)`.
    #[inline]
    fn frobenius_inv_pow(self, power: usize) -> Self {
        let d = Self::DEGREE;
        self.frobenius_pow((d - (power % d)) % d)
    }
}

/// Parameters for a quadratic extension `F[u]/(u^2 − NR)` over `F`.
///
/// Implemented by zero-sized config types so the non-residue choice is a
/// compile-time property of the extension type.
pub trait Ext2Config<F: Ring> {
    /// Whether the non-residue is −1: multiplication by `NR` is then a free
    /// negation and the Karatsuba/squaring routines save a base multiply.
    const IS_NEG_ONE: bool = false;

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
        if Self::IS_NEG_ONE {
            from_base(F::zero()) - x
        } else {
            from_base(Self::non_residue()) * x
        }
    }
}

/// [`Ext2Config`] with non-residue −1; valid when `p ≡ 3 (mod 4)`.
pub struct NegOneNr;

impl<F: Ring> Ext2Config<F> for NegOneNr {
    const IS_NEG_ONE: bool = true;

    #[inline]
    fn non_residue() -> F {
        -F::one()
    }
}

/// [`Ext2Config`] with non-residue 2; valid when `p ≡ 5 (mod 8)`, which
/// holds for every registered pseudo-Mersenne prime (`2^k − c`, `c ≡ 3 mod 8`).
pub struct TwoNr;

impl<F: Ring> Ext2Config<F> for TwoNr {
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
