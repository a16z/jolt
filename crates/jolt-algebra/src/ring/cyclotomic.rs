//! Cyclotomic ring `Z_q[X]/(X^D + 1)` in coefficient form.

mod decomposition;
mod shift_windows;
#[cfg(test)]
mod tests;
mod traits;
mod wide;

use crate::{AdditiveGroup, CanonicalEncoding, Field, One, Ring, Zero};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::RngCore;
use std::array::from_fn;
use std::fmt;
use std::io::{Read, Write};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use decomposition::{
    balanced_decompose_coefficients_pow2_i8_into, decompose_centering_threshold,
    BalancedDecomposePow2Params, SmallSignedValue,
};
pub use shift_windows::NegacyclicShiftWindows;
pub use wide::WideCyclotomicRing;

/// Element of the cyclotomic ring `Z_q[X]/(X^D + 1)`.
///
/// Stored as `D` coefficients in the base field `F`, representing
/// `a_0 + a_1*X + ... + a_{D-1}*X^{D-1}`.
///
/// Multiplication is negacyclic convolution: `X^D = -1`, so a product
/// term at index `i + j >= D` wraps to index `(i + j) - D` with a sign flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CyclotomicRing<F: Field, const D: usize> {
    /// Coefficients in ascending degree order.
    pub coeffs: [F; D],
}

impl<F: Field, const D: usize> CyclotomicRing<F, D> {
    /// Construct from a coefficient array.
    #[inline]
    pub fn from_coefficients(coeffs: [F; D]) -> Self {
        Self { coeffs }
    }

    /// Construct from a slice, zero-padding if shorter than `D`.
    ///
    /// Avoids creating a `[F; D]` stack temporary when `D` is large.
    #[inline]
    pub fn from_slice(slice: &[F]) -> Self {
        let mut coeffs = [F::zero(); D];
        let len = slice.len().min(D);
        coeffs[..len].copy_from_slice(&slice[..len]);
        Self { coeffs }
    }

    /// Borrow the coefficient array.
    #[inline]
    pub fn coefficients(&self) -> &[F; D] {
        &self.coeffs
    }

    /// Mutably borrow the coefficient array.
    #[inline]
    pub fn coefficients_mut(&mut self) -> &mut [F; D] {
        &mut self.coeffs
    }

    /// The additive identity (all-zero polynomial).
    #[inline]
    pub fn zero() -> Self {
        Self {
            coeffs: [F::zero(); D],
        }
    }

    /// The multiplicative identity (`1 + 0*X + ... + 0*X^{D-1}`).
    #[inline]
    pub fn one() -> Self {
        let mut coeffs = [F::zero(); D];
        coeffs[0] = F::one();
        Self { coeffs }
    }

    /// The monomial `X` (i.e., `[0, 1, 0, ..., 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `D < 2`.
    #[inline]
    pub fn x() -> Self {
        assert!(D >= 2, "ring degree must be at least 2");
        let mut coeffs = [F::zero(); D];
        coeffs[1] = F::one();
        Self { coeffs }
    }

    /// Scalar multiplication: multiply every coefficient by `k`.
    #[inline]
    pub fn scale(&self, k: &F) -> Self {
        let mut out = self.coeffs;
        for c in &mut out {
            *c *= *k;
        }
        Self { coeffs: out }
    }

    /// Fused scalar multiply and accumulate: `dst += self * k`.
    ///
    /// This avoids a temporary ring and lets fields combine multiplication,
    /// addition, and modular reduction when they have a specialized path.
    #[inline]
    pub fn scale_accumulate_into(&self, dst: &mut Self, k: F) {
        for (d, s) in dst.coeffs.iter_mut().zip(self.coeffs.iter().copied()) {
            *d = s.mul_add(k, *d);
        }
    }

    /// Apply the cyclotomic automorphism `sigma_k: X -> X^k` for odd `k`.
    ///
    /// In `Z_q[X]/(X^D + 1)`, this permutes/sign-flips coefficients using
    /// exponent reduction modulo `2D`.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0` or `k` is not odd modulo `2D`.
    pub fn sigma(&self, k: usize) -> Self {
        assert!(D > 0, "ring degree must be non-zero");
        let two_d = 2 * D;
        let k_mod = k % two_d;
        assert!(k_mod % 2 == 1, "sigma_k requires odd k in Z_q[X]/(X^D + 1)");

        let mut out = [F::zero(); D];
        for (j, coeff) in self.coeffs.iter().copied().enumerate() {
            let idx = (j * k_mod) % two_d;
            if idx < D {
                out[idx] += coeff;
            } else {
                out[idx - D] -= coeff;
            }
        }
        Self { coeffs: out }
    }

    /// Apply `sigma_{-1}` (`X -> X^{-1} = X^{2D-1}` in this ring).
    ///
    /// # Panics
    ///
    /// Panics if `D == 0`.
    pub fn sigma_m1(&self) -> Self {
        assert!(D > 0, "ring degree must be non-zero");
        self.sigma(2 * D - 1)
    }

    /// Multiply by `X^k` in `Z_q[X]/(X^D + 1)` via O(D) coefficient rotation.
    /// It supports general exponents `k >= D` by reducing modulo `2D`.
    ///
    /// Since `X^D = -1`, coefficients that wrap past index `D` get negated.
    #[inline]
    pub fn negacyclic_shift(&self, k: usize) -> Self {
        let k = k % (D << 1);
        if k == 0 {
            return *self;
        }

        let global_neg = k >= D;
        let shift = k % D;

        if shift == 0 {
            return self.neg();
        }

        let mut out = [F::zero(); D];
        for i in 0..D {
            let target = i + shift;
            let wrap_neg = target >= D;
            let coeff = if global_neg ^ wrap_neg {
                -self.coeffs[i]
            } else {
                self.coeffs[i]
            };

            if target < D {
                out[target] = coeff;
            } else {
                out[target - D] = coeff;
            }
        }
        Self { coeffs: out }
    }

    /// Fused negacyclic shift + accumulate: `dst += self * X^k`.
    ///
    /// Requires `k < D`.
    /// Equivalent to `*dst += self.negacyclic_shift(k)` within the contract domain of `k < D`,
    /// but avoids allocating a temporary ring element.
    ///
    /// For arbitrary exponents (including `k >= D`), use [`Self::negacyclic_shift`].
    #[inline]
    pub fn shift_accumulate_into(&self, dst: &mut Self, k: usize) {
        debug_assert!(
            k < D,
            "fused method shift_accumulate_into: k={k} must be < D={D}"
        );

        let (lo, hi) = dst.coeffs.split_at_mut(k);
        let (self_lo, self_hi) = self.coeffs.split_at(D - k);
        for (d, s) in hi.iter_mut().zip(self_lo) {
            *d += *s; // i + k < D
        }
        for (d, s) in lo.iter_mut().zip(self_hi) {
            *d -= *s; // i + k >= D
        }
    }

    /// Fused negacyclic shift + subtract: `dst -= self * X^k`.
    ///
    /// Requires `k < D`.
    /// Equivalent to `*dst -= self.negacyclic_shift(k)` within the
    /// contract domain of `k < D`, but avoids allocating a temporary ring element.
    ///
    /// For arbitrary exponents (including `k >= D`), use [`Self::negacyclic_shift`].
    #[inline]
    pub fn shift_sub_into(&self, dst: &mut Self, k: usize) {
        debug_assert!(k < D, "fused method shift_sub_into: k={k} must be < D={D}");

        let (lo, hi) = dst.coeffs.split_at_mut(k);
        let (self_lo, self_hi) = self.coeffs.split_at(D - k);
        for (d, s) in hi.iter_mut().zip(self_lo) {
            *d -= *s; // i + k < D
        }
        for (d, s) in lo.iter_mut().zip(self_hi) {
            *d += *s; // i + k >= D
        }
    }

    /// Fused negacyclic shift + scaled accumulate: `dst += scale * self * X^k`.
    ///
    /// Requires `k < D`.
    /// Equivalent to `*dst += self.scale(&scale).negacyclic_shift(k)` within the
    /// contract domain of `k < D`, but avoids allocating a temporary ring element.
    ///
    /// For arbitrary exponents (including `k >= D`), use [`Self::negacyclic_shift`].
    #[inline]
    pub fn shift_scale_accumulate_into(&self, dst: &mut Self, k: usize, scale: F) {
        debug_assert!(
            k < D,
            "fused method shift_scale_accumulate_into: k={k} must be < D={D}"
        );

        let (lo, hi) = dst.coeffs.split_at_mut(k);
        let (self_lo, self_hi) = self.coeffs.split_at(D - k);
        for (d, s) in hi.iter_mut().zip(self_lo) {
            *d = s.mul_add(scale, *d); // i + k < D
        }
        for (d, s) in lo.iter_mut().zip(self_hi) {
            *d -= *s * scale; // i + k >= D
        }
    }

    /// Fused `dst += self * rhs` when `rhs` is coefficient-sparse.
    ///
    /// This is exact for any field coefficients in `rhs`, but runs in
    /// `O(hw(rhs) * D)` instead of `O(D^2)`.
    pub fn mul_accumulate_sparse_rhs_into(&self, rhs: &Self, dst: &mut Self)
    where
        F: CanonicalEncoding,
    {
        for (pos, coeff) in rhs.coeffs.iter().copied().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if coeff == F::one() {
                self.shift_accumulate_into(dst, pos);
            } else if coeff == -F::one() {
                self.shift_sub_into(dst, pos);
            } else {
                self.shift_scale_accumulate_into(dst, pos, coeff);
            }
        }
    }

    /// Fused `dst += self * rhs` via schoolbook negacyclic convolution.
    ///
    /// Accumulates the product directly into `dst` without allocating a
    /// temporary ring element, saving D additions and one memory pass.
    #[inline]
    pub fn mul_accumulate_into(&self, rhs: &Self, dst: &mut Self) {
        for i in 0..D {
            let ai = self.coeffs[i];
            if ai.is_zero() {
                continue;
            }

            let (dst_wrap, dst_direct) = dst.coeffs.split_at_mut(i);
            let (rhs_direct, rhs_wrap) = rhs.coeffs.split_at(D - i);

            for (dst_coeff, rhs_coeff) in dst_direct.iter_mut().zip(rhs_direct.iter()) {
                *dst_coeff += ai * *rhs_coeff;
            }
            for (dst_coeff, rhs_coeff) in dst_wrap.iter_mut().zip(rhs_wrap.iter()) {
                *dst_coeff -= ai * *rhs_coeff;
            }
        }
    }

    /// Check whether all coefficients are zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    /// Extension seam: constant-term helpers shared with the zero-knowledge
    /// building blocks tracked in LayerZero-Labs/akita#120.
    ///
    /// The constant coefficient `a_0` (zero when `D == 0`).
    #[inline]
    pub fn constant_term(&self) -> F {
        self.coeffs.first().copied().unwrap_or_else(F::zero)
    }

    /// Extension seam: constant-term helpers shared with the zero-knowledge
    /// building blocks tracked in LayerZero-Labs/akita#120.
    ///
    /// The coefficient inner product `sum_i a_i b_i`. This equals the
    /// constant coefficient of `self * other.sigma_m1()`, because
    /// `X^i X^{-j}` has a nonzero constant coefficient only when `i == j`,
    /// but it costs `D` multiplications instead of a ring product.
    #[inline]
    pub fn coefficient_inner_product(&self, other: &Self) -> F {
        self.coeffs
            .iter()
            .zip(other.coeffs.iter())
            .fold(F::zero(), |acc, (a, b)| a.mul_add(*b, acc))
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> CyclotomicRing<F, D> {
    /// Coefficients as centered integers in `(-q/2, q/2]`.
    pub fn centered_coefficients_i128(&self) -> [i128; D] {
        let modulus = (-F::one())
            .to_u128_checked()
            .expect("Akita field element must fit in u128")
            + 1;
        let half_modulus = modulus / 2;
        self.coeffs.map(|coefficient| {
            let canonical = coefficient
                .to_u128_checked()
                .expect("Akita field element must fit in u128");
            if canonical > half_modulus {
                -((modulus - canonical) as i128)
            } else {
                canonical as i128
            }
        })
    }
}
