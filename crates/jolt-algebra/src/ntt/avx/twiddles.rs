//! Precomputed Montgomery quotients for the x86 `i16` and `i32` transforms.
//!
//! For a constant `w` the quotient is `w * p^{-1} mod 2^k`, with `k` the lane
//! width. With it, the Montgomery product of any `x` by `w` is
//! `hi(x*w) - hi(lo(x*q) * p)`, which equals `mont_mul(x, w)` bit for bit.
//! The low-half product `lo(x*q)` no longer waits on `x*w`, and the `i32`
//! kernels need no `mullo`.

use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};

/// Quotients for the constant tables of `NttTwiddles`, entry for entry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct MontQuotients<W: PrimeWidth, const D: usize> {
    /// Quotients of `fwd_twiddles`.
    pub(crate) fwd: [W; D],
    /// Quotients of `inv_twiddles`.
    pub(crate) inv: [W; D],
    /// Quotients of `psi_pows`.
    pub(crate) psi: [W; D],
    /// Quotients of `psi_pows_r2`.
    pub(crate) psi_r2: [W; D],
    /// Quotients of `d_inv_psi_inv`.
    pub(crate) d_inv_psi_inv: [W; D],
    /// Quotient of `d_inv`.
    pub(crate) d_inv: W,
}

impl<W: PrimeWidth, const D: usize> MontQuotients<W, D> {
    /// Derive the quotients of the Montgomery-form tables.
    pub(crate) fn compute(
        prime: NttPrime<W>,
        fwd_twiddles: &[MontCoeff<W>; D],
        inv_twiddles: &[MontCoeff<W>; D],
        psi_pows: &[MontCoeff<W>; D],
        psi_pows_r2: &[W; D],
        d_inv_psi_inv: &[MontCoeff<W>; D],
        d_inv: MontCoeff<W>,
    ) -> Self {
        let quotient = |value: W| value.wrapping_mul(prime.pinv);
        Self {
            fwd: fwd_twiddles.map(|w| quotient(w.raw())),
            inv: inv_twiddles.map(|w| quotient(w.raw())),
            psi: psi_pows.map(|w| quotient(w.raw())),
            psi_r2: psi_pows_r2.map(quotient),
            d_inv_psi_inv: d_inv_psi_inv.map(|w| quotient(w.raw())),
            d_inv: quotient(d_inv.raw()),
        }
    }
}
