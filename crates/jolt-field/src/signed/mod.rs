//! Signed big integer types for the Jolt prover.
//!
//! These types represent signed integers with configurable bit widths using
//! sign-magnitude representation. They are used extensively in R1CS evaluation,
//! accumulation, and increment witness generation where intermediate values
//! can be negative but have bounded magnitude.
//!
//! Two families are provided:
//!
//! - [`SignedBigInt<N>`]: magnitude stored as `Limbs<N>` (width = `N * 64` bits)
//! - [`SignedBigIntHi32<N>`]: magnitude stored as `[u64; N]` + `u32` tail (width = `N * 64 + 32` bits)
//!
//! Common type aliases:
//! - `S64`, `S128`, `S192`, `S256` (from `SignedBigInt`)
//! - `S96`, `S160`, `S224` (from `SignedBigIntHi32`)

mod signed_bigint;
mod signed_bigint_hi32;

pub use signed_bigint::*;
pub use signed_bigint_hi32::*;

/// Generates the 5 standard operator impls for each `(Op, OpAssign)` pair:
/// val-val, OpAssign-val, val-ref, OpAssign-ref, ref-ref.
///
/// Each operator delegates to an `&self`-taking `_assign_in_place` method.
macro_rules! impl_signed_assign_ops {
    ($T:ident {
        $($Op:ident, $OpAssign:ident, $method:ident, $assign_method:ident => $assign_fn:ident;)*
    }) => { $(
        impl<const N: usize> $Op for $T<N> {
            type Output = Self;
            #[inline]
            fn $method(mut self, rhs: Self) -> Self {
                self.$assign_fn(&rhs);
                self
            }
        }

        impl<const N: usize> $OpAssign for $T<N> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                self.$assign_fn(&rhs);
            }
        }

        impl<const N: usize> $Op<&$T<N>> for $T<N> {
            type Output = $T<N>;
            #[inline]
            fn $method(mut self, rhs: &$T<N>) -> $T<N> {
                self.$assign_fn(rhs);
                self
            }
        }

        impl<const N: usize> $OpAssign<&$T<N>> for $T<N> {
            #[inline]
            fn $assign_method(&mut self, rhs: &$T<N>) {
                self.$assign_fn(rhs);
            }
        }

        impl<const N: usize> $Op for &$T<N> {
            type Output = $T<N>;
            #[inline]
            fn $method(self, rhs: Self) -> $T<N> {
                let mut out = *self;
                out.$assign_fn(rhs);
                out
            }
        }
    )* };
}

pub(crate) use impl_signed_assign_ops;
