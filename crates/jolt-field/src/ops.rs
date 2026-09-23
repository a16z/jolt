//! Exported stamping macros.
//!
//! Concrete types provide raw add/sub/mul/neg bodies and the two identities;
//! these macros emit the full operator matrix, iterator sums/products, and
//! canonical-bytes serde. They are exported so third-party field
//! implementors pay the same near-zero boilerplate as the in-crate backends.
//!
//! Generic parameters are passed as a raw token list: `impl[const P: u64]`,
//! `impl[F: Field, C: Config<F>]`, or `impl[]` for concrete types.

/// Implements the additive operator matrix for a group type: `Add`/`Sub`
/// (owned and by-ref), `AddAssign`/`SubAssign`, `Neg`, `Zero`, and the
/// [`AdditiveGroup`](crate::AdditiveGroup) marker.
///
/// `is_zero` compares against the zero expression, which is correct for
/// types whose stored representation is canonical.
#[macro_export]
macro_rules! impl_group_ops {
    (impl[$($g:tt)*] $ty:ty {
        add($aa:ident, $ab:ident): $add:expr,
        sub($sa:ident, $sb:ident): $sub:expr,
        neg($na:ident): $neg:expr,
        zero: $zero:expr $(,)?
    }) => {
        impl<$($g)*> ::core::ops::Add for $ty {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let ($aa, $ab) = (self, rhs);
                $add
            }
        }
        impl<'a, $($g)*> ::core::ops::Add<&'a $ty> for $ty {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: &'a $ty) -> Self {
                self + *rhs
            }
        }
        impl<$($g)*> ::core::ops::AddAssign for $ty {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl<$($g)*> ::core::ops::Sub for $ty {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let ($sa, $sb) = (self, rhs);
                $sub
            }
        }
        impl<'a, $($g)*> ::core::ops::Sub<&'a $ty> for $ty {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: &'a $ty) -> Self {
                self - *rhs
            }
        }
        impl<$($g)*> ::core::ops::SubAssign for $ty {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl<$($g)*> ::core::ops::Neg for $ty {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                let $na = self;
                $neg
            }
        }
        impl<$($g)*> ::num_traits::Zero for $ty {
            #[inline(always)]
            fn zero() -> Self {
                $zero
            }
            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == $zero
            }
        }
        impl<$($g)*> $crate::AdditiveGroup for $ty {}
    };
}

/// Implements the full ring operator matrix: everything in
/// [`impl_group_ops!`] plus `Mul` (owned and by-ref), `MulAssign`, `One`,
/// and iterator `Sum`/`Product` (owned and by-ref).
#[macro_export]
macro_rules! impl_ring_ops {
    (impl[$($g:tt)*] $ty:ty {
        add($aa:ident, $ab:ident): $add:expr,
        sub($sa:ident, $sb:ident): $sub:expr,
        mul($ma:ident, $mb:ident): $mul:expr,
        neg($na:ident): $neg:expr,
        zero: $zero:expr,
        one: $one:expr $(,)?
    }) => {
        $crate::impl_group_ops!(impl[$($g)*] $ty {
            add($aa, $ab): $add,
            sub($sa, $sb): $sub,
            neg($na): $neg,
            zero: $zero,
        });
        impl<$($g)*> ::core::ops::Mul for $ty {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                let ($ma, $mb) = (self, rhs);
                $mul
            }
        }
        impl<'a, $($g)*> ::core::ops::Mul<&'a $ty> for $ty {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: &'a $ty) -> Self {
                self * *rhs
            }
        }
        impl<$($g)*> ::core::ops::MulAssign for $ty {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl<$($g)*> ::num_traits::One for $ty {
            #[inline(always)]
            fn one() -> Self {
                $one
            }
        }
        impl<$($g)*> ::core::iter::Sum for $ty {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($zero, |acc, x| acc + x)
            }
        }
        impl<'a, $($g)*> ::core::iter::Sum<&'a $ty> for $ty {
            #[inline]
            fn sum<I: Iterator<Item = &'a $ty>>(iter: I) -> Self {
                iter.fold($zero, |acc, x| acc + *x)
            }
        }
        impl<$($g)*> ::core::iter::Product for $ty {
            #[inline]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($one, |acc, x| acc * x)
            }
        }
        impl<'a, $($g)*> ::core::iter::Product<&'a $ty> for $ty {
            #[inline]
            fn product<I: Iterator<Item = &'a $ty>>(iter: I) -> Self {
                iter.fold($one, |acc, x| acc * *x)
            }
        }
    };
}

/// Implements canonical-bytes serde for a [`CanonicalEncoding`](crate::CanonicalEncoding)
/// type: serializes the exact `NUM_BYTES` little-endian canonical encoding
/// as a fixed-size byte array (no length prefix under bincode) and rejects
/// non-canonical or wrong-length encodings on deserialize.
///
/// `$n` must equal the type's `NUM_BYTES` (debug-asserted).
#[macro_export]
macro_rules! impl_serde_bytes {
    (impl[$($g:tt)*] $ty:ty, $n:expr) => {
        impl<$($g)*> ::serde::Serialize for $ty {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                debug_assert_eq!($n, <$ty as $crate::CanonicalBytes>::NUM_BYTES);
                let mut buf = [0u8; $n];
                $crate::CanonicalBytes::to_bytes_le(self, &mut buf);
                <[u8; $n]>::serialize(&buf, serializer)
            }
        }
        impl<'de, $($g)*> ::serde::Deserialize<'de> for $ty {
            fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let buf = <[u8; $n]>::deserialize(deserializer)?;
                <$ty as $crate::CanonicalEncoding>::from_bytes_le_checked(&buf)
                    .ok_or_else(|| ::serde::de::Error::custom("non-canonical field element encoding"))
            }
        }
    };
}
