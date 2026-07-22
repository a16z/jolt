//! Shared macros for the mechanical supertrait obligations of the algebra
//! hierarchy.
//!
//! Each macro is invoked from the concrete type's own file, so a type's full
//! trait surface stays visible where the type is defined; only the expansion
//! is shared.

/// Implements `Zero`, `One`, `Display`, `Hash`, owned and by-reference
/// `Sum`/`Product`, and the `AdditiveGroup` marker for a ring-like type from
/// per-type leaf expressions.
macro_rules! impl_native_ring_algebra {
    (
        impl[$($g:tt)*] $ty:ty {
            zero: $zero:expr,
            is_zero($isz:ident): $is_zero:expr,
            one: $one:expr,
            display($dv:ident, $f:ident): $display:expr,
            hash($hv:ident, $st:ident): $hash:expr $(,)?
        }
    ) => {
        impl<$($g)*> ::num_traits::Zero for $ty {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn is_zero(&self) -> bool {
                let $isz = self;
                $is_zero
            }
        }

        impl<$($g)*> ::num_traits::One for $ty {
            #[inline]
            fn one() -> Self {
                $one
            }
        }

        impl<$($g)*> ::std::fmt::Display for $ty {
            fn fmt(&self, $f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                let $dv = self;
                $display
            }
        }

        impl<$($g)*> ::std::hash::Hash for $ty {
            fn hash<JfHasher: ::std::hash::Hasher>(&self, $st: &mut JfHasher) {
                let $hv = self;
                $hash
            }
        }

        impl<$($g)*> ::std::iter::Sum for $ty {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as ::num_traits::Zero>::zero(), |acc, x| acc + x)
            }
        }

        impl<'jf_ref, $($g)*> ::std::iter::Sum<&'jf_ref Self> for $ty {
            fn sum<I: Iterator<Item = &'jf_ref Self>>(iter: I) -> Self {
                iter.fold(<Self as ::num_traits::Zero>::zero(), |acc, x| acc + *x)
            }
        }

        impl<$($g)*> ::std::iter::Product for $ty {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as ::num_traits::One>::one(), |acc, x| acc * x)
            }
        }

        impl<'jf_ref, $($g)*> ::std::iter::Product<&'jf_ref Self> for $ty {
            fn product<I: Iterator<Item = &'jf_ref Self>>(iter: I) -> Self {
                iter.fold(<Self as ::num_traits::One>::one(), |acc, x| acc * *x)
            }
        }

        impl<$($g)*> $crate::AdditiveGroup for $ty {}
    };
}

/// Implements `Zero`, the by-reference `Add`/`Sub` forwarders, and the
/// `AdditiveGroup` marker for wide accumulator types (no multiplication, no
/// multiplicative identity).
macro_rules! impl_native_additive {
    (
        impl[$($g:tt)*] $ty:ty {
            zero: $zero:expr,
            is_zero($isz:ident): $is_zero:expr $(,)?
        }
    ) => {
        impl<$($g)*> ::num_traits::Zero for $ty {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn is_zero(&self) -> bool {
                let $isz = self;
                $is_zero
            }
        }

        impl<'jf_ref, $($g)*> ::std::ops::Add<&'jf_ref Self> for $ty {
            type Output = Self;

            #[inline]
            fn add(self, rhs: &'jf_ref Self) -> Self::Output {
                self + *rhs
            }
        }

        impl<'jf_ref, $($g)*> ::std::ops::Sub<&'jf_ref Self> for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: &'jf_ref Self) -> Self::Output {
                self - *rhs
            }
        }

        impl<$($g)*> $crate::AdditiveGroup for $ty {}
    };
}

/// Implements the value, assignment, and by-reference operator matrix for a
/// const-generic Solinas prime type by delegating to its `add_raw`/`sub_raw`/
/// `mul_raw` kernels. The reduction logic itself stays hand-written per type.
macro_rules! impl_prime_ops {
    ($ty:ident<$p:ident: $p_ty:ty>, zero_raw: $zero_raw:expr) => {
        impl<const $p: $p_ty> ::std::ops::Add for $ty<$p> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                Self(Self::add_raw(self.0, rhs.0))
            }
        }

        impl<const $p: $p_ty> ::std::ops::Sub for $ty<$p> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self(Self::sub_raw(self.0, rhs.0))
            }
        }

        impl<const $p: $p_ty> ::std::ops::Mul for $ty<$p> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                Self(Self::mul_raw(self.0, rhs.0))
            }
        }

        impl<const $p: $p_ty> ::std::ops::Neg for $ty<$p> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self(Self::sub_raw($zero_raw, self.0))
            }
        }

        impl<const $p: $p_ty> ::std::ops::AddAssign for $ty<$p> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<const $p: $p_ty> ::std::ops::SubAssign for $ty<$p> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<const $p: $p_ty> ::std::ops::MulAssign for $ty<$p> {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<'jf_ref, const $p: $p_ty> ::std::ops::Add<&'jf_ref Self> for $ty<$p> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: &'jf_ref Self) -> Self::Output {
                self + *rhs
            }
        }

        impl<'jf_ref, const $p: $p_ty> ::std::ops::Sub<&'jf_ref Self> for $ty<$p> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: &'jf_ref Self) -> Self::Output {
                self - *rhs
            }
        }

        impl<'jf_ref, const $p: $p_ty> ::std::ops::Mul<&'jf_ref Self> for $ty<$p> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: &'jf_ref Self) -> Self::Output {
                self * *rhs
            }
        }
    };
}

pub(crate) use {impl_native_additive, impl_native_ring_algebra, impl_prime_ops};
