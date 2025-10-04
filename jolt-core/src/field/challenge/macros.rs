//! Macros for challenge field operations

/// Implements standard arithmetic operators (+, -, *) for F as JoltField types
#[macro_export]
macro_rules! impl_field_ops_inline {
    ($t:ty, $f:ty, $mul_mode:tt) => {
        // Challenge + Challenge operations
        impl Add<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }

        impl Sub<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }

        impl Mul<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }

        // Challenge + Field operations
        impl Add<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a> Add<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) + *rhs
            }
        }
        impl<'a> Add<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a, 'b> Add<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(*self) + *rhs
            }
        }

        impl Sub<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a> Sub<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) - *rhs
            }
        }
        impl<'a> Sub<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a, 'b> Sub<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(*self) - *rhs
            }
        }

        // Multiplication Challenge * Field
        impl Mul<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                $crate::impl_field_ops_inline!(@mul_challenge_field $mul_mode, $f, self, rhs)
            }
        }
        impl<'a> Mul<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $f) -> $f {
                $crate::impl_field_ops_inline!(@mul_challenge_field $mul_mode, $f, self, *rhs)
            }
        }
        impl<'a> Mul<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                $crate::impl_field_ops_inline!(@mul_challenge_field $mul_mode, $f, *self, rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $f) -> $f {
                $crate::impl_field_ops_inline!(@mul_challenge_field $mul_mode, $f, *self, *rhs)
            }
        }

        // Field + Challenge operations
        impl Add<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                self + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                *self + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                *self + Into::<$f>::into(rhs)
            }
        }

        impl Sub<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                self - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                *self - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                *self - Into::<$f>::into(rhs)
            }
        }

        // Multiplication Field * Challenge with mode-specific behavior
        impl Mul<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                $crate::impl_field_ops_inline!(@mul_field_challenge $mul_mode, $f, self, rhs)
            }
        }
        impl<'a> Mul<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                $crate::impl_field_ops_inline!(@mul_field_challenge $mul_mode, $f, self, *rhs)
            }
        }
        impl<'a> Mul<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                $crate::impl_field_ops_inline!(@mul_field_challenge $mul_mode, $f, *self, rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                $crate::impl_field_ops_inline!(@mul_field_challenge $mul_mode, $f, *self, *rhs)
            }
        }
    };

    (@mul_challenge_field optimized, $f:ty, $lhs:expr, $rhs:expr) => {
        $rhs.mul_hi_bigint_u128($lhs.value())
    };
    (@mul_challenge_field standard, $f:ty, $lhs:expr, $rhs:expr) => {
        Into::<$f>::into($lhs) * $rhs
    };

    (@mul_field_challenge optimized, $f:ty, $lhs:expr, $rhs:expr) => {
        $lhs.mul_hi_bigint_u128($rhs.value())
    };
    (@mul_field_challenge standard, $f:ty, $lhs:expr, $rhs:expr) => {
        $lhs * Into::<$f>::into($rhs)
    };
}
