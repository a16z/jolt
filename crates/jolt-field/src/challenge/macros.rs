//! Macros for challenge field operations

/// Implements standard arithmetic operators (+, -, *) for challenge × field type pairs.
///
/// Generates all 4 ownership variants (val-val, val-ref, ref-val, ref-ref) for:
/// - Challenge × Challenge: Add, Sub, Mul
/// - Challenge × Field: Add, Sub, Mul (Mul uses mode-specific dispatch)
/// - Field × Challenge: Add, Sub, Mul (Mul uses mode-specific dispatch)
///
/// `$mul_mode` is either `optimized` (uses `mul_by_hi_2limbs` for 125-bit challenges)
/// or `standard` (converts to field element first).
#[macro_export]
macro_rules! impl_field_ops_inline {
    ($t:ty, $f:ty, $mul_mode:tt) => {
        // Challenge × Challenge
        $crate::impl_field_ops_inline!(@binop Add, add, $t, $t, $f,
            |lhs, rhs| { Into::<$f>::into(lhs) + Into::<$f>::into(rhs) });
        $crate::impl_field_ops_inline!(@binop Sub, sub, $t, $t, $f,
            |lhs, rhs| { Into::<$f>::into(lhs) - Into::<$f>::into(rhs) });
        $crate::impl_field_ops_inline!(@binop Mul, mul, $t, $t, $f,
            |lhs, rhs| { Into::<$f>::into(lhs) * Into::<$f>::into(rhs) });

        // Challenge × Field
        $crate::impl_field_ops_inline!(@binop Add, add, $t, $f, $f,
            |lhs, rhs| { Into::<$f>::into(lhs) + rhs });
        $crate::impl_field_ops_inline!(@binop Sub, sub, $t, $f, $f,
            |lhs, rhs| { Into::<$f>::into(lhs) - rhs });
        $crate::impl_field_ops_inline!(@binop Mul, mul, $t, $f, $f,
            |lhs, rhs| { $crate::impl_field_ops_inline!(@mul_challenge_field $mul_mode, $f, lhs, rhs) });

        // Field × Challenge
        $crate::impl_field_ops_inline!(@binop Add, add, $f, $t, $f,
            |lhs, rhs| { lhs + Into::<$f>::into(rhs) });
        $crate::impl_field_ops_inline!(@binop Sub, sub, $f, $t, $f,
            |lhs, rhs| { lhs - Into::<$f>::into(rhs) });
        $crate::impl_field_ops_inline!(@binop Mul, mul, $f, $t, $f,
            |lhs, rhs| { $crate::impl_field_ops_inline!(@mul_field_challenge $mul_mode, $f, lhs, rhs) });
    };

    // Generates all 4 ownership variants for a single binary operator.
    // Both $Lhs and $Rhs must be Copy.
    (@binop $Op:ident, $method:ident, $Lhs:ty, $Rhs:ty, $Out:ty,
        |$lhs:ident, $rhs:ident| { $($body:tt)* }) => {
        impl $Op<$Rhs> for $Lhs {
            type Output = $Out;
            #[inline(always)]
            fn $method(self, rhs: $Rhs) -> $Out {
                let ($lhs, $rhs) = (self, rhs);
                $($body)*
            }
        }
        impl<'a> $Op<&'a $Rhs> for $Lhs {
            type Output = $Out;
            #[inline(always)]
            fn $method(self, rhs: &'a $Rhs) -> $Out {
                let ($lhs, $rhs) = (self, *rhs);
                $($body)*
            }
        }
        impl<'a> $Op<$Rhs> for &'a $Lhs {
            type Output = $Out;
            #[inline(always)]
            fn $method(self, rhs: $Rhs) -> $Out {
                let ($lhs, $rhs) = (*self, rhs);
                $($body)*
            }
        }
        impl<'a, 'b> $Op<&'b $Rhs> for &'a $Lhs {
            type Output = $Out;
            #[inline(always)]
            fn $method(self, rhs: &'b $Rhs) -> $Out {
                let ($lhs, $rhs) = (*self, *rhs);
                $($body)*
            }
        }
    };

    (@mul_challenge_field optimized, $f:ty, $lhs:expr, $rhs:expr) => {
        $rhs.mul_by_hi_2limbs($lhs.low, $lhs.high)
    };
    (@mul_challenge_field standard, $f:ty, $lhs:expr, $rhs:expr) => {
        Into::<$f>::into($lhs) * $rhs
    };

    (@mul_field_challenge optimized, $f:ty, $lhs:expr, $rhs:expr) => {
        $lhs.mul_by_hi_2limbs($rhs.low, $rhs.high)
    };
    (@mul_field_challenge standard, $f:ty, $lhs:expr, $rhs:expr) => {
        $lhs * Into::<$f>::into($rhs)
    };
}
