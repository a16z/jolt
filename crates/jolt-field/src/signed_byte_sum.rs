use crate::Field;

pub(crate) fn fold<'a, F: Field + 'a>(terms: impl IntoIterator<Item = (&'a F, i8)>) -> F {
    terms
        .into_iter()
        .fold(F::zero(), |sum, (value, coefficient)| {
            sum + *value * F::from_i64(i64::from(coefficient))
        })
}

macro_rules! field_methods {
    () => {
        /// Sum of field values multiplied by signed bytes, including -128.
        /// The empty sum is zero. Coefficients remain integers until consumed.
        fn sum_signed_bytes<'a>(terms: impl IntoIterator<Item = (&'a Self, i8)>) -> Self
        where
            Self: 'a,
        {
            $crate::signed_byte_sum::fold(terms)
        }
    };
}
pub(crate) use field_methods;

#[cfg(all(feature = "field-inline-guest", target_arch = "riscv64"))]
macro_rules! fp128_methods {
    () => {
        fn sum_signed_bytes<'a>(terms: impl IntoIterator<Item = (&'a Self, i8)>) -> Self
        where
            Self: 'a,
        {
            if !Self::inline_accumulators() {
                return $crate::signed_byte_sum::fold(terms);
            }
            Self::from_inline_limbs($crate::field_inline::sum_signed_bytes(
                terms
                    .into_iter()
                    .map(|(value, coefficient)| (&value.0, coefficient)),
            ))
        }
    };
}
#[cfg(all(feature = "field-inline-guest", target_arch = "riscv64"))]
pub(crate) use fp128_methods;
