// Keep the head's existing source locations unchanged when the feature is off.
macro_rules! field_methods {
    () => {
        /// A weighted sum of row dot products, with checked rectangular shape.
        fn weighted_dot_product(rows: &[&[Self]], weights: &[Self], shared: &[Self]) -> Self {
            assert_eq!(rows.len(), weights.len(), "one weight per row");
            assert!(rows.iter().all(|row| row.len() == shared.len()));
            rows.iter()
                .zip(weights)
                .fold(Self::zero(), |sum, (row, weight)| {
                    sum + Self::dot_product(row, shared) * *weight
                })
        }

        /// Dot product with signed 16-bit coefficients, over the shorter input.
        /// Coefficients remain integers until their multiplication by the field
        /// weight; implementations must preserve negative and minimum values.
        fn dot_product_i16(weights: &[Self], coefficients: impl IntoIterator<Item = i16>) -> Self {
            $crate::algebra::signed_i16_dot_fold(weights, coefficients)
        }
    };
}
pub(crate) use field_methods;

macro_rules! fp128_methods {
    () => {
        fn weighted_dot_product(rows: &[&[Self]], weights: &[Self], shared: &[Self]) -> Self {
            <Self as $crate::MulBaseUnreduced<Self>>::weighted_dot_base_rows(rows, weights, shared)
        }

        fn dot_product_i16(weights: &[Self], coefficients: impl IntoIterator<Item = i16>) -> Self {
            #[cfg(all(feature = "field-inline-guest", target_arch = "riscv64"))]
            if Self::inline_accumulators() {
                // SAFETY: Fp128 is repr(transparent) over two canonical limbs.
                let limbs: &[[u64; 2]] =
                    unsafe { core::slice::from_raw_parts(weights.as_ptr().cast(), weights.len()) };
                return Self::from_inline_limbs($crate::field_inline::signed_i16_dot(
                    limbs,
                    coefficients,
                ));
            }
            $crate::algebra::signed_i16_dot_fold(weights, coefficients)
        }
    };
}
pub(crate) use fp128_methods;
