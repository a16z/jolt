macro_rules! field_methods {
    () => {
        /// Adds `scale * src[i]` to `dst[i]` over the shorter slice.
        /// Elements outside that prefix are unchanged.
        #[inline(always)]
        fn add_scaled_slice(scale: Self, src: &[Self], dst: &mut [Self]) {
            for (dst, src) in dst.iter_mut().zip(src) {
                *dst += scale * *src;
            }
        }
    };
}
pub(crate) use field_methods;

#[cfg(all(feature = "field-inline-guest", target_arch = "riscv64"))]
macro_rules! fp128_methods {
    () => {
        #[inline(always)]
        fn add_scaled_slice(scale: Self, src: &[Self], dst: &mut [Self]) {
            if !Self::inline_accumulators() {
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst += scale * *src;
                }
                return;
            }
            let len = src.len().min(dst.len());
            // SAFETY: Fp128 is transparent over [u64; 2]; len is within src.
            let src: &[[u64; 2]] = unsafe { core::slice::from_raw_parts(src.as_ptr().cast(), len) };
            for (src, dst) in src.chunks(4).zip(dst[..len].chunks_mut(4)) {
                // SAFETY: transparent layout and exclusive dst borrow. No field
                // arithmetic consumes these limbs before canonical validation.
                let limbs: &mut [[u64; 2]] =
                    unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
                $crate::field_inline::setup_weight_tiles::add_scaled_tile(&scale.0, src, limbs);
                for value in dst {
                    *value = Self::from_inline_limbs(value.0);
                }
            }
        }
    };
}
#[cfg(all(feature = "field-inline-guest", target_arch = "riscv64"))]
pub(crate) use fp128_methods;
