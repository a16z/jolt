use std::arch::x86_64::{_mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_srli_si128};

#[inline]
pub(super) fn clmul(a: u64, b: u64) -> u128 {
    // SAFETY: this module requires cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"));
    // SSE2 is baseline on x86_64, and the cfg enables carry-less multiplication.
    unsafe {
        let product =
            _mm_clmulepi64_si128::<0>(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64));
        let low = _mm_cvtsi128_si64(product) as u64;
        let high = _mm_cvtsi128_si64(_mm_srli_si128::<8>(product)) as u64;
        u128::from(low) | (u128::from(high) << 64)
    }
}
