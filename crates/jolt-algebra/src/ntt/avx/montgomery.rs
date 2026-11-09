#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Montgomery-reduce the signed 64-bit lanes of `c` whose low halves are
/// arbitrary, leaving the reduced `i32` in the low half of each lane. The
/// caller keeps `|c| < 2^31 p` so the result fits.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_reduce_i32_products_avx2(
    c: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    // Only the low 32 bits of `t` matter, so the unsigned even-lane product
    // replaces a `mullo`.
    let t = _mm256_mul_epu32(c, pinv);
    let tp = _mm256_mul_epi32(t, p);
    let diff = _mm256_sub_epi64(c, tp);
    // Keep the high 32-bit two's-complement pattern from each 64-bit lane.
    // AVX2 has no arithmetic i64 shift, but the low half after this logical
    // shift is exactly the scalar `(diff >> 32) as i32` bit pattern.
    _mm256_srli_epi64::<32>(diff)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduce_range_8x_i32_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi32(1);
    let p_minus_one = _mm256_sub_epi32(p, one);
    let ge_mask = _mm256_cmpgt_epi32(a, p_minus_one);
    let after_sub = _mm256_sub_epi32(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi32(zero, after_sub);
    _mm256_add_epi32(after_sub, _mm256_and_si256(p, lt_mask))
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn reduce_range_16x_i32_avx512(a: __m512i, p: __m512i) -> __m512i {
    let one = _mm512_set1_epi32(1);
    let p_minus_one = _mm512_sub_epi32(p, one);
    let ge_mask = _mm512_cmpgt_epi32_mask(a, p_minus_one);
    let after_sub = _mm512_mask_sub_epi32(a, ge_mask, a, p);

    let zero = _mm512_setzero_si512();
    let lt_mask = _mm512_cmplt_epi32_mask(after_sub, zero);
    _mm512_mask_add_epi32(after_sub, lt_mask, after_sub, p)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_mul_16x_i16_avx2(
    a: __m256i,
    b: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    // Signed Montgomery reduction with R = 2^16:
    //   c = a*b, t = low(c)*pinv mod R, out = (c - t*p)/R.
    // `mulhi` exposes the signed high half directly, while `mullo` computes
    // the two low halves modulo R. Hence high(c) - high(t*p) is exactly the
    // desired quotient, with no lane widening or packing.
    let c_lo = _mm256_mullo_epi16(a, b);
    let c_hi = _mm256_mulhi_epi16(a, b);
    let t = _mm256_mullo_epi16(c_lo, pinv);
    let tp_hi = _mm256_mulhi_epi16(t, p);
    _mm256_sub_epi16(c_hi, tp_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduce_range_16x_i16_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi16(1);
    let p_minus_one = _mm256_sub_epi16(p, one);
    let ge_mask = _mm256_cmpgt_epi16(a, p_minus_one);
    let after_sub = _mm256_sub_epi16(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi16(zero, after_sub);
    _mm256_add_epi16(after_sub, _mm256_and_si256(p, lt_mask))
}
