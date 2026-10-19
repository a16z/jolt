#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::lanes::mont_mul;
use super::montgomery::{
    mont_mul_16x_i16_avx2, reduce_range_16x_i16_avx2, reduce_range_16x_i32_avx512,
    reduce_range_8x_i32_avx2,
};
use crate::ntt::prime::{MontCoeff, NttPrime, I32_LAZY_DOT_BATCH};

/// AVX2 pointwise dot-product accumulation for up to six i32 CRT entries.
///
/// Raw signed products are accumulated in i64 lanes and Montgomery-reduced
/// once per batch. For `B <= 6` and `p < 2^30`, the reduction numerator is
/// bounded by `B*2^60 + 2^61 < 2^63`; the reduced result therefore fits i32.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` must be valid for `d`
/// writable i32 elements. Each of the first `count` pointers in `lhs` and
/// `rhs` must be valid for `d` readable i32 elements. The pointed-to ranges
/// must obey Rust's aliasing rules with `acc`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_dot_acc_i32(
    acc: *mut i32,
    lhs: *const *const i32,
    rhs: *const *const i32,
    count: usize,
    d: usize,
    p: i32,
    pinv: i32,
) {
    debug_assert!(count <= I32_LAZY_DOT_BATCH);
    macro_rules! dispatch_count {
        ($count:literal) => {{
            // SAFETY: inherited pointer contract and AVX2 target feature.
            unsafe { pointwise_dot_acc_i32_count::<$count>(acc, lhs, rhs, d, p, pinv) }
        }};
    }
    match count {
        0 => {}
        1 => dispatch_count!(1),
        2 => dispatch_count!(2),
        3 => dispatch_count!(3),
        4 => dispatch_count!(4),
        5 => dispatch_count!(5),
        6 => dispatch_count!(6),
        _ => unreachable!("pointwise dot exceeds lazy reduction bound"),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pointwise_dot_acc_i32_count<const COUNT: usize>(
    acc: *mut i32,
    lhs: *const *const i32,
    rhs: *const *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    debug_assert!(COUNT > 0 && COUNT <= I32_LAZY_DOT_BATCH);
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let mut i = 0;
    while i + 8 <= d {
        let mut even_sum = _mm256_setzero_si256();
        let mut odd_sum = _mm256_setzero_si256();
        for product in 0..COUNT {
            // SAFETY: guaranteed by this function's pointer contract and loop bounds.
            unsafe {
                let l = _mm256_loadu_si256((*lhs.add(product)).add(i).cast());
                let r = _mm256_loadu_si256((*rhs.add(product)).add(i).cast());
                even_sum = _mm256_add_epi64(even_sum, _mm256_mul_epi32(l, r));
                odd_sum = _mm256_add_epi64(
                    odd_sum,
                    _mm256_mul_epi32(_mm256_srli_epi64::<32>(l), _mm256_srli_epi64::<32>(r)),
                );
            }
        }
        // SAFETY: the six-product bound above keeps both signed reductions exact.
        unsafe {
            let even = super::montgomery::mont_reduce_i32_products_avx2(even_sum, p_v, pinv_v);
            let odd = super::montgomery::mont_reduce_i32_products_avx2(odd_sum, p_v, pinv_v);
            let batch = _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd));
            let batch = reduce_range_8x_i32_avx2(batch, p_v);
            let accumulator_pointer = acc.add(i);
            let accumulator = _mm256_loadu_si256(accumulator_pointer.cast());
            _mm256_storeu_si256(
                accumulator_pointer.cast(),
                reduce_range_8x_i32_avx2(_mm256_add_epi32(accumulator, batch), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            let mut raw_sum = 0_i64;
            for product in 0..COUNT {
                // SAFETY: guaranteed by this function's pointer contract and tail bound.
                unsafe {
                    raw_sum += i64::from(*(*lhs.add(product)).add(i))
                        * i64::from(*(*rhs.add(product)).add(i));
                }
            }
            let correction = (raw_sum as i32).wrapping_mul(pinv);
            let reduced = ((raw_sum - i64::from(correction) * i64::from(p)) >> 32) as i32;
            let reduced = prime.reduce_range(MontCoeff::from_raw(reduced));
            // SAFETY: guaranteed by this function's pointer contract.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(reduced.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i32` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi32(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(sum, p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `acc`, `lhs`, and
/// `rhs` must be valid for `d` `i32` elements. `acc` must be writable and must
/// not alias in a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(crate) unsafe fn pointwise_mul_acc_i32_avx512(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let pinv_v = _mm512_set1_epi32(pinv);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let l = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let r = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            let prod = mont_mul(l, r, p_v, pinv_v);
            let sum = _mm512_add_epi32(a, prod);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(sum, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i32` CRT limb.
///
/// Computes `out[i] = reduce_range(lhs[i] + rhs[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. All pointers must be valid for
/// `d` `i32` elements and `out` must be writable. `out` may equal `lhs`.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i32(out: *mut i32, lhs: *const i32, rhs: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            _mm256_storeu_si256(
                out.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(_mm256_add_epi32(a, b), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*lhs.add(i)).wrapping_add(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 add-and-reduce for one `i32` CRT limb.
///
/// Computes `out[i] = reduce_range(lhs[i] + rhs[i])`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. All pointers must be
/// valid for `d` `i32` elements and `out` must be writable. `out` may equal
/// `lhs`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn add_reduce_i32_avx512(
    out: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(_mm512_add_epi32(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*lhs.add(i)).wrapping_add(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 pointwise multiply-accumulate for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i16` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_v = _mm256_set1_epi16(p);
    let pinv_v = _mm256_set1_epi16(pinv);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_16x_i16_avx2(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi16(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(sum, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` and `other` must be valid
/// for `d` `i16` elements. `acc` must be writable and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_v = _mm256_set1_epi16(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(_mm256_add_epi16(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}
