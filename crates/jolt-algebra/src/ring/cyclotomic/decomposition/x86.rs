#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{
    balanced_decompose_coefficients_pow2_signed_kernel, BalancedDecomposePow2Params,
    BalancedSignedDigit,
};
use crate::CanonicalEncoding;

#[inline(always)]
unsafe fn cmpgt_epu32_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign = _mm256_set1_epi32(i32::MIN);
    _mm256_cmpgt_epi32(_mm256_xor_si256(lhs, sign), _mm256_xor_si256(rhs, sign))
}

/// Broadcast constants of the fp32 add-bias decomposition.
struct U32BiasConstants {
    threshold: __m256i,
    /// 32-bit halves of `H` and of `H - q mod 2^64`.
    nonnegative_low: __m256i,
    nonnegative_high: __m256i,
    negative_low: __m256i,
    negative_high: __m256i,
    mask: __m256i,
    half_b: __m256i,
}

/// Low and high 32-bit words of `x + H mod 2^64` for eight canonical residues.
#[inline(always)]
unsafe fn biased_words_avx2(values: __m256i, constants: &U32BiasConstants) -> [__m256i; 2] {
    let negative = cmpgt_epu32_avx2(values, constants.threshold);
    let low = _mm256_add_epi32(
        values,
        _mm256_blendv_epi8(constants.nonnegative_low, constants.negative_low, negative),
    );
    // All ones where the low word wrapped.
    let carry = cmpgt_epu32_avx2(values, low);
    let high = _mm256_sub_epi32(
        _mm256_blendv_epi8(
            constants.nonnegative_high,
            constants.negative_high,
            negative,
        ),
        carry,
    );
    [low, high]
}

/// Decompose `N` groups of eight coefficients starting at `base`, reading
/// `WORDS` 32-bit words of `x + H`.
///
/// Every digit plane is a shift of the biased words, so the planes carry no
/// dependency on each other. Groups of four vectors pack into one 32-byte
/// store per plane.
#[inline(always)]
unsafe fn balanced_decompose_u32_block_avx2<const N: usize, const WORDS: usize>(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
    base: usize,
    constants: &U32BiasConstants,
) {
    // Plain loops, not closures: a closure does not inherit this function's
    // target features, so the intrinsics inside it would not inline.
    let width = canonical.len();
    let zero = _mm256_setzero_si256();
    let mut words = [[zero; 2]; N];
    for (group, word) in words.iter_mut().enumerate() {
        let values = _mm256_loadu_si256(canonical.as_ptr().add(base + 8 * group).cast());
        *word = biased_words_avx2(values, constants);
    }
    let mut fields = [zero; N];
    // `packus_epi16` interleaves 128-bit lanes; this restores coefficient order.
    let lane_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for level in 0..params.levels {
        let bit = level as u32 * params.log_basis;
        for (field, &[low, high]) in fields.iter_mut().zip(&words) {
            let shifted = if WORDS == 1 || bit + params.log_basis <= 32 {
                _mm256_srl_epi32(low, _mm_cvtsi32_si128(bit as i32))
            } else if bit < 32 {
                _mm256_or_si256(
                    _mm256_srl_epi32(low, _mm_cvtsi32_si128(bit as i32)),
                    _mm256_sll_epi32(high, _mm_cvtsi32_si128(32 - bit as i32)),
                )
            } else {
                _mm256_srl_epi32(high, _mm_cvtsi32_si128(bit as i32 - 32))
            };
            // Masking first keeps the saturating packs below exact.
            *field = _mm256_and_si256(shifted, constants.mask);
        }
        let plane = out.as_mut_ptr().add(level * width + base);
        let mut group = 0;
        while group + 4 <= N {
            let bytes = _mm256_packus_epi16(
                _mm256_packus_epi32(fields[group], fields[group + 1]),
                _mm256_packus_epi32(fields[group + 2], fields[group + 3]),
            );
            let bytes = _mm256_permutevar8x32_epi32(bytes, lane_order);
            let digits = _mm256_sub_epi8(bytes, constants.half_b);
            _mm256_storeu_si256(plane.add(8 * group).cast(), digits);
            group += 4;
        }
        while group < N {
            let words = _mm_packus_epi32(
                _mm256_castsi256_si128(fields[group]),
                _mm256_extracti128_si256::<1>(fields[group]),
            );
            let bytes = _mm_packus_epi16(words, words);
            let digits = _mm_sub_epi8(bytes, _mm256_castsi256_si128(constants.half_b));
            _mm_storel_epi64(plane.add(8 * group).cast(), digits);
            group += 1;
        }
    }
}

/// AVX2 add-bias decomposition of canonical fp32 representatives.
///
/// Requires `levels * log_basis <= 64`, so the digits read only the low
/// 64 bits of `x + H`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn balanced_decompose_canonical_u32_pow2_i8_avx2(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
) {
    debug_assert!(canonical.len().is_multiple_of(8));
    debug_assert_eq!(out.len(), canonical.len() * params.levels);
    debug_assert!(params.log_basis <= 8);
    debug_assert!(params.digit_bits() <= 64);
    debug_assert!(params.q <= u32::MAX.into());

    let [nonnegative, negative] = [params.bias_nonnegative[0], params.bias_negative[0]];
    let constants = U32BiasConstants {
        threshold: _mm256_set1_epi32(params.threshold as i32),
        nonnegative_low: _mm256_set1_epi32(nonnegative as i32),
        nonnegative_high: _mm256_set1_epi32((nonnegative >> 32) as i32),
        negative_low: _mm256_set1_epi32(negative as i32),
        negative_high: _mm256_set1_epi32((negative >> 32) as i32),
        mask: _mm256_set1_epi32((1i32 << params.log_basis) - 1),
        half_b: _mm256_set1_epi8((1u32 << (params.log_basis - 1)) as u8 as i8),
    };
    // Digits inside the low word skip the high word entirely.
    if params.digit_bits() <= 32 {
        balanced_decompose_u32_blocks_avx2::<1>(canonical, out, params, &constants);
    } else {
        balanced_decompose_u32_blocks_avx2::<2>(canonical, out, params, &constants);
    }
}

#[inline(always)]
unsafe fn balanced_decompose_u32_blocks_avx2<const WORDS: usize>(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
    constants: &U32BiasConstants,
) {
    let width = canonical.len();
    let mut base = 0usize;
    while base + 32 <= width {
        balanced_decompose_u32_block_avx2::<4, WORDS>(canonical, out, params, base, constants);
        base += 32;
    }
    while base < width {
        balanced_decompose_u32_block_avx2::<1, WORDS>(canonical, out, params, base, constants);
        base += 8;
    }
}

/// AVX-512 instantiation of the add-bias balanced decomposition kernel.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub(super) unsafe fn balanced_decompose_coefficients_pow2_signed_avx512<
    F: jolt_field::Field + CanonicalEncoding,
    T: BalancedSignedDigit,
>(
    coefficients: &[F],
    out: &mut [T],
    params: &BalancedDecomposePow2Params<F>,
) {
    balanced_decompose_coefficients_pow2_signed_kernel(coefficients, out, params);
}

/// AVX2 instantiation of the add-bias balanced decomposition kernel.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn balanced_decompose_coefficients_pow2_signed_avx2<
    F: jolt_field::Field + CanonicalEncoding,
    T: BalancedSignedDigit,
>(
    coefficients: &[F],
    out: &mut [T],
    params: &BalancedDecomposePow2Params<F>,
) {
    balanced_decompose_coefficients_pow2_signed_kernel(coefficients, out, params);
}
