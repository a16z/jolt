use std::arch::aarch64::*;

use super::BalancedDecomposePow2Params;
use crate::CanonicalEncoding;

/// Broadcast constants of the fp32 add-bias decomposition.
struct U32BiasConstants {
    threshold: uint32x4_t,
    /// 32-bit halves of `H` and of `H - q mod 2^64`.
    nonnegative_low: uint32x4_t,
    nonnegative_high: uint32x4_t,
    negative_low: uint32x4_t,
    negative_high: uint32x4_t,
    mask: uint8x16_t,
    half_b: uint8x16_t,
}

/// Low and high 32-bit words of `x + H mod 2^64` for four canonical residues.
#[inline(always)]
unsafe fn biased_words_neon(values: uint32x4_t, constants: &U32BiasConstants) -> [uint32x4_t; 2] {
    let negative = vcgtq_u32(values, constants.threshold);
    let low = vaddq_u32(
        values,
        vbslq_u32(negative, constants.negative_low, constants.nonnegative_low),
    );
    // All ones where the low word wrapped.
    let carry = vcltq_u32(low, values);
    let high = vsubq_u32(
        vbslq_u32(
            negative,
            constants.negative_high,
            constants.nonnegative_high,
        ),
        carry,
    );
    [low, high]
}

/// Decompose `N` groups of four coefficients starting at `base`, reading
/// `WORDS` 32-bit words of `x + H`.
///
/// Every digit plane is a shift of the biased words, so the planes carry no
/// dependency on each other. Groups of four vectors narrow into one 16-byte
/// store per plane.
#[inline(always)]
unsafe fn balanced_decompose_u32_block_neon<const N: usize, const WORDS: usize>(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
    base: usize,
    constants: &U32BiasConstants,
) {
    let width = canonical.len();
    let zero = vdupq_n_u32(0);
    let mut words = [[zero; 2]; N];
    for (group, word) in words.iter_mut().enumerate() {
        *word = biased_words_neon(
            vld1q_u32(canonical.as_ptr().add(base + 4 * group)),
            constants,
        );
    }
    let mut fields = [zero; N];
    for level in 0..params.levels {
        let bit = level as u32 * params.log_basis;
        // USHL shifts right for negative counts, so `high_shift` also serves
        // fields above bit 32.
        let low_shift = vdupq_n_s32(-(bit as i32));
        let high_shift = vdupq_n_s32(32 - bit as i32);
        for (field, &[low, high]) in fields.iter_mut().zip(&words) {
            *field = if WORDS == 1 || bit + params.log_basis <= 32 {
                vshlq_u32(low, low_shift)
            } else if bit >= 32 {
                vshlq_u32(high, high_shift)
            } else {
                vorrq_u32(vshlq_u32(low, low_shift), vshlq_u32(high, high_shift))
            };
        }
        let plane = out.as_mut_ptr().add(level * width + base);
        let mut group = 0;
        while group + 4 <= N {
            let halves = [
                vuzp1q_u16(
                    vreinterpretq_u16_u32(fields[group]),
                    vreinterpretq_u16_u32(fields[group + 1]),
                ),
                vuzp1q_u16(
                    vreinterpretq_u16_u32(fields[group + 2]),
                    vreinterpretq_u16_u32(fields[group + 3]),
                ),
            ];
            let bytes = vuzp1q_u8(
                vreinterpretq_u8_u16(halves[0]),
                vreinterpretq_u8_u16(halves[1]),
            );
            let digits = vsubq_u8(vandq_u8(bytes, constants.mask), constants.half_b);
            vst1q_u8(plane.add(4 * group).cast(), digits);
            group += 4;
        }
        while group < N {
            let halves = vmovn_u32(fields[group]);
            let bytes = vmovn_u16(vcombine_u16(halves, vdup_n_u16(0)));
            let digits = vsub_u8(
                vand_u8(bytes, vget_low_u8(constants.mask)),
                vget_low_u8(constants.half_b),
            );
            vst1_lane_u32(plane.add(4 * group).cast(), vreinterpret_u32_u8(digits), 0);
            group += 1;
        }
    }
}

/// NEON add-bias decomposition of canonical fp32 representatives.
///
/// Requires `levels * log_basis <= 64`, so the digits read only the low
/// 64 bits of `x + H`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn balanced_decompose_canonical_u32_pow2_i8_neon(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
) {
    debug_assert!(canonical.len().is_multiple_of(4));
    debug_assert_eq!(out.len(), canonical.len() * params.levels);
    debug_assert!(params.log_basis <= 8);
    debug_assert!(params.digit_bits() <= 64);
    debug_assert!(params.q <= u32::MAX.into());

    let [nonnegative, negative] = [params.bias_nonnegative[0], params.bias_negative[0]];
    let constants = U32BiasConstants {
        threshold: vdupq_n_u32(params.threshold as u32),
        nonnegative_low: vdupq_n_u32(nonnegative as u32),
        nonnegative_high: vdupq_n_u32((nonnegative >> 32) as u32),
        negative_low: vdupq_n_u32(negative as u32),
        negative_high: vdupq_n_u32((negative >> 32) as u32),
        mask: vdupq_n_u8(((1u32 << params.log_basis) - 1) as u8),
        half_b: vdupq_n_u8((1u32 << (params.log_basis - 1)) as u8),
    };
    // Digits inside the low word skip the high word entirely.
    if params.digit_bits() <= 32 {
        balanced_decompose_u32_blocks_neon::<1>(canonical, out, params, &constants);
    } else {
        balanced_decompose_u32_blocks_neon::<2>(canonical, out, params, &constants);
    }
}

#[inline(always)]
unsafe fn balanced_decompose_u32_blocks_neon<const WORDS: usize>(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<impl jolt_field::Field + CanonicalEncoding>,
    constants: &U32BiasConstants,
) {
    let width = canonical.len();
    let mut base = 0usize;
    while base + 16 <= width {
        balanced_decompose_u32_block_neon::<4, WORDS>(canonical, out, params, base, constants);
        base += 16;
    }
    while base < width {
        balanced_decompose_u32_block_neon::<1, WORDS>(canonical, out, params, base, constants);
        base += 4;
    }
}
