//! NEON reduction of canonical field coefficients into i32 CRT residues.

use std::arch::aarch64::*;

use crate::ntt::field_limbs::{FieldLimbScales, FieldModulusLimbs, MAX_FIELD_LIMBS};
use crate::ntt::prime::MontCoeff;

/// NEON form of [`crate::ntt::field_limbs::field_residues`] for i32 primes.
///
/// `vld4q_u32` transposes four coefficients into their 32-bit words, which are
/// compared against `⌊q/2⌋` and cut into signed 26-bit limbs. The limbs stay
/// in registers while every prime accumulates them with widening lane
/// multiplies and applies one signed Montgomery reduction.
///
/// # Safety
///
/// `canonical.len()` must be a multiple of four, `start + canonical.len()`
/// must not exceed `D`, `L` must cover `modulus`, and every coefficient must
/// be below that modulus.
pub(crate) unsafe fn field_residues_i32<const L: usize, const K: usize, const D: usize>(
    out: &mut [[MontCoeff<i32>; D]; K],
    start: usize,
    canonical: &[u128],
    modulus: &FieldModulusLimbs,
    scales: &[FieldLimbScales; K],
) {
    let mask = vdupq_n_u32((1 << 26) - 1);
    let half = [
        modulus.half as u32,
        (modulus.half >> 32) as u32,
        (modulus.half >> 64) as u32,
        (modulus.half >> 96) as u32,
    ]
    .map(|word| vdupq_n_u32(word));
    let q = modulus.limbs.map(|limb| vdupq_n_s32(limb));
    let src = canonical.as_ptr().cast::<u32>();
    for block in (0..canonical.len()).step_by(4) {
        // SAFETY: the length is a multiple of four, so the 64 bytes of
        // coefficients `block..block + 4` are in bounds.
        let words = unsafe { vld4q_u32(src.add(4 * block)) };
        let (w0, w1, w2, w3) = (words.0, words.1, words.2, words.3);
        // Unsigned lexicographic `c > ⌊q/2⌋`: an equal word defers to the
        // words below it. Coefficients lie below `2^(26 L)`, so word `j >= L`
        // is zero on both sides.
        let defer =
            |center, word, half| vbslq_u32(vceqq_u32(word, half), center, vcgtq_u32(word, half));
        let mut center = vcgtq_u32(w0, half[0]);
        if L > 1 {
            center = defer(center, w1, half[1]);
        }
        if L > 2 {
            center = defer(center, w2, half[2]);
        }
        if L > 3 {
            center = defer(center, w3, half[3]);
        }
        let center = vreinterpretq_s32_u32(center);
        let slices = [
            vandq_u32(w0, mask),
            vandq_u32(vsliq_n_u32::<6>(vshrq_n_u32::<26>(w0), w1), mask),
            vandq_u32(vsliq_n_u32::<12>(vshrq_n_u32::<20>(w1), w2), mask),
            vandq_u32(vsliq_n_u32::<18>(vshrq_n_u32::<14>(w2), w3), mask),
            vshrq_n_u32::<8>(w3),
        ];
        let limbs: [int32x4_t; MAX_FIELD_LIMBS] = std::array::from_fn(|j| {
            vsubq_s32(vreinterpretq_s32_u32(slices[j]), vandq_s32(q[j], center))
        });

        for (residues, scales) in out.iter_mut().zip(scales) {
            // SAFETY: `scales.scales` holds five lanes.
            let (low_scales, top_scale) = unsafe {
                (
                    vld1q_s32(scales.scales.as_ptr()),
                    vld1q_dup_s32(scales.scales.as_ptr().add(4)),
                )
            };
            let mut lanes01 = vmull_laneq_s32::<0>(vget_low_s32(limbs[0]), low_scales);
            let mut lanes23 = vmull_high_laneq_s32::<0>(limbs[0], low_scales);
            if L > 1 {
                lanes01 = vmlal_laneq_s32::<1>(lanes01, vget_low_s32(limbs[1]), low_scales);
                lanes23 = vmlal_high_laneq_s32::<1>(lanes23, limbs[1], low_scales);
            }
            if L > 2 {
                lanes01 = vmlal_laneq_s32::<2>(lanes01, vget_low_s32(limbs[2]), low_scales);
                lanes23 = vmlal_high_laneq_s32::<2>(lanes23, limbs[2], low_scales);
            }
            if L > 3 {
                lanes01 = vmlal_laneq_s32::<3>(lanes01, vget_low_s32(limbs[3]), low_scales);
                lanes23 = vmlal_high_laneq_s32::<3>(lanes23, limbs[3], low_scales);
            }
            if L > 4 {
                lanes01 = vmlal_laneq_s32::<0>(lanes01, vget_low_s32(limbs[4]), top_scale);
                lanes23 = vmlal_high_laneq_s32::<0>(lanes23, limbs[4], top_scale);
            }
            let p = vdupq_n_s32(scales.p);
            let m = vmulq_s32(
                vuzp1q_s32(
                    vreinterpretq_s32_s64(lanes01),
                    vreinterpretq_s32_s64(lanes23),
                ),
                vdupq_n_s32(scales.pinv),
            );
            lanes01 = vmlsl_s32(lanes01, vget_low_s32(m), vget_low_s32(p));
            lanes23 = vmlsl_high_s32(lanes23, m, p);
            let reduced = vuzp2q_s32(
                vreinterpretq_s32_s64(lanes01),
                vreinterpretq_s32_s64(lanes23),
            );
            // SAFETY: `start + block + 4 <= D`, and MontCoeff<i32> is
            // transparent.
            unsafe {
                vst1q_s32(
                    residues.as_mut_ptr().add(start + block).cast::<i32>(),
                    reduced,
                );
            }
        }
    }
}
