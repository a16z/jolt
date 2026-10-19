use std::arch::aarch64::*;

use crate::ntt::batched_four_point_eligible;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// Convert signed i16 coefficients directly into an i16 Montgomery limb.
/// Montgomery reduction accepts the signed representative directly, including
/// source magnitudes larger than the prime.
///
/// # Safety
/// `dst` and `src` must be valid for `d` elements and must not overlap.
pub(crate) unsafe fn centered_i16_to_mont_i16(
    dst: *mut i16,
    src: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
    montsq: i16,
) {
    let p_q = vdupq_n_s16(p);
    let pinv_q = vdupq_n_s16(pinv);
    let montsq_q = vdupq_n_s16(montsq);
    let mut i = 0usize;
    while i + 8 <= d {
        let coefficients = vld1q_s16(src.add(i));
        vst1q_s16(
            dst.add(i),
            mont_mul_8x_i16(coefficients, montsq_q, p_q, pinv_q),
        );
        i += 8;
    }

    let prime = NttPrime::compute(p);
    while i < d {
        *dst.add(i) = prime.from_canonical(*src.add(i)).raw();
        i += 1;
    }
}

/// 4-wide Montgomery multiply for i16 primes.
///
/// Natural 4-wide: `vmull_s16` produces `int32x4_t`.
#[inline(always)]
unsafe fn mont_mul_4x_i16(a: int16x4_t, b: int16x4_t, p: int16x4_t, pinv: int16x4_t) -> int16x4_t {
    let c = vmull_s16(a, b);
    let t = vmul_s16(vmovn_s32(c), pinv);
    let tp = vmull_s16(t, p);
    vmovn_s32(vshrq_n_s32::<16>(vsubq_s32(c, tp)))
}

/// 8-wide Montgomery multiply for i16 primes without widening lanes.
///
/// `vqdmulhq_s16(a, b) >> 1` is the exact signed high half of `a * b` for
/// these primes: saturation is impossible because one factor is always
/// strictly smaller than `2^14`. This is the NEON analogue of AVX2
/// `mulhi_epi16` and lets one vector carry eight independent residues.
#[inline(always)]
unsafe fn mont_mul_8x_i16(a: int16x8_t, b: int16x8_t, p: int16x8_t, pinv: int16x8_t) -> int16x8_t {
    let c_low = vmulq_s16(a, b);
    let c_high = vshrq_n_s16::<1>(vqdmulhq_s16(a, b));
    let t = vmulq_s16(c_low, pinv);
    let tp_high = vshrq_n_s16::<1>(vqdmulhq_s16(t, p));
    vsubq_s16(c_high, tp_high)
}

/// 8-wide range reduction for i16: `(-2p, 2p)` → `(-p, p)`.
///
/// Same comparison-first approach as i32 but on `int16x8_t`.
#[inline(always)]
unsafe fn reduce_range_8x_i16(a: int16x8_t, p: int16x8_t) -> int16x8_t {
    let zero = vdupq_n_s16(0);
    let ge_mask = vcgeq_s16(a, p);
    let after_sub = vsubq_s16(
        a,
        vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(p), ge_mask)),
    );
    let lt_mask = vcltq_s16(after_sub, zero);
    vaddq_s16(
        after_sub,
        vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(p), lt_mask)),
    )
}

/// Vectorized final two DIF stages for eight independent size-four i16 transforms.
#[inline(always)]
unsafe fn forward_dif_tail_i16<const D: usize>(
    a_ptr: *mut i16,
    fwd_twiddles: *const i16,
    p_q: int16x8_t,
    pinv_q: int16x8_t,
) {
    let tw0 = vdupq_n_s16(*fwd_twiddles);
    let tw1 = vdupq_n_s16(*fwd_twiddles.add(1));
    let tw2 = vdupq_n_s16(*fwd_twiddles.add(2));
    let mut base = 0usize;
    while base < D {
        let q = vld4q_s16(a_ptr.add(base));
        let (r0, r1, r2, r3) = (q.0, q.1, q.2, q.3);

        let s0 = reduce_range_8x_i16(vaddq_s16(r0, r2), p_q);
        let d0 = mont_mul_8x_i16(vsubq_s16(r0, r2), tw1, p_q, pinv_q);
        let s1 = reduce_range_8x_i16(vaddq_s16(r1, r3), p_q);
        let d1 = mont_mul_8x_i16(vsubq_s16(r1, r3), tw2, p_q, pinv_q);

        let o0 = reduce_range_8x_i16(vaddq_s16(s0, s1), p_q);
        let o1 = mont_mul_8x_i16(vsubq_s16(s0, s1), tw0, p_q, pinv_q);
        let o2 = reduce_range_8x_i16(vaddq_s16(d0, d1), p_q);
        let o3 = mont_mul_8x_i16(vsubq_s16(d0, d1), tw0, p_q, pinv_q);
        vst4q_s16(a_ptr.add(base), int16x8x4_t(o0, o1, o2, o3));
        base += 32;
    }
}

/// Vectorized first two DIT stages for eight independent size-four i16 transforms.
#[inline(always)]
unsafe fn inverse_dit_head_i16<const D: usize>(
    a_ptr: *mut i16,
    inv_twiddles: *const i16,
    p_q: int16x8_t,
    pinv_q: int16x8_t,
) {
    let tw0 = vdupq_n_s16(*inv_twiddles);
    let tw1 = vdupq_n_s16(*inv_twiddles.add(1));
    let tw2 = vdupq_n_s16(*inv_twiddles.add(2));
    let mut base = 0usize;
    while base < D {
        let q = vld4q_s16(a_ptr.add(base));
        let (r0, r1, r2, r3) = (q.0, q.1, q.2, q.3);

        let v1 = mont_mul_8x_i16(r1, tw0, p_q, pinv_q);
        let v3 = mont_mul_8x_i16(r3, tw0, p_q, pinv_q);
        let s0 = reduce_range_8x_i16(vaddq_s16(r0, v1), p_q);
        let d0 = reduce_range_8x_i16(vsubq_s16(r0, v1), p_q);
        let s1 = reduce_range_8x_i16(vaddq_s16(r2, v3), p_q);
        let d1 = reduce_range_8x_i16(vsubq_s16(r2, v3), p_q);

        let v2 = mont_mul_8x_i16(s1, tw1, p_q, pinv_q);
        let v3 = mont_mul_8x_i16(d1, tw2, p_q, pinv_q);
        let o0 = reduce_range_8x_i16(vaddq_s16(s0, v2), p_q);
        let o2 = reduce_range_8x_i16(vsubq_s16(s0, v2), p_q);
        let o1 = reduce_range_8x_i16(vaddq_s16(d0, v3), p_q);
        let o3 = reduce_range_8x_i16(vsubq_s16(d0, v3), p_q);
        vst4q_s16(a_ptr.add(base), int16x8x4_t(o0, o1, o2, o3));
        base += 32;
    }
}

/// NEON-accelerated forward negacyclic NTT for i16 primes.
///
/// Processes eight butterflies per iteration in the main stages and packs the
/// final two stages across eight independent size-four transforms.
pub(crate) unsafe fn forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let pinv_q = vdupq_n_s16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    // Pre-twist by psi^i
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i16;
        let mut i = 0;
        while i + 8 <= D {
            let ai = vld1q_s16(a_ptr.add(i));
            let psi = vld1q_s16(psi_ptr.add(i));
            vst1q_s16(a_ptr.add(i), mont_mul_8x_i16(ai, psi, p_q, pinv_q));
            i += 8;
        }
    }

    // DIF butterfly stages
    let mut len = D / 2;
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let u = vld1q_s16(a_ptr.add(start + j));
                    let v = vld1q_s16(a_ptr.add(start + j + len));
                    let w = vld1q_s16(tw_ptr.add(twiddle_base + j));
                    vst1q_s16(
                        a_ptr.add(start + j),
                        reduce_range_8x_i16(vaddq_s16(u, v), p_q),
                    );
                    vst1q_s16(
                        a_ptr.add(start + j + len),
                        mont_mul_8x_i16(vsubq_s16(u, v), w, p_q, pinv_q),
                    );
                    j += 8;
                }
            } else {
                let mut j = 0;
                while j < len {
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v = vld1_s16(a_ptr.add(start + j + len));
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);

                    // reduce_range on 4-wide i16 (use 8-wide by padding)
                    let sum_q = vcombine_s16(sum, vdup_n_s16(0));
                    let sum_reduced = vget_low_s16(reduce_range_8x_i16(sum_q, p_q));

                    let diff_mul_w = mont_mul_4x_i16(diff, w, p_d, pinv_d);
                    vst1_s16(a_ptr.add(start + j), sum_reduced);
                    vst1_s16(a_ptr.add(start + j + len), diff_mul_w);
                    j += 4;
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    if batched_four_point_eligible::<D>(8) {
        forward_dif_tail_i16::<D>(a_ptr, tw.fwd_twiddles.as_ptr() as *const i16, p_q, pinv_q);
    } else {
        while len > 0 {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
                start += 2 * len;
            }
            len /= 2;
        }
    }

    // Final reduce_range pass
    reduce_range_in_place_i16(a, p_q);
}

/// NEON-accelerated inverse negacyclic NTT for i16 primes.
pub(crate) unsafe fn inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let pinv_q = vdupq_n_s16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = if batched_four_point_eligible::<D>(8) {
        inverse_dit_head_i16::<D>(a_ptr, tw.inv_twiddles.as_ptr() as *const i16, p_q, pinv_q);
        4usize
    } else {
        1usize
    };
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let w = vld1q_s16(tw_ptr.add(twiddle_base + j));
                    let u = vld1q_s16(a_ptr.add(start + j));
                    let v = mont_mul_8x_i16(vld1q_s16(a_ptr.add(start + j + len)), w, p_q, pinv_q);
                    vst1q_s16(
                        a_ptr.add(start + j),
                        reduce_range_8x_i16(vaddq_s16(u, v), p_q),
                    );
                    vst1q_s16(
                        a_ptr.add(start + j + len),
                        reduce_range_8x_i16(vsubq_s16(u, v), p_q),
                    );
                    j += 8;
                }
            } else if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v_raw = vld1_s16(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i16(v_raw, w, p_d, pinv_d);
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let reduced = reduce_range_8x_i16(vcombine_s16(sum, diff), p_q);
                    vst1_s16(a_ptr.add(start + j), vget_low_s16(reduced));
                    vst1_s16(a_ptr.add(start + j + len), vget_high_s16(reduced));
                    j += 4;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    // Fused D^{-1} * psi^{-i} untwist
    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i16;
        let mut i = 0;
        while i + 8 <= D {
            let ai = vld1q_s16(a_ptr.add(i));
            let f = vld1q_s16(fused_ptr.add(i));
            vst1q_s16(a_ptr.add(i), mont_mul_8x_i16(ai, f, p_q, pinv_q));
            i += 8;
        }
    }
}

/// NEON-accelerated forward cyclic NTT for i16.
pub(crate) unsafe fn forward_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v = vld1_s16(a_ptr.add(start + j + len));
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let sum_q = vcombine_s16(sum, vdup_n_s16(0));
                    vst1_s16(
                        a_ptr.add(start + j),
                        vget_low_s16(reduce_range_8x_i16(sum_q, p_q)),
                    );
                    vst1_s16(
                        a_ptr.add(start + j + len),
                        mont_mul_4x_i16(diff, w, p_d, pinv_d),
                    );
                    j += 4;
                }
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }
    reduce_range_in_place_i16(a, p_q);
}

/// NEON-accelerated inverse cyclic NTT for i16.
pub(crate) unsafe fn inverse_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v_raw = vld1_s16(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i16(v_raw, w, p_d, pinv_d);
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let reduced = reduce_range_8x_i16(vcombine_s16(sum, diff), p_q);
                    vst1_s16(a_ptr.add(start + j), vget_low_s16(reduced));
                    vst1_s16(a_ptr.add(start + j + len), vget_high_s16(reduced));
                    j += 4;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    // D^{-1} scaling
    {
        let d_inv = tw.d_inv;
        let d_inv_d = vdup_n_s16(d_inv.raw());
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1_s16(a_ptr.add(i));
            vst1_s16(a_ptr.add(i), mont_mul_4x_i16(ai, d_inv_d, p_d, pinv_d));
            i += 4;
        }
    }
}

/// 8-wide pointwise multiply-accumulate for a single CRT limb (i16).
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_d = vdup_n_s16(p);
    let pinv_d = vdup_n_s16(pinv);
    let p_q = vdupq_n_s16(p);
    let pinv_q = vdupq_n_s16(pinv);
    let mut i = 0;
    while i + 8 <= d {
        let a = vld1q_s16(acc.add(i));
        let l = vld1q_s16(lhs.add(i));
        let r = vld1q_s16(rhs.add(i));
        let prod = mont_mul_8x_i16(l, r, p_q, pinv_q);
        let sum = vaddq_s16(a, prod);
        vst1q_s16(acc.add(i), reduce_range_8x_i16(sum, p_q));
        i += 8;
    }
    while i + 4 <= d {
        let a = vld1_s16(acc.add(i));
        let l = vld1_s16(lhs.add(i));
        let r = vld1_s16(rhs.add(i));
        let prod = mont_mul_4x_i16(l, r, p_d, pinv_d);
        let sum = vadd_s16(a, prod);
        let sum_q = vcombine_s16(sum, vdup_n_s16(0));
        vst1_s16(acc.add(i), vget_low_s16(reduce_range_8x_i16(sum_q, p_q)));
        i += 4;
    }
}

/// Return whether every signed value lies in `[-bound, bound)`.
///
/// # Safety
///
/// `values` must be valid for `len` readable elements and `bound` must be
/// positive.
pub(crate) unsafe fn all_i16_in_balanced_range(values: *const i16, len: usize, bound: i16) -> bool {
    let lower_q = vdupq_n_s16(-bound);
    let upper_q = vdupq_n_s16(bound);
    let mut i = 0usize;
    while i + 8 <= len {
        let value = vld1q_s16(values.add(i));
        let below = vcltq_s16(value, lower_q);
        let above = vcgeq_s16(value, upper_q);
        if vmaxvq_u16(vorrq_u16(below, above)) != 0 {
            return false;
        }
        i += 8;
    }
    while i < len {
        let value = *values.add(i);
        if value < -bound || value >= bound {
            return false;
        }
        i += 1;
    }
    true
}

/// 8-wide add-and-reduce for a single CRT limb (i16).
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements, properly aligned for
/// NEON loads/stores, and must not alias in a way that violates Rust's
/// mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_q = vdupq_n_s16(p);
    let mut i = 0;
    while i + 8 <= d {
        let a = vld1q_s16(acc.add(i));
        let b = vld1q_s16(other.add(i));
        vst1q_s16(acc.add(i), reduce_range_8x_i16(vaddq_s16(a, b), p_q));
        i += 8;
    }
    while i + 4 <= d {
        let a = vld1_s16(acc.add(i));
        let b = vld1_s16(other.add(i));
        let sum_q = vcombine_s16(vadd_s16(a, b), vdup_n_s16(0));
        vst1_s16(acc.add(i), vget_low_s16(reduce_range_8x_i16(sum_q, p_q)));
        i += 4;
    }
}

/// In-place reduce_range over a full i16 array.
unsafe fn reduce_range_in_place_i16<const D: usize>(a: &mut [MontCoeff<i16>; D], p_q: int16x8_t) {
    let ptr = a.as_mut_ptr() as *mut i16;
    let mut i = 0;
    while i + 8 <= D {
        let val = vld1q_s16(ptr.add(i));
        vst1q_s16(ptr.add(i), reduce_range_8x_i16(val, p_q));
        i += 8;
    }
    while i + 4 <= D {
        let val = vld1_s16(ptr.add(i));
        let padded = vcombine_s16(val, vdup_n_s16(0));
        vst1_s16(ptr.add(i), vget_low_s16(reduce_range_8x_i16(padded, p_q)));
        i += 4;
    }
}
