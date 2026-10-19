use std::arch::aarch64::*;

use super::twiddles::{BarrettConstant, BarrettTable, BarrettTwiddles};
use crate::ntt::butterfly::{self, NttTwiddles};
use crate::ntt::prime::{MontCoeff, NttPrime, I32_LAZY_DOT_BATCH};
use crate::ntt::NttKernelPlan;

/// Smallest degree handled by the vector transforms. The negacyclic head
/// consumes eight coefficients from each quarter, and both four-point
/// deinterleaved passes consume sixteen coefficients per step. Smaller
/// degrees use the scalar transforms.
const MIN_VECTOR_DEGREE: usize = 32;

/// True 4-wide signed Montgomery multiply for i32 primes.
///
/// Computes `a * b * R^{-1} mod p` (R = 2^32) for all four lanes at once,
/// matching the scalar [`NttPrime::mont_mul_raw`] convention exactly
/// (`pinv = p^{-1} mod 2^32`, signed). Result lies in `(-p, p)`.
///
/// Uses the high-multiply formulation from Becker–Hwang–Kannwischer–Yang
/// ("Neon NTT"): `vqdmulhq_s32` yields `(2·a·b) >> 32` with a single 4-lane
/// multiply, so the reduction needs two `vqdmulhq_s32`, two `vmulq_s32`, and one
/// halving subtract instead of two 2-lane `vmull_s32` widening chains.
/// Every NTT prime here is `< 2^30`, so neither `2·a·b` nor `2·m·p` saturates
/// an `int32x4_t` after the `>> 32`.
#[inline(always)]
unsafe fn mont_mul_4x_i32(
    a: int32x4_t,
    b: int32x4_t,
    p_q: int32x4_t,
    pinv_q: int32x4_t,
) -> int32x4_t {
    // top = (2·a·b) >> 32, the doubled high word of the full product.
    let top = vqdmulhq_s32(a, b);
    // lo = (a·b) mod 2^32 = low word of the product.
    let lo = vmulq_s32(a, b);
    // m = lo · pinv mod 2^32 (signed), the Montgomery quotient digit.
    let m = vmulq_s32(lo, pinv_q);
    // sub = (2·m·p) >> 32, the doubled high word of m·p.
    let sub = vqdmulhq_s32(m, p_q);
    // (top - sub) >> 1 = (a·b - m·p) >> 32, exact since the low 32 bits cancel.
    vhsubq_s32(top, sub)
}

/// Broadcast modulus constants for a prime `3 <= p < 2^30`.
#[derive(Clone, Copy)]
struct Modulus {
    p: int32x4_t,
    two_p: int32x4_t,
    neg_p: int32x4_t,
    neg_two_p: int32x4_t,
    /// `round(2^31 / p)`, consumed by [`centered_reduce_4x_i32`].
    reciprocal: int32x4_t,
}

impl Modulus {
    #[inline(always)]
    unsafe fn new(p: i32) -> Self {
        let reciprocal = ((1i64 << 31) + i64::from(p) / 2) / i64::from(p);
        Self {
            p: vdupq_n_s32(p),
            two_p: vdupq_n_s32(2 * p),
            neg_p: vdupq_n_s32(-p),
            neg_two_p: vdupq_n_s32(-2 * p),
            reciprocal: vdupq_n_s32(reciprocal as i32),
        }
    }
}

/// 4-wide canonical reduction for i32: maps `(-2p, 2p)` → `[0, p)`.
///
/// Viewed as unsigned lanes, `min(x, x + 2p)` picks the representative in
/// `[0, 2p)` and `min(y, y - p)` then picks the one in `[0, p)`: since
/// `4p < 2^32`, the rejected candidate always wraps above both. Four
/// instructions, with no compare masks.
#[inline(always)]
unsafe fn reduce_range_4x_i32(x: int32x4_t, p: int32x4_t, two_p: int32x4_t) -> int32x4_t {
    let x = vreinterpretq_u32_s32(x);
    let y = vminq_u32(x, vaddq_u32(x, vreinterpretq_u32_s32(two_p)));
    vreinterpretq_s32_u32(vminq_u32(y, vsubq_u32(y, vreinterpretq_u32_s32(p))))
}

/// Folds a sum or difference into `[0, 2p)`, viewing lanes as unsigned.
///
/// Returns `min(t, t + c)`. With `c = -2p` it accepts `t` in `[0, 4p)`, with
/// `c = 2p` it accepts `t` in `(-2p, 2p)`, and with `c = -p` it maps `[0, 2p)`
/// to `[0, p)`. Since `4p < 2^32`, the rejected candidate always wraps above
/// the kept one. Two ALU instructions, with no multiply.
#[inline(always)]
unsafe fn fold_4x_i32(t: int32x4_t, c: int32x4_t) -> int32x4_t {
    let t = vreinterpretq_u32_s32(t);
    vreinterpretq_s32_u32(vminq_u32(t, vaddq_u32(t, vreinterpretq_u32_s32(c))))
}

/// Centered reduction of `|x| < 2p`, for a modulus `3 <= p < 2^30`.
///
/// Let `v = round(2^31 / p)` and `qhat = round(x*v / 2^31)`. Then
/// `|qhat - x/p| <= 1/2 + |x|/2^32 < 1`, so `x - qhat*p` is congruent
/// to `x` and lies strictly in `(-p, p)`. Also `|qhat| <= 2`, hence the
/// product and result fit i32. `sqrdmulh` cannot reach its saturation case.
/// The same bound keeps the result in `(-p, p)` for every `|x| < 2^31`.
#[inline(always)]
pub(super) unsafe fn centered_reduce_4x_i32(
    x: int32x4_t,
    p: int32x4_t,
    reciprocal: int32x4_t,
) -> int32x4_t {
    vmlsq_s32(x, vqrdmulhq_s32(x, reciprocal), p)
}

/// A constant multiplier in Barrett form, broadcast or loaded per lane.
///
/// `value` is the centered plain residue `w` and `quotient` is
/// `w' = round(w * 2^31 / p)`; see [`BarrettTwiddles`].
#[derive(Clone, Copy)]
struct Multiplier {
    value: int32x4_t,
    quotient: int32x4_t,
}

impl Multiplier {
    /// Load entries `index..index + 4` of `table`.
    #[inline(always)]
    unsafe fn load<const D: usize>(table: &BarrettTable<i32, D>, index: usize) -> Self {
        Self {
            value: vld1q_s32(table.values.as_ptr().add(index)),
            quotient: vld1q_s32(table.quotients.as_ptr().add(index)),
        }
    }

    /// Broadcast entry `index` of `table`.
    #[inline(always)]
    unsafe fn broadcast<const D: usize>(table: &BarrettTable<i32, D>, index: usize) -> Self {
        Self {
            value: vld1q_dup_s32(table.values.as_ptr().add(index)),
            quotient: vld1q_dup_s32(table.quotients.as_ptr().add(index)),
        }
    }

    #[inline(always)]
    unsafe fn splat(constant: BarrettConstant<i32>) -> Self {
        Self {
            value: vdupq_n_s32(constant.value),
            quotient: vdupq_n_s32(constant.quotient),
        }
    }

    /// `x * w mod p` in `(-p, p)` for any `|x| < 2^31`.
    ///
    /// With `q = round(x * w' / 2^31)`, `|x*w/p - q| <= 1/2 + |x|/2^32 < 1`
    /// because `|w' - w*2^31/p| <= 1/2`. The exact result `x*w - q*p` fits
    /// i32, so computing it with wrapping `mul` and `mls` is exact. A plain
    /// `w` preserves whatever Montgomery scaling `x` carries.
    #[inline(always)]
    unsafe fn mul(self, x: int32x4_t, p: int32x4_t) -> int32x4_t {
        vmlsq_s32(vmulq_s32(x, self.value), vqrdmulhq_s32(x, self.quotient), p)
    }

    /// `x * w mod p` in `(0, 2p)` for any `|x| < 2^31`: [`Self::mul`] plus
    /// `p`, with the addition folded into a multiply-accumulate.
    #[inline(always)]
    unsafe fn mul_biased(self, x: int32x4_t, p: int32x4_t) -> int32x4_t {
        vmlsq_s32(
            vmlaq_s32(p, x, self.value),
            vqrdmulhq_s32(x, self.quotient),
            p,
        )
    }
}

/// Gentleman–Sande butterfly `(u + v, w (u - v))` with outputs in `[0, 2p)`.
///
/// Inputs lie in `[0, 2p)` with `fold = -2p`, or in `(-p, p)` with
/// `fold = 2p`. Only the difference is multiplied; the sum is folded with ALU
/// instructions, which keeps the multiply pipes for the twiddle product.
#[inline(always)]
unsafe fn dif_butterfly(
    u: int32x4_t,
    v: int32x4_t,
    w: Multiplier,
    m: Modulus,
    fold: int32x4_t,
) -> (int32x4_t, int32x4_t) {
    (
        fold_4x_i32(vaddq_s32(u, v), fold),
        w.mul_biased(vsubq_s32(u, v), m.p),
    )
}

/// Cooley–Tukey butterfly `(u + w v, u - w v)` for `u` in `[0, 2p)` and any
/// `v`. Both outputs lie in `[0, 2p)`, and only the twiddle product uses the
/// multiply pipes.
#[inline(always)]
unsafe fn dit_butterfly(
    u: int32x4_t,
    v: int32x4_t,
    w: Multiplier,
    m: Modulus,
) -> (int32x4_t, int32x4_t) {
    let v = w.mul_biased(v, m.p);
    (
        fold_4x_i32(vaddq_s32(u, v), m.neg_two_p),
        fold_4x_i32(vsubq_s32(u, v), m.two_p),
    )
}

/// First forward pass: DIF stages `D/2` and `D/4` with a per-output scaling
/// of stage `D/2`, in one load and store of the array.
///
/// With `NEGACYCLIC` and `i = psi^(D/2)`, stage `D/2` of the twisted input
/// `psi^j x_j` is `psi^j (x_j + i x_{j+D/2})` and
/// `psi^j w_j (x_j - i x_{j+D/2})`, so the twist costs one multiply by `i`
/// per pair instead of a separate pass. `twist` then holds
/// `[psi^j | psi^j w_j]`, possibly scaled by `R` so that plain integer inputs
/// enter Montgomery form here. Without `NEGACYCLIC`, stage `D/2` is the plain
/// cyclic butterfly and `twist` holds `[R | R w_j]`, which converts plain
/// integer inputs at the cost of one extra multiply per pair.
///
/// `load(i)` returns lanes `i..i + 8` of the input. With `CENTER`, the
/// operands that are added without a preceding multiply are first reduced,
/// so any `|x| < 2^31` input is accepted; otherwise inputs must lie in
/// `(-p, p)`; without `NEGACYCLIC` the pair sums must as well. Outputs lie
/// in `[0, 2p)`.
#[inline(always)]
unsafe fn forward_head<const D: usize, const CENTER: bool, const NEGACYCLIC: bool>(
    a: *mut i32,
    barrett: &BarrettTwiddles<i32, D>,
    twist: &BarrettTable<i32, D>,
    m: Modulus,
    load: impl Fn(usize) -> [int32x4_t; 2],
) {
    let quarter = D / 4;
    let half = D / 2;
    let root = Multiplier::splat(barrett.quarter_root);
    let center = |x: int32x4_t| {
        if CENTER {
            centered_reduce_4x_i32(x, m.p, m.reciprocal)
        } else {
            x
        }
    };
    let mut j = 0usize;
    while j < quarter {
        let x0 = load(j);
        let x1 = load(j + quarter);
        let x2 = load(j + half);
        let x3 = load(j + half + quarter);
        for lane in 0..2 {
            let o = j + 4 * lane;
            let (x0, x1) = (center(x0[lane]), center(x1[lane]));
            let (t0, t1) = if NEGACYCLIC {
                (root.mul(x2[lane], m.p), root.mul(x3[lane], m.p))
            } else {
                (center(x2[lane]), center(x3[lane]))
            };
            let y0 = Multiplier::load(twist, o).mul(vaddq_s32(x0, t0), m.p);
            let y2 = Multiplier::load(twist, half + o).mul(vsubq_s32(x0, t0), m.p);
            let y1 = Multiplier::load(twist, quarter + o).mul(vaddq_s32(x1, t1), m.p);
            let y3 = Multiplier::load(twist, half + quarter + o).mul(vsubq_s32(x1, t1), m.p);

            let w = Multiplier::load(&barrett.fwd, quarter - 1 + o);
            let (z0, z1) = dif_butterfly(y0, y1, w, m, m.two_p);
            let (z2, z3) = dif_butterfly(y2, y3, w, m, m.two_p);
            vst1q_s32(a.add(o), z0);
            vst1q_s32(a.add(o + quarter), z1);
            vst1q_s32(a.add(o + half), z2);
            vst1q_s32(a.add(o + half + quarter), z3);
        }
        j += 8;
    }
}

/// DIF stages `len, len/2, ..., 1`, with canonical `[0, p)` outputs in
/// bit-reversed order.
///
/// Inputs lie in `[0, 2p)`, or in `(-p, p)` with `centered`; every stage
/// output lies in `[0, 2p)`. Stages are paired into radix-4 passes so each
/// pass loads and stores the array once for two stages; the last two stages
/// run deinterleaved in [`forward_dif_tail`]. Requires `D >= 32` and
/// `4 <= len <= D/2`.
#[inline(always)]
unsafe fn forward_dif_stages<const D: usize>(
    a: *mut i32,
    fwd: &BarrettTable<i32, D>,
    m: Modulus,
    mut len: usize,
    centered: bool,
) {
    let mut fold = if centered { m.two_p } else { m.neg_two_p };
    while len >= 8 {
        let half = len / 2;
        let mut start = 0usize;
        while start < D {
            let unit = |j: usize| {
                let x = start + j;
                let u0 = vld1q_s32(a.add(x));
                let u1 = vld1q_s32(a.add(x + half));
                let u2 = vld1q_s32(a.add(x + len));
                let u3 = vld1q_s32(a.add(x + len + half));
                let (v0, v2) = dif_butterfly(u0, u2, Multiplier::load(fwd, len - 1 + j), m, fold);
                let (v1, v3) =
                    dif_butterfly(u1, u3, Multiplier::load(fwd, len - 1 + j + half), m, fold);
                let w = Multiplier::load(fwd, half - 1 + j);
                let (z0, z1) = dif_butterfly(v0, v1, w, m, m.neg_two_p);
                let (z2, z3) = dif_butterfly(v2, v3, w, m, m.neg_two_p);
                vst1q_s32(a.add(x), z0);
                vst1q_s32(a.add(x + half), z1);
                vst1q_s32(a.add(x + len), z2);
                vst1q_s32(a.add(x + len + half), z3);
            };
            // Two independent units per iteration overlap their twiddle
            // multiply chains.
            if half >= 8 {
                let mut j = 0usize;
                while j < half {
                    unit(j);
                    unit(j + 4);
                    j += 8;
                }
            } else {
                unit(0);
            }
            start += 2 * len;
        }
        len /= 4;
        fold = m.neg_two_p;
    }
    if len == 4 {
        let w = Multiplier::load(fwd, 3);
        let mut start = 0usize;
        while start < D {
            let u = vld1q_s32(a.add(start));
            let v = vld1q_s32(a.add(start + 4));
            let (sum, diff) = dif_butterfly(u, v, w, m, fold);
            vst1q_s32(a.add(start), sum);
            vst1q_s32(a.add(start + 4), diff);
            start += 8;
        }
    }
    forward_dif_tail::<D>(a, fwd, m);
}

/// Final two DIF stages (`len = 2`, then `len = 1`) on inputs in `[0, 2p)`,
/// with canonical output.
///
/// A stride-4 `vld4q` deinterleave places element `e` of four independent
/// size-4 sub-transforms in the lanes of `r_e`, so both stages run 4-wide
/// with broadcast twiddles, and `vst4q` re-interleaves. Stage `len = 1` and
/// the first `len = 2` butterfly have twiddle 1, leaving one multiply.
#[inline(always)]
unsafe fn forward_dif_tail<const D: usize>(a: *mut i32, fwd: &BarrettTable<i32, D>, m: Modulus) {
    let w = Multiplier::broadcast(fwd, 2);
    let mut base = 0usize;
    while base < D {
        let int32x4x4_t(r0, r1, r2, r3) = vld4q_s32(a.add(base));

        // len = 2: butterflies (e0, e2) with twiddle 1 and (e1, e3) with w.
        let s0 = fold_4x_i32(vaddq_s32(r0, r2), m.neg_two_p);
        let d0 = fold_4x_i32(vsubq_s32(r0, r2), m.two_p);
        let (s1, d1) = dif_butterfly(r1, r3, w, m, m.neg_two_p);

        // len = 1: butterflies (e0, e1) and (e2, e3) with twiddle 1.
        let canonical = |t, fold| fold_4x_i32(fold_4x_i32(t, fold), m.neg_p);
        let o0 = canonical(vaddq_s32(s0, s1), m.neg_two_p);
        let o1 = canonical(vsubq_s32(s0, s1), m.two_p);
        let o2 = canonical(vaddq_s32(d0, d1), m.neg_two_p);
        let o3 = canonical(vsubq_s32(d0, d1), m.two_p);

        vst4q_s32(a.add(base), int32x4x4_t(o0, o1, o2, o3));
        base += 16;
    }
}

/// Inverse DIT stages `1, 2, ..., D/2` on inputs in `(-p, p)`, multiplying
/// output `i` by `scale(i)` inside the last stage. Outputs lie in `(-p, p)`.
///
/// The first two stages run deinterleaved, middle stages are paired into
/// radix-4 passes (with one radix-8 pass when their count is odd and
/// `D >= 512`), and the last two stages form one pass together with the
/// scaling. Requires `D >= 32`.
#[inline(always)]
unsafe fn inverse_dit_stages<const D: usize>(
    a: *mut i32,
    inv: &BarrettTable<i32, D>,
    m: Modulus,
    scale: impl Fn(usize) -> Multiplier,
) {
    // Stages 1 and 2. Stage 1 and the first stage-2 butterfly have twiddle 1.
    // Every stage output lies in `[0, 2p)`.
    let w = Multiplier::broadcast(inv, 2);
    let mut base = 0usize;
    while base < D {
        let int32x4x4_t(r0, r1, r2, r3) = vld4q_s32(a.add(base));
        let s0 = fold_4x_i32(vaddq_s32(r0, r1), m.two_p);
        let d0 = fold_4x_i32(vsubq_s32(r0, r1), m.two_p);
        let s1 = fold_4x_i32(vaddq_s32(r2, r3), m.two_p);
        let d1 = fold_4x_i32(vsubq_s32(r2, r3), m.two_p);
        let o0 = fold_4x_i32(vaddq_s32(s0, s1), m.neg_two_p);
        let o2 = fold_4x_i32(vsubq_s32(s0, s1), m.two_p);
        let (o1, o3) = dit_butterfly(d0, d1, w, m);
        vst4q_s32(a.add(base), int32x4x4_t(o0, o1, o2, o3));
        base += 16;
    }

    // Stages 4 through D/8, two at a time where possible. An odd number of
    // them would leave a lone radix-2 pass; from D = 512 the first three run
    // as one radix-8 pass instead. At D = 128 those three are the whole range
    // and the radix-8 pass measured slower than radix 4 plus radix 2.
    let quarter = D / 4;
    let mut len = 4usize;
    if quarter >= 128 && quarter.trailing_zeros() % 2 == 1 {
        let mut x = 0usize;
        while x < D {
            let u: [int32x4_t; 8] = core::array::from_fn(|k| vld1q_s32(a.add(x + 4 * k)));
            let w = Multiplier::load(inv, 3);
            let (v0, v1) = dit_butterfly(u[0], u[1], w, m);
            let (v2, v3) = dit_butterfly(u[2], u[3], w, m);
            let (v4, v5) = dit_butterfly(u[4], u[5], w, m);
            let (v6, v7) = dit_butterfly(u[6], u[7], w, m);
            let w0 = Multiplier::load(inv, 7);
            let w1 = Multiplier::load(inv, 11);
            let (y0, y2) = dit_butterfly(v0, v2, w0, m);
            let (y1, y3) = dit_butterfly(v1, v3, w1, m);
            let (y4, y6) = dit_butterfly(v4, v6, w0, m);
            let (y5, y7) = dit_butterfly(v5, v7, w1, m);
            let w = |k: usize| Multiplier::load(inv, 15 + 4 * k);
            let z = [
                dit_butterfly(y0, y4, w(0), m),
                dit_butterfly(y1, y5, w(1), m),
                dit_butterfly(y2, y6, w(2), m),
                dit_butterfly(y3, y7, w(3), m),
            ];
            for (k, (lo, hi)) in z.into_iter().enumerate() {
                vst1q_s32(a.add(x + 4 * k), lo);
                vst1q_s32(a.add(x + 4 * k + 16), hi);
            }
            x += 32;
        }
        len = 32;
    }
    while 4 * len <= quarter {
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            while j < len {
                let x = start + j;
                let u0 = vld1q_s32(a.add(x));
                let u1 = vld1q_s32(a.add(x + len));
                let u2 = vld1q_s32(a.add(x + 2 * len));
                let u3 = vld1q_s32(a.add(x + 3 * len));
                let w = Multiplier::load(inv, len - 1 + j);
                let (v0, v1) = dit_butterfly(u0, u1, w, m);
                let (v2, v3) = dit_butterfly(u2, u3, w, m);
                let (z0, z2) = dit_butterfly(v0, v2, Multiplier::load(inv, 2 * len - 1 + j), m);
                let (z1, z3) =
                    dit_butterfly(v1, v3, Multiplier::load(inv, 2 * len - 1 + j + len), m);
                vst1q_s32(a.add(x), z0);
                vst1q_s32(a.add(x + len), z1);
                vst1q_s32(a.add(x + 2 * len), z2);
                vst1q_s32(a.add(x + 3 * len), z3);
                j += 4;
            }
            start += 4 * len;
        }
        len *= 4;
    }
    if len < quarter {
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            while j < len {
                let x = start + j;
                let u = vld1q_s32(a.add(x));
                let v = vld1q_s32(a.add(x + len));
                let (sum, diff) = dit_butterfly(u, v, Multiplier::load(inv, len - 1 + j), m);
                vst1q_s32(a.add(x), sum);
                vst1q_s32(a.add(x + len), diff);
                j += 4;
            }
            start += 2 * len;
        }
    }

    // Stages D/4 and D/2 with the output scaling. Recentering the stage-D/4
    // sums to `[-p, p)` keeps the stage-D/2 sums and differences in (-2p, 2p),
    // so they go straight into the scaling multiply.
    let half = D / 2;
    let mut o = 0usize;
    while o < quarter {
        let x0 = vld1q_s32(a.add(o));
        let x1 = vld1q_s32(a.add(o + quarter));
        let x2 = vld1q_s32(a.add(o + half));
        let x3 = vld1q_s32(a.add(o + half + quarter));
        let w = Multiplier::load(inv, quarter - 1 + o);
        let (y0, y1) = dit_butterfly(x0, x1, w, m);
        let (y2, y3) = dit_butterfly(x2, x3, w, m);
        let (y0, y1) = (vsubq_s32(y0, m.p), vsubq_s32(y1, m.p));
        let v0 = Multiplier::load(inv, half - 1 + o).mul(y2, m.p);
        let v1 = Multiplier::load(inv, half - 1 + o + quarter).mul(y3, m.p);
        vst1q_s32(a.add(o), scale(o).mul(vaddq_s32(y0, v0), m.p));
        vst1q_s32(a.add(o + half), scale(o + half).mul(vsubq_s32(y0, v0), m.p));
        vst1q_s32(
            a.add(o + quarter),
            scale(o + quarter).mul(vaddq_s32(y1, v1), m.p),
        );
        vst1q_s32(
            a.add(o + half + quarter),
            scale(o + half + quarter).mul(vsubq_s32(y1, v1), m.p),
        );
        o += 4;
    }
}

/// NEON-accelerated forward negacyclic NTT for i32 primes.
///
/// Accepts any representative `|x| < 2^31`; outputs are canonical `[0, p)`.
pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    if D < MIN_VECTOR_DEGREE {
        butterfly::forward_ntt(a, prime, tw, NttKernelPlan::SCALAR);
        return;
    }
    let m = Modulus::new(prime.p);
    let a_ptr = a.as_mut_ptr().cast::<i32>();
    let barrett = &tw.barrett;
    forward_head::<D, true, true>(a_ptr, barrett, &barrett.twist, m, |i| {
        [vld1q_s32(a_ptr.add(i)), vld1q_s32(a_ptr.add(i + 4))]
    });
    forward_dif_stages::<D>(a_ptr, &barrett.fwd, m, D / 8, false);
}

/// Signed-integer conversion and forward negacyclic (or, without
/// `NEGACYCLIC`, cyclic) NTT for i32 primes.
///
/// The first pass reads the inputs directly and multiplies by `R`-scaled
/// stage factors, so conversion to Montgomery form, any twist, and the first
/// two stages share one pass. `load(ptr)` widens the eight inputs at `ptr`,
/// and every input must have magnitude below `p / 2`, so the cyclic head's
/// unmultiplied pair sums stay in `(-p, p)`.
#[inline(always)]
unsafe fn forward_ntt_small_i32<T: Copy + Into<i32>, const D: usize, const NEGACYCLIC: bool>(
    a: &mut [MontCoeff<i32>; D],
    inputs: &[T; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    load: impl Fn(*const T) -> [int32x4_t; 2],
) {
    if D < MIN_VECTOR_DEGREE {
        for ((coefficient, &input), &psi_r2) in a.iter_mut().zip(inputs).zip(&tw.psi_pows_r2) {
            let factor = if NEGACYCLIC { psi_r2 } else { prime.montsq };
            *coefficient = MontCoeff::from_raw(prime.mont_mul_raw(input.into(), factor));
        }
        butterfly::forward_ntt_cyclic(a, prime, tw, NttKernelPlan::SCALAR);
        return;
    }
    let m = Modulus::new(prime.p);
    let a_ptr = a.as_mut_ptr().cast::<i32>();
    let inputs_ptr = inputs.as_ptr();
    let barrett = &tw.barrett;
    let first = if NEGACYCLIC {
        &barrett.twist_digits
    } else {
        &barrett.cyclic_digits
    };
    forward_head::<D, false, NEGACYCLIC>(a_ptr, barrett, first, m, |i| load(inputs_ptr.add(i)));
    forward_dif_stages::<D>(a_ptr, &barrett.fwd, m, D / 8, false);
}

/// Widen eight signed digits to two i32 vectors.
#[inline(always)]
unsafe fn load_i8x8(ptr: *const i8) -> [int32x4_t; 2] {
    let wide = vmovl_s8(vld1_s8(ptr));
    [vmovl_s16(vget_low_s16(wide)), vmovl_high_s16(wide)]
}

/// NEON-accelerated signed-i8 conversion and forward negacyclic NTT.
///
/// Kept out of line, like the centered-i16 entry: inlined into the digit
/// fill loop, it slows the i8 matvec by about 8%.
#[inline(never)]
pub(crate) unsafe fn forward_ntt_i8_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    forward_ntt_small_i32::<_, D, true>(a, digits, prime, tw, |ptr| load_i8x8(ptr));
}

/// NEON-accelerated signed-i8 conversion and forward cyclic NTT.
///
/// Kept out of line for the same reason as the negacyclic entry.
#[inline(never)]
pub(crate) unsafe fn forward_ntt_cyclic_i8_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    forward_ntt_small_i32::<_, D, false>(a, digits, prime, tw, |ptr| load_i8x8(ptr));
}

/// NEON-accelerated centered-i16 conversion and forward negacyclic NTT.
///
/// The protocol's i32 CRT primes all exceed the complete i16 range, so sign
/// extension already gives a representative in `(-p, p)`.
#[inline(never)]
pub(crate) unsafe fn forward_ntt_centered_i16_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    coefficients: &[i16; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    forward_ntt_small_i32::<_, D, true>(a, coefficients, prime, tw, |ptr| {
        let x = vld1q_s16(ptr);
        [vmovl_s16(vget_low_s16(x)), vmovl_high_s16(x)]
    });
}

/// NEON-accelerated inverse negacyclic NTT for i32 primes.
///
/// Inputs must lie in `(-p, p)`; outputs lie in `(-p, p)`.
pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    if D < MIN_VECTOR_DEGREE {
        butterfly::inverse_ntt(a, prime, tw, NttKernelPlan::SCALAR);
        return;
    }
    let barrett = &tw.barrett;
    inverse_dit_stages::<D>(
        a.as_mut_ptr().cast::<i32>(),
        &barrett.inv,
        Modulus::new(prime.p),
        |i| Multiplier::load(&barrett.untwist, i),
    );
}

/// NEON-accelerated forward cyclic NTT for i32 (no negacyclic twist).
///
/// Inputs must lie in `(-p, p)`; outputs are canonical `[0, p)`.
#[inline]
pub(crate) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    if D < MIN_VECTOR_DEGREE {
        butterfly::forward_ntt_cyclic(a, prime, tw, NttKernelPlan::SCALAR);
        return;
    }
    forward_dif_stages::<D>(
        a.as_mut_ptr().cast::<i32>(),
        &tw.barrett.fwd,
        Modulus::new(prime.p),
        D / 2,
        true,
    );
}

/// NEON-accelerated inverse cyclic NTT for i32 (no negacyclic untwist).
///
/// Inputs must lie in `(-p, p)`; outputs lie in `(-p, p)`.
pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    if D < MIN_VECTOR_DEGREE {
        butterfly::inverse_ntt_cyclic(a, prime, tw, NttKernelPlan::SCALAR);
        return;
    }
    let d_inv = Multiplier::splat(tw.barrett.d_inv);
    inverse_dit_stages::<D>(
        a.as_mut_ptr().cast::<i32>(),
        &tw.barrett.inv,
        Modulus::new(prime.p),
        |_| d_inv,
    );
}

/// 4-wide pointwise multiply-accumulate for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))` for `i in 0..d`.
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_q = vdupq_n_s32(p);
    let two_p_q = vdupq_n_s32(2 * p);
    let pinv_q = vdupq_n_s32(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 4 <= d {
        let a = vld1q_s32(acc.add(i));
        let l = vld1q_s32(lhs.add(i));
        let r = vld1q_s32(rhs.add(i));
        let prod = mont_mul_4x_i32(l, r, p_q, pinv_q);
        let sum = vaddq_s32(a, prod);
        vst1q_s32(acc.add(i), reduce_range_4x_i32(sum, p_q, two_p_q));
        i += 4;
    }
    while i < d {
        let prod = prime.mul(
            MontCoeff::from_raw(*lhs.add(i)),
            MontCoeff::from_raw(*rhs.add(i)),
        );
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// NEON pointwise dot-product accumulation for up to six i32 CRT entries.
///
/// Raw signed products are accumulated in i64 lanes and Montgomery-reduced
/// once per batch. For `B <= 6` and `p < 2^30`, the reduction numerator is
/// bounded by `B*2^60 + 2^61 < 2^63`, and the reduced batch lies in
/// `(-2p, 2p)`.
///
/// # Safety
///
/// `acc` must be valid for `d` writable i32 elements. Each of the first
/// `count` pointers in `lhs` and `rhs` must be valid for `d` readable i32
/// elements. The pointed-to ranges must obey Rust's aliasing rules with `acc`.
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
        ($count:literal) => {
            pointwise_dot_acc_i32_count::<$count>(acc, lhs, rhs, d, p, pinv)
        };
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

#[inline(always)]
unsafe fn pointwise_dot_acc_i32_count<const COUNT: usize>(
    acc: *mut i32,
    lhs: *const *const i32,
    rhs: *const *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let lhs: [*const i32; COUNT] = core::array::from_fn(|product| *lhs.add(product));
    let rhs: [*const i32; COUNT] = core::array::from_fn(|product| *rhs.add(product));
    let m = Modulus::new(p);
    let p_d = vdup_n_s32(p);
    let pinv_d = vdup_n_s32(pinv);
    let mut i = 0usize;
    let finish = |i: usize, low_sum: int64x2_t, high_sum: int64x2_t| {
        let low_correction = vmull_s32(vmul_s32(vmovn_s64(low_sum), pinv_d), p_d);
        let high_correction = vmull_s32(vmul_s32(vmovn_s64(high_sum), pinv_d), p_d);
        let low = vshrn_n_s64::<32>(vsubq_s64(low_sum, low_correction));
        let high = vshrn_n_s64::<32>(vsubq_s64(high_sum, high_correction));
        let batch = centered_reduce_4x_i32(vcombine_s32(low, high), m.p, m.reciprocal);
        let accumulator = vld1q_s32(acc.add(i));
        vst1q_s32(
            acc.add(i),
            reduce_range_4x_i32(vaddq_s32(accumulator, batch), m.p, m.two_p),
        );
    };
    while i + 8 <= d {
        let mut low0 = vdupq_n_s64(0);
        let mut high0 = vdupq_n_s64(0);
        let mut low1 = vdupq_n_s64(0);
        let mut high1 = vdupq_n_s64(0);
        for product in 0..COUNT {
            let l0 = vld1q_s32(lhs[product].add(i));
            let r0 = vld1q_s32(rhs[product].add(i));
            let l1 = vld1q_s32(lhs[product].add(i + 4));
            let r1 = vld1q_s32(rhs[product].add(i + 4));
            low0 = vmlal_s32(low0, vget_low_s32(l0), vget_low_s32(r0));
            high0 = vmlal_high_s32(high0, l0, r0);
            low1 = vmlal_s32(low1, vget_low_s32(l1), vget_low_s32(r1));
            high1 = vmlal_high_s32(high1, l1, r1);
        }
        finish(i, low0, high0);
        finish(i + 4, low1, high1);
        i += 8;
    }
    while i + 4 <= d {
        let mut low_sum = vdupq_n_s64(0);
        let mut high_sum = vdupq_n_s64(0);
        for product in 0..COUNT {
            let l = vld1q_s32(lhs[product].add(i));
            let r = vld1q_s32(rhs[product].add(i));
            low_sum = vmlal_s32(low_sum, vget_low_s32(l), vget_low_s32(r));
            high_sum = vmlal_high_s32(high_sum, l, r);
        }
        finish(i, low_sum, high_sum);
        i += 4;
    }

    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            let mut raw_sum = 0_i64;
            for product in 0..COUNT {
                raw_sum += i64::from(*lhs[product].add(i)) * i64::from(*rhs[product].add(i));
            }
            let correction = (raw_sum as i32).wrapping_mul(pinv);
            let reduced = ((raw_sum - i64::from(correction) * i64::from(p)) >> 32) as i32;
            let reduced = prime.reduce_range(MontCoeff::from_raw(reduced));
            let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(reduced.raw()));
            *acc.add(i) = prime.reduce_range(sum).raw();
            i += 1;
        }
    }
}

/// Convert signed i8 coefficients directly into an i32 Montgomery limb.
///
/// # Safety
///
/// `dst` and `src` must be valid for `d` elements and must not overlap. The
/// modulus must be larger than the complete signed i8 range.
pub(crate) unsafe fn centered_i8_to_mont_i32(
    dst: *mut i32,
    src: *const i8,
    d: usize,
    p: i32,
    pinv: i32,
    montsq: i32,
) {
    let p_q = vdupq_n_s32(p);
    let pinv_q = vdupq_n_s32(pinv);
    let montsq_q = vdupq_n_s32(montsq);
    let mut i = 0usize;
    while i + 16 <= d {
        let coefficients = vld1q_s8(src.add(i));
        let low_i16 = vmovl_s8(vget_low_s8(coefficients));
        let high_i16 = vmovl_high_s8(coefficients);
        let values = [
            vmovl_s16(vget_low_s16(low_i16)),
            vmovl_high_s16(low_i16),
            vmovl_s16(vget_low_s16(high_i16)),
            vmovl_high_s16(high_i16),
        ];
        for (chunk, values) in values.into_iter().enumerate() {
            vst1q_s32(
                dst.add(i + chunk * 4),
                mont_mul_4x_i32(values, montsq_q, p_q, pinv_q),
            );
        }
        i += 16;
    }

    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            *dst.add(i) = prime.from_canonical(i32::from(*src.add(i))).raw();
            i += 1;
        }
    }
}

/// 4-wide add-and-reduce for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + other[i])` for `i in 0..d`.
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements, properly aligned for
/// NEON loads/stores, and must not alias in a way that violates Rust's
/// mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_q = vdupq_n_s32(p);
    let two_p_q = vdupq_n_s32(2 * p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 4 <= d {
        let a = vld1q_s32(acc.add(i));
        let b = vld1q_s32(other.add(i));
        vst1q_s32(
            acc.add(i),
            reduce_range_4x_i32(vaddq_s32(a, b), p_q, two_p_q),
        );
        i += 4;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}
