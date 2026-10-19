//! AVX-512IFMA kernels for 50-bit residues.
//!
//! The forward transform is the negacyclic Cooley–Tukey NTT with the twist
//! merged into its twiddles, and the inverse is the matching Gentleman–Sande
//! transform with `D^{-1}` folded into its last stage. Each runs:
//!
//! - one radix-2, -4, or -8 pass across vectors per group of stages whose
//!   butterfly half-length is at least 64, with broadcast twiddles;
//! - one pass per 64 values for the six stages nearest the leaves. Its three
//!   in-register stages interleave the two halves of each 16-value block
//!   through self-inverse shuffles, and store that order
//!   ([`super::STORED_ORDER`]).
//!
//! Every multiply is a Shoup product that accepts any input below
//! `2^52 > 4p` and returns `[0, 2p)`. Forward values stay in `[0, 4p)`
//! because Harvey's butterfly reduces only its unmultiplied input; inverse
//! values stay in `[0, 2p)`, and its last stage returns canonical residues.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{
    Ifma52Accumulator, Ifma52Prime, Ifma52Residues, Ifma52Twiddles, LaneMultiplier, MASK, RADIX,
};

#[derive(Clone, Copy)]
struct Modulus {
    p: __m512i,
    two_p: __m512i,
    complement: __m512i,
    mask: __m512i,
}

impl Modulus {
    #[inline(always)]
    unsafe fn new(prime: Ifma52Prime) -> Self {
        // SAFETY: the caller enables AVX-512F.
        unsafe {
            Self {
                p: _mm512_set1_epi64(prime.modulus as i64),
                two_p: _mm512_set1_epi64((2 * prime.modulus) as i64),
                complement: _mm512_set1_epi64((RADIX - prime.modulus) as i64),
                mask: _mm512_set1_epi64(MASK as i64),
            }
        }
    }

    /// `[0, 4p) -> [0, 2p)`.
    #[inline(always)]
    unsafe fn reduce_4p(self, x: __m512i) -> __m512i {
        // SAFETY: the caller enables AVX-512F.
        unsafe { _mm512_min_epu64(x, _mm512_sub_epi64(x, self.two_p)) }
    }

    /// `[0, 2p) -> [0, p)`.
    #[inline(always)]
    unsafe fn reduce_2p(self, x: __m512i) -> __m512i {
        // SAFETY: the caller enables AVX-512F.
        unsafe { _mm512_min_epu64(x, _mm512_sub_epi64(x, self.p)) }
    }
}

/// A constant multiplier with its Shoup quotient `floor(w 2^52 / p)`.
#[derive(Clone, Copy)]
struct Multiplier {
    value: __m512i,
    precon: __m512i,
}

impl Multiplier {
    #[inline(always)]
    unsafe fn splat(value: u64, precon: u64) -> Self {
        // SAFETY: the caller enables AVX-512F.
        unsafe {
            Self {
                value: _mm512_set1_epi64(value as i64),
                precon: _mm512_set1_epi64(precon as i64),
            }
        }
    }

    #[inline(always)]
    unsafe fn lanes(lanes: &LaneMultiplier) -> Self {
        // SAFETY: the caller enables AVX-512F; `LaneMultiplier` is 64-byte
        // aligned and each field is one vector.
        unsafe {
            Self {
                value: _mm512_load_si512(lanes.value.as_ptr().cast()),
                precon: _mm512_load_si512(lanes.precon.as_ptr().cast()),
            }
        }
    }

    /// `x w mod p` in `[0, 2p)` for `x < 2^52`.
    #[inline(always)]
    unsafe fn mul(self, x: __m512i, m: Modulus) -> __m512i {
        // SAFETY: the caller enables AVX-512F and AVX-512IFMA.
        unsafe {
            let zero = _mm512_setzero_si512();
            let quotient = _mm512_madd52hi_epu64(zero, self.precon, x);
            let product = _mm512_madd52lo_epu64(zero, self.value, x);
            _mm512_and_si512(
                _mm512_madd52lo_epu64(product, quotient, m.complement),
                m.mask,
            )
        }
    }
}

/// Cooley–Tukey butterfly on `[0, 4p)` inputs with `[0, 4p)` outputs. With
/// `small`, `a` is already below `2p`.
#[inline(always)]
unsafe fn ct(a: __m512i, b: __m512i, w: Multiplier, m: Modulus, small: bool) -> (__m512i, __m512i) {
    // SAFETY: the caller enables AVX-512F and AVX-512IFMA.
    unsafe {
        let t = w.mul(b, m);
        let a = if small { a } else { m.reduce_4p(a) };
        (
            _mm512_add_epi64(a, t),
            _mm512_sub_epi64(_mm512_add_epi64(a, m.two_p), t),
        )
    }
}

/// Gentleman–Sande butterfly on `[0, 2p)` inputs with `[0, 2p)` outputs.
#[inline(always)]
unsafe fn gs(a: __m512i, b: __m512i, w: Multiplier, m: Modulus) -> (__m512i, __m512i) {
    // SAFETY: the caller enables AVX-512F and AVX-512IFMA.
    unsafe {
        (
            m.reduce_4p(_mm512_add_epi64(a, b)),
            w.mul(_mm512_sub_epi64(_mm512_add_epi64(a, m.two_p), b), m),
        )
    }
}

/// The last inverse butterfly, scaled by `D^{-1}` into `[0, p)`.
#[inline(always)]
unsafe fn gs_last<const D: usize>(
    a: __m512i,
    b: __m512i,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
) -> (__m512i, __m512i) {
    // SAFETY: the caller enables AVX-512F and AVX-512IFMA.
    unsafe {
        let scale = Multiplier::splat(tw.inverse_scale[0], tw.inverse_scale_precon[0]);
        let twisted = Multiplier::splat(tw.inverse_scale[1], tw.inverse_scale_precon[1]);
        (
            m.reduce_2p(scale.mul(_mm512_add_epi64(a, b), m)),
            m.reduce_2p(twisted.mul(_mm512_sub_epi64(_mm512_add_epi64(a, m.two_p), b), m)),
        )
    }
}

/// Stages `s .. s + log2 R` of the forward transform on `R` vectors spaced
/// evenly across block `index - 2^s` of stage `s`. With `small`, the inputs
/// are below `2p`.
///
/// Stage `s` is peeled so that every loop body is uniform: a flag that varies
/// by level stops LLVM from fully unrolling the nest, which leaves `x` on the
/// stack.
#[inline(always)]
unsafe fn forward_butterflies<const D: usize, const R: usize>(
    x: &mut [__m512i; R],
    index: usize,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
    small: bool,
) {
    let half = R / 2;
    // SAFETY: stage `s` has `2^s` blocks, so `index < D`; the caller enables
    // AVX-512F and AVX-512IFMA.
    unsafe {
        let w = Multiplier::splat(
            *tw.forward.get_unchecked(index),
            *tw.forward_precon.get_unchecked(index),
        );
        for t in 0..half {
            (x[t], x[t + half]) = ct(x[t], x[t + half], w, m, small);
        }
    }
    let mut half = R / 4;
    let mut level = 1;
    while half > 0 {
        for group in 0..1 << level {
            let entry = (index << level) + group;
            // SAFETY: stage `s + level` has `2^(s + level)` blocks, so
            // `entry < D`; the caller enables AVX-512F and AVX-512IFMA.
            unsafe {
                let w = Multiplier::splat(
                    *tw.forward.get_unchecked(entry),
                    *tw.forward_precon.get_unchecked(entry),
                );
                for t in 2 * half * group..2 * half * group + half {
                    (x[t], x[t + half]) = ct(x[t], x[t + half], w, m, false);
                }
            }
        }
        half /= 2;
        level += 1;
    }
}

/// The mirror image of [`forward_butterflies`]. With `LAST`, block `index ==
/// 1` is stage 0 and its butterflies apply `D^{-1}`.
#[inline(always)]
unsafe fn inverse_butterflies<const D: usize, const R: usize, const LAST: bool>(
    x: &mut [__m512i; R],
    index: usize,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
) {
    let mut half = 1;
    let mut level = R.trailing_zeros();
    while level > 0 {
        level -= 1;
        for group in 0..1 << level {
            let entry = (index << level) + group;
            // SAFETY: as in `forward_butterflies`.
            unsafe {
                let w = Multiplier::splat(
                    *tw.inverse.get_unchecked(entry),
                    *tw.inverse_precon.get_unchecked(entry),
                );
                for t in 2 * half * group..2 * half * group + half {
                    (x[t], x[t + half]) = if LAST && level == 0 {
                        gs_last(x[t], x[t + half], tw, m)
                    } else {
                        gs(x[t], x[t + half], w, m)
                    };
                }
            }
        }
        half *= 2;
    }
}

/// One forward pass over stages `stage .. stage + log2 R`, all with
/// butterfly half-length at least 8. `load(i)` reads the vector at value `i`,
/// which lets the first pass convert its inputs.
#[inline(always)]
unsafe fn forward_pass<const D: usize, const R: usize>(
    values: *mut u64,
    stage: u32,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
    load: &impl Fn(usize) -> __m512i,
    small: bool,
) {
    let blocks = 1 << stage;
    let stride = D / blocks / R;
    for block in 0..blocks {
        let base = block * D / blocks;
        for offset in (0..stride).step_by(8) {
            let start = base + offset;
            // SAFETY: `stride >= 8`, every index below `start + R stride`
            // lies in block `block`, and `values` is 64-byte aligned.
            unsafe {
                let mut x: [__m512i; R] = core::array::from_fn(|t| load(start + t * stride));
                forward_butterflies::<D, R>(&mut x, blocks + block, tw, m, small);
                for (t, value) in x.into_iter().enumerate() {
                    _mm512_store_si512(values.add(start + t * stride).cast(), value);
                }
            }
        }
    }
}

/// The mirror image of [`forward_pass`], in place.
#[inline(always)]
unsafe fn inverse_pass<const D: usize, const R: usize, const LAST: bool>(
    values: *mut u64,
    stage: u32,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
) {
    let blocks = 1 << stage;
    let stride = D / blocks / R;
    for block in 0..blocks {
        let base = block * D / blocks;
        for offset in (0..stride).step_by(8) {
            // SAFETY: as in `forward_pass`.
            unsafe {
                let pointer = values.add(base + offset);
                let mut x: [__m512i; R] =
                    core::array::from_fn(|t| _mm512_load_si512(pointer.add(t * stride).cast()));
                inverse_butterflies::<D, R, LAST>(&mut x, blocks + block, tw, m);
                for (t, value) in x.into_iter().enumerate() {
                    _mm512_store_si512(pointer.add(t * stride).cast(), value);
                }
            }
        }
    }
}

const LOW_PAIRS: [i64; 8] = [0, 1, 8, 9, 4, 5, 12, 13];
const HIGH_PAIRS: [i64; 8] = [2, 3, 10, 11, 6, 7, 14, 15];

/// The last six forward stages, 64 values at a time.
#[inline(always)]
unsafe fn forward_tail<const D: usize>(
    values: *mut u64,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
    load: &impl Fn(usize) -> __m512i,
    small: bool,
) {
    // SAFETY: the caller enables AVX-512F and AVX-512IFMA; `values` is 64-byte
    // aligned and holds `D` values; `forward_lanes` has one entry per
    // 16-value block.
    unsafe {
        let low_pairs = _mm512_loadu_si512(LOW_PAIRS.as_ptr().cast());
        let high_pairs = _mm512_loadu_si512(HIGH_PAIRS.as_ptr().cast());
        for group in 0..D / 64 {
            let base = 64 * group;
            let mut x: [__m512i; 8] = core::array::from_fn(|t| load(base + 8 * t));
            forward_butterflies::<D, 8>(&mut x, D / 64 + group, tw, m, small);
            for pair in 0..4 {
                let lanes = tw.forward_lanes.get_unchecked(4 * group + pair);
                let (a, b) = (x[2 * pair], x[2 * pair + 1]);
                let (a, b) = (
                    _mm512_shuffle_i64x2::<0x44>(a, b),
                    _mm512_shuffle_i64x2::<0xee>(a, b),
                );
                let (a, b) = ct(a, b, Multiplier::lanes(&lanes[0]), m, false);
                let (a, b) = (
                    _mm512_permutex2var_epi64(a, low_pairs, b),
                    _mm512_permutex2var_epi64(a, high_pairs, b),
                );
                let (a, b) = ct(a, b, Multiplier::lanes(&lanes[1]), m, false);
                let (a, b) = (_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
                let (a, b) = ct(a, b, Multiplier::lanes(&lanes[2]), m, false);
                let pointer = values.add(base + 16 * pair);
                _mm512_store_si512(pointer.cast(), a);
                _mm512_store_si512(pointer.add(8).cast(), b);
            }
        }
    }
}

/// The first six inverse stages, 64 values at a time. With `LAST` (`D ==
/// 64`), the sixth is stage 0.
#[inline(always)]
unsafe fn inverse_head<const D: usize, const LAST: bool>(
    values: *mut u64,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
) {
    // SAFETY: as in `forward_tail`.
    unsafe {
        let low_pairs = _mm512_loadu_si512(LOW_PAIRS.as_ptr().cast());
        let high_pairs = _mm512_loadu_si512(HIGH_PAIRS.as_ptr().cast());
        for group in 0..D / 64 {
            let base = 64 * group;
            let mut x = [_mm512_setzero_si512(); 8];
            for pair in 0..4 {
                let lanes = tw.inverse_lanes.get_unchecked(4 * group + pair);
                let pointer = values.add(base + 16 * pair);
                let (a, b) = (
                    _mm512_load_si512(pointer.cast()),
                    _mm512_load_si512(pointer.add(8).cast()),
                );
                let (a, b) = gs(a, b, Multiplier::lanes(&lanes[2]), m);
                let (a, b) = (_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
                let (a, b) = gs(a, b, Multiplier::lanes(&lanes[1]), m);
                let (a, b) = (
                    _mm512_permutex2var_epi64(a, low_pairs, b),
                    _mm512_permutex2var_epi64(a, high_pairs, b),
                );
                let (a, b) = gs(a, b, Multiplier::lanes(&lanes[0]), m);
                x[2 * pair] = _mm512_shuffle_i64x2::<0x44>(a, b);
                x[2 * pair + 1] = _mm512_shuffle_i64x2::<0xee>(a, b);
            }
            inverse_butterflies::<D, 8, LAST>(&mut x, D / 64 + group, tw, m);
            for (t, value) in x.into_iter().enumerate() {
                _mm512_store_si512(values.add(base + 8 * t).cast(), value);
            }
        }
    }
}

/// Forward transform whose first pass reads its inputs, each below `4p` (or
/// `2p` with `small`), through `load`.
#[inline(always)]
unsafe fn forward_from<const D: usize>(
    values: *mut u64,
    tw: &Ifma52Twiddles<D>,
    m: Modulus,
    load: impl Fn(usize) -> __m512i,
    small: bool,
) {
    let outer = D.trailing_zeros() - 6;
    let head = outer % 3;
    // SAFETY: the caller enables AVX-512F and AVX-512IFMA; `values` is 64-byte
    // aligned and holds `D` values.
    unsafe {
        if outer == 0 {
            return forward_tail(values, tw, m, &load, small);
        }
        match head {
            1 => forward_pass::<D, 2>(values, 0, tw, m, &load, small),
            2 => forward_pass::<D, 4>(values, 0, tw, m, &load, small),
            _ => forward_pass::<D, 8>(values, 0, tw, m, &load, small),
        }
        let stored = |i: usize| _mm512_load_si512(values.add(i).cast());
        let mut stage = if head == 0 { 3 } else { head };
        while stage < outer {
            forward_pass::<D, 8>(values, stage, tw, m, &stored, false);
            stage += 3;
        }
        forward_tail(values, tw, m, &stored, false);
    }
}

/// Forward transform of residues below `4p`, into `[0, 4p)`.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn forward<const D: usize>(
    values: &mut Ifma52Residues<D>,
    prime: Ifma52Prime,
    tw: &Ifma52Twiddles<D>,
) {
    let values = values.0.as_mut_ptr();
    // SAFETY: inherited target features; `values` is 64-byte aligned.
    unsafe {
        forward_from(
            values,
            tw,
            Modulus::new(prime),
            #[inline(always)]
            |i| _mm512_load_si512(values.add(i).cast()),
            false,
        )
    }
}

/// Forward transform of signed `i16` coefficients, into `[0, 4p)`.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn forward_i16<const D: usize>(
    values: &mut Ifma52Residues<D>,
    coefficients: &[i16; D],
    prime: Ifma52Prime,
    tw: &Ifma52Twiddles<D>,
) {
    let coefficients = coefficients.as_ptr();
    // SAFETY: inherited target features; each load reads eight of the `D`
    // coefficients, and `c + p` lies in `[0, 2p)` because `p > 2^15`.
    unsafe {
        let m = Modulus::new(prime);
        forward_from(
            values.0.as_mut_ptr(),
            tw,
            m,
            #[inline(always)]
            |i| {
                _mm512_add_epi64(
                    _mm512_cvtepi16_epi64(_mm_loadu_si128(coefficients.add(i).cast())),
                    m.p,
                )
            },
            true,
        )
    }
}

/// Inverse transform of `[0, 2p)` residues, into `[0, p)`.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn inverse<const D: usize>(
    values: &mut Ifma52Residues<D>,
    prime: Ifma52Prime,
    tw: &Ifma52Twiddles<D>,
) {
    let values = values.0.as_mut_ptr();
    let outer = D.trailing_zeros() - 6;
    let head = outer % 3;
    // SAFETY: inherited target features; `values` is 64-byte aligned.
    unsafe {
        let m = Modulus::new(prime);
        if outer == 0 {
            return inverse_head::<D, true>(values, tw, m);
        }
        inverse_head::<D, false>(values, tw, m);
        let mut stage = outer;
        while stage > head.max(3) {
            stage -= 3;
            inverse_pass::<D, 8, false>(values, stage, tw, m);
        }
        match head {
            1 => inverse_pass::<D, 2, true>(values, 0, tw, m),
            2 => inverse_pass::<D, 4, true>(values, 0, tw, m),
            _ => inverse_pass::<D, 8, true>(values, 0, tw, m),
        }
    }
}

/// Vectors of each accumulator row that one pass over the terms updates.
const DOT_VECTORS: usize = 4;

/// Add the low and high 52-bit halves of `lhs[i] * rhs[i]` to the
/// accumulator, lane by lane.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn dot_accumulate<const D: usize>(
    accumulator: &mut Ifma52Accumulator<D>,
    lhs: &[Ifma52Residues<D>],
    rhs: &[Ifma52Residues<D>],
) {
    let low = accumulator.low.0.as_mut_ptr();
    let high = accumulator.high.0.as_mut_ptr();
    for index in (0..D).step_by(8 * DOT_VECTORS) {
        // SAFETY: `D` is a multiple of 64 and every array is 64-byte aligned.
        unsafe {
            let mut sums: [(__m512i, __m512i); DOT_VECTORS] = core::array::from_fn(|v| {
                (
                    _mm512_load_si512(low.add(index + 8 * v).cast()),
                    _mm512_load_si512(high.add(index + 8 * v).cast()),
                )
            });
            for (lhs, rhs) in lhs.iter().zip(rhs) {
                let (lhs, rhs) = (lhs.0.as_ptr().add(index), rhs.0.as_ptr().add(index));
                for (v, (low, high)) in sums.iter_mut().enumerate() {
                    let a = _mm512_load_si512(lhs.add(8 * v).cast());
                    let b = _mm512_load_si512(rhs.add(8 * v).cast());
                    *low = _mm512_madd52lo_epu64(*low, a, b);
                    *high = _mm512_madd52hi_epu64(*high, a, b);
                }
            }
            for (v, (low_sum, high_sum)) in sums.into_iter().enumerate() {
                _mm512_store_si512(low.add(index + 8 * v).cast(), low_sum);
                _mm512_store_si512(high.add(index + 8 * v).cast(), high_sum);
            }
        }
    }
}

/// Canonical residues of `high 2^52 + low`.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn dot_reduce<const D: usize>(
    accumulator: &Ifma52Accumulator<D>,
    prime: Ifma52Prime,
) -> Ifma52Residues<D> {
    let mut out = Ifma52Residues([0; D]);
    // SAFETY: inherited target features; every array is 64-byte aligned.
    unsafe {
        let m = Modulus::new(prime);
        let radix = Multiplier::splat(prime.radix, prime.radix_precon);
        let radix_squared = Multiplier::splat(prime.radix_squared, prime.radix_squared_precon);
        let four_p = _mm512_add_epi64(m.two_p, m.two_p);
        let eight_p = _mm512_add_epi64(four_p, four_p);
        for index in (0..D).step_by(8) {
            let low = _mm512_load_si512(accumulator.low.0.as_ptr().add(index).cast());
            let high = _mm512_load_si512(accumulator.high.0.as_ptr().add(index).cast());
            // high 2^52 + low = h1 2^104 + h0 2^52 + l0 with h1 < 2^12.
            let high = _mm512_add_epi64(high, _mm512_srli_epi64::<52>(low));
            let x = _mm512_add_epi64(
                _mm512_add_epi64(
                    radix.mul(_mm512_and_si512(high, m.mask), m),
                    radix_squared.mul(_mm512_srli_epi64::<52>(high), m),
                ),
                _mm512_and_si512(low, m.mask),
            );
            // x < 2p + 2p + 2^52 < 12p, since p > 2^49.
            let x = _mm512_min_epu64(x, _mm512_sub_epi64(x, eight_p));
            let x = _mm512_min_epu64(x, _mm512_sub_epi64(x, four_p));
            let x = m.reduce_2p(m.reduce_4p(x));
            _mm512_store_si512(out.0.as_mut_ptr().add(index).cast(), x);
        }
    }
    out
}
