//! x86 negacyclic and cyclic NTTs for one `i32` or `i16` CRT limb.
//!
//! The forward transform is Gentleman–Sande DIF and the inverse is
//! Cooley–Tukey DIT, both over bit-reversed evaluations, exactly as in the
//! scalar `butterfly` module. The passes are:
//!
//! - Forward: one radix-4 pass per two stages from `D/2` down, the first of
//!   which also applies the negacyclic twist (or converts signed `i8` digits
//!   or centered `i16` coefficients), then
//!   a 256-bit tail that runs the last four stages (`log2 D` even) or five
//!   stages (`log2 D` odd) in registers.
//! - Inverse: the mirror-image 256-bit head, then radix-4 passes, the last of
//!   which also applies the untwist and `D^{-1}` scaling.
//!
//! Every radix-4 pass therefore has a butterfly half-length of at least 16, so
//! the passes are generic over [`Lanes`]: `i32` runs at 256 or 512 bits and
//! `i16` at 256 bits. Forward passes store sums in `[0, 2p)` and products in
//! `(-p, p)`; inverse passes store `[0, 2p)`. See [`super::lanes`]. Constant
//! multiplies use the precomputed quotients in [`super::twiddles`]. The tail
//! and head are specific to the element width, see [`Width`].
//!
//! Range contracts: the negacyclic forward transform accepts any input, the
//! cyclic forward transform and both inverses accept `(-p, p)`. Forward output
//! is canonical `[0, p)`; inverse output lies in `(-p, p)`. Degrees below 64
//! use the scalar transforms.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::lanes::{I16Multiplier, I16x16, I32Multiplier, Lanes, Modulus, Multiply};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};

/// A constant table with its Montgomery quotients.
#[derive(Clone, Copy)]
struct Table<E> {
    values: *const E,
    quotients: *const E,
}

impl<E: Copy> Table<E> {
    fn new<T, const D: usize>(values: &[T; D], quotients: &[E; D]) -> Self {
        Self {
            values: values.as_ptr().cast(),
            quotients: quotients.as_ptr(),
        }
    }

    #[inline(always)]
    unsafe fn load<V: Lanes<Elem = E>>(self, index: usize) -> V::Multiplier {
        // SAFETY: the caller keeps `index + LANES` within the table.
        unsafe {
            V::Multiplier::new(
                V::load(self.values.add(index)),
                V::load(self.quotients.add(index)),
            )
        }
    }
}

/// The register-resident ends of the transform for one element width.
trait Width: PrimeWidth {
    /// Last four (or, with `FIVE`, five) forward stages, 32 values at a time,
    /// from the last pass's `[0, 2p)` sums and `(-p, p)` products to canonical
    /// `[0, p)`.
    unsafe fn forward_tail<const D: usize, const FIVE: bool>(
        a: *mut Self,
        fwd: Table<Self>,
        p: Self,
    );

    /// First four (or, with `FIVE`, five) inverse stages, 32 values at a time,
    /// from `(-p, p)` to `[0, 2p)`.
    unsafe fn inverse_head<const D: usize, const FIVE: bool>(
        a: *mut Self,
        inv: Table<Self>,
        p: Self,
    );
}

/// Stages `len` and `len / 2` of the forward DIF transform on the four
/// quarters at `start + {0, len/2, len, 3len/2}`, read through `load` and
/// stored to `a`. Sums return in `[0, 2p)` (quarters 0 and 2) and products in
/// `(-p, p)` (quarters 1 and 3).
///
/// `SIGNED` inputs lie in `(-p, p)`; otherwise inputs lie in `[0, 2p)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn forward_radix4<V: Lanes, const SIGNED: bool>(
    a: *mut V::Elem,
    start: usize,
    len: usize,
    load: &impl Fn(usize) -> V,
    outer0: V::Multiplier,
    outer1: V::Multiplier,
    inner: V::Multiplier,
    m: Modulus<V>,
) {
    let index = [start, start + len / 2, start + len, start + len + len / 2];
    // SAFETY: the caller keeps every quarter in bounds, and its target
    // features cover `V`.
    unsafe {
        let q = [
            load(index[0]),
            load(index[1]),
            load(index[2]),
            load(index[3]),
        ];
        let e0 = outer0.mul(q[0].sub(q[2]), m);
        let e1 = outer1.mul(q[1].sub(q[3]), m);
        let (x, y) = if SIGNED {
            (m.fold(q[0].add(q[2])), m.fold(q[1].add(q[3])))
        } else {
            (m.reduce_4p(q[0].add(q[2])), m.reduce_4p(q[1].add(q[3])))
        };
        m.reduce_4p(x.add(y)).store(a.add(index[0]));
        inner.mul(x.sub(y), m).store(a.add(index[1]));
        m.fold(e0.add(e1)).store(a.add(index[2]));
        inner.mul(e0.sub(e1), m).store(a.add(index[3]));
    }
}

/// One forward radix-4 pass over stages `len` and `len / 2`. `load(i)` reads
/// the input vector at index `i`, which lets the first pass twist or convert
/// its inputs on the way in.
///
/// The `FIRST` pass reads `(-p, p)` inputs. A later pass reads the quarters
/// of the previous pass, which alternate between `[0, 2p)` sums and `(-p, p)`
/// products every `2 len` values.
#[inline(always)]
unsafe fn forward_pass<V: Lanes, const D: usize, const FIRST: bool>(
    a: *mut V::Elem,
    len: usize,
    fwd: Table<V::Elem>,
    m: Modulus<V>,
    load: impl Fn(usize) -> V,
) {
    let half = len / 2;
    let mut j = 0;
    while j < half {
        // SAFETY: `len <= D/2` is a stage length and `half >= LANES`, so every
        // twiddle and data index below is in bounds.
        unsafe {
            let outer0 = fwd.load::<V>(len - 1 + j);
            let outer1 = fwd.load::<V>(len - 1 + half + j);
            let inner = fwd.load::<V>(half - 1 + j);
            if FIRST {
                forward_radix4::<V, true>(a, j, len, &load, outer0, outer1, inner, m);
            } else {
                let mut start = j;
                while start < D {
                    forward_radix4::<V, false>(a, start, len, &load, outer0, outer1, inner, m);
                    forward_radix4::<V, true>(
                        a,
                        start + 2 * len,
                        len,
                        &load,
                        outer0,
                        outer1,
                        inner,
                        m,
                    );
                    start += 4 * len;
                }
            }
        }
        j += V::LANES;
    }
}

/// Radix-2 DIF butterfly whose inputs share, per lane, the range selected by
/// `offset` (see [`Modulus::sum_offsets`]). The sum returns in `[0, 2p)` and
/// the product in `(-p, p)`.
#[inline(always)]
unsafe fn dif<V: Lanes>(u: V, v: V, w: V::Multiplier, offset: V, m: Modulus<V>) -> (V, V) {
    // SAFETY: the caller's target features cover `V`.
    unsafe { (m.reduce_sum(u.add(v), offset), w.mul(u.sub(v), m)) }
}

/// Radix-2 DIT butterfly on `[0, 2p)` inputs with `[0, 2p)` outputs.
#[inline(always)]
unsafe fn dit<V: Lanes>(u: V, v: V, w: V::Multiplier, m: Modulus<V>) -> (V, V) {
    // SAFETY: the caller's target features cover `V`.
    unsafe {
        let u = u.sub(m.p);
        let v = w.mul(v, m);
        (m.fold(u.add(v)), m.fold(u.sub(v)))
    }
}

/// Split two vectors into their low and high 128-bit halves:
/// `([x.lo, y.lo], [x.hi, y.hi])`. The map is its own inverse.
#[inline(always)]
unsafe fn swap_halves(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        (
            _mm256_permute2x128_si256::<0x20>(x, y),
            _mm256_permute2x128_si256::<0x31>(x, y),
        )
    }
}

/// Transpose the 4×4 `i32` blocks held in each 128-bit lane of four vectors.
#[inline(always)]
unsafe fn transpose4(y: [__m256i; 4]) -> [__m256i; 4] {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        let t0 = _mm256_unpacklo_epi32(y[0], y[1]);
        let t1 = _mm256_unpacklo_epi32(y[2], y[3]);
        let t2 = _mm256_unpackhi_epi32(y[0], y[1]);
        let t3 = _mm256_unpackhi_epi32(y[2], y[3]);
        [
            _mm256_unpacklo_epi64(t0, t1),
            _mm256_unpackhi_epi64(t0, t1),
            _mm256_unpacklo_epi64(t2, t3),
            _mm256_unpackhi_epi64(t2, t3),
        ]
    }
}

/// Broadcast four consecutive `i32` table entries to both 128-bit lanes.
#[inline(always)]
unsafe fn load_broadcast4(table: Table<i32>, index: usize) -> I32Multiplier<__m256i> {
    // SAFETY: the caller keeps `index + 4` within the table; AVX2 is enabled.
    unsafe {
        I32Multiplier::new(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.values.add(index).cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.quotients.add(index).cast())),
        )
    }
}

impl Width for i32 {
    #[inline(always)]
    unsafe fn forward_tail<const D: usize, const FIVE: bool>(a: *mut i32, fwd: Table<i32>, p: i32) {
        // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and
        // every block of 32 values is in bounds.
        unsafe {
            let m = Modulus::<__m256i>::splat(p);
            let t16 = [fwd.load::<__m256i>(15), fwd.load::<__m256i>(23)];
            let t8 = fwd.load::<__m256i>(7);
            let t4 = load_broadcast4(fwd, 3);
            let t2 = I32Multiplier::new(
                __m256i::splat(*fwd.values.add(2)),
                __m256i::splat(*fwd.quotients.add(2)),
            );
            // The last pass leaves `[0, 2p)` sums and `(-p, p)` products in
            // alternating quarters: 32 values each for `FIVE`, else 16. Every
            // stage below keeps that pairing, so each sum reduces with the
            // offset of its lane.
            let (unsigned, signed) = m.sum_offsets();
            let lower_unsigned = swap_halves(unsigned, signed).0;
            let mut base = 0;
            while base < D {
                for (half, offset16) in [(0, unsigned), (32, signed)] {
                    let p = a.add(base + half);
                    let mut y = [
                        __m256i::load(p),
                        __m256i::load(p.add(8)),
                        __m256i::load(p.add(16)),
                        __m256i::load(p.add(24)),
                    ];
                    if FIVE {
                        (y[0], y[2]) = dif(y[0], y[2], t16[0], offset16, m);
                        (y[1], y[3]) = dif(y[1], y[3], t16[1], offset16, m);
                    }
                    (y[0], y[1]) = dif(y[0], y[1], t8, unsigned, m);
                    (y[2], y[3]) = dif(y[2], y[3], t8, signed, m);
                    for k in [0, 2] {
                        let (u, v) = swap_halves(y[k], y[k + 1]);
                        let (u, v) = dif(u, v, t4, lower_unsigned, m);
                        (y[k], y[k + 1]) = swap_halves(u, v);
                    }

                    // Each lane position now holds one 4-point group `r0..r3`
                    // whose values share a range: `[0, 2p)` in the low
                    // 128-bit lane and `(-p, p)` in the high one. Its stage-2
                    // multiply by `fwd[1]` and stage-1 multiplies by `fwd[0]`
                    // are by the Montgomery one and are skipped.
                    let [r0, r1, r2, r3] = transpose4(y);
                    let s0 = m.canonical(m.reduce_sum(r0.add(r2), lower_unsigned));
                    let s1 = m.canonical(m.reduce_sum(r1.add(r3), lower_unsigned));
                    let d0 = m.canonical(m.fold(r0.sub(r2)));
                    let d1 = m.canonical_signed(t2.mul(r1.sub(r3), m));
                    let out = transpose4([
                        m.canonical(s0.add(s1)),
                        m.canonical_signed(s0.sub(s1)),
                        m.canonical(d0.add(d1)),
                        m.canonical_signed(d0.sub(d1)),
                    ]);
                    for (k, value) in out.into_iter().enumerate() {
                        value.store(p.add(8 * k));
                    }
                }
                base += 64;
            }
        }
    }

    #[inline(always)]
    unsafe fn inverse_head<const D: usize, const FIVE: bool>(a: *mut i32, inv: Table<i32>, p: i32) {
        // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and
        // every block of 32 values is in bounds.
        unsafe {
            let m = Modulus::<__m256i>::splat(p);
            let t2 = I32Multiplier::new(
                __m256i::splat(*inv.values.add(2)),
                __m256i::splat(*inv.quotients.add(2)),
            );
            let t4 = load_broadcast4(inv, 3);
            let t8 = inv.load::<__m256i>(7);
            let t16 = [inv.load::<__m256i>(15), inv.load::<__m256i>(23)];
            let mut base = 0;
            while base < D {
                let p = a.add(base);
                let [r0, r1, r2, r3] = transpose4([
                    __m256i::load(p),
                    __m256i::load(p.add(8)),
                    __m256i::load(p.add(16)),
                    __m256i::load(p.add(24)),
                ]);
                let a0 = m.fold(r0.add(r1));
                let a1 = m.fold(r0.sub(r1)).sub(m.p);
                let a2 = m.fold(r2.add(r3));
                let v = t2.mul(r2.sub(r3), m);
                let mut y = transpose4([
                    m.reduce_4p(a0.add(a2)),
                    m.fold(a1.add(v)),
                    m.fold(a0.sub(a2)),
                    m.fold(a1.sub(v)),
                ]);
                for k in [0, 2] {
                    let (u, v) = swap_halves(y[k], y[k + 1]);
                    let (u, v) = dit(u, v, t4, m);
                    (y[k], y[k + 1]) = swap_halves(u, v);
                }
                (y[0], y[1]) = dit(y[0], y[1], t8, m);
                (y[2], y[3]) = dit(y[2], y[3], t8, m);
                if FIVE {
                    (y[0], y[2]) = dit(y[0], y[2], t16[0], m);
                    (y[1], y[3]) = dit(y[1], y[3], t16[1], m);
                }
                for (k, value) in y.into_iter().enumerate() {
                    value.store(p.add(8 * k));
                }
                base += 32;
            }
        }
    }
}

/// Exchange the high 64 bits of each 128-bit lane of `x` with the low 64 bits
/// of the same lane of `y`. The map is its own inverse.
#[inline(always)]
unsafe fn swap_quads(x: I16x16, y: I16x16) -> (I16x16, I16x16) {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        (
            I16x16(_mm256_unpacklo_epi64(x.0, y.0)),
            I16x16(_mm256_unpackhi_epi64(x.0, y.0)),
        )
    }
}

/// Exchange the high 32 bits of each 64-bit lane of `x` with the low 32 bits
/// of the same lane of `y`. The map is its own inverse.
#[inline(always)]
unsafe fn swap_pairs(x: I16x16, y: I16x16) -> (I16x16, I16x16) {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        (
            I16x16(_mm256_blend_epi32::<0xAA>(
                x.0,
                _mm256_slli_epi64::<32>(y.0),
            )),
            I16x16(_mm256_blend_epi32::<0xAA>(
                _mm256_srli_epi64::<32>(x.0),
                y.0,
            )),
        )
    }
}

/// Exchange the high 16 bits of each 32-bit lane of `x` with the low 16 bits
/// of the same lane of `y`. The map is its own inverse.
#[inline(always)]
unsafe fn swap_words(x: I16x16, y: I16x16) -> (I16x16, I16x16) {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        (
            I16x16(_mm256_blend_epi16::<0xAA>(
                x.0,
                _mm256_slli_epi32::<16>(y.0),
            )),
            I16x16(_mm256_blend_epi16::<0xAA>(
                _mm256_srli_epi32::<16>(x.0),
                y.0,
            )),
        )
    }
}

/// Repeat the `i16` table entries `index..index + len` across a vector, for
/// `len` of 2, 4, or 8.
#[inline(always)]
unsafe fn load_repeated(table: Table<i16>, index: usize, len: usize) -> I16Multiplier {
    #[inline(always)]
    unsafe fn repeat(ptr: *const i16, len: usize) -> I16x16 {
        // SAFETY: forwarded from `load_repeated`.
        unsafe {
            I16x16(match len {
                2 => _mm256_set1_epi32(ptr.cast::<i32>().read_unaligned()),
                4 => _mm256_set1_epi64x(ptr.cast::<i64>().read_unaligned()),
                _ => _mm256_broadcastsi128_si256(_mm_loadu_si128(ptr.cast())),
            })
        }
    }
    // SAFETY: the caller keeps `index + len` within the table; AVX2 is enabled.
    unsafe {
        I16Multiplier::new(
            repeat(table.values.add(index), len),
            repeat(table.quotients.add(index), len),
        )
    }
}

/// The `i16` tail and head keep 32 values in two vectors `x` and `y`. Stage 16
/// pairs `x` with `y`. Before each shorter stage a `swap_*` map regroups the
/// values so every butterfly pairs lane `i` of one vector with lane `i` of the
/// other: after [`swap_halves`] the stage-8 halves, after [`swap_quads`] the
/// stage-4 halves, after [`swap_pairs`] the stage-2 halves, and after
/// [`swap_words`] the even and odd values.
impl Width for i16 {
    #[inline(always)]
    unsafe fn forward_tail<const D: usize, const FIVE: bool>(a: *mut i16, fwd: Table<i16>, p: i16) {
        // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and
        // every block of 32 values is in bounds.
        unsafe {
            let m = Modulus::<I16x16>::splat(p);
            let t16 = fwd.load::<I16x16>(15);
            let t8 = load_repeated(fwd, 7, 8);
            let t4 = load_repeated(fwd, 3, 4);
            let t2 = load_repeated(fwd, 1, 2);
            // The last pass leaves `[0, 2p)` sums and `(-p, p)` products in
            // alternating quarters: 32 values each for `FIVE`, else 16. Each
            // `swap_*` map keeps a stage's sums in the first vector and its
            // products in the second, so applying it to the offsets gives the
            // per-lane sum offset of the next stage.
            let (unsigned, signed) = m.sum_offsets();
            let (o8, _) = swap_halves(unsigned.0, signed.0);
            let (o4, _) = swap_quads(unsigned, signed);
            let (o2, _) = swap_pairs(unsigned, signed);
            let (o1, _) = swap_words(unsigned, signed);
            let mut base = 0;
            while base < D {
                for (half, offset16) in [(0, unsigned), (32, signed)] {
                    let p = a.add(base + half);
                    let (mut x, mut y) = (I16x16::load(p), I16x16::load(p.add(16)));
                    if FIVE {
                        (x, y) = dif(x, y, t16, offset16, m);
                    }
                    let (u, v) = swap_halves(x.0, y.0);
                    let (u, v) = dif(I16x16(u), I16x16(v), t8, I16x16(o8), m);
                    let (u, v) = swap_quads(u, v);
                    let (u, v) = dif(u, v, t4, o4, m);
                    let (u, v) = swap_pairs(u, v);
                    let (u, v) = dif(u, v, t2, o2, m);
                    let (u, v) = swap_words(u, v);
                    // Stage 1 multiplies by `fwd[0]`, the Montgomery one.
                    let even = m.canonical(m.reduce_sum(u.add(v), o1));
                    let odd = m.canonical(m.fold(u.sub(v)));
                    let (x, y) = swap_halves(
                        _mm256_unpacklo_epi16(even.0, odd.0),
                        _mm256_unpackhi_epi16(even.0, odd.0),
                    );
                    I16x16(x).store(p);
                    I16x16(y).store(p.add(16));
                }
                base += 64;
            }
        }
    }

    #[inline(always)]
    unsafe fn inverse_head<const D: usize, const FIVE: bool>(a: *mut i16, inv: Table<i16>, p: i16) {
        // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and
        // every block of 32 values is in bounds.
        unsafe {
            let m = Modulus::<I16x16>::splat(p);
            let t2 = load_repeated(inv, 1, 2);
            let t4 = load_repeated(inv, 3, 4);
            let t8 = load_repeated(inv, 7, 8);
            let t16 = inv.load::<I16x16>(15);
            // Gathers the even values of each 128-bit lane into its low 64
            // bits and the odd values into its high 64 bits.
            let split = _mm256_setr_epi8(
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2,
                3, 6, 7, 10, 11, 14, 15,
            );
            let mut base = 0;
            while base < D {
                let p = a.add(base);
                let (x, y) = swap_halves(
                    _mm256_loadu_si256(p.cast()),
                    _mm256_loadu_si256(p.add(16).cast()),
                );
                let (u, v) = swap_quads(
                    I16x16(_mm256_shuffle_epi8(x, split)),
                    I16x16(_mm256_shuffle_epi8(y, split)),
                );
                // Stage 1 multiplies by `inv[0]`, the Montgomery one.
                let (u, v) = (m.fold(u.add(v)), m.fold(u.sub(v)));
                let (u, v) = swap_words(u, v);
                let (u, v) = dit(u, v, t2, m);
                let (u, v) = swap_pairs(u, v);
                let (u, v) = dit(u, v, t4, m);
                let (u, v) = swap_quads(u, v);
                let (u, v) = dit(u, v, t8, m);
                let (x, y) = swap_halves(u.0, v.0);
                let (mut x, mut y) = (I16x16(x), I16x16(y));
                if FIVE {
                    (x, y) = dit(x, y, t16, m);
                }
                x.store(p);
                y.store(p.add(16));
                base += 32;
            }
        }
    }
}

/// One inverse radix-4 pass over stages `len` and `2 len`, from `[0, 2p)`.
/// Each output in `(-2p, 2p)` is written as `finish(index, output)`.
#[inline(always)]
unsafe fn inverse_pass<V: Lanes, const D: usize>(
    a: *mut V::Elem,
    len: usize,
    inv: Table<V::Elem>,
    m: Modulus<V>,
    finish: impl Fn(usize, V) -> V,
) {
    let mut j = 0;
    while j < len {
        // SAFETY: `len <= D/4` is a stage length and `len >= LANES`, so every
        // twiddle and data index below is in bounds.
        unsafe {
            let inner = inv.load::<V>(len - 1 + j);
            let outer0 = inv.load::<V>(2 * len - 1 + j);
            let outer1 = inv.load::<V>(3 * len - 1 + j);
            let mut start = j;
            while start < D {
                let p = a.add(start);
                let q0 = V::load(p).add(m.p);
                let q1 = inner.mul(V::load(p.add(len)), m);
                let q2 = V::load(p.add(2 * len)).sub(m.p);
                let q3 = inner.mul(V::load(p.add(3 * len)), m);
                let sum0 = m.reduce_4p(q0.add(q1)).sub(m.p);
                let diff0 = m.reduce_4p(q0.sub(q1)).sub(m.p);
                let sum1 = outer0.mul(q2.add(q3), m);
                let diff1 = outer1.mul(q2.sub(q3), m);
                finish(start, sum0.add(sum1)).store(p);
                finish(start + len, diff0.add(diff1)).store(p.add(len));
                finish(start + 2 * len, sum0.sub(sum1)).store(p.add(2 * len));
                finish(start + 3 * len, diff0.sub(diff1)).store(p.add(3 * len));
                start += 4 * len;
            }
        }
        j += V::LANES;
    }
}

/// Forward DIF transform whose first pass reads its inputs through
/// `load_first`, which must return values in `(-p, p)`.
#[inline(always)]
unsafe fn forward<V: Lanes, const D: usize>(
    a: *mut V::Elem,
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
    load_first: impl Fn(usize) -> V,
) where
    V::Elem: Width,
{
    debug_assert!(D >= 64 && D.is_power_of_two());
    // SAFETY: the caller's target features cover `V` and AVX2, and `a` holds
    // `D` values.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        let fwd = Table::new(&tw.fwd_twiddles, &tw.quotients.fwd);
        forward_pass::<V, D, true>(a, D / 2, fwd, m, load_first);
        let five = D.trailing_zeros() % 2 == 1;
        let tail_len = if five { 16 } else { 8 };
        let mut len = D / 8;
        while len > tail_len {
            forward_pass::<V, D, false>(
                a,
                len,
                fwd,
                m,
                #[inline(always)]
                |i| V::load(a.add(i)),
            );
            len /= 4;
        }
        if five {
            <V::Elem as Width>::forward_tail::<D, true>(a, fwd, prime.p);
        } else {
            <V::Elem as Width>::forward_tail::<D, false>(a, fwd, prime.p);
        }
    }
}

/// Inverse DIT transform whose last pass writes `finish(index, output)`.
#[inline(always)]
unsafe fn inverse<V: Lanes, const D: usize>(
    a: *mut V::Elem,
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
    finish: impl Fn(usize, V) -> V,
) where
    V::Elem: Width,
{
    debug_assert!(D >= 64 && D.is_power_of_two());
    // SAFETY: the caller's target features cover `V` and AVX2, and `a` holds
    // `D` values.
    unsafe {
        let inv = Table::new(&tw.inv_twiddles, &tw.quotients.inv);
        let five = D.trailing_zeros() % 2 == 1;
        if five {
            <V::Elem as Width>::inverse_head::<D, true>(a, inv, prime.p);
        } else {
            <V::Elem as Width>::inverse_head::<D, false>(a, inv, prime.p);
        }
        let m = Modulus::<V>::splat(prime.p);
        let mut len = if five { 32 } else { 16 };
        while len < D / 4 {
            inverse_pass::<V, D>(
                a,
                len,
                inv,
                m,
                #[inline(always)]
                |_, x| m.fold(x),
            );
            len *= 4;
        }
        inverse_pass::<V, D>(a, len, inv, m, finish);
    }
}

#[inline(always)]
unsafe fn forward_negacyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let psi = Table::new(&tw.psi_pows, &tw.quotients.psi);
    let a = a.as_mut_ptr().cast::<V::Elem>();
    // SAFETY: inherited from the caller; `psi` has `D` entries.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        forward::<V, D>(
            a,
            prime,
            tw,
            #[inline(always)]
            |i| psi.load::<V>(i).mul(V::load(a.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_digits<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    digits: &[i8; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let psi_r2 = Table::new(&tw.psi_pows_r2, &tw.quotients.psi_r2);
    let digits = digits.as_ptr();
    // SAFETY: inherited from the caller; `digits` and `psi_r2` have `D` entries.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        forward::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i| psi_r2.load::<V>(i).mul(V::load_i8(digits.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_cyclic_digits<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    digits: &[i8; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let digits = digits.as_ptr();
    // SAFETY: inherited from the caller; `digits` has `D` entries.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        let r2 = V::Multiplier::new(
            V::splat(prime.montsq),
            V::splat(prime.montsq.wrapping_mul(prime.pinv)),
        );
        forward::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i| r2.mul(V::load_i8(digits.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_centered_i16<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    coefficients: &[i16; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let psi_r2 = Table::new(&tw.psi_pows_r2, &tw.quotients.psi_r2);
    let coefficients = coefficients.as_ptr();
    // SAFETY: inherited from the caller; `coefficients` and `psi_r2` have `D`
    // entries.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        forward::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i| psi_r2.load::<V>(i).mul(V::load_i16(coefficients.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_cyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let a = a.as_mut_ptr().cast::<V::Elem>();
    // SAFETY: inherited from the caller.
    unsafe {
        forward::<V, D>(
            a,
            prime,
            tw,
            #[inline(always)]
            |i| V::load(a.add(i)),
        )
    }
}

#[inline(always)]
unsafe fn inverse_negacyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    let untwist = Table::new(&tw.d_inv_psi_inv, &tw.quotients.d_inv_psi_inv);
    // SAFETY: inherited from the caller; `untwist` has `D` entries.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        inverse::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i, x| untwist.load::<V>(i).mul(x, m),
        )
    }
}

#[inline(always)]
unsafe fn inverse_cyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<V::Elem>; D],
    prime: NttPrime<V::Elem>,
    tw: &NttTwiddles<V::Elem, D>,
) where
    V::Elem: Width,
{
    // SAFETY: inherited from the caller.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        let d_inv = V::Multiplier::new(V::splat(tw.d_inv.raw()), V::splat(tw.quotients.d_inv));
        inverse::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |_, x| d_inv.mul(x, m),
        )
    }
}

macro_rules! x86_transforms {
    ($(
        $(#[$doc:meta])*
        $generic:ident, ($($arg:ident: $ty:ty),*) =>
            $name:ident, $avx512:ident, $name_i16:ident;
    )*) => {$(
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The caller must ensure AVX2 is available, and AVX-512F/DQ/BW when
        /// `use_avx512` is set. `D` must be a power of two of at least 64.
        #[target_feature(enable = "avx2")]
        pub(crate) unsafe fn $name<const D: usize>(
            a: &mut [MontCoeff<i32>; D],
            $($arg: $ty,)*
            prime: NttPrime<i32>,
            tw: &NttTwiddles<i32, D>,
            use_avx512: bool,
        ) {
            if use_avx512 {
                // SAFETY: forwarded from this function's contract.
                unsafe { $avx512(a, $($arg,)* prime, tw) }
            } else {
                // SAFETY: forwarded from this function's contract.
                unsafe { $generic::<__m256i, D>(a, $($arg,)* prime, tw) }
            }
        }

        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
        unsafe fn $avx512<const D: usize>(
            a: &mut [MontCoeff<i32>; D],
            $($arg: $ty,)*
            prime: NttPrime<i32>,
            tw: &NttTwiddles<i32, D>,
        ) {
            // SAFETY: forwarded from the caller's contract.
            unsafe { $generic::<__m512i, D>(a, $($arg,)* prime, tw) }
        }

        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The caller must ensure AVX2 is available. `D` must be a power of
        /// two of at least 64.
        #[target_feature(enable = "avx2")]
        pub(crate) unsafe fn $name_i16<const D: usize>(
            a: &mut [MontCoeff<i16>; D],
            $($arg: $ty,)*
            prime: NttPrime<i16>,
            tw: &NttTwiddles<i16, D>,
        ) {
            // SAFETY: forwarded from this function's contract.
            unsafe { $generic::<I16x16, D>(a, $($arg,)* prime, tw) }
        }
    )*};
}

x86_transforms! {
    /// Forward negacyclic NTT.
    forward_negacyclic, () => forward_ntt_i32, forward_ntt_i32_avx512, forward_ntt_i16;
    /// Signed-digit conversion and forward negacyclic NTT. The `psi^i R^2`
    /// twist enters Montgomery form and twists in one product.
    forward_digits, (digits: &[i8; D]) =>
        forward_ntt_i8_i32, forward_ntt_i8_i32_avx512, forward_ntt_i8_i16;
    /// Centered-i16 conversion and forward negacyclic NTT, through the same
    /// `psi^i R^2` product as the signed-digit entry.
    forward_centered_i16, (coefficients: &[i16; D]) =>
        forward_ntt_centered_i16_i32, forward_ntt_centered_i16_i32_avx512,
        forward_ntt_centered_i16_i16;
    /// Inverse negacyclic NTT.
    inverse_negacyclic, () => inverse_ntt_i32, inverse_ntt_i32_avx512, inverse_ntt_i16;
    /// Signed-digit conversion and forward cyclic NTT. One Montgomery product
    /// by `R^2` enters Montgomery form on the first pass's loads.
    forward_cyclic_digits, (digits: &[i8; D]) =>
        forward_ntt_cyclic_i8_i32, forward_ntt_cyclic_i8_i32_avx512, forward_ntt_cyclic_i8_i16;
    /// Forward cyclic NTT.
    forward_cyclic, () =>
        forward_ntt_cyclic_i32, forward_ntt_cyclic_i32_avx512, forward_ntt_cyclic_i16;
    /// Inverse cyclic NTT.
    inverse_cyclic, () =>
        inverse_ntt_cyclic_i32, inverse_ntt_cyclic_i32_avx512, inverse_ntt_cyclic_i16;
}
