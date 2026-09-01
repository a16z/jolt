//! Per-ISA SIMD primitive vocabularies: [`SimdWord`] is the instruction-set
//! contract the shared packed algebra (`engine.rs`, `fp128.rs`) is written
//! against, implemented by the [`Neon`], [`Avx2`], and [`Avx512`] markers.
//!
//! Only *algorithmic* per-ISA differences live here — e.g. AVX2 has no
//! 64-bit widening multiply (emulated from 32×32→64 partial products),
//! AVX-512 comparisons produce mask registers (converted to lane masks),
//! and only NEON has a 32-bit high-multiply (`mul_pm31`). Comparison
//! results are all-ones lane masks on every ISA.

#![cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]

/// One SIMD instruction set's primitive vocabulary over vectors of `u32`
/// and `u64` lanes.
///
/// # Invariants
///
/// - `lt_*` return all-ones lane masks (unsigned compare); `select*` takes
///   such a mask and picks `t`/`f` lane-wise.
/// - `mul_small`/`mul_small_wide` require `c < 2^32`.
/// - Shift amounts satisfy `0 < k < 64`.
/// - `narrow_pack` is the layout inverse of `widen_mul`: packing the
///   (reduced) halves restores the original lane order.
pub trait SimdWord: 'static {
    /// Vector of [`W32`](Self::W32) `u32` lanes.
    type V32: Copy + Send + Sync;
    /// Vector of [`W64`](Self::W64) `u64` lanes.
    type V64: Copy + Send + Sync;
    /// `u32` lanes per vector.
    const W32: usize;
    /// `u64` lanes per vector.
    const W64: usize;
    /// Whether packed `Fp64` multiplication should go lane-by-lane through
    /// the scalar kernel (no efficient 64×64 vector multiply on this ISA).
    const FP64_MUL_BY_LANES: bool;

    fn v32_from_fn(f: impl FnMut(usize) -> u32) -> Self::V32;
    fn v32_lane(v: Self::V32, lane: usize) -> u32;
    fn v64_from_fn(f: impl FnMut(usize) -> u64) -> Self::V64;
    fn v64_lane(v: Self::V64, lane: usize) -> u64;
    fn splat32(x: u32) -> Self::V32;
    fn splat64(x: u64) -> Self::V64;
    fn add32(a: Self::V32, b: Self::V32) -> Self::V32;
    fn sub32(a: Self::V32, b: Self::V32) -> Self::V32;
    fn min_u32(a: Self::V32, b: Self::V32) -> Self::V32;
    fn lt_u32(a: Self::V32, b: Self::V32) -> Self::V32;
    fn select32(m: Self::V32, t: Self::V32, f: Self::V32) -> Self::V32;
    fn add64(a: Self::V64, b: Self::V64) -> Self::V64;
    fn sub64(a: Self::V64, b: Self::V64) -> Self::V64;
    fn and64(a: Self::V64, b: Self::V64) -> Self::V64;
    fn or64(a: Self::V64, b: Self::V64) -> Self::V64;
    fn shr64(v: Self::V64, k: u32) -> Self::V64;
    fn shl64(v: Self::V64, k: u32) -> Self::V64;
    fn lt_u64(a: Self::V64, b: Self::V64) -> Self::V64;
    fn select64(m: Self::V64, t: Self::V64, f: Self::V64) -> Self::V64;
    /// Low 64 bits of `v * c` per lane.
    fn mul_small(v: Self::V64, c: u64) -> Self::V64;
    /// `[lo, hi]` of the full `v * c` per lane.
    fn mul_small_wide(v: Self::V64, c: u64) -> [Self::V64; 2];
    /// `[lo, hi]` of the full 128-bit `a * b` per lane.
    fn mul64_wide(a: Self::V64, b: Self::V64) -> [Self::V64; 2];
    /// 32×32→64 widening multiply of all `u32` lanes; the two half-vectors
    /// are in an ISA-specific layout (see `narrow_pack`).
    fn widen_mul(a: Self::V32, b: Self::V32) -> [Self::V64; 2];
    /// Low 32 bits of each 64-bit lane, reassembled in original lane order.
    fn narrow_pack(x: [Self::V64; 2]) -> Self::V32;

    /// Multiply for 31-bit pseudo-Mersenne primes `p = 2^31 − c` entirely in
    /// 32-bit lanes, when the ISA has a 32-bit high-multiply. Inputs must be
    /// canonical and `p` must be exactly 31 bits with `c(c+1) < p`.
    #[inline(always)]
    fn mul_pm31(_a: Self::V32, _b: Self::V32, _p: u32, _c: u32) -> Option<Self::V32> {
        None
    }

    /// Scalar-lane multiply for 63-bit pseudo-Mersenne primes when the ISA
    /// has a dedicated carry-preserving sequence.
    #[inline(always)]
    fn mul_pm63(_a: u64, _b: u64, _p: u64, _c: u64) -> Option<u64> {
        None
    }
}

/// Stamps vocabulary methods whose body is a single (possibly block)
/// intrinsic expression, wrapped in the requisite `unsafe` block.
macro_rules! fwd {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $body:expr;)*) => {
        $(
            #[inline(always)]
            fn $name($($arg: $ty),*) -> $ret {
                unsafe { $body }
            }
        )*
    };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::Neon;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "register-only NEON intrinsics over plain integer lanes; the per-fn contracts are documented on the SimdWord trait"
)]
mod neon {
    use super::SimdWord;
    use core::arch::aarch64::{
        uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vandq_u32, vandq_u64, vbslq_u32, vbslq_u64,
        vcltq_u32, vcltq_u64, vcombine_u32, vdup_n_u32, vdupq_n_s64, vdupq_n_u32, vdupq_n_u64,
        vget_low_u32, vminq_u32, vmovn_u64, vmull_high_u32, vmull_u32, vmulq_u32, vorrq_u64,
        vqdmulhq_s32, vreinterpretq_s32_u32, vreinterpretq_u32_s32, vshlq_n_u64, vshlq_u64,
        vshrq_n_u32, vshrq_n_u64, vsubq_u32, vsubq_u64,
    };
    use core::mem::transmute;

    /// AArch64 NEON: 128-bit vectors (4 × u32, 2 × u64).
    pub enum Neon {}

    impl SimdWord for Neon {
        type V32 = uint32x4_t;
        type V64 = uint64x2_t;
        const W32: usize = 4;
        const W64: usize = 2;
        // No 64×64 vector multiply: per-lane scalar folds win at width 2.
        const FP64_MUL_BY_LANES: bool = true;

        fwd! {
            v32_from_fn(f: impl FnMut(usize) -> u32) -> uint32x4_t =
                transmute::<[u32; 4], uint32x4_t>(std::array::from_fn(f));
            v32_lane(v: uint32x4_t, lane: usize) -> u32 =
                transmute::<uint32x4_t, [u32; 4]>(v)[lane];
            v64_from_fn(f: impl FnMut(usize) -> u64) -> uint64x2_t =
                transmute::<[u64; 2], uint64x2_t>(std::array::from_fn(f));
            v64_lane(v: uint64x2_t, lane: usize) -> u64 =
                transmute::<uint64x2_t, [u64; 2]>(v)[lane];
            splat32(x: u32) -> uint32x4_t = vdupq_n_u32(x);
            splat64(x: u64) -> uint64x2_t = vdupq_n_u64(x);
            add32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t = vaddq_u32(a, b);
            sub32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t = vsubq_u32(a, b);
            min_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t = vminq_u32(a, b);
            lt_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t = vcltq_u32(a, b);
            select32(m: uint32x4_t, t: uint32x4_t, f: uint32x4_t) -> uint32x4_t =
                vbslq_u32(m, t, f);
            add64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t = vaddq_u64(a, b);
            sub64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t = vsubq_u64(a, b);
            and64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t = vandq_u64(a, b);
            or64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t = vorrq_u64(a, b);
            shr64(v: uint64x2_t, k: u32) -> uint64x2_t = vshlq_u64(v, vdupq_n_s64(-i64::from(k)));
            shl64(v: uint64x2_t, k: u32) -> uint64x2_t = vshlq_u64(v, vdupq_n_s64(i64::from(k)));
            lt_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t = vcltq_u64(a, b);
            select64(m: uint64x2_t, t: uint64x2_t, f: uint64x2_t) -> uint64x2_t =
                vbslq_u64(m, t, f);
            // v*c = (lo32(v)·c) + ((hi32(v)·c) << 32): two vmull widening muls.
            mul_small(v: uint64x2_t, c: u64) -> uint64x2_t = {
                let c32 = vdup_n_u32(c as u32);
                let lo = vmull_u32(vmovn_u64(v), c32);
                let hi = vmull_u32(vmovn_u64(vshrq_n_u64::<32>(v)), c32);
                vaddq_u64(lo, vshlq_n_u64::<32>(hi))
            };
            // Cold on NEON (only the vectorized fp64 reduce uses it).
            mul_small_wide(v: uint64x2_t, c: u64) -> [uint64x2_t; 2] = {
                let p =
                    transmute::<uint64x2_t, [u64; 2]>(v).map(|x| u128::from(x) * u128::from(c));
                [
                    Self::v64_from_fn(|i| p[i] as u64),
                    Self::v64_from_fn(|i| (p[i] >> 64) as u64),
                ]
            };
            // Cold on NEON: packed fp64 multiplies go lane-by-lane instead.
            mul64_wide(a: uint64x2_t, b: uint64x2_t) -> [uint64x2_t; 2] = {
                let x = transmute::<uint64x2_t, [u64; 2]>(a);
                let y = transmute::<uint64x2_t, [u64; 2]>(b);
                let p: [u128; 2] = std::array::from_fn(|i| u128::from(x[i]) * u128::from(y[i]));
                [
                    Self::v64_from_fn(|i| p[i] as u64),
                    Self::v64_from_fn(|i| (p[i] >> 64) as u64),
                ]
            };
            widen_mul(a: uint32x4_t, b: uint32x4_t) -> [uint64x2_t; 2] =
                [vmull_u32(vget_low_u32(a), vget_low_u32(b)), vmull_high_u32(a, b)];
            narrow_pack(x: [uint64x2_t; 2]) -> uint32x4_t =
                vcombine_u32(vmovn_u64(x[0]), vmovn_u64(x[1]));
        }

        /// Packed multiply for 31-bit pseudo-Mersenne primes `p = 2^31 − c`,
        /// reducing entirely in 32-bit lanes via two `vqdmulhq_s32`
        /// high-multiplies (no 64-bit widening).
        ///
        /// # Correctness (exact, no estimation)
        ///
        /// Precondition: lanes `a, b ∈ [0, p)`; `c(c+1) < p = 2^31 − c` is
        /// equivalent to `c(c+2) < 2^31` and gives `c² < p`. Write
        /// `z = a·b < 2^62`.
        ///
        /// 1. `h = sqdmulh(a,b) = ⌊2z/2^32⌋ = ⌊z/2^31⌋` (exact, `2z < 2^63`
        ///    so no saturation), `z_lo31 = z mod 2^31`, so `z = h·2^31 + z_lo31`.
        /// 2. `2^31 ≡ c (mod p)` ⇒ `z ≡ t = c·h + z_lo31`.
        /// 3. `hh = ⌊c·h/2^31⌋ < c` and `ch_lo31 = c·h mod 2^31`, so
        ///    `s = ch_lo31 + z_lo31 < 2^32` (no u32 overflow); with
        ///    `hp = hh + (s ≫ 31) ≤ c` and `lo31p = s mod 2^31`,
        ///    `t = hp·2^31 + lo31p`.
        /// 4. `t ≡ t' = c·hp + lo31p` with `c·hp ≤ c² < 2^31` (exact 32-bit
        ///    multiply) and `t' < c² + 2^31 < 2p` (from `c(c+2) < 2^31`), so
        ///    one min-subtract canonicalizes.
        #[inline(always)]
        fn mul_pm31(a: uint32x4_t, b: uint32x4_t, p: u32, c: u32) -> Option<uint32x4_t> {
            unsafe {
                let mask31 = vdupq_n_u32((1u32 << 31) - 1);
                let cvec = vdupq_n_u32(c);
                let pvec = vdupq_n_u32(p);
                let h = vreinterpretq_u32_s32(vqdmulhq_s32(
                    vreinterpretq_s32_u32(a),
                    vreinterpretq_s32_u32(b),
                ));
                let z_lo31 = vandq_u32(vmulq_u32(a, b), mask31);
                let hh = vreinterpretq_u32_s32(vqdmulhq_s32(
                    vreinterpretq_s32_u32(h),
                    vreinterpretq_s32_u32(cvec),
                ));
                let ch_lo31 = vandq_u32(vmulq_u32(h, cvec), mask31);
                let s = vaddq_u32(ch_lo31, z_lo31);
                let hp = vaddq_u32(hh, vshrq_n_u32::<31>(s));
                let lo31p = vandq_u32(s, mask31);
                let tprime = vaddq_u32(vmulq_u32(hp, cvec), lo31p);
                Some(vminq_u32(tprime, vsubq_u32(tprime, pvec)))
            }
        }

        /// Carry-preserving two-fold reducer for `p = 2^63 - c`.
        #[inline(always)]
        fn mul_pm63(lhs: u64, rhs: u64, _p: u64, c: u64) -> Option<u64> {
            let result: u64;
            let reduction_bias = (1u64 << 63) + c;

            // SAFETY: the assembly has no memory or stack effects and all
            // temporaries are declared outputs. The caller dispatches here
            // only for a 63-bit modulus satisfying the field invariant.
            unsafe {
                core::arch::asm!(
                    "umulh {high}, {lhs}, {rhs}",
                    "mul {result}, {lhs}, {rhs}",
                    "extr {quotient}, {high}, {result}, #63",
                    "and {result}, {result}, #0x7fffffffffffffff",
                    "umulh {product_hi}, {quotient}, {c}",
                    "mul {product_lo}, {quotient}, {c}",
                    "adds {result}, {result}, {product_lo}",
                    "adc {high}, {product_hi}, xzr",
                    "extr {quotient}, {high}, {result}, #63",
                    "and {result}, {result}, #0x7fffffffffffffff",
                    "madd {result}, {quotient}, {c}, {result}",
                    "adds {high}, {result}, {reduction_bias}",
                    "csel {result}, {high}, {result}, hs",
                    lhs = in(reg) lhs,
                    rhs = in(reg) rhs,
                    c = in(reg) c,
                    reduction_bias = in(reg) reduction_bias,
                    result = out(reg) result,
                    high = out(reg) _,
                    quotient = out(reg) _,
                    product_lo = out(reg) _,
                    product_hi = out(reg) _,
                    options(pure, nomem, nostack),
                );
            }
            Some(result)
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub use avx2::Avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "register-only AVX2 intrinsics over plain integer lanes; the per-fn contracts are documented on the SimdWord trait"
)]
mod avx2 {
    use super::SimdWord;
    use core::arch::x86_64::*;
    use core::mem::transmute;

    /// x86-64 AVX2: 256-bit vectors (8 × u32, 4 × u64). No native unsigned
    /// compares (sign-bit-XOR trick) and no 64-bit widening multiply
    /// (assembled from `_mm256_mul_epu32` 32×32→64 partial products).
    pub enum Avx2 {}

    /// Duplicate the high 32 bits of each 64-bit lane into the low 32 bits.
    /// The float `movehdup` runs on port 5, off the multiply ports.
    #[inline(always)]
    unsafe fn movehdup_epi32(x: __m256i) -> __m256i {
        unsafe { _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x))) }
    }

    #[inline(always)]
    unsafe fn moveldup_epi32(x: __m256i) -> __m256i {
        unsafe { _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(x))) }
    }

    impl SimdWord for Avx2 {
        type V32 = __m256i;
        type V64 = __m256i;
        const W32: usize = 8;
        const W64: usize = 4;
        const FP64_MUL_BY_LANES: bool = false;

        fwd! {
            v32_from_fn(f: impl FnMut(usize) -> u32) -> __m256i =
                transmute::<[u32; 8], __m256i>(std::array::from_fn(f));
            v32_lane(v: __m256i, lane: usize) -> u32 = transmute::<__m256i, [u32; 8]>(v)[lane];
            v64_from_fn(f: impl FnMut(usize) -> u64) -> __m256i =
                transmute::<[u64; 4], __m256i>(std::array::from_fn(f));
            v64_lane(v: __m256i, lane: usize) -> u64 = transmute::<__m256i, [u64; 4]>(v)[lane];
            splat32(x: u32) -> __m256i = _mm256_set1_epi32(x as i32);
            splat64(x: u64) -> __m256i = _mm256_set1_epi64x(x as i64);
            add32(a: __m256i, b: __m256i) -> __m256i = _mm256_add_epi32(a, b);
            sub32(a: __m256i, b: __m256i) -> __m256i = _mm256_sub_epi32(a, b);
            min_u32(a: __m256i, b: __m256i) -> __m256i = _mm256_min_epu32(a, b);
            lt_u32(a: __m256i, b: __m256i) -> __m256i = {
                let s = _mm256_set1_epi32(i32::MIN);
                _mm256_cmpgt_epi32(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s))
            };
            select32(m: __m256i, t: __m256i, f: __m256i) -> __m256i = _mm256_blendv_epi8(f, t, m);
            add64(a: __m256i, b: __m256i) -> __m256i = _mm256_add_epi64(a, b);
            sub64(a: __m256i, b: __m256i) -> __m256i = _mm256_sub_epi64(a, b);
            and64(a: __m256i, b: __m256i) -> __m256i = _mm256_and_si256(a, b);
            or64(a: __m256i, b: __m256i) -> __m256i = _mm256_or_si256(a, b);
            shr64(v: __m256i, k: u32) -> __m256i =
                _mm256_srl_epi64(v, _mm_set_epi64x(0, i64::from(k)));
            shl64(v: __m256i, k: u32) -> __m256i =
                _mm256_sll_epi64(v, _mm_set_epi64x(0, i64::from(k)));
            lt_u64(a: __m256i, b: __m256i) -> __m256i = {
                let s = _mm256_set1_epi64x(i64::MIN);
                _mm256_cmpgt_epi64(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s))
            };
            select64(m: __m256i, t: __m256i, f: __m256i) -> __m256i = _mm256_blendv_epi8(f, t, m);
            // No 64-bit multiply: v*c = (v_lo·c) + ((v_hi·c) << 32) mod 2^64.
            mul_small(v: __m256i, c: u64) -> __m256i = {
                let cv = _mm256_set1_epi64x(c as i64);
                let lo = _mm256_mul_epu32(v, cv);
                let hi = _mm256_mul_epu32(_mm256_srli_epi64::<32>(v), cv);
                _mm256_add_epi64(lo, _mm256_slli_epi64::<32>(hi))
            };
            mul_small_wide(v: __m256i, c: u64) -> [__m256i; 2] = {
                let cv = _mm256_set1_epi64x(c as i64);
                let lo_p = _mm256_mul_epu32(v, cv);
                let hi_p = _mm256_mul_epu32(_mm256_srli_epi64::<32>(v), cv);
                let lo = _mm256_add_epi64(lo_p, _mm256_slli_epi64::<32>(hi_p));
                // Subtracting an all-ones carry mask adds one.
                let carry = Self::lt_u64(lo, lo_p);
                let hi = _mm256_sub_epi64(_mm256_srli_epi64::<32>(hi_p), carry);
                [lo, hi]
            };
            // Schoolbook 64×64→128 from 32×32→64 partial products
            // (plonky2/plonky3 Goldilocks technique).
            mul64_wide(x: __m256i, y: __m256i) -> [__m256i; 2] = {
                let x_hi = movehdup_epi32(x);
                let y_hi = movehdup_epi32(y);
                let mul_ll = _mm256_mul_epu32(x, y);
                let mul_lh = _mm256_mul_epu32(x, y_hi);
                let mul_hl = _mm256_mul_epu32(x_hi, y);
                let mul_hh = _mm256_mul_epu32(x_hi, y_hi);
                let t0 = _mm256_add_epi64(mul_hl, _mm256_srli_epi64::<32>(mul_ll));
                let t0_lo = _mm256_and_si256(t0, _mm256_set1_epi64x(0xFFFF_FFFF_i64));
                let t1 = _mm256_add_epi64(mul_lh, t0_lo);
                let t2 = _mm256_add_epi64(mul_hh, _mm256_srli_epi64::<32>(t0));
                let hi = _mm256_add_epi64(t2, _mm256_srli_epi64::<32>(t1));
                let lo = _mm256_blend_epi32::<0b1010_1010>(mul_ll, moveldup_epi32(t1));
                [lo, hi]
            };
            widen_mul(a: __m256i, b: __m256i) -> [__m256i; 2] =
                [_mm256_mul_epu32(a, b), _mm256_mul_epu32(movehdup_epi32(a), movehdup_epi32(b))];
            narrow_pack(x: [__m256i; 2]) -> __m256i =
                _mm256_blend_epi32::<0b1010_1010>(x[0], _mm256_slli_epi64::<32>(x[1]));
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub use avx512::Avx512;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "register-only AVX-512 intrinsics over plain integer lanes; the per-fn contracts are documented on the SimdWord trait"
)]
mod avx512 {
    use super::SimdWord;
    use core::arch::x86_64::*;
    use core::mem::transmute;

    /// x86-64 AVX-512 (F + DQ): 512-bit vectors (16 × u32, 8 × u64). Native
    /// unsigned compares produce mask registers, converted to all-ones lane
    /// masks with `movm`; selects use one `vpternlogq` (truth table `0xCA`
    /// computes `(m & t) | (!m & f)`).
    pub enum Avx512 {}

    #[inline(always)]
    unsafe fn movehdup_epi32_512(x: __m512i) -> __m512i {
        unsafe { _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x))) }
    }

    #[inline(always)]
    unsafe fn moveldup_epi32_512(x: __m512i) -> __m512i {
        unsafe { _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(x))) }
    }

    impl SimdWord for Avx512 {
        type V32 = __m512i;
        type V64 = __m512i;
        const W32: usize = 16;
        const W64: usize = 8;
        const FP64_MUL_BY_LANES: bool = false;

        fwd! {
            v32_from_fn(f: impl FnMut(usize) -> u32) -> __m512i =
                transmute::<[u32; 16], __m512i>(std::array::from_fn(f));
            v32_lane(v: __m512i, lane: usize) -> u32 = transmute::<__m512i, [u32; 16]>(v)[lane];
            v64_from_fn(f: impl FnMut(usize) -> u64) -> __m512i =
                transmute::<[u64; 8], __m512i>(std::array::from_fn(f));
            v64_lane(v: __m512i, lane: usize) -> u64 = transmute::<__m512i, [u64; 8]>(v)[lane];
            splat32(x: u32) -> __m512i = _mm512_set1_epi32(x as i32);
            splat64(x: u64) -> __m512i = _mm512_set1_epi64(x as i64);
            add32(a: __m512i, b: __m512i) -> __m512i = _mm512_add_epi32(a, b);
            sub32(a: __m512i, b: __m512i) -> __m512i = _mm512_sub_epi32(a, b);
            min_u32(a: __m512i, b: __m512i) -> __m512i = _mm512_min_epu32(a, b);
            lt_u32(a: __m512i, b: __m512i) -> __m512i =
                _mm512_movm_epi32(_mm512_cmplt_epu32_mask(a, b));
            select32(m: __m512i, t: __m512i, f: __m512i) -> __m512i =
                _mm512_ternarylogic_epi64::<0xCA>(m, t, f);
            add64(a: __m512i, b: __m512i) -> __m512i = _mm512_add_epi64(a, b);
            sub64(a: __m512i, b: __m512i) -> __m512i = _mm512_sub_epi64(a, b);
            and64(a: __m512i, b: __m512i) -> __m512i = _mm512_and_si512(a, b);
            or64(a: __m512i, b: __m512i) -> __m512i = _mm512_or_si512(a, b);
            shr64(v: __m512i, k: u32) -> __m512i =
                _mm512_srl_epi64(v, _mm_set_epi64x(0, i64::from(k)));
            shl64(v: __m512i, k: u32) -> __m512i =
                _mm512_sll_epi64(v, _mm_set_epi64x(0, i64::from(k)));
            lt_u64(a: __m512i, b: __m512i) -> __m512i =
                _mm512_movm_epi64(_mm512_cmplt_epu64_mask(a, b));
            select64(m: __m512i, t: __m512i, f: __m512i) -> __m512i =
                _mm512_ternarylogic_epi64::<0xCA>(m, t, f);
            // AVX-512DQ has a true 64-bit low multiply.
            mul_small(v: __m512i, c: u64) -> __m512i =
                _mm512_mullo_epi64(v, _mm512_set1_epi64(c as i64));
            mul_small_wide(v: __m512i, c: u64) -> [__m512i; 2] = {
                let cv = _mm512_set1_epi64(c as i64);
                let lo_p = _mm512_mul_epu32(v, cv);
                let hi_p = _mm512_mul_epu32(_mm512_srli_epi64::<32>(v), cv);
                let lo = _mm512_add_epi64(lo_p, _mm512_slli_epi64::<32>(hi_p));
                let carry = _mm512_cmplt_epu64_mask(lo, lo_p);
                let hi_base = _mm512_srli_epi64::<32>(hi_p);
                let hi = _mm512_mask_add_epi64(hi_base, carry, hi_base, _mm512_set1_epi64(1));
                [lo, hi]
            };
            // Schoolbook 64×64→128 from 32×32→64 partial products
            // (plonky3 Goldilocks AVX-512 technique).
            mul64_wide(x: __m512i, y: __m512i) -> [__m512i; 2] = {
                let x_hi = movehdup_epi32_512(x);
                let y_hi = movehdup_epi32_512(y);
                let mul_ll = _mm512_mul_epu32(x, y);
                let mul_lh = _mm512_mul_epu32(x, y_hi);
                let mul_hl = _mm512_mul_epu32(x_hi, y);
                let mul_hh = _mm512_mul_epu32(x_hi, y_hi);
                let t0 = _mm512_add_epi64(mul_hl, _mm512_srli_epi64::<32>(mul_ll));
                let t0_lo = _mm512_and_si512(t0, _mm512_set1_epi64(0xFFFF_FFFF_i64));
                let t1 = _mm512_add_epi64(mul_lh, t0_lo);
                let t2 = _mm512_add_epi64(mul_hh, _mm512_srli_epi64::<32>(t0));
                let hi = _mm512_add_epi64(t2, _mm512_srli_epi64::<32>(t1));
                let lo =
                    _mm512_mask_blend_epi32(0b0101_0101_0101_0101, moveldup_epi32_512(t1), mul_ll);
                [lo, hi]
            };
            widen_mul(a: __m512i, b: __m512i) -> [__m512i; 2] = [
                _mm512_mul_epu32(a, b),
                _mm512_mul_epu32(movehdup_epi32_512(a), movehdup_epi32_512(b)),
            ];
            narrow_pack(x: [__m512i; 2]) -> __m512i =
                _mm512_mask_blend_epi32(0b1010_1010_1010_1010, x[0], _mm512_slli_epi64::<32>(x[1]));
        }
    }
}
