use std::array::from_fn;
use std::marker::PhantomData;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx;
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic};
use crate::ntt::montgomery::{inverse_i32, reduce_i32};
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::NttTwiddles;

use super::CrtNttParamSet;

/// Number of balanced-digit slots covered by [`DigitMontLut`].
///
/// Balanced i8 decomposition uses `log_basis` in `1..=8`, so every digit lands
/// in `[-128, 127]`. One fixed 256-entry table therefore covers all bases,
/// which lets the lookup drop the const-generic `L` and the per-`log_basis`
/// monomorphization it forced.
const DIGIT_LUT_LEN: usize = 256;
const DIGIT_LUT_OFFSET: i16 = (DIGIT_LUT_LEN / 2) as i16;

/// Precomputed Montgomery forms for balanced i8 digit values in `[-128, 127]`.
///
/// Storing the Montgomery representation eliminates one `from_canonical`
/// Montgomery multiply per coefficient in the hot digit path. The table is
/// built once per mat-vec and is independent of the decomposition `log_basis`.
#[derive(Debug, Clone)]
pub struct DigitMontLut<W: PrimeWidth, const K: usize> {
    vals: [[MontCoeff<W>; DIGIT_LUT_LEN]; K],
    len: usize,
    offset: i16,
}

/// Precomputed Montgomery forms for centered integer coefficients in
/// `[-max_abs, max_abs]`.
#[derive(Debug, Clone)]
pub struct CenteredMontLut<W: PrimeWidth, const K: usize> {
    vals: [Vec<MontCoeff<W>>; K],
    offset: i32,
}

/// Division-free map from a centered integer to its Montgomery form modulo one
/// CRT prime, for either limb width.
///
/// Every product is reduced by one signed 32-bit Montgomery step
/// `redc(t) = t * 2^-32 mod p`, which maps `|t| < 2^31 p` into `(-p, p)`.
/// With the centered constants `scale[j] = 2^(32 j) * R * 2^32 mod p`, where
/// `R = 2^W::R_LOG`, `redc(x * scale[j])` is the Montgomery form of
/// `x * 2^(32 j)`.
#[derive(Debug, Clone, Copy)]
pub(super) struct CenteredMontReducer<W: PrimeWidth> {
    p: i64,
    /// `p^-1 mod 2^32`.
    pinv: i32,
    scale: [i64; 4],
    _width: PhantomData<W>,
}

impl<W: PrimeWidth> CenteredMontReducer<W> {
    #[inline(always)]
    pub(super) fn new(prime: NttPrime<W>) -> Self {
        let p = prime.p.to_i64();
        debug_assert!(p > 1 && p % 2 == 1 && p < 1 << (W::R_LOG - 1));
        let two_32 = (1i64 << 32) % p;
        let mut power = ((1u128 << (W::R_LOG + 32)) % p as u128) as i64;
        let scale = from_fn(|_| {
            let centered = if power > p / 2 { power - p } else { power };
            power = power * two_32 % p;
            centered
        });
        Self {
            p,
            pinv: inverse_i32(p as i32),
            scale,
            _width: PhantomData,
        }
    }

    #[inline(always)]
    fn redc(self, t: i64) -> i64 {
        i64::from(reduce_i32(t, self.p as i32, self.pinv))
    }

    /// Montgomery form of `value`, in `(-p, p)`.
    #[inline(always)]
    pub(super) fn reduce_i32(self, value: i32) -> MontCoeff<W> {
        // |value * scale[0]| <= 2^31 (p - 1) / 2.
        MontCoeff::from_raw(W::from_i64(self.redc(i64::from(value) * self.scale[0])))
    }

    /// Montgomery form, in `(-p, p)`, of the value with the given
    /// [`balanced_limbs`].
    #[inline(always)]
    pub(super) fn reduce_limbs(self, limbs: [i64; 4]) -> MontCoeff<W> {
        // Each pair sum is at most 2 * 2^31 * (p - 1) / 2 in magnitude, so
        // both reductions land in (-p, p) and their sum in (-2p, 2p).
        let low = self.redc(limbs[0] * self.scale[0] + limbs[1] * self.scale[1]);
        let high = self.redc(limbs[2] * self.scale[2] + limbs[3] * self.scale[3]);
        let mut sum = low + high;
        if sum >= self.p {
            sum -= self.p;
        } else if sum <= -self.p {
            sum += self.p;
        }
        MontCoeff::from_raw(W::from_i64(sum))
    }
}

/// Split `value` into balanced 32-bit limbs with
/// `value = sum_j limbs[j] * 2^(32 j)`.
///
/// The low three limbs lie in `[-2^31, 2^31)` and the top limb in
/// `[-2^31, 2^31]` for every `|value| <= 2^127`, which covers any centered
/// residue of a modulus up to `2^128`.
#[inline(always)]
pub(super) fn balanced_limbs(value: i128) -> [i64; 4] {
    let mut rest = value;
    let mut limbs = [0i64; 4];
    for limb in &mut limbs[..3] {
        let low = rest as i32;
        *limb = i64::from(low);
        // Equals (rest - low) >> 32 without overflowing near 2^127.
        rest = (rest >> 32) + i128::from(low < 0);
    }
    limbs[3] = rest as i64;
    limbs
}

impl<W: PrimeWidth, const K: usize> DigitMontLut<W, K> {
    #[inline(always)]
    fn debug_assert_active_digits<const D: usize>(&self, digits: &[i8; D]) {
        debug_assert!(
            digits.iter().all(|&digit| {
                let idx = i16::from(digit) + self.offset;
                idx >= 0 && (idx as usize) < self.len
            }),
            "digit LUT conversion outside active balanced range"
        );
    }

    /// Build a lookup table for the active balanced range `[-bound, bound)`.
    ///
    /// This keeps the fixed non-monomorphized LUT type while avoiding needless
    /// Montgomery conversions for common small bases (`log_basis` 2, 3, or 4).
    pub fn new_with_digit_bound<const D: usize>(
        params: &CrtNttParamSet<W, K, D>,
        digit_abs_bound: u64,
    ) -> Self {
        debug_assert!(digit_abs_bound.is_power_of_two());
        debug_assert!((1..=DIGIT_LUT_OFFSET as u64).contains(&digit_abs_bound));
        let digit_abs_bound = digit_abs_bound
            .max(1)
            .next_power_of_two()
            .min(DIGIT_LUT_OFFSET as u64);
        let len = (digit_abs_bound as usize) * 2;
        let offset = digit_abs_bound as i16;
        let mut vals = [[MontCoeff::from_raw(W::default()); DIGIT_LUT_LEN]; K];
        for (k, limb) in vals.iter_mut().enumerate() {
            let prime = params.primes[k];
            for (idx, dst) in limb.iter_mut().enumerate().take(len) {
                let v = idx as i64 - i64::from(offset);
                *dst = prime.from_canonical(W::from_i64(v));
            }
        }
        Self { vals, len, offset }
    }

    /// Look up the Montgomery form of a balanced digit for CRT prime `k`.
    ///
    /// Contract: `digit` is in this LUT's active balanced range and `k < K`.
    /// The i8-NTT kernel boundary upholds this with validated `log_basis` and
    /// per-block digit range checks. Because the active table length is a power
    /// of two, masking keeps the lookup in-bounds and branch-free without a
    /// per-coefficient bounds check. The debug assertion surfaces any contract
    /// violation in debug builds.
    #[inline(always)]
    pub fn get(&self, k: usize, digit: i8) -> MontCoeff<W> {
        let idx = (i16::from(digit) + self.offset) as usize;
        debug_assert!(
            idx < self.len,
            "digit LUT lookup outside active balanced range"
        );
        self.vals[k][idx & (self.len - 1)]
    }

    /// Fill one CRT limb with Montgomery representations of signed digits.
    #[inline]
    pub(super) fn fill_limb<const D: usize>(
        &self,
        k: usize,
        digits: &[i8; D],
        _params: &CrtNttParamSet<W, K, D>,
        dst: &mut [MontCoeff<W>; D],
    ) {
        self.debug_assert_active_digits(digits);
        #[cfg(target_arch = "aarch64")]
        if _params.kernel_plan().uses_neon() && size_of::<W>() == size_of::<i32>() {
            let prime = _params.primes[k];
            // SAFETY: the width check proves the transparent i32
            // representation, and both arrays contain D elements.
            unsafe {
                neon::centered_i8_to_mont_i32(
                    dst.as_mut_ptr().cast::<i32>(),
                    digits.as_ptr(),
                    D,
                    prime.p.to_i64() as i32,
                    prime.pinv.to_i64() as i32,
                    prime.montsq.to_i64() as i32,
                );
            }
            return;
        }

        for (dst, &digit) in dst.iter_mut().zip(digits) {
            *dst = self.get(k, digit);
        }
    }

    /// Convert one signed-digit limb and apply its forward negacyclic NTT, or
    /// with `CYCLIC` its forward cyclic NTT.
    #[inline]
    pub(super) fn fill_ntt_limb<const CYCLIC: bool, const D: usize>(
        &self,
        k: usize,
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        dst: &mut [MontCoeff<W>; D],
    ) {
        self.debug_assert_active_digits(digits);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::ntt::butterfly::use_x86_transform_ntt::<D>(params.kernel_plan()) {
            let prime = params.primes[k];
            let tw = &params.twiddles[k];
            let use_avx512 = params.kernel_plan().uses_avx512_transform();
            // SAFETY: PrimeWidth is sealed to i16 and i32, so the width check
            // identifies W. MontCoeff is transparent, while NttPrime and
            // NttTwiddles have stable C layouts. Both arrays contain D >= 64
            // elements, do not overlap, and the prepared plan proves AVX2.
            unsafe {
                if size_of::<W>() == size_of::<i16>() {
                    let dst = &mut *(dst as *mut _ as *mut [MontCoeff<i16>; D]);
                    let prime = *(&prime as *const _ as *const NttPrime<i16>);
                    let tw = &*(tw as *const _ as *const NttTwiddles<i16, D>);
                    if CYCLIC {
                        avx::forward_ntt_cyclic_i8_i16(dst, digits, prime, tw);
                    } else {
                        avx::forward_ntt_i8_i16(dst, digits, prime, tw);
                    }
                } else {
                    let dst = &mut *(dst as *mut _ as *mut [MontCoeff<i32>; D]);
                    let prime = *(&prime as *const _ as *const NttPrime<i32>);
                    let tw = &*(tw as *const _ as *const NttTwiddles<i32, D>);
                    if CYCLIC {
                        avx::forward_ntt_cyclic_i8_i32(dst, digits, prime, tw, use_avx512);
                    } else {
                        avx::forward_ntt_i8_i32(dst, digits, prime, tw, use_avx512);
                    }
                }
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if params.kernel_plan().uses_neon() && size_of::<W>() == size_of::<i32>() {
            let prime = params.primes[k];
            let tw = &params.twiddles[k];
            // SAFETY: PrimeWidth is sealed to i16 and i32, so the width check
            // identifies W as i32. MontCoeff is transparent, while NttPrime
            // and NttTwiddles have stable C layouts. Both input arrays have D
            // elements and do not overlap.
            unsafe {
                let dst = &mut *(dst as *mut _ as *mut [MontCoeff<i32>; D]);
                let prime = *(&prime as *const _ as *const NttPrime<i32>);
                let tw = &*(tw as *const _ as *const NttTwiddles<i32, D>);
                if CYCLIC {
                    neon::forward_ntt_cyclic_i8_i32(dst, digits, prime, tw);
                } else {
                    neon::forward_ntt_i8_i32(dst, digits, prime, tw);
                }
            }
            return;
        }

        self.fill_limb(k, digits, params, dst);
        let (prime, tw, plan) = (params.primes[k], &params.twiddles[k], params.kernel_plan());
        if CYCLIC {
            forward_ntt_cyclic(dst, prime, tw, plan);
        } else {
            forward_ntt(dst, prime, tw, plan);
        }
    }
}

impl<W: PrimeWidth, const K: usize> CenteredMontLut<W, K> {
    /// Build a lookup table for all centered coefficients in `[-max_abs, max_abs]`.
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>, max_abs: i32) -> Self {
        let max_abs = max_abs.max(0);
        let vals = from_fn(|k| {
            let reducer = CenteredMontReducer::new(params.primes[k]);
            (-max_abs..=max_abs)
                .map(|v| reducer.reduce_i32(v))
                .collect()
        });
        Self {
            vals,
            offset: max_abs,
        }
    }

    /// Look up the Montgomery form of a centered coefficient for CRT prime `k`.
    #[inline(always)]
    pub fn get(&self, k: usize, coeff: i32) -> Option<MontCoeff<W>> {
        let idx = coeff.checked_add(self.offset)?;
        self.vals.get(k)?.get(usize::try_from(idx).ok()?).copied()
    }

    /// Look up the Montgomery form of a caller-validated centered coefficient.
    ///
    /// # Safety
    ///
    /// `k` must be less than `K`, and `coeff` must be within this LUT's
    /// covered centered range.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, k: usize, coeff: i32) -> MontCoeff<W> {
        let idx = coeff + self.offset;
        debug_assert!(idx >= 0);
        debug_assert!((idx as usize) < self.vals[k].len());
        unsafe { *self.vals.get_unchecked(k).get_unchecked(idx as usize) }
    }
}
