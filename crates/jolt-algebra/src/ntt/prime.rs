//! NTT prime arithmetic kernels generic over coefficient width.
//!
//! Per-prime scalar operations:
//! - Montgomery multiplication ([`NttPrime::mul`])
//! - Branchless conditional add/sub and range reduction
//!
//! Coefficients in Montgomery domain are wrapped in [`MontCoeff`] to prevent
//! accidental mixing with canonical values.
//!
//! The [`PrimeWidth`] trait abstracts over `i16` (R = 2^16, for primes < 2^14)
//! and `i32` (R = 2^32, for primes < 2^30). All NTT types are generic over
//! `W: PrimeWidth`; monomorphization produces optimal code for each width.

use std::fmt;

/// Largest raw i32 Montgomery dot batch whose signed reduction is safe for
/// every supported prime below `2^30`.
pub const I32_LAZY_DOT_BATCH: usize = 6;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
}

/// Integer width abstraction for NTT prime arithmetic.
///
/// Sealed with exactly two implementations: `i16` and `i32`.
pub trait PrimeWidth:
    sealed::Sealed + Copy + Clone + Eq + Default + fmt::Debug + Send + Sync + 'static
{
    /// Double-width type for intermediate Montgomery products.
    type Wide: Copy + Clone;

    /// Tables the NEON transforms read beside the Montgomery tables: the
    /// Barrett-form tables for `i32`, and nothing for `i16`, whose kernels
    /// never read them.
    #[cfg(target_arch = "aarch64")]
    #[doc(hidden)]
    type NeonTables<const D: usize>: Clone + fmt::Debug + PartialEq + Eq + Send + Sync;

    /// log2(R) for Montgomery reduction: 16 for `i16`, 32 for `i32`.
    const R_LOG: u32;

    /// Widening multiply: `a * b` as `Wide`.
    fn wide_mul(a: Self, b: Self) -> Self::Wide;

    /// Truncate wide value to narrow (low half, i.e. mod R).
    fn truncate(w: Self::Wide) -> Self;

    /// Arithmetic right shift of wide value by `R_LOG` bits.
    fn wide_shift(w: Self::Wide) -> Self;

    /// Wide subtraction (wrapping).
    fn wide_sub(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    /// Wrapping addition.
    fn wrapping_add(self, rhs: Self) -> Self;
    /// Wrapping subtraction.
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// Wrapping multiplication.
    fn wrapping_mul(self, rhs: Self) -> Self;
    /// Wrapping negation.
    fn wrapping_neg(self) -> Self;

    /// Arithmetic right shift by `BITS - 1`: all-1s if negative, all-0s otherwise.
    fn sign_mask(self) -> Self;

    /// Bitwise AND.
    fn bitand(self, rhs: Self) -> Self;

    /// Convert from `i64` (truncating).
    fn from_i64(v: i64) -> Self;
    /// Convert to `i64` (sign-extending).
    fn to_i64(self) -> i64;

    /// Derive [`Self::NeonTables`] from the Montgomery-form tables.
    #[cfg(target_arch = "aarch64")]
    #[doc(hidden)]
    fn neon_tables<const D: usize>(
        prime: NttPrime<Self>,
        fwd_twiddles: &[MontCoeff<Self>; D],
        inv_twiddles: &[MontCoeff<Self>; D],
        psi_pows: &[MontCoeff<Self>; D],
        d_inv_psi_inv: &[MontCoeff<Self>; D],
        d_inv: MontCoeff<Self>,
    ) -> Self::NeonTables<D>;
}

impl PrimeWidth for i16 {
    type Wide = i32;
    #[cfg(target_arch = "aarch64")]
    type NeonTables<const D: usize> = ();
    const R_LOG: u32 = 16;

    #[inline]
    fn wide_mul(a: Self, b: Self) -> i32 {
        (a as i32) * (b as i32)
    }
    #[inline]
    fn truncate(w: i32) -> Self {
        w as i16
    }
    #[inline]
    fn wide_shift(w: i32) -> Self {
        (w >> 16) as i16
    }
    #[inline]
    fn wide_sub(a: i32, b: i32) -> i32 {
        a.wrapping_sub(b)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        i16::wrapping_add(self, rhs)
    }
    #[inline]
    fn wrapping_sub(self, rhs: Self) -> Self {
        i16::wrapping_sub(self, rhs)
    }
    #[inline]
    fn wrapping_mul(self, rhs: Self) -> Self {
        i16::wrapping_mul(self, rhs)
    }
    #[inline]
    fn wrapping_neg(self) -> Self {
        i16::wrapping_neg(self)
    }
    #[inline]
    fn sign_mask(self) -> Self {
        self >> 15
    }
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i16
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[cfg(target_arch = "aarch64")]
    fn neon_tables<const D: usize>(
        _: NttPrime<Self>,
        _: &[MontCoeff<Self>; D],
        _: &[MontCoeff<Self>; D],
        _: &[MontCoeff<Self>; D],
        _: &[MontCoeff<Self>; D],
        _: MontCoeff<Self>,
    ) {
    }
}

impl PrimeWidth for i32 {
    type Wide = i64;
    #[cfg(target_arch = "aarch64")]
    type NeonTables<const D: usize> = super::neon::BarrettTwiddles<i32, D>;
    const R_LOG: u32 = 32;

    #[inline]
    fn wide_mul(a: Self, b: Self) -> i64 {
        (a as i64) * (b as i64)
    }
    #[inline]
    fn truncate(w: i64) -> Self {
        w as i32
    }
    #[inline]
    fn wide_shift(w: i64) -> Self {
        (w >> 32) as i32
    }
    #[inline]
    fn wide_sub(a: i64, b: i64) -> i64 {
        a.wrapping_sub(b)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        i32::wrapping_add(self, rhs)
    }
    #[inline]
    fn wrapping_sub(self, rhs: Self) -> Self {
        i32::wrapping_sub(self, rhs)
    }
    #[inline]
    fn wrapping_mul(self, rhs: Self) -> Self {
        i32::wrapping_mul(self, rhs)
    }
    #[inline]
    fn wrapping_neg(self) -> Self {
        i32::wrapping_neg(self)
    }
    #[inline]
    fn sign_mask(self) -> Self {
        self >> 31
    }
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i32
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[cfg(target_arch = "aarch64")]
    fn neon_tables<const D: usize>(
        prime: NttPrime<Self>,
        fwd_twiddles: &[MontCoeff<Self>; D],
        inv_twiddles: &[MontCoeff<Self>; D],
        psi_pows: &[MontCoeff<Self>; D],
        d_inv_psi_inv: &[MontCoeff<Self>; D],
        d_inv: MontCoeff<Self>,
    ) -> Self::NeonTables<D> {
        super::neon::BarrettTwiddles::compute(
            prime,
            fwd_twiddles,
            inv_twiddles,
            psi_pows,
            d_inv_psi_inv,
            d_inv,
        )
    }
}

/// A coefficient in Montgomery domain for an NTT prime.
///
/// Wraps a `W` representing `a * R mod p` (where `R = 2^{W::R_LOG}`).
/// Use [`NttPrime::from_canonical`] to enter and [`NttPrime::to_canonical`]
/// to leave Montgomery domain.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct MontCoeff<W: PrimeWidth>(W);

impl<W: PrimeWidth> MontCoeff<W> {
    /// Wrap a raw Montgomery-domain value.
    #[inline]
    pub fn from_raw(val: W) -> Self {
        Self(val)
    }

    /// Extract the raw value (still in Montgomery domain).
    #[inline]
    pub fn raw(self) -> W {
        self.0
    }
}

impl<W: PrimeWidth> fmt::Debug for MontCoeff<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mont({:?})", self.0)
    }
}

/// Per-prime constants for NTT arithmetic.
///
/// Generic over `W: PrimeWidth` — use `i16` for primes below 2^14 (R = 2^16),
/// or `i32` for primes below 2^30 (R = 2^32).
///
/// The C representation is part of the SIMD dispatch contract. `PrimeWidth`
/// is sealed to `i16` and `i32`, and architecture-specific kernels reinterpret
/// a value after checking which of those two widths is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct NttPrime<W: PrimeWidth> {
    /// Prime modulus.
    pub p: W,
    /// `p^{-1} mod R` (centered signed). Used in Montgomery reduction.
    pub pinv: W,
    /// `R mod p` (centered signed). Montgomery form of 1.
    pub mont: W,
    /// `R^2 mod p` (centered signed). Used for canonical → Montgomery conversion.
    pub montsq: W,
}

impl<W: PrimeWidth> NttPrime<W> {
    /// Check that `p` is an odd prime below `R/4` and derive its constants.
    ///
    /// The kernels' lazy ranges assume `p < R/4`: `2^14` for `i16` and `2^30`
    /// for `i32`. The Fermat inverses in
    /// [`NttTwiddles::compute`](super::butterfly::NttTwiddles::compute) assume
    /// `p` is prime.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not an odd prime below `R/4`.
    pub fn new(p: W) -> Self {
        let p_i64 = p.to_i64();
        assert!(
            p_i64 > 2 && p_i64 < 1 << (W::R_LOG - 2) && is_prime(p_i64),
            "NTT modulus {p_i64} must be an odd prime below 2^{}",
            W::R_LOG - 2
        );
        Self::compute(p)
    }

    /// Derive all Montgomery constants from a raw prime value.
    ///
    /// Does not check `p`; [`Self::new`] does.
    pub fn compute(p: W) -> Self {
        let p_i64 = p.to_i64();
        debug_assert!(p_i64 > 1 && p_i64 % 2 == 1, "NTT prime must be odd and > 1");

        // pinv via Newton's method: x_{n+1} = x_n * (2 - p * x_n).
        // 5 iterations gives correctness mod 2^32 (sufficient for both i16 and i32).
        let mut pinv: i64 = 1;
        for _ in 0..5 {
            pinv = pinv.wrapping_mul(2i64.wrapping_sub(p_i64.wrapping_mul(pinv)));
        }
        let pinv = W::from_i64(pinv);

        let half = p_i64 / 2;
        let center = |x: i64| -> W { W::from_i64(if x > half { x - p_i64 } else { x }) };

        let r_mod_p = ((1i128 << W::R_LOG) % (p_i64 as i128)) as i64;
        let mont = center(r_mod_p);

        let rsq_mod_p = ((1i128 << (2 * W::R_LOG)) % (p_i64 as i128)) as i64;
        let montsq = center(rsq_mod_p);

        Self {
            p,
            pinv,
            mont,
            montsq,
        }
    }

    /// Montgomery product: `a * b * R^{-1} mod p`.
    #[inline]
    pub fn mul(self, a: MontCoeff<W>, b: MontCoeff<W>) -> MontCoeff<W> {
        MontCoeff(self.mont_mul_raw(a.0, b.0))
    }

    /// Raw Montgomery multiply on bare `W` values.
    #[inline]
    pub(crate) fn mont_mul_raw(self, a: W, b: W) -> W {
        let c = W::wide_mul(a, b);
        let t = W::truncate(c).wrapping_mul(self.pinv);
        let tp = W::wide_mul(t, self.p);
        W::wide_shift(W::wide_sub(c, tp))
    }

    /// Conditionally subtract `p` if `a >= p` (branchless).
    ///
    /// Input may be in `(-2p, 2p)` during butterflies. Both i16 and i32 paths
    /// widen to avoid signed overflow: i16→i32, i32→i64.
    #[inline]
    pub fn csubp(self, a: MontCoeff<W>) -> MontCoeff<W> {
        if W::R_LOG == 16 {
            let ai = a.0.to_i64() as i32;
            let pi = self.p.to_i64() as i32;
            let diff = ai - pi;
            let mask = diff >> 31;
            MontCoeff(W::from_i64((diff + (mask & pi)) as i64))
        } else {
            let ai = a.0.to_i64();
            let pi = self.p.to_i64();
            let diff = ai - pi;
            let mask = diff >> 63;
            MontCoeff(W::from_i64(diff + (mask & pi)))
        }
    }

    /// Conditionally add `p` if `a < 0` (branchless).
    ///
    /// Widened arithmetic mirrors `csubp` to avoid i16 overflow edge cases.
    #[inline]
    pub fn caddp(self, a: MontCoeff<W>) -> MontCoeff<W> {
        if W::R_LOG == 16 {
            let ai = a.0.to_i64() as i32;
            let pi = self.p.to_i64() as i32;
            let mask = ai >> 31;
            MontCoeff(W::from_i64((ai + (mask & pi)) as i64))
        } else {
            let ai = a.0.to_i64();
            let pi = self.p.to_i64();
            let mask = ai >> 63;
            MontCoeff(W::from_i64(ai + (mask & pi)))
        }
    }

    /// Range-reduce from `(-2p, 2p)` to `(-p, p)`.
    #[inline]
    pub fn reduce_range(self, a: MontCoeff<W>) -> MontCoeff<W> {
        self.caddp(self.csubp(a))
    }

    /// Fully normalize a Montgomery coefficient to `[0, p)`.
    #[inline]
    pub fn normalize(self, a: MontCoeff<W>) -> MontCoeff<W> {
        self.csubp(self.caddp(a))
    }

    /// Convert a canonical value into Montgomery domain: `a ↦ a * R mod p`.
    #[inline]
    pub fn from_canonical(self, a: W) -> MontCoeff<W> {
        MontCoeff(self.mont_mul_raw(a, self.montsq))
    }

    /// Convert from Montgomery domain to canonical `[0, p)`.
    #[inline]
    pub fn to_canonical(self, a: MontCoeff<W>) -> W {
        let raw = MontCoeff(self.mont_mul_raw(a.0, W::from_i64(1)));
        self.normalize(raw).0
    }

    /// Center a canonical value from approximately `(-p, p)` into `[-p/2, p/2)`.
    #[inline]
    pub fn center(self, a: W) -> W {
        let mask_neg = a.sign_mask();
        let canonical = a.wrapping_add(mask_neg.bitand(self.p));
        let half = W::from_i64(self.p.to_i64() / 2);
        let needs_sub = half.wrapping_sub(canonical).sign_mask();
        canonical.wrapping_add(needs_sub.bitand(self.p.wrapping_neg()))
    }

    /// In-place range reduction on a coefficient slice.
    #[inline]
    pub fn reduce_range_in_place(self, coeffs: &mut [MontCoeff<W>]) {
        for c in coeffs {
            *c = self.reduce_range(*c);
        }
    }
}

/// Whether `n` is prime, by deterministic Miller-Rabin.
pub(crate) fn is_prime(n: i64) -> bool {
    // Bases 2, 7 and 61 decide every n below 4_759_123_141; these seven
    // decide every n below 2^64.
    let bases: &[i64] = if n < 4_759_123_141 {
        &[2, 7, 61]
    } else {
        &[2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022]
    };
    if n < 2 {
        return false;
    }
    if let Some(&base) = bases.iter().find(|&&base| n % base == 0) {
        return n == base;
    }
    let shift = (n - 1).trailing_zeros();
    let odd = (n - 1) >> shift;
    bases.iter().all(|&base| {
        let mut x = pow_mod(base, odd, n);
        if x == 1 || x == n - 1 {
            return true;
        }
        (1..shift).any(|_| {
            x = pow_mod(x, 2, n);
            x == n - 1
        })
    })
}

/// Modular exponentiation: `base^exp mod modulus`, for `base >= 0`.
pub(crate) fn pow_mod(base: i64, exp: i64, modulus: i64) -> i64 {
    fn pow(mut base: i64, mut exp: i64, mul: impl Fn(i64, i64) -> i64) -> i64 {
        let mut result = 1;
        while exp > 0 {
            if exp & 1 == 1 {
                result = mul(result, base);
            }
            base = mul(base, base);
            exp >>= 1;
        }
        result
    }
    // Products of residues below 2^31 fit i64; wider moduli need i128.
    if modulus < 1 << 31 {
        pow(base % modulus, exp, |lhs, rhs| lhs * rhs % modulus)
    } else {
        pow(base % modulus, exp, |lhs, rhs| {
            (i128::from(lhs) * i128::from(rhs) % i128::from(modulus)) as i64
        })
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn is_prime_matches_trial_division() {
        let trial = |n: i64| n > 1 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
        // Every value below 2^16, including the base-2 strong pseudoprimes
        // 2047, 3277 and 4033, and a window just below 2^30.
        for n in (0..1 << 16).chain((1 << 30) - 8192..1 << 30) {
            assert_eq!(is_prime(n), trial(n), "n={n}");
        }
    }

    #[test]
    fn is_prime_rejects_strong_pseudoprimes_past_2_31() {
        // 3215031751 = 151 * 751 * 28351 is a strong pseudoprime to bases 2
        // and 7, and 3825123056546413051 = 149491 * 747451 * 34233211 to every
        // prime base through 23; 1125899772623531 = 33554393 * 33554467.
        for n in [
            3_215_031_751,
            3_825_123_056_546_413_051,
            1_125_899_772_623_531,
        ] {
            assert!(!is_prime(n), "n={n}");
        }
        for p in [2_147_483_647, 4_294_967_291, 1_125_899_906_826_241] {
            assert!(is_prime(p), "p={p}");
        }
    }

    #[test]
    fn new_accepts_only_odd_primes_below_a_quarter_radix() {
        assert_eq!(NttPrime::new(12289_i16), NttPrime::compute(12289_i16));
        assert_eq!(
            NttPrime::new(1073707009_i32),
            NttPrime::compute(1073707009_i32)
        );
        // 1537 = 29 * 53 and 94391809 = 7681 * 12289 pass the `2D | p - 1`
        // twiddle check for D = 256; 18433 and 2013265921 are primes above R/4.
        for p in [2_i16, 1537, 18433] {
            assert!(catch_unwind(|| NttPrime::new(p)).is_err(), "p={p}");
        }
        for p in [2_i32, 1073707008, 94391809, 2013265921] {
            assert!(catch_unwind(|| NttPrime::new(p)).is_err(), "p={p}");
        }
    }
}
