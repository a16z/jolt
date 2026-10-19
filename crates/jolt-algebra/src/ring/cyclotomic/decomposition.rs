use super::*;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(test)]
mod tests;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

/// Compute the centering threshold for balanced decomposition.
///
/// When `levels * log_basis == field_bits`, uses asymmetric centering (T_k).
/// Otherwise falls back to symmetric centering (q/2).
pub fn decompose_centering_threshold(levels: usize, log_basis: u32, q: u128) -> u128 {
    let half_q = q / 2;
    let field_bits = 128u32 - q.saturating_sub(1).leading_zeros();
    let total_decomp_bits = (levels as u32).saturating_mul(log_basis);
    if total_decomp_bits == field_bits {
        let b: u128 = 1u128 << log_basis;
        let b_k_minus_1 = if total_decomp_bits >= 128 {
            u128::MAX
        } else {
            (1u128 << total_decomp_bits) - 1
        };
        let t_k = (b / 2 - 1) * (b_k_minus_1 / (b - 1));
        t_k.min(half_q)
    } else {
        half_q
    }
}

trait BalancedSignedDigit: Copy {
    const MAX_LOG_BASIS: u32;
    /// Convert a biased digit field `digit + b/2` (low `log_basis` bits of
    /// `field`) to the balanced digit.
    fn from_biased_field(field: u16, mask: u16, half_b: u16) -> Self;
}

impl BalancedSignedDigit for i8 {
    const MAX_LOG_BASIS: u32 = 8;

    #[inline(always)]
    fn from_biased_field(field: u16, mask: u16, half_b: u16) -> Self {
        (field & mask).wrapping_sub(half_b) as Self
    }
}

/// Truncation of a digit lane to the low 16 bits a digit field reads.
trait LowU16 {
    fn low_u16(self) -> u16;
}

macro_rules! low_u16 {
    ($($lane:ty),*) => {$(
        impl LowU16 for $lane {
            #[inline(always)]
            fn low_u16(self) -> u16 {
                self as u16
            }
        }
    )*};
}
low_u16!(u16, u32, u64);

impl BalancedSignedDigit for i16 {
    const MAX_LOG_BASIS: u32 = 16;

    #[inline(always)]
    fn from_biased_field(field: u16, mask: u16, half_b: u16) -> Self {
        (field & mask).wrapping_sub(half_b) as Self
    }
}

/// Signed integer inputs of [`BalancedDecomposePow2Params::decompose_small_signed_into`].
#[doc(hidden)]
pub trait SmallSignedValue: Copy {
    const MIN: i32;
    const MAX: i32;
    fn widen(self) -> i32;
}

macro_rules! small_signed_value {
    ($($carrier:ty),*) => {$(
        impl SmallSignedValue for $carrier {
            const MIN: i32 = <$carrier>::MIN as i32;
            const MAX: i32 = <$carrier>::MAX as i32;

            #[inline(always)]
            fn widen(self) -> i32 {
                i32::from(self)
            }
        }
    )*};
}
small_signed_value!(i8, i16);

/// Precomputed parameters for balanced power-of-two signed decomposition.
///
/// The balanced digit recurrence `d_k = balanced(c_k mod b)`,
/// `c_{k+1} = (c_k - d_k) / b` is the carry chain of one wide addition: with
/// `H = sum_{k < levels} (b/2) b^k` and `x` the centered value,
/// `x + H = sum_k (d_k + b/2) b^k + c_levels b^levels` and every
/// `d_k + b/2` lies in `[0, b)`. So digit `k` is bit field `k` of `x + H`,
/// minus `b/2`. Because `levels * log_basis <= 128 + log_basis < 192`, the
/// low fields of `x + H` are exact in 192-bit two's-complement arithmetic,
/// and `x + H` is the canonical residue plus `H` (nonnegative side) or plus
/// `H - q` (negative side). The bias words below store those two constants.
#[derive(Clone, Copy, Debug)]
pub struct BalancedDecomposePow2Params<F: Field + CanonicalEncoding> {
    levels: usize,
    log_basis: u32,
    q: u128,
    threshold: u128,
    /// `H` as little-endian 192-bit words, added to residues `<= threshold`.
    bias_nonnegative: [u64; 3],
    /// `H - q mod 2^192`, added to residues `> threshold`.
    bias_negative: [u64; 3],
    field: std::marker::PhantomData<F>,
}

impl<F: Field + CanonicalEncoding> BalancedDecomposePow2Params<F> {
    /// Build decomposition parameters for `levels` digits in base `2^log_basis`.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is outside `1..=16`, if the requested digit
    /// budget exceeds the supported field-width guard, or if `log_basis` is 1
    /// and `levels` exceeds the bit width of `q`.
    #[expect(
        clippy::expect_used,
        reason = "documented panics: invalid decomposition parameters or output shapes are caller bugs"
    )]
    pub fn new(levels: usize, log_basis: u32) -> Self {
        let q = (-F::one())
            .to_u128_checked()
            .expect("Akita field modulus must fit in u128")
            .checked_add(1)
            .expect("Akita field modulus must fit in u128");
        assert!(
            log_basis > 0 && log_basis <= 16,
            "log_basis must be in 1..=16 for signed i16 output"
        );
        let level_count = u32::try_from(levels).expect("levels must fit in u32");
        assert!(
            level_count.saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );
        // Base-2 balanced digits lie in {-1, 0}. Past the field width the
        // threshold is `q / 2`, and the digits cannot reach the positive
        // centered values below it.
        let field_bits = 128 - q.saturating_sub(1).leading_zeros();
        assert!(
            log_basis > 1 || level_count <= field_bits,
            "log_basis 1 needs levels <= the field width"
        );

        let mut bias_nonnegative = [0u64; 3];
        for level in 0..level_count {
            let bit = level * log_basis + log_basis - 1;
            bias_nonnegative[(bit / 64) as usize] |= 1u64 << (bit % 64);
        }
        let (negative0, borrow0) = bias_nonnegative[0].overflowing_sub(q as u64);
        let (partial1, borrow1a) = bias_nonnegative[1].overflowing_sub((q >> 64) as u64);
        let (negative1, borrow1b) = partial1.overflowing_sub(u64::from(borrow0));
        let negative2 = bias_nonnegative[2].wrapping_sub(u64::from(borrow1a | borrow1b));

        Self {
            levels,
            log_basis,
            q,
            threshold: decompose_centering_threshold(levels, log_basis, q),
            bias_nonnegative,
            bias_negative: [negative0, negative1, negative2],
            field: std::marker::PhantomData,
        }
    }

    /// Number of digit planes.
    #[inline]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Base-2 logarithm of the digit basis.
    #[inline]
    pub fn log_basis(&self) -> u32 {
        self.log_basis
    }

    /// Decompose a small signed ring using its canonical field residues.
    ///
    /// Output planes use the same order as field decomposition.
    ///
    /// # Panics
    ///
    /// Panics if the plane count differs from `levels()` or a digit does not fit
    /// the output carrier (`i8` or `i16`).
    #[doc(hidden)]
    #[expect(private_bounds)]
    #[inline]
    pub fn decompose_small_signed_into<T, O, const D: usize>(
        &self,
        ring: &[T; D],
        out: &mut [[O; D]],
    ) where
        T: SmallSignedValue,
        O: BalancedSignedDigit,
    {
        assert_eq!(
            out.len(),
            self.levels,
            "digit plane count must match levels"
        );
        assert!(
            self.log_basis <= O::MAX_LOG_BASIS,
            "basis exceeds output carrier"
        );
        // When every carrier value is its own centered representative,
        // `x + H` is `v + H` in two's complement and the digits read its low
        // `digit_bits()` bits, so the narrowest lane holding them is exact.
        let centered = self.threshold >= T::MAX as u128
            && self.q - self.threshold > T::MIN.unsigned_abs() as u128;
        let bias = self.bias_nonnegative[0];
        match self.digit_bits() {
            bits if centered && bits <= 16 => {
                self.small_signed_lanes(ring, out, |value| {
                    (value as u16).wrapping_add(bias as u16)
                });
            }
            bits if centered && bits <= 32 => {
                self.small_signed_lanes(ring, out, |value| {
                    (value as u32).wrapping_add(bias as u32)
                });
            }
            bits if centered && bits <= 64 => {
                self.small_signed_lanes(ring, out, |value| (value as u64).wrapping_add(bias));
            }
            _ => self.small_signed_residues(ring, out),
        }
    }

    /// Digit planes of `word(v)`, the low bits of `v + H` in one lane type.
    #[inline(always)]
    fn small_signed_lanes<T, O, W, const D: usize>(
        &self,
        ring: &[T; D],
        out: &mut [[O; D]],
        word: impl Fn(i32) -> W,
    ) where
        T: SmallSignedValue,
        O: BalancedSignedDigit,
        W: Copy + std::ops::Shr<u32, Output = W> + LowU16,
    {
        let mask = ((1u32 << self.log_basis) - 1) as u16;
        let half_b = (1u32 << (self.log_basis - 1)) as u16;
        for (level, plane) in out.iter_mut().enumerate() {
            let shift = level as u32 * self.log_basis;
            for (digit, &value) in plane.iter_mut().zip(ring) {
                *digit =
                    O::from_biased_field((word(value.widen()) >> shift).low_u16(), mask, half_b);
            }
        }
    }

    /// Exact path through each value's canonical residue, for parameters
    /// that recenter part of the carrier or read past 64 bits.
    fn small_signed_residues<T, O, const D: usize>(&self, ring: &[T; D], out: &mut [[O; D]])
    where
        T: SmallSignedValue,
        O: BalancedSignedDigit,
    {
        let mask = ((1u32 << self.log_basis) - 1) as u16;
        let half_b = (1u32 << (self.log_basis - 1)) as u16;
        for (coefficient, value) in ring.iter().copied().enumerate() {
            let value = value.widen();
            let canonical = if value >= 0 {
                value as u128
            } else {
                self.q - u128::from(value.unsigned_abs())
            };
            let words = self.biased(canonical);
            for (level, plane) in out.iter_mut().enumerate() {
                let shift = level as u32 * self.log_basis;
                let word = (shift / 64) as usize;
                let bit = shift % 64;
                let mut field = words[word] >> bit;
                if bit + self.log_basis > 64 {
                    field |= words[word + 1] << (64 - bit);
                }
                plane[coefficient] = O::from_biased_field(field as u16, mask, half_b);
            }
        }
    }

    /// Low bits of `x + H` that the digits read.
    #[inline]
    fn digit_bits(&self) -> usize {
        self.levels * self.log_basis as usize
    }

    /// `x + H mod 2^192` for the centered value `x` of `canonical`.
    ///
    /// Branch-free word selection keeps the coefficient loop vectorizable.
    #[inline(always)]
    fn biased(&self, canonical: u128) -> [u64; 3] {
        let low = canonical as u64;
        let high = (canonical >> 64) as u64;
        let threshold_low = self.threshold as u64;
        let threshold_high = (self.threshold >> 64) as u64;
        let negative = (high > threshold_high) || (high == threshold_high) && (low > threshold_low);
        let select = u64::from(negative).wrapping_neg();
        let [nonnegative0, nonnegative1, nonnegative2] = self.bias_nonnegative;
        let [negative0, negative1, negative2] = self.bias_negative;
        let bias0 = nonnegative0 ^ ((nonnegative0 ^ negative0) & select);
        let bias1 = nonnegative1 ^ ((nonnegative1 ^ negative1) & select);
        let bias2 = nonnegative2 ^ ((nonnegative2 ^ negative2) & select);
        let word0 = low.wrapping_add(bias0);
        let carry0 = u64::from(word0 < low);
        let partial1 = high.wrapping_add(bias1);
        let carry1a = u64::from(partial1 < high);
        let word1 = partial1.wrapping_add(carry0);
        let carry1b = u64::from(word1 < partial1);
        [word0, word1, bias2.wrapping_add(carry1a | carry1b)]
    }
}

/// Decompose flat field coefficients into digit-major signed `i8` values.
///
/// The digit for `coefficients[coefficient]` at `level` is written to
/// `out[level * coefficients.len() + coefficient]`. This is the canonical
/// coefficient decomposition primitive for callers whose data is not already
/// grouped into cyclotomic ring elements.
///
/// # Panics
///
/// Panics if `out.len() != coefficients.len() * params.levels`, or if the
/// precomputed parameters use a basis wider than signed `i8` digits.
#[expect(
    clippy::expect_used,
    reason = "documented panics: invalid decomposition parameters or output shapes are caller bugs"
)]
#[inline]
pub fn balanced_decompose_coefficients_pow2_i8_into<F: Field + CanonicalEncoding>(
    coefficients: &[F],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<F>,
) {
    let expected_len = coefficients
        .len()
        .checked_mul(params.levels)
        .expect("flat digit output length overflow");
    assert_eq!(
        out.len(),
        expected_len,
        "flat digit output length must match coefficients * levels",
    );
    assert!(
        params.log_basis <= <i8 as BalancedSignedDigit>::MAX_LOG_BASIS,
        "log_basis must be in 1..=8 for i8 output"
    );
    if coefficients.is_empty() || params.levels == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if coefficients.len().is_multiple_of(4) && params.digit_bits() <= 64 {
        if let Some(canonical) = F::canonical_u32_slice(coefficients) {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: runtime feature detection guarantees NEON, and the
                // length check guarantees every load/store covers four lanes.
                unsafe {
                    aarch64::balanced_decompose_canonical_u32_pow2_i8_neon(canonical, out, params);
                };
                return;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if coefficients.len().is_multiple_of(8) && params.digit_bits() <= 64 {
        if let Some(canonical) = F::canonical_u32_slice(coefficients) {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime feature detection guarantees AVX2, and the
                // length check guarantees every load/store covers eight lanes.
                unsafe {
                    x86::balanced_decompose_canonical_u32_pow2_i8_avx2(canonical, out, params);
                };
                return;
            }
        }
    }
    balanced_decompose_coefficients_pow2_signed_into(coefficients, out, params);
}

/// Coefficients staged per pass. Their biased words stay in L1 while every
/// digit plane is written.
const BIASED_CHUNK: usize = 64;

/// Decompose flat coefficients into digit-major signed digits.
///
/// Dispatches to the widest available instantiation of
/// [`balanced_decompose_coefficients_pow2_signed_kernel`].
///
/// # Panics
///
/// Panics if `out.len() != coefficients.len() * params.levels`.
#[expect(
    clippy::expect_used,
    reason = "documented panics: invalid decomposition parameters or output shapes are caller bugs"
)]
#[inline]
fn balanced_decompose_coefficients_pow2_signed_into<
    F: Field + CanonicalEncoding,
    T: BalancedSignedDigit,
>(
    coefficients: &[F],
    out: &mut [T],
    params: &BalancedDecomposePow2Params<F>,
) {
    let expected_len = coefficients
        .len()
        .checked_mul(params.levels)
        .expect("flat digit output length overflow");
    assert_eq!(
        out.len(),
        expected_len,
        "flat digit output length must match coefficients * levels",
    );
    debug_assert!(params.log_basis <= T::MAX_LOG_BASIS);
    if coefficients.is_empty() || params.levels == 0 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512dq")
        {
            // SAFETY: runtime feature detection guarantees every enabled feature.
            unsafe {
                x86::balanced_decompose_coefficients_pow2_signed_avx512(coefficients, out, params);
            };
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2.
            unsafe {
                x86::balanced_decompose_coefficients_pow2_signed_avx2(coefficients, out, params);
            };
            return;
        }
    }
    balanced_decompose_coefficients_pow2_signed_kernel(coefficients, out, params);
}

/// Add-bias balanced decomposition (see [`BalancedDecomposePow2Params`]).
///
/// Digits only read the low `levels * log_basis` bits of `x + H`, so the
/// kernel stages just the 64-bit words those bits span.
#[inline(always)]
fn balanced_decompose_coefficients_pow2_signed_kernel<
    F: Field + CanonicalEncoding,
    T: BalancedSignedDigit,
>(
    coefficients: &[F],
    out: &mut [T],
    params: &BalancedDecomposePow2Params<F>,
) {
    match params.digit_bits().div_ceil(64) {
        0 | 1 => balanced_decompose_biased_words::<F, T, 1>(coefficients, out, params),
        2 => balanced_decompose_biased_words::<F, T, 2>(coefficients, out, params),
        _ => balanced_decompose_biased_words::<F, T, 3>(coefficients, out, params),
    }
}

/// Each chunk of coefficients is biased into `WORDS` rows of `u64`, then each
/// digit plane is one shift, mask, and subtract per coefficient. Both loops
/// are independent across coefficients, so they vectorize for the target
/// features of the instantiating function.
#[expect(
    clippy::expect_used,
    reason = "documented panics: invalid decomposition parameters or output shapes are caller bugs"
)]
#[inline(always)]
fn balanced_decompose_biased_words<
    F: Field + CanonicalEncoding,
    T: BalancedSignedDigit,
    const WORDS: usize,
>(
    coefficients: &[F],
    out: &mut [T],
    params: &BalancedDecomposePow2Params<F>,
) {
    let width = coefficients.len();
    let log_basis = params.log_basis;
    let mask = ((1u32 << log_basis) - 1) as u16;
    let half_b = (1u32 << (log_basis - 1)) as u16;
    let mut words = [[0u64; BIASED_CHUNK]; WORDS];
    for (chunk_index, chunk) in coefficients.chunks(BIASED_CHUNK).enumerate() {
        // `chunks` never yields more than `BIASED_CHUNK` coefficients, but
        // LLVM cannot see that. Without the explicit bound, the row stores
        // keep a bounds check that forces a scalar, branchy tail.
        let len = chunk.len().min(BIASED_CHUNK);
        for lane in 0..len {
            let biased = params.biased(
                chunk[lane]
                    .to_u128_checked()
                    .expect("Akita field element must fit in u128"),
            );
            for (row, word) in words.iter_mut().zip(biased) {
                row[lane] = word;
            }
        }

        let start = chunk_index * BIASED_CHUNK;
        for (level, plane) in out.chunks_exact_mut(width).enumerate() {
            let shift = level as u32 * log_basis;
            let word = (shift / 64) as usize;
            let bit = shift % 64;
            let digits = &mut plane[start..start + len];
            if bit + log_basis <= 64 {
                for (digit, &low) in digits.iter_mut().zip(&words[word][..len]) {
                    *digit = T::from_biased_field((low >> bit) as u16, mask, half_b);
                }
            } else {
                // A straddling field ends inside `levels * log_basis` bits, so
                // `word + 1 < WORDS`.
                for ((digit, &low), &high) in digits
                    .iter_mut()
                    .zip(&words[word][..len])
                    .zip(&words[word + 1][..len])
                {
                    let field = (low >> bit) | (high << (64 - bit));
                    *digit = T::from_biased_field(field as u16, mask, half_b);
                }
            }
        }
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> CyclotomicRing<F, D> {
    /// Recompose from i8 digit planes; the inverse oracle for
    /// [`Self::balanced_decompose_pow2_i8_into_with_params`].
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is zero or >= 128.
    #[cfg(test)]
    pub(crate) fn gadget_recompose_pow2_i8(digits: &[[i8; D]], log_basis: u32) -> Self
    where
        F: Field + CanonicalEncoding,
    {
        if digits.is_empty() {
            return Self::zero();
        }
        assert!(
            log_basis > 0 && log_basis <= 128,
            "invalid log_basis: {log_basis}"
        );

        if digits.len() == 1 {
            let coeffs = from_fn(|i| F::from_i64(digits[0][i] as i64));
            return Self { coeffs };
        }

        let b = F::from_u128_reduced(1u128 << log_basis);
        let coeffs = from_fn(|i| {
            let mut acc = F::zero();
            let mut power = F::one();
            for plane in digits {
                acc += F::from_i64(plane[i] as i64) * power;
                power *= b;
            }
            acc
        });
        Self { coeffs }
    }

    #[inline]
    /// Decompose using caller-supplied precomputed decomposition parameters.
    pub fn balanced_decompose_pow2_i8_into_with_params(
        &self,
        out: &mut [[i8; D]],
        params: &BalancedDecomposePow2Params<F>,
    ) where
        F: Field + CanonicalEncoding,
    {
        assert!(
            params.log_basis <= <i8 as BalancedSignedDigit>::MAX_LOG_BASIS,
            "log_basis must be in 1..=8 for i8 output"
        );
        balanced_decompose_coefficients_pow2_i8_into(&self.coeffs, out.as_flattened_mut(), params);
    }

    /// Balanced decomposition directly into signed i16 digit planes.
    ///
    /// This is the canonical large-basis path. `log_basis` may be in
    /// `1..=16`; bases 10 and 11 map to `[-512, 511]` and `[-1024, 1023]`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != params.levels()`.
    #[inline]
    pub fn balanced_decompose_pow2_i16_into(
        &self,
        out: &mut [[i16; D]],
        params: &BalancedDecomposePow2Params<F>,
    ) where
        F: Field + CanonicalEncoding,
    {
        balanced_decompose_coefficients_pow2_signed_into(
            &self.coeffs,
            out.as_flattened_mut(),
            params,
        );
    }
}
