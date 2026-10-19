//! Canonical 50-bit NTT arithmetic for AVX-512IFMA hosts.

use akita_error::AkitaError;

use super::prime::is_prime;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

/// The three largest primes below `2^50` with `2^14 | (p - 1)`.
pub const IFMA52_PRIMES: [u64; 3] = [
    1_125_899_906_826_241,
    1_125_899_906_629_633,
    1_125_899_905_744_897,
];

pub(crate) const RADIX: u64 = 1 << 52;
pub(crate) const MASK: u64 = RADIX - 1;

/// One canonical-residue prime and its reduction constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ifma52Prime {
    /// Prime modulus.
    pub modulus: u64,
    /// `2^52 mod p`, with its Shoup quotient.
    pub(crate) radix: u64,
    pub(crate) radix_precon: u64,
    /// `2^104 mod p`, with its Shoup quotient.
    pub(crate) radix_squared: u64,
    pub(crate) radix_squared_precon: u64,
}

impl Ifma52Prime {
    /// Validate and prepare one IFMA52 prime.
    pub fn new(modulus: u64) -> Result<Self, AkitaError> {
        // The range check runs first, so the cast is lossless.
        if modulus <= (1 << 49) || modulus >= (1 << 50) || !is_prime(modulus as i64) {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 modulus must be a prime between 2^49 and 2^50".into(),
            ));
        }
        let mut prime = Self {
            modulus,
            radix: 0,
            radix_precon: 0,
            radix_squared: 0,
            radix_squared_precon: 0,
        };
        prime.radix = RADIX % modulus;
        prime.radix_squared = prime.mul(prime.radix, prime.radix);
        prime.radix_precon = prime.precondition(prime.radix);
        prime.radix_squared_precon = prime.precondition(prime.radix_squared);
        Ok(prime)
    }

    #[inline]
    pub(crate) fn add(self, lhs: u64, rhs: u64) -> u64 {
        let sum = lhs + rhs;
        if sum >= self.modulus {
            sum - self.modulus
        } else {
            sum
        }
    }

    #[inline]
    pub(crate) fn sub(self, lhs: u64, rhs: u64) -> u64 {
        if lhs >= rhs {
            lhs - rhs
        } else {
            lhs + self.modulus - rhs
        }
    }

    #[inline]
    pub(crate) fn mul(self, lhs: u64, rhs: u64) -> u64 {
        ((u128::from(lhs) * u128::from(rhs)) % u128::from(self.modulus)) as u64
    }

    #[inline]
    pub(crate) fn canonical_i16(self, value: i16) -> u64 {
        if value >= 0 {
            value as u64
        } else {
            self.modulus - u64::from(value.unsigned_abs())
        }
    }

    fn pow(self, mut base: u64, mut exponent: u64) -> u64 {
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.mul(result, base);
            }
            base = self.mul(base, base);
            exponent >>= 1;
        }
        result
    }

    fn inverse(self, value: u64) -> u64 {
        self.pow(value, self.modulus - 2)
    }

    pub(crate) fn precondition(self, value: u64) -> u64 {
        ((u128::from(value) << 52) / u128::from(self.modulus)) as u64
    }
}

/// One ring's residues modulo an IFMA52 prime.
///
/// The 64-byte alignment keeps every 512-bit access within one cache line;
/// the transform and dot kernels are load-bound, and line-split accesses made
/// their cost depend on where the caller's stack or heap happened to land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Ifma52Residues<const D: usize>(pub [u64; D]);

/// Eight per-lane multipliers with their Shoup quotients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub(crate) struct LaneMultiplier {
    pub(crate) value: [u64; 8],
    pub(crate) precon: [u64; 8],
}

impl LaneMultiplier {
    fn new(prime: Ifma52Prime, value: [u64; 8]) -> Self {
        Self {
            value,
            precon: value.map(|value| prime.precondition(value)),
        }
    }
}

/// Position in standard order of the value that a transform stores at offset
/// `t` of each 16-value block.
///
/// The SIMD forward finishes its last three stages with the two halves of a
/// block interleaved, and stores them that way rather than spending shuffles
/// on restoring the standard order: offset `t` holds standard offset
/// `rotl4(t)`. Pointwise products do not depend on the order, and the inverse
/// reads it directly.
const STORED_ORDER: [usize; 16] = {
    let mut order = [0; 16];
    let mut t = 0;
    while t < 16 {
        order[t] = ((t << 1) | (t >> 3)) & 15;
        t += 1;
    }
    order
};

/// Negacyclic NTT tables for one 50-bit prime.
///
/// `forward[m]` is `psi^brv(m)` for the primitive `2D`-th root `psi` and the
/// `log2 D`-bit reversal `brv`; stage `s` of the forward transform multiplies
/// block `k` by `forward[2^s + k]`, which merges the negacyclic twist into
/// the butterflies. `inverse[m]` is its inverse.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Ifma52Twiddles<const D: usize> {
    pub(crate) forward: [u64; D],
    pub(crate) forward_precon: [u64; D],
    pub(crate) inverse: [u64; D],
    pub(crate) inverse_precon: [u64; D],
    /// Per 16-value block, the lane multipliers of the last three forward
    /// stages in their interleaved lane order.
    pub(crate) forward_lanes: Vec<[LaneMultiplier; 3]>,
    /// Per 16-value block, the lane multipliers of the first three inverse
    /// stages, indexed like `forward_lanes`.
    pub(crate) inverse_lanes: Vec<[LaneMultiplier; 3]>,
    /// `D^{-1}` and `D^{-1} inverse[1]`, which the last inverse stage applies.
    pub(crate) inverse_scale: [u64; 2],
    pub(crate) inverse_scale_precon: [u64; 2],
}

impl<const D: usize> Ifma52Twiddles<D> {
    /// Compute tables for a supported power-of-two ring degree.
    pub fn compute(prime: Ifma52Prime) -> Result<Self, AkitaError> {
        if D < 64 || !D.is_power_of_two() || !(prime.modulus - 1).is_multiple_of(2 * D as u64) {
            return Err(AkitaError::InvalidSetup(format!(
                "IFMA52 prime does not support ring degree {D}"
            )));
        }
        let exponent = (prime.modulus - 1) / (2 * D as u64);
        // For a prime modulus the search stops at the least quadratic
        // non-residue, which is small; the bound turns any other outcome into
        // an error instead of an endless search.
        let psi = (2..1 << 16)
            .map(|candidate| prime.pow(candidate, exponent))
            .find(|&candidate| prime.pow(candidate, D as u64) == prime.modulus - 1)
            .ok_or_else(|| AkitaError::InvalidSetup("IFMA52 root search failed".into()))?;
        let psi_inverse = prime.inverse(psi);
        let bits = D.trailing_zeros();
        let bit_reversed = |index: usize| (index.reverse_bits() >> (usize::BITS - bits)) as u64;
        let forward: [u64; D] = std::array::from_fn(|index| prime.pow(psi, bit_reversed(index)));
        let inverse: [u64; D] =
            std::array::from_fn(|index| prime.pow(psi_inverse, bit_reversed(index)));
        let lanes = |table: &[u64; D], block: usize| {
            [
                std::array::from_fn(|lane| table[D / 8 + 2 * block + lane / 4]),
                std::array::from_fn(|lane| table[D / 4 + 4 * block + lane / 2]),
                std::array::from_fn(|lane| table[D / 2 + 8 * block + lane]),
            ]
            .map(|value| LaneMultiplier::new(prime, value))
        };
        let inverse_d = prime.inverse(D as u64);
        let inverse_scale = [inverse_d, prime.mul(inverse_d, inverse[1])];
        Ok(Self {
            forward,
            forward_precon: forward.map(|value| prime.precondition(value)),
            inverse,
            inverse_precon: inverse.map(|value| prime.precondition(value)),
            forward_lanes: (0..D / 16).map(|block| lanes(&forward, block)).collect(),
            inverse_lanes: (0..D / 16).map(|block| lanes(&inverse, block)).collect(),
            inverse_scale,
            inverse_scale_precon: inverse_scale.map(|value| prime.precondition(value)),
        })
    }
}

/// Whether this process can execute the AVX-512IFMA kernels.
#[must_use]
pub fn ifma52_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Whether IFMA52 is available and SIMD has not been globally disabled.
#[must_use]
pub fn ifma52_enabled() -> bool {
    ifma52_available() && std::env::var("AKITA_SCALAR_NTT").ok().as_deref() != Some("1")
}

/// Forward transform of canonical residues.
///
/// The output is in transform order (see [`STORED_ORDER`]) and below `4p`
/// on the IFMA path; the scalar path returns canonical residues.
#[inline(always)]
pub(crate) fn forward<const D: usize>(
    values: &mut Ifma52Residues<D>,
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
    use_ifma: bool,
) {
    debug_assert!(values.0.iter().all(|&value| value < prime.modulus));
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::forward(values, prime, twiddles) };
        debug_assert!(values.0.iter().all(|&value| value < 4 * prime.modulus));
        return;
    }
    scalar_forward(&mut values.0, prime, twiddles);
}

/// [`forward`] of signed `i16` coefficients.
#[inline(always)]
pub(crate) fn forward_i16<const D: usize>(
    values: &mut Ifma52Residues<D>,
    coefficients: &[i16; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
    use_ifma: bool,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::forward_i16(values, coefficients, prime, twiddles) };
        debug_assert!(values.0.iter().all(|&value| value < 4 * prime.modulus));
        return;
    }
    values.0 = coefficients.map(|coefficient| prime.canonical_i16(coefficient));
    scalar_forward(&mut values.0, prime, twiddles);
}

/// Inverse transform of canonical residues in transform order, into
/// canonical residues in standard order. The IFMA kernel also accepts
/// residues below `2p`.
#[inline(always)]
pub(crate) fn inverse<const D: usize>(
    values: &mut Ifma52Residues<D>,
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
    use_ifma: bool,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        debug_assert!(values.0.iter().all(|&value| value < 2 * prime.modulus));
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::inverse(values, prime, twiddles) };
        debug_assert!(values.0.iter().all(|&value| value < prime.modulus));
        return;
    }
    debug_assert!(values.0.iter().all(|&value| value < prime.modulus));
    scalar_inverse(&mut values.0, prime, twiddles);
}

/// Most products one [`Ifma52Accumulator`] absorbs between folds.
pub(crate) const IFMA52_ACCUMULATOR_TERMS: usize = 1 << 11;

/// Unreduced pointwise dot products: coefficient `c` holds
/// `high[c] 2^52 + low[c]`.
///
/// Each product of two residues below `2^52` adds its low and high 52 bits
/// to `low` and `high`. A fold leaves `low` below `p` and `high` zero, so
/// for up to [`IFMA52_ACCUMULATOR_TERMS`] products after it `low` stays
/// below `p + 2^63` and `high` below `2^63`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct Ifma52Accumulator<const D: usize> {
    pub(crate) low: Ifma52Residues<D>,
    pub(crate) high: Ifma52Residues<D>,
}

impl<const D: usize> Ifma52Accumulator<D> {
    pub(crate) const ZERO: Self = Self {
        low: Ifma52Residues([0; D]),
        high: Ifma52Residues([0; D]),
    };

    /// Add `sum_i lhs[i] * rhs[i]` pointwise. Every residue must lie below
    /// `2^52`.
    #[inline(always)]
    pub(crate) fn accumulate(
        &mut self,
        lhs: &[Ifma52Residues<D>],
        rhs: &[Ifma52Residues<D>],
        use_ifma: bool,
    ) {
        debug_assert_eq!(lhs.len(), rhs.len());
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = use_ifma;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_ifma {
            // SAFETY: runtime feature detection covers every enabled instruction.
            unsafe { x86::dot_accumulate(self, lhs, rhs) };
            return;
        }
        for (lhs, rhs) in lhs.iter().zip(rhs) {
            for (((low, high), &lhs), &rhs) in self
                .low
                .0
                .iter_mut()
                .zip(&mut self.high.0)
                .zip(&lhs.0)
                .zip(&rhs.0)
            {
                let product = u128::from(lhs) * u128::from(rhs);
                *low += product as u64 & MASK;
                *high += (product >> 52) as u64;
            }
        }
    }

    /// Canonical residues of the accumulated sums.
    #[inline(always)]
    pub(crate) fn reduce(&self, prime: Ifma52Prime, use_ifma: bool) -> Ifma52Residues<D> {
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = use_ifma;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_ifma {
            // SAFETY: runtime feature detection covers every enabled instruction.
            return unsafe { x86::dot_reduce(self, prime) };
        }
        Ifma52Residues(std::array::from_fn(|index| {
            let sum = (u128::from(self.high.0[index]) << 52) + u128::from(self.low.0[index]);
            (sum % u128::from(prime.modulus)) as u64
        }))
    }

    /// Replace the sums by their canonical residues, which restores the
    /// capacity for [`IFMA52_ACCUMULATOR_TERMS`] more products.
    #[inline(always)]
    pub(crate) fn fold(&mut self, prime: Ifma52Prime, use_ifma: bool) {
        self.low = self.reduce(prime, use_ifma);
        self.high = Ifma52Residues([0; D]);
    }
}

/// Merged-twist Cooley–Tukey transform of canonical residues, stored in
/// [`STORED_ORDER`].
fn scalar_forward<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    let (mut len, mut blocks) = (D / 2, 1);
    while len != 0 {
        for block in 0..blocks {
            let twiddle = twiddles.forward[blocks + block];
            for index in 2 * len * block..2 * len * block + len {
                let product = prime.mul(values[index + len], twiddle);
                let value = values[index];
                values[index] = prime.add(value, product);
                values[index + len] = prime.sub(value, product);
            }
        }
        len /= 2;
        blocks *= 2;
    }
    for block in values.chunks_exact_mut(16) {
        let standard: [u64; 16] = std::array::from_fn(|offset| block[offset]);
        for (value, &offset) in block.iter_mut().zip(&STORED_ORDER) {
            *value = standard[offset];
        }
    }
}

/// Inverse of [`scalar_forward`] on canonical residues.
fn scalar_inverse<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    for block in values.chunks_exact_mut(16) {
        let stored: [u64; 16] = std::array::from_fn(|offset| block[offset]);
        for (&value, &offset) in stored.iter().zip(&STORED_ORDER) {
            block[offset] = value;
        }
    }
    let (mut len, mut blocks) = (1, D / 2);
    while len < D {
        for block in 0..blocks {
            let twiddle = twiddles.inverse[blocks + block];
            for index in 2 * len * block..2 * len * block + len {
                let (x, y) = (values[index], values[index + len]);
                values[index] = prime.add(x, y);
                values[index + len] = prime.mul(prime.sub(x, y), twiddle);
            }
        }
        len *= 2;
        blocks /= 2;
    }
    for value in values.iter_mut() {
        *value = prime.mul(*value, twiddles.inverse_scale[0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<const D: usize>(prime: Ifma52Prime, seed: u64) -> Ifma52Residues<D> {
        Ifma52Residues(std::array::from_fn(|index| {
            (index as u64 * 1_000_003 + seed).wrapping_mul(0x9e37_79b9_7f4a_7c15) % prime.modulus
        }))
    }

    fn canonical<const D: usize>(
        values: Ifma52Residues<D>,
        prime: Ifma52Prime,
    ) -> Ifma52Residues<D> {
        Ifma52Residues(values.0.map(|value| value % prime.modulus))
    }

    fn round_trip<const D: usize>(use_ifma: bool) {
        for &modulus in &IFMA52_PRIMES {
            let prime = Ifma52Prime::new(modulus).expect("prime");
            let twiddles = Ifma52Twiddles::<D>::compute(prime).expect("twiddles");
            let original = sample::<D>(prime, 17);
            let mut transformed = original;
            forward(&mut transformed, prime, &twiddles, use_ifma);
            assert!(transformed.0.iter().all(|&value| value < 4 * modulus));
            let mut transformed = canonical(transformed, prime);
            inverse(&mut transformed, prime, &twiddles, use_ifma);
            assert_eq!(transformed, original, "modulus={modulus}, D={D}");
        }
    }

    /// Negacyclic products through the transforms match schoolbook products.
    fn negacyclic_product<const D: usize>(use_ifma: bool) {
        let prime = Ifma52Prime::new(IFMA52_PRIMES[1]).expect("prime");
        let twiddles = Ifma52Twiddles::<D>::compute(prime).expect("twiddles");
        let lhs = sample::<D>(prime, 3);
        let rhs: [i16; D] = std::array::from_fn(|index| (index as i16).wrapping_mul(-7919));
        let mut expected = [0; D];
        for (i, &a) in lhs.0.iter().enumerate() {
            for (j, &b) in rhs.iter().enumerate() {
                let product = prime.mul(a, prime.canonical_i16(b));
                let slot = &mut expected[(i + j) % D];
                *slot = if i + j < D {
                    prime.add(*slot, product)
                } else {
                    prime.sub(*slot, product)
                };
            }
        }
        let mut lhs_transformed = lhs;
        forward(&mut lhs_transformed, prime, &twiddles, use_ifma);
        let mut rhs_transformed = Ifma52Residues([0; D]);
        forward_i16(&mut rhs_transformed, &rhs, prime, &twiddles, use_ifma);
        let mut accumulator = Ifma52Accumulator::ZERO;
        accumulator.accumulate(&[lhs_transformed], &[rhs_transformed], use_ifma);
        let mut product = accumulator.reduce(prime, use_ifma);
        inverse(&mut product, prime, &twiddles, use_ifma);
        assert_eq!(product.0, expected, "D={D}, use_ifma={use_ifma}");
    }

    /// A full accumulator of maximal lazy products reduces exactly.
    fn saturated_accumulator<const D: usize>(use_ifma: bool) {
        for &modulus in &IFMA52_PRIMES {
            let prime = Ifma52Prime::new(modulus).expect("prime");
            let largest = Ifma52Residues([4 * modulus - 1; D]);
            let terms = vec![largest; IFMA52_ACCUMULATOR_TERMS];
            let mut accumulator = Ifma52Accumulator::ZERO;
            accumulator.accumulate(&terms, &terms, use_ifma);
            accumulator.fold(prime, use_ifma);
            accumulator.accumulate(&terms, &terms, use_ifma);
            let square = prime.mul(largest.0[0] % modulus, largest.0[0] % modulus);
            let expected = prime.mul(square, (2 * IFMA52_ACCUMULATOR_TERMS) as u64);
            assert_eq!(accumulator.reduce(prime, use_ifma).0, [expected; D]);
        }
    }

    macro_rules! at_degrees {
        ($check:ident, $use_ifma:expr, $($degree:literal),+) => {
            $($check::<$degree>($use_ifma);)+
        };
    }

    #[test]
    fn prime_requires_a_prime_modulus_between_2_49_and_2_50() {
        // 2^49 + 1 and 2^50 - 1 are multiples of 3, and
        // 1125899772623531 = 33554393 * 33554467.
        for modulus in [
            (1 << 49) - 1,
            1 << 49,
            (1 << 49) + 1,
            1_125_899_772_623_531,
            (1 << 50) - 1,
            (1 << 50) + 1,
        ] {
            assert!(Ifma52Prime::new(modulus).is_err(), "{modulus}");
        }
        for modulus in IFMA52_PRIMES {
            assert!(Ifma52Prime::new(modulus).is_ok(), "{modulus}");
        }
    }

    #[test]
    fn scalar_transforms_round_trip_and_multiply() {
        at_degrees!(round_trip, false, 64, 128, 256, 512, 1024, 2048);
        at_degrees!(negacyclic_product, false, 64, 128, 256);
        at_degrees!(saturated_accumulator, false, 64);
    }

    #[test]
    fn dispatched_transforms_match_scalar() {
        if !ifma52_available() {
            return;
        }
        at_degrees!(round_trip, true, 64, 128, 256, 512, 1024, 2048);
        at_degrees!(negacyclic_product, true, 64, 128, 256, 512, 1024, 2048);
        at_degrees!(saturated_accumulator, true, 64, 512);
        simd_matches_scalar::<64>();
        simd_matches_scalar::<128>();
        simd_matches_scalar::<256>();
        simd_matches_scalar::<512>();
        simd_matches_scalar::<1024>();
        simd_matches_scalar::<2048>();
    }

    fn simd_matches_scalar<const D: usize>() {
        for &modulus in &IFMA52_PRIMES {
            let prime = Ifma52Prime::new(modulus).expect("prime");
            let twiddles = Ifma52Twiddles::<D>::compute(prime).expect("twiddles");
            // The extreme canonical and `i16` inputs drive the lazy forward
            // outputs toward `4p`.
            for fill in [0, modulus - 1] {
                let (mut simd, mut scalar) = (Ifma52Residues([fill; D]), Ifma52Residues([fill; D]));
                forward(&mut simd, prime, &twiddles, true);
                forward(&mut scalar, prime, &twiddles, false);
                assert!(simd.0.iter().all(|&value| value < 4 * modulus));
                assert_eq!(canonical(simd, prime), scalar, "forward {fill}, D={D}");
            }
            for digits in [
                std::array::from_fn(|index| (index as i16).wrapping_mul(12_345) ^ 0x5a5a),
                [i16::MIN; D],
                [i16::MAX; D],
                std::array::from_fn(|index| [i16::MIN, i16::MAX][index % 2]),
            ] {
                let (mut simd, mut scalar) = (Ifma52Residues([0; D]), Ifma52Residues([0; D]));
                forward_i16(&mut simd, &digits, prime, &twiddles, true);
                forward_i16(&mut scalar, &digits, prime, &twiddles, false);
                assert!(simd.0.iter().all(|&value| value < 4 * modulus));
                assert_eq!(canonical(simd, prime), scalar, "forward_i16, D={D}");
            }

            let lazy = Ifma52Residues(sample::<D>(prime, 5).0.map(|value| value + modulus));
            for lazy in [lazy, Ifma52Residues([2 * modulus - 1; D])] {
                let (mut simd, mut scalar) = (lazy, canonical(lazy, prime));
                inverse(&mut simd, prime, &twiddles, true);
                inverse(&mut scalar, prime, &twiddles, false);
                assert_eq!(simd, scalar, "inverse, D={D}");
            }
        }
    }

    #[test]
    #[ignore = "requires AVX-512F/DQ/IFMA hardware or emulation"]
    fn ifma52_hardware_round_trip_does_not_fallback() {
        assert!(ifma52_available(), "AVX-512F/DQ/IFMA is unavailable");
        assert!(ifma52_enabled(), "IFMA52 dispatch was not selected");
        dispatched_transforms_match_scalar();
    }
}
