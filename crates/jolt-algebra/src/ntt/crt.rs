//! CRT helpers: Garner reconstruction and limb-based modular arithmetic.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

use super::prime::{NttPrime, PrimeWidth};
use crate::Field;
use akita_error::AkitaError;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmallNat {
    limbs: Vec<u32>,
}

impl SmallNat {
    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn mul_u128(&mut self, rhs: u128) {
        if rhs == 0 {
            self.limbs = vec![0];
            return;
        }
        let mut rhs_limbs = Vec::new();
        let mut value = rhs;
        while value != 0 {
            rhs_limbs.push(value as u32);
            value >>= 32;
        }
        let mut out = vec![0u32; self.limbs.len() + rhs_limbs.len()];
        for (i, &lhs) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs) in rhs_limbs.iter().enumerate() {
                let index = i + j;
                let accum = u128::from(out[index]) + u128::from(lhs) * u128::from(rhs) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
            }
            let mut index = i + rhs_limbs.len();
            while carry != 0 {
                if index == out.len() {
                    out.push(0);
                }
                let accum = u128::from(out[index]) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
                index += 1;
            }
        }
        while out.len() > 1 && out.last() == Some(&0) {
            out.pop();
        }
        self.limbs = out;
    }
}

impl Ord for SmallNat {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for SmallNat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact product capacity of a CRT residue profile.
///
/// The product is independent of the residue representation and execution
/// kernels. It can therefore compare homogeneous i16/i32 profiles, mixed
/// profiles, and wider SIMD-specific profiles through one exact bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrtCapacity {
    product: SmallNat,
}

impl CrtCapacity {
    /// Build a capacity from canonical prime moduli.
    pub fn from_prime_moduli(primes: impl IntoIterator<Item = u128>) -> Self {
        let mut product = SmallNat::one();
        for prime in primes {
            product.mul_u128(prime);
        }
        Self { product }
    }

    /// Extend this capacity with one additional canonical prime modulus.
    #[must_use]
    pub fn with_prime_modulus(mut self, prime: u128) -> Self {
        self.product.mul_u128(prime);
        self
    }

    /// Whether this CRT product can reconstruct the requested accumulation.
    ///
    /// The strict exactness condition is
    /// `2 * width * D * floor(q / 2) * rhs_abs_bound < product(primes)`.
    pub fn supports<F: crate::Field + crate::CanonicalEncoding, const D: usize>(
        &self,
        width: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        self.supports_modulus(
            width,
            D,
            (-F::one())
                .to_u128_checked()
                .expect("Akita field element must fit in u128")
                + 1,
            rhs_abs_bound,
        )
    }

    /// Whether this CRT product can reconstruct an accumulation for an
    /// explicitly identified field modulus and runtime ring dimension.
    pub fn supports_modulus(
        &self,
        width: usize,
        ring_dimension: usize,
        modulus: u128,
        rhs_abs_bound: u64,
    ) -> bool {
        let mut required = SmallNat::one();
        required.mul_u128(2);
        required.mul_u128(width as u128);
        required.mul_u128(ring_dimension as u128);
        required.mul_u128(modulus / 2);
        required.mul_u128(u128::from(rhs_abs_bound));
        required < self.product
    }

    /// Conservative maximum matrix width supported at one coefficient bound.
    pub fn max_safe_width<F: crate::Field + crate::CanonicalEncoding, const D: usize>(
        &self,
        rhs_abs_bound: u64,
    ) -> Option<usize> {
        self.max_safe_width_for_modulus(
            D,
            (-F::one())
                .to_u128_checked()
                .expect("Akita field element must fit in u128")
                + 1,
            rhs_abs_bound,
        )
    }

    /// Conservative maximum matrix width for an explicitly identified field
    /// modulus and runtime ring dimension.
    pub fn max_safe_width_for_modulus(
        &self,
        ring_dimension: usize,
        modulus: u128,
        rhs_abs_bound: u64,
    ) -> Option<usize> {
        if rhs_abs_bound == 0 {
            return Some(usize::MAX);
        }
        if modulus <= 1
            || ring_dimension == 0
            || !self.supports_modulus(1, ring_dimension, modulus, rhs_abs_bound)
        {
            return None;
        }
        let mut low = 1usize;
        let mut high = 2usize;
        while self.supports_modulus(high, ring_dimension, modulus, rhs_abs_bound) {
            low = high;
            let Some(next) = high.checked_mul(2) else {
                if self.supports_modulus(usize::MAX, ring_dimension, modulus, rhs_abs_bound) {
                    return Some(usize::MAX);
                }
                high = usize::MAX;
                break;
            };
            high = next;
        }
        while low + 1 < high {
            let mid = low + (high - low) / 2;
            if self.supports_modulus(mid, ring_dimension, modulus, rhs_abs_bound) {
                low = mid;
            } else {
                high = mid;
            }
        }
        Some(low)
    }
}

/// Limb radix bit-width (`2^14`).
pub const RADIX_BITS: u32 = 14;
const RADIX: i32 = 1 << RADIX_BITS;
const RADIX_MASK: i32 = RADIX - 1;

/// Precomputed Garner inverse table for CRT reconstruction.
///
/// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`. Upper triangle and
/// diagonal entries are zero (unused).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarnerData<const K: usize> {
    /// CRT moduli `p_i`, each in `[2, 2^62)`.
    moduli: [u64; K],
    /// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`.
    gamma: [[u64; K]; K],
    /// Shoup quotients `floor(gamma[i][j] * 2^64 / p_i)`.
    gamma_shoup: [[u64; K]; K],
    /// Least multiple of `p_i` that is at least `floor(p_j / 2)`, so adding it
    /// before subtracting a centered digit modulo `p_j` stays nonnegative.
    lift: [[u64; K]; K],
    /// First limb of the run containing limb `i`. Each run's modulus product
    /// is below 2^126, so its centered mixed-radix value fits in i128.
    run_start: [usize; K],
}

impl<const K: usize> GarnerData<K> {
    /// CRT moduli used to construct this immutable precomputation.
    pub fn moduli(&self) -> &[u64; K] {
        &self.moduli
    }

    /// Garner inverses used to construct the cached Shoup quotients.
    pub fn gamma(&self) -> &[[u64; K]; K] {
        &self.gamma
    }

    /// Compute Garner constants from a set of NTT primes.
    pub fn compute<W: PrimeWidth>(primes: &[NttPrime<W>; K]) -> Self {
        Self::try_from_moduli(primes.map(|prime| prime.p.to_i64() as u64))
            .expect("CRT primes must be pairwise coprime")
    }

    pub(crate) fn try_from_moduli(moduli: [u64; K]) -> Result<Self, AkitaError> {
        if moduli
            .iter()
            .any(|&modulus| !(2..1 << 62).contains(&modulus))
        {
            return Err(AkitaError::InvalidSetup(
                "CRT moduli must lie in [2, 2^62)".into(),
            ));
        }
        let mut gamma = [[0; K]; K];
        let mut gamma_shoup = [[0; K]; K];
        let mut lift = [[0; K]; K];
        for i in 1..K {
            let pi = moduli[i];
            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                gamma[i][j] = modular_inverse(moduli[j] % pi, pi)?;
                gamma_shoup[i][j] = ((u128::from(gamma[i][j]) << 64) / u128::from(pi)) as u64;
                lift[i][j] = (moduli[j] / 2).div_ceil(pi) * pi;
            }
        }
        let mut run_start = [0; K];
        let (mut start, mut product) = (0, 1u128);
        for (index, &modulus) in moduli.iter().enumerate() {
            match product.checked_mul(u128::from(modulus)) {
                Some(next) if next < 1 << 126 => product = next,
                _ => (start, product) = (index, u128::from(modulus)),
            }
            run_start[index] = start;
        }
        Ok(Self {
            moduli,
            gamma,
            gamma_shoup,
            lift,
            run_start,
        })
    }

    /// Field images of the radix weights `prod_{j < i} p_j`, followed by the
    /// image of the full CRT modulus.
    pub(crate) fn field_weights<F: Field>(&self) -> ([F; K], F) {
        let mut product = F::one();
        let weights = core::array::from_fn(|index| {
            let weight = product;
            product *= F::from_u64(self.moduli[index]);
            weight
        });
        (weights, product)
    }

    /// Field image of the integer whose centered mixed-radix digits are
    /// `digits`, given the radix weights from [`Self::field_weights`].
    ///
    /// Each run of limbs is Horner-evaluated exactly in i128, so a run costs
    /// one field conversion and at most one field multiplication.
    pub(crate) fn digits_to_field<F: Field>(&self, digits: &[i64; K], weights: &[F; K]) -> F {
        let mut result = F::zero();
        let mut end = K;
        while end > 0 {
            let start = self.run_start[end - 1];
            let mut value = 0i128;
            for index in (start..end).rev() {
                value = value * i128::from(self.moduli[index]) + i128::from(digits[index]);
            }
            let value = F::from_i128(value);
            result += if start == 0 {
                value
            } else {
                value * weights[start]
            };
            end = start;
        }
        result
    }

    /// Replace each column of residues with its centered mixed-radix digits.
    ///
    /// On entry `limbs[i][c]` is the residue of coefficient `c` modulo `p_i`,
    /// which must lie in `(-p_i, p_i)`. On exit it is digit `i`, in
    /// `[-floor(p_i / 2), floor(p_i / 2)]`, where coefficient `c` equals
    /// `sum_i limbs[i][c] * prod_{j < i} p_j`. Each step is a Shoup
    /// multiplication by a precomputed inverse, and each inner loop runs over
    /// independent coefficients, so the per-coefficient chains overlap.
    pub(crate) fn centered_mixed_radix<const D: usize>(&self, limbs: &mut [[i64; D]; K]) {
        for index in 0..K {
            let modulus = self.moduli[index];
            let (prior_limbs, rest) = limbs.split_at_mut(index);
            let Some(column) = rest.first_mut() else {
                return;
            };
            for value in column.iter_mut() {
                *value += i64::from(*value < 0) * modulus as i64;
            }
            for (prior, prior_digits) in prior_limbs.iter().enumerate() {
                let (gamma, gamma_shoup) =
                    (self.gamma[index][prior], self.gamma_shoup[index][prior]);
                let lift = self.lift[index][prior];
                for (value, &prior_digit) in column.iter_mut().zip(prior_digits) {
                    // `0 <= value < p_i`, `lift >= floor(p_j / 2) >= |prior_digit|`,
                    // and all moduli are below 2^62, so the sum is nonnegative
                    // and below 2^64.
                    let shifted = (*value as u64 + lift).wrapping_add_signed(-prior_digit);
                    let quotient = ((u128::from(shifted) * u128::from(gamma_shoup)) >> 64) as u64;
                    let digit = shifted
                        .wrapping_mul(gamma)
                        .wrapping_sub(quotient.wrapping_mul(modulus));
                    *value = (digit - u64::from(digit >= modulus) * modulus) as i64;
                }
            }
            for value in column.iter_mut() {
                *value -= i64::from(*value > (modulus / 2) as i64) * modulus as i64;
            }
        }
    }
}

pub(crate) fn modular_inverse(value: u64, modulus: u64) -> Result<u64, AkitaError> {
    let (mut old_remainder, mut remainder) = (i128::from(modulus), i128::from(value));
    let (mut old_coefficient, mut coefficient) = (0i128, 1i128);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_coefficient, coefficient) = (coefficient, old_coefficient - quotient * coefficient);
    }
    if old_remainder != 1 {
        return Err(AkitaError::InvalidSetup(
            "CRT primes are not pairwise coprime".into(),
        ));
    }
    Ok(old_coefficient.rem_euclid(i128::from(modulus)) as u64)
}

/// Fixed-width radix-`2^14` integer.
///
/// Limbs are little-endian: `limbs[0]` is least significant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimbQ<const L: usize> {
    /// Little-endian limbs.
    pub limbs: [u16; L],
}

impl<const L: usize> Default for LimbQ<L> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const L: usize> LimbQ<L> {
    /// Zero value.
    #[inline]
    pub const fn zero() -> Self {
        Self { limbs: [0; L] }
    }
}

impl<const L: usize> From<u128> for LimbQ<L> {
    fn from(mut x: u128) -> Self {
        let mut out = [0u16; L];
        for (i, limb) in out.iter_mut().enumerate() {
            if i + 1 < L {
                *limb = (x & (RADIX_MASK as u128)) as u16;
                x >>= RADIX_BITS;
            } else {
                *limb = x as u16;
            }
        }
        Self { limbs: out }
    }
}

impl<const L: usize> TryFrom<LimbQ<L>> for u128 {
    type Error = &'static str;

    fn try_from(limb: LimbQ<L>) -> Result<Self, Self::Error> {
        if (L as u32) * RADIX_BITS > 128 {
            return Err("LimbQ too wide for u128");
        }
        let mut acc = 0u128;
        for i in (0..L).rev() {
            acc <<= RADIX_BITS;
            acc |= limb.limbs[i] as u128;
        }
        Ok(acc)
    }
}

impl<const L: usize> PartialOrd for LimbQ<L> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const L: usize> Ord for LimbQ<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..L).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl<const L: usize> Add for LimbQ<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut carry = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let s = self.limbs[i] as i32 + rhs.limbs[i] as i32 + carry;
            if i + 1 < L {
                carry = s >> RADIX_BITS;
                *out_limb = (s & RADIX_MASK) as u16;
            } else {
                *out_limb = s as u16;
            }
        }
        Self { limbs: out }
    }
}

impl<const L: usize> Sub for LimbQ<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut borrow = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let d = self.limbs[i] as i32 - rhs.limbs[i] as i32 + borrow;
            if i + 1 < L {
                borrow = d >> 31;
                *out_limb = (d - borrow * RADIX) as u16;
            } else {
                *out_limb = d as u16;
            }
        }
        Self { limbs: out }
    }
}

impl<const L: usize> fmt::Display for LimbQ<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(val) = u128::try_from(*self) {
            write!(f, "{val}")
        } else {
            write!(f, "LimbQ{:?}", self.limbs)
        }
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::CrtCapacity;
    use crate::ntt::tables::{I16_TAIL_PRIME, Q32_PRIMES};
    use crate::PrimeWidth;
    use jolt_field::Prime32Offset99;

    #[test]
    fn q32_capacity_matches_existing_exact_widths() {
        let base =
            CrtCapacity::from_prime_moduli(Q32_PRIMES.iter().map(|prime| prime.p.to_i64() as u128));
        assert_eq!(
            base.max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(63)
        );
        assert_eq!(
            base.clone()
                .with_prime_modulus(I16_TAIL_PRIME.p as u128)
                .max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(786_406)
        );
    }

    #[test]
    fn mixed_wide_and_small_capacity_is_representation_independent() {
        let mixed =
            CrtCapacity::from_prime_moduli([1_125_899_906_826_241u128, I16_TAIL_PRIME.p as u128]);
        assert_eq!(
            mixed.max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(768)
        );
        assert!(mixed.supports::<Prime32Offset99, 128>(128, 32_768));
    }
}

#[cfg(test)]
mod garner_tests {
    use super::GarnerData;
    use crate::ntt::ifma52::IFMA52_PRIMES;
    use crate::ntt::tables::{I16_TAIL_PRIME, I32_RAW_PRIMES};
    use crate::Field;
    use jolt_field::{Prime128OffsetA7F7, Prime32Offset99};

    /// Checks range and `sum_k d_k prod_{j<k} p_j == r_i (mod p_i)` for all i.
    fn mixed_radix<const K: usize>(garner: &GarnerData<K>, residues: [i64; K]) -> [i64; K] {
        let mut limbs = residues.map(|residue| [residue]);
        garner.centered_mixed_radix(&mut limbs);
        limbs.map(|[digit]| digit)
    }

    fn assert_mixed_radix<const K: usize>(garner: &GarnerData<K>, residues: [i64; K]) {
        let digits = mixed_radix(garner, residues);
        for (i, &modulus) in garner.moduli.iter().enumerate() {
            let modulus = i128::from(modulus);
            assert!(i128::from(digits[i]).abs() <= modulus / 2);
            let mut value = 0i128;
            let mut weight = 1i128;
            for (k, &digit) in digits.iter().enumerate() {
                value = (value + i128::from(digit) * weight).rem_euclid(modulus);
                weight = (weight * i128::from(garner.moduli[k])).rem_euclid(modulus);
            }
            assert_eq!(value, i128::from(residues[i]).rem_euclid(modulus));
        }
    }

    /// Checks the i128 run evaluation against one field product per digit.
    fn assert_field_lift<F: Field, const K: usize>(garner: &GarnerData<K>, digits: &[i64; K]) {
        let (weights, _) = garner.field_weights::<F>();
        let expected = digits
            .iter()
            .zip(weights)
            .fold(F::zero(), |sum, (&digit, weight)| {
                sum + F::from_i64(digit) * weight
            });
        assert_eq!(garner.digits_to_field(digits, &weights), expected);
    }

    fn check_profile<const K: usize>(moduli: [u64; K]) {
        let garner = GarnerData::try_from_moduli(moduli).unwrap();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..4096 {
            let residues = core::array::from_fn(|limb| {
                let modulus = moduli[limb] as i64;
                let half = modulus / 2;
                match (round + limb) % 5 {
                    0 => -half,
                    1 => half,
                    2 => modulus - 1,
                    3 => 1 - modulus,
                    _ => {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state % (2 * modulus as u64 - 1)) as i64 - (modulus - 1)
                    }
                }
            });
            assert_mixed_radix(&garner, residues);
            let digits = mixed_radix(&garner, residues);
            assert_field_lift::<Prime128OffsetA7F7, K>(&garner, &digits);
            assert_field_lift::<Prime32Offset99, K>(&garner, &digits);
        }
    }

    #[test]
    fn shoup_mixed_radix_matches_crt_congruences() {
        check_profile(I32_RAW_PRIMES.map(|prime| prime as u64));
        check_profile(IFMA52_PRIMES);
        check_profile([IFMA52_PRIMES[0], 12_289, I32_RAW_PRIMES[0] as u64]);
        check_profile([(1 << 62) - 57, (1 << 61) - 1, 3]);
        check_profile([I16_TAIL_PRIME.p as u64; 1]);
        check_profile([
            I32_RAW_PRIMES[0] as u64,
            I32_RAW_PRIMES[1] as u64,
            I32_RAW_PRIMES[2] as u64,
            I32_RAW_PRIMES[3] as u64,
            I32_RAW_PRIMES[4] as u64,
            I32_RAW_PRIMES[5] as u64,
            I16_TAIL_PRIME.p as u64,
        ]);
    }

    #[test]
    fn rejects_moduli_outside_shoup_range() {
        assert!(GarnerData::try_from_moduli([1u64 << 62, 3]).is_err());
        assert!(GarnerData::try_from_moduli([1u64, 3]).is_err());
    }
}
