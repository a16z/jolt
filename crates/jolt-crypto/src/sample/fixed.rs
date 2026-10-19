//! Constant-time 192-bit fixed point and the public high-precision routines
//! that compute its constants.

use num_bigint::BigUint;
use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable};

/// A value in `[0, 1)` stored as `Σ limb_i 2^{64i} / 2^{192}`.
///
/// Every operation runs a fixed instruction sequence independent of the
/// limb values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fix192(pub(crate) [u64; 3]);

impl ConditionallySelectable for Fix192 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let [a0, a1, a2] = a.0;
        let [b0, b1, b2] = b.0;
        Self([
            u64::conditional_select(&a0, &b0, choice),
            u64::conditional_select(&a1, &b1, choice),
            u64::conditional_select(&a2, &b2, choice),
        ])
    }
}

impl Fix192 {
    /// `1 − 2^{−192}`, the multiplicative start of a product.
    pub(crate) const ALMOST_ONE: Self = Self([u64::MAX; 3]);

    pub(crate) fn random<R: RngCore>(rng: &mut R) -> Self {
        Self([rng.next_u64(), rng.next_u64(), rng.next_u64()])
    }

    /// The top 192 bits of the 384-bit product: `⌊a·b·2^{192}⌋ / 2^{192}`.
    #[expect(
        clippy::indexing_slicing,
        reason = "loop indices are bounded by the fixed limb counts"
    )]
    pub(crate) fn mul(&self, rhs: &Self) -> Self {
        let mut wide = [0u64; 6];
        for i in 0..3 {
            let mut carry = 0u128;
            for j in 0..3 {
                let cur =
                    u128::from(wide[i + j]) + u128::from(self.0[i]) * u128::from(rhs.0[j]) + carry;
                wide[i + j] = cur as u64;
                carry = cur >> 64;
            }
            wide[i + 3] = carry as u64;
        }
        Self([wide[3], wide[4], wide[5]])
    }

    /// `self < rhs`, from the borrow of `self − rhs`.
    pub(crate) fn ct_lt(&self, rhs: &Self) -> Choice {
        let mut borrow = 0u64;
        for (a, b) in self.0.iter().zip(rhs.0.iter()) {
            let (d, c1) = a.overflowing_sub(*b);
            let (_, c2) = d.overflowing_sub(borrow);
            borrow = u64::from(c1 | c2);
        }
        Choice::from(borrow as u8)
    }

    /// Right shift by a public amount.
    pub(crate) fn shr(&self, shift: u32) -> Self {
        let wide = (u128::from(self.0[1]) << 64) | u128::from(self.0[0]);
        let top = u128::from(self.0[2]);
        let value = (top, wide);
        let shifted = match shift {
            0 => value,
            1..=127 => (top >> shift, (wide >> shift) | (top << (128 - shift))),
            128..=191 => (0, top >> (shift - 128)),
            _ => (0, 0),
        };
        Self([shifted.1 as u64, (shifted.1 >> 64) as u64, shifted.0 as u64])
    }

    /// Left shift by a public `shift ∈ [1, 63]`, and whether any set bit was
    /// shifted out.
    pub(crate) fn shl_overflowing(&self, shift: u32) -> (Self, Choice) {
        let [l0, l1, l2] = self.0;
        let lost = l2 >> (64 - shift);
        let out = Self([
            l0 << shift,
            (l1 << shift) | (l0 >> (64 - shift)),
            (l2 << shift) | (l1 >> (64 - shift)),
        ]);
        (out, !ct_is_zero_u64(lost))
    }

    /// Right shift by a secret `shift ∈ [0, 255]`: a barrel shifter whose
    /// eight stages always run.
    pub(crate) fn ct_shr(&self, shift: u8) -> Self {
        let mut out = *self;
        for stage in 0..8 {
            let bit = Choice::from((shift >> stage) & 1);
            out = Self::conditional_select(&out, &out.shr(1 << stage), bit);
        }
        out
    }

    /// `∏ factors[i]^{bit_i(exponent)}` over every listed factor, starting
    /// from [`Self::ALMOST_ONE`]. The number of multiplications is the
    /// table length, never the exponent's bit pattern.
    pub(crate) fn ct_product(factors: &[Self], exponent: u128) -> Self {
        let mut acc = Self::ALMOST_ONE;
        for (i, factor) in factors.iter().enumerate() {
            let bit = Choice::from(((exponent >> i) & 1) as u8);
            acc = Self::conditional_select(&acc, &acc.mul(factor), bit);
        }
        acc
    }
}

fn ct_is_zero_u64(x: u64) -> Choice {
    // `x | −x` has its top bit set exactly when `x ≠ 0`.
    Choice::from((((x | x.wrapping_neg()) >> 63) ^ 1) as u8)
}

/// `a < b` for `u128`, from the borrow of `a − b`.
pub(crate) fn ct_lt_u128(a: u128, b: u128) -> Choice {
    Choice::from(u8::from(a.overflowing_sub(b).1))
}

/// `a < b` for `i128`.
pub(crate) fn ct_lt_i128(a: i128, b: i128) -> Choice {
    const BIAS: u128 = 1 << 127;
    ct_lt_u128((a as u128) ^ BIAS, (b as u128) ^ BIAS)
}

/// `a < b` for `i64`.
pub(crate) fn ct_lt_i64(a: i64, b: i64) -> Choice {
    const BIAS: u64 = 1 << 63;
    Choice::from(u8::from(
        ((a as u64) ^ BIAS).overflowing_sub((b as u64) ^ BIAS).1,
    ))
}

/// Floor division of a secret `dividend` by a public `divisor ∈ [1, 2^{126})`:
/// returns `(n, r)` with `dividend = n·divisor + r` and `r ∈ [0, divisor)`.
/// Restoring division with 128 fixed iterations; no hardware divide.
pub(crate) fn ct_floor_div(dividend: i128, divisor: u128) -> (i128, u128) {
    let negative = Choice::from((dividend as u128 >> 127) as u8);
    let magnitude = u128::conditional_select(
        &(dividend as u128),
        &(dividend.wrapping_neg() as u128),
        negative,
    );
    let mut quotient = 0u128;
    let mut remainder = 0u128;
    for i in (0..128).rev() {
        remainder = (remainder << 1) | ((magnitude >> i) & 1);
        let fits = !ct_lt_u128(remainder, divisor);
        remainder = u128::conditional_select(&remainder, &remainder.wrapping_sub(divisor), fits);
        quotient |= u128::from(fits.unwrap_u8()) << i;
    }
    let inexact = !subtle::ConstantTimeEq::ct_eq(&remainder, &0u128);
    let negative_quotient = (quotient as i128).wrapping_neg() - i128::from(inexact.unwrap_u8());
    let negative_remainder =
        u128::conditional_select(&0, &divisor.wrapping_sub(remainder), inexact);
    (
        i128::conditional_select(&(quotient as i128), &negative_quotient, negative),
        u128::conditional_select(&remainder, &negative_remainder, negative),
    )
}

/// Working precision of the public constant routines.
const WORK_BITS: u32 = 320;

fn work_one() -> BigUint {
    BigUint::from(1u8) << WORK_BITS
}

/// `⌊2^{W}·ln 2⌋` up to `W + 16` ulps, from `ln 2 = Σ_{k≥1} 1/(k·2^k)`.
fn ln2() -> BigUint {
    (1..=WORK_BITS + 16)
        .map(|k| (work_one() >> k) / BigUint::from(k))
        .sum()
}

/// `e^{−y}` for a scaled `y ∈ [0, 1)`, by Taylor series. Each term carries
/// at most two ulps of truncation, and the series stops at its first zero
/// term (about 70 terms at this precision).
fn exp_neg_small(y: &BigUint) -> BigUint {
    let mut term = work_one();
    let mut positive = term.clone();
    let mut negative = BigUint::default();
    let mut k = 1u32;
    while term != BigUint::default() {
        term = ((term * y) >> WORK_BITS) / BigUint::from(k);
        if k % 2 == 1 {
            negative += &term;
        } else {
            positive += &term;
        }
        k += 1;
    }
    positive - negative
}

/// `2^{−a}` for a scaled exponent `a ≥ 0`.
fn pow2_neg(a: &BigUint, ln2: &BigUint) -> BigUint {
    let integer = a >> WORK_BITS;
    let fraction = a - (&integer << WORK_BITS);
    match u32::try_from(&integer) {
        Ok(shift) if shift < WORK_BITS => exp_neg_small(&((fraction * ln2) >> WORK_BITS)) >> shift,
        _ => BigUint::default(),
    }
}

/// Floors a scaled value in `[0, 1]` to 192 bits; the value 1 maps to
/// [`Fix192::ALMOST_ONE`]. The working error (below `2^{−300}`) is first
/// subtracted, so the result never exceeds the exact value except on an
/// input that is itself a multiple of `2^{−192}` to within that error.
fn floor_fix192(scaled: &BigUint) -> Fix192 {
    let margin = BigUint::from(1u8) << (WORK_BITS - 300);
    let lowered = if scaled > &margin {
        scaled - margin
    } else {
        BigUint::default()
    };
    let floored = lowered >> (WORK_BITS - 192);
    if &floored >> 192u32 != BigUint::default() {
        return Fix192::ALMOST_ONE;
    }
    let mut digits = floored.to_u64_digits().into_iter();
    Fix192([
        digits.next().unwrap_or(0),
        digits.next().unwrap_or(0),
        digits.next().unwrap_or(0),
    ])
}

/// `[2^{−2^i/d} : i < len]`, each floored to 192 bits.
pub(crate) fn pow2_neg_table(denominator: u128, len: u32) -> Vec<Fix192> {
    let ln2 = ln2();
    let denominator = BigUint::from(denominator);
    (0..len)
        .map(|i| {
            let exponent = (BigUint::from(1u8) << (WORK_BITS + i)) / &denominator;
            floor_fix192(&pow2_neg(&exponent, &ln2))
        })
        .collect()
}

/// CDT thresholds `⌊2^{192}·Pr[X ≤ j]⌋`, `j < len`, for `Pr[X = x] ∝ 2^{−x²}`
/// on `x ≥ 0`. Terms with `x² ≥ W` vanish at working precision.
pub(crate) fn binary_gaussian_cdt(len: u32) -> Vec<Fix192> {
    let term = |x: u32| work_one() >> (x * x);
    let total: BigUint = (0..32).map(term).sum();
    let mut partial = BigUint::default();
    (0..len)
        .map(|j| {
            partial += term(j);
            floor_fix192(&((&partial << WORK_BITS) / &total))
        })
        .collect()
}

/// `M^{−1}` for `M = exp(t/α + 1/(2α²))`, `t = √(2·ε_bits·ln 2)`
/// (Whiteout Def. `wo:def:joint-rejection` with `ε = 2^{−ε_bits}`),
/// floored, so the rule runs with some `M' ≥ M`.
pub(crate) fn inverse_repetition_rate(alpha: u32, epsilon_bits: u32) -> Fix192 {
    let ln2 = ln2();
    let t = ((BigUint::from(2 * epsilon_bits) * &ln2) << WORK_BITS).sqrt();
    let exponent =
        t / BigUint::from(alpha) + work_one() / BigUint::from(2 * u64::from(alpha).pow(2));
    // e^{−e} = 2^{−e/ln 2}.
    floor_fix192(&pow2_neg(&((exponent << WORK_BITS) / &ln2), &ln2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn floor_division_matches_euclid() {
        let mut rng = ChaCha20Rng::seed_from_u64(6);
        for _ in 0..10_000 {
            let divisor = ((u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()))
                >> (2 + rng.next_u32() % 125))
                .max(1);
            let dividend = ((u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) >> 3)
                as i128
                * if rng.next_u32() & 1 == 1 { -1 } else { 1 };
            let expected = (
                dividend.div_euclid(divisor as i128),
                dividend.rem_euclid(divisor as i128) as u128,
            );
            assert_eq!(
                ct_floor_div(dividend, divisor),
                expected,
                "{dividend} / {divisor}"
            );
        }
    }

    #[test]
    fn shifts_match_integer_shifts() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let x = Fix192::random(&mut rng);
        let value = |f: &Fix192| {
            BigUint::from(f.0[2]) << 128u32 | BigUint::from(f.0[1]) << 64u32 | BigUint::from(f.0[0])
        };
        for shift in 0..=255u8 {
            assert_eq!(value(&x.ct_shr(shift)), value(&x) >> u32::from(shift));
        }
    }
}
