//! BN254 Fr Montgomery/Barrett arithmetic kernel and the wide accumulator.
//!
//! Ported from jolt-field's `arkworks/bn254_ops.rs` + `wide_accumulator.rs`
//! with identical algorithms: Barrett folding for scalar multiplication,
//! a compile-time Montgomery table for small-integer conversion, and the
//! folded 4×4 product accumulator with deferred Montgomery reduction.

use crate::{signed::S256, Accumulator, Limbs};
use ark_bn254::FrConfig;
use ark_ff::{BigInt, Fp, MontConfig};
use num_traits::Zero;

use super::Fr;

type InnerFr = ark_bn254::Fr;

const N: usize = 4;
const MODULUS: [u64; N] = <FrConfig as MontConfig<N>>::MODULUS.0;
const INV: u64 = <FrConfig as MontConfig<N>>::INV;
const R: BigInt<N> = <FrConfig as MontConfig<N>>::R;

const MODULUS_HAS_SPARE_BIT: bool = MODULUS[N - 1] >> 63 == 0;
const MODULUS_NUM_SPARE_BITS: u32 = MODULUS[N - 1].leading_zeros();

/// a + b * c + carry → (result, new carry)
#[inline(always)]
fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128) * (c as u128) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// *a += b + carry → new carry
#[inline(always)]
fn adc(a: &mut u64, b: u64, carry: u64) -> u64 {
    let tmp = (*a as u128) + (b as u128) + (carry as u128);
    *a = tmp as u64;
    (tmp >> 64) as u64
}

/// *a -= b + borrow → new borrow (1 if underflow)
#[inline(always)]
fn sbb(a: &mut u64, b: u64, borrow: u64) -> u64 {
    let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
    *a = tmp as u64;
    u64::from(tmp >> 64 == 0)
}

/// `k * p` for small `k`, as (low N limbs, carry limb).
const fn modulus_times(k: u64) -> ([u64; N], u64) {
    let mut lo = [0u64; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let v = (MODULUS[i] as u128) * (k as u128) + carry as u128;
        lo[i] = v as u64;
        carry = (v >> 64) as u64;
        i += 1;
    }
    (lo, carry)
}

const MODULUS_TIMES_2: ([u64; N], u64) = modulus_times(2);
const MODULUS_TIMES_3: ([u64; N], u64) = modulus_times(3);

/// Barrett mu = floor(2^(N*64 + 64 - spare_bits - 1) / MODULUS), computed via
/// normalized Knuth long division. The quotient fits in a single u64.
const BARRETT_MU: u64 = {
    let shift = MODULUS_NUM_SPARE_BITS;
    let p_hi = if shift > 0 {
        (MODULUS[3] << shift) | (MODULUS[2] >> (64 - shift))
    } else {
        MODULUS[3]
    };
    let p_lo = if shift > 0 {
        (MODULUS[2] << shift) | (MODULUS[1] >> (64 - shift))
    } else {
        MODULUS[2]
    };
    // Normalized dividend top limbs are [1 << 63, 0].
    let dividend_top = (1u128 << 63) << 64;
    let mut q = dividend_top / (p_hi as u128);
    let mut r = dividend_top - q * (p_hi as u128);
    while r < (1u128 << 64) && q * (p_lo as u128) > (r << 64) {
        q -= 1;
        r += p_hi as u128;
    }
    q as u64
};

/// `PRECOMP_TABLE[i]` = Montgomery form of `i`, for fast small-int conversion.
const PRECOMP_TABLE_SIZE: usize = 1 << 14;
static PRECOMP_TABLE: [InnerFr; PRECOMP_TABLE_SIZE] = {
    let mut table = [Fp::new_unchecked(BigInt([0u64; N])); PRECOMP_TABLE_SIZE];
    let mut i = 1usize;
    while i < PRECOMP_TABLE_SIZE {
        let mut limbs = [0u64; N];
        limbs[0] = i as u64;
        table[i] = Fp::new(BigInt::new(limbs));
        i += 1;
    }
    table
};

/// Compare two 4-limb numbers.
#[inline(always)]
fn compare_4(a: [u64; N], b: [u64; N]) -> core::cmp::Ordering {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return if a[i] > b[i] {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Less
            };
        }
    }
    core::cmp::Ordering::Equal
}

/// a - b for 4-limb numbers. Caller guarantees a >= b.
#[inline(always)]
fn sub_4(a: [u64; N], b: [u64; N]) -> [u64; N] {
    let mut result = a;
    let mut borrow = 0u64;
    borrow = sbb(&mut result[0], b[0], borrow);
    borrow = sbb(&mut result[1], b[1], borrow);
    borrow = sbb(&mut result[2], b[2], borrow);
    let _ = sbb(&mut result[3], b[3], borrow);
    result
}

/// Reduce a 5-limb Barrett intermediate known to be < 4p down to < p.
///
/// BN254 has two spare bits, so 2p and 3p fit in N limbs and the top
/// intermediate limb is always zero here.
#[inline(always)]
fn barrett_cond_subtract(r_tmp: BigInt<5>) -> BigInt<N> {
    let r_n: [u64; N] = [r_tmp.0[0], r_tmp.0[1], r_tmp.0[2], r_tmp.0[3]];
    if compare_4(r_n, MODULUS_TIMES_2.0) != core::cmp::Ordering::Less {
        if compare_4(r_n, MODULUS_TIMES_3.0) != core::cmp::Ordering::Less {
            BigInt(sub_4(r_n, MODULUS_TIMES_3.0))
        } else {
            BigInt(sub_4(r_n, MODULUS_TIMES_2.0))
        }
    } else if compare_4(r_n, MODULUS) != core::cmp::Ordering::Less {
        BigInt(sub_4(r_n, MODULUS))
    } else {
        BigInt(r_n)
    }
}

/// Barrett reduction kernel: reduce 5 limbs → 4 limbs (mod p).
#[inline(always)]
fn barrett_reduce_5_to_4(c: BigInt<5>) -> BigInt<N> {
    let tilde_c: u64 = if MODULUS_HAS_SPARE_BIT {
        (c.0[N] << MODULUS_NUM_SPARE_BITS) + (c.0[N - 1] >> (64 - MODULUS_NUM_SPARE_BITS))
    } else {
        c.0[N]
    };
    let m: u64 = ((tilde_c as u128 * BARRETT_MU as u128) >> 64) as u64;

    // r_tmp = c - m * 2p
    let (m2p_lo, m2p_hi) = MODULUS_TIMES_2;
    let mut m2p = BigInt([m2p_lo[0], m2p_lo[1], m2p_lo[2], m2p_lo[3], m2p_hi]);
    let mut carry = 0u64;
    for limb in &mut m2p.0 {
        let prod = (*limb as u128) * (m as u128) + (carry as u128);
        *limb = prod as u64;
        carry = (prod >> 64) as u64;
    }
    let mut r_tmp = c.0;
    let mut borrow = 0u64;
    for (r, &sub) in r_tmp.iter_mut().zip(m2p.0.iter()) {
        borrow = sbb(r, sub, borrow);
    }
    debug_assert!(borrow == 0, "borrow in Barrett c - m*2p");

    barrett_cond_subtract(BigInt(r_tmp))
}

/// N Montgomery reduction steps on a buffer of L >= 2N limbs; returns the
/// final carry.
#[inline(always)]
fn montgomery_reduce_in_place<const L: usize>(limbs: &mut [u64; L]) -> u64 {
    let mut carry2 = 0u64;
    for i in 0..N {
        let tmp = limbs[i].wrapping_mul(INV);
        let mut carry = 0u64;
        let _ = mac_with_carry(limbs[i], tmp, MODULUS[0], &mut carry);
        for j in 1..N {
            limbs[i + j] = mac_with_carry(limbs[i + j], tmp, MODULUS[j], &mut carry);
        }
        carry2 = adc(&mut limbs[i + N], carry, carry2);
    }
    carry2
}

/// Montgomery reduce an L-limb integer (L >= 2N) to a field element.
///
/// For L > 2N the tail is first folded down via Barrett, then the standard
/// N-step Montgomery REDC runs.
#[inline(always)]
pub(crate) fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> InnerFr {
    debug_assert!(L >= 2 * N, "montgomery_reduce requires L >= 2N");
    let mut buf = unreduced.0;

    if L > 2 * N {
        let mut acc = [0u64; N];
        let mut i = L;
        while i > N {
            i -= 1;
            let c5 = BigInt([buf[i], acc[0], acc[1], acc[2], acc[3]]);
            acc = barrett_reduce_5_to_4(c5).0;
        }
        buf[N..2 * N].copy_from_slice(&acc);
        for slot in &mut buf[2 * N..L] {
            *slot = 0;
        }
    }

    let carry = montgomery_reduce_in_place(&mut buf);

    let mut result_limbs = [0u64; N];
    result_limbs.copy_from_slice(&buf[N..2 * N]);
    let mut result = Fp::new_unchecked(BigInt::<N>(result_limbs));

    let needs_sub = if MODULUS_HAS_SPARE_BIT {
        compare_4(result.0 .0, MODULUS) != core::cmp::Ordering::Less
    } else {
        carry != 0 || compare_4(result.0 .0, MODULUS) != core::cmp::Ordering::Less
    };
    if needs_sub {
        result.0 = BigInt(sub_4(result.0 .0, MODULUS));
    }
    result
}

/// Multiply BigInt<4> by u64, producing BigInt<5>.
#[inline(always)]
fn bigint4_mul_u64(a: &BigInt<N>, b: u64) -> BigInt<5> {
    let mut res = BigInt::<5>([0u64; 5]);
    let mut carry = 0u64;
    for i in 0..N {
        res.0[i] = mac_with_carry(0, a.0[i], b, &mut carry);
    }
    res.0[N] = carry;
    res
}

/// Multiply BigInt<4> by u128, producing BigInt<6>.
#[inline(always)]
fn bigint4_mul_u128(a: &BigInt<N>, b: u128) -> BigInt<6> {
    let (b_lo, b_hi) = (b as u64, (b >> 64) as u64);
    let mut res = BigInt::<6>([0u64; 6]);
    let mut carry = 0u64;
    for i in 0..N {
        res.0[i] = mac_with_carry(res.0[i], a.0[i], b_lo, &mut carry);
    }
    res.0[N] = carry;
    let mut carry2 = 0u64;
    for i in 0..N {
        res.0[i + 1] = mac_with_carry(res.0[i + 1], a.0[i], b_hi, &mut carry2);
    }
    res.0[N + 1] = carry2;
    res
}

/// Barrett reduce BigInt<6> → Fr via two rounds.
#[inline(always)]
fn from_unchecked_nplus2(element: BigInt<6>) -> InnerFr {
    let c1 = BigInt::<5>([
        element.0[1],
        element.0[2],
        element.0[3],
        element.0[4],
        element.0[5],
    ]);
    let r1 = barrett_reduce_5_to_4(c1);
    let c2 = BigInt([element.0[0], r1.0[0], r1.0[1], r1.0[2], r1.0[3]]);
    Fp::new_unchecked(barrett_reduce_5_to_4(c2))
}

/// Multiply a field element by u64 via one Barrett round.
#[inline(always)]
pub(crate) fn mul_u64(a: InnerFr, b: u64) -> InnerFr {
    if b == 0 || Zero::is_zero(&a) {
        return InnerFr::zero();
    }
    if b == 1 {
        return a;
    }
    Fp::new_unchecked(barrett_reduce_5_to_4(bigint4_mul_u64(&a.0, b)))
}

/// Multiply a field element by u128 via up to two Barrett rounds.
#[inline(always)]
pub(crate) fn mul_u128(a: InnerFr, b: u128) -> InnerFr {
    if b >> 64 == 0 {
        mul_u64(a, b as u64)
    } else {
        from_unchecked_nplus2(bigint4_mul_u128(&a.0, b))
    }
}

/// Convert u64 → Fr: table lookup for small values, `mul_u64(R, n)` otherwise.
#[inline(always)]
pub(crate) fn from_u64(n: u64) -> InnerFr {
    if n < PRECOMP_TABLE_SIZE as u64 {
        PRECOMP_TABLE[n as usize]
    } else {
        mul_u64(Fp::new_unchecked(R), n)
    }
}

/// Convert u128 → Fr: table lookup for small values, `mul_u128(R, n)` otherwise.
#[inline(always)]
pub(crate) fn from_u128(n: u128) -> InnerFr {
    if n < PRECOMP_TABLE_SIZE as u128 {
        PRECOMP_TABLE[n as usize]
    } else {
        mul_u128(Fp::new_unchecked(R), n)
    }
}

/// Folded 4×4 product accumulator for BN254 Fr deferred reduction.
///
/// Stores the running sum of Montgomery-form products in positional `u128`
/// slots; each fmadd defers all carry propagation, and
/// [`Accumulator::reduce`] performs one carry pass plus one Montgomery
/// reduction. The `u128` slots give ~2^63 fmadds of headroom.
#[derive(Clone, Copy)]
pub struct WideAccumulator {
    slots: [u128; 8],
}

/// BN254 Fr accumulator for signed small-scalar products.
///
/// Positive and negative terms are held separately as unreduced five-limb
/// integers and reduced once at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrSmallScalarAccumulator {
    pos: Limbs<5>,
    neg: Limbs<5>,
}

impl Default for FrSmallScalarAccumulator {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: Limbs::zero(),
            neg: Limbs::zero(),
        }
    }
}

impl FrSmallScalarAccumulator {
    #[inline(always)]
    fn add_to(slots: &mut Limbs<5>, value: Fr) {
        slots.add_assign_trunc(&Limbs(value.inner_limbs()));
    }

    #[inline(always)]
    fn fmadd_magnitude(slots: &mut Limbs<5>, value: Fr, scalar: u64) {
        if scalar == 0 {
            return;
        }
        if scalar == 1 {
            Self::add_to(slots, value);
            return;
        }
        slots.add_assign_trunc(&Limbs(bigint4_mul_u64(&value.0 .0, scalar).0));
    }
}

impl Accumulator for FrSmallScalarAccumulator {
    type Element = Fr;

    #[inline(always)]
    fn add(&mut self, value: Fr) {
        Self::add_to(&mut self.pos, value);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.pos.add_assign_trunc(&other.pos);
        self.neg.add_assign_trunc(&other.neg);
    }

    #[inline(always)]
    fn reduce(self) -> Fr {
        if self.pos >= self.neg {
            Fr(Fp::new_unchecked(barrett_reduce_5_to_4(BigInt(
                self.pos.sub_trunc::<5, 5>(&self.neg).0,
            ))))
        } else {
            -Fr(Fp::new_unchecked(barrett_reduce_5_to_4(BigInt(
                self.neg.sub_trunc::<5, 5>(&self.pos).0,
            ))))
        }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        self.add(a * b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, value: Fr, scalar: u64) {
        Self::fmadd_magnitude(&mut self.pos, value, scalar);
    }

    #[inline(always)]
    fn fmadd_i64(&mut self, value: Fr, scalar: i64) {
        let magnitude = scalar.unsigned_abs();
        if scalar >= 0 {
            Self::fmadd_magnitude(&mut self.pos, value, magnitude);
        } else {
            Self::fmadd_magnitude(&mut self.neg, value, magnitude);
        }
    }
}

/// BN254 Fr accumulator for field elements multiplied by signed 256-bit
/// integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrSignedProductAccumulator {
    pos: [u128; 8],
    neg: [u128; 8],
}

impl Default for FrSignedProductAccumulator {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: [0; 8],
            neg: [0; 8],
        }
    }
}

impl FrSignedProductAccumulator {
    #[inline(always)]
    fn fmadd_magnitude(slots: &mut [u128; 8], value: Fr, magnitude: Limbs<4>) {
        for (i, value_limb) in value.inner_limbs().into_iter().enumerate() {
            for (j, magnitude_limb) in magnitude.0.into_iter().enumerate() {
                let product = (value_limb as u128) * (magnitude_limb as u128);
                slots[i + j] += (product as u64) as u128;
                slots[i + j + 1] += ((product >> 64) as u64) as u128;
            }
        }
    }

    #[inline]
    fn normalize(slots: [u128; 8]) -> BigInt<9> {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (index, slot) in slots.into_iter().enumerate() {
            let (sum, overflow) = slot.overflowing_add(carry);
            out[index] = sum as u64;
            carry = (sum >> 64) + ((overflow as u128) << 64);
        }
        out[8] = carry as u64;
        BigInt(out)
    }

    #[inline(always)]
    fn fmadd_unsigned(&mut self, value: Fr, scalar: u64) {
        Self::fmadd_magnitude(&mut self.pos, value, Limbs::from_u64(scalar));
    }
}

impl Accumulator for FrSignedProductAccumulator {
    type Element = Fr;

    #[inline(always)]
    fn add(&mut self, value: Fr) {
        self.fmadd_unsigned(value, 1);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        for (lhs, rhs) in self.pos.iter_mut().zip(other.pos) {
            *lhs += rhs;
        }
        for (lhs, rhs) in self.neg.iter_mut().zip(other.neg) {
            *lhs += rhs;
        }
    }

    #[inline]
    fn reduce(self) -> Fr {
        let pos = Self::normalize(self.pos);
        let neg = Self::normalize(self.neg);
        let correction = Fr(Fp::new_unchecked(<FrConfig as MontConfig<4>>::R2));
        let reduced = if pos >= neg {
            Fr(from_montgomery_reduce(BigInt(
                Limbs(pos.0).sub_trunc::<9, 9>(&Limbs(neg.0)).0,
            )))
        } else {
            -Fr(from_montgomery_reduce(BigInt(
                Limbs(neg.0).sub_trunc::<9, 9>(&Limbs(pos.0)).0,
            )))
        };
        reduced * correction
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        self.add(a * b);
    }

    #[inline(always)]
    fn fmadd_u64(&mut self, value: Fr, scalar: u64) {
        self.fmadd_unsigned(value, scalar);
    }

    #[inline(always)]
    fn fmadd_i64(&mut self, value: Fr, scalar: i64) {
        let magnitude = Limbs::from_u64(scalar.unsigned_abs());
        if scalar >= 0 {
            Self::fmadd_magnitude(&mut self.pos, value, magnitude);
        } else {
            Self::fmadd_magnitude(&mut self.neg, value, magnitude);
        }
    }

    #[inline(always)]
    fn fmadd_signed_u64(&mut self, value: Fr, magnitude: u64, is_positive: bool) {
        let magnitude = Limbs::from_u64(magnitude);
        if is_positive {
            Self::fmadd_magnitude(&mut self.pos, value, magnitude);
        } else {
            Self::fmadd_magnitude(&mut self.neg, value, magnitude);
        }
    }

    #[inline(always)]
    fn fmadd_s256(&mut self, value: Fr, scalar: &S256) {
        if scalar.magnitude.is_zero() {
            return;
        }
        if scalar.is_positive {
            Self::fmadd_magnitude(&mut self.pos, value, scalar.magnitude);
        } else {
            Self::fmadd_magnitude(&mut self.neg, value, scalar.magnitude);
        }
    }
}

impl Default for WideAccumulator {
    #[inline]
    fn default() -> Self {
        Self { slots: [0; 8] }
    }
}

impl WideAccumulator {
    /// Carry-propagate the positional slots into a 9-limb integer.
    #[inline]
    fn normalize(self) -> BigInt<9> {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (index, slot) in self.slots.into_iter().enumerate() {
            let (sum, overflow) = slot.overflowing_add(carry);
            out[index] = sum as u64;
            carry = (sum >> 64) + ((overflow as u128) << 64);
        }
        out[8] = carry as u64;
        BigInt::new(out)
    }
}

impl Accumulator for WideAccumulator {
    type Element = Fr;

    /// Adds `value` in four limb additions instead of a full 4×4 `fmadd` by
    /// `one()`. The slots hold products of Montgomery forms, so a plain
    /// element must enter as `value * R`. Since `2^256 = R (mod p)`, placing
    /// the element limbs at positions 4 through 8 contributes exactly that
    /// value, and [`Accumulator::reduce`] folds the high limbs modulo `p`.
    #[inline(always)]
    fn add(&mut self, value: Fr) {
        for (slot, limb) in self.slots[4..].iter_mut().zip(value.inner_limbs()) {
            *slot += limb as u128;
        }
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        for (lhs, rhs) in self.slots.iter_mut().zip(other.slots) {
            *lhs += rhs;
        }
    }

    fn reduce(self) -> Fr {
        Fr(from_montgomery_reduce(self.normalize()))
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        let (a, b) = (a.inner_limbs(), b.inner_limbs());
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let product = (ai as u128) * (bj as u128);
                self.slots[i + j] += (product as u64) as u128;
                self.slots[i + j + 1] += ((product >> 64) as u64) as u128;
            }
        }
    }

    /// One Barrett round beats the default's `from_u64` conversion plus a
    /// full 4×4 limb product.
    #[inline(always)]
    fn fmadd_u64(&mut self, value: Fr, scalar: u64) {
        self.add(Fr(mul_u64(value.0, scalar)));
    }

    #[inline(always)]
    fn fmadd_u128(&mut self, value: Fr, scalar: u128) {
        self.add(Fr(mul_u128(value.0, scalar)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ring;
    use ark_ff::UniformRand;
    use num_traits::One;
    use rand::{Rng, SeedableRng};

    fn spread(seed: u64) -> Fr {
        let a = Fr::from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let b = Fr::from_u64(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1);
        a * b * b + a
    }

    fn s256_to_fr(value: &S256) -> Fr {
        let mut magnitude = Fr::zero();
        for limb in value.magnitude_limbs().into_iter().rev() {
            magnitude = magnitude.mul_pow_2(64) + Fr::from_u64(limb);
        }
        if value.is_positive {
            magnitude
        } else {
            -magnitude
        }
    }

    #[test]
    fn accumulator_add_matches_fmadd_by_one() {
        for seed in 0..100u64 {
            let value = spread(seed);
            let a = spread(seed + 1000);
            let b = spread(seed + 2000);

            let mut via_add = WideAccumulator::default();
            via_add.fmadd(a, b);
            via_add.add(value);

            let mut via_fmadd = WideAccumulator::default();
            via_fmadd.fmadd(a, b);
            via_fmadd.fmadd(value, <Fr as One>::one());

            assert_eq!(via_add.reduce(), via_fmadd.reduce());
            assert_eq!(via_add.reduce(), a * b + value);
        }
    }

    #[test]
    fn accumulator_repeated_add_reduces_exactly() {
        let mut acc = WideAccumulator::default();
        let mut expected = Fr::from_u64(0);
        for seed in 0..1000u64 {
            let value = spread(seed);
            acc.add(value);
            expected += value;
        }
        assert_eq!(acc.reduce(), expected);
    }

    #[test]
    fn small_scalar_accumulator_reduces_signed_terms() {
        let mut left = FrSmallScalarAccumulator::default();
        left.fmadd_u64(Fr::from_u64(3), 16);
        left.fmadd_i64(Fr::from_u64(5), -7);

        let mut right = FrSmallScalarAccumulator::default();
        right.add(Fr::from_u64(11));
        right.fmadd_i64(Fr::from_u64(9), -13);
        right.fmadd_u64(Fr::from_u64(2), 7);

        left.merge(right);
        assert_eq!(left.reduce(), -Fr::from_u64(79));
    }

    #[test]
    fn signed_product_accumulator_reduces_and_merges() {
        let terms = [
            (Fr::from_u64(3), S256::from_i128(17)),
            (Fr::from_u64(11), S256::from_i128(-9)),
            (Fr::from_u64(42), S256::new([7, 5, 3, 1], true)),
            (Fr::from_u64(6), S256::new([u64::MAX, 19, 0, 0], false)),
        ];

        let mut left = FrSignedProductAccumulator::default();
        let mut right = FrSignedProductAccumulator::default();
        let mut expected = Fr::zero();
        for (index, (field, scalar)) in terms.into_iter().enumerate() {
            if index % 2 == 0 {
                left.fmadd_s256(field, &scalar);
            } else {
                right.fmadd_s256(field, &scalar);
            }
            expected += field * s256_to_fr(&scalar);
        }

        left.merge(right);
        assert_eq!(left.reduce(), expected);
    }

    #[test]
    fn kernel_matches_arkworks() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(7);
        for _ in 0..500 {
            let a = InnerFr::rand(&mut rng);
            let b: u64 = rng.gen();
            let c: u128 = rng.gen();
            assert_eq!(mul_u64(a, b), a * InnerFr::from(b));
            assert_eq!(mul_u128(a, c), a * InnerFr::from(c));
            assert_eq!(from_u64(b), InnerFr::from(b));
            assert_eq!(from_u128(c), InnerFr::from(c));
        }
        let boundary = PRECOMP_TABLE_SIZE as u64;
        assert_eq!(from_u64(boundary - 1), InnerFr::from(boundary - 1));
        assert_eq!(from_u64(boundary), InnerFr::from(boundary));
    }

    #[test]
    fn montgomery_reduce_roundtrip() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(8);
        for _ in 0..200 {
            let a = InnerFr::rand(&mut rng);
            let b = InnerFr::rand(&mut rng);
            let mut prod = BigInt::<8>([0u64; 8]);
            for (i, &ai) in a.0 .0.iter().enumerate() {
                let mut carry = 0u64;
                for (j, &bj) in b.0 .0.iter().enumerate() {
                    prod.0[i + j] = mac_with_carry(prod.0[i + j], ai, bj, &mut carry);
                }
                prod.0[i + 4] = carry;
            }
            assert_eq!(from_montgomery_reduce::<8>(prod), a * b);
        }
    }
}
