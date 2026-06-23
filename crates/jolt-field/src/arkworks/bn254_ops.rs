//! BN254 Fr field arithmetic operations.
//!
//! Low-level field arithmetic (Montgomery/Barrett reduction, scalar multiplication,
//! precomputed lookup tables).
use ark_bn254::FrConfig;
use ark_ff::{BigInt, Fp, MontConfig};
use num_traits::Zero;

type Fr = ark_bn254::Fr;

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

const N: usize = 4;

const MODULUS: [u64; N] = <FrConfig as MontConfig<N>>::MODULUS.0;
const INV: u64 = <FrConfig as MontConfig<N>>::INV;
const R: BigInt<N> = <FrConfig as MontConfig<N>>::R;

const MODULUS_HAS_SPARE_BIT: bool = MODULUS[N - 1] >> 63 == 0;
const MODULUS_NUM_SPARE_BITS: u32 = MODULUS[N - 1].leading_zeros();

/// 2*p as ([u64; 4], u64) — low N limbs and carry
const MODULUS_TIMES_2: ([u64; N], u64) = {
    let mut lo = [0u64; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let doubled = (MODULUS[i] as u128) * 2 + carry as u128;
        lo[i] = doubled as u64;
        carry = (doubled >> 64) as u64;
        i += 1;
    }
    (lo, carry)
};

/// 3*p as ([u64; 4], u64) — low N limbs and carry
const MODULUS_TIMES_3: ([u64; N], u64) = {
    let (m2_lo, m2_hi) = MODULUS_TIMES_2;
    let mut lo = [0u64; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let sum = (MODULUS[i] as u128) + (m2_lo[i] as u128) + (carry as u128);
        lo[i] = sum as u64;
        carry = (sum >> 64) as u64;
        i += 1;
    }
    (lo, m2_hi + carry)
};

/// Barrett mu = floor(2^(N*64 + 64 - spare_bits - 1) / MODULUS)
///
/// Computed via normalized Knuth long division. The quotient fits in a single u64.
const BARRETT_MU: u64 = {
    // Dividend = 2^(319 - spare_bits). For BN254 (spare_bits=2): 2^317
    // Represented as 5 limbs: [0, 0, 0, 0, 1 << (63 - spare_bits)]
    let shift = MODULUS_NUM_SPARE_BITS;

    // Normalize divisor: shift left by `shift` so MSB of top limb is set
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

    // Normalized dividend top two limbs: [1 << 63, 0]
    // (original top limb 1<<(63-shift), shifted left by shift → 1<<63)
    let dn4 = 1u64 << 63;

    // q_hat = floor((dn4 * 2^64) / p_hi)
    let dividend_top = (dn4 as u128) << 64;
    let mut q = dividend_top / (p_hi as u128);

    // Knuth refinement: while q * p_lo > remainder * 2^64, decrement q
    let mut r = dividend_top - q * (p_hi as u128);
    while r < (1u128 << 64) && q * (p_lo as u128) > (r << 64) {
        q -= 1;
        r += p_hi as u128;
    }

    q as u64
};

/// 16384-entry lookup table mapping small integers to their Montgomery form.
const PRECOMP_TABLE_SIZE: usize = 1 << 14;

/// `PRECOMP_TABLE[i]` = Montgomery form of `i` for BN254 Fr.
///
/// Uses `Fp::new()` which converts standard form → Montgomery form at compile time.
static PRECOMP_TABLE: [Fr; PRECOMP_TABLE_SIZE] = {
    let mut table: [Fr; PRECOMP_TABLE_SIZE] =
        [Fp::new_unchecked(BigInt([0u64; N])); PRECOMP_TABLE_SIZE];
    let mut i = 1usize;
    while i < PRECOMP_TABLE_SIZE {
        let mut limbs = [0u64; N];
        limbs[0] = i as u64;
        table[i] = Fp::new(BigInt::new(limbs));
        i += 1;
    }
    table
};

/// Pack (low_limb, [u64; N]) into BigInt<5>: low_limb at index 0, rest at 1..5
#[inline(always)]
fn nplus1_from_low_and_high(low: u64, high: [u64; N]) -> BigInt<5> {
    let mut limbs = [0u64; 5];
    limbs[0] = low;
    limbs[1] = high[0];
    limbs[2] = high[1];
    limbs[3] = high[2];
    limbs[4] = high[3];
    BigInt(limbs)
}

/// Pack ([u64; N], high_limb) into BigInt<5>: N limbs then high_limb
#[inline(always)]
fn nplus1_from_low_n_and_top(low_n: [u64; N], top: u64) -> BigInt<5> {
    let mut limbs = [0u64; 5];
    limbs[0] = low_n[0];
    limbs[1] = low_n[1];
    limbs[2] = low_n[2];
    limbs[3] = low_n[3];
    limbs[4] = top;
    BigInt(limbs)
}

/// Conditional subtraction for Barrett reduction: reduce a 5-limb intermediate
/// that is known to be < 4p down to < p (4 limbs).
#[inline(always)]
fn barrett_cond_subtract(r_tmp: BigInt<5>) -> BigInt<N> {
    let (m2_lo, _m2_hi) = MODULUS_TIMES_2;
    let (m3_lo, _m3_hi) = MODULUS_TIMES_3;

    // BN254 has MODULUS_NUM_SPARE_BITS = 2, so 2p and 3p both fit in N limbs.
    // This means r_tmp.0[4] == 0 for all branches below.

    let r_n: [u64; N] = [r_tmp.0[0], r_tmp.0[1], r_tmp.0[2], r_tmp.0[3]];

    if compare_4(r_n, m2_lo) != core::cmp::Ordering::Less {
        // r_tmp >= 2p
        if compare_4(r_n, m3_lo) != core::cmp::Ordering::Less {
            // r_tmp >= 3p → subtract 3p
            BigInt(sub_4(r_n, m3_lo))
        } else {
            // 2p <= r_tmp < 3p → subtract 2p
            BigInt(sub_4(r_n, m2_lo))
        }
    } else if compare_4(r_n, MODULUS) != core::cmp::Ordering::Less {
        // p <= r_tmp < 2p → subtract p
        BigInt(sub_4(r_n, MODULUS))
    } else {
        // r_tmp < p → no subtraction
        BigInt(r_n)
    }
}

/// Compare two 4-limb numbers (big-endian comparison)
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

/// Subtract two 4-limb numbers: a - b. Caller guarantees a >= b.
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

/// Barrett reduction kernel: reduce 5 limbs → 4 limbs (mod p).
///
/// Input `c` is a BigInt<5>. Computes `c mod p` via one Barrett estimate step.
#[inline(always)]
fn barrett_reduce_5_to_4(c: BigInt<5>) -> BigInt<N> {
    // Compute tilde_c = floor(c / R') where R' = 2^modulus_bits
    let tilde_c: u64 = if MODULUS_HAS_SPARE_BIT {
        let high = c.0[N];
        let second_high = c.0[N - 1];
        (high << MODULUS_NUM_SPARE_BITS) + (second_high >> (64 - MODULUS_NUM_SPARE_BITS))
    } else {
        c.0[N]
    };

    // Estimate m = floor(tilde_c * mu / 2^64)
    let m: u64 = ((tilde_c as u128 * BARRETT_MU as u128) >> 64) as u64;

    // Compute m * 2p (result fits in 5 limbs)
    let (m2p_lo, m2p_hi) = MODULUS_TIMES_2;
    let mut m2p = nplus1_from_low_n_and_top(m2p_lo, m2p_hi);
    // Multiply m2p by the scalar m in place
    mul_bigint5_by_u64_in_place(&mut m2p, m);

    // Compute r_tmp = c - m * 2p
    let mut r_tmp = c.0;
    let mut borrow = 0u64;
    for (r, &m) in r_tmp.iter_mut().zip(m2p.0.iter()) {
        borrow = sbb(r, m, borrow);
    }
    debug_assert!(borrow == 0, "Borrow in Barrett c - m*2p");

    barrett_cond_subtract(BigInt(r_tmp))
}

/// Multiply a BigInt<5> by a u64 scalar in place.
#[inline(always)]
fn mul_bigint5_by_u64_in_place(a: &mut BigInt<5>, b: u64) {
    let mut carry = 0u64;
    for limb in &mut a.0 {
        let prod = (*limb as u128) * (b as u128) + (carry as u128);
        *limb = prod as u64;
        carry = (prod >> 64) as u64;
    }
    // Overflow is discarded (caller ensures result fits in 5 limbs)
}

/// Perform N Montgomery reduction steps on a mutable buffer of L >= 2N limbs.
/// Returns carry from the final step.
#[inline(always)]
#[expect(clippy::needless_range_loop)]
fn montgomery_reduce_in_place<const L: usize>(limbs: &mut [u64; L]) -> u64 {
    debug_assert!(L >= 2 * N);
    let mut carry2 = 0u64;
    for i in 0..N {
        let tmp = limbs[i].wrapping_mul(INV);
        let mut carry = 0u64;
        // Discard low word: limbs[i] + tmp * MODULUS[0] → carry only
        let _ = mac_with_carry(limbs[i], tmp, MODULUS[0], &mut carry);
        for j in 1..N {
            let k = i + j;
            limbs[k] = mac_with_carry(limbs[k], tmp, MODULUS[j], &mut carry);
        }
        carry2 = adc(&mut limbs[i + N], carry, carry2);
    }
    carry2
}

/// Montgomery reduce an L-limb BigInt (L >= 2N) to a field element.
///
/// For L > 2N, first folds the tail (indices N..L) via Barrett, then runs
/// the standard N-step Montgomery REDC.
#[inline(always)]
pub(crate) fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> Fr {
    debug_assert!(L >= 2 * N, "montgomery_reduce requires L >= 2N");
    let mut buf = unreduced.0;

    // If L > 2N, fold excess high limbs down via Barrett
    if L > 2 * N {
        let mut acc = BigInt::<N>([0u64; N]);
        let mut i = L;
        while i > N {
            i -= 1;
            let c5 = nplus1_from_low_and_high(buf[i], acc.0);
            acc = barrett_reduce_5_to_4(c5);
        }
        buf[N..N + N].copy_from_slice(&acc.0);
        for slot in &mut buf[2 * N..L] {
            *slot = 0;
        }
    }

    let carry = montgomery_reduce_in_place(&mut buf);

    let mut result_limbs = [0u64; N];
    result_limbs.copy_from_slice(&buf[N..N + N]);
    let mut result = Fp::new_unchecked(BigInt::<N>(result_limbs));

    // Final conditional subtraction
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
    if b == 0 {
        return BigInt::<6>([0u64; 6]);
    }
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;

    let mut res = BigInt::<6>([0u64; 6]);

    // Pass 1: res += a * b_lo
    let mut carry = 0u64;
    for i in 0..N {
        res.0[i] = mac_with_carry(res.0[i], a.0[i], b_lo, &mut carry);
    }
    res.0[N] = carry;

    // Pass 2: res[1..] += a * b_hi
    let mut carry2 = 0u64;
    for i in 0..N {
        res.0[i + 1] = mac_with_carry(res.0[i + 1], a.0[i], b_hi, &mut carry2);
    }
    res.0[N + 1] = carry2;

    res
}

/// Barrett reduce BigInt<5> → Fr (N+1 → field element)
#[inline(always)]
fn from_unchecked_nplus1(element: BigInt<5>) -> Fr {
    let r = barrett_reduce_5_to_4(element);
    Fp::new_unchecked(r)
}

/// Barrett reduce BigInt<6> → Fr via two rounds
#[inline(always)]
fn from_unchecked_nplus2(element: BigInt<6>) -> Fr {
    // Round 1: reduce top 5 limbs (indices 1..6)
    let c1 = BigInt::<5>([
        element.0[1],
        element.0[2],
        element.0[3],
        element.0[4],
        element.0[5],
    ]);
    let r1 = barrett_reduce_5_to_4(c1);

    // Round 2: reduce [element[0], r1]
    let c2 = nplus1_from_low_and_high(element.0[0], r1.0);
    let r2 = barrett_reduce_5_to_4(c2);
    Fp::new_unchecked(r2)
}

/// Multiply a field element by u64.
#[inline(always)]
pub(crate) fn mul_u64(a: Fr, b: u64) -> Fr {
    if b == 0 || Zero::is_zero(&a) {
        return Fr::zero();
    }
    if b == 1 {
        return a;
    }
    let prod = bigint4_mul_u64(&a.0, b);
    from_unchecked_nplus1(prod)
}

/// Multiply a field element by i64.
#[inline(always)]
pub(crate) fn mul_i64(a: Fr, b: i64) -> Fr {
    let abs = b.unsigned_abs();
    let res = mul_u64(a, abs);
    if b < 0 {
        -res
    } else {
        res
    }
}

/// Multiply a field element by u128.
#[inline(always)]
pub(crate) fn mul_u128(a: Fr, b: u128) -> Fr {
    if b >> 64 == 0 {
        mul_u64(a, b as u64)
    } else {
        let prod = bigint4_mul_u128(&a.0, b);
        from_unchecked_nplus2(prod)
    }
}

/// Multiply a field element by i128.
#[inline(always)]
pub(crate) fn mul_i128(a: Fr, b: i128) -> Fr {
    if b == 0 || Zero::is_zero(&a) {
        return Fr::zero();
    }
    if b == 1 {
        return a;
    }
    let abs = b.unsigned_abs();
    let res = if abs <= u64::MAX as u128 {
        mul_u64(a, abs as u64)
    } else {
        let prod = bigint4_mul_u128(&a.0, abs);
        from_unchecked_nplus2(prod)
    };
    if b < 0 {
        -res
    } else {
        res
    }
}

/// Convert u64 → Fr using precomp table for small values, mul_u64(R, n) otherwise.
#[inline(always)]
pub(crate) fn from_u64(n: u64) -> Fr {
    if (n as usize) < PRECOMP_TABLE_SIZE {
        PRECOMP_TABLE[n as usize]
    } else {
        mul_u64(Fp::new_unchecked(R), n)
    }
}

/// Convert u128 → Fr using precomp table for small values, mul_u128(R, n) otherwise.
#[inline(always)]
pub(crate) fn from_u128(n: u128) -> Fr {
    if n < PRECOMP_TABLE_SIZE as u128 {
        PRECOMP_TABLE[n as usize]
    } else {
        mul_u128(Fp::new_unchecked(R), n)
    }
}

/// Wrap a raw BigInt<4> as Fr without any reduction (caller guarantees it's valid).
#[inline(always)]
pub(crate) fn from_bigint_unchecked(r: BigInt<N>) -> Fr {
    Fp::new_unchecked(r)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_ff::{PrimeField, UniformRand};
    use ark_std::test_rng;
    use rand::Rng;

    #[test]
    fn barrett_mu_sanity() {
        assert_ne!(BARRETT_MU, 0);
    }

    #[test]
    fn modulus_times_2_correct() {
        let (lo, hi) = MODULUS_TIMES_2;
        // Verify 2*MODULUS by manual doubling
        let mut expected = [0u64; N];
        let mut carry = 0u128;
        for i in 0..N {
            let doubled = (MODULUS[i] as u128) * 2 + carry;
            expected[i] = doubled as u64;
            carry = doubled >> 64;
        }
        assert_eq!(lo, expected);
        assert_eq!(hi, carry as u64);
    }

    #[test]
    fn modulus_times_3_correct() {
        let (lo, hi) = MODULUS_TIMES_3;
        // Verify 3*MODULUS by tripling
        let mut expected = [0u64; N];
        let mut carry = 0u128;
        for i in 0..N {
            let tripled = (MODULUS[i] as u128) * 3 + carry;
            expected[i] = tripled as u64;
            carry = tripled >> 64;
        }
        assert_eq!(lo, expected);
        assert_eq!(hi, carry as u64);
    }

    #[test]
    fn precomp_table_spot_check() {
        // PRECOMP_TABLE[i] should equal Montgomery form of i
        assert_eq!(PRECOMP_TABLE[0], Fr::from(0u64));
        assert_eq!(PRECOMP_TABLE[1], Fr::from(1u64));
        assert_eq!(PRECOMP_TABLE[42], Fr::from(42u64));
        assert_eq!(PRECOMP_TABLE[16383], Fr::from(16383u64));
    }

    #[test]
    fn from_u64_matches() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let val: u64 = rng.gen();
            let expected = Fr::from(val);
            let got = from_u64(val);
            assert_eq!(got, expected, "from_u64 mismatch for {}", val);
        }
        assert_eq!(from_u64(0), Fr::from(0u64));
        assert_eq!(from_u64(1), Fr::from(1u64));
        assert_eq!(from_u64(u64::MAX), Fr::from(u64::MAX));
    }

    #[test]
    fn from_u128_matches() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let val: u128 = ((rng.gen::<u64>() as u128) << 64) | (rng.gen::<u64>() as u128);
            let expected = {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                Fr::from_bigint(bigint).unwrap()
            };
            let got = from_u128(val);
            assert_eq!(got, expected, "from_u128 mismatch for {}", val);
        }
    }

    #[test]
    fn mul_u64_correct() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let a = Fr::rand(&mut rng);
            let b: u64 = rng.gen();
            let expected = a * Fr::from(b);
            let got = mul_u64(a, b);
            assert_eq!(got, expected, "mul_u64 mismatch: b={}", b);
        }
        // Edge cases
        let a = Fr::rand(&mut rng);
        assert_eq!(mul_u64(a, 0), Fr::zero());
        assert_eq!(mul_u64(a, 1), a);
    }

    #[test]
    fn mul_i64_correct() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let a = Fr::rand(&mut rng);
            let b: i64 = rng.gen();
            let expected = if b >= 0 {
                a * Fr::from(b as u64)
            } else {
                -(a * Fr::from((-b) as u64))
            };
            let got = mul_i64(a, b);
            assert_eq!(got, expected, "mul_i64 mismatch: b={}", b);
        }
    }

    #[test]
    fn mul_u128_correct() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let a = Fr::rand(&mut rng);
            let b: u128 = ((rng.gen::<u64>() as u128) << 64) | (rng.gen::<u64>() as u128);
            let b_fr = {
                let bigint = BigInt::new([b as u64, (b >> 64) as u64, 0, 0]);
                Fr::from_bigint(bigint).unwrap()
            };
            let expected = a * b_fr;
            let got = mul_u128(a, b);
            assert_eq!(got, expected, "mul_u128 mismatch");
        }
    }

    #[test]
    fn mul_i128_correct() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let a = Fr::rand(&mut rng);
            let b: i128 = rng.gen();
            let abs_b = b.unsigned_abs();
            let b_fr = {
                let bigint = BigInt::new([abs_b as u64, (abs_b >> 64) as u64, 0, 0]);
                Fr::from_bigint(bigint).unwrap()
            };
            let expected = if b >= 0 { a * b_fr } else { -(a * b_fr) };
            let got = mul_i128(a, b);
            assert_eq!(got, expected, "mul_i128 mismatch");
        }
    }

    #[test]
    fn montgomery_reduce_roundtrip() {
        let mut rng = test_rng();
        // Multiply the raw Montgomery-form BigInts: a_mont * b_mont = (aR)(bR).
        // Montgomery reduce divides by R → abR = Montgomery form of a*b.
        for _ in 0..200 {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let expected = a * b;

            // Access internal Montgomery representation directly
            let a_mont = (a.0).0;
            let b_mont = (b.0).0;
            let mut prod = BigInt::<8>::zero();
            for (i, &ai) in a_mont.iter().enumerate() {
                let mut carry = 0u64;
                for (j, &bj) in b_mont.iter().enumerate() {
                    prod.0[i + j] = mac_with_carry(prod.0[i + j], ai, bj, &mut carry);
                }
                prod.0[i + N] = carry;
            }
            let got = from_montgomery_reduce::<8>(prod);
            assert_eq!(got, expected, "Montgomery reduce roundtrip mismatch");
        }
    }
}
