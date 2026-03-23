//! RNS-Montgomery arithmetic for BN254 Fr on Metal GPU.
//!
//! Uses the Bajard/VROOM approach: elements are stored in Montgomery form
//! `ã = a × M mod r` decomposed into two independent RNS bases (B and B'),
//! each with 9 pseudo-Mersenne primes (p = 2^31 - c).
//!
//! # Architecture
//!
//! - **Base B** (primary): 9 primes, product M = Π m_i > r.
//!   Montgomery constant R = M. Elements in B represent `ã mod m_i`.
//! - **Base B'** (secondary): 9 primes, product M' = Π m'_j > r.
//!   Used for accumulation — sumcheck accumulators live only in B'.
//!
//! # RNS-Montgomery multiplication
//!
//! Given ã = a·M mod r and b̃ = b·M mod r (both in B ∪ B'):
//! 1. Component-wise product: t_i = ã_i · b̃_i mod m_i (in B), same in B'
//! 2. Compute q = -t · r⁻¹ mod B (component-wise in B)
//! 3. Base-extend q from B → B' via exact Garner CRT
//! 4. Combine: s'_j = (t'_j + q'_j · r_j) · M⁻¹_j mod m'_j (in B')
//!
//! Result s̃ = (a·b)·M mod r in B'. All operations are 32-bit.
//!
//! # Register pressure
//!
//! Per thread with D=4 eval points: ~108 u32 → 2.3 simdgroups occupancy
//! (vs current CIOS: 280-500 u32 → 0.5-0.9 simdgroups).

use jolt_field::Fr;

/// Primes per basis. 9 primes × 31 bits = 279-bit range per basis (> 254-bit r).
pub const BASIS_SIZE: usize = 9;

/// Total primes across both bases.
pub const NUM_PRIMES: usize = 2 * BASIS_SIZE;

/// Primary basis B: first 9 pseudo-Mersenne primes.
/// Product M = Π m_i serves as Montgomery constant R.
pub const PRIMARY_PRIMES: [u32; BASIS_SIZE] = [
    2_147_483_647, // 2^31 - 1   (M31)
    2_147_483_629, // 2^31 - 19
    2_147_483_587, // 2^31 - 61
    2_147_483_579, // 2^31 - 69
    2_147_483_563, // 2^31 - 85
    2_147_483_549, // 2^31 - 99
    2_147_483_543, // 2^31 - 105
    2_147_483_497, // 2^31 - 151
    2_147_483_489, // 2^31 - 159
];

/// Secondary basis B': last 9 pseudo-Mersenne primes.
/// Accumulators live in B' only during sumcheck inner loop.
pub const SECONDARY_PRIMES: [u32; BASIS_SIZE] = [
    2_147_483_477, // 2^31 - 171
    2_147_483_423, // 2^31 - 225
    2_147_483_399, // 2^31 - 249
    2_147_483_353, // 2^31 - 295
    2_147_483_323, // 2^31 - 325
    2_147_483_269, // 2^31 - 379
    2_147_483_249, // 2^31 - 399
    2_147_483_237, // 2^31 - 411
    2_147_483_179, // 2^31 - 469
];

/// All 18 primes: B then B'.
pub const PRIMES: [u32; NUM_PRIMES] = {
    let mut all = [0u32; NUM_PRIMES];
    let mut i = 0;
    while i < BASIS_SIZE {
        all[i] = PRIMARY_PRIMES[i];
        all[BASIS_SIZE + i] = SECONDARY_PRIMES[i];
        i += 1;
    }
    all
};

/// Pseudo-Mersenne offsets: c_i where p_i = 2^31 - c_i.
pub const C_VALUES: [u32; NUM_PRIMES] = {
    let mut c = [0u32; NUM_PRIMES];
    let mut i = 0;
    while i < NUM_PRIMES {
        c[i] = (1u32 << 31) - PRIMES[i];
        i += 1;
    }
    c
};

pub const PRIMARY_C: [u32; BASIS_SIZE] = {
    let mut c = [0u32; BASIS_SIZE];
    let mut i = 0;
    while i < BASIS_SIZE {
        c[i] = (1u32 << 31) - PRIMARY_PRIMES[i];
        i += 1;
    }
    c
};

pub const SECONDARY_C: [u32; BASIS_SIZE] = {
    let mut c = [0u32; BASIS_SIZE];
    let mut i = 0;
    while i < BASIS_SIZE {
        c[i] = (1u32 << 31) - SECONDARY_PRIMES[i];
        i += 1;
    }
    c
};

/// RNS-Montgomery element: residues in both bases.
/// `[0..9]` = primary basis B, `[9..18]` = secondary basis B'.
pub type RnsMontElement = [u32; NUM_PRIMES];

/// Residues in a single basis.
pub type BasisResidue = [u32; BASIS_SIZE];

// ---------------------------------------------------------------------------
// Big-integer arithmetic (u64 limbs, LE)
// ---------------------------------------------------------------------------

/// Number of u64 limbs for big-integer operations.
/// 10 limbs = 640 bits, enough for products of 279-bit values.
const BIG_LIMBS: usize = 10;

type BigInt = [u64; BIG_LIMBS];

const BIG_ZERO: BigInt = [0u64; BIG_LIMBS];

fn big_from_u32_le(limbs: &[u32; 8]) -> BigInt {
    let mut r = BIG_ZERO;
    for i in 0..4 {
        r[i] = (limbs[2 * i] as u64) | ((limbs[2 * i + 1] as u64) << 32);
    }
    r
}

fn big_from_bytes_le(bytes: &[u8; 32]) -> BigInt {
    let mut r = BIG_ZERO;
    for i in 0..4 {
        r[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    r
}

fn big_to_bytes_le(a: &BigInt) -> [u8; 80] {
    let mut bytes = [0u8; 80];
    for (i, &limb) in a.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

fn big_add(a: &BigInt, b: &BigInt) -> BigInt {
    let mut r = BIG_ZERO;
    let mut carry = 0u64;
    for i in 0..BIG_LIMBS {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        r[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
    r
}

/// a - b, assuming a >= b.
fn big_sub(a: &BigInt, b: &BigInt) -> BigInt {
    let mut r = BIG_ZERO;
    let mut borrow = 0i64;
    for i in 0..BIG_LIMBS {
        let diff = (a[i] as i128) - (b[i] as i128) - (borrow as i128);
        r[i] = diff as u64;
        borrow = i64::from(diff < 0);
    }
    r
}

fn big_cmp(a: &BigInt, b: &BigInt) -> std::cmp::Ordering {
    for i in (0..BIG_LIMBS).rev() {
        let ord = a[i].cmp(&b[i]);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

fn big_mul(a: &BigInt, b: &BigInt) -> BigInt {
    let mut out = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..BIG_LIMBS {
            if i + j >= BIG_LIMBS {
                break;
            }
            let prod = (a[i] as u128) * (b[j] as u128) + (out[i + j] as u128) + carry;
            out[i + j] = prod as u64;
            carry = prod >> 64;
        }
    }
    out
}

/// a mod m, where m fits in a BigInt. Uses repeated subtraction with shifts for simplicity.
/// Only used in precomputation, not on hot path.
fn big_mod(a: &BigInt, m: &BigInt) -> BigInt {
    if big_cmp(a, m) == std::cmp::Ordering::Less {
        return *a;
    }

    // Find highest bit of a
    let a_bits = big_bit_len(a);
    let m_bits = big_bit_len(m);
    assert!(m_bits > 0, "division by zero");

    let mut remainder = *a;
    if a_bits <= m_bits {
        if big_cmp(&remainder, m) != std::cmp::Ordering::Less {
            remainder = big_sub(&remainder, m);
        }
        return remainder;
    }

    let shift = a_bits - m_bits;
    let mut shifted_m = big_shl(m, shift);

    for _ in 0..=shift {
        if big_cmp(&remainder, &shifted_m) != std::cmp::Ordering::Less {
            remainder = big_sub(&remainder, &shifted_m);
        }
        shifted_m = big_shr1(&shifted_m);
    }
    remainder
}

fn big_bit_len(a: &BigInt) -> usize {
    for i in (0..BIG_LIMBS).rev() {
        if a[i] != 0 {
            return i * 64 + (64 - a[i].leading_zeros() as usize);
        }
    }
    0
}

fn big_shl(a: &BigInt, shift: usize) -> BigInt {
    if shift == 0 {
        return *a;
    }
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut r = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        if i + word_shift >= BIG_LIMBS {
            break;
        }
        r[i + word_shift] |= if bit_shift == 0 {
            a[i]
        } else {
            a[i] << bit_shift
        };
        if bit_shift > 0 && i + word_shift + 1 < BIG_LIMBS {
            r[i + word_shift + 1] |= a[i] >> (64 - bit_shift);
        }
    }
    r
}

fn big_shr1(a: &BigInt) -> BigInt {
    let mut r = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        r[i] = a[i] >> 1;
        if i + 1 < BIG_LIMBS {
            r[i] |= a[i + 1] << 63;
        }
    }
    r
}

fn big_is_zero(a: &BigInt) -> bool {
    a.iter().all(|&x| x == 0)
}

// ---------------------------------------------------------------------------
// Precomputed constants (lazily initialized)
// ---------------------------------------------------------------------------

/// Modular exponentiation: base^exp mod modulus.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Reduce x mod p where p = 2^31 - c (pseudo-Mersenne).
#[inline]
pub fn reduce_mod(x: u64, p: u32, c: u32) -> u32 {
    let hi = x >> 31;
    let lo = x & 0x7FFF_FFFF;
    let r = lo + (c as u64) * hi;
    let hi2 = (r >> 31) as u32;
    let lo2 = (r & 0x7FFF_FFFF) as u32;
    let result = lo2 + c * hi2;
    if result >= p {
        result - p
    } else {
        result
    }
}

/// `2^(32*k) mod p_i` for k=0..7, for decomposing 256-bit integers.
fn pow2k_mod_p() -> &'static [[u32; 8]; NUM_PRIMES] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; 8]; NUM_PRIMES]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; 8]; NUM_PRIMES];
        for i in 0..NUM_PRIMES {
            let p = PRIMES[i] as u64;
            let mut power = 1u64;
            for slot in &mut table[i] {
                *slot = (power % p) as u32;
                power = (power << 32) % p;
            }
        }
        table
    })
}

/// `2^(32*k) mod p_i` for k=0..19 (extended to 640-bit range).
/// Used for decomposing big-integer M*a which can be up to ~533 bits.
fn pow2k_extended() -> &'static [[u32; 20]; NUM_PRIMES] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; 20]; NUM_PRIMES]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; 20]; NUM_PRIMES];
        for i in 0..NUM_PRIMES {
            let p = PRIMES[i] as u64;
            let mut power = 1u64;
            for slot in &mut table[i] {
                *slot = (power % p) as u32;
                power = (power << 32) % p;
            }
        }
        table
    })
}

/// Garner CRT inverse constants for the primary basis B:
/// `garner_inv_b[i][j] = m_j^{-1} mod m_i` for j < i, where m are PRIMARY_PRIMES.
pub(crate) fn garner_inv_b() -> &'static [[u32; BASIS_SIZE]; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; BASIS_SIZE]; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; BASIS_SIZE]; BASIS_SIZE];
        for i in 1..BASIS_SIZE {
            let pi = PRIMARY_PRIMES[i] as u64;
            for j in 0..i {
                let pj = PRIMARY_PRIMES[j] as u64;
                table[i][j] = mod_pow(pj, pi - 2, pi) as u32;
            }
        }
        table
    })
}

/// Garner CRT inverse constants for both bases (used for full 18-prime reconstruction).
fn garner_inv_full() -> &'static [[u32; NUM_PRIMES]; NUM_PRIMES] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; NUM_PRIMES]; NUM_PRIMES]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; NUM_PRIMES]; NUM_PRIMES];
        for i in 1..NUM_PRIMES {
            let pi = PRIMES[i] as u64;
            for j in 0..i {
                let pj = PRIMES[j] as u64;
                table[i][j] = mod_pow(pj, pi - 2, pi) as u32;
            }
        }
        table
    })
}

/// M = product of primary basis primes. Serves as the Montgomery constant R.
fn big_m() -> &'static BigInt {
    use std::sync::OnceLock;
    static VAL: OnceLock<BigInt> = OnceLock::new();
    VAL.get_or_init(|| {
        let mut product = BIG_ZERO;
        product[0] = PRIMARY_PRIMES[0] as u64;
        for &p in &PRIMARY_PRIMES[1..] {
            let mut p_big = BIG_ZERO;
            p_big[0] = p as u64;
            product = big_mul(&product, &p_big);
        }
        product
    })
}

/// r = BN254 scalar field order.
fn big_r() -> &'static BigInt {
    use std::sync::OnceLock;
    static VAL: OnceLock<BigInt> = OnceLock::new();
    VAL.get_or_init(|| {
        // BN254 Fr order: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
        let bytes: [u8; 32] = [
            0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8,
            0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ];
        big_from_bytes_le(&bytes)
    })
}

/// M mod r (used in Fr → RNS-Montgomery conversion).
fn big_m_mod_r() -> &'static BigInt {
    use std::sync::OnceLock;
    static VAL: OnceLock<BigInt> = OnceLock::new();
    VAL.get_or_init(|| big_mod(big_m(), big_r()))
}

/// M^{-1} mod r (used in RNS-Montgomery → Fr conversion, de-Montgomerize).
fn big_m_inv_mod_r() -> &'static BigInt {
    use std::sync::OnceLock;
    static VAL: OnceLock<BigInt> = OnceLock::new();
    VAL.get_or_init(|| {
        // M^{-1} mod r via Fermat: M^{r-2} mod r.
        // But r is 254-bit so we need big modpow.
        let m = big_m_mod_r();
        let r = big_r();
        // r - 2
        let mut exp = *r;
        // Subtract 2 from exp
        // r[0] > 2 always (BN254 order is odd and > 2), so single-limb subtract suffices.
        exp[0] -= 2;
        big_mod_pow(m, &exp, r)
    })
}

fn big_mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    let mut result = BIG_ZERO;
    result[0] = 1;
    let mut b = big_mod(base, modulus);
    let bits = big_bit_len(exp);

    for bit in 0..bits {
        let word = bit / 64;
        let pos = bit % 64;
        if (exp[word] >> pos) & 1 == 1 {
            result = big_mod(&big_mul(&result, &b), modulus);
        }
        b = big_mod(&big_mul(&b, &b), modulus);
    }
    result
}

/// `-r^{-1} mod m_i` for each primary prime m_i.
/// Used in RNS-Montgomery multiplication: q_i = -t_i · neg_r_inv_i mod m_i.
pub fn neg_r_inv_b() -> &'static [u32; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let r = big_r();
        let mut result = [0u32; BASIS_SIZE];
        for i in 0..BASIS_SIZE {
            let mi = PRIMARY_PRIMES[i] as u64;
            // r mod m_i
            let r_mod_mi = big_mod_small(r, mi);
            // r^{-1} mod m_i = r_mod_mi^{m_i - 2} mod m_i (Fermat)
            let r_inv = mod_pow(r_mod_mi, mi - 2, mi);
            // -r^{-1} mod m_i
            result[i] = (mi - r_inv) as u32;
        }
        result
    })
}

/// `M^{-1} mod m'_j` for each secondary prime m'_j.
/// Used in the final combine step of RNS-Montgomery multiplication.
pub fn m_inv_bp() -> &'static [u32; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let m = big_m();
        let mut result = [0u32; BASIS_SIZE];
        for j in 0..BASIS_SIZE {
            let mj = SECONDARY_PRIMES[j] as u64;
            let m_mod_mj = big_mod_small(m, mj);
            result[j] = mod_pow(m_mod_mj, mj - 2, mj) as u32;
        }
        result
    })
}

/// `r mod m'_j` for each secondary prime m'_j.
/// Used in combine step: s'_j = (t'_j + q'_j · r_j) · M^{-1}_j.
pub fn r_mod_bp() -> &'static [u32; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let r = big_r();
        let mut result = [0u32; BASIS_SIZE];
        for j in 0..BASIS_SIZE {
            let mj = SECONDARY_PRIMES[j] as u64;
            result[j] = big_mod_small(r, mj) as u32;
        }
        result
    })
}

/// `m_i mod m'_j` matrix for base extension B → B' (backward evaluation).
/// `b_mod_bp[i][j] = PRIMARY_PRIMES[i] mod SECONDARY_PRIMES[j]`.
pub fn b_mod_bp() -> &'static [[u32; BASIS_SIZE]; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; BASIS_SIZE]; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; BASIS_SIZE]; BASIS_SIZE];
        for i in 0..BASIS_SIZE {
            for j in 0..BASIS_SIZE {
                table[i][j] = PRIMARY_PRIMES[i] % SECONDARY_PRIMES[j];
            }
        }
        table
    })
}

/// `m'_j mod m_i` matrix for base extension B' → B (backward evaluation).
/// `bp_mod_b[j][i] = SECONDARY_PRIMES[j] mod PRIMARY_PRIMES[i]`.
pub fn bp_mod_b() -> &'static [[u32; BASIS_SIZE]; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; BASIS_SIZE]; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; BASIS_SIZE]; BASIS_SIZE];
        for j in 0..BASIS_SIZE {
            for i in 0..BASIS_SIZE {
                table[j][i] = SECONDARY_PRIMES[j] % PRIMARY_PRIMES[i];
            }
        }
        table
    })
}

/// Base extension from B' → B using exact Garner CRT.
///
/// Given residues in secondary basis B', compute residues in primary basis B.
/// Used to recover B-side representation after RNS-Montgomery multiply
/// (which produces results in B' only).
pub fn base_extend_bp_to_b(q_bp: &BasisResidue) -> BasisResidue {
    let garner = garner_inv_bp();
    let bp_mod = bp_mod_b();

    // Forward pass: Garner mixed-radix digits from B' residues
    let mut v = *q_bp;
    for i in 1..BASIS_SIZE {
        let pi = SECONDARY_PRIMES[i] as u64;
        for j in 0..i {
            let diff = if v[i] >= v[j] {
                (v[i] - v[j]) as u64
            } else {
                pi - (v[j] - v[i]) as u64
            };
            let inv = garner[i][j] as u64;
            v[i] = ((diff * inv) % pi) as u32;
        }
    }

    // Backward evaluation: q_i = Horner eval of mixed-radix digits at m_i
    let mut q_b = [0u32; BASIS_SIZE];
    let last = BASIS_SIZE - 1;
    for i in 0..BASIS_SIZE {
        let ci = PRIMARY_C[i];
        let mut val = v[last] as u64;
        for k in (0..last).rev() {
            val = reduce_mod(val * (bp_mod[k][i] as u64), PRIMARY_PRIMES[i], ci) as u64;
            val = reduce_mod(val + v[k] as u64, PRIMARY_PRIMES[i], ci) as u64;
        }
        q_b[i] = val as u32;
    }

    q_b
}

/// Reduce a BigInt modulo a small (< 2^32) modulus.
fn big_mod_small(a: &BigInt, m: u64) -> u64 {
    let mut result = 0u128;
    for i in (0..BIG_LIMBS).rev() {
        result = ((result << 64) + a[i] as u128) % m as u128;
    }
    result as u64
}

/// Decompose a BigInt into RNS residues across all 18 primes.
fn big_to_rns(a: &BigInt) -> RnsMontElement {
    let pow2k = pow2k_extended();
    // Convert BigInt to u32 limbs
    let mut u32_limbs = [0u32; 20];
    for i in 0..BIG_LIMBS {
        u32_limbs[2 * i] = a[i] as u32;
        u32_limbs[2 * i + 1] = (a[i] >> 32) as u32;
    }

    let mut residues = [0u32; NUM_PRIMES];
    for i in 0..NUM_PRIMES {
        let p = PRIMES[i];
        let c = C_VALUES[i];
        let mut acc: u64 = 0;
        for k in 0..20 {
            acc += (u32_limbs[k] as u64) * (pow2k[i][k] as u64);
            if k % 2 == 1 {
                acc = reduce_mod(acc, p, c) as u64;
            }
        }
        residues[i] = reduce_mod(acc, p, c);
    }
    residues
}

// ---------------------------------------------------------------------------
// Conversion: Fr ↔ RNS-Montgomery
// ---------------------------------------------------------------------------

/// Convert Fr → RNS-Montgomery element.
///
/// Computes `ã = a × M mod r` then decomposes into both bases.
/// This is the only conversion needed at the start of a sumcheck — no per-round conversions.
pub fn fr_to_rns_mont(fr: &Fr) -> RnsMontElement {
    use jolt_field::Field;
    let bytes = fr.to_bytes(); // standard integer (non-Montgomery)
    let a = big_from_bytes_le(&bytes);

    // ã = a × M mod r
    let a_mont = big_mod(&big_mul(&a, big_m_mod_r()), big_r());

    big_to_rns(&a_mont)
}

/// Convert RNS-Montgomery element → Fr.
///
/// Reconstructs the integer from B' residues (or full B∪B'),
/// reduces mod r, then de-Montgomerizes: `a = ã × M^{-1} mod r`.
#[cfg(test)]
pub fn rns_mont_to_fr(elem: &RnsMontElement) -> Fr {
    // Full CRT reconstruction using all 18 primes
    let bytes = rns_to_le_bytes_full(elem);
    let big_val = {
        let mut v = BIG_ZERO;
        for i in 0..BIG_LIMBS {
            v[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        v
    };

    // Reduce mod r
    let a_mont = big_mod(&big_val, big_r());

    // De-Montgomerize: a = a_mont × M^{-1} mod r
    let a = big_mod(&big_mul(&a_mont, big_m_inv_mod_r()), big_r());

    // Convert to Fr
    let mut le_bytes = [0u8; 32];
    for i in 0..4 {
        le_bytes[i * 8..(i + 1) * 8].copy_from_slice(&a[i].to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&le_bytes)
}

/// Convert Fr → standard RNS residues (non-Montgomery).
/// Used for integer-sum approach (legacy, still useful for weights).
#[cfg(test)]
pub fn fr_to_rns(fr: &Fr) -> [u32; NUM_PRIMES] {
    use jolt_field::Field;
    let bytes = fr.to_bytes();
    let limbs = u32_limbs_from_le_bytes(&bytes);
    decompose_u32_limbs(&limbs)
}

fn u32_limbs_from_le_bytes(bytes: &[u8; 32]) -> [u32; 8] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    limbs
}

fn decompose_u32_limbs(limbs: &[u32; 8]) -> [u32; NUM_PRIMES] {
    let pow2k = pow2k_mod_p();
    let mut residues = [0u32; NUM_PRIMES];
    for i in 0..NUM_PRIMES {
        let p = PRIMES[i];
        let c = C_VALUES[i];
        let mut acc: u64 = 0;
        for k in 0..8 {
            acc += (limbs[k] as u64) * (pow2k[i][k] as u64);
            if k % 2 == 1 {
                acc = reduce_mod(acc, p, c) as u64;
            }
        }
        residues[i] = reduce_mod(acc, p, c);
    }
    residues
}

/// Reconstruct integer from full 18-prime RNS via Garner CRT.
/// Returns 80 LE bytes (640 bits).
fn rns_to_le_bytes_full(residues: &RnsMontElement) -> [u8; 80] {
    let garner = garner_inv_full();

    let mut v = *residues;
    for i in 1..NUM_PRIMES {
        let pi = PRIMES[i] as u64;
        for j in 0..i {
            let diff = if v[i] >= v[j] {
                (v[i] - v[j]) as u64
            } else {
                pi - (v[j] - v[i]) as u64
            };
            let inv = garner[i][j] as u64;
            v[i] = ((diff * inv) % pi) as u32;
        }
    }

    let mut result = [0u64; BIG_LIMBS];
    let mut factor = [0u64; BIG_LIMBS];
    factor[0] = 1;

    for i in 0..NUM_PRIMES {
        let vi = v[i] as u64;
        let mut carry = 0u128;
        for limb in 0..BIG_LIMBS {
            carry += (factor[limb] as u128) * (vi as u128) + (result[limb] as u128);
            result[limb] = carry as u64;
            carry >>= 64;
        }

        if i < NUM_PRIMES - 1 {
            let pi = PRIMES[i] as u64;
            let mut carry_f = 0u128;
            for f in &mut factor {
                carry_f += (*f as u128) * (pi as u128);
                *f = carry_f as u64;
                carry_f >>= 64;
            }
        }
    }

    let mut bytes = [0u8; 80];
    for (i, &limb) in result.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// RNS-Montgomery multiplication (CPU reference, matches GPU kernel)
// ---------------------------------------------------------------------------

/// Base extension from B → B' using exact Garner CRT.
///
/// Given residues `q[0..9]` in primary basis B, compute `q'[0..9]` in secondary basis B'.
/// Forward pass: Garner mixed-radix digits from B residues.
/// Backward evaluation: evaluate at each m'_j.
pub fn base_extend_b_to_bp(q_b: &BasisResidue) -> BasisResidue {
    let garner = garner_inv_b();
    let b_mod = b_mod_bp();

    // Forward pass: compute mixed-radix digits v[i]
    let mut v = *q_b;
    for i in 1..BASIS_SIZE {
        let pi = PRIMARY_PRIMES[i] as u64;
        for j in 0..i {
            let diff = if v[i] >= v[j] {
                (v[i] - v[j]) as u64
            } else {
                pi - (v[j] - v[i]) as u64
            };
            let inv = garner[i][j] as u64;
            v[i] = ((diff * inv) % pi) as u32;
        }
    }

    // Backward evaluation: q'_j = v[0] + v[1]*m[0] + v[2]*m[0]*m[1] + ... mod m'_j
    let mut q_bp = [0u32; BASIS_SIZE];
    for j in 0..BASIS_SIZE {
        let mj = SECONDARY_PRIMES[j] as u64;
        let cj = SECONDARY_C[j];
        // Horner evaluation: starting from highest digit
        let mut val = v[BASIS_SIZE - 1] as u64;
        for i in (0..BASIS_SIZE - 1).rev() {
            // val = val * m_i + v[i], mod m'_j
            val = reduce_mod(val * (b_mod[i][j] as u64), SECONDARY_PRIMES[j], cj) as u64;
            val = reduce_mod(val + v[i] as u64, SECONDARY_PRIMES[j], cj) as u64;
        }
        q_bp[j] = val as u32;
        let _ = mj; // suppress unused warning
    }

    q_bp
}

/// RNS-Montgomery multiplication (CPU reference implementation).
///
/// Given ã, b̃ in RNS-Montgomery form (both bases), computes c̃ = (a·b)·M mod r.
/// Result is valid in B' only (caller must base-extend to B if needed).
#[cfg(test)]
pub fn rns_mont_mul(a: &RnsMontElement, b: &RnsMontElement) -> RnsMontElement {
    let neg_r_inv = neg_r_inv_b();
    let m_inv = m_inv_bp();
    let r_mod = r_mod_bp();

    // Step 1: component-wise product in B: t_i = ã_i · b̃_i mod m_i
    let mut t_b = [0u32; BASIS_SIZE];
    for i in 0..BASIS_SIZE {
        let mi = PRIMARY_PRIMES[i] as u64;
        let ci = PRIMARY_C[i];
        t_b[i] = reduce_mod((a[i] as u64) * (b[i] as u64), PRIMARY_PRIMES[i], ci);
        let _ = mi;
    }

    // Step 1b: component-wise product in B': t'_j = ã_j · b̃_j mod m'_j
    let mut t_bp = [0u32; BASIS_SIZE];
    for j in 0..BASIS_SIZE {
        let cj = SECONDARY_C[j];
        t_bp[j] = reduce_mod(
            (a[BASIS_SIZE + j] as u64) * (b[BASIS_SIZE + j] as u64),
            SECONDARY_PRIMES[j],
            cj,
        );
    }

    // Step 2: q_i = t_i · neg_r_inv_i mod m_i (in B)
    let mut q_b = [0u32; BASIS_SIZE];
    for i in 0..BASIS_SIZE {
        let ci = PRIMARY_C[i];
        q_b[i] = reduce_mod(
            (t_b[i] as u64) * (neg_r_inv[i] as u64),
            PRIMARY_PRIMES[i],
            ci,
        );
    }

    // Step 3: base-extend q from B → B'
    let q_bp = base_extend_b_to_bp(&q_b);

    // Step 4: combine in B': s'_j = (t'_j + q'_j · r_j) · M^{-1}_j mod m'_j
    let mut result = [0u32; NUM_PRIMES];

    // B side: we can reconstruct if needed, but for pure accumulation we only need B'
    // Store t_b as the B-side (not the final result, but useful for further ops)
    for i in 0..BASIS_SIZE {
        // For the B side of the result, we'd need the full formula with M^{-1} mod m_i.
        // Since M mod m_i = 0 (M = Π m_k), M^{-1} mod m_i doesn't exist!
        // This means the result in B must be computed differently.
        // Actually, s = (t + q·r) / M. In B: m_i | M, so (t_i + q_i·r_i) must be ≡ 0 mod m_i.
        // The B-side residue of s must be obtained via base extension from B'.
        // For now, store 0 — we reconstruct from B' when needed.
        result[i] = 0;
    }

    for j in 0..BASIS_SIZE {
        let mj = SECONDARY_PRIMES[j] as u64;
        let cj = SECONDARY_C[j];
        // t'_j + q'_j · r_j mod m'_j
        let qr = reduce_mod(
            (q_bp[j] as u64) * (r_mod[j] as u64),
            SECONDARY_PRIMES[j],
            cj,
        );
        let sum = reduce_mod((t_bp[j] as u64) + (qr as u64), SECONDARY_PRIMES[j], cj);
        // × M^{-1}_j
        result[BASIS_SIZE + j] =
            reduce_mod((sum as u64) * (m_inv[j] as u64), SECONDARY_PRIMES[j], cj);
        let _ = mj;
    }

    result
}

/// RNS-Montgomery addition in B': component-wise (a + b) mod m'_j.
#[cfg(test)]
pub fn rns_mont_add_bp(a: &RnsMontElement, b: &RnsMontElement) -> BasisResidue {
    let mut result = [0u32; BASIS_SIZE];
    for j in 0..BASIS_SIZE {
        let s = (a[BASIS_SIZE + j] as u64) + (b[BASIS_SIZE + j] as u64);
        result[j] = reduce_mod(s, SECONDARY_PRIMES[j], SECONDARY_C[j]);
    }
    result
}

/// RNS-Montgomery subtraction in B': component-wise (a - b) mod m'_j.
#[cfg(test)]
pub fn rns_mont_sub_bp(a: &RnsMontElement, b: &RnsMontElement) -> BasisResidue {
    let mut result = [0u32; BASIS_SIZE];
    for j in 0..BASIS_SIZE {
        let mj = SECONDARY_PRIMES[j];
        if a[BASIS_SIZE + j] >= b[BASIS_SIZE + j] {
            result[j] = a[BASIS_SIZE + j] - b[BASIS_SIZE + j];
        } else {
            result[j] = mj - (b[BASIS_SIZE + j] - a[BASIS_SIZE + j]);
        }
    }
    result
}

/// Convert B'-only residues back to Fr (for reading accumulator results).
///
/// Reconstructs integer from B' via Garner CRT, reduces mod r, de-Montgomerizes.
pub fn bp_residues_to_fr(bp: &BasisResidue) -> Fr {
    let garner = garner_inv_bp();

    let mut v = *bp;
    for i in 1..BASIS_SIZE {
        let pi = SECONDARY_PRIMES[i] as u64;
        for j in 0..i {
            let diff = if v[i] >= v[j] {
                (v[i] - v[j]) as u64
            } else {
                pi - (v[j] - v[i]) as u64
            };
            let inv = garner[i][j] as u64;
            v[i] = ((diff * inv) % pi) as u32;
        }
    }

    // Backward reconstruction
    let mut result = [0u64; BIG_LIMBS];
    let mut factor = [0u64; BIG_LIMBS];
    factor[0] = 1;

    for i in 0..BASIS_SIZE {
        let vi = v[i] as u64;
        let mut carry = 0u128;
        for limb in 0..BIG_LIMBS {
            carry += (factor[limb] as u128) * (vi as u128) + (result[limb] as u128);
            result[limb] = carry as u64;
            carry >>= 64;
        }
        if i < BASIS_SIZE - 1 {
            let pi = SECONDARY_PRIMES[i] as u64;
            let mut carry_f = 0u128;
            for f in &mut factor {
                carry_f += (*f as u128) * (pi as u128);
                *f = carry_f as u64;
                carry_f >>= 64;
            }
        }
    }

    // result is ã = a·M mod r (but could be larger than r if CRT range > r)
    let a_mont_big = result;
    let r = big_r();
    let a_mont = big_mod(&a_mont_big, r);
    let a = big_mod(&big_mul(&a_mont, big_m_inv_mod_r()), r);

    let mut le_bytes = [0u8; 32];
    for i in 0..4 {
        le_bytes[i * 8..(i + 1) * 8].copy_from_slice(&a[i].to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&le_bytes)
}

/// Garner CRT inverse constants for the secondary basis B'.
pub fn garner_inv_bp() -> &'static [[u32; BASIS_SIZE]; BASIS_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; BASIS_SIZE]; BASIS_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u32; BASIS_SIZE]; BASIS_SIZE];
        for i in 1..BASIS_SIZE {
            let pi = SECONDARY_PRIMES[i] as u64;
            for j in 0..i {
                let pj = SECONDARY_PRIMES[j] as u64;
                table[i][j] = mod_pow(pj, pi - 2, pi) as u32;
            }
        }
        table
    })
}

/// Batch conversion: `Fr[]` → flat u32 buffer in SoA layout for GPU upload.
///
/// Layout: `[B_0: n vals | B_1: n vals | ... | B_8: n vals | B'_0: n vals | ... | B'_8: n vals]`
/// Total output length: `NUM_PRIMES × data.len()`.
pub fn batch_fr_to_rns_mont(data: &[Fr]) -> Vec<u32> {
    let n = data.len();
    let mut out = vec![0u32; NUM_PRIMES * n];

    for (i, fr) in data.iter().enumerate() {
        let elem = fr_to_rns_mont(fr);
        for (j, &r) in elem.iter().enumerate() {
            out[j * n + i] = r;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use num_traits::{One, Zero};

    fn verify_primality(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        let mut d = n - 1;
        let mut s = 0u32;
        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }
        for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if a >= n {
                continue;
            }
            let mut x = mod_pow(a, d, n);
            if x == 1 || x == n - 1 {
                continue;
            }
            let mut found = false;
            for _ in 0..s - 1 {
                x = ((x as u128 * x as u128) % n as u128) as u64;
                if x == n - 1 {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }

    #[test]
    fn all_primes_are_prime() {
        for (i, &p) in PRIMES.iter().enumerate() {
            assert!(verify_primality(p as u64), "PRIMES[{i}] = {p} is not prime");
            assert_eq!(
                p,
                (1u32 << 31) - C_VALUES[i],
                "PRIMES[{i}] doesn't match C_VALUES[{i}]"
            );
        }
    }

    #[test]
    fn primary_basis_covers_field_order() {
        // M = Π m_i > r ≈ 2^254. log2(M) = Σ log2(m_i) ≈ 9 × 31 = 279.
        let log2_m: f64 = PRIMARY_PRIMES.iter().map(|&p| (p as f64).log2()).sum();
        assert!(
            log2_m > 254.0,
            "Primary basis product too small: log2(M) = {log2_m:.1}, need > 254"
        );
    }

    #[test]
    fn secondary_basis_covers_field_order() {
        let log2_mp: f64 = SECONDARY_PRIMES.iter().map(|&p| (p as f64).log2()).sum();
        assert!(
            log2_mp > 254.0,
            "Secondary basis product too small: log2(M') = {log2_mp:.1}, need > 254"
        );
    }

    #[test]
    fn big_r_is_correct() {
        let r = big_r();
        // Verify first limb
        assert_eq!(r[0], 0x43e1f593f0000001);
        assert_eq!(r[1], 0x2833e84879b97091);
        assert_eq!(r[2], 0xb85045b68181585d);
        assert_eq!(r[3], 0x30644e72e131a029);
    }

    #[test]
    fn m_times_m_inv_is_one_mod_r() {
        let m = big_m_mod_r();
        let m_inv = big_m_inv_mod_r();
        let r = big_r();
        let product = big_mod(&big_mul(m, m_inv), r);
        let mut one = BIG_ZERO;
        one[0] = 1;
        assert_eq!(product, one, "M × M^{{-1}} mod r should be 1");
    }

    #[test]
    fn neg_r_inv_correct() {
        let neg_r_inv = neg_r_inv_b();
        let r = big_r();
        for i in 0..BASIS_SIZE {
            let mi = PRIMARY_PRIMES[i] as u64;
            let r_mod_mi = big_mod_small(r, mi);
            // neg_r_inv[i] * r ≡ -1 mod m_i, i.e. (neg_r_inv[i] * r_mod_mi + 1) % mi == 0
            let product = ((neg_r_inv[i] as u64) * r_mod_mi) % mi;
            // product should be mi - 1 (≡ -1 mod mi)
            assert_eq!(product, mi - 1, "neg_r_inv[{i}] × r should ≡ -1 mod m_{i}");
        }
    }

    #[test]
    fn roundtrip_rns_mont_zero() {
        let fr = Fr::zero();
        let elem = fr_to_rns_mont(&fr);
        // Zero maps to zero in RNS-Montgomery (0 × M mod r = 0)
        assert_eq!(
            elem, [0u32; NUM_PRIMES],
            "RNS-Mont of 0 should be all zeros"
        );
        let back = rns_mont_to_fr(&elem);
        assert_eq!(back, fr);
    }

    #[test]
    fn roundtrip_rns_mont_one() {
        let fr = Fr::one();
        let elem = fr_to_rns_mont(&fr);
        // 1 × M mod r = M mod r. Residues should be (M mod r) mod m_i for each prime.
        let m_mod_r = big_m_mod_r();
        for i in 0..NUM_PRIMES {
            let expected = big_mod_small(m_mod_r, PRIMES[i] as u64) as u32;
            assert_eq!(
                elem[i], expected,
                "RNS-Mont of 1: prime {i} expected {expected}, got {}",
                elem[i]
            );
        }
        let back = rns_mont_to_fr(&elem);
        assert_eq!(back, fr, "RNS-Mont roundtrip of 1 failed");
    }

    #[test]
    fn roundtrip_rns_mont_small() {
        for val in [2u64, 42, 1000, u32::MAX as u64, u64::MAX] {
            let fr = Fr::from_u64(val);
            let elem = fr_to_rns_mont(&fr);
            let back = rns_mont_to_fr(&elem);
            assert_eq!(back, fr, "RNS-Mont roundtrip failed for {val}");
        }
    }

    #[test]
    fn roundtrip_rns_mont_random() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let fr = Fr::random(&mut rng);
            let elem = fr_to_rns_mont(&fr);
            let back = rns_mont_to_fr(&elem);
            assert_eq!(back, fr, "RNS-Mont roundtrip failed for random element");
        }
    }

    #[test]
    fn rns_mont_mul_matches_field() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(77);

        for _ in 0..50 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let expected = a * b;

            let a_rns = fr_to_rns_mont(&a);
            let b_rns = fr_to_rns_mont(&b);
            let c_rns = rns_mont_mul(&a_rns, &b_rns);

            // Result is in B' only; reconstruct from B' residues
            let mut bp = [0u32; BASIS_SIZE];
            bp.copy_from_slice(&c_rns[BASIS_SIZE..]);
            let result = bp_residues_to_fr(&bp);

            assert_eq!(result, expected, "RNS-Mont multiplication mismatch");
        }
    }

    #[test]
    fn rns_mont_mul_one_is_identity() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);

        let one = fr_to_rns_mont(&Fr::one());
        for _ in 0..20 {
            let a_fr = Fr::random(&mut rng);
            let a = fr_to_rns_mont(&a_fr);
            let product = rns_mont_mul(&a, &one);

            let mut bp = [0u32; BASIS_SIZE];
            bp.copy_from_slice(&product[BASIS_SIZE..]);
            let result = bp_residues_to_fr(&bp);
            assert_eq!(result, a_fr, "Multiplication by 1 should be identity");
        }
    }

    #[test]
    fn base_extend_roundtrip() {
        // Take a value in B, extend to B', verify correctness by reconstructing full integer.
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..50 {
            let fr = Fr::random(&mut rng);
            let full = fr_to_rns_mont(&fr);

            // Extract B residues, extend to B'
            let mut q_b = [0u32; BASIS_SIZE];
            q_b.copy_from_slice(&full[..BASIS_SIZE]);
            let q_bp = base_extend_b_to_bp(&q_b);

            // The extended B' residues should match the directly computed ones
            for j in 0..BASIS_SIZE {
                assert_eq!(
                    q_bp[j],
                    full[BASIS_SIZE + j],
                    "Base extension mismatch at j={j}: direct={}, extended={}",
                    full[BASIS_SIZE + j],
                    q_bp[j]
                );
            }
        }
    }

    #[test]
    fn batch_rns_mont_matches_individual() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(55);

        let n = 64;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let batch = batch_fr_to_rns_mont(&data);

        for (i, fr) in data.iter().enumerate() {
            let individual = fr_to_rns_mont(fr);
            for j in 0..NUM_PRIMES {
                assert_eq!(
                    batch[j * n + i],
                    individual[j],
                    "batch mismatch at element {i}, residue {j}"
                );
            }
        }
    }

    #[test]
    fn garner_inv_b_correct() {
        let garner = garner_inv_b();
        for i in 1..BASIS_SIZE {
            let pi = PRIMARY_PRIMES[i] as u64;
            for j in 0..i {
                let pj = PRIMARY_PRIMES[j] as u64;
                let inv = garner[i][j] as u64;
                let product = ((pj as u128 * inv as u128) % pi as u128) as u64;
                assert_eq!(
                    product, 1,
                    "Garner B inv wrong: p_B[{j}]^{{-1}} mod p_B[{i}]"
                );
            }
        }
    }

    #[test]
    fn garner_inv_bp_correct() {
        let garner = garner_inv_bp();
        for i in 1..BASIS_SIZE {
            let pi = SECONDARY_PRIMES[i] as u64;
            for j in 0..i {
                let pj = SECONDARY_PRIMES[j] as u64;
                let inv = garner[i][j] as u64;
                let product = ((pj as u128 * inv as u128) % pi as u128) as u64;
                assert_eq!(
                    product, 1,
                    "Garner B' inv wrong: p_B'[{j}]^{{-1}} mod p_B'[{i}]"
                );
            }
        }
    }
}
