use jolt_kernels::metal::solinas::{Fp128, OFFSET_275};

const MODULUS: u128 = u128::MAX - OFFSET_275 as u128 + 1;

#[inline(always)]
pub(crate) fn add(lhs: Fp128, rhs: Fp128) -> Fp128 {
    let (sum, carry) = lhs.to_u128().overflowing_add(rhs.to_u128());
    let sum = if carry { sum + OFFSET_275 as u128 } else { sum };
    Fp128::from_u128(if sum >= MODULUS { sum - MODULUS } else { sum })
}

#[inline(always)]
pub(crate) fn sub(lhs: Fp128, rhs: Fp128) -> Fp128 {
    let lhs = lhs.to_u128();
    let rhs = rhs.to_u128();
    Fp128::from_u128(if lhs >= rhs {
        lhs - rhs
    } else {
        MODULUS - (rhs - lhs)
    })
}

#[inline(always)]
pub(crate) fn bind(lo: Fp128, hi: Fp128, challenge: Fp128) -> Fp128 {
    add(lo, mul(challenge, sub(hi, lo)))
}

/// Temporary portable control for a matched CPU/GPU pointwise workload.
///
/// This is replaced by `jolt_field::Prime128Offset275` when that type lands.
#[inline(always)]
pub(crate) fn mul(lhs: Fp128, rhs: Fp128) -> Fp128 {
    let lhs = lhs.to_u128();
    let rhs = rhs.to_u128();
    let (a0, a1) = (lhs as u64, (lhs >> 64) as u64);
    let (b0, b1) = (rhs as u64, (rhs >> 64) as u64);
    let (p00_lo, p00_hi) = mul64_wide(a0, b0);
    let (p01_lo, p01_hi) = mul64_wide(a0, b1);
    let (p10_lo, p10_hi) = mul64_wide(a1, b0);
    let (p11_lo, p11_hi) = mul64_wide(a1, b1);

    let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
    let r0 = p00_lo;
    let r1 = row1 as u64;
    let carry1 = (row1 >> 64) as u64;
    let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
    let r2 = row2 as u64;
    let carry2 = (row2 >> 64) as u64;
    let r3 = (p11_hi as u128 + carry2 as u128) as u64;

    Fp128::from_u128(reduce(r0, r1, r2, r3))
}

#[inline(always)]
fn mul64_wide(lhs: u64, rhs: u64) -> (u64, u64) {
    let product = lhs as u128 * rhs as u128;
    (product as u64, (product >> 64) as u64)
}

#[inline(always)]
fn reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> u128 {
    let offset = OFFSET_275 as u64;
    let cr2 = offset as u128 * r2 as u128;
    let cr3 = offset as u128 * r3 as u128;
    let t0_sum = r0 as u128 + cr2 as u64 as u128;
    let t0 = t0_sum as u64;
    let t1_sum = r1 as u128 + (cr2 >> 64) as u64 as u128 + cr3 as u64 as u128 + (t0_sum >> 64);
    let t1 = t1_sum as u64;
    let t2 = ((cr3 >> 64) + (t1_sum >> 64)) as u64;

    let ct2 = offset as u128 * t2 as u128;
    let (s0, carry0) = t0.overflowing_add(ct2 as u64);
    let (s1a, carry1a) = t1.overflowing_add((ct2 >> 64) as u64);
    let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
    let overflow = carry1a | carry1b;
    let (corrected0, carry2) = s0.overflowing_add(offset);
    let (corrected1, canonical_carry) = s1.overflowing_add(carry2 as u64);
    let [lo, hi] = if overflow | canonical_carry {
        [corrected0, corrected1]
    } else {
        [s0, s1]
    };
    lo as u128 | (hi as u128) << 64
}
