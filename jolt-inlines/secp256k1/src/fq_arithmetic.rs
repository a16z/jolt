//! Arithmetic on canonical, non-Montgomery secp256k1 base-field limbs.
//!
//! With p = 2^256 - C, an overflowing sum needs a correction of +C and
//! a borrowing difference needs -C. Usually only the low limb changes.

const C: u64 = (1u64 << 32) + 977;
const P_LOW: u64 = 0u64.wrapping_sub(C);

#[inline(always)]
fn adc(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (sum, c1) = a.overflowing_add(b);
    let (sum, c2) = sum.overflowing_add(carry as u64);
    (sum, c1 || c2)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: bool) -> (u64, bool) {
    let (difference, b1) = a.overflowing_sub(b);
    let (difference, b2) = difference.overflowing_sub(borrow as u64);
    (difference, b1 || b2)
}

/// Reduce a sum of two canonical elements, including its 257th bit.
#[inline(always)]
fn reduce_sum(mut sum: [u64; 4], carry: bool) -> [u64; 4] {
    if carry {
        // a+b <= 2p-2, so (a+b-2^256)+C <= p-2: no second reduction.
        let (low, mut carry) = sum[0].overflowing_add(C);
        sum[0] = low;
        // Spell out propagation so LLVM keeps the limbs in registers.
        if carry {
            (sum[1], carry) = sum[1].overflowing_add(1);
            if carry {
                (sum[2], carry) = sum[2].overflowing_add(1);
                if carry {
                    sum[3] = sum[3].wrapping_add(1);
                }
            }
        }
    } else if sum[1] & sum[2] & sum[3] == u64::MAX && sum[0] >= P_LOW {
        // Without overflow, sum >= p requires three all-one high limbs.
        sum = [sum[0] - P_LOW, 0, 0, 0];
    }
    sum
}

#[inline(always)]
pub(super) fn add(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut sum = [0; 4];
    let mut carry = false;
    for i in 0..4 {
        (sum[i], carry) = adc(a[i], b[i], carry);
    }
    reduce_sum(sum, carry)
}

#[inline(always)]
pub(super) fn sub(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut difference = [0; 4];
    let mut borrow = false;
    for i in 0..4 {
        (difference[i], borrow) = sbb(a[i], b[i], borrow);
    }
    if borrow {
        // The wrapped difference is a-b+2^256. Subtracting C yields
        // a-b+p in [1,p-1], so the correction cannot underflow 256 bits.
        let (low, mut borrow) = difference[0].overflowing_sub(C);
        difference[0] = low;
        if borrow {
            (difference[1], borrow) = difference[1].overflowing_sub(1);
            if borrow {
                (difference[2], borrow) = difference[2].overflowing_sub(1);
                if borrow {
                    difference[3] = difference[3].wrapping_sub(1);
                }
            }
        }
    }
    difference
}

#[inline(always)]
pub(super) fn double(a: [u64; 4]) -> [u64; 4] {
    reduce_sum(
        [
            a[0] << 1,
            (a[1] << 1) | (a[0] >> 63),
            (a[2] << 1) | (a[1] >> 63),
            (a[3] << 1) | (a[2] >> 63),
        ],
        a[3] >> 63 != 0,
    )
}
