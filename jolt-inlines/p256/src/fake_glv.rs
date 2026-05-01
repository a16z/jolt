/// Half-GCD decomposition for P-256 "Fake GLV" scalar multiplication.
///
/// Given scalar s, finds (a, b) with b*s ≡ a (mod n) and |a|, |b| <= √n.
/// Based on the extended GCD algorithm truncated at √n.
use num_bigint::{BigInt as NBigInt, BigUint as NBigUint, Sign};

use crate::P256_ORDER;

fn order_bigint() -> NBigInt {
    let mut bytes = Vec::with_capacity(32);
    for &limb in &P256_ORDER {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    NBigInt::from_bytes_le(Sign::Plus, &bytes)
}

/// Decompose scalar `s` via half-GCD: returns `(a, b)` with `b*s ≡ a (mod n)`,
/// `|a|, |b| <= √n`.
pub(crate) fn decompose_scalar(s: &NBigInt) -> (NBigInt, NBigInt) {
    let n = order_bigint();
    let sqrt_n: NBigInt = {
        let n_uint: NBigUint = n.to_biguint().unwrap();
        n_uint.sqrt().into()
    };

    let (mut old_u, mut u_val) = (n.clone(), s.clone());
    let (mut old_v, mut v_val) = (NBigInt::from(0i64), NBigInt::from(1i64));
    while u_val.magnitude() >= sqrt_n.magnitude() {
        let q = &old_u / &u_val;
        let (new_u, new_v) = (&old_u - &q * &u_val, &old_v - &q * &v_val);
        old_u = u_val;
        u_val = new_u;
        old_v = v_val;
        v_val = new_v;
    }

    (u_val, v_val)
}

/// Serialize a signed big-integer as `(lo, hi, sign)` triple of u64 values.
fn serialize_128(val: &NBigInt) -> (u64, u64, u64) {
    let sign = if val.sign() == Sign::Minus {
        1u64
    } else {
        0u64
    };
    let bytes = val.magnitude().to_bytes_le();
    let mut lo = 0u64;
    let mut hi = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        if i < 8 {
            lo |= (b as u64) << (i * 8);
        } else if i < 16 {
            hi |= (b as u64) << ((i - 8) * 8);
        }
    }
    (lo, hi, sign)
}

/// Decompose scalar and serialize as 6 u64 values:
/// `[a_lo, a_hi, a_sign, b_lo, b_hi, b_sign]`.
pub(crate) fn decompose_to_u64s(s: &NBigInt) -> [u64; 6] {
    let (a, b) = decompose_scalar(s);
    let (a_lo, a_hi, a_sign) = serialize_128(&a);
    let (b_lo, b_hi, b_sign) = serialize_128(&b);
    [a_lo, a_hi, a_sign, b_lo, b_hi, b_sign]
}

/// Decompose scalar and return as `(u128, bool)` pairs (value, is_negative).
pub(crate) fn decompose_to_u128s(s: &NBigInt) -> (u128, bool, u128, bool) {
    let (a, b) = decompose_scalar(s);
    let to_u128 = |val: &NBigInt| -> (u128, bool) {
        let sign = val.sign() == Sign::Minus;
        let bytes = val.magnitude().to_bytes_le();
        let mut lo = 0u64;
        let mut hi = 0u64;
        for (i, &byte) in bytes.iter().enumerate() {
            if i < 8 {
                lo |= (byte as u64) << (i * 8);
            } else if i < 16 {
                hi |= (byte as u64) << ((i - 8) * 8);
            }
        }
        ((lo as u128) | ((hi as u128) << 64), sign)
    };
    let (a_val, a_sign) = to_u128(&a);
    let (b_val, b_sign) = to_u128(&b);
    (a_val, a_sign, b_val, b_sign)
}
