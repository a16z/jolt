//! BN254 modular arithmetic for constant folding.
//!
//! Operates on `[u64; 4]` limbs in little-endian order. All arithmetic is
//! modular over the BN254 scalar field prime:
//!
//! ```text
//! p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//! ```

use num_bigint::BigUint;

/// BN254 scalar field prime as `[u64; 4]` (little-endian limbs).
pub const MODULUS: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

fn to_biguint(val: [u64; 4]) -> BigUint {
    let mut bytes = [0u8; 32];
    for (i, limb) in val.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

fn from_biguint(n: &BigUint) -> [u64; 4] {
    let bytes = n.to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        if start < bytes.len() {
            let end = (start + 8).min(bytes.len());
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}

fn modulus() -> BigUint {
    to_biguint(MODULUS)
}

/// Zero element.
pub const ZERO: [u64; 4] = [0, 0, 0, 0];

/// One element.
pub const ONE: [u64; 4] = [1, 0, 0, 0];

pub fn is_zero(val: [u64; 4]) -> bool {
    val == ZERO
}

pub fn is_one(val: [u64; 4]) -> bool {
    val == ONE
}

/// Modular addition: `(a + b) mod p`.
pub fn add(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let p = modulus();
    let sum = (to_biguint(a) + to_biguint(b)) % &p;
    from_biguint(&sum)
}

/// Modular subtraction: `(a - b) mod p`.
pub fn sub(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let p = modulus();
    let a_big = to_biguint(a);
    let b_big = to_biguint(b);
    let result = if a_big >= b_big {
        (a_big - b_big) % &p
    } else {
        (&p - (b_big - a_big) % &p) % &p
    };
    from_biguint(&result)
}

/// Modular multiplication: `(a * b) mod p`.
pub fn mul(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let p = modulus();
    let prod = (to_biguint(a) * to_biguint(b)) % &p;
    from_biguint(&prod)
}

/// Modular negation: `(-a) mod p`.
pub fn neg(a: [u64; 4]) -> [u64; 4] {
    if is_zero(a) {
        ZERO
    } else {
        sub(ZERO, a)
    }
}

/// Modular inverse: `a^{-1} mod p`, or `None` for zero.
pub fn inv(a: [u64; 4]) -> Option<[u64; 4]> {
    if is_zero(a) {
        return None;
    }
    let p = modulus();
    let a_big = to_biguint(a);
    // a^{p-2} mod p (Fermat's little theorem)
    let exp = &p - BigUint::from(2u64);
    let result = a_big.modpow(&exp, &p);
    Some(from_biguint(&result))
}

/// Modular division: `(a / b) mod p`, or `None` if `b` is zero.
pub fn div(a: [u64; 4], b: [u64; 4]) -> Option<[u64; 4]> {
    inv(b).map(|b_inv| mul(a, b_inv))
}

pub fn from_u64(n: u64) -> [u64; 4] {
    [n, 0, 0, 0]
}

/// Converts a `u128` to limbs, reduced mod p.
pub fn from_u128(n: u128) -> [u64; 4] {
    let p = modulus();
    let big = BigUint::from(n) % &p;
    from_biguint(&big)
}

/// Converts a signed `i64` to its field representative.
pub fn from_i64(n: i64) -> [u64; 4] {
    if n >= 0 {
        from_u64(n as u64)
    } else {
        neg(from_u64(n.unsigned_abs()))
    }
}

/// Converts a signed `i128` to its field representative.
pub fn from_i128(n: i128) -> [u64; 4] {
    if n >= 0 {
        from_u128(n as u128)
    } else {
        let abs = from_u128(n.unsigned_abs());
        neg(abs)
    }
}

pub fn square(a: [u64; 4]) -> [u64; 4] {
    mul(a, a)
}

pub fn to_decimal_string(val: [u64; 4]) -> String {
    to_biguint(val).to_string()
}

pub fn to_bytes_le(val: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, limb) in val.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

/// Converts little-endian bytes to limbs, reduced mod p.
pub fn from_bytes_le(bytes: &[u8]) -> [u64; 4] {
    let p = modulus();
    let big = BigUint::from_bytes_le(bytes) % &p;
    from_biguint(&big)
}

pub fn num_bits(val: [u64; 4]) -> u32 {
    to_biguint(val).bits() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_laws() {
        let a = from_u64(42);
        assert_eq!(add(a, ZERO), a);
        assert_eq!(mul(a, ONE), a);
        assert_eq!(sub(a, ZERO), a);
    }

    #[test]
    fn additive_inverse() {
        let a = from_u64(12345);
        let neg_a = neg(a);
        assert_eq!(add(a, neg_a), ZERO);
    }

    #[test]
    fn multiplicative_inverse() {
        let a = from_u64(7);
        let a_inv = inv(a).unwrap();
        assert_eq!(mul(a, a_inv), ONE);
    }

    #[test]
    fn zero_inverse_is_none() {
        assert!(inv(ZERO).is_none());
    }

    #[test]
    fn sub_underflow_wraps() {
        let a = from_u64(3);
        let b = from_u64(5);
        // 3 - 5 mod p = p - 2
        let result = sub(a, b);
        assert_eq!(add(result, b), a);
    }

    #[test]
    fn signed_conversion() {
        let pos = from_i64(42);
        let neg_val = from_i64(-42);
        assert_eq!(add(pos, neg_val), ZERO);

        let pos128 = from_i128(100);
        let neg128 = from_i128(-100);
        assert_eq!(add(pos128, neg128), ZERO);
    }

    #[test]
    fn square_consistency() {
        let a = from_u64(17);
        assert_eq!(square(a), mul(a, a));
    }

    #[test]
    fn bytes_roundtrip() {
        let a = from_u64(0xDEAD_BEEF);
        let bytes = to_bytes_le(a);
        assert_eq!(from_bytes_le(&bytes), a);
    }

    #[test]
    fn decimal_string() {
        let a = from_u64(42);
        assert_eq!(to_decimal_string(a), "42");
    }

    #[test]
    fn division() {
        let a = from_u64(21);
        let b = from_u64(7);
        let result = div(a, b).unwrap();
        assert_eq!(result, from_u64(3));
    }

    #[test]
    fn large_multiplication_wraps() {
        // p - 1 squared should equal 1 (since (p-1)^2 = p^2 - 2p + 1 ≡ 1 mod p)
        let p_minus_1 = sub(from_biguint(&modulus()), ONE);
        assert_eq!(mul(p_minus_1, p_minus_1), ONE);
    }

    #[test]
    fn u128_conversion() {
        let large: u128 = (1u128 << 127) + 42;
        let val = from_u128(large);
        // Should be reduced mod p
        assert!(!is_zero(val));
        let p = modulus();
        let expected = BigUint::from(large) % &p;
        assert_eq!(to_biguint(val), expected);
    }
}
