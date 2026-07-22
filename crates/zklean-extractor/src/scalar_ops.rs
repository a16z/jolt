//! Scalar arithmetic operations for BN254 field elements.
//!
//! This module provides modular arithmetic for 256-bit scalars represented as `[u64; 4]`.
//! These functions are used for constant folding during code generation.
//!
//! All operations are performed modulo the BN254 scalar field modulus:
//! p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

use num_bigint::BigUint;

use crate::mle_ast::Scalar;

// =============================================================================
// Constants
// =============================================================================

/// BN254 scalar field modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const BN254_MODULUS: [u64; 4] = [
    0x43E1F593F0000001,
    0x2833E84879B97091,
    0xB85045B68181585D,
    0x30644E72E131A029,
];

/// Zero scalar: [0, 0, 0, 0]
pub const SCALAR_ZERO: Scalar = [0, 0, 0, 0];

/// One scalar: [1, 0, 0, 0]
pub const SCALAR_ONE: Scalar = [1, 0, 0, 0];

// =============================================================================
// Public arithmetic functions
// =============================================================================

/// Convert a Scalar to a decimal string.
pub fn scalar_to_decimal_string(limbs: &Scalar) -> String {
    if *limbs == SCALAR_ZERO {
        return "0".to_string();
    }
    scalar_to_biguint(limbs).to_string()
}

/// Add two 256-bit numbers with modular reduction.
///
/// Computes (a + b) mod p where p is the BN254 scalar field modulus.
pub fn scalar_add_mod(a: Scalar, b: Scalar) -> Scalar {
    let mut result = [0u64; 4];
    let mut carry = 0u128;

    for i in 0..4 {
        let sum = a[i] as u128 + b[i] as u128 + carry;
        result[i] = sum as u64;
        carry = sum >> 64;
    }

    // Reduce mod p if needed
    if carry > 0 || scalar_ge(&result, &BN254_MODULUS) {
        scalar_sub_assuming_ge(&result, &BN254_MODULUS)
    } else {
        result
    }
}

/// Subtract two 256-bit numbers with modular reduction.
///
/// Computes (a - b) mod p where p is the BN254 scalar field modulus.
pub fn scalar_sub_mod(a: Scalar, b: Scalar) -> Scalar {
    // a - b mod p = a + (p - b) mod p
    let neg_b = scalar_neg_mod(b);
    scalar_add_mod(a, neg_b)
}

/// Negate a scalar: -a mod p = p - a.
pub fn scalar_neg_mod(a: Scalar) -> Scalar {
    if a == SCALAR_ZERO {
        return SCALAR_ZERO;
    }
    scalar_sub_assuming_ge(&BN254_MODULUS, &a)
}

/// Multiply two 256-bit numbers with modular reduction.
///
/// Computes (a * b) mod p where p is the BN254 scalar field modulus.
///
/// Note: This uses BigUint for simplicity. For performance-critical code,
/// Montgomery multiplication would be more efficient.
pub fn scalar_mul_mod(a: Scalar, b: Scalar) -> Scalar {
    let result =
        (scalar_to_biguint(&a) * scalar_to_biguint(&b)) % scalar_to_biguint(&BN254_MODULUS);
    biguint_to_scalar(result)
}

// =============================================================================
// Internal helper functions
// =============================================================================

/// Convert a `[u64; 4]` scalar to BigUint (little-endian limb order).
fn scalar_to_biguint(s: &Scalar) -> BigUint {
    BigUint::from_slice(&[
        s[0] as u32,
        (s[0] >> 32) as u32,
        s[1] as u32,
        (s[1] >> 32) as u32,
        s[2] as u32,
        (s[2] >> 32) as u32,
        s[3] as u32,
        (s[3] >> 32) as u32,
    ])
}

/// Convert a BigUint to `[u64; 4]` scalar (little-endian limb order).
fn biguint_to_scalar(n: BigUint) -> Scalar {
    let digits = n.to_u64_digits();
    let mut out = [0u64; 4];
    for (i, &d) in digits.iter().take(4).enumerate() {
        out[i] = d;
    }
    out
}

/// Compare two 256-bit numbers: returns true if a >= b.
pub(crate) fn scalar_ge(a: &Scalar, b: &Scalar) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true // equal
}

/// Subtract b from a, assuming a >= b (no modular wrap).
pub(crate) fn scalar_sub_assuming_ge(a: &Scalar, b: &Scalar) -> Scalar {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = a[i] as i128 - b[i] as i128 - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_zero_one() {
        assert_eq!(SCALAR_ZERO, [0, 0, 0, 0]);
        assert_eq!(SCALAR_ONE, [1, 0, 0, 0]);
    }

    #[test]
    fn test_scalar_add_simple() {
        let a = [1, 0, 0, 0];
        let b = [2, 0, 0, 0];
        let result = scalar_add_mod(a, b);
        assert_eq!(result, [3, 0, 0, 0]);
    }

    #[test]
    fn test_scalar_neg() {
        let a = [1, 0, 0, 0];
        let neg_a = scalar_neg_mod(a);
        // -1 mod p = p - 1
        let expected = scalar_sub_assuming_ge(&BN254_MODULUS, &a);
        assert_eq!(neg_a, expected);

        // a + (-a) = 0
        let sum = scalar_add_mod(a, neg_a);
        assert_eq!(sum, SCALAR_ZERO);
    }

    #[test]
    fn test_scalar_to_decimal_string() {
        assert_eq!(scalar_to_decimal_string(&SCALAR_ZERO), "0");
        assert_eq!(scalar_to_decimal_string(&SCALAR_ONE), "1");
        assert_eq!(scalar_to_decimal_string(&[42, 0, 0, 0]), "42");
    }

    #[test]
    fn test_scalar_sub_mod() {
        // 3 - 1 = 2
        let result = scalar_sub_mod([3, 0, 0, 0], SCALAR_ONE);
        assert_eq!(result, [2, 0, 0, 0]);

        // 0 - 1 = p - 1 (wraps around)
        let result = scalar_sub_mod(SCALAR_ZERO, SCALAR_ONE);
        let expected = scalar_sub_assuming_ge(&BN254_MODULUS, &SCALAR_ONE);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_mul_mod() {
        // 2 * 3 = 6
        let result = scalar_mul_mod([2, 0, 0, 0], [3, 0, 0, 0]);
        assert_eq!(result, [6, 0, 0, 0]);

        // 1 * x = x
        let x = [42, 0, 0, 0];
        assert_eq!(scalar_mul_mod(SCALAR_ONE, x), x);

        // 0 * x = 0
        assert_eq!(scalar_mul_mod(SCALAR_ZERO, x), SCALAR_ZERO);
    }

    #[test]
    fn test_add_overflow_reduces_mod_p() {
        // (p - 1) + 1 = 0 mod p
        let p_minus_1 = scalar_sub_assuming_ge(&BN254_MODULUS, &SCALAR_ONE);
        let result = scalar_add_mod(p_minus_1, SCALAR_ONE);
        assert_eq!(result, SCALAR_ZERO);

        // (p - 1) + 2 = 1 mod p
        let result = scalar_add_mod(p_minus_1, [2, 0, 0, 0]);
        assert_eq!(result, SCALAR_ONE);
    }

    #[test]
    fn test_mul_large_values() {
        // (p - 1) * (p - 1) mod p
        // Using Fermat: (p-1)^2 = p^2 - 2p + 1 ≡ 1 mod p
        let p_minus_1 = scalar_sub_assuming_ge(&BN254_MODULUS, &SCALAR_ONE);
        let result = scalar_mul_mod(p_minus_1, p_minus_1);
        assert_eq!(result, SCALAR_ONE);

        // 2 * (p - 1) = 2p - 2 ≡ -2 ≡ p - 2 mod p
        let result = scalar_mul_mod([2, 0, 0, 0], p_minus_1);
        let expected = scalar_sub_assuming_ge(&BN254_MODULUS, &[2, 0, 0, 0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_modulus_decimal_string() {
        // Verify modulus converts to the expected decimal string
        let p_str = scalar_to_decimal_string(&BN254_MODULUS);
        assert_eq!(
            p_str,
            "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        );
    }

    #[test]
    fn test_large_value_decimal_string() {
        // p - 1 should be p - 1
        let p_minus_1 = scalar_sub_assuming_ge(&BN254_MODULUS, &SCALAR_ONE);
        let s = scalar_to_decimal_string(&p_minus_1);
        assert_eq!(
            s,
            "21888242871839275222246405745257275088548364400416034343698204186575808495616"
        );
    }
}
