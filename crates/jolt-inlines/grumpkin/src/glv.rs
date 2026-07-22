/// GLV decomposition for the Grumpkin scalar field.
///
/// Given scalar k in Fr, compute k1, k2 such that k = k1 + k2 * lambda (mod r)
/// with |k1|, |k2| <= 2^128. Based on arkworks' GLV decomposition.
use ark_ff::{BigInteger, PrimeField};
use ark_grumpkin::Fr;
use num_bigint::BigInt as NBigInt;
use num_bigint::Sign;
use num_integer::Integer;

fn scalar_modulus() -> NBigInt {
    NBigInt::from_bytes_le(Sign::Plus, &Fr::MODULUS.to_bytes_le())
}

fn rounded_div(k: &NBigInt, coeff: &NBigInt, modulus: &NBigInt) -> NBigInt {
    let (mut div, rem) = (k * coeff).div_rem(modulus);
    if (&rem + &rem) > *modulus {
        div += NBigInt::from(1u8);
    }
    div
}

/// Decompose scalar k into (sign1, |k1|) and (sign2, |k2|) where
/// k = k1 + k2 * lambda (mod r). Sign is `true` for negative.
pub fn decompose_scalar(k: NBigInt) -> [(bool, u128); 2] {
    let modulus = scalar_modulus();
    let n11 = NBigInt::from(147946756881789319000765030803803410729i128);
    let n12 = NBigInt::from(-9931322734385697762i128);
    let n21 = NBigInt::from(9931322734385697762i128);
    let n22 = NBigInt::from(147946756881789319010696353538189108491i128);

    let beta_1 = rounded_div(&k, &n22, &modulus);
    let beta_2 = rounded_div(&k, &(-n12.clone()), &modulus);
    let k1 = &k - &beta_1 * &n11 - &beta_2 * &n21;
    let k2 = -(&beta_1 * &n12 + &beta_2 * &n22);

    [to_sign_abs(k1), to_sign_abs(k2)]
}

fn to_sign_abs(n: NBigInt) -> (bool, u128) {
    let (sign, bytes) = n.to_bytes_le();
    assert!(
        bytes.len() <= 16,
        "GLV decomposition produced out-of-range half-scalar"
    );
    let mut padded = [0u8; 16];
    padded[..bytes.len()].copy_from_slice(&bytes);
    (sign == Sign::Minus, u128::from_le_bytes(padded))
}

/// Serialize decomposition as 6 u64 values: [sign1, k1_lo, k1_hi, sign2, k2_lo, k2_hi].
pub fn decompose_scalar_to_u64s(k: NBigInt) -> [u64; 6] {
    let [(s1, k1), (s2, k2)] = decompose_scalar(k);
    [
        s1 as u64,
        k1 as u64,
        (k1 >> 64) as u64,
        s2 as u64,
        k2 as u64,
        (k2 >> 64) as u64,
    ]
}
