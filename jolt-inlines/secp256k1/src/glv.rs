/// GLV decomposition for secp256k1 scalar field.
///
/// Given scalar k in Fr, compute k1, k2 such that k = k1 + k2 * lambda (mod r)
/// with |k1|, |k2| <= 2^128. Based on the algorithm from arkworks ec/src/scalar_mul/glv.rs.
use num_bigint::BigInt as NBigInt;
use num_bigint::Sign;
use num_integer::Integer;

fn r() -> NBigInt {
    NBigInt::from_bytes_le(
        Sign::Plus,
        &[
            65, 65, 54, 208, 140, 94, 210, 191, 59, 160, 72, 175, 230, 220, 174, 186, 254, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ],
    )
}

fn a1() -> NBigInt {
    NBigInt::from_bytes_le(
        Sign::Plus,
        &[
            21, 235, 132, 146, 228, 144, 108, 232, 205, 107, 212, 167, 33, 210, 134, 48,
        ],
    )
}

fn b1() -> NBigInt {
    NBigInt::from_bytes_le(
        Sign::Plus,
        &[
            195, 228, 191, 10, 169, 127, 84, 111, 40, 136, 14, 1, 214, 126, 67, 228,
        ],
    )
}

fn a2() -> NBigInt {
    NBigInt::from_bytes_le(
        Sign::Plus,
        &[
            216, 207, 68, 157, 141, 16, 193, 87, 246, 243, 226, 168, 247, 80, 202, 20, 1,
        ],
    )
}

fn rounded_div(k: &NBigInt, coeff: &NBigInt, r: &NBigInt) -> NBigInt {
    let (mut div, rem) = (k * coeff).div_rem(r);
    if (&rem + &rem) > *r {
        div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
    }
    div
}

/// Decompose scalar k into (sign1, |k1|) and (sign2, |k2|) where k = k1 + k2*lambda (mod r).
/// sign is `true` for negative.
pub fn decompose_scalar(k: NBigInt) -> [(bool, u128); 2] {
    let r = r();
    let a1 = a1();
    let b1 = b1();
    let a2 = a2();

    let beta_1 = rounded_div(&k, &a1, &r);
    let beta_2 = rounded_div(&k, &b1, &r);

    let k1 = &k - &beta_1 * &a1 - &beta_2 * &a2;
    let k2 = &beta_1 * &b1 - &beta_2 * &a1;

    [to_sign_abs(k1), to_sign_abs(k2)]
}

fn to_sign_abs(n: NBigInt) -> (bool, u128) {
    let (sign, bytes) = n.to_bytes_le();
    let mut padded = [0u8; 16];
    let len = bytes.len().min(16);
    padded[..len].copy_from_slice(&bytes[..len]);
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
