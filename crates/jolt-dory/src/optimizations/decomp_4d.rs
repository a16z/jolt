use super::constants::POWER_OF_2_DECOMPOSITIONS;
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, Sign};

fn fr_to_bigint(fr: Fr) -> BigInt {
    let bytes = fr.into_bigint().to_bytes_be();
    BigInt::from_bytes_be(Sign::Plus, &bytes)
}

/// Table-based 4-dimensional scalar decomposition.
/// Decomposes k into (k0, k1, k2, k3) such that
/// k ≡ k0 + k1·λ + k2·λ² + k3·λ³ (mod r).
/// Each coefficient is at most ~66 bits.
fn decompose_scalar_table_based(scalar: &BigInt) -> ([u128; 4], [bool; 4]) {
    let mut k0 = 0u128;
    let mut k1 = 0u128;
    let mut k2 = 0u128;
    let mut k3 = 0u128;

    let mut temp_scalar = scalar.clone();
    let mut bit_position = 0;

    while temp_scalar > BigInt::from(0) && bit_position < 254 {
        if &temp_scalar & BigInt::from(1) == BigInt::from(1) {
            let (decomp_k0, decomp_k1, decomp_k2, decomp_k3, neg0, neg1, neg2, neg3) =
                POWER_OF_2_DECOMPOSITIONS[bit_position];

            if neg0 {
                k0 = k0.wrapping_sub(decomp_k0);
            } else {
                k0 = k0.wrapping_add(decomp_k0);
            }
            if neg1 {
                k1 = k1.wrapping_sub(decomp_k1);
            } else {
                k1 = k1.wrapping_add(decomp_k1);
            }
            if neg2 {
                k2 = k2.wrapping_sub(decomp_k2);
            } else {
                k2 = k2.wrapping_add(decomp_k2);
            }
            if neg3 {
                k3 = k3.wrapping_sub(decomp_k3);
            } else {
                k3 = k3.wrapping_add(decomp_k3);
            }
        }

        temp_scalar >>= 1;
        bit_position += 1;
    }

    let (final_k0, neg_flag0) = if (k0 as i128) < 0 {
        (k0.wrapping_neg(), true)
    } else {
        (k0, false)
    };

    let (final_k1, neg_flag1) = if (k1 as i128) < 0 {
        (k1.wrapping_neg(), true)
    } else {
        (k1, false)
    };

    let (final_k2, neg_flag2) = if (k2 as i128) < 0 {
        (k2.wrapping_neg(), true)
    } else {
        (k2, false)
    };

    let (final_k3, neg_flag3) = if (k3 as i128) < 0 {
        (k3.wrapping_neg(), true)
    } else {
        (k3, false)
    };

    (
        [final_k0, final_k1, final_k2, final_k3],
        [neg_flag0, neg_flag1, neg_flag2, neg_flag3],
    )
}

/// Decompose a BN254 scalar for 4D GLV multiplication (G2).
/// Returns coefficients as `BigInt` and their sign flags.
pub fn decompose_scalar_4d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs_u128, signs) = decompose_scalar_table_based(&scalar_bigint);

    let coeffs = [
        Fr::from(coeffs_u128[0]).into_bigint(),
        Fr::from(coeffs_u128[1]).into_bigint(),
        Fr::from(coeffs_u128[2]).into_bigint(),
        Fr::from(coeffs_u128[3]).into_bigint(),
    ];

    (coeffs, signs)
}
