//! Radix-16 signed-digit recoding of the verifier scalars: `s = Σ_w d_w·16^w`
//! with centered digits `d_w ∈ [-8, 8)`, committed as one-hot selectors
//! `j = d + 8 ∈ [0, 16)` per (base, window).

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};

pub const WINDOW: usize = 4;
/// `64 · 4 = 256` bits cover every scalar below `r < 2^254` with a centered top digit.
pub const WINDOWS: usize = 64;
pub const CANDIDATES: usize = 1 << WINDOW;

/// The digit value of selector `j`.
pub const fn digit_value(j: u8) -> i32 {
    j as i32 - 8
}

fn to_bigint(scalar: Fr) -> BigInt {
    BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_be(&scalar.into_bigint().to_bytes_be()),
    )
}

/// Selectors `j_w = d_w + 8`, least significant window first.
pub fn digits(scalar: Fr) -> [u8; WINDOWS] {
    let radix = BigInt::from(1u32 << WINDOW);
    let half = BigInt::from(1u32 << (WINDOW - 1));
    let mut value = to_bigint(scalar);
    let mut out = [0u8; WINDOWS];
    for slot in &mut out {
        let mut residue = &value % &radix;
        if residue.sign() == Sign::Minus {
            residue += &radix;
        }
        let digit = if residue >= half {
            residue - &radix
        } else {
            residue
        };
        let digit = digit
            .to_i32()
            .unwrap_or_else(|| unreachable!("centered digit fits i32"));
        *slot = (digit + 8) as u8;
        value = (value - BigInt::from(digit)) / &radix;
    }
    assert!(value.is_zero(), "scalar exceeds the fixed window count");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn digits_recompose() {
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        for _ in 0..20 {
            let s = Fr::rand(&mut rng);
            let recomposed = digits(s).iter().rev().fold(Fr::from(0u64), |acc, &j| {
                let d = digit_value(j);
                let d = if d < 0 {
                    -Fr::from(d.unsigned_abs() as u64)
                } else {
                    Fr::from(d as u64)
                };
                acc * Fr::from(16u64) + d
            });
            assert_eq!(recomposed, s);
        }
        assert_eq!(digits(Fr::from(0u64)), [8u8; WINDOWS]);
    }
}
