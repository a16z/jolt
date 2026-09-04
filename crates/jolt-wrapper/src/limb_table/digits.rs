//! Radix-16 signed-digit recoding of the verifier scalars: `s = Σ_w d_w·16^w`
//! with centered digits `d_w ∈ [-8, 8)`, committed as one-hot selectors
//! `j = d + 8 ∈ [0, 16)` per (base, window).

use ark_bn254::{Fq, Fr};
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};

pub const WINDOW: usize = 4;
/// `64 · 4 = 256` bits cover every scalar below `r < 2^254` with a centered top digit.
pub const WINDOWS: usize = 64;
/// Top digits the recoding window check gathers: `V_hi = Σ_{i=48}^{63} 16^{i−48}·d_i`.
pub const WINDOW_TOP_DIGITS: usize = 16;
/// The scalar modulus' top 64 bits, `r >> 192`.
pub const R_HI: u64 = 0x3064_4e72_e131_a029;
/// Largest admitted `V_hi`. With `|Σ_{i<48} 16^i·d_i| < 2^192·8/15`, every
/// recoding with `0 ≤ V_hi ≤ R_HI − 2` represents an integer in
/// `[−2^192·8/15, (R_HI − 2)·2^192 + 2^192·8/15]`, an interval shorter than
/// `r`, so each residue class has at most one admitted recoding: an
/// occurrence's digit string is a function of its scalar. The canonical
/// recoding of an honest scalar `s < r` fails only when its top window
/// exceeds the bound, i.e. `s > (R_HI − 2)·2^192 + 2^192·7/15`: fewer than
/// `3/R_HI ≈ 2^−60` of the scalars (no witness; the prover's saturated window
/// row fails the link).
pub const WINDOW_BOUND: u64 = R_HI - 2;
/// Window rows per layout: one per link occurrence, in a `256`-row block
/// (rows without an occurrence hold `V_hi = 0`).
pub const WINDOW_ROWS: usize = 256;

/// `V_hi` of a recoding from its window digits (`window_digits[w]` is the
/// digit processed in window `w`, the most significant first).
pub fn window_value(window_digits: &[i64]) -> i64 {
    window_digits
        .iter()
        .take(WINDOW_TOP_DIGITS)
        .fold(0, |acc, d| 16 * acc + d)
}

/// The window row's value `V_hi + 2^64·(WINDOW_BOUND − V_hi)`; a `V_hi`
/// outside `0..=WINDOW_BOUND` saturates (no witness: the link rejects).
pub fn window_row_value(v_hi: i64) -> Fq {
    let v_hi = u64::try_from(v_hi).unwrap_or(0).min(WINDOW_BOUND);
    Fq::from(u128::from(v_hi) | (u128::from(WINDOW_BOUND - v_hi) << 64))
}
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
