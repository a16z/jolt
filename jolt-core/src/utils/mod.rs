use crate::field::{ChallengeFieldOps, JoltField};

use rayon::prelude::*;

pub mod accumulation;
pub mod counters;
pub mod errors;
pub mod expanding_table;
pub mod gaussian_elimination;
pub mod lookup_bits;
pub mod math;
#[cfg(feature = "monitor")]
pub mod monitor;
pub mod profiling;
pub mod small_scalar;
pub mod thread;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use jolt_core::utils::index_to_field_bitvector;
/// # use ark_bn254::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: JoltField + ChallengeFieldOps<F>>(
    value: u128,
    bits: usize,
) -> Vec<F> {
    if bits != 128 {
        assert!(value < 1u128 << bits);
    }

    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: JoltField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

#[inline(always)]
pub fn mul_0_optimized<F: JoltField>(likely_zero: &F, x: &F) -> F {
    if likely_zero.is_zero() {
        F::zero()
    } else {
        *likely_zero * *x
    }
}

/// Splits a 128-bit value into two 64-bit values by separating even and odd bits.
/// The even bits (indices 0,2,4,...) go into the first returned value, and odd bits (indices 1,3,5,...) into the second.
///
/// # Arguments
///
/// * `val` - The 128-bit input value to split
///
/// # Returns
///
/// A tuple (x, y) where:
/// - x contains the bits from even indices (0,2,4,...) compacted into the low 64 bits
/// - y contains the bits from odd indices (1,3,5,...) compacted into the low 64 bits
pub fn uninterleave_bits(val: u128) -> (u64, u64) {
    // Isolate even and odd bits.
    let mut x_bits = (val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    let mut y_bits = val & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    // Compact the bits into the lower part of `x_bits`
    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    // And do the same for `y_bits`
    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    (x_bits as u64, y_bits as u64)
}

/// Combines two 64-bit values into a single 128-bit value by interleaving their bits.
/// Takes even bits from the first argument and odd bits from the second argument.
///
/// # Arguments
///
/// * `even_bits` - A 64-bit value whose bits will be placed at even indices (0,2,4,...)
/// * `odd_bits` - A 64-bit value whose bits will be placed at odd indices (1,3,5,...)
///
/// # Returns
///
/// A 128-bit value containing interleaved bits from the input values, with even_bits shifted into even positions
/// and odd_bits in odd positions.
///
/// # Examples
///
/// ```
/// # use jolt_core::utils::interleave_bits;
/// assert_eq!(interleave_bits(0b01, 0b10), 0b110);
/// ```
pub fn interleave_bits(even_bits: u64, odd_bits: u64) -> u128 {
    // Insert zeros between each bit of `x_bits`
    let mut x_bits = even_bits as u128;
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // And do the same for `y_bits`
    let mut y_bits = odd_bits as u128;
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    (x_bits << 1) | y_bits
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use rand_core::RngCore;

    use super::*;

    #[test]
    fn interleave_uninterleave_bits() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let val = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;
            let (even, odd) = uninterleave_bits(val);
            assert_eq!(val, interleave_bits(even, odd));
        }

        for _ in 0..1000 {
            let even = rng.next_u64();
            let odd = rng.next_u64();
            assert_eq!((even, odd), uninterleave_bits(interleave_bits(even, odd)));
        }
    }
}
