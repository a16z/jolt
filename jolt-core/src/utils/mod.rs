#![allow(dead_code)]
use crate::field::JoltField;

use ark_std::test_rng;
use rayon::prelude::*;

pub mod errors;
pub mod gaussian_elimination;
pub mod instruction_utils;
pub mod math;
pub mod profiling;
pub mod sol_types;
pub mod thread;
pub mod transcript;

/// Macros that determine the optimal iterator type based on the feature flags.
///
/// For some cases (ex. offloading to GPU), we may not want to use a parallel iterator.
/// Specifically when icicle is enabled we want to be careful to use serial iteration in the right places.
/// Based on observations; multiple calls into icicle_msm functions can dramatically slow down GPU performance.
#[macro_export]
macro_rules! optimal_iter {
    ($T:expr) => {{
        #[cfg(feature = "icicle")]
        {
            $T.iter()
        }
        #[cfg(not(feature = "icicle"))]
        {
            $T.par_iter()
        }
    }};
}

#[macro_export]
macro_rules! into_optimal_iter {
    ($T:expr) => {{
        #[cfg(feature = "icicle")]
        {
            $T.into_iter()
        }
        #[cfg(not(feature = "icicle"))]
        {
            $T.into_par_iter()
        }
    }};
}

#[macro_export]
macro_rules! optimal_iter_mut {
    ($T:expr) => {{
        #[cfg(feature = "icicle")]
        {
            $T.iter_mut()
        }
        #[cfg(not(feature = "icicle"))]
        {
            $T.par_iter_mut()
        }
    }};
}

#[macro_export]
macro_rules! join_conditional {
    ($f1:expr, $f2:expr) => {{
        #[cfg(feature = "icicle")]
        {
            ($f1(), $f2())
        }
        #[cfg(not(feature = "icicle"))]
        {
            rayon::join($f1, $f2)
        }
    }};
}

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
pub fn index_to_field_bitvector<F: JoltField>(value: u64, bits: usize) -> Vec<F> {
    assert!((value as u128) < 1 << bits);

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

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && num.is_power_of_two()
}

/// Take the first two `num_bits` chunks of `item` (from the right / LSB) and return them as a tuple `(high_chunk, low_chunk)`.
///
/// If `item` is shorter than `2 * num_bits`, the remaining bits are zero-padded.
///
/// If `item` is longer than `2 * num_bits`, the remaining bits are discarded.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::split_bits;
///
/// assert_eq!(split_bits(0b101000, 2), (0b10, 0b00));
/// assert_eq!(split_bits(0b101000, 3), (0b101, 0b000));
/// assert_eq!(split_bits(0b101000, 4), (0b0010, 0b1000));
/// ```
pub fn split_bits(item: usize, num_bits: usize) -> (usize, usize) {
    let max_value = (1 << num_bits) - 1; // Calculate the maximum value that can be represented with num_bits

    let low_chunk = item & max_value; // Extract the lower bits
    let high_chunk = (item >> num_bits) & max_value; // Shift the item to the right and extract the next set of bits

    (high_chunk, low_chunk)
}

/// Generate a random point with `memory_bits` field elements.
pub fn gen_random_point<F: JoltField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
        r_i.push(F::random(&mut rng));
    }
    r_i
}

/// Splits a 64-bit value into two 32-bit values by separating even and odd bits.
/// The even bits (indices 0,2,4,...) go into the first returned value, and odd bits (indices 1,3,5,...) into the second.
///
/// # Arguments
///
/// * `val` - The 64-bit input value to split
///
/// # Returns
///
/// A tuple (x, y) where:
/// - x contains the bits from even indices (0,2,4,...) compacted into the low 32 bits
/// - y contains the bits from odd indices (1,3,5,...) compacted into the low 32 bits
pub fn uninterleave_bits(val: u64) -> (u32, u32) {
    // Isolate even and odd bits.
    let mut x_bits = (val >> 1) & 0x5555_5555_5555_5555;
    let mut y_bits = val & 0x5555_5555_5555_5555;

    // Compact the bits into the lower part of `x_bits`
    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF;

    // And do the same for `y_bits`
    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF;

    (x_bits as u32, y_bits as u32)
}

/// Combines two 32-bit values into a single 64-bit value by interleaving their bits.
/// Takes even bits from the first argument and odd bits from the second argument.
///
/// # Arguments
///
/// * `even_bits` - A 32-bit value whose bits will be placed at even indices (0,2,4,...)
/// * `odd_bits` - A 32-bit value whose bits will be placed at odd indices (1,3,5,...)
///
/// # Returns
///
/// A 64-bit value containing interleaved bits from the input values, with even_bits shifted into even positions
/// and odd_bits in odd positions.
///
/// # Examples
///
/// ```
/// # use jolt_core::utils::interleave_bits;
/// assert_eq!(interleave_bits(0b01, 0b10), 0b100);
/// ```
pub fn interleave_bits(even_bits: u32, odd_bits: u32) -> u64 {
    // Insert zeros between each bit of `x_bits`
    let mut x_bits = even_bits as u64;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555;

    // And do the same for `y_bits`
    let mut y_bits = odd_bits as u64;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555;

    (x_bits << 1) | y_bits
}

#[cfg(test)]
mod tests {
    use rand_core::RngCore;

    use super::*;

    #[test]
    fn interleave_uninterleave_bits() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let val = rng.next_u64();
            let (even, odd) = uninterleave_bits(val);
            assert_eq!(val, interleave_bits(even, odd));
        }

        for _ in 0..1000 {
            let even = rng.next_u32();
            let odd = rng.next_u32();
            assert_eq!((even, odd), uninterleave_bits(interleave_bits(even, odd)));
        }
    }

    #[test]
    fn split() {
        assert_eq!(split_bits(0b00_01, 2), (0, 1));
        assert_eq!(split_bits(0b10_01, 2), (2, 1));
    }
}
