#![allow(dead_code)]

use ark_ff::PrimeField;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

#[cfg(test)]
pub mod test;

pub mod errors;
pub mod gaussian_elimination;
pub mod math;
pub mod random;
pub mod transcript;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use liblasso::utils::index_to_field_bitvector;
/// # use ark_bls12_381::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: PrimeField>(value: usize, bits: usize) -> Vec<F> {
  assert!(value < 1 << bits);

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

/// Convert Vec<F> which should represent a bitvector to a packed string of bits {0, 1, ?}
pub fn ff_bitvector_dbg<F: PrimeField>(f: &Vec<F>) -> String {
  let mut result = "".to_owned();
  for bit in f {
    if *bit == F::one() {
      result.push('1');
    } else if *bit == F::zero() {
      result.push('0');
    } else {
      result.push('?');
    }
  }
  result
}

#[tracing::instrument(skip_all, name = "compute_dotproduct")]
pub fn compute_dotproduct<F: PrimeField>(a: &[F], b: &[F]) -> F {
  assert_eq!(a.len(), b.len());

  #[cfg(feature = "multicore")]
  let dot_product = (0..a.len()).into_par_iter().map(|i| a[i] * b[i]).sum();
  #[cfg(not(feature = "multicore"))]
  let dot_product = (0..a.len()).map(|i| a[i] * b[i]).sum();

  dot_product
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
  num != 0 && (num & (num - 1)) == 0
}

/// Splits `item` into two chunks of `num_bits` size where each is less than 2^num_bits.
/// Ex: split_bits(0b101_000, 3) -> (101, 000)
pub fn split_bits(item: usize, num_bits: usize) -> (usize, usize) {
  let max_value = (1 << num_bits) - 1; // Calculate the maximum value that can be represented with num_bits

  let low_chunk = item & max_value; // Extract the lower bits
  let high_chunk = (item >> num_bits) & max_value; // Shift the item to the right and extract the next set of bits

  (high_chunk, low_chunk)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn split() {
    assert_eq!(split_bits(0b00_01, 2), (0, 1));
    assert_eq!(split_bits(0b10_01, 2), (2, 1));
  }
}
