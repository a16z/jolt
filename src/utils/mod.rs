use ark_ff::PrimeField;

#[cfg(test)]
pub mod test;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```
/// use libspartan::utils::index_to_field_bitvector;
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
      result.push_str("1");
    } else if *bit == F::zero() {
      result.push_str("0");
    } else {
      result.push_str("?");
    }
  }
  result
}

pub fn compute_dotproduct<F: PrimeField>(a: &[F], b: &[F]) -> F {
  assert_eq!(a.len(), b.len());
  (0..a.len()).map(|i| a[i] * b[i]).sum()
}
