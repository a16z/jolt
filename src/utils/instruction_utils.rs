use ark_ff::PrimeField;
use ark_std::log2;

pub fn concatenate_lookups<F: PrimeField>(vals: &[F], C: usize, shift_bits: usize) -> F {
  assert_eq!(vals.len(), C);

  let mut sum = F::zero();
  let mut weight = F::one();
  let shift = F::from(1u64 << shift_bits);
  for i in 0..C {
    sum += weight * vals[C - i - 1];
    weight *= shift;
  }
  sum
}

pub fn chunk_and_concatenate_operands(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
  let operand_bits: usize = log_M / 2;
  let operand_bit_mask: usize = (1 << operand_bits) - 1;
  (0..C)
    .map(|i| {
      let shift = ((C - i - 1) * operand_bits) as u32;
      let left = x.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
      let right = y.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
      (left << operand_bits) | right
    })
    .collect()
}

pub fn add_and_chunk_operands(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
  let sum_chunk_bits: usize = log_M;
  let sum_chunk_bit_mask: usize = (1 << sum_chunk_bits) - 1;
  let z: u128 = (x as u128) + (y as u128);
  (0..C)
    .map(|i| {
      let shift = ((C - i - 1) * sum_chunk_bits) as u32;
      let chunk = z.checked_shr(shift).unwrap_or(0) as usize & sum_chunk_bit_mask;
      chunk
    })
    .collect()
}
