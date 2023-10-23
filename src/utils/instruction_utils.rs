use ark_ff::PrimeField;
use ark_std::log2;

pub fn concatenate_lookups<F: PrimeField>(vals: &[F], C: usize, M: usize) -> F {
  assert_eq!(vals.len(), C);

  let mut sum = F::zero();
  let mut weight = F::one();
  let shift = F::from(1u64 << (log2(M) / 2));
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
      let left = (x as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
      let right = (y as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
      (left << operand_bits) | right
    })
    .collect()
}