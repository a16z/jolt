use crate::field::JoltField;

/// Asserts that `C * log_M` is at least as large as the word size.
pub fn assert_valid_parameters(word_size: usize, C: usize, log_M: usize) {
    assert!(C * log_M >= word_size);
}

/// Concatenates a slice of field elements into a single field element.
///
/// # Arguments
///
/// * `vals` - A slice of `C` field elements, each assumed to be an `operand_bits`-bit number.
/// * `C` - The number of field elements in `vals`.
/// * `operand_bits` - The number of bits required to represent each element in `vals`.
///
/// # Notes
///
/// If an element of `vals` is larger than `operand_bits`, it will not be truncated.
/// This behavior is commonly used by the collation functions of instructions.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::concatenate_lookups;
/// use ark_bn254::Fr;
///
/// let vals = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
/// let result = concatenate_lookups(&vals, 3, 2);
/// assert_eq!(result, Fr::from(0b01_10_11));
/// ```
pub fn concatenate_lookups<F: JoltField>(vals: &[F], C: usize, operand_bits: usize) -> F {
    assert_eq!(vals.len(), C);

    let mut sum = F::zero();
    let mut weight = F::one();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    for i in 0..C {
        sum += weight * vals[C - i - 1];
        weight *= shift;
    }
    sum
}

/// Splits a 64-bit unsigned integer `x` into a `C`-length vector of `u64`, each representing a
/// `chunk_len`-bit chunk.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::chunk_operand;
///
/// // Normal usage
/// let x = 0b1100_1010_1111_0000;
/// assert_eq!(chunk_operand(x, 4, 4), vec![12, 10, 15, 0]);
/// // Edge cases
/// // More chunks than bits in x
/// assert_eq!(chunk_operand(0xFF, 5, 2), vec![0, 3, 3, 3, 3]);
/// // Fewer chunks * chunk_len than bits in x (remaining bits discarded)
/// assert_eq!(chunk_operand(0xFFF, 2, 4), vec![15, 15]);
/// ```
pub fn chunk_operand(x: u64, C: usize, chunk_len: usize) -> Vec<u64> {
    let bit_mask = (1 << chunk_len) - 1;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * chunk_len) as u32;
            x.checked_shr(shift).unwrap_or(0) & bit_mask
        })
        .collect()
}

/// Splits a 64-bit unsigned integer `x` into a `C`-length vector of `usize`, each representing a
/// `chunk_len`-bit chunk. Only different from `chunk_operand` in that it returns `usize` instead of
/// `u64`.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::chunk_operand_usize;
///
/// // Normal usage
/// let x = 0b1100_1010_1111_0000;
/// assert_eq!(chunk_operand_usize(x, 4, 4), vec![12, 10, 15, 0]);
/// // Edge cases
/// // More chunks than bits in x
/// assert_eq!(chunk_operand_usize(0xFF, 5, 2), vec![0, 3, 3, 3, 3]);
/// // Fewer chunks * chunk_len than bits in x (remaining bits discarded)
/// assert_eq!(chunk_operand_usize(0xFFF, 2, 4), vec![15, 15]);
/// ```
pub fn chunk_operand_usize(x: u64, C: usize, chunk_len: usize) -> Vec<usize> {
    let bit_mask = (1 << chunk_len) - 1;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * chunk_len) as u32;
            (x.checked_shr(shift).unwrap_or(0) & bit_mask) as usize
        })
        .collect()
}

/// Chunks and concatenates two 64-bit unsigned integers `x` and `y` into a `C`-length vector of `usize`,
/// where each element represents a `log_M`-bit chunk that is the concatenation of `log_M / 2`-bit
/// chunks from each of `x` and `y`.
///
/// # Arguments
///
/// * `x` - The first 64-bit unsigned integer to chunk and concatenate.
/// * `y` - The second 64-bit unsigned integer to chunk and concatenate.
/// * `C` - The number of chunks to produce.
/// * `log_M` - The number of bits in each resulting chunk.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::chunk_and_concatenate_operands;
///
/// // Normal usage
/// let x = 0b1100_1010;
/// let y = 0b1111_0000;
/// assert_eq!(chunk_and_concatenate_operands(x, y, 2, 8), vec![0b1100_1111, 0b1010_0000]);
///
/// let x = 0b11_00_11;
/// let y = 0b00_11_00;
/// assert_eq!(chunk_and_concatenate_operands(x, y, 3, 4), vec![0b1100, 0b0011, 0b1100]);
///
/// // More chunks than bits in x | y
/// assert_eq!(chunk_and_concatenate_operands(0b11, 0b11, 5, 2), vec![0, 0, 0, 3, 3]);
///
/// // Fewer chunks than bits in x | y (remaining bits discarded)
/// assert_eq!(chunk_and_concatenate_operands(0xFF, 0xFF, 2, 4), vec![15, 15]);
/// ```
pub fn chunk_and_concatenate_operands(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;

    #[cfg(test)]
    {
        let max_operand_bits = C * log_M / 2;
        if max_operand_bits != 64 {
            // if 64, handled by normal overflow checking
            let max_operand: u64 = (1 << max_operand_bits) - 1;
            assert!(x <= max_operand);
            assert!(y <= max_operand);
        }
    }

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

/// Chunks `z` into `C` chunks bitwise where `z = x + y`.
/// `log_M` is the number of bits for each of the `C` chunks of `z`.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::add_and_chunk_operands;
///
/// // Normal usage
/// assert_eq!(add_and_chunk_operands(12, 3, 2, 2), vec![0b11, 0b11]);
///
/// // More chunks than bits in x | y
/// assert_eq!(add_and_chunk_operands(3, 4, 2, 2), vec![0b01, 0b11]);
///
/// // Fewer chunks than bits in x | y (remaining bits discarded)
/// assert_eq!(add_and_chunk_operands(31, 31, 3, 2), vec![0b11, 0b11, 0b10]);
/// ```
pub fn add_and_chunk_operands(x: u128, y: u128, C: usize, log_M: usize) -> Vec<usize> {
    let sum_chunk_bits: usize = log_M;
    let sum_chunk_bit_mask: usize = (1 << sum_chunk_bits) - 1;
    let z: u128 = x + y;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * sum_chunk_bits) as u32;

            z.checked_shr(shift).unwrap_or(0) as usize & sum_chunk_bit_mask
        })
        .collect()
}

/// Chunks `z` into `C` chunks bitwise where `z = x * y`.
/// `log_M` is the number of bits for each of the `C` chunks of `z`.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::multiply_and_chunk_operands;
///
/// // Normal usage
/// assert_eq!(multiply_and_chunk_operands(5, 3, 2, 2), vec![0b11, 0b11]);
///
/// // More chunks than bits in x | y
/// assert_eq!(multiply_and_chunk_operands(7, 1, 2, 2), vec![0b01, 0b11]);
///
/// // Fewer chunks than bits in x | y (remaining bits discarded)
/// assert_eq!(multiply_and_chunk_operands(2, 31, 3, 2), vec![0b11, 0b11, 0b10]);
/// ```
pub fn multiply_and_chunk_operands(x: u128, y: u128, C: usize, log_M: usize) -> Vec<usize> {
    let product_chunk_bits: usize = log_M;
    let product_chunk_bit_mask: usize = (1 << product_chunk_bits) - 1;
    let z: u128 = x * y;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * product_chunk_bits) as u32;
            z.checked_shr(shift).unwrap_or(0) as usize & product_chunk_bit_mask
        })
        .collect()
}

/// Chunks and concatenates two 64-bit unsigned integers `x` and `y` into a vector of concatenated chunks,
/// where the second half of each concatenated chunk is always `y_0`, the last chunk of `y` (from left to right).
///
/// # Arguments
///
/// * `x` - The first 64-bit unsigned integer to chunk.
/// * `y` - The second 64-bit unsigned integer to chunk.
/// * `C` - The number of chunks to produce.
/// * `log_M` - The number of bits in each resulting concatenated chunk.
///
/// # Result
/// [ x_{C-1} || y_0, ..., x_0 || y_0 ],
/// where `x_{C-1}` is the big end of `x`, and `y_0` is the small end of `y`.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::instruction_utils::chunk_and_concatenate_for_shift;
///
/// // Normal usage
/// assert_eq!(chunk_and_concatenate_for_shift(0b1001, 0b0010, 2, 4), vec![0b1010, 0b0110]);
///
/// // More chunks than bits in x | y
/// assert_eq!(chunk_and_concatenate_for_shift(0b10_11, 0b00_01, 3, 4), vec![0b00_01, 0b10_01, 0b11_01]);
///
/// // Fewer chunks than bits in x | y (chunks larger than `C` discarded)
/// assert_eq!(chunk_and_concatenate_for_shift(0b10_01_11, 0b00_00_01, 2, 4), vec![0b01_01, 0b11_01]);
/// ```
pub fn chunk_and_concatenate_for_shift(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;
    let operand_bit_mask: usize = (1 << operand_bits) - 1;

    let y_lowest_chunk: usize = y as usize & operand_bit_mask;

    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * operand_bits) as u32;
            let left = x.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
            (left << operand_bits) | y_lowest_chunk
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn concatenate_lookups_test() {
        let vals = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let concat = concatenate_lookups(&vals, 3, 2);
        assert_eq!(concat, Fr::from(0b01_10_11));

        let vals = vec![Fr::from(7), Fr::from(1), Fr::from(2), Fr::from(3)];
        let concat = concatenate_lookups(&vals, 4, 3);
        assert_eq!(concat, Fr::from(0b111_001_010_011));
    }

    #[test]
    fn chunk_and_concatenate_operands_test() {
        let chunks = chunk_and_concatenate_operands(0b11, 0b10, 2, 2);
        assert_eq!(chunks, vec![0b11, 0b10]);

        let chunks = chunk_and_concatenate_operands(0b11_00_11, 0b10_01_10, 3, 4);
        assert_eq!(chunks, vec![0b11_10, 0b00_01, 0b_11_10]);
    }

    #[test]
    #[should_panic]
    fn chunk_and_concatenate_operands_too_large() {
        // Fail to split 2 integers of length 3-bits in to 2 chunks of length 2-bits
        chunk_and_concatenate_operands(0b111, 0b101, 2, 2);
    }

    #[test]
    #[should_panic]
    fn chunk_and_concatenate_operands_too_large_2() {
        // Fail to split 2 integers of length 6-bits into 3 chunks of length 3-bits
        chunk_and_concatenate_operands(0b11_11_11, 0b10_10_10, 3, 3);
    }

    #[test]
    fn add_and_chunk_operands_test() {
        // x = 0b0011
        // y = 0b1100
        // z = 0b1111
        let chunks = add_and_chunk_operands(0b0011, 0b1100, 2, 2);
        assert_eq!(chunks, vec![0b11, 0b11]);

        // x = 20
        // y = 30
        // z = 50 = 0b11_00_10
        let chunks = add_and_chunk_operands(20u128, 30u128, 3, 2);
        assert_eq!(chunks, vec![0b11, 0b00, 0b10]);
    }

    #[test]
    fn chunk_and_concatenate_for_shift_test() {
        let x = 0b1001;
        let y = 2;
        let C = 2;
        let log_M = 4;
        let chunks = chunk_and_concatenate_for_shift(x, y, C, log_M);
        assert_eq!(chunks.len(), 2);

        // 2-bit operands
        // x_0 | y = 0b10 | 0b10
        assert_eq!(chunks[0], 0b10_10);
        // x_1 | y = 0b01 | 0b10
        assert_eq!(chunks[1], 0b01_10);
    }

    #[test]
    fn chunk_and_concatenate_for_shift_test_larger() {
        let x = 0b10_01_11;
        let y = 0b00_00_01;
        let C = 3;
        let log_M = 4;
        let chunks = chunk_and_concatenate_for_shift(x, y, C, log_M);
        assert_eq!(chunks.len(), 3);

        // 2-bit operands
        // x_0 | y = 0b10 | 0b01
        assert_eq!(chunks[0], 0b10_01);
        // x_1 | y = 0b01 | 0b01
        assert_eq!(chunks[1], 0b01_01);
        // x_2 | y = 0b11 | 0b01
        assert_eq!(chunks[2], 0b11_01);
    }
}
