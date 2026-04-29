//! Primitive oracle construction kernels for Bolt-generated Jolt code.
//!
//! This crate is intentionally not a runtime/provider abstraction. Generated
//! code calls these kernels after the Bolt lowering pipeline has made oracle
//! generation explicit in IR.

use jolt_field::Field;

/// Converts an i128 trace column to field elements and pads it to `target_len`.
///
/// The input is normally trace-length data; commitment domains can be larger
/// than the trace domain, so generated code asks for the final committed length.
pub fn dense_i128_column_to_field<F: Field>(values: &[i128], target_len: usize) -> Vec<F> {
    assert!(
        values.len() <= target_len,
        "dense trace column has {} values, target length is {target_len}",
        values.len()
    );
    let mut output: Vec<F> = values.iter().map(|&value| F::from_i128(value)).collect();
    output.resize(target_len, F::zero());
    output
}

/// Pads an optional field-valued oracle to `target_len`.
///
/// `None` stays `None`; zero-skipping policy is deliberately left to the
/// generated commitment code because skip semantics are protocol metadata.
pub fn optional_field_oracle<F: Field>(values: Option<&[F]>, target_len: usize) -> Option<Vec<F>> {
    values.map(|values| pad_field_oracle(values, target_len))
}

/// Pads a field-valued oracle to `target_len`.
pub fn pad_field_oracle<F: Field>(values: &[F], target_len: usize) -> Vec<F> {
    assert!(
        values.len() <= target_len,
        "field oracle has {} values, target length is {target_len}",
        values.len()
    );
    let mut output = values.to_vec();
    output.resize(target_len, F::zero());
    output
}

/// Builds one address-major one-hot chunk polynomial.
///
/// Layout is `output[chunk_value * trace_len + cycle]`. Chunk `0` is the most
/// significant chunk, matching jolt-core's committed RA decomposition.
pub fn one_hot_chunk_address_major<F: Field>(
    values: &[Option<u128>],
    chunk: usize,
    num_chunks: usize,
    chunk_bits: usize,
    trace_len: usize,
    padding_value: Option<u128>,
) -> Vec<F> {
    assert!(
        values.len() <= trace_len,
        "one-hot source has {} values, trace length is {trace_len}",
        values.len()
    );
    assert!(
        chunk < num_chunks,
        "chunk index {chunk} out of bounds for {num_chunks} chunks"
    );
    assert!(
        chunk_bits < usize::BITS as usize,
        "chunk_bits must fit in usize"
    );

    let chunk_domain = 1usize << chunk_bits;
    let shift = chunk_bits * (num_chunks - 1 - chunk);
    let mask = (chunk_domain - 1) as u128;
    let mut output = vec![F::zero(); chunk_domain * trace_len];

    for cycle in 0..trace_len {
        let value = values.get(cycle).copied().flatten().or(padding_value);
        if let Some(value) = value {
            let index = ((value >> shift) & mask) as usize;
            output[index * trace_len + cycle] = F::one();
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    #[test]
    fn dense_column_converts_and_pads() {
        let output = dense_i128_column_to_field::<Fr>(&[5, -3], 4);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], Fr::from_i128(5));
        assert_eq!(output[1], Fr::from_i128(-3));
        assert_eq!(output[2], Fr::from_u64(0));
        assert_eq!(output[3], Fr::from_u64(0));
    }

    #[test]
    fn one_hot_chunks_are_address_major_and_msb_first() {
        let values = [Some(0xABu128), Some(0x12), None];
        let output = one_hot_chunk_address_major::<Fr>(&values, 0, 2, 4, 4, Some(0));

        assert_eq!(output.len(), 16 * 4);
        assert_eq!(output[0xA * 4], Fr::from_u64(1));
        assert_eq!(output[5], Fr::from_u64(1));
        assert_eq!(output[2], Fr::from_u64(1));
        assert_eq!(output[3], Fr::from_u64(1));
    }

    #[test]
    fn one_hot_none_padding_skips_entries() {
        let values = [Some(3u128), None];
        let output = one_hot_chunk_address_major::<Fr>(&values, 0, 1, 2, 3, None);

        assert_eq!(output[3 * 3], Fr::from_u64(1));
        assert!(output
            .iter()
            .enumerate()
            .all(|(index, value)| index == 9 || *value == Fr::from_u64(0)));
    }
}
