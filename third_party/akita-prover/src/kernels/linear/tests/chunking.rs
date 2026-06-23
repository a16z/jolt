use super::*;

#[test]
fn mat_vec_mul_ntt_i8_dense_single_row_chunks_q128() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let log_basis = 6;
    let num_digits = 1;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let digit_ring = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let block = vec![digit_ring; cols];
    let block_slices: Vec<&[CyclotomicRing<F, D>]> = vec![block.as_slice()];

    let got =
        mat_vec_mul_ntt_i8_dense_single_row(&slot, cols, &block_slices, num_digits, log_basis)
            .expect("single-row dense mat-vec");

    let product = row * digit_ring;
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += product;
        acc
    });

    assert_eq!(got, vec![expected]);
}

#[test]
fn q128_many_blocks_digits_chunk_instead_of_unsafe_block_parallel() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let num_blocks = 16;
    let log_basis = 6;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|_| vec![CyclotomicRing::from_coefficients([half; D]); cols])
        .collect();
    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..cols)
                .map(|col| {
                    let digit = if (block_idx + col) % 2 == 0 { -32 } else { 31 };
                    [digit; D]
                })
                .collect()
        })
        .collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q128(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let got = mat_vec_mul_digits_i8_with_params_for_log_basis::<F, i32, Q128_NUM_PRIMES, D>(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            let expected = schoolbook_digit_mat_vec(&mat, &digit_blocks);

            assert_eq!(got, expected);
        }
        _ => panic!("unexpected parameter family"),
    }
}
