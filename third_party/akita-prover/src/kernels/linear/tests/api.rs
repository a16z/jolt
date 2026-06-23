use super::*;

#[test]
fn aligned_i8_tile_width_keeps_full_tiles_on_digit_boundaries() {
    assert_eq!(aligned_i8_tile_width(130, 512, 64), 128);
    assert_eq!(aligned_i8_tile_width(63, 512, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 65, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 48, 64), 48);
}

#[test]
fn predecomposed_digit_api_rejects_digits_outside_log_basis_range() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let row = CyclotomicRing::<F, D>::one();
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = build_ntt_slot(flat.ring_view::<D>(1, 1).expect("valid matrix"))
        .expect("Q32 dispatch should support this field and ring dimension");
    let bad_digits = vec![[4i8; D]];
    let blocks: Vec<&[[i8; D]]> = vec![bad_digits.as_slice()];

    assert!(matches!(
        mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, 1, &blocks, 3),
        Err(akita_field::AkitaError::InvalidInput(_))
    ));
}

#[test]
fn raw_i8_strided_accepts_signed_unit_outside_binary_digit_range() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let num_rows = 2;
    let block_len = 3;
    let num_blocks = 2;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
        .map(|row| {
            (0..block_len)
                .map(|col| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        F::from_u64((row + col + k + 1) as u64)
                    }))
                })
                .collect()
        })
        .collect();
    let flat_rows: Vec<_> = mat.iter().flatten().copied().collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(num_rows, block_len)
            .expect("valid matrix view"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");
    let coeffs: Vec<[i8; D]> = (0..block_len)
        .flat_map(|col| {
            (0..num_blocks).map(move |block| {
                if (col + block) % 2 == 0 {
                    [1i8; D]
                } else {
                    [-1i8; D]
                }
            })
        })
        .collect();
    let blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block| {
            (0..block_len)
                .map(|col| coeffs[col * num_blocks + block])
                .collect()
        })
        .collect();

    let got = mat_vec_mul_ntt_raw_i8_strided::<F, D>(
        &slot, num_rows, block_len, &coeffs, num_blocks, block_len,
    )
    .expect("raw signed-i8 strided mat-vec");
    let expected = schoolbook_digit_mat_vec(&mat, &blocks);

    assert_eq!(got, expected);
}
