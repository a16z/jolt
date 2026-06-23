use super::*;

#[test]
fn mat_vec_mul_digits_i8_matches_num_digits_one_roundtrip() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = (i as i64 * 19 + j as i64 * 7 + k as i64) % 7;
                        F::from_i64(raw - 3)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = vec![
        (0..6)
            .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 7) as i8 - 3))
            .collect(),
        (0..4)
            .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
            .collect(),
        vec![],
    ];

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|digit| {
                    let coeffs = std::array::from_fn(|k| F::from_i64(digit[k] as i64));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                1,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_matches_direct_digits_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 3 + digit_idx as i16 * 5 + k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_matches_direct_digits_on_multi_tile_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;
    let num_blocks = 4;
    let rings_per_block = 1_400;
    let digits_per_block = rings_per_block * num_digits;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..digits_per_block)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 3 + digit_idx as i16 * 5 + k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let dense = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
            assert_eq!(dense, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_fast_path_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((13 * i as i64 + 7 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let dense = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            assert_eq!(dense, generic);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_single_row_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = vec![(0..6)
        .map(|j| {
            let coeffs = std::array::from_fn(|k| {
                let raw = ((7 * j as i64 + k as i64) % 9) - 4;
                F::from_i64(raw)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()];

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let single = super::mat_vec_mul_i8_dense_single_row_with_params(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let generic_single: Vec<CyclotomicRing<F, D>> =
                generic.into_iter().map(|row| row[0]).collect();
            assert_eq!(single, generic_single);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_three_rows_match_schoolbook_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 9 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let digit_blocks: Vec<Vec<[i8; D]>> = ring_blocks
                .iter()
                .map(|block| decompose_block_i8(block, num_digits, log_basis))
                .collect();
            let reference = schoolbook_digit_mat_vec(&mat, &digit_blocks);
            assert_eq!(generic, reference);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_digits_i8_three_rows_match_schoolbook_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 9 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i64 + 2 * digit_idx as i64 + k as i64) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_digits_i8_with_params_for_log_basis::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
            >(&ntt_mat, &digit_block_slices, log_basis, &params);
            let reference = schoolbook_digit_mat_vec(&mat, &digit_blocks);
            assert_eq!(generic, reference);
        }
        _ => panic!("unexpected parameter family"),
    }
}

// Block-parallel and column-tiled matvec must produce byte-identical ring output for
// wide rows (n_a 5..=7) across EVERY CRT+NTT parameter family the commit path can
// select: Q32 (<= 32-bit moduli), Q64 (64-bit), and Q128 (fp128). Raising
// `SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS` only changes which path runs, never the result,
// so this guards that invariant for all field families, not just fp32. One `#[test]`
// per family pinpoints a regression to the offending CRT family.
macro_rules! block_parallel_matches_column_tiled_wide_rows_test {
    ($name:ident, $field:ty, $d:literal, $variant:ident, $width:ident, $num_primes:ident) => {
        #[test]
        fn $name() {
            type F = $field;
            const D: usize = $d;
            // Wide enough to push the column-tiled reference path across >= 2 cache tiles
            // on both aarch64 (4 MiB L2) and x86_64 (1 MiB L2), so the block-parallel
            // single-accumulator result is checked against the tiled
            // accumulate-then-`add_ntt_into` combine.
            let width = 5_500;
            // Keep the reference on the column-tiled path:
            // `mat_vec_mul_digits_i8_with_params` only takes the block-parallel branch
            // when `num_blocks >= 16`.
            let num_blocks = 2;

            let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..7)
                .map(|i| {
                    (0..width)
                        .map(|j| {
                            let coeffs = std::array::from_fn(|k| {
                                let raw = ((17 * i as i64 + 9 * j as i64 + k as i64) % 9) - 4;
                                F::from_i64(raw)
                            });
                            CyclotomicRing::from_coefficients(coeffs)
                        })
                        .collect()
                })
                .collect();

            let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
                .map(|block_idx| {
                    (0..width)
                        .map(|digit_idx| {
                            std::array::from_fn(|k| {
                                (((block_idx as i64 + 2 * digit_idx as i64 + k as i64) % 7) - 3)
                                    as i8
                            })
                        })
                        .collect()
                })
                .collect();
            let digit_block_slices: Vec<&[[i8; D]]> =
                digit_blocks.iter().map(Vec::as_slice).collect();

            match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
                ProtocolCrtNttParams::$variant(params) => {
                    let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
                    let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
                    let log_basis = 3;
                    for n_a in 5..=7 {
                        let column_tiled =
                            mat_vec_mul_digits_i8_with_params::<F, $width, $num_primes, D>(
                                &ntt_mat[..n_a],
                                &digit_block_slices,
                                log_basis,
                                &params,
                            );
                        let block_parallel = super::mat_vec_mul_digits_i8_block_parallel::<
                            F,
                            $width,
                            $num_primes,
                            D,
                            true,
                        >(
                            &ntt_mat[..n_a],
                            &digit_block_slices,
                            super::balanced_digit_abs_bound(log_basis),
                            &params,
                        );
                        assert_eq!(
                            block_parallel, column_tiled,
                            "block-parallel must match column-tiled for n_a={n_a}"
                        );
                    }
                }
                _ => panic!("unexpected parameter family"),
            }
        }
    };
}

block_parallel_matches_column_tiled_wide_rows_test!(
    mat_vec_mul_digits_i8_block_parallel_matches_column_tiled_for_wide_rows,
    Fp64<4294967197>,
    64,
    Q32,
    i32,
    Q32_NUM_PRIMES
);
block_parallel_matches_column_tiled_wide_rows_test!(
    mat_vec_mul_digits_i8_block_parallel_matches_column_tiled_for_wide_rows_q64,
    Prime64Offset59,
    64,
    Q64,
    i32,
    Q64_NUM_PRIMES
);
block_parallel_matches_column_tiled_wide_rows_test!(
    mat_vec_mul_digits_i8_block_parallel_matches_column_tiled_for_wide_rows_q128,
    Prime128Offset275,
    64,
    Q128,
    i32,
    Q128_NUM_PRIMES
);

#[test]
fn mat_vec_mul_digits_i8_strided_three_row_matches_block_path_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((13 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((2 * block_idx as i64 + digit_idx as i64 + 3 * k as i64) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();
    let strided_digits: Vec<[i8; D]> = (0..6)
        .flat_map(|col| digit_blocks.iter().map(move |block| block[col]))
        .collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let block_path = mat_vec_mul_digits_i8_with_params_for_log_basis::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
            >(&ntt_mat, &digit_block_slices, log_basis, &params);
            let strided_path = super::mat_vec_mul_digits_i8_strided_block_parallel(
                &ntt_mat,
                &strided_digits,
                digit_blocks.len(),
                digit_blocks[0].len(),
                super::balanced_digit_abs_bound(log_basis),
                &params,
            );
            assert_eq!(strided_path, block_path);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_digits_i8_strided_matches_block_path_across_batch_boundary() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_blocks = 20;
    let digits_per_block = 9;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 11 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..digits_per_block)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 7 + digit_idx as i16 * 3 + 2 * k as i16) % 7) - 3)
                            as i8
                    })
                })
                .collect()
        })
        .collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    let mut strided_digits = Vec::with_capacity(num_blocks * digits_per_block);
    for col in 0..digits_per_block {
        for block in &digit_blocks {
            strided_digits.push(block[col]);
        }
    }

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let block_path = mat_vec_mul_digits_i8_with_params_for_log_basis::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
            >(&ntt_mat, &digit_block_slices, log_basis, &params);
            let strided_path = mat_vec_mul_digits_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_digits,
                num_blocks,
                digits_per_block,
                log_basis,
                &params,
            );
            assert_eq!(block_path, strided_path);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_digits_i8_sparse_batches_match_schoolbook() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_blocks = 7;
    let digits_per_block = 5;
    let total_planes = num_blocks * digits_per_block;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((13 * i as i64 + 17 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let positions: Vec<(usize, usize)> = (0..total_planes)
        .map(|idx| {
            let permuted = (idx * 11) % total_planes;
            (permuted % num_blocks, permuted / num_blocks)
        })
        .collect();

    for live_count in [0usize, 15, 16, 17, 31] {
        let mut digit_blocks = vec![vec![[0i8; D]; digits_per_block]; num_blocks];
        for (live_idx, &(block_idx, col)) in positions.iter().take(live_count).enumerate() {
            digit_blocks[block_idx][col] = std::array::from_fn(|k| {
                let raw = ((live_idx as i16 * 5 + col as i16 * 3 + k as i16) % 7) - 3;
                if raw == 0 {
                    1
                } else {
                    raw as i8
                }
            });
        }

        let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();
        let mut strided_digits = Vec::with_capacity(total_planes);
        for col in 0..digits_per_block {
            for block in &digit_blocks {
                strided_digits.push(block[col]);
            }
        }

        match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
            ProtocolCrtNttParams::Q32(params) => {
                let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
                let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
                let expected = schoolbook_digit_mat_vec(&mat, &digit_blocks);
                let block_path =
                    mat_vec_mul_digits_i8_with_params_for_log_basis::<F, i32, Q32_NUM_PRIMES, D>(
                        &ntt_mat,
                        &digit_block_slices,
                        log_basis,
                        &params,
                    );
                let strided_path = mat_vec_mul_digits_i8_strided_with_params_for_log_basis(
                    &ntt_mat,
                    &strided_digits,
                    num_blocks,
                    digits_per_block,
                    log_basis,
                    &params,
                );
                assert_eq!(block_path, expected, "block path mismatch at {live_count}");
                assert_eq!(
                    strided_path, expected,
                    "strided path mismatch at {live_count}"
                );
            }
            _ => panic!("unexpected parameter family"),
        }
    }
}

#[test]
fn mat_vec_mul_digits_i8_sparse_strided_ragged_tail_matches_schoolbook() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_blocks = 7;
    let digits_per_block = 5;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((19 * i as i64 + 7 * j as i64 + 3 * k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let mut digit_blocks = vec![vec![[0i8; D]; digits_per_block]; num_blocks];
    for col in 0..(digits_per_block - 1) {
        for (block_idx, block) in digit_blocks.iter_mut().enumerate() {
            if (block_idx + 2 * col) % 3 == 0 {
                block[col] = std::array::from_fn(|k| {
                    let raw = ((block_idx as i16 + col as i16 * 5 + k as i16) % 7) - 3;
                    if raw == 0 {
                        -1
                    } else {
                        raw as i8
                    }
                });
            }
        }
    }

    let mut strided_digits = Vec::with_capacity(num_blocks * digits_per_block);
    for col in 0..digits_per_block {
        for block in &digit_blocks {
            strided_digits.push(block[col]);
        }
    }
    strided_digits.truncate(strided_digits.len() - 3);

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let expected = schoolbook_digit_mat_vec(&mat, &digit_blocks);
            let strided_path = mat_vec_mul_digits_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_digits,
                num_blocks,
                digits_per_block,
                log_basis,
                &params,
            );
            assert_eq!(strided_path, expected);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_strided_matches_block_path_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((19 * i as i64 + 11 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    let mut strided_coeffs = Vec::with_capacity(ring_blocks.len() * ring_blocks[0].len());
    for col in 0..ring_blocks[0].len() {
        for block in &ring_blocks {
            strided_coeffs.push(block[col]);
        }
    }

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let block_path = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let strided_path = mat_vec_mul_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_coeffs,
                ring_blocks.len(),
                ring_blocks[0].len(),
                num_digits,
                log_basis,
                &params,
            );
            assert_eq!(block_path, strided_path);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_strided_matches_direct_digits_on_multi_tile_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;
    let num_blocks = 4;
    let rings_per_block = 1_400;
    let digits_per_block = rings_per_block * num_digits;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((19 * i as i64 + 11 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..digits_per_block)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 5 + digit_idx as i16 + 3 * k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let mut strided_coeffs = Vec::with_capacity(num_blocks * rings_per_block);
    for col in 0..rings_per_block {
        for block in &ring_blocks {
            strided_coeffs.push(block[col]);
        }
    }

    let mut strided_digits = Vec::with_capacity(num_blocks * digits_per_block);
    for col in 0..digits_per_block {
        for block in &digit_blocks {
            strided_digits.push(block[col]);
        }
    }

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_coeffs,
                num_blocks,
                rings_per_block,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_digits,
                num_blocks,
                digits_per_block,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}
