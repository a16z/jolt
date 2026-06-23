use super::*;

#[test]
fn dense_mat_vec_matches_schoolbook_q32_d64() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();
    let vec: Vec<CyclotomicRing<F, D>> = (0..4)
        .map(|j| {
            let coeffs = std::array::from_fn(|k| F::from_u64((j as u64 * 50 + k as u64 + 3) % 89));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
    let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q32 dispatch should succeed");
    assert_eq!(schoolbook, crt_ntt);
}

#[test]
fn dense_mat_vec_matches_schoolbook_q64_dispatch_for_large_d() {
    type F = Fp64<4294967197>;
    const D: usize = 128;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 20_000 + j as u64 * 300 + k as u64 + 7) % 113)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();
    let vec: Vec<CyclotomicRing<F, D>> = (0..2)
        .map(|j| {
            let coeffs =
                std::array::from_fn(|k| F::from_u64((j as u64 * 70 + k as u64 + 11) % 101));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
    let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q64 dispatch should succeed");
    assert_eq!(schoolbook, crt_ntt);
}

#[test]
fn dense_mat_vec_many_matches_individual_crt_ntt_q32_d64() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let vecs: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|seed| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((seed as u64 * 700 + j as u64 * 50 + k as u64 + 3) % 89)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let expected: Vec<Vec<CyclotomicRing<F, D>>> = vecs
        .iter()
        .map(|v| mat_vec_mul_crt_ntt(&mat, v).expect("single CRT+NTT mat-vec should succeed"))
        .collect();

    let got =
        mat_vec_mul_crt_ntt_many(&mat, &vecs).expect("batched CRT+NTT mat-vec should succeed");
    assert_eq!(expected, got);
}
