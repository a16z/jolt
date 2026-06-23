use super::*;

#[test]
fn fused_split_eq_quotients_uses_all_cyclic_role_rows() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let rows = 3;
    let cols = 5;
    let flat_rows: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let raw = (idx as i64 * 17 + k as i64 * 5) % 31;
                F::from_i64(raw - 15)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(rows, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

    let e_hat: Vec<[i8; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 7) as i8 - 3))
        .collect();
    let t_hat: Vec<[i8; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((3 * j + k) % 5) as i8 - 2))
        .collect();
    let z_pre: Vec<[i32; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((j + k) % 3) as i32 - 1))
        .collect();

    let log_basis = 3;
    let expected_d = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, rows, cols, &e_hat, log_basis)
        .expect("expected D rows");
    let expected_b = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, rows, cols, &t_hat, log_basis)
        .expect("expected B rows");
    let (d_rows, b_rows, _a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, rows, rows, 1, &e_hat, &t_hat, &z_pre, 1)
            .expect("fused split-eq rows");

    assert_eq!(d_rows, expected_d);
    assert_eq!(b_rows, expected_b);
}

#[test]
fn fused_split_eq_q128_quotient_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 32_768)
            .expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_q128_quotient_falls_back_when_one_term_exceeds_crt() {
    type F = Prime128Offset275;
    const D: usize = 128;
    let cols = 1;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 32_768)
            .expect("fused split-eq rows");

    let z = centered_i32_ring(&z_pre[0]);
    let expected = quotient_from_cyclic_and_negacyclic(&cyclic_product(&row, &z), &(row * z));

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_uses_actual_centered_bound_when_hint_is_underreported() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 1)
            .expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_q128_cyclic_i8_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let e_hat = vec![[-32i8; D]; cols];

    let (d_rows, _b_rows, _a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 1, 0, 0, &e_hat, &[], &[], 0)
            .expect("fused split-eq rows");

    let digit = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += cyclic_product(&row, &digit);
        acc
    });

    assert_eq!(d_rows, vec![expected]);
}

#[test]
fn fused_split_eq_quotients_uses_role_local_packed_widths() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let n_d = 2;
    let n_b = 3;
    let n_a = 2;
    let d_width = 2;
    let b_width = 4;
    let a_width = 3;
    let total_len = (n_d * d_width).max(n_b * b_width).max(n_a * a_width);
    let flat_rows: Vec<CyclotomicRing<F, D>> = (0..total_len)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let raw = (idx as i64 * 19 + k as i64 * 7) % 37;
                F::from_i64(raw - 18)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, total_len)
            .expect("valid packed setup prefix"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

    let e_hat: Vec<[i8; D]> = (0..d_width)
        .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 5) as i8 - 2))
        .collect();
    let t_hat: Vec<[i8; D]> = (0..b_width)
        .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
        .collect();
    let z_pre: Vec<[i32; D]> = (0..a_width)
        .map(|j| std::array::from_fn(|k| ((3 * j + k) % 7) as i32 - 3))
        .collect();
    let z_rings: Vec<CyclotomicRing<F, D>> = z_pre
        .iter()
        .map(|row| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| F::from_i64(row[k] as i64)))
        })
        .collect();

    let log_basis = 3;
    let expected_d =
        mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, n_d, d_width, &e_hat, log_basis)
            .expect("expected D rows");
    let expected_b =
        mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, n_b, b_width, &t_hat, log_basis)
            .expect("expected B rows");
    let expected_a = (0..n_a)
        .map(|row_idx| {
            (0..a_width).fold(CyclotomicRing::<F, D>::zero(), |mut acc, col_idx| {
                let lhs = flat_rows[row_idx * a_width + col_idx];
                let z = z_rings[col_idx];
                let cyclic = cyclic_product(&lhs, &z);
                let negacyclic = lhs * z;
                acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
                acc
            })
        })
        .collect::<Vec<_>>();
    let (d_rows, b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, n_d, n_b, n_a, &e_hat, &t_hat, &z_pre, 3)
            .expect("fused split-eq rows");

    assert_eq!(d_rows, expected_d);
    assert_eq!(b_rows, expected_b);
    assert_eq!(a_rows, expected_a);
}
