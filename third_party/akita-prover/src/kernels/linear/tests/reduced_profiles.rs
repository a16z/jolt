use super::{centered_i32_ring, cyclic_product, quotient_from_cyclic_and_negacyclic};
use crate::kernels::crt_ntt::build_ntt_slot;
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};
use akita_algebra::CyclotomicRing;
use akita_field::{
    CanonicalField, FieldCore, HalvingField, Prime128Offset275, Prime32Offset99, Prime64Offset59,
};
use akita_types::layout::FlatMatrix;

fn assert_single_i8_chunk_paths<F: FieldCore + CanonicalField, const D: usize>(cols: usize) {
    let log_basis = 6;
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
    .expect("CRT+NTT dispatch should support this field and ring dimension");
    let digits = vec![[-32i8; D]; cols];

    let negacyclic = mat_vec_mul_ntt_single_i8::<F, D>(&slot, 1, cols, &digits, log_basis)
        .expect("single predecomposed digit mat-vec");
    let cyclic = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, 1, cols, &digits, log_basis)
        .expect("single cyclic predecomposed digit mat-vec");

    let negacyclic_product = row * digit_ring;
    let expected_negacyclic = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += negacyclic_product;
        acc
    });
    let cyclic_term = cyclic_product(&row, &digit_ring);
    let expected_cyclic = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += cyclic_term;
        acc
    });

    assert_eq!(negacyclic, vec![expected_negacyclic]);
    assert_eq!(cyclic, vec![expected_cyclic]);
}

fn assert_fused_split_eq_zpre_chunks<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    cols: usize,
) {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("CRT+NTT dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 32_768)
            .expect("fused split-eq rows");

    let z = centered_i32_ring(&z_pre[0]);
    let term = quotient_from_cyclic_and_negacyclic(&cyclic_product(&row, &z), &(row * z));
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += term;
        acc
    });

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_zpre_chunks_reduced_profiles() {
    assert_fused_split_eq_zpre_chunks::<Prime32Offset99, 256>(32);
    assert_fused_split_eq_zpre_chunks::<Prime64Offset59, 256>(8);
}

#[test]
fn mat_vec_mul_ntt_single_i8_chunks_q128() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let log_basis = 6;
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
    let digits = vec![[-32i8; D]; cols];

    let got = mat_vec_mul_ntt_single_i8::<F, D>(&slot, 1, cols, &digits, log_basis)
        .expect("single predecomposed digit mat-vec");

    let product = row * digit_ring;
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += product;
        acc
    });

    assert_eq!(got, vec![expected]);
}

#[test]
fn mat_vec_mul_ntt_single_i8_cyclic_chunks_q128() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let log_basis = 6;
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
    let digits = vec![[-32i8; D]; cols];

    let got = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, 1, cols, &digits, log_basis)
        .expect("single cyclic predecomposed digit mat-vec");

    let product = cyclic_product(&row, &digit_ring);
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += product;
        acc
    });

    assert_eq!(got, vec![expected]);
}

#[test]
fn mat_vec_mul_ntt_single_i8_chunks_reduced_profiles() {
    assert_single_i8_chunk_paths::<Prime32Offset99, 256>(900);
    assert_single_i8_chunk_paths::<Prime64Offset59, 256>(8_200);
}
