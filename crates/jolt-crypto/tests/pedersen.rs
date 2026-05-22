//! Pedersen commitment scheme tests over BN254 G1.

#![expect(clippy::unwrap_used, reason = "tests should fail loudly")]

use jolt_crypto::{
    Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorCommitmentOpening,
    VectorOpeningError,
};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_poly::EqPolynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn deterministic_setup(count: usize) -> PedersenSetup<Bn254G1> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead);
    let message_generators: Vec<Bn254G1> = (0..count).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding_generator = Bn254::random_g1(&mut rng);
    PedersenSetup::new(message_generators, blinding_generator)
}

fn mle_eval(flattened_rows: &[Fr], row_len: usize, row_point: &[Fr], entry_point: &[Fr]) -> Fr {
    let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
    let entry_weights = EqPolynomial::new(entry_point.to_vec()).evaluations();
    let mut result = Fr::from_u64(0);
    for (row_index, row_weight) in row_weights.iter().copied().enumerate() {
        let row_offset = row_index * row_len;
        for (entry_index, entry_weight) in entry_weights.iter().copied().enumerate() {
            if let Some(value) = flattened_rows.get(row_offset + entry_index) {
                result += row_weight * entry_weight * *value;
            }
        }
    }
    result
}

fn row_commitments(
    setup: &PedersenSetup<Bn254G1>,
    flattened_rows: &[Fr],
    row_len: usize,
    row_blindings: &[Fr],
) -> Vec<Bn254G1> {
    row_blindings
        .iter()
        .enumerate()
        .map(|(row_index, blinding)| {
            let row_offset = row_index * row_len;
            let mut row = vec![Fr::from_u64(0); row_len];
            for (entry_index, row_entry) in row.iter_mut().enumerate() {
                if let Some(value) = flattened_rows.get(row_offset + entry_index) {
                    *row_entry = *value;
                }
            }
            Pedersen::<Bn254G1>::commit(setup, &row, blinding)
        })
        .collect()
}

#[test]
fn commit_verify_roundtrip() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(1);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
    assert!(Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &blinding
    ));
}

#[test]
fn wrong_values_rejected() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(2);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);

    let mut wrong_values = values.clone();
    wrong_values[0] += Fr::from_u64(1);
    assert!(!Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &wrong_values,
        &blinding
    ));
}

#[test]
fn wrong_blinding_rejected() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(3);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);

    let wrong_blinding = blinding + Fr::from_u64(1);
    assert!(!Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &wrong_blinding
    ));
}

#[test]
fn different_blinding_different_commitment() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(4);

    let values: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let r1 = Fr::random(&mut rng);
    let r2 = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &values, &r1);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &values, &r2);
    assert_ne!(
        c1, c2,
        "different blindings should produce different commitments"
    );
}

#[test]
fn commitment_is_binding() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(5);

    let v1: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let v2: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &v1, &blinding);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &v2, &blinding);
    assert_ne!(
        c1, c2,
        "different messages with same blinding should differ"
    );
}

#[test]
fn additive_homomorphism() {
    // C(m₁, r₁) + C(m₂, r₂) == C(m₁+m₂, r₁+r₂)
    let setup = deterministic_setup(3);
    let mut rng = ChaCha20Rng::seed_from_u64(6);

    let v1: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let v2: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let r1 = Fr::random(&mut rng);
    let r2 = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &v1, &r1);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &v2, &r2);

    let v_sum: Vec<Fr> = v1.iter().zip(v2.iter()).map(|(a, b)| *a + *b).collect();
    let r_sum = r1 + r2;
    let c_sum = Pedersen::<Bn254G1>::commit(&setup, &v_sum, &r_sum);

    assert_eq!(c1 + c2, c_sum, "Pedersen should be additively homomorphic");
}

#[test]
fn zero_blinding_commit() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(7);

    let values: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let zero = Fr::from_u64(0);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &zero);
    // Without blinding, commitment is purely MSM of message generators.
    let expected = Bn254G1::msm(&setup.message_generators[..2], &values);
    assert_eq!(commitment, expected);
}

#[test]
fn capacity_returns_generator_count() {
    let setup = deterministic_setup(10);
    assert_eq!(Pedersen::<Bn254G1>::capacity(&setup), 10);
}

#[test]
#[should_panic(expected = "exceeds generator count")]
fn commit_panics_on_exceeding_capacity() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(9);

    let values: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let _ = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
}

#[test]
#[should_panic(expected = "at least one message generator")]
fn setup_panics_on_empty_generators() {
    let _ = PedersenSetup::new(Vec::<Bn254G1>::new(), Bn254G1::identity());
}

#[test]
fn partial_values_uses_prefix_generators() {
    let setup = deterministic_setup(8);
    let mut rng = ChaCha20Rng::seed_from_u64(8);

    let values: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
    assert!(Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &blinding
    ));
}

#[test]
fn committed_rows_opening_roundtrip_returns_mle_eval() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(10);

    let row_len = 4;
    let row_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let entry_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let flattened_rows: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();
    let row_blindings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let row_commitments = row_commitments(&setup, &flattened_rows, row_len, &row_blindings);

    let (opening, opened_eval) = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        row_len,
        &row_point,
        &entry_point,
    )
    .unwrap();
    let verified_eval = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap();

    let expected = mle_eval(&flattened_rows, row_len, &row_point, &entry_point);
    assert_eq!(opened_eval, expected);
    assert_eq!(verified_eval, expected);
}

#[test]
fn committed_rows_opening_accepts_zero_padded_flattened_rows() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(11);

    let row_len = 4;
    let row_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let entry_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let flattened_rows: Vec<Fr> = (0..10).map(|_| Fr::random(&mut rng)).collect();
    let row_blindings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let row_commitments = row_commitments(&setup, &flattened_rows, row_len, &row_blindings);

    let (opening, opened_eval) = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        row_len,
        &row_point,
        &entry_point,
    )
    .unwrap();
    let verified_eval = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap();

    let expected = mle_eval(&flattened_rows, row_len, &row_point, &entry_point);
    assert_eq!(opened_eval, expected);
    assert_eq!(verified_eval, expected);
}

#[test]
fn committed_rows_opening_handles_single_row_scalar_entry() {
    let setup = deterministic_setup(1);
    let value = Fr::from_u64(42);
    let blinding = Fr::from_u64(17);
    let flattened_rows = [value];
    let row_blindings = [blinding];
    let row_commitments = [Pedersen::<Bn254G1>::commit(
        &setup,
        &flattened_rows,
        &blinding,
    )];

    let (opening, opened_eval) =
        Pedersen::<Bn254G1>::open_committed_rows(&flattened_rows, &row_blindings, 1, &[], &[])
            .unwrap();
    let verified_eval =
        Pedersen::<Bn254G1>::verify_committed_rows(&setup, &row_commitments, &[], &[], &opening)
            .unwrap();

    assert_eq!(opened_eval, value);
    assert_eq!(verified_eval, value);
}

#[test]
fn committed_rows_opening_rejects_tampered_combined_vector() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(12);

    let row_len = 4;
    let row_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let entry_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let flattened_rows: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();
    let row_blindings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let row_commitments = row_commitments(&setup, &flattened_rows, row_len, &row_blindings);
    let (mut opening, _) = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        row_len,
        &row_point,
        &entry_point,
    )
    .unwrap();

    opening.combined_vector[0] += Fr::from_u64(1);
    let err = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap_err();
    assert!(matches!(err, VectorOpeningError::CommitmentMismatch));
}

#[test]
fn committed_rows_opening_rejects_tampered_blinding() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(13);

    let row_len = 4;
    let row_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let entry_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let flattened_rows: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();
    let row_blindings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let row_commitments = row_commitments(&setup, &flattened_rows, row_len, &row_blindings);
    let (mut opening, _) = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        row_len,
        &row_point,
        &entry_point,
    )
    .unwrap();

    opening.combined_blinding += Fr::from_u64(1);
    let err = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap_err();
    assert!(matches!(err, VectorOpeningError::CommitmentMismatch));
}

#[test]
fn committed_rows_opening_rejects_wrong_row_commitment() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(14);

    let row_len = 4;
    let row_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let entry_point: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let flattened_rows: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();
    let row_blindings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let mut row_commitments = row_commitments(&setup, &flattened_rows, row_len, &row_blindings);
    row_commitments[0] = row_commitments[1];
    let (opening, _) = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        row_len,
        &row_point,
        &entry_point,
    )
    .unwrap();

    let err = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap_err();
    assert!(matches!(err, VectorOpeningError::CommitmentMismatch));
}

#[test]
fn committed_rows_opening_rejects_invalid_dimensions() {
    let row_point = [Fr::from_u64(2), Fr::from_u64(3)];
    let entry_point = [Fr::from_u64(5), Fr::from_u64(7)];
    let flattened_rows = vec![Fr::from_u64(1); 16];
    let row_blindings = vec![Fr::from_u64(9); 4];

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        0,
        &row_point,
        &entry_point,
    )
    .unwrap_err();
    assert!(matches!(err, VectorOpeningError::RowLenZero));

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        3,
        &row_point,
        &entry_point,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::RowLenNotPowerOfTwo { row_len: 3 }
    ));

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        4,
        &row_point,
        &entry_point[..1],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::EntryPointLengthMismatch {
            expected: 2,
            got: 1
        }
    ));

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings[..3],
        4,
        &row_point,
        &entry_point,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::RowBlindingsLengthMismatch {
            expected: 4,
            got: 3
        }
    ));

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &[Fr::from_u64(1); 17],
        &row_blindings,
        4,
        &row_point,
        &entry_point,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::FlattenedRowsTooLong { max: 16, got: 17 }
    ));

    let err = Pedersen::<Bn254G1>::open_committed_rows(
        &flattened_rows,
        &row_blindings,
        4,
        &vec![Fr::from_u64(1); usize::BITS as usize],
        &entry_point,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::PointTooLarge { point_len } if point_len == usize::BITS as usize
    ));
}

#[test]
fn committed_rows_verifier_rejects_invalid_dimensions() {
    let setup = deterministic_setup(2);
    let row_point = [Fr::from_u64(2), Fr::from_u64(3)];
    let entry_point = [Fr::from_u64(5), Fr::from_u64(7)];
    let row_commitments = vec![Bn254G1::identity(); 3];
    let opening = VectorCommitmentOpening {
        combined_vector: vec![Fr::from_u64(1); 4],
        combined_blinding: Fr::from_u64(9),
    };

    let err = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::RowCommitmentsLengthMismatch {
            expected: 4,
            got: 3
        }
    ));

    let row_commitments = vec![Bn254G1::identity(); 4];
    let err = Pedersen::<Bn254G1>::verify_committed_rows(
        &setup,
        &row_commitments,
        &row_point,
        &entry_point,
        &opening,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorOpeningError::CommitmentCapacityExceeded {
            capacity: 2,
            row_len: 4
        }
    ));
}
