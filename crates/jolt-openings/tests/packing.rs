#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{OpeningsError, PrefixPacking};
use jolt_poly::{boolean_point_msb, eq_index_msb, Polynomial};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[path = "support/packed.rs"]
mod packed_support;

use packed_support::{materialize_packed, MaterializedPackedWitness};

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn build_packed(polynomials: &[(u64, Polynomial<Fr>)]) -> MaterializedPackedWitness<u64, Fr> {
    materialize_packed(polynomials).expect("packed polynomial should build")
}

#[test]
fn prefix_packing_is_deterministic_for_mixed_arities_and_id_ties() {
    let packing =
        PrefixPacking::new([(30_u64, 1), (20, 2), (10, 2), (40, 0)]).expect("packing should build");

    assert_eq!(packing.packed_num_vars, 4);
    assert_eq!((&packing).into_iter().count(), 4);
    assert_eq!(packing[&10].prefix, vec![false, false]);
    assert_eq!(packing[&20].prefix, vec![false, true]);
    assert_eq!(packing[&30].prefix, vec![true, false, false]);
    assert_eq!(packing[&40].prefix, vec![true, false, true, false]);
}

#[test]
fn prefix_packing_rejects_empty_and_duplicate_polynomials() {
    let empty = PrefixPacking::new(Vec::<(u64, usize)>::new());
    assert!(matches!(empty, Err(OpeningsError::InvalidSetup(_))));

    let duplicate = PrefixPacking::new([(7_u64, 2), (7, 1)]);
    assert!(matches!(duplicate, Err(OpeningsError::InvalidSetup(_))));
}

#[test]
fn prefix_packing_builds_directly_from_specs_without_witness_materialization() {
    let specs = [(30_u64, 1), (20, 2), (10, 2), (40, 0)];
    let packing = PrefixPacking::new(specs).expect("direct specs should produce packing");
    assert_eq!(packing.packed_num_vars, 4);
    assert_eq!(packing[&10].prefix, vec![false, false]);
}

#[test]
fn prefix_packing_helpers_validate_unknown_ids_and_point_dimensions() {
    let packing = PrefixPacking::new([(10_u64, 2), (20, 2)]).expect("packing should build");

    assert_eq!(packing[&20].prefix, vec![true]);
    assert_eq!(eq_index_msb(&[fr(2)], 1), fr(2));

    let packed_point = packing
        .pack_point(&[fr(11)], &[fr(2), fr(5)])
        .expect("packed point should combine prefix and logical coordinates");
    assert_eq!(packed_point, vec![fr(11), fr(2), fr(5)]);
    assert_eq!(
        packing
            .logical_point(&20, &packed_point)
            .expect("logical point should be the slot suffix"),
        vec![fr(2), fr(5)]
    );
    assert_eq!(
        eq_index_msb(&packed_point[..packing[&20].prefix.len()], 1),
        fr(11)
    );

    let wrong_prefix_len = packing.pack_point(&[fr(11), fr(13)], &[fr(2), fr(5)]);
    assert!(matches!(
        wrong_prefix_len,
        Err(OpeningsError::InvalidBatch(_))
    ));

    let logical_point_too_long = packing.pack_point(&[], &[fr(1), fr(2), fr(3), fr(4)]);
    assert!(matches!(
        logical_point_too_long,
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn materialized_packed_witness_places_polynomials_under_boolean_prefixes_at_random_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(7010);
    let poly_a = Polynomial::<Fr>::random(3, &mut rng);
    let poly_b = Polynomial::<Fr>::random(2, &mut rng);
    let poly_c = Polynomial::<Fr>::random(1, &mut rng);

    let packed = build_packed(&[
        (2, poly_b.clone()),
        (3, poly_c.clone()),
        (1, poly_a.clone()),
    ]);

    for (id, polynomial) in [(1_u64, &poly_a), (2, &poly_b), (3, &poly_c)] {
        let slot = &packed.packing[&id];
        let logical_point = (0..slot.num_vars)
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>();
        let prefix_point = slot
            .prefix
            .iter()
            .map(|bit| if *bit { fr(1) } else { fr(0) })
            .collect::<Vec<_>>();
        let packed_point = packed
            .packing
            .pack_point(&prefix_point, &logical_point)
            .expect("packed point should be valid");
        assert_eq!(
            packed.polynomial.evaluate(&packed_point),
            polynomial.evaluate(&logical_point)
        );
    }
}

#[test]
fn materialized_packed_witness_zero_fills_padding_cells() {
    let poly_a = Polynomial::new(vec![fr(3), fr(5), fr(7), fr(11)]);
    let poly_b = Polynomial::new(vec![fr(13)]);
    let packed = build_packed(&[(0, poly_a), (1, poly_b)]);

    assert_eq!(packed.packing.packed_num_vars, 3);
    for index in 5..8 {
        let point = boolean_point_msb(packed.packing.packed_num_vars, index);
        assert_eq!(packed.polynomial.evaluate(&point), fr(0));
    }
}

#[test]
fn materialized_packed_witness_rejects_duplicate_ids_and_empty_builds() {
    let polynomial = Polynomial::new(vec![fr(1), fr(2)]);
    let duplicate = materialize_packed(&[(7_u64, polynomial.clone()), (7, polynomial)]);
    assert!(matches!(duplicate, Err(OpeningsError::InvalidSetup(_))));

    let empty = materialize_packed::<u64, Fr>(&[]);
    assert!(matches!(empty, Err(OpeningsError::InvalidSetup(_))));
}
