#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, PackedBatch,
    PackedWitness, PrefixPackedClaim, PrefixPackedProverSetup, PrefixPackedStatement,
    PrefixPackedVerifierSetup, PrefixPacking,
};
use jolt_poly::{boolean_point_msb, eq_index_msb, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[path = "support/mock.rs"]
mod mock;
#[path = "support/packed.rs"]
mod packed_support;

use mock::{MockCommitment, MockCommitmentScheme};
use packed_support::{materialize_packed, MaterializedPackedWitness};

type MockPCS = MockCommitmentScheme<Fr>;
type PackedMockPCS = PackedBatch<MockPCS, u64>;
type TestClaim = PrefixPackedClaim<Fr, u64>;
type TestStatement = PrefixPackedStatement<Fr, u64, MockCommitment<Fr>>;

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn packed_setup(
    packing: PrefixPacking<u64>,
) -> (
    PrefixPackedProverSetup<MockPCS, u64>,
    PrefixPackedVerifierSetup<MockPCS, u64>,
) {
    (
        PrefixPackedProverSetup {
            pcs: (),
            packing: packing.clone(),
        },
        PrefixPackedVerifierSetup { pcs: (), packing },
    )
}

fn build_packed(polynomials: &[(u64, Polynomial<Fr>)]) -> MaterializedPackedWitness<u64, Fr> {
    materialize_packed(polynomials).expect("packed polynomial should build")
}

fn same_arity_polynomials() -> [(u64, Polynomial<Fr>); 2] {
    [
        (
            0,
            Polynomial::new((0..4).map(|index| fr(3 + 2 * index)).collect()),
        ),
        (
            1,
            Polynomial::new((0..4).map(|index| fr(17 + 3 * index)).collect()),
        ),
    ]
}

fn claims_for(
    polynomials: &[(u64, Polynomial<Fr>)],
    ids: &[u64],
    logical_point: Vec<Fr>,
) -> Vec<TestClaim> {
    ids.iter()
        .map(|id| {
            let polynomial = polynomials
                .iter()
                .find_map(|(candidate_id, polynomial)| (candidate_id == id).then_some(polynomial))
                .expect("claim id should have a source polynomial");
            PrefixPackedClaim::new(
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

fn claims_for_packed_point(
    polynomials: &[(u64, Polynomial<Fr>)],
    packing: &PrefixPacking<u64>,
    packed_point: &[Fr],
) -> Vec<TestClaim> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce a logical suffix");
            PrefixPackedClaim::new(
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

fn prove_packed(
    packed: &MaterializedPackedWitness<u64, Fr>,
    statement: TestStatement,
    hint: (),
    label: &'static [u8],
) -> <PackedMockPCS as BatchOpeningScheme>::Proof {
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let mut transcript = Blake2bTranscript::new(label);
    <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    )
    .expect("packed proof should prove")
}

fn verify_packed(
    packing: PrefixPacking<u64>,
    statement: TestStatement,
    proof: &<PackedMockPCS as BatchOpeningScheme>::Proof,
    label: &'static [u8],
) -> Result<(), OpeningsError> {
    let (_, verifier_setup) = packed_setup(packing);
    let mut transcript = Blake2bTranscript::new(label);
    <PackedMockPCS as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        proof,
        &mut transcript,
    )
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

#[test]
fn prefix_packed_batch_roundtrip_with_statement() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let mut prover_transcript = Blake2bTranscript::new(b"packing-roundtrip");
    let proof = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        PackedWitness::new(&packed.polynomial, hint),
        &mut prover_transcript,
    )
    .expect("packed proof should prove");
    let mut verifier_transcript = Blake2bTranscript::new(b"packing-roundtrip");
    <PackedMockPCS as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("packed proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn prefix_packed_batch_roundtrip_mixed_arities_without_padding() {
    let polynomials = [
        (
            0,
            Polynomial::new((0..8).map(|index| fr(1 + 2 * index)).collect()),
        ),
        (1, Polynomial::new(vec![fr(23), fr(29)])),
        (2, Polynomial::new(vec![fr(31)])),
    ];
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 4);

    let packed_point = vec![fr(2), fr(3), fr(5), fr(7)];
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = claims_for_packed_point(&polynomials, &packed.packing, &packed_point);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let proof = prove_packed(
        &packed,
        statement.clone(),
        hint,
        b"packing-mixed-arity-roundtrip",
    );
    verify_packed(
        packed.packing,
        statement,
        &proof,
        b"packing-mixed-arity-roundtrip",
    )
    .expect("mixed-arity packed proof should verify");
}

#[test]
fn prefix_packed_batch_rejects_missing_packed_slot() {
    let polynomials = [
        (0, Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)])),
        (1, Polynomial::new(vec![fr(11), fr(13), fr(17), fr(19)])),
        (2, Polynomial::new(vec![fr(23), fr(29)])),
    ];
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());

    let claims_arity_two = claims_for(&polynomials, &[0, 1], vec![fr(3), fr(5)]);
    let statement = PrefixPackedStatement::new(commitment, claims_arity_two);

    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let mut transcript = Blake2bTranscript::new(b"packing-mixed-arity-packing");
    let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn prefix_packed_batch_rejects_empty_claims() {
    let packed = build_packed(&same_arity_polynomials());
    let (prover_setup, _) = packed_setup(packed.packing.clone());

    let mut transcript = Blake2bTranscript::new(b"packing-empty-claims");
    let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(MockCommitment::default(), Vec::new()),
        PackedWitness::new(&packed.polynomial, ()),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn prefix_packed_batch_rejects_unknown_id() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let mut claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let proof = prove_packed(
        &packed,
        PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"packing-unknown-id",
    );
    claims[0].id = 99;

    let result = verify_packed(
        packed.packing,
        PrefixPackedStatement::new(commitment, claims),
        &proof,
        b"packing-unknown-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn prefix_packed_batch_rejects_suffix_incompatible_mixed_arity_claims() {
    let polynomials = [
        (0, Polynomial::new(vec![fr(3), fr(5), fr(7), fr(11)])),
        (1, Polynomial::new(vec![fr(13), fr(17)])),
    ];
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = vec![
        PrefixPackedClaim::new(
            0,
            EvaluationClaim::new(
                vec![fr(2), fr(5)],
                polynomials[0].1.evaluate(&[fr(2), fr(5)]),
            ),
        ),
        PrefixPackedClaim::new(
            1,
            EvaluationClaim::new(vec![fr(2)], polynomials[1].1.evaluate(&[fr(2)])),
        ),
    ];
    let statement = PrefixPackedStatement::new(commitment, claims);
    let (prover_setup, _) = packed_setup(packed.packing.clone());

    let mut transcript = Blake2bTranscript::new(b"packing-mixed-arities");
    let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn prefix_packed_batch_rejects_mismatched_logical_points() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let mut claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    claims[1].evaluation.point = vec![fr(8), fr(13)].into();
    let statement = PrefixPackedStatement::new(commitment, claims);
    let (prover_setup, _) = packed_setup(packed.packing.clone());

    let mut transcript = Blake2bTranscript::new(b"packing-mismatched-points");
    let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn prefix_packed_batch_rejects_wrong_packed_commitment() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let other_packed = build_packed(&[(0, Polynomial::new(vec![fr(1), fr(1), fr(1), fr(1)]))]);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let (other_commitment, ()) = MockPCS::commit(&other_packed.polynomial, &());
    let claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let proof = prove_packed(
        &packed,
        PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"packing-wrong-commitment",
    );
    let result = verify_packed(
        packed.packing,
        PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        b"packing-wrong-commitment",
    );
    assert!(result.is_err(), "wrong packed commitment should reject");
}

#[test]
fn prefix_packed_batch_rejects_tampered_id_even_when_id_is_known() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let proof = prove_packed(
        &packed,
        PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"packing-tampered-id",
    );
    let mut tampered = claims;
    tampered[0].id = 1;

    let result = verify_packed(
        packed.packing,
        PrefixPackedStatement::new(commitment, tampered),
        &proof,
        b"packing-tampered-id",
    );
    assert!(result.is_err(), "tampered known id should reject");
}

#[test]
fn prefix_packed_batch_rejects_tampered_value() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let proof = prove_packed(
        &packed,
        PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"packing-tampered-value",
    );
    let mut tampered = claims;
    tampered[0].evaluation.value += fr(1);

    let result = verify_packed(
        packed.packing,
        PrefixPackedStatement::new(commitment, tampered),
        &proof,
        b"packing-tampered-value",
    );
    assert!(result.is_err(), "tampered value should reject");
}

#[test]
fn prefix_packed_batch_rejects_wrong_witness_dimension() {
    let polynomials = same_arity_polynomials();
    let packed = build_packed(&polynomials);
    let (commitment, hint) = MockPCS::commit(&packed.polynomial, &());
    let claims = claims_for(&polynomials, &[0, 1], vec![fr(2), fr(5)]);
    let statement = PrefixPackedStatement::new(commitment, claims);
    let (prover_setup, _) = packed_setup(packed.packing);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2)]);

    let mut transcript = Blake2bTranscript::new(b"packing-wrong-witness");
    let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement,
        PackedWitness::new(&wrong_witness, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
