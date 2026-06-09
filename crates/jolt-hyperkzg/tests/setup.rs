//! HyperKZG setup, SRS file, and derived-vector-commitment tests.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

mod common;

use common::{make_setup, KzgPCS};
use jolt_crypto::{Bn254, Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use num_traits::One;

fn commit_open_verify(
    poly: &Polynomial<Fr>,
    point: &[Fr],
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
    label: &'static [u8],
) {
    let eval = poly.evaluate(point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), pk);

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof =
        <KzgPCS as CommitmentScheme>::open(poly, point, eval, pk, None, &mut prover_transcript);

    let mut verifier_transcript = Blake2bTranscript::new(label);
    <KzgPCS as CommitmentScheme>::verify(
        &commitment,
        point,
        eval,
        &proof,
        vk,
        &mut verifier_transcript,
    )
    .expect("verification should succeed");
}

#[test]
fn deterministic_setup_from_secret() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let beta = Fr::from_u64(12345);

    let pk1 = KzgPCS::setup_from_secret(beta, 16, g1, g2);
    let pk2 = KzgPCS::setup_from_secret(beta, 16, g1, g2);
    let vk2 = KzgPCS::verifier_setup(&pk2);

    let poly = Polynomial::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
    let (com1, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk1);
    let (com2, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk2);
    assert_eq!(
        com1, com2,
        "deterministic setups must produce same commitments"
    );

    let point = vec![Fr::from_u64(7)];
    let eval = poly.evaluate(&point);
    let mut prover_transcript = Blake2bTranscript::new(b"det-setup");
    let proof =
        <KzgPCS as CommitmentScheme>::open(&poly, &point, eval, &pk1, None, &mut prover_transcript);
    let mut verifier_transcript = Blake2bTranscript::new(b"det-setup");
    <KzgPCS as CommitmentScheme>::verify(
        &com1,
        &point,
        eval,
        &proof,
        &vk2,
        &mut verifier_transcript,
    )
    .expect("cross-setup verification must work");
}

#[test]
fn trait_setup_uses_fresh_randomness() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let (pk1, _) = <KzgPCS as CommitmentScheme>::setup((4, g1, g2));
    let (pk2, _) = <KzgPCS as CommitmentScheme>::setup((4, g1, g2));

    let poly = Polynomial::new((1..=16).map(Fr::from_u64).collect());
    let (com1, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk1);
    let (com2, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk2);
    assert_ne!(com1, com2, "trait setup should sample a fresh trapdoor");
}

#[test]
fn srs_file_roundtrip_uses_canonical_name() {
    let k = 3;
    let (pk, _) = make_setup(1 << k);

    let dir = std::env::temp_dir().join(format!(
        "jolt-hyperkzg-srs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp SRS dir");

    KzgPCS::write_srs_to_dir(&pk, &dir, k).expect("write SRS file");
    let path = dir.join("hyperkzg_3.srs");
    assert!(path.exists(), "canonical SRS file should exist");

    let loaded = KzgPCS::read_srs_from_dir(&dir, k).expect("read SRS file");
    let vk = KzgPCS::verifier_setup(&loaded);

    let poly = Polynomial::new(vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ]);
    let point = vec![Fr::from_u64(9), Fr::from_u64(10), Fr::from_u64(11)];
    commit_open_verify(&poly, &point, &loaded, &vk, b"kzg-srs-file");

    std::fs::remove_dir_all(&dir).expect("remove temp SRS dir");
}

#[test]
fn extract_vc_setup_produces_valid_pedersen() {
    let (pk, _) = make_setup(1 << 4);
    let capacity = 5;
    let vc_setup = PedersenSetup::<Bn254G1>::derive(&pk, capacity);

    assert_eq!(
        <Pedersen<Bn254G1> as VectorCommitment>::capacity(&vc_setup),
        capacity,
    );

    let values = vec![Fr::one(), Fr::from_u64(2), Fr::from_u64(3)];
    let blinding = Fr::from_u64(42);
    let commitment = <Pedersen<Bn254G1> as VectorCommitment>::commit(&vc_setup, &values, &blinding);
    assert!(<Pedersen<Bn254G1> as VectorCommitment>::verify(
        &vc_setup,
        &commitment,
        &values,
        &blinding,
    ));
}
