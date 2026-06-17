//! ZK HyperKZG SRS import/export tests.

#![cfg(feature = "zk")]
#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_hyperkzg::error::HyperKZGError;
use jolt_hyperkzg::HyperKZGScheme;

type KzgPCS = HyperKZGScheme<Bn254>;

fn temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "jolt-hyperkzg-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_nanos()
    ))
}

fn make_zk_setup(max_degree: usize) -> jolt_hyperkzg::HyperKZGProverSetup<Bn254> {
    let g1 = Bn254::g1_generator();
    let hiding_g1 = g1.scalar_mul(&Fr::from_u64(17));
    let g2 = Bn254::g2_generator();
    KzgPCS::setup_zk_from_secret(Fr::from_u64(12345), max_degree, g1, hiding_g1, g2)
}

#[test]
fn zk_srs_file_roundtrip_uses_canonical_name() {
    let k = 3;
    let pk = make_zk_setup(1 << k);
    let dir = temp_dir("zk-srs");
    std::fs::create_dir_all(&dir).expect("create temp SRS dir");

    KzgPCS::write_zk_srs_to_dir(&pk, &dir, k).expect("write ZK SRS file");
    let path = dir.join("hyperkzg_zk_3.srs");
    assert!(path.exists(), "canonical ZK SRS file should exist");

    let _loaded = KzgPCS::read_zk_srs_from_dir(&dir, k).expect("read ZK SRS file");

    std::fs::remove_dir_all(&dir).expect("remove temp SRS dir");
}

#[test]
fn plain_loader_rejects_zk_srs_file() {
    let pk = make_zk_setup(8);
    let dir = temp_dir("plain-rejects-zk");
    std::fs::create_dir_all(&dir).expect("create temp SRS dir");
    let path = dir.join("mismatch.srs");

    KzgPCS::write_zk_srs_file(&pk, &path).expect("write ZK SRS file");
    let result = KzgPCS::read_srs_file(&path);

    assert!(matches!(
        result,
        Err(HyperKZGError::SrsFileKindMismatch { .. })
    ));
    std::fs::remove_dir_all(&dir).expect("remove temp SRS dir");
}

#[test]
fn zk_loader_rejects_plain_srs_file() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup_from_secret(Fr::from_u64(12345), 8, g1, g2);
    let dir = temp_dir("zk-rejects-plain");
    std::fs::create_dir_all(&dir).expect("create temp SRS dir");
    let path = dir.join("mismatch.srs");

    KzgPCS::write_srs_file(&pk, &path).expect("write plain SRS file");
    let result = KzgPCS::read_zk_srs_file(&path);

    assert!(matches!(
        result,
        Err(HyperKZGError::SrsFileKindMismatch { .. })
    ));
    std::fs::remove_dir_all(&dir).expect("remove temp SRS dir");
}

#[test]
fn plain_writer_rejects_zk_setup() {
    let pk = make_zk_setup(8);
    let dir = temp_dir("plain-writer-rejects-zk");
    std::fs::create_dir_all(&dir).expect("create temp SRS dir");
    let path = dir.join("wrong-writer.srs");

    let result = KzgPCS::write_srs_file(&pk, &path);

    assert!(matches!(
        result,
        Err(HyperKZGError::WrongSrsSetupKind { .. })
    ));
    std::fs::remove_dir_all(&dir).expect("remove temp SRS dir");
}
