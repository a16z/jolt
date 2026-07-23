//! Writes checked-in accepted-proof bundles consumed by the `jolt-verifier`
//! fuzz workspace. Transparent and ZK builds generate their own protocol-
//! compatible fixtures.
//!
//! Run explicitly and commit the output:
//! `cargo nextest run -p jolt-verifier --features prover-fixtures \
//!   --test generate_fuzz_fixture --run-ignored ignored-only`

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    dead_code,
    reason = "the ignored fixture generator should fail loudly and report written paths"
)]

use std::fs;
use std::path::PathBuf;

use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};

mod support;

type FuzzPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type FuzzProof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type FuzzBundle = (
    FuzzPreprocessing,
    common::jolt_device::JoltDevice,
    FuzzProof,
    Option<DoryCommitment>,
);

#[test]
#[ignore = "writes the checked-in fuzz fixture; run explicitly and commit the output"]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn generate_transparent_fuzz_fixtures() {
    for (filename, case) in [
        (
            "muldiv-bundle.bin",
            support::verifier_fixtures::standard_muldiv_case(),
        ),
        (
            "advice-consumer-bundle.bin",
            support::verifier_fixtures::standard_advice_consumer_case(),
        ),
        (
            "committed-muldiv-bundle.bin",
            support::verifier_fixtures::standard_committed_muldiv_case(),
        ),
    ] {
        support::assert_accepts(case.verify());
        write_bundle(
            filename,
            (
                case.preprocessing,
                case.public_io,
                case.proof,
                case.trusted_advice_commitment,
            ),
        );
    }
}

#[test]
#[ignore = "writes checked-in ZK fuzz fixtures; run explicitly and commit the output"]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn generate_zk_fuzz_fixtures() {
    for (filename, case) in [
        (
            "zk-muldiv-bundle.bin",
            support::verifier_fixtures::zk_muldiv_case(),
        ),
        (
            "zk-committed-muldiv-bundle.bin",
            support::verifier_fixtures::zk_committed_muldiv_case(),
        ),
    ] {
        support::assert_zk_accepts(case.verify());
        write_bundle(
            filename,
            (case.preprocessing, case.public_io, case.proof, None),
        );
    }
}

fn write_bundle(filename: &str, bundle: FuzzBundle) {
    let bytes = bincode::serde::encode_to_vec(&bundle, bincode::config::standard())
        .expect("serialize fuzz bundle");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("fixtures")
        .join(filename);
    fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture directory");
    fs::write(&path, &bytes).expect("write fuzz bundle");

    // Round-trip through the public deserialization the fuzz target uses.
    let (decoded, consumed): (FuzzBundle, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("decode fuzz bundle");
    assert_eq!(consumed, bytes.len(), "fuzz bundle has trailing bytes");
    support::assert_accepts(jolt_verifier::verify::<
        jolt_field::Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        jolt_transcript::LegacyBlake2bTranscript,
    >(&decoded.0, &decoded.1, &decoded.2, decoded.3.as_ref()));

    println!("wrote {} bytes to {}", bytes.len(), path.display());
}
