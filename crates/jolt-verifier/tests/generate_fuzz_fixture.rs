//! Writes the checked-in fuzz fixture consumed by the `jolt-verifier` fuzz
//! workspace: a bincode bundle of a genuine, accepted muldiv verification
//! case (preprocessing, public I/O, proof, trusted-advice commitment) using
//! the public serde types. The fuzz targets load it into a `OnceLock`, so the
//! expensive host-side proof generation happens here once, not per iteration.
//!
//! Run explicitly and commit the output:
//! `cargo test -p jolt-verifier --features prover-fixtures --test \
//!   generate_fuzz_fixture -- --ignored`

#![cfg(all(feature = "prover-fixtures", not(feature = "zk")))]

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
fn generate_muldiv_fuzz_fixture() {
    let case = support::verifier_fixtures::standard_muldiv_case();
    support::assert_accepts(case.verify());

    let bundle: FuzzBundle = (
        case.preprocessing,
        case.public_io,
        case.proof,
        case.trusted_advice_commitment,
    );
    let bytes = bincode::serde::encode_to_vec(&bundle, bincode::config::standard())
        .expect("serialize fuzz bundle");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("fixtures")
        .join("muldiv-bundle.bin");
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
    >(
        &decoded.0, &decoded.1, &decoded.2, decoded.3.as_ref(),
    ));

    println!("wrote {} bytes to {}", bytes.len(), path.display());
}
