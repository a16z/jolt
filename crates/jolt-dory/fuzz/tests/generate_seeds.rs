//! Regenerates the checked-in `seeds/deser_commitment/` corpus from a genuine
//! serialized Dory commitment and proof, so mutation explores the
//! neighborhood of valid encodings instead of pure garbage.
//!
//! Run explicitly, then commit the outputs:
//! `cargo test --manifest-path crates/jolt-dory/fuzz/Cargo.toml -- --ignored`

use std::fs;
use std::path::Path;

use jolt_dory::DoryScheme;
use jolt_field::{Fr, RandomSampling};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
#[ignore = "writes the checked-in seed corpus; run explicitly and commit the outputs"]
fn generate_deser_commitment_seeds() {
    let seed_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("seeds/deser_commitment");
    fs::create_dir_all(&seed_dir).expect("seed directory");
    let config = bincode::config::standard();

    let num_vars = 2;
    let mut rng = ChaCha20Rng::seed_from_u64(0x5EED);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        DoryScheme::commit(poly.evaluations(), &prover_setup).expect("commit");
    let mut transcript = Blake2bTranscript::new(b"seed-gen");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut transcript)
        .expect("open");

    let commitment_bytes =
        bincode::serde::encode_to_vec(&commitment, config).expect("encode commitment");
    fs::write(seed_dir.join("valid-commitment-bincode"), commitment_bytes)
        .expect("write commitment seed");
    let proof_bytes = bincode::serde::encode_to_vec(&proof, config).expect("encode proof");
    fs::write(seed_dir.join("valid-proof-bincode"), proof_bytes).expect("write proof seed");
}
