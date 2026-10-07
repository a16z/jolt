//! Regenerates the checked-in `seeds/deser_commitment/` corpus from a genuine
//! Dory commitment and genuine argument strings, so mutation explores the
//! neighborhood of valid encodings instead of pure garbage.
//!
//! The opening seeds use the target's protocol id, session, and statement
//! (point `(1, ..., NUM_VARS)`, evaluation zero), so they parse in full and
//! reach `dory::verify`. Whether they also accept depends on the SRS: dory-pcs
//! caches a randomly generated URS per machine.
//!
//! Run explicitly, then commit the outputs:
//! `cargo nextest run --manifest-path crates/jolt-dory/fuzz/Cargo.toml --run-ignored only`

use std::fs;
use std::path::Path;

use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: usize = 2;
const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-dory-fuzz/deser");
const SESSION: &[u8] = b"fuzz-deser";

#[test]
#[ignore = "writes the checked-in seed corpus; run explicitly and commit the outputs"]
fn generate_deser_commitment_seeds() {
    let seed_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("seeds/deser_commitment");
    fs::create_dir_all(&seed_dir).expect("seed directory");
    let config = bincode::config::standard();

    let mut rng = ChaCha20Rng::seed_from_u64(0x5EED);
    let prover_setup = DoryScheme::setup_prover(NUM_VARS);
    let point: Vec<Fr> = (1..=NUM_VARS as u64).map(Fr::from_u64).collect();
    // Shift a random polynomial so it evaluates to zero at `point`.
    let random = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
    let shift = random.evaluate(&point);
    let poly = Polynomial::new(random.evaluations().iter().map(|&e| e - shift).collect());
    let eval = poly.evaluate(&point);
    assert_eq!(eval, Fr::from_u64(0));

    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).expect("commit");
    let commitment_bytes =
        bincode::serde::encode_to_vec(&commitment, config).expect("encode commitment");
    fs::write(seed_dir.join("valid-commitment-bincode"), commitment_bytes)
        .expect("write commitment seed");
    let mut atom = vec![0u8; <DoryCommitment as CanonicalBytes>::NUM_BYTES];
    commitment.to_bytes_le(&mut atom);
    fs::write(seed_dir.join("valid-commitment-atom"), atom).expect("write atom seed");

    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    DoryScheme::send_commitment(&commitment, &mut transcript);
    DoryScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut transcript,
    )
    .expect("open");
    fs::write(seed_dir.join("valid-opening-narg"), transcript.finish())
        .expect("write opening seed");

    let (commitment, hint) =
        DoryScheme::commit_zk(poly.evaluations(), &prover_setup).expect("commit_zk");
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    DoryScheme::send_commitment(&commitment, &mut transcript);
    let (_y_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript)
            .expect("open_zk");
    fs::write(seed_dir.join("valid-zk-opening-narg"), transcript.finish())
        .expect("write ZK opening seed");
}
