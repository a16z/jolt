//! Frozen known-answer vectors, one per sponge: a fixed script exercising every
//! message kind, challenge kind, and grinding. Any change to the protocol-id
//! derivation, framing, encoding, or challenge mapping changes these bytes.

#![cfg(all(
    feature = "transcript-blake2b",
    feature = "transcript-keccak",
    feature = "transcript-poseidon"
))]
#![expect(clippy::unwrap_used, reason = "test crate")]

use jolt_field::{CanonicalBytes, Fr};
use jolt_transcript::{
    Blake2b512, Channel, Keccak, PoseidonSponge, ProtocolId, ProverTranscript, Sponge,
    VerifierTranscript,
};

/// The script's argument string and the bytes of every challenge it draws.
fn run<H: Sponge>() -> (Vec<u8>, Vec<u8>) {
    let protocol = ProtocolId::new::<H>("jolt-transcript/known-answers");
    let mut prover = ProverTranscript::<H>::new(&protocol, b"session");
    let mut challenges = Vec::new();
    prover.public_bytes(b"public input");
    prover.public(&Fr::from(5u64));
    prover.send(&Fr::from(7u64));
    prover.send_all(&[11u64, 13]);
    prover.send_bounded_bytes(b"variable", 16).unwrap();
    challenges.extend(prover.challenge::<Fr>().to_bytes_le_vec());
    challenges.extend(prover.challenge_small::<Fr>().to_bytes_le_vec());
    let _nonce = prover.grind(6).unwrap();
    challenges.extend(prover.challenge_bytes::<32>());
    let narg = prover.finish();

    let mut verifier = VerifierTranscript::<H>::new(&protocol, b"session", &narg);
    let mut replayed = Vec::new();
    verifier.public_bytes(b"public input");
    verifier.public(&Fr::from(5u64));
    assert_eq!(verifier.receive::<Fr>().unwrap(), Fr::from(7u64));
    assert_eq!(verifier.receive_n::<u64>(2).unwrap(), vec![11, 13]);
    assert_eq!(verifier.receive_bounded_bytes(16).unwrap(), b"variable");
    replayed.extend(verifier.challenge::<Fr>().to_bytes_le_vec());
    replayed.extend(verifier.challenge_small::<Fr>().to_bytes_le_vec());
    let _nonce = verifier.check_grind(6).unwrap();
    replayed.extend(verifier.challenge_bytes::<32>());
    verifier.finish().unwrap();
    assert_eq!(replayed, challenges);

    (narg, challenges)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

fn assert_vectors(name: &str, (narg, challenges): (Vec<u8>, Vec<u8>), expected: (&str, &str)) {
    assert_eq!(hex(&narg), expected.0, "{name} argument string");
    assert_eq!(hex(&challenges), expected.1, "{name} challenges");
}

/// The argument string differs across sponges only in its grinding nonce.
const NARG_PREFIX: &str = "07000000000000000000000000000000000000000000000000000000000000000b000000000000000d00000000000000080000007661726961626c65";

#[test]
fn blake2b_known_answers() {
    assert_vectors(
        "blake2b",
        run::<Blake2b512>(),
        (
            &format!("{NARG_PREFIX}ad01"),
            "3d4e089306afb69df886095eee15fbf14517c4ff924ec6bfd09265fc6a8a60054bb2626cc97cccb74e359d255207da9e39f4eee144c71404fe870f184ac4e31b422aaeb6c1a4720024b98c7f60fe220acf9f414231a6d2fd4dea709ae2c4eb14",
        ),
    );
}

#[test]
fn keccak_known_answers() {
    assert_vectors(
        "keccak",
        run::<Keccak>(),
        (
            &format!("{NARG_PREFIX}8101"),
            "87b238e77e1881d46c35f627091e902ad1936d867b6c2fa7765547f44db00a230329d92cd97029872e99afbf8b9bdd57c59c13e503303e0c10e5a79fc642e11224bf861e6ff0b9bfa0c72d5ebcfe46562d97d675ea1b3814a87f1ebcaf118eb3",
        ),
    );
}

#[test]
fn poseidon_known_answers() {
    assert_vectors(
        "poseidon",
        run::<PoseidonSponge>(),
        (
            &format!("{NARG_PREFIX}49"),
            "0be942b2a23ad3aa6fd8abf1580f85e78b2df6d4bf46d6564202c4f16cf2f022ce5db4a5e3f7288c0b37e0912b8214568272d5160be2592b4c87e9a74c374304fc9b0b01515cf511c4480ab9d67d3b1b88dba661536bf44d7b380cf04dd32604",
        ),
    );
}
