//! Frozen known-answer vectors, one per sponge: a fixed script exercising every
//! message kind, challenge kind, and grinding. Any change to the protocol-id
//! derivation, framing, encoding, or challenge mapping changes these bytes.

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
    prover.send_all(&[11u32, 13]);
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
    assert_eq!(verifier.receive_n::<u32>(2).unwrap(), vec![11, 13]);
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

/// The argument string differs across sponges only in its `u32` grinding nonce.
const NARG_PREFIX: &str = "07000000000000000000000000000000000000000000000000000000000000000b0000000d000000080000007661726961626c65";

#[test]
fn blake2b_known_answers() {
    assert_vectors(
        "blake2b",
        run::<Blake2b512>(),
        (
            &format!("{NARG_PREFIX}78"),
            "f14acf6a0e8500083aff06c75c0e690988c707180bdd447ee79c61ae1a7af622b4b8f1266fc13197f631ede2d2c8407a823d4388e79fba572325bdbd113cdf140edab4fa7e6bf134a286a792d365eea0671007657c372b32315757aca6c9990f",
        ),
    );
}

#[test]
fn keccak_known_answers() {
    assert_vectors(
        "keccak",
        run::<Keccak>(),
        (
            &format!("{NARG_PREFIX}1f"),
            "d8255bbe1b4ab7de6af9a08d7f52525044f0faa2d1f77159e6f5dbca1a897210b54afa2edf054d0a9e9d21f565d947dba3c96d3a662cbc5dc8f371998554c30845ffeaf9f80a897b327911715039879241ea7de1a6b98b5546c5f7fe87110275",
        ),
    );
}

#[test]
fn poseidon_known_answers() {
    assert_vectors(
        "poseidon",
        run::<PoseidonSponge>(),
        (
            &format!("{NARG_PREFIX}0d"),
            "6186b569421edf0a7490f9020cf5124637e90bcd2ff7c905ec433482548d541557d1d4a63d6ea9a8847ad61506d5ebba81550033c00c4991a4a586a3db288810bfbea94f9af0dd473dff8a154a7cc1f5e6f1f6dad5bfdbcf1025b3510cd8ab2a",
        ),
    );
}
