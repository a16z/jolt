//! Tests for the Poseidon transcript: the legacy facade and the native
//! spongefish surface, which absorbs and squeezes BN254 `Fr` elements.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use ark_ff::{BigInteger, PrimeField};
use jolt_field::{CanonicalBytes, Fr};
use jolt_transcript::{
    prover_transcript, session_id, verifier_transcript, BytesMsg, ChallengeSponge, FieldMsg,
    FrUnit, PoseidonSponge, PoseidonTranscript, ProverTranscript, VerifierTranscript,
    MAX_CHALLENGE_BYTES,
};
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, Encoding};

type Pos = PoseidonTranscript;

transcript_tests!(Pos);

const SESSION: &[u8] = b"jolt/poseidon/v1";
const INSTANCE: [u8; 32] = [0x5a; 32];

const LEGACY_VECTOR: [u8; 32] = [
    0x69, 0x6d, 0xf6, 0x8a, 0x7c, 0x9b, 0xdd, 0xd3, 0x46, 0xa0, 0xc3, 0xe9, 0xca, 0x8b, 0x25, 0x76,
    0x8e, 0xc1, 0xac, 0x62, 0x76, 0x09, 0x34, 0x9c, 0xbd, 0xc3, 0x2b, 0xa3, 0xab, 0x3b, 0x34, 0x03,
];

const NATIVE_VECTOR: [[u8; 32]; 2] = [
    [
        0x3c, 0x78, 0xd3, 0x4f, 0x26, 0xf7, 0x10, 0xea, 0x55, 0x9b, 0x1e, 0x78, 0xc8, 0x0e, 0x1c,
        0x06, 0x7b, 0xb0, 0xfc, 0xbb, 0x6b, 0x06, 0xc1, 0x12, 0x98, 0x93, 0xff, 0x75, 0x94, 0xac,
        0x16, 0x26,
    ],
    [
        0xfb, 0x77, 0x0b, 0x8c, 0xc0, 0x04, 0x4a, 0x4b, 0xcc, 0x65, 0x1b, 0x96, 0x90, 0x3b, 0xa4,
        0xfb, 0xfa, 0x55, 0xed, 0x81, 0xae, 0xc3, 0x69, 0x53, 0x3a, 0xfd, 0x18, 0x25, 0xa4, 0xd0,
        0x6a, 0x21,
    ],
];

fn le_bytes(value: Fr) -> [u8; 32] {
    let mut out = [0u8; 32];
    value.to_bytes_le(&mut out);
    out
}

fn fr(value: u64) -> Fr {
    Fr::from(value)
}

/// Pins the session derived from `PROTOCOL_ID`, the packed label, the framed
/// and packed append, and the native challenge.
#[test]
fn legacy_known_vector() {
    let mut transcript = PoseidonTranscript::new(b"Jolt");
    Transcript::append_bytes(&mut transcript, &12345u64.to_be_bytes());
    let challenge = Transcript::challenge(&mut transcript);
    assert_eq!(le_bytes(challenge), LEGACY_VECTOR);
}

/// One prover message of each kind, then two challenges.
fn sample_prover() -> spongefish::ProverState<PoseidonSponge> {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
    prover.public_message(&BytesMsg(b"public".to_vec()));
    prover.prover_message(&BytesMsg(b"jolt".to_vec()));
    prover.prover_message(&FieldMsg(fr(7)));
    prover
}

/// Pins the native layout: the session and instance packing, `BytesMsg` and
/// `FieldMsg` encodings, the NARG bytes, and two native challenges.
#[test]
fn native_known_vector() {
    let mut prover = sample_prover();
    let challenges = ProverTranscript::field_challenges::<Fr>(&mut prover, 2).unwrap();
    let mut narg = 4u64.to_le_bytes().to_vec();
    narg.extend_from_slice(b"jolt");
    narg.extend_from_slice(&le_bytes(fr(7)));
    assert_eq!(prover.narg_string(), narg);
    assert_eq!(
        challenges.iter().copied().map(le_bytes).collect::<Vec<_>>(),
        NATIVE_VECTOR
    );
}

/// Byte strings packed as the module documents: the length, then 31-byte
/// little-endian chunks.
fn packed(bytes: &[u8]) -> Vec<FrUnit> {
    let mut units = vec![FrUnit::from(fr(bytes.len() as u64))];
    for chunk in bytes.chunks(31) {
        units.push(FrUnit::from(Fr::from(
            ark_bn254::Fr::from_le_bytes_mod_order(chunk),
        )));
    }
    units
}

/// Replays the sample transcript on a bare sponge from the documented
/// packing: challenges are the squeezed elements themselves.
#[test]
fn challenges_are_squeezed_elements_after_the_documented_packing() {
    let mut sponge = PoseidonSponge::init(session_id().as_bytes());
    let _ = sponge
        .absorb(&[packed(SESSION), packed(&INSTANCE)].concat())
        .absorb(&packed(b"public"))
        .absorb(&packed(b"jolt"))
        .absorb(&[FrUnit::from(fr(7))]);
    let expected: Vec<Fr> = sponge.squeeze_array::<3>().map(Fr::from).to_vec();

    let mut prover = sample_prover();
    assert_eq!(
        ProverTranscript::field_challenges::<Fr>(&mut prover, 3).unwrap(),
        expected
    );
    assert_eq!(
        <(BytesMsg, [u8; 32]) as Encoding<FrUnit>>::encode(&(BytesMsg(SESSION.to_vec()), INSTANCE))
            .as_ref(),
        [packed(SESSION), packed(&INSTANCE)].concat()
    );
}

#[test]
fn round_trips_field_and_byte_messages() {
    let bodies: [&[u8]; 5] = [b"", b"x", &[0xab; 31], &[0xcd; 32], &[0xef; 100]];
    let scalars = [fr(0), fr(1), -fr(1), fr(u64::MAX)];

    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
    prover.public_message(&FieldMsg(fr(3)));
    let mut sent = Vec::new();
    for body in bodies {
        prover.prover_message(&BytesMsg(body.to_vec()));
        sent.push(ProverTranscript::field_challenges::<Fr>(&mut prover, 1).unwrap());
    }
    for scalar in scalars {
        prover.prover_message(&FieldMsg(scalar));
    }
    sent.push(ProverTranscript::field_challenges::<Fr>(&mut prover, 5).unwrap());
    assert!(ProverTranscript::field_challenges::<Fr>(&mut prover, 0)
        .unwrap()
        .is_empty());

    let mut verifier =
        verifier_transcript::<PoseidonSponge>(SESSION, INSTANCE, prover.narg_string());
    verifier.public_message(&FieldMsg(fr(3)));
    let mut read = Vec::new();
    for body in bodies {
        assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, body);
        read.push(VerifierTranscript::field_challenges::<Fr>(&mut verifier, 1).unwrap());
    }
    for scalar in scalars {
        assert_eq!(verifier.prover_message::<FieldMsg<Fr>>().unwrap().0, scalar);
    }
    read.push(VerifierTranscript::field_challenges::<Fr>(&mut verifier, 5).unwrap());
    assert_eq!(read, sent);
    verifier.check_eof().unwrap();
}

/// A different public message, session or instance changes the challenges.
#[test]
fn challenges_bind_messages_session_and_instance() {
    let draw = |session: &[u8], instance: [u8; 32], body: &[u8]| {
        let mut prover = prover_transcript::<PoseidonSponge>(session, instance);
        prover.public_message(&BytesMsg(body.to_vec()));
        ProverTranscript::field_challenges::<Fr>(&mut prover, 1).unwrap()
    };
    let base = draw(SESSION, INSTANCE, b"");
    assert_ne!(base, draw(SESSION, INSTANCE, &[0]));
    assert_ne!(base, draw(b"jolt/poseidon/v2", INSTANCE, b""));
    assert_ne!(base, draw(SESSION, [0x5b; 32], b""));
}

fn field_narg(value: Fr) -> Vec<u8> {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
    prover.prover_message(&FieldMsg(value));
    prover.narg_string().to_vec()
}

/// Truncated and non-canonical elements are rejected, and the rejection
/// poisons the verifier so `check_eof` fails too. Trailing bytes fail
/// `check_eof`.
#[test]
fn rejects_truncated_non_canonical_and_trailing_input() {
    let narg = field_narg(fr(42));
    let modulus = ark_bn254::Fr::MODULUS.to_bytes_le();
    let mut above = modulus.clone();
    above[0] += 1;
    let rejected = [
        narg[..31].to_vec(),
        Vec::new(),
        modulus,
        above,
        vec![0xff; 32],
    ];
    for narg in rejected {
        let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, INSTANCE, &narg);
        assert!(verifier.prover_message::<FieldMsg<Fr>>().is_err());
        assert!(VerifierTranscript::field_challenges::<Fr>(&mut verifier, 1).is_err());
        assert!(verifier.check_eof().is_err());
    }

    let max = field_narg(-fr(1));
    let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, INSTANCE, &max);
    assert_eq!(verifier.prover_message::<FieldMsg<Fr>>().unwrap().0, -fr(1));
    verifier.check_eof().unwrap();

    let trailing = [narg.as_slice(), &[0]].concat();
    let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, INSTANCE, &trailing);
    assert_eq!(verifier.prover_message::<FieldMsg<Fr>>().unwrap().0, fr(42));
    assert!(verifier.check_eof().is_err());
}

/// The cap counts 32 bytes per element; a larger request fails before
/// touching the sponge and poisons the verifier.
#[test]
fn oversized_challenge_messages_fail_and_poison() {
    let cap = MAX_CHALLENGE_BYTES / 32;
    assert_eq!(
        <PoseidonSponge as ChallengeSponge<Fr>>::challenge_units(cap),
        Some(cap)
    );
    for len in [cap + 1, usize::MAX] {
        assert_eq!(
            <PoseidonSponge as ChallengeSponge<Fr>>::challenge_units(len),
            None
        );

        let mut prover = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
        assert!(ProverTranscript::field_challenges::<Fr>(&mut prover, len).is_err());
        let mut fresh = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
        assert_eq!(
            ProverTranscript::field_challenges::<Fr>(&mut prover, 1).unwrap(),
            ProverTranscript::field_challenges::<Fr>(&mut fresh, 1).unwrap()
        );

        let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, INSTANCE, &[]);
        assert!(VerifierTranscript::field_challenges::<Fr>(&mut verifier, len).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

/// Native challenges are full-width: across many draws the top byte takes
/// values a 128-bit challenge never would.
#[test]
fn challenges_are_full_width() {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, INSTANCE);
    let challenges = ProverTranscript::field_challenges::<Fr>(&mut prover, 64).unwrap();
    assert!(challenges
        .iter()
        .any(|&c| le_bytes(c)[16..].iter().any(|&b| b != 0)));
    let mut legacy = PoseidonTranscript::new(b"width");
    let wide = (0..64).any(|_| le_bytes(Transcript::challenge(&mut legacy))[16..] != [0; 16]);
    assert!(wide);
}
