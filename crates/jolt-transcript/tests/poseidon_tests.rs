//! Tests for the Poseidon transcript, which packs each message's bytes into
//! BN254 `Fr` elements and draws `Fr` challenges as squeezed elements.

#![expect(clippy::unwrap_used, reason = "tests")]

use ark_bn254::Fr as BnFr;
use ark_ff::PrimeField;
use jolt_field::{CanonicalBytes, Fr};
use jolt_transcript::{
    prover_transcript, session_id, verifier_transcript, BytesMsg, ChallengeSponge, FieldTranscript,
    FrUnit, PoseidonSponge, ProverTranscript, Transcript, VerifierTranscript, MAX_CHALLENGE_BYTES,
};
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, ProverState};

const SESSION: &[u8] = b"jolt/poseidon/v1";
const INSTANCE: [u8; 32] = [0x5a; 32];

const NATIVE_VECTOR: [[u8; 32]; 2] = [
    [
        0x3f, 0x85, 0x29, 0x7b, 0xe8, 0x4d, 0x5f, 0xe5, 0xc0, 0x9d, 0x7a, 0xc0, 0x12, 0x57, 0xd6,
        0x90, 0xe4, 0x59, 0xfd, 0xf6, 0x1b, 0x92, 0x1e, 0x94, 0x7f, 0x03, 0x1d, 0xe0, 0xc5, 0xdf,
        0x29, 0x08,
    ],
    [
        0xfa, 0x74, 0x0d, 0x13, 0x9c, 0x87, 0xb9, 0xf1, 0xd8, 0x42, 0xe8, 0xc5, 0xa0, 0x6a, 0x63,
        0xcc, 0xd7, 0xc5, 0xb4, 0xeb, 0x85, 0xf0, 0x54, 0xf7, 0x4e, 0x24, 0x6d, 0xed, 0x8c, 0x6e,
        0xeb, 0x27,
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

fn challenges(transcript: &mut impl FieldTranscript<Fr>, len: usize) -> Vec<Fr> {
    transcript.challenge_vector(len).unwrap()
}

/// A public message and a prover message.
fn sample_prover() -> ProverState<PoseidonSponge> {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    prover.append(&BytesMsg(b"public".to_vec()));
    ProverTranscript::prover_message(&mut prover, &BytesMsg(b"jolt".to_vec()));
    prover
}

/// Pins the session and instance packing, the message packing, the NARG
/// bytes and two native challenges.
#[test]
fn native_known_vector() {
    let mut prover = sample_prover();
    let drawn = challenges(&mut prover, 2);
    let mut narg = 4u64.to_le_bytes().to_vec();
    narg.extend_from_slice(b"jolt");
    assert_eq!(prover.narg_string(), narg);
    assert_eq!(
        drawn.iter().copied().map(le_bytes).collect::<Vec<_>>(),
        NATIVE_VECTOR
    );
}

/// A byte string packed as the module documents: the length, then 31-byte
/// little-endian chunks.
fn packed(bytes: &[u8]) -> Vec<FrUnit> {
    let mut units = vec![FrUnit::from(fr(bytes.len() as u64))];
    for chunk in bytes.chunks(31) {
        units.push(FrUnit::from(Fr::from(BnFr::from_le_bytes_mod_order(chunk))));
    }
    units
}

/// `bytes` with an eight-byte little-endian length in front.
fn framed(bytes: &[u8]) -> Vec<u8> {
    [&(bytes.len() as u64).to_le_bytes()[..], bytes].concat()
}

/// Replays the sample transcript on a bare sponge from the documented
/// packing: the instance is one packed message, each message is packed on its
/// own, and challenges are the squeezed elements themselves.
#[test]
fn challenges_are_squeezed_elements_after_the_documented_packing() {
    let instance = [framed(SESSION), framed(&INSTANCE)].concat();
    let mut sponge = PoseidonSponge::init(session_id().as_bytes());
    let _ = sponge
        .absorb(&packed(&instance))
        .absorb(&packed(&framed(b"public")))
        .absorb(&packed(&framed(b"jolt")));
    let expected: Vec<Fr> = sponge.squeeze_array::<3>().map(Fr::from).to_vec();

    let mut prover = sample_prover();
    assert_eq!(challenges(&mut prover, 3), expected);
}

#[test]
fn round_trips_byte_and_scalar_messages() {
    let bodies: [&[u8]; 5] = [b"", b"x", &[0xab; 23], &[0xcd; 24], &[0xef; 100]];
    let scalars = [fr(0), fr(1), -fr(1), fr(u64::MAX)];

    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    prover.append(&le_bytes(fr(3)));
    let mut sent = Vec::new();
    for body in bodies {
        ProverTranscript::prover_message(&mut prover, &BytesMsg(body.to_vec()));
        sent.push(challenges(&mut prover, 1));
    }
    for scalar in scalars {
        ProverTranscript::prover_message(&mut prover, &le_bytes(scalar));
    }
    sent.push(challenges(&mut prover, 5));
    assert!(challenges(&mut prover, 0).is_empty());

    let mut verifier =
        verifier_transcript::<PoseidonSponge>(SESSION, &INSTANCE, prover.narg_string());
    verifier.append(&le_bytes(fr(3)));
    let mut read = Vec::new();
    for body in bodies {
        let message = VerifierTranscript::prover_message::<BytesMsg>(&mut verifier).unwrap();
        assert_eq!(message.0, body);
        read.push(challenges(&mut verifier, 1));
    }
    for scalar in scalars {
        let message = VerifierTranscript::prover_message::<[u8; 32]>(&mut verifier).unwrap();
        assert_eq!(message, le_bytes(scalar));
    }
    read.push(challenges(&mut verifier, 5));
    assert_eq!(read, sent);
    verifier.check_eof().unwrap();
}

fn draw_after(
    session: &[u8],
    instance: [u8; 32],
    absorb: impl FnOnce(&mut ProverState<PoseidonSponge>),
) -> Vec<Fr> {
    let mut prover = prover_transcript::<PoseidonSponge>(session, &instance);
    absorb(&mut prover);
    challenges(&mut prover, 1)
}

/// A different message, session or instance changes the challenges. Each
/// message is packed on its own, so splitting raw bytes across messages, or
/// appending a zero byte, does too, even without a length in the encoding.
/// The same messages give the same challenges, and swapping two messages
/// changes them.
#[test]
fn challenges_bind_messages_session_and_instance() {
    let base = draw_after(SESSION, INSTANCE, |t| t.append(&[1u8, 2]));
    let others = [
        draw_after(SESSION, INSTANCE, |t| {
            t.append(&[1u8]);
            t.append(&[2u8]);
        }),
        draw_after(SESSION, INSTANCE, |t| t.append(&[1u8, 2, 0])),
        draw_after(SESSION, INSTANCE, |t| t.append(&[1u8, 3])),
        draw_after(b"jolt/poseidon/v2", INSTANCE, |t| t.append(&[1u8, 2])),
        draw_after(SESSION, [0x5b; 32], |t| t.append(&[1u8, 2])),
    ];
    for other in others {
        assert_ne!(base, other);
    }

    let in_order = |first: u8, second: u8| {
        draw_after(SESSION, INSTANCE, |t| {
            t.append(&[first]);
            t.append(&[second]);
        })
    };
    assert_eq!(in_order(1, 2), in_order(1, 2));
    assert_ne!(in_order(1, 2), in_order(2, 1));
}

/// A truncated message is rejected and poisons the verifier, so later draws
/// and `check_eof` fail. Trailing bytes fail `check_eof`.
#[test]
fn rejects_truncated_and_trailing_input() {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    ProverTranscript::prover_message(&mut prover, &BytesMsg(b"payload".to_vec()));
    let narg = prover.narg_string().to_vec();

    for cut in 1..=narg.len() {
        let truncated = &narg[..narg.len() - cut];
        let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, &INSTANCE, truncated);
        assert!(VerifierTranscript::prover_message::<BytesMsg>(&mut verifier).is_err());
        assert!(FieldTranscript::<Fr>::challenge_vector(&mut verifier, 1).is_err());
        assert!(verifier.check_eof().is_err());
    }

    let trailing = [narg.as_slice(), &[0]].concat();
    let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, &INSTANCE, &trailing);
    let message = VerifierTranscript::prover_message::<BytesMsg>(&mut verifier).unwrap();
    assert_eq!(message.0, b"payload");
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

        let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
        assert!(FieldTranscript::<Fr>::challenge_vector(&mut prover, len).is_err());
        let mut fresh = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
        assert_eq!(challenges(&mut prover, 1), challenges(&mut fresh, 1));

        let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, &INSTANCE, &[]);
        assert!(FieldTranscript::<Fr>::challenge_vector(&mut verifier, len).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

/// Native challenges are full-width: across many draws the top half takes
/// values a 128-bit challenge never would.
#[test]
fn challenges_are_full_width() {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    assert!(challenges(&mut prover, 64)
        .iter()
        .any(|&c| le_bytes(c)[16..].iter().any(|&b| b != 0)));
}

/// A short challenge squeezes two elements `x0` and `x1`, keeps the low `b`
/// bits of `x0 + p * x1`, and squeezes nothing more. The verifier draws the
/// same value. The expected value is computed on unbounded integers.
#[test]
fn short_challenge_keeps_the_low_bits_of_two_elements() {
    use jolt_field::CanonicalEncoding;
    use num_bigint::BigUint;

    let int = |x: Fr| BigUint::from_bytes_le(&le_bytes(x));
    let modulus = int(-fr(1)) + 1u32;
    for (loss, bits) in [(1u64, 128u32), (4, 130), (u64::MAX, 192)] {
        let mut widest = 0;
        for round in 0..64u32 {
            let statement = round.to_be_bytes();
            let mut short = prover_transcript::<PoseidonSponge>(SESSION, &statement);
            let mut manual = prover_transcript::<PoseidonSponge>(SESSION, &statement);
            let value: Fr = short.challenge_short_scalar(loss).unwrap();

            let drawn = challenges(&mut manual, 3);
            let expected =
                (int(drawn[0]) + &modulus * int(drawn[1])) % (BigUint::from(1u32) << bits);
            let mut bytes = expected.to_bytes_le();
            bytes.resize(bits.div_ceil(8) as usize, 0);
            assert_eq!(value, Fr::from_bytes_le_reduced(&bytes));
            assert_eq!(challenges(&mut short, 1), [drawn[2]]);

            let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, &statement, &[]);
            let verified: Fr = verifier.challenge_short_scalar(loss).unwrap();
            assert_eq!(verified, value);
            widest = widest.max(value.num_bits());
        }
        // The top bit is set with probability 1/2 per draw.
        assert_eq!(widest, bits, "loss={loss}");
    }
}

/// A zero loss factor is refused before anything is squeezed, and poisons the
/// verifier.
#[test]
fn short_challenge_refuses_zero_loss_and_poisons() {
    let mut prover = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    assert!(FieldTranscript::<Fr>::challenge_short_scalar(&mut prover, 0).is_err());
    let mut fresh = prover_transcript::<PoseidonSponge>(SESSION, &INSTANCE);
    assert_eq!(challenges(&mut prover, 1), challenges(&mut fresh, 1));

    let mut verifier = verifier_transcript::<PoseidonSponge>(SESSION, &INSTANCE, &[]);
    assert!(FieldTranscript::<Fr>::challenge_short_scalar(&mut verifier, 0).is_err());
    assert!(verifier.check_eof().is_err());
}
