//! Field-aligned Poseidon transcript tests (spec §4): encoding injectivity,
//! tag domain-split, GT chunking, and prover/verifier NARG roundtrip with
//! challenge agreement through the REAL `ProverState`/`VerifierState`
//! factories.

#![cfg(feature = "transcript-poseidon")]
#![expect(
    clippy::unwrap_used,
    reason = "test code; failures should panic loudly"
)]

use ark_bn254::Fr;
use ark_ff::Zero;
use jolt_field::Fr as JoltFr;
use jolt_transcript::{
    poseidon_prover_transcript, poseidon_verifier_transcript, prover_transcript,
    push_byte_rule_units, serialize_slice, verifier_transcript, CommitmentsMsg,
    DuplexSpongeInterface, Encoding, FieldFrameMsg, FsAbsorb, NativeChallenge, PoseidonSponge,
    ProverTranscript, RawBytesMsg, VerifierTranscript,
};

/// A Dory-GT-shaped commitment stand-in: 48 × u64 = 384 canonical bytes,
/// no length prefix (ark serializes fixed-size arrays without one).
type FakeGt = [u64; 48];

const SESSION: &[u8] = b"jolt-poseidon-field-aligned-test/v1";

fn squeeze1(s: &mut PoseidonSponge) -> Fr {
    let mut out = [Fr::zero(); 1];
    let _ = s.squeeze(&mut out);
    out[0]
}

fn absorb_msg<T: Encoding<[Fr]>>(s: &mut PoseidonSponge, msg: &T) {
    let _ = s.absorb(msg.encode().as_ref());
}

/// `absorb([a, b])` must differ from `absorb([a]) ; absorb([b])` — the
/// count-led tag binds message boundaries.
#[test]
fn field_frame_message_boundaries_bind() {
    let (a, b) = (Fr::from(123u64), Fr::from(456u64));
    let mut s1 = PoseidonSponge::new();
    absorb_msg(&mut s1, &FieldFrameMsg(vec![a, b]));
    let mut s2 = PoseidonSponge::new();
    absorb_msg(&mut s2, &FieldFrameMsg(vec![a]));
    absorb_msg(&mut s2, &FieldFrameMsg(vec![b]));
    assert_ne!(squeeze1(&mut s1), squeeze1(&mut s2));
}

/// An empty field frame (`[Fr(1), 0]`) must evolve the state differently
/// than a squeeze refill (`permute(0,0)`).
#[test]
fn empty_field_frame_distinct_from_squeeze_refill() {
    let mut s1 = PoseidonSponge::new();
    absorb_msg(&mut s1, &FieldFrameMsg(vec![]));
    let after_empty_frame = squeeze1(&mut s1);

    let mut s2 = PoseidonSponge::new();
    let _ = squeeze1(&mut s2); // permute(0,0) — what an aliasing empty frame would be
    let second_squeeze = squeeze1(&mut s2);

    assert_ne!(after_empty_frame, second_squeeze);
}

/// Tag domain-split: a byte message and a field frame with IDENTICAL payload
/// units must diverge purely on the `2L` (even) vs `2k+1` (odd) leading tag.
#[test]
fn byte_message_and_field_frame_tag_split() {
    // 62-byte payload = two 31-byte chunks decoding to units u1, u2.
    let mut payload = vec![0u8; 62];
    payload[0] = 17; // chunk 1 ↦ Fr(17)
    payload[31] = 99; // chunk 2 ↦ Fr(99)
    let byte_units = RawBytesMsg(payload).encode().as_ref().to_vec();
    let frame_units = FieldFrameMsg(vec![Fr::from(17u64), Fr::from(99u64)])
        .encode()
        .as_ref()
        .to_vec();

    // Both encode to 4 units (tag, u1, u2, pad); only the tags differ.
    assert_eq!(byte_units.len(), 4);
    assert_eq!(frame_units.len(), 4);
    assert_eq!(byte_units[1..], frame_units[1..]);
    assert_eq!(byte_units[0], Fr::from(124u64)); // 2 · 62
    assert_eq!(frame_units[0], Fr::from(5u64)); // 2 · 2 + 1

    let mut s1 = PoseidonSponge::new();
    let _ = s1.absorb(&byte_units);
    let mut s2 = PoseidonSponge::new();
    let _ = s2.absorb(&frame_units);
    assert_ne!(squeeze1(&mut s1), squeeze1(&mut s2));
}

/// A GT-sized (384-byte) payload chunks to exactly 14 units:
/// `[Fr(2·384), 12 × 31-byte chunks, 1 × 12-byte chunk]` — already even,
/// no padding.
#[test]
fn gt_sized_payload_chunks_to_exactly_14_units() {
    let mut units = Vec::new();
    push_byte_rule_units(&mut units, &[0xA5u8; 384]);
    assert_eq!(units.len(), 14);
    assert_eq!(units[0], Fr::from(768u64));

    // A commitments frame of k GTs leads with the count pair
    // `[Fr(2k+1), 0]`, then k pair-aligned 14-unit tag-led groups.
    let gts: Vec<FakeGt> = vec![[7u64; 48], [9u64; 48]];
    let frame_units = CommitmentsMsg(gts).encode().as_ref().to_vec();
    assert_eq!(frame_units.len(), 30);
    assert_eq!(frame_units[0], Fr::from(5u64)); // 2 · 2 + 1
    assert_eq!(frame_units[1], Fr::zero()); // count padded to a permute pair
    assert_eq!(frame_units[2], Fr::from(768u64));
    assert_eq!(frame_units[16], Fr::from(768u64));

    // The empty commitments frame stays count-led: `[Fr(1), pad]`, distinct
    // from the empty byte message's `Fr(0)` tag.
    let empty_units = CommitmentsMsg::<FakeGt>(vec![]).encode().as_ref().to_vec();
    assert_eq!(empty_units, vec![Fr::from(1u64), Fr::zero()]);
}

/// The frame-level count unit binds the partition of commitments into
/// frames: `[c1,c2] + [c3]` and `[c1] + [c2,c3]` carry the same units in the
/// same order at the group level, so without the count they would alias and
/// a NARG malleation re-partitioning adjacent frames would leave every
/// challenge unchanged.
#[test]
fn commitment_frame_repartition_diverges_challenges() {
    let instance = [0x33u8; 32];
    let (c1, c2, c3): (FakeGt, FakeGt, FakeGt) = ([1u64; 48], [2u64; 48], [3u64; 48]);

    let mut a = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    ProverTranscript::<PoseidonSponge>::public_message(&mut a, &CommitmentsMsg(vec![c1, c2]));
    ProverTranscript::<PoseidonSponge>::public_message(&mut a, &CommitmentsMsg(vec![c3]));
    let ca: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut a);

    let mut b = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    ProverTranscript::<PoseidonSponge>::public_message(&mut b, &CommitmentsMsg(vec![c1]));
    ProverTranscript::<PoseidonSponge>::public_message(&mut b, &CommitmentsMsg(vec![c2, c3]));
    let cb: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut b);

    assert_ne!(ca.0, cb.0, "re-partitioned commitment frames must diverge");
}

/// Mixed absorb/write/squeeze schedule through the REAL factories: the
/// verifier reads every frame back from the NARG, all challenges agree, the
/// NARG bytes are exactly the byte-sponge framing (8-byte LE length ‖
/// compressed payload), and `check_eof` passes.
#[test]
fn narg_roundtrip_mixed_schedule_challenges_agree() {
    let instance = [0x42u8; 32];
    let scalars: Vec<Fr> = (1..=5).map(|i| Fr::from(i * 31 + 7u64)).collect();
    let gts: Vec<FakeGt> = vec![[3u64; 48], [0xFFFF_FFFF_FFFF_FFFFu64; 48]];

    let mut p = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_scalar(&mut p, &Fr::from(2026u64));
    ProverTranscript::<PoseidonSponge>::prover_message(&mut p, &FieldFrameMsg(scalars.clone()));
    let p_c1: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut p);
    ProverTranscript::<PoseidonSponge>::prover_message(&mut p, &CommitmentsMsg(gts.clone()));
    ProverTranscript::<PoseidonSponge>::prover_message(&mut p, &CommitmentsMsg::<FakeGt>(vec![]));
    FsAbsorb::absorb_commitment(&mut p, &gts[0]);
    FsAbsorb::absorb_scalars(&mut p, &scalars);
    let p_c2: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut p);
    let narg = ProverTranscript::<PoseidonSponge>::narg_string(&p).to_vec();

    // The NARG transport is byte-identical to the byte-sponge `BytesMsg`
    // framing of the same frames.
    let mut expected = Vec::new();
    for body in [serialize_slice(&scalars), serialize_slice(&gts), Vec::new()] {
        expected.extend_from_slice(&(body.len() as u64).to_le_bytes());
        expected.extend_from_slice(&body);
    }
    assert_eq!(narg, expected, "NARG framing changed");

    let mut v = poseidon_verifier_transcript(SESSION, instance, PoseidonSponge::default(), &narg);
    FsAbsorb::absorb_scalar(&mut v, &Fr::from(2026u64));
    let frame: FieldFrameMsg =
        VerifierTranscript::<PoseidonSponge>::prover_message(&mut v).unwrap();
    assert_eq!(frame.0, scalars, "scalar frame reconstructed incorrectly");
    let v_c1: NativeChallenge = VerifierTranscript::<PoseidonSponge>::verifier_message(&mut v);
    let read_gts: CommitmentsMsg<FakeGt> =
        VerifierTranscript::<PoseidonSponge>::prover_message(&mut v).unwrap();
    assert_eq!(
        read_gts.0, gts,
        "commitments frame reconstructed incorrectly"
    );
    let presence: CommitmentsMsg<FakeGt> =
        VerifierTranscript::<PoseidonSponge>::prover_message(&mut v).unwrap();
    assert!(presence.0.is_empty(), "presence frame must be empty");
    FsAbsorb::absorb_commitment(&mut v, &gts[0]);
    FsAbsorb::absorb_scalars(&mut v, &scalars);
    let v_c2: NativeChallenge = VerifierTranscript::<PoseidonSponge>::verifier_message(&mut v);
    VerifierTranscript::<PoseidonSponge>::check_eof(v).unwrap();

    assert_eq!(p_c1.0, v_c1.0, "mid-schedule challenge diverged");
    assert_eq!(p_c2.0, v_c2.0, "final challenge diverged");
}

/// FIX #9 regression: on Poseidon, `absorb_field`/`absorb_field_slice` must
/// emit the SAME count-led field frame as `absorb_scalar`/`absorb_scalars`
/// (the byte-sponge trait defaults would diverge to the byte rule). No prover
/// in the gated suite calls `absorb_field` on Poseidon, so this equivalence is
/// otherwise unexercised — lock it here. `absorb_field` bounds on
/// `jolt_field::Field` ([`JoltFr`]); `absorb_scalar` on `CanonicalSerialize`
/// (`ark_bn254::Fr`); the same value through both must yield one challenge.
#[test]
fn absorb_field_matches_absorb_scalar_on_poseidon() {
    let instance = [0x9au8; 32];
    let n: u64 = 0x1234_5678_9abc_def0;

    let mut a = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_field(&mut a, &JoltFr::from(n));
    let ca: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut a);

    let mut b = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_scalar(&mut b, &Fr::from(n));
    let cb: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut b);

    assert_eq!(
        ca.0, cb.0,
        "absorb_field diverged from absorb_scalar on Poseidon"
    );

    let js: Vec<JoltFr> = (1..=4).map(|i| JoltFr::from(i * 97 + 5u64)).collect();
    let xs: Vec<Fr> = (1..=4).map(|i| Fr::from(i * 97 + 5u64)).collect();

    let mut c = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_field_slice(&mut c, &js);
    let cc: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut c);

    let mut d = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_scalars(&mut d, &xs);
    let cd: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut d);

    assert_eq!(
        cc.0, cd.0,
        "absorb_field_slice diverged from absorb_scalars on Poseidon"
    );
}

/// The unit-generic `prover_transcript`/`verifier_transcript` factories
/// dispatch to the Poseidon-specific ones for `PoseidonSponge`.
#[test]
fn generic_factories_dispatch_to_poseidon_path() {
    let instance = [0x07u8; 32];

    let mut a = prover_transcript(SESSION, instance, PoseidonSponge::default());
    let mut b = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    let ca: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut a);
    let cb: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut b);
    assert_eq!(ca.0, cb.0);

    let v = verifier_transcript(SESSION, instance, PoseidonSponge::default(), &[]);
    let mut v = v;
    let cv: NativeChallenge = VerifierTranscript::<PoseidonSponge>::verifier_message(&mut v);
    assert_eq!(cv.0, ca.0, "prover and verifier domain separators diverged");
    VerifierTranscript::<PoseidonSponge>::check_eof(v).unwrap();
}

/// Different sessions / instances diverge the transcript.
#[test]
fn domain_separator_binds_session_and_instance() {
    let mut base = poseidon_prover_transcript(SESSION, [1u8; 32], PoseidonSponge::default());
    let mut other_session =
        poseidon_prover_transcript(b"other-session", [1u8; 32], PoseidonSponge::default());
    let mut other_instance =
        poseidon_prover_transcript(SESSION, [2u8; 32], PoseidonSponge::default());

    let c0: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut base);
    let c1: NativeChallenge =
        ProverTranscript::<PoseidonSponge>::verifier_message(&mut other_session);
    let c2: NativeChallenge =
        ProverTranscript::<PoseidonSponge>::verifier_message(&mut other_instance);
    assert_ne!(c0.0, c1.0);
    assert_ne!(c0.0, c2.0);
}

/// `FieldFrameMsg` NARG reads reject non-canonical (≥ r) elements and bodies
/// that are not a multiple of 32 bytes — mirroring the native `read_all`
/// strictness — and reject truncation without advancing the cursor.
#[test]
fn field_frame_narg_rejects_malformed_bodies() {
    use jolt_transcript::NargDeserialize;

    // Non-canonical element (0xFF…FF ≥ r).
    let mut narg = 32u64.to_le_bytes().to_vec();
    narg.extend_from_slice(&[0xFF; 32]);
    let mut cursor: &[u8] = &narg;
    assert!(FieldFrameMsg::deserialize_from_narg(&mut cursor).is_err());
    assert_eq!(cursor.len(), narg.len(), "cursor must not advance on error");

    // Body length not a multiple of 32.
    let mut narg = 31u64.to_le_bytes().to_vec();
    narg.extend_from_slice(&[0u8; 31]);
    let mut cursor: &[u8] = &narg;
    assert!(FieldFrameMsg::deserialize_from_narg(&mut cursor).is_err());

    // Truncated body.
    let mut narg = 64u64.to_le_bytes().to_vec();
    narg.extend_from_slice(&[0u8; 32]);
    let mut cursor: &[u8] = &narg;
    assert!(FieldFrameMsg::deserialize_from_narg(&mut cursor).is_err());
}

/// The typed `FsAbsorb` methods and the raw message encodings agree: the
/// vocabulary surface and the codec layer cannot drift.
#[test]
fn typed_absorb_methods_match_message_encodings() {
    let instance = [0x55u8; 32];
    let x = Fr::from(40_961u64);

    // absorb_scalar == public_message(FieldFrameMsg([x]))
    let mut a = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_scalar(&mut a, &x);
    let mut b = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    ProverTranscript::<PoseidonSponge>::public_message(&mut b, &FieldFrameMsg(vec![x]));
    let ca: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut a);
    let cb: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut b);
    assert_eq!(ca.0, cb.0);

    // absorb_commitment == the byte rule over the compressed serialization
    // (one per-GT group; a lone commitment absorb carries no frame count, so
    // it is NOT a singleton `CommitmentsMsg` frame).
    let gt: FakeGt = [11u64; 48];
    let mut c = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    FsAbsorb::absorb_commitment(&mut c, &gt);
    let mut d = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    ProverTranscript::<PoseidonSponge>::public_message(
        &mut d,
        &RawBytesMsg(serialize_slice(&[gt])),
    );
    let cc: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut c);
    let cd: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut d);
    assert_eq!(cc.0, cd.0);

    let mut e = poseidon_prover_transcript(SESSION, instance, PoseidonSponge::default());
    ProverTranscript::<PoseidonSponge>::public_message(&mut e, &CommitmentsMsg(vec![gt]));
    let ce: NativeChallenge = ProverTranscript::<PoseidonSponge>::verifier_message(&mut e);
    assert_ne!(cc.0, ce.0, "a commitments frame leads with a count unit");
}
