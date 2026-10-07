//! Prover/verifier symmetry, rejection, and wire-format pins for the NARG
//! transcript.

#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]

use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, Ring};
use jolt_transcript::{
    Blake2b512, Channel, Keccak, Nonce, PoseidonSponge, ProtocolId, ProverTranscript, Sponge,
    TranscriptError, VerifierTranscript, GRINDING_NONCE_SLACK_BITS,
};
use spongefish::Encoding as _;

const SESSION: &[u8] = b"narg-tests";

fn protocol<H: Sponge>() -> ProtocolId {
    ProtocolId::new::<H>("jolt-transcript/tests")
}

/// A toy protocol exercising every message kind. The verifier's only
/// acceptance check is the final product, so tampering is detected only
/// through the challenges the tampered bytes feed.
fn prove<H: Sponge>() -> Vec<u8> {
    let mut prover = ProverTranscript::<H>::new(&protocol::<H>(), SESSION);
    prover.public(&Fr::from_u64(11));
    prover.public_bytes(b"public statement");
    let x = Fr::from_u64(5);
    prover.send(&x);
    prover.send_all(&[Fr::from_u64(6), Fr::from_u64(7)]);
    prover.send_bytes(&[1, 2, 3]);
    prover.send_bounded_bytes(&[9; 10], 16).unwrap();
    let r: Fr = prover.challenge();
    let s: Fr = prover.challenge_small();
    let _nonce = prover.grind(6).unwrap();
    prover.send(&(x * r + s));
    prover.finish()
}

fn verify<H: Sponge>(narg: &[u8]) -> Result<(), TranscriptError> {
    let mut verifier = VerifierTranscript::<H>::new(&protocol::<H>(), SESSION, narg);
    verifier.public(&Fr::from_u64(11));
    verifier.public_bytes(b"public statement");
    let x: Fr = verifier.receive()?;
    let _pair: Vec<Fr> = verifier.receive_n(2)?;
    let _bytes = verifier.receive_bytes(3)?;
    let _bounded = verifier.receive_bounded_bytes(16)?;
    let r: Fr = verifier.challenge();
    let s: Fr = verifier.challenge_small();
    let _nonce = verifier.check_grind(6)?;
    let y: Fr = verifier.receive()?;
    if y != x * r + s {
        return Err(TranscriptError::NonCanonical);
    }
    verifier.finish()
}

fn for_each_sponge(
    check: impl Fn(&dyn Fn() -> Vec<u8>, &dyn Fn(&[u8]) -> Result<(), TranscriptError>),
) {
    check(&prove::<Blake2b512>, &|narg| verify::<Blake2b512>(narg));
    check(&prove::<Keccak>, &|narg| verify::<Keccak>(narg));
    check(&prove::<PoseidonSponge>, &|narg| {
        verify::<PoseidonSponge>(narg)
    });
}

#[test]
fn honest_proof_verifies() {
    for_each_sponge(|prove, verify| assert_eq!(verify(&prove()), Ok(())));
}

#[test]
fn every_byte_flip_rejects() {
    for_each_sponge(|prove, verify| {
        let narg = prove();
        for index in 0..narg.len() {
            let mut tampered = narg.clone();
            tampered[index] ^= 1;
            assert!(verify(&tampered).is_err(), "flip at byte {index} accepted");
        }
    });
}

#[test]
fn trailing_and_truncated_proofs_reject() {
    for_each_sponge(|prove, verify| {
        let mut narg = prove();
        narg.push(0);
        assert_eq!(verify(&narg), Err(TranscriptError::TrailingBytes));
        narg.truncate(narg.len() - 2);
        assert_eq!(verify(&narg), Err(TranscriptError::Truncated));
    });
}

#[test]
fn sponges_and_protocol_names_separate_domains() {
    fn first_challenge<H: Sponge>(protocol: &ProtocolId, session: &[u8]) -> Fr {
        ProverTranscript::<H>::new(protocol, session).challenge()
    }
    let blake = first_challenge::<Blake2b512>(&protocol::<Blake2b512>(), SESSION);
    assert_ne!(
        blake,
        first_challenge::<Keccak>(&protocol::<Keccak>(), SESSION)
    );
    assert_ne!(
        blake,
        first_challenge::<Blake2b512>(&ProtocolId::new::<Blake2b512>("other"), SESSION)
    );
    assert_ne!(
        blake,
        first_challenge::<Blake2b512>(&protocol::<Blake2b512>(), b"other")
    );
    assert_ne!(
        ProtocolId::new::<Blake2b512>("ab").as_bytes(),
        ProtocolId::new::<Blake2b512>("a").as_bytes()
    );
}

#[test]
fn public_bytes_are_framed() {
    let protocol = protocol::<Blake2b512>();
    let mut split = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    split.public_bytes(b"ab");
    split.public_bytes(b"c");
    let mut joined = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    joined.public_bytes(b"a");
    joined.public_bytes(b"bc");
    assert_ne!(
        split.challenge_bytes::<32>(),
        joined.challenge_bytes::<32>()
    );
}

#[test]
fn failed_receive_poisons_without_consuming() {
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION);
    prover.send(&7u32);
    let narg = prover.finish();
    let mut verifier =
        VerifierTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION, &narg);
    assert_eq!(verifier.receive::<Fr>(), Err(TranscriptError::Truncated));
    assert_eq!(verifier.remaining(), narg.len());
    assert_eq!(verifier.receive::<u32>(), Err(TranscriptError::Poisoned));
    assert_eq!(verifier.finish(), Err(TranscriptError::Poisoned));
}

#[test]
fn non_canonical_field_element_rejects() {
    let modulus_bytes = {
        let minus_one = -Fr::from_u64(1);
        let mut bytes = minus_one.to_bytes_le_vec();
        bytes[0] += 1;
        bytes
    };
    let mut verifier =
        VerifierTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION, &modulus_bytes);
    assert_eq!(verifier.receive::<Fr>(), Err(TranscriptError::NonCanonical));
}

#[test]
fn bounds_are_checked_before_reading() {
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION);
    assert_eq!(
        prover.send_bounded_bytes(&[0; 5], 4),
        Err(TranscriptError::OutOfBounds)
    );
    prover.send_bounded_bytes(&[0; 5], 5).unwrap();
    let narg = prover.finish();
    let mut verifier =
        VerifierTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION, &narg);
    assert_eq!(
        verifier.receive_bounded_bytes(4),
        Err(TranscriptError::OutOfBounds)
    );

    let mut verifier =
        VerifierTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION, &narg);
    assert_eq!(
        verifier.receive_n::<Fr>(usize::MAX / Fr::NUM_BYTES),
        Err(TranscriptError::Truncated)
    );
}

#[test]
fn grinding_rejects_a_stronger_claim_and_unsupported_difficulty() {
    let protocol = protocol::<Blake2b512>();
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    let nonce = prover.grind(4).unwrap();
    let narg = prover.finish();

    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, SESSION, &narg);
    assert_eq!(verifier.check_grind(4), Ok(nonce));
    assert_eq!(verifier.finish(), Ok(()));

    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    assert_eq!(prover.grind(26), Err(TranscriptError::UnsupportedGrinding));
    assert_eq!(prover.grind(0), Ok(0));
    assert!(prover.narg().is_empty());
}

#[test]
fn small_challenge_decodes_the_next_squeezed_bytes() {
    let mut small = ProverTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION);
    let mut raw = ProverTranscript::<Blake2b512>::new(&protocol::<Blake2b512>(), SESSION);
    for _ in 0..8 {
        let expected = Fr::from_challenge_bytes(&raw.challenge_bytes::<16>());
        assert_eq!(small.challenge_small::<Fr>(), expected);
    }
}

/// A grind squeezes a seed and sends a `u32` nonce; the verifier rejects a
/// nonce outside the search range and a nonce that fails the predicate.
#[test]
fn grinding_rejects_out_of_range_and_failing_nonces() {
    let protocol = protocol::<Blake2b512>();
    let bits = 6;
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    let nonce = prover.grind(bits).unwrap();
    let honest = prover.finish();
    assert_eq!(honest, Nonce(nonce).encode().as_ref());

    let out_of_range = Nonce(1u32 << (bits + GRINDING_NONCE_SLACK_BITS))
        .encode()
        .as_ref()
        .to_vec();
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, SESSION, &out_of_range);
    assert_eq!(
        verifier.check_grind(bits),
        Err(TranscriptError::OutOfBounds)
    );

    let failing = Nonce((0..nonce).next().unwrap_or(nonce + 1))
        .encode()
        .as_ref()
        .to_vec();
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, SESSION, &failing);
    assert_eq!(
        verifier.check_grind(bits),
        Err(TranscriptError::GrindingRejected)
    );

    // The same nonce under a different seed (another session) is rejected.
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, b"other", &honest);
    assert_eq!(
        verifier.check_grind(bits),
        Err(TranscriptError::GrindingRejected)
    );
}

#[test]
fn exchange_is_send_for_the_prover_and_receive_for_the_verifier() {
    fn shared<C: Channel>(channel: &mut C, value: &mut Fr) -> Fr {
        channel.exchange(value).unwrap();
        channel.challenge()
    }
    let protocol = protocol::<Blake2b512>();
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    let prover_challenge = shared(&mut prover, &mut Fr::from_u64(42));
    let narg = prover.finish();
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, SESSION, &narg);
    let mut slot = Fr::from_u64(0);
    assert_eq!(shared(&mut verifier, &mut slot), prover_challenge);
    assert_eq!(slot, Fr::from_u64(42));
    assert_eq!(verifier.finish(), Ok(()));
}

#[cfg(feature = "logging")]
#[test]
fn prover_and_verifier_log_identical_sites() {
    use jolt_transcript::SiteId;
    let protocol = protocol::<Blake2b512>();
    let mut prover = ProverTranscript::<Blake2b512>::new(&protocol, SESSION);
    prover.site(SiteId::label("statement"));
    prover.public(&Fr::from_u64(3));
    prover.site(SiteId::label("message"));
    prover.send(&Fr::from_u64(4));
    let _: Fr = prover.challenge();
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&protocol, SESSION, prover.narg());
    verifier.site(SiteId::label("statement"));
    verifier.public(&Fr::from_u64(3));
    verifier.site(SiteId::label("message"));
    let _: Fr = verifier.receive().unwrap();
    let _: Fr = verifier.challenge();
    assert_eq!(prover.events(), verifier.events());
    assert_eq!(prover.events()[1].narg, Some(0..32));
}
