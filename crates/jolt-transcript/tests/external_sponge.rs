#![cfg(feature = "transcript-keccak")]
#![expect(clippy::unwrap_used, reason = "tests")]

use std::cell::Cell;

use jolt_transcript::{
    grind_bits_for_loss, protocol_tag, prover_transcript, session_id, verifier_transcript,
    BytesMsg, Encoding, FieldTranscript, Keccak, ProverTranscript, RuntimePrimeChallenge,
    UniformBytes, VerifierState, VerifierTranscript, RUNTIME_PRIME_WIDE_BYTES,
};
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface};

const PERSONALIZATION: [u8; 32] = protocol_tag(b"jolt/test/personalized-keccak");

#[derive(Clone)]
struct ExternalSponge(Keccak);

impl DuplexSpongeInterface for ExternalSponge {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        let _ = self.0.absorb(input);
        self
    }

    fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        let _ = self.0.squeeze(output);
        self
    }
}

impl DuplexSpongeInit for ExternalSponge {
    fn init(session_id: &[u8; 32]) -> Self {
        let mut sponge = Keccak::init(session_id);
        let _ = sponge.absorb(&PERSONALIZATION);
        Self(sponge)
    }
}

// SAFETY: The fixed personalization extends Keccak's session prefix. Cloning,
// absorption, and squeezing preserve the underlying duplex behavior.
unsafe impl UniformBytes for ExternalSponge {}

#[derive(Debug, PartialEq, Eq)]
struct Challenge(u32);

thread_local! {
    static MODULUS: Cell<u32> = const { Cell::new(257) };
}

impl RuntimePrimeChallenge for Challenge {
    fn modulus() -> u128 {
        u128::from(MODULUS.get())
    }

    fn from_le_bytes_wide(bytes: &[u8; RUNTIME_PRIME_WIDE_BYTES]) -> Self {
        let modulus = MODULUS.get();
        Self(
            bytes
                .iter()
                .rev()
                .fold(0, |acc, &byte| (acc * 256 + u32::from(byte)) % modulus),
        )
    }
}

#[test]
fn external_sponge_draws_matching_challenges() {
    MODULUS.set(257);
    let session = protocol_tag(b"jolt/external-sponge-test");
    let statement = [0x42u8; 32];
    let commitment = BytesMsg(b"commitment");

    let mut sponge = Keccak::init(session_id().as_bytes());
    let _ = sponge.absorb(&PERSONALIZATION);
    let _ = sponge.absorb(&session);
    let _ = sponge.absorb(BytesMsg(&statement).encode().as_ref());
    let _ = sponge.absorb(commitment.encode().as_ref());
    let expected = Challenge::from_le_bytes_wide(&sponge.squeeze_array());

    let mut prover = prover_transcript::<ExternalSponge>(&session, &statement);
    prover.send(&commitment);
    let challenge: Challenge = prover.challenge().unwrap();
    assert_eq!(challenge, expected);

    let mut verifier =
        verifier_transcript::<ExternalSponge>(&session, &statement, prover.narg_string());
    let message: BytesMsg = verifier.prover_message().unwrap();
    assert_eq!(message.as_slice(), commitment.as_slice());
    let challenge: Challenge = verifier.challenge().unwrap();
    assert_eq!(challenge, expected);
    VerifierTranscript::check_eof(verifier).unwrap();
}

#[test]
fn verifier_prices_work_from_each_proofs_modulus() {
    let session = protocol_tag(b"jolt/runtime-prime-pricing-test");
    let loss = 65;
    let security_bits = 6;
    let target = u128::from(loss) << security_bits;

    for prime in [257u32, 263] {
        MODULUS.set(prime);
        let mut verifier =
            verifier_transcript::<ExternalSponge>(&session, &prime.to_le_bytes(), &[]);
        let challenge_set =
            <VerifierState<'_, ExternalSponge> as FieldTranscript<Challenge>>::challenge_set();
        let bits = grind_bits_for_loss(loss, challenge_set, security_bits).unwrap();

        assert!(u128::from(prime) << bits >= target, "prime={prime}");
        assert!(
            bits == 0 || u128::from(prime) << (bits - 1) < target,
            "prime={prime}"
        );
        let challenge: Challenge = verifier.challenge().unwrap();
        assert!(challenge.0 < prime);
        VerifierTranscript::check_eof(verifier).unwrap();
    }
}
