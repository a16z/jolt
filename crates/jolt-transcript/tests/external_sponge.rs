#![cfg(feature = "transcript-keccak")]
#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, BytesMsg, FieldTranscript, Keccak,
    ProverTranscript, RuntimePrimeChallenge, UniformBytes, VerifierTranscript,
    RUNTIME_PRIME_WIDE_BYTES,
};
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface};

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
        Self(Keccak::init(session_id))
    }
}

// SAFETY: Initialization, absorbs, and squeezes preserve Keccak's behavior.
unsafe impl UniformBytes for ExternalSponge {}

#[derive(Debug, PartialEq, Eq)]
struct Challenge(u32);

impl Challenge {
    const MODULUS: u32 = 257;
}

impl RuntimePrimeChallenge for Challenge {
    fn modulus() -> u128 {
        u128::from(Self::MODULUS)
    }

    fn from_le_bytes_wide(bytes: &[u8; RUNTIME_PRIME_WIDE_BYTES]) -> Self {
        Self(bytes.iter().rev().fold(0, |acc, &byte| {
            (acc * 256 + u32::from(byte)) % Self::MODULUS
        }))
    }
}

#[test]
fn external_sponge_draws_matching_challenges() {
    let session = protocol_tag(b"jolt/external-sponge-test");
    let statement = [0x42u8; 32];
    let mut prover = prover_transcript::<ExternalSponge>(&session, &statement);
    prover.send(&BytesMsg(b"commitment".to_vec()));
    let expected: Challenge = prover.challenge().unwrap();

    let mut verifier =
        verifier_transcript::<ExternalSponge>(&session, &statement, prover.narg_string());
    let message: BytesMsg = verifier.prover_message().unwrap();
    assert_eq!(message.as_slice(), b"commitment");
    let challenge: Challenge = verifier.challenge().unwrap();
    assert_eq!(challenge, expected);
    VerifierTranscript::check_eof(verifier).unwrap();
}
