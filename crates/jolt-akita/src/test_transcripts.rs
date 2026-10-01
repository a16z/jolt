//! Standalone transcripts for exercising openings outside a Jolt proof.

use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-akita/test");

pub(crate) fn new_prover_transcript(session: &[u8]) -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&PROTOCOL, session)
}

pub(crate) fn new_verifier_transcript<'a>(
    session: &[u8],
    proof: &'a [u8],
) -> VerifierTranscript<'a, Blake2b512> {
    VerifierTranscript::new(&PROTOCOL, session, proof)
}

/// The verifier consumed exactly the prover's argument string and ended in the
/// same sponge state.
#[expect(clippy::expect_used, reason = "test assertion")]
pub(crate) fn assert_transcripts_agree(
    mut prover: ProverTranscript<Blake2b512>,
    mut verifier: VerifierTranscript<'_, Blake2b512>,
) {
    assert_eq!(
        prover.challenge_bytes::<32>(),
        verifier.challenge_bytes::<32>()
    );
    verifier
        .finish()
        .expect("the verifier consumes the whole proof");
}
