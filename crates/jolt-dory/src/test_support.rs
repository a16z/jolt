//! Transcripts for the crate's unit tests.

use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, VerifierTranscript};

fn protocol() -> ProtocolId {
    ProtocolId::new::<Blake2b512>("jolt-dory/tests")
}

pub(crate) fn prover(session: &[u8]) -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&protocol(), session)
}

pub(crate) fn verifier<'a>(session: &[u8], narg: &'a [u8]) -> VerifierTranscript<'a, Blake2b512> {
    VerifierTranscript::new(&protocol(), session, narg)
}
