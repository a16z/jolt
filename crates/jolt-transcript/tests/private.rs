//! The clear-mode private-message seam, extension and base forms: values
//! round-trip, derived claims bind without proof bytes, and no pads are used.

#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_transcript::{
    Blake2b512, Channel, PadSlots, PrivateChannel, ProtocolId, ProverTranscript, TranscriptError,
    VerifierTranscript,
};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-transcript/private");
const SESSION: &[u8] = b"private";

fn values() -> [Fr; 3] {
    [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)]
}

fn prove(derived: Fr) -> (Vec<u8>, Fr) {
    let mut prover = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    prover.bind_derived::<Fr, Fr>(&derived).unwrap();
    prover.bind_derived_base(&derived).unwrap();
    assert!(prover.narg().is_empty());
    let mut sent = values();
    let (head, tail) = sent.split_at_mut(1);
    assert_eq!(
        prover.exchange_private::<Fr, Fr>(head).unwrap(),
        PadSlots::default()
    );
    assert_eq!(
        prover.exchange_private_base(tail).unwrap(),
        PadSlots::default()
    );
    assert_eq!(sent, values());
    let challenge: Fr = prover.challenge();
    (prover.finish(), challenge)
}

fn verify(proof: &[u8], derived: Fr) -> Result<([Fr; 3], Fr), TranscriptError> {
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, proof);
    verifier.bind_derived::<Fr, Fr>(&derived)?;
    verifier.bind_derived_base(&derived)?;
    let mut received = [Fr::from_u64(0); 3];
    let (head, tail) = received.split_at_mut(1);
    assert_eq!(
        verifier.exchange_private::<Fr, Fr>(head)?,
        PadSlots::default()
    );
    assert_eq!(verifier.exchange_private_base(tail)?, PadSlots::default());
    let challenge: Fr = verifier.challenge();
    verifier.finish()?;
    Ok((received, challenge))
}

#[test]
fn clear_exchange_round_trips_and_binds_derived_claims() {
    let derived = Fr::from_u64(11);
    let (proof, challenge) = prove(derived);
    assert_eq!(proof.len(), 3 * Fr::NUM_BYTES);
    assert_eq!(verify(&proof, derived).unwrap(), (values(), challenge));

    let (_, rebound) = verify(&proof, derived + Fr::from_u64(1)).unwrap();
    assert_ne!(rebound, challenge);
    assert_eq!(
        verify(&proof[..proof.len() - 1], derived),
        Err(TranscriptError::Truncated)
    );
}
