//! The clear-mode private-message seam, extension and base forms: values
//! round-trip, derived claims bind without proof bytes, and no pads are used.

#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_algebra::{CanonicalBytes, Prime128Offset275 as Fp, Ring};
use jolt_transcript::{
    Blake2b512, Channel, ClaimDomain, ClaimMismatch, ClearDomain, PadSlots, PrivateChannel,
    ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-transcript/private");
const SESSION: &[u8] = b"private";

fn values() -> [Fp; 3] {
    [Fp::from_u64(3), Fp::from_u64(5), Fp::from_u64(7)]
}

fn prove(derived: Fp) -> (Vec<u8>, Fp) {
    let mut prover = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    prover.bind_derived::<Fp, Fp>(&derived).unwrap();
    prover.bind_derived_base(&derived).unwrap();
    assert!(prover.narg().is_empty());
    let mut sent = values();
    let (head, tail) = sent.split_at_mut(1);
    assert_eq!(
        prover.exchange_private::<Fp, Fp>(head).unwrap(),
        PadSlots::default()
    );
    assert_eq!(
        prover.exchange_private_base(tail).unwrap(),
        PadSlots::default()
    );
    assert_eq!(sent, values());
    let challenge: Fp = prover.challenge();
    (prover.finish(), challenge)
}

fn verify(proof: &[u8], derived: Fp) -> Result<([Fp; 3], Fp), TranscriptError> {
    let mut verifier = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, proof);
    verifier.bind_derived::<Fp, Fp>(&derived)?;
    verifier.bind_derived_base(&derived)?;
    let mut received = [Fp::from_u64(0); 3];
    let (head, tail) = received.split_at_mut(1);
    assert_eq!(
        verifier.exchange_private::<Fp, Fp>(head)?,
        PadSlots::default()
    );
    assert_eq!(verifier.exchange_private_base(tail)?, PadSlots::default());
    let challenge: Fp = verifier.challenge();
    verifier.finish()?;
    Ok((received, challenge))
}

#[test]
fn clear_exchange_round_trips_and_binds_derived_claims() {
    let derived = Fp::from_u64(11);
    let (proof, challenge) = prove(derived);
    assert_eq!(proof.len(), 3 * Fp::NUM_BYTES);
    assert_eq!(verify(&proof, derived).unwrap(), (values(), challenge));

    let (_, rebound) = verify(&proof, derived + Fp::from_u64(1)).unwrap();
    assert_ne!(rebound, challenge);
    assert_eq!(
        verify(&proof[..proof.len() - 1], derived),
        Err(TranscriptError::Truncated)
    );
}

/// Akita's stage-1 leaf check, claim = Σ_{k≤4} c_k·r^k, lowered as Akita
/// `specs/whiteout-akita.md` §4.4 writes it: one product slot u = r·r, then
/// the final product lands on the claim.
fn stage1_leaf<D: ClaimDomain<Fp>>(
    domain: &mut D,
    r: D::Value,
    c: [Fp; 5],
    claim: D::Value,
) -> Result<(), D::Error> {
    let u = domain.product(r.clone(), r.clone())?;
    let y = domain.add(
        &domain.add(&domain.scale(&u, c[4]), &domain.scale(&r, c[3])),
        &domain.constant(c[2]),
    );
    let target = domain.add(
        &claim,
        &domain.add(&domain.scale(&r, -c[1]), &domain.constant(-c[0])),
    );
    domain.product_into(u, y, target)
}

#[test]
fn clear_domain_compares_checks() {
    let c = [3, 5, 7, 11, 13].map(Fp::from_u64);
    let r = Fp::from_u64(17);
    let claim = c
        .iter()
        .rev()
        .fold(Fp::from_u64(0), |sum, &coefficient| sum * r + coefficient);
    assert_eq!(stage1_leaf(&mut ClearDomain, r, c, claim), Ok(()));
    assert_eq!(
        stage1_leaf(&mut ClearDomain, r, c, claim + Fp::from_u64(1)),
        Err(ClaimMismatch)
    );
    assert_eq!(
        ClaimDomain::<Fp>::check_zero(&mut ClearDomain, Fp::from_u64(0)),
        Ok(())
    );
    assert_eq!(ClearDomain.check_zero(Fp::from_u64(1)), Err(ClaimMismatch));
}
