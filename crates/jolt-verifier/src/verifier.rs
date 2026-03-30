//! Low-level verification primitives reused by [`verify`](crate::verify).

use jolt_openings::{
    AdditivelyHomomorphic, OpeningReduction, OpeningsError, RlcReduction, VerifierClaim,
};
use jolt_transcript::Transcript;

use crate::error::JoltError;

/// Verifies batch PCS opening proofs via RLC reduction.
pub fn verify_openings<PCS, T>(
    claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
    opening_proofs: &[PCS::Proof],
    verifier_setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), JoltError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = PCS::Field>,
{
    let reduced = <RlcReduction as OpeningReduction<PCS>>::reduce_verifier(claims, &(), transcript)
        .map_err(JoltError::Opening)?;

    if reduced.len() != opening_proofs.len() {
        return Err(JoltError::Opening(OpeningsError::VerificationFailed));
    }

    for (claim, proof) in reduced.iter().zip(opening_proofs.iter()) {
        PCS::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            verifier_setup,
            transcript,
        )
        .map_err(JoltError::Opening)?;
    }

    Ok(())
}
