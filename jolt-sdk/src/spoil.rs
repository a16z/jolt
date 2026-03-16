use crate::hcf;

/// Unwrap `Result`, spoiling the proof on error instead of producing a valid proof of panic.
///
/// Use when a malicious prover should not be able to produce any proof if the condition fails
/// (e.g. cryptographic assertions). Do NOT use for input validation or expected error cases.
pub trait UnwrapOrSpoilProof<T> {
    fn unwrap_or_spoil_proof(self) -> T;
}

impl<T, E> UnwrapOrSpoilProof<T> for Result<T, E> {
    #[inline(always)]
    fn unwrap_or_spoil_proof(self) -> T {
        match self {
            Ok(v) => v,
            Err(_) => hcf(),
        }
    }
}
