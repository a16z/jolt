//! Canonical, size-bounded (de)serialization for [`JoltProof`].
//!
//! A standalone proof arrives as untrusted bytes from a prover. Two properties
//! matter at this boundary:
//!
//! 1. **Bounded work.** A malformed length prefix must not drive an unbounded
//!    allocation. The proof contains many length-prefixed `Vec`s (RA commitment
//!    vectors, per-round sumcheck vectors, the BlindFold row/commitment
//!    vectors); without a ceiling, a prover can claim astronomically long
//!    vectors. (One specific inner allocation — the Dory proof round count — is
//!    already capped at the element level by `MAX_SERIALIZED_PROOF_ROUNDS` in
//!    `jolt-dory`; this is the systematic outer bound covering every other
//!    `Vec`.)
//! 2. **A single accepted encoding.** Pinning one wire format and rejecting
//!    trailing bytes removes proof malleability.
//!
//! Element-level validation (curve points in-subgroup, field elements reduced)
//! is enforced independently by the `serde` implementations of the field and
//! group types, which route through `CanonicalDeserialize` with validation.

use serde::{de::DeserializeOwned, Serialize};

use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;

use crate::{proof::JoltProof, VerifierError};

/// Hard ceiling on accepted serialized proof size, enforced both before
/// decoding (input length) and during decoding (the bincode read limit).
///
/// Jolt proofs are succinct — polylogarithmic in the trace length — so even
/// large executions serialize well under this bound; it exists only to cap the
/// deserialization work a malicious proof can induce. Callers that can compute
/// a tighter bound from their preprocessing may pass it to
/// [`JoltProof::from_canonical_bytes_bounded`].
pub const MAX_PROOF_BYTES: usize = 1 << 27; // 128 MiB

impl<PCS, VC, ZkProof> JoltProof<PCS, VC, ZkProof>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    /// Decodes a proof from its canonical byte encoding.
    ///
    /// Rejects inputs larger than [`MAX_PROOF_BYTES`] before allocating, caps
    /// allocation during decoding at the same bound, and rejects any trailing
    /// bytes after a complete proof.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, VerifierError>
    where
        Self: DeserializeOwned,
    {
        Self::from_canonical_bytes_bounded(bytes, MAX_PROOF_BYTES)
    }

    /// Like [`from_canonical_bytes`](Self::from_canonical_bytes) but with a
    /// caller-supplied upper bound on the input length. The bound is clamped to
    /// [`MAX_PROOF_BYTES`], which always remains the absolute allocation ceiling
    /// enforced during decoding.
    pub fn from_canonical_bytes_bounded(
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<Self, VerifierError>
    where
        Self: DeserializeOwned,
    {
        let max_bytes = max_bytes.min(MAX_PROOF_BYTES);
        if bytes.len() > max_bytes {
            return Err(VerifierError::ProofTooLarge {
                got: bytes.len(),
                max: max_bytes,
            });
        }

        let config = bincode::config::standard().with_limit::<MAX_PROOF_BYTES>();
        let (proof, consumed) = bincode::serde::decode_from_slice::<Self, _>(bytes, config)
            .map_err(|error| VerifierError::ProofDeserializationFailed {
                reason: error.to_string(),
            })?;
        if consumed != bytes.len() {
            return Err(VerifierError::TrailingProofBytes {
                consumed,
                total: bytes.len(),
            });
        }
        Ok(proof)
    }

    /// Encodes the proof to its canonical byte form, the inverse of
    /// [`from_canonical_bytes`](Self::from_canonical_bytes).
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VerifierError>
    where
        Self: Serialize,
    {
        let config = bincode::config::standard().with_limit::<MAX_PROOF_BYTES>();
        bincode::serde::encode_to_vec(self, config).map_err(|error| {
            VerifierError::ProofSerializationFailed {
                reason: error.to_string(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;

    use crate::proof::JoltProof;

    fn assert_canonical_bytes_capable<T: serde::Serialize + serde::de::DeserializeOwned>() {}

    #[test]
    fn dory_pedersen_proof_supports_canonical_bytes() {
        // Compile-time assurance that the production (Dory PCS, Pedersen VC)
        // instantiation satisfies the `serde` bounds that `from_canonical_bytes`
        // and `to_canonical_bytes` require, so the canonical decode boundary is
        // usable for real proofs (the `core-fixtures` integration round-trip
        // exercises it dynamically once the compat layer compiles).
        assert_canonical_bytes_capable::<JoltProof<DoryScheme, Pedersen<Bn254G1>>>();
    }
}
