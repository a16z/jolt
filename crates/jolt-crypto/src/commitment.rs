use std::fmt::Debug;

use jolt_field::Field;
use jolt_transcript::AppendToTranscript;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Base commitment abstraction: defines only the output type.
///
/// This is the root of the commitment trait hierarchy, shared by both
/// vector commitments ([`JoltCommitment`], `VectorCommitment`) and
/// polynomial commitment schemes (`jolt_openings::CommitmentScheme`).
/// The `Output` associated type is the single piece of connective tissue
/// between these different levels of abstraction.
pub trait Commitment {
    /// The commitment value (e.g., a group element, a Merkle root, a lattice vector).
    type Output: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
}

/// Backend-agnostic vector commitment.
///
/// Abstracts the ability to commit to a vector of field elements with a
/// blinding factor. ZK protocols (BlindFold, sumcheck) should be generic
/// over this trait rather than hardcoded to Pedersen or elliptic curves.
///
/// The `Setup` associated type represents transparent, shared parameters
/// (e.g., generators from a URS, lattice parameters). Setup data is
/// expected to be derivable from or shared with a polynomial commitment
/// scheme's structured reference string.
pub trait JoltCommitment: Clone + Send + Sync + 'static {
    /// Transparent setup parameters (generators, public parameters, etc.).
    type Setup: Clone + Send + Sync;

    /// The commitment output (e.g., a group element, a lattice vector).
    ///
    /// Requires [`AppendToTranscript`] so commitments can be absorbed
    /// into Fiat-Shamir transcripts during ZK sumcheck.
    type Commitment: Clone
        + Copy
        + Debug
        + Default
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + for<'de> Deserialize<'de>
        + AppendToTranscript;

    /// Maximum number of values this setup can commit to.
    fn capacity(setup: &Self::Setup) -> usize;

    /// Commits to `values` with the given `blinding` factor.
    ///
    /// # Panics
    ///
    /// May panic if `values.len()` exceeds [`Self::capacity()`].
    fn commit<F: Field>(setup: &Self::Setup, values: &[F], blinding: &F) -> Self::Commitment;

    /// Returns `true` if `commitment` opens to `(values, blinding)`.
    fn verify<F: Field>(
        setup: &Self::Setup,
        commitment: &Self::Commitment,
        values: &[F],
        blinding: &F,
    ) -> bool;
}
