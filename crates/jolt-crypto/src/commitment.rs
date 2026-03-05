use std::fmt::Debug;

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Backend-agnostic vector commitment scheme.
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
    type Commitment: Clone
        + Copy
        + Debug
        + Default
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + for<'de> Deserialize<'de>;

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
