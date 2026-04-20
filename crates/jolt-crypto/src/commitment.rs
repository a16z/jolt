use std::fmt::Debug;

use jolt_field::Field;
use jolt_transcript::AppendToTranscript;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Base commitment abstraction: defines only the output type.
///
/// This is the root of the commitment trait hierarchy, shared by both
/// vector commitments ([`VectorCommitment`]) and
/// polynomial commitment schemes (`jolt_openings::CommitmentScheme`).
/// The `Output` associated type is the single piece of connective tissue
/// between these different levels of abstraction.
pub trait Commitment {
    /// The commitment value (e.g., a group element, a Merkle root, a lattice vector).
    type Output: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
}

/// Backend-agnostic vector commitment.
///
/// Extends [`Commitment`] with the ability to commit to a vector of field
/// elements with a blinding factor. Uses `Self::Output` from the supertrait
/// as the commitment value type.
pub trait VectorCommitment: Commitment + Clone + Send + Sync + 'static
where
    Self::Output: Copy + Default + AppendToTranscript + Serialize + for<'de> Deserialize<'de>,
{
    /// Transparent setup parameters (generators, public parameters, etc.).
    type Setup: Clone + Send + Sync;

    /// Maximum number of values this setup can commit to.
    #[must_use]
    fn capacity(setup: &Self::Setup) -> usize;

    /// Commits to `values` with the given `blinding` factor.
    ///
    /// # Panics
    ///
    /// May panic if `values.len()` exceeds [`Self::capacity()`].
    #[must_use]
    fn commit<F: Field>(setup: &Self::Setup, values: &[F], blinding: &F) -> Self::Output;

    /// Returns `true` if `commitment` opens to `(values, blinding)`.
    #[must_use]
    fn verify<F: Field>(
        setup: &Self::Setup,
        commitment: &Self::Output,
        values: &[F],
        blinding: &F,
    ) -> bool;
}

/// Additive homomorphism on commitment values over a scalar field `F`.
///
/// Captures the ability to linearly combine two commitments without
/// knowing the committed values:
/// ```text
/// linear_combine(c1, c2, s) = c1 ⊕ s ⊗ c2
/// ```
///
/// Required by Nova folding for instance-level commitment operations.
/// Not all commitment schemes have this property (e.g., hash-based schemes
/// do not). Pedersen and lattice-based schemes do.
///
/// Blanket-implemented for [`JoltGroup`](crate::JoltGroup) over any field
/// (via `scalar_mul` + addition). Non-group commitment types (e.g., lattice
/// vectors) can implement this trait directly for their native scalar field.
pub trait HomomorphicCommitment<F: Field>: Clone {
    /// Computes `c1 + scalar * c2`.
    #[must_use]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self;
}

/// Derives a commitment setup from a source setup (e.g., PCS SRS → Pedersen generators).
///
/// This is the bridge between a polynomial commitment scheme's structured
/// reference string and a vector commitment's setup parameters. Each PCS
/// implements this for the vector commitment setups it can derive.
///
/// Backend-agnostic: works for EC (Pedersen from Dory/KZG SRS), lattice
/// (matrix columns from lattice SRS), or hash-based (Merkle params) schemes.
pub trait DeriveSetup<Source> {
    fn derive(source: &Source, capacity: usize) -> Self;
}

impl<G: crate::JoltGroup, F: Field> HomomorphicCommitment<F> for G {
    #[inline]
    fn linear_combine(c1: &G, c2: &G, scalar: &F) -> G {
        *c1 + c2.scalar_mul(scalar)
    }
}
