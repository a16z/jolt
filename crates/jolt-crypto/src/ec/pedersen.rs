use jolt_field::Field;
use serde::{Deserialize, Serialize};

use super::group::JoltGroup;
use crate::commitment::{Commitment, VectorCommitment};

/// Pedersen vector commitment scheme, generic over any `JoltGroup`.
///
/// Commitment: `C = Σᵢ values[i] * message_generators[i] + blinding * blinding_generator`
///
/// This provides a blanket `VectorCommitment` implementation for any group
/// that implements `JoltGroup`, so concrete backends (BN254, etc.) inherit
/// Pedersen commitments.
#[derive(Clone, Debug)]
pub struct Pedersen<G: JoltGroup> {
    _marker: std::marker::PhantomData<G>,
}

/// Setup parameters for Pedersen commitments: a vector of message generators
/// and a separate blinding generator.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PedersenSetup<G: JoltGroup> {
    pub message_generators: Vec<G>,
    pub blinding_generator: G,
}

impl<G: JoltGroup> PedersenSetup<G> {
    /// Constructs setup from externally-provided generators (e.g., from a PCS URS).
    ///
    /// # Panics
    ///
    /// Panics if `message_generators` is empty.
    pub fn new(message_generators: Vec<G>, blinding_generator: G) -> Self {
        assert!(
            !message_generators.is_empty(),
            "Pedersen setup requires at least one message generator"
        );
        Self {
            message_generators,
            blinding_generator,
        }
    }
}

impl<G: JoltGroup> Commitment for Pedersen<G> {
    type Output = G;
}

impl<G: JoltGroup> VectorCommitment for Pedersen<G> {
    type Setup = PedersenSetup<G>;

    #[inline]
    fn capacity(setup: &Self::Setup) -> usize {
        setup.message_generators.len()
    }

    /// Computes `C = Σᵢ values[i] · message_generators[i] + blinding · blinding_generator`.
    ///
    /// Short inputs commit over a prefix of the generators, which is equivalent
    /// to zero-padding: `commit(setup, [a], r) == commit(setup, [a, 0], r)`.
    /// This matters for binding: Pedersen only binds inputs of a fixed length.
    /// Callers that accept variable-length inputs must pin the length
    /// independently (e.g., append it to the Fiat-Shamir transcript) to
    /// avoid accepting two different-length openings of the same commitment.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() > Self::capacity(setup)`.
    fn commit<F: Field>(setup: &Self::Setup, values: &[F], blinding: &F) -> G {
        assert!(
            values.len() <= setup.message_generators.len(),
            "values length ({}) exceeds generator count ({})",
            values.len(),
            setup.message_generators.len(),
        );
        let msg = G::msm(&setup.message_generators[..values.len()], values);
        let blind = setup.blinding_generator.scalar_mul(blinding);
        msg + blind
    }

    fn verify<F: Field>(setup: &Self::Setup, commitment: &G, values: &[F], blinding: &F) -> bool {
        *commitment == Self::commit(setup, values, blinding)
    }
}
