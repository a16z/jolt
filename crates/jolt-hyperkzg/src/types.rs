//! Commitment, proof, and setup types for HyperKZG.
//!
//! All types are generic over `P: PairingGroup` — no arkworks leakage.

use jolt_crypto::{JoltGroup, PairingGroup};
use serde::{Deserialize, Serialize};

/// Commitment to a multilinear polynomial: a single G1 element.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGCommitment<P: PairingGroup> {
    pub(crate) point: P::G1,
}

impl<P: PairingGroup> Copy for HyperKZGCommitment<P> {}

#[allow(clippy::expl_impl_clone_on_copy)]
impl<P: PairingGroup> Clone for HyperKZGCommitment<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: PairingGroup> std::fmt::Debug for HyperKZGCommitment<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HyperKZGCommitment")
            .field("point", &self.point)
            .finish()
    }
}

impl<P: PairingGroup> PartialEq for HyperKZGCommitment<P> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl<P: PairingGroup> Eq for HyperKZGCommitment<P> {}

impl<P: PairingGroup> Default for HyperKZGCommitment<P> {
    fn default() -> Self {
        Self {
            point: P::G1::identity(),
        }
    }
}

/// Opening proof for the HyperKZG protocol.
///
/// - `com`: intermediate polynomial commitments from the Gemini folding (ell - 1 elements)
/// - `w`: KZG witness commitments for the three evaluation points `[r, -r, r^2]`
/// - `v`: evaluations of all intermediate polynomials at the three points
///   (`v[t][k]` = polynomial k evaluated at point t)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGProof<P: PairingGroup> {
    pub com: Vec<P::G1>,
    pub w: Vec<P::G1>,
    pub v: Vec<Vec<P::ScalarField>>,
}

/// Prover setup: SRS G1 and G2 powers.
///
/// G1 powers: `[g1, beta * g1, beta^2 * g1, ..., beta^n * g1]`
/// G2 powers: `[g2, beta * g2]` (only two needed for KZG verification).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::G2: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGProverSetup<P: PairingGroup> {
    pub(crate) g1_powers: Vec<P::G1>,
    pub(crate) g2_powers: Vec<P::G2>,
}

/// Verifier setup: the four G1/G2 elements needed for pairing checks.
///
/// - `g1`: generator $g$
/// - `g2`: generator $h$
/// - `beta_g2`: $\beta \cdot h$ (for KZG pairing check)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::G2: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGVerifierSetup<P: PairingGroup> {
    pub(crate) g1: P::G1,
    pub(crate) g2: P::G2,
    pub(crate) beta_g2: P::G2,
}

impl<P: PairingGroup> From<&HyperKZGProverSetup<P>> for HyperKZGVerifierSetup<P> {
    fn from(prover: &HyperKZGProverSetup<P>) -> Self {
        Self {
            g1: prover.g1_powers[0],
            g2: prover.g2_powers[0],
            beta_g2: prover.g2_powers[1],
        }
    }
}
