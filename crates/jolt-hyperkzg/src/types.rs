//! Commitment, proof, and setup types for HyperKZG.
//!
//! All types are generic over `P: PairingGroup` — no arkworks leakage.

use jolt_crypto::{HomomorphicCommitment, JoltGroup, PairingGroup};
use jolt_transcript::{AppendToTranscript, Transcript};
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

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "explicit impl is required because PairingGroup is not bounded by Clone"
)]
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

impl<P: PairingGroup> AppendToTranscript for HyperKZGCommitment<P> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.point.append_to_transcript(transcript);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        self.point.transcript_payload_len()
    }
}

impl<P: PairingGroup> PartialEq for HyperKZGCommitment<P> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl<P: PairingGroup> Eq for HyperKZGCommitment<P> {}

impl<P: PairingGroup, F: jolt_field::JoltField> HomomorphicCommitment<F> for HyperKZGCommitment<P> {
    #[inline]
    fn add(c1: &Self, c2: &Self) -> Self {
        Self {
            point: <P::G1 as HomomorphicCommitment<F>>::add(&c1.point, &c2.point),
        }
    }

    #[inline]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        Self {
            point: HomomorphicCommitment::linear_combine(&c1.point, &c2.point, scalar),
        }
    }
}

impl<P: PairingGroup> Default for HyperKZGCommitment<P> {
    fn default() -> Self {
        Self {
            point: <P::G1 as JoltGroup>::identity(),
        }
    }
}

impl<P: PairingGroup> HyperKZGCommitment<P> {
    /// Wraps a commitment computed outside the scheme (e.g. a batch-addition
    /// commitment to a 0/1 column over the same SRS powers).
    pub fn new(point: P::G1) -> Self {
        Self { point }
    }
}

/// Opening proof for the HyperKZG protocol.
///
/// - `com`: intermediate polynomial commitments from the Gemini folding (ell - 1 elements)
/// - `w`: KZG witness commitment for the three evaluation points `[r, -r, r^2]`
/// - `v`: evaluations of all intermediate polynomials at the three points
///   (`v[t][k]` = polynomial k evaluated at point t)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGProof<P: PairingGroup> {
    pub com: Vec<P::G1>,
    pub w: P::G1,
    pub v: [Vec<P::ScalarField>; 3],
}

/// Prover setup: SRS G1 and G2 powers.
///
/// G1 powers: `[g1, beta * g1, beta^2 * g1, ..., beta^n * g1]`
/// G2 powers through `beta^3 * g2` for the degree-three batch opening divisor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1Affine: Serialize, P::G2: Serialize",
    deserialize = "P::G1Affine: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGProverSetup<P: PairingGroup> {
    pub(crate) g1_powers: Vec<P::G1Affine>,
    pub(crate) g2_powers: Vec<P::G2>,
}

/// Verifier setup powers needed for the degree-three KZG pairing check.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::G2: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGVerifierSetup<P: PairingGroup> {
    pub(crate) g1: P::G1,
    pub(crate) beta_g1: P::G1,
    pub(crate) beta_sq_g1: P::G1,
    pub(crate) g2: P::G2,
    pub(crate) beta_g2: P::G2,
    pub(crate) beta_sq_g2: P::G2,
    pub(crate) beta_cu_g2: P::G2,
}

impl<P: PairingGroup> From<&HyperKZGProverSetup<P>> for HyperKZGVerifierSetup<P> {
    /// # Panics
    ///
    /// Panics on a hand-built setup with fewer than three G1 powers or four
    /// G2 powers. `setup_from_secret` produces both required prefixes.
    #[expect(
        clippy::indexing_slicing,
        reason = "setup_from_secret produces at least 3 G1 powers and exactly 4 G2 powers (see Panics)"
    )]
    fn from(prover: &HyperKZGProverSetup<P>) -> Self {
        Self {
            g1: P::g1_from_affine(&prover.g1_powers[0]),
            beta_g1: P::g1_from_affine(&prover.g1_powers[1]),
            beta_sq_g1: P::g1_from_affine(&prover.g1_powers[2]),
            g2: prover.g2_powers[0],
            beta_g2: prover.g2_powers[1],
            beta_sq_g2: prover.g2_powers[2],
            beta_cu_g2: prover.g2_powers[3],
        }
    }
}
