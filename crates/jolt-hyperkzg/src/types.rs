//! Commitment, proof, and setup types for HyperKZG.
//!
//! All types are generic over `P: PairingGroup` — no arkworks leakage.

use jolt_crypto::{HomomorphicCommitment, JoltGroup, PairingGroup};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

pub(crate) const HYPERKZG_SRS_NAME: &str = "HYPERKZG_SRS";
pub(crate) const HYPERKZG_SRS_VERSION: u32 = 1;

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

impl<P: PairingGroup> PartialEq for HyperKZGCommitment<P> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl<P: PairingGroup> Eq for HyperKZGCommitment<P> {}

impl<P> AppendToTranscript for HyperKZGCommitment<P>
where
    P: PairingGroup,
    P::G1: AppendToTranscript,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.point.append_to_transcript(transcript);
    }
}

impl<P: PairingGroup> HomomorphicCommitment<P::ScalarField> for HyperKZGCommitment<P> {

    #[inline]
    fn add(c1: &Self, c2: &Self) -> Self {
        Self {
            point: <P::G1 as HomomorphicCommitment<P::ScalarField>>::add(&c1.point, &c2.point),
        }
    }

    #[inline]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &P::ScalarField) -> Self {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperKZGProofKind {
    Clear,
    Zk,
}

/// Mode-specific HyperKZG opening proof data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub enum HyperKZGProofPayload<P: PairingGroup> {
    /// Clear evaluations of all intermediate polynomials at `[r, -r, r^2]`.
    Clear { v: [Vec<P::ScalarField>; 3] },
    /// Hidden evaluation commitments for ZK HyperKZG.
    Zk { y: [Vec<P::G1>; 3], y_out: P::G1 },
}

impl<P: PairingGroup> HyperKZGProofPayload<P> {
    pub(crate) const fn kind(&self) -> HyperKZGProofKind {
        match self {
            Self::Clear { .. } => HyperKZGProofKind::Clear,
            Self::Zk { .. } => HyperKZGProofKind::Zk,
        }
    }
}

#[cfg(feature = "zk")]
pub type HyperKZGHiddenEvaluationCommitments<P> = [Vec<<P as PairingGroup>::G1>; 3];

#[cfg(feature = "zk")]
pub(crate) type HyperKZGZkOpenOutput<P> = (
    HyperKZGProof<P>,
    <P as PairingGroup>::G1,
    <P as PairingGroup>::ScalarField,
);

/// Opening proof for the HyperKZG protocol.
///
/// - `com`: intermediate polynomial commitments from the Gemini folding (ell - 1 elements)
/// - `w`: KZG witness commitments for the three evaluation points `[r, -r, r^2]`
/// - `payload`: mode-specific clear or ZK evaluation data
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGProof<P: PairingGroup> {
    pub com: Vec<P::G1>,
    pub w: [P::G1; 3],
    pub payload: HyperKZGProofPayload<P>,
}

impl<P: PairingGroup> HyperKZGProof<P> {
    pub(crate) fn clear(com: Vec<P::G1>, w: [P::G1; 3], v: [Vec<P::ScalarField>; 3]) -> Self {
        Self {
            com,
            w,
            payload: HyperKZGProofPayload::Clear { v },
        }
    }

    #[cfg(feature = "zk")]
    pub(crate) fn zk(com: Vec<P::G1>, w: [P::G1; 3], y: [Vec<P::G1>; 3], y_out: P::G1) -> Self {
        Self {
            com,
            w,
            payload: HyperKZGProofPayload::Zk { y, y_out },
        }
    }

    pub(crate) const fn payload_kind(&self) -> HyperKZGProofKind {
        self.payload.kind()
    }

    pub fn clear_evaluations(&self) -> Option<&[Vec<P::ScalarField>; 3]> {
        match &self.payload {
            HyperKZGProofPayload::Clear { v } => Some(v),
            HyperKZGProofPayload::Zk { .. } => None,
        }
    }

    pub fn clear_evaluations_mut(&mut self) -> Option<&mut [Vec<P::ScalarField>; 3]> {
        match &mut self.payload {
            HyperKZGProofPayload::Clear { v } => Some(v),
            HyperKZGProofPayload::Zk { .. } => None,
        }
    }

    #[cfg(feature = "zk")]
    pub fn hidden_evaluation_commitments(
        &self,
    ) -> Option<(&HyperKZGHiddenEvaluationCommitments<P>, &P::G1)> {
        match &self.payload {
            HyperKZGProofPayload::Clear { .. } => None,
            HyperKZGProofPayload::Zk { y, y_out } => Some((y, y_out)),
        }
    }

    #[cfg(feature = "zk")]
    pub fn hidden_evaluation_commitments_mut(
        &mut self,
    ) -> Option<(&mut HyperKZGHiddenEvaluationCommitments<P>, &mut P::G1)> {
        match &mut self.payload {
            HyperKZGProofPayload::Clear { .. } => None,
            HyperKZGProofPayload::Zk { y, y_out } => Some((y, y_out)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::ScalarField: Serialize",
    deserialize = "P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct HyperKZGOpeningHint<P: PairingGroup> {
    pub(crate) blind: Option<P::ScalarField>,
}

impl<P: PairingGroup> HyperKZGOpeningHint<P> {
    pub(crate) const fn clear() -> Self {
        Self { blind: None }
    }

    #[cfg(feature = "zk")]
    pub(crate) const fn zk(blind: P::ScalarField) -> Self {
        Self { blind: Some(blind) }
    }

    pub const fn is_zk(&self) -> bool {
        self.blind.is_some()
    }

    #[cfg(feature = "zk")]
    pub(crate) fn into_zk_blind(self) -> Option<P::ScalarField> {
        self.blind
    }
}

impl<P: PairingGroup> Default for HyperKZGOpeningHint<P> {
    fn default() -> Self {
        Self::clear()
    }
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
    pub(crate) hiding_g1_sequence: Option<Vec<P::G1>>,
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
    pub(crate) hiding_g1: Option<P::G1>,
}

impl<P: PairingGroup> From<&HyperKZGProverSetup<P>> for HyperKZGVerifierSetup<P> {
    fn from(prover: &HyperKZGProverSetup<P>) -> Self {
        Self {
            g1: prover.g1_powers[0],
            g2: prover.g2_powers[0],
            beta_g2: prover.g2_powers[1],
            hiding_g1: prover.hiding_g1_sequence.as_ref().map(|powers| powers[0]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperKZGSrsKind {
    Plain,
    Zk,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::G2: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub(crate) struct HyperKZGSrsFile<P: PairingGroup> {
    pub(crate) name: String,
    pub(crate) version: u32,
    pub(crate) kind: HyperKZGSrsKind,
    pub(crate) k: usize,
    pub(crate) capacity: usize,
    pub(crate) setup: HyperKZGProverSetup<P>,
}
