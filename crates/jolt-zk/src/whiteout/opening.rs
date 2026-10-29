//! The masked group opening (`specs/whiteout-jolt-integration.md` §3): the
//! verifier's seam between a zero-knowledge proof's masked stages and the
//! lattice PCS that opens its committed groups through masked levels, the
//! Σ-protocol and the response proof. jolt-openings stays free of jolt-zk,
//! so the seam lives here.

use jolt_algebra::JoltField;
use jolt_transcript::{Sponge, VerifierTranscript};

use super::LatticePadPayload;
use crate::masking::{
    AffineRow, CommittedPadsView, OperandShapeDomain, PadLayout, ResidualRelation, ZkError,
};

/// One committed group's claims as a masked opening states them: the
/// commitment, the opening point (in the caller's order), and one affine form
/// over the pads per member, in the group's member order. The forms are rows
/// over pad slots, so a masked opening can only run after the pads and every
/// slot its claims reference are committed and sent (spec §8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedGroupClaim<F, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub evaluations: Vec<AffineRow<F>>,
}

/// The committed coefficients of a precommitted dense group, which fix the
/// producer contract its one-shot fold digits are sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DenseSource {
    /// Coefficients of centered magnitude at most `u64::MAX` (advice words,
    /// committed programs, the signed RV64 increments `FusedInc`).
    Bounded,
    /// Arbitrary field elements (field-register increments).
    FullField,
}

/// One precommitted group a masked opening folds beside its final group:
/// a single dense polynomial in `num_vars` variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecommittedShape {
    pub num_vars: usize,
    pub source: DenseSource,
}

/// The groups one masked opening opens, as the statement fixes them: the
/// precommitted groups in canonical order, then the final group of
/// `num_polys` polynomials in `num_vars` variables.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskedOpeningShape {
    pub precommitted: Vec<PrecommittedShape>,
    pub num_vars: usize,
    pub num_polys: usize,
}

/// The verifier side of a masked group opening.
pub trait MaskedGroupOpening<F: JoltField> {
    type Setup;
    type GroupCommitment;
    /// The public setup of one opening shape, derived from the verifier
    /// setup and the groups' commitments.
    type Shape;
    /// What the verifier holds after the masked levels: the checks they
    /// state and the residual claim at the cut.
    type Prefix;

    /// The setup of the opening of `groups`. The verifier derives them from
    /// the statement and checks the commitments' metadata against them
    /// first: a commitment header is prover-controlled.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the setup has no masked opening of that shape.
    fn shape(setup: &Self::Setup, groups: &MaskedOpeningShape) -> Result<Self::Shape, ZkError>;

    /// Reads the pad commitment of `layout`, at its transcript position
    /// before the first challenge.
    ///
    /// # Errors
    ///
    /// [`ZkError`] on a malformed commitment.
    fn receive_pads<H: Sponge>(
        shape: &Self::Shape,
        layout: PadLayout,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError>;

    /// Replays the masked levels of the opening of `groups` on `pads`. The
    /// groups are in the shape's order: precommitted, then final.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if a masked level rejects.
    fn verify_masked<H: Sponge, P>(
        shape: &Self::Shape,
        groups: Vec<MaskedGroupClaim<F, Self::GroupCommitment>>,
        pads: &mut CommittedPadsView<P>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self::Prefix, ZkError>;

    /// The pad-layout replay of [`Self::verify_masked`], over a counting
    /// view: it counts the masked messages' pads and checks nothing.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the replay cannot run the shape.
    fn count_masked<H: Sponge>(
        shape: &Self::Shape,
        groups: Vec<MaskedGroupClaim<F, Self::GroupCommitment>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self::Prefix, ZkError>;

    /// Allocates the masked levels' auxiliary slots and records their
    /// product operands in `shape_domain`.
    ///
    /// # Errors
    ///
    /// As [`OperandShapeDomain`].
    fn prefix_shape(
        prefix: &Self::Prefix,
        shape_domain: &mut OperandShapeDomain,
    ) -> Result<(), ZkError>;

    /// Appends the masked levels' rows over `masked_auxiliary`, their part of
    /// the auxiliary block, whose slots start at the layout's auxiliary
    /// range, and the cut row.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if a row cannot be built.
    fn append_prefix_rows(
        prefix: &Self::Prefix,
        masked_auxiliary: &[F],
        relation: &mut ResidualRelation<F>,
    ) -> Result<(), ZkError>;

    /// Verifies the Σ-protocol and the response proof of `relation` over the
    /// committed pads and the prefix's residual claim.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if either rejects.
    fn verify_tail<H: Sponge>(
        shape: &Self::Shape,
        prefix: &Self::Prefix,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<LatticePadPayload<F>>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}
