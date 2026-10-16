use core::marker::PhantomData;

use jolt_field::{JoltField, Ring};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::ram_val_final;
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltRelationId, RamOutputCheckPublic,
    ReadWriteDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, ChallengeDrawError, InputClaims, OutputClaims, SumcheckChallenges};

/// The produced RAM `val_final` opening, sharing the single output-check opening
/// point. Generic over the opening cell (`F` for the serialized wire value,
/// `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamOutputCheck)]
pub struct RamOutputCheckOutputClaims<C> {
    #[opening(RamValFinal)]
    pub val_final: C,
}

/// The RAM output check consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamOutputCheckInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamOutputCheckInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: JoltField> InputClaims<F> for RamOutputCheckInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The RAM output-check Fiat-Shamir draw: the address reference point the
/// `EqAddress` public is evaluated against, drawn as one raw `challenge()` per
/// RAM address variable right after the stage-2 batch gammas (the relation's
/// `draw_challenges` override in `jolt-verifier` performs the draw).
///
/// The vector field rules out the `SumcheckChallenges` derive, so the impl is
/// hand-written: the vector is not challenge-id-resolvable (it never appears
/// as an `Expr` leaf), and the struct cannot be built from a per-field scalar
/// stream — `from_transcript_values` fails rather than fabricate a point.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RamOutputCheckChallenges<F> {
    pub output_address: Vec<F>,
}

impl<F: JoltField> SumcheckChallenges<F> for RamOutputCheckChallenges<F> {
    fn from_transcript_values<I: Iterator<Item = F>>(
        _values: I,
    ) -> Result<Self, ChallengeDrawError> {
        Err(ChallengeDrawError::NotStreamConstructible)
    }

    fn resolve_challenge(&self, _id: &JoltChallengeId) -> Option<F> {
        None
    }
}

/// The RAM output-check sumcheck: pins `Val_final` against the committed
/// public I/O value on the I/O region — `eq · mask · (val_final − val_io)` —
/// with each derived leaf one multilinear; no input claim.
#[derive(Clone)]
pub struct OutputCheck {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for OutputCheck {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;
    type Challenges<F> = RamOutputCheckChallenges<F>;
    type Inputs<C> = RamOutputCheckInputClaims<C>;
    type Outputs<C> = RamOutputCheckOutputClaims<C>;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamOutputCheck
    }

    fn rounds(&self) -> usize {
        self.shape.output_check_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(RamOutputCheckPublic::EqAddress)
            * derived(RamOutputCheckPublic::IoMask)
            * opening(ram_val_final())
            - derived(RamOutputCheckPublic::EqAddress)
                * derived(RamOutputCheckPublic::IoMask)
                * derived(RamOutputCheckPublic::ValIo)
    }
}
