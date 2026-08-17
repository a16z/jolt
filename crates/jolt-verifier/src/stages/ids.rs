//! Verifier-local composite ids naming both protocol families.
//!
//! The verifier composes the ordinary Jolt protocol with optional protocol
//! extensions (today: field-inline). Diagnostics and reporting surfaces that
//! outlive a single relation carry these composites so they can name ids from
//! either family without collapsing the namespaces in jolt-claims.

use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId, FieldInlineRelationId,
};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId};

/// A relation id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierRelationId {
    Jolt(JoltRelationId),
    FieldInline(FieldInlineRelationId),
}

impl From<JoltRelationId> for VerifierRelationId {
    fn from(id: JoltRelationId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineRelationId> for VerifierRelationId {
    fn from(id: FieldInlineRelationId) -> Self {
        Self::FieldInline(id)
    }
}

/// An opening id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierOpeningId {
    Jolt(JoltOpeningId),
    FieldInline(FieldInlineOpeningId),
}

impl From<JoltOpeningId> for VerifierOpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineOpeningId> for VerifierOpeningId {
    fn from(id: FieldInlineOpeningId) -> Self {
        Self::FieldInline(id)
    }
}

/// Downcast from the composite [`VerifierOpeningId`] to one protocol family's
/// opening id: `None` when the composite carries the other family. The inverse
/// of the `From` embeddings above, letting family-typed resolvers (each batch
/// member's claim struct speaks its own id family) participate in one
/// composite-keyed lookup — see `relations::resolve_member_opening`.
pub trait FromVerifierOpeningId: Sized {
    fn from_verifier(id: VerifierOpeningId) -> Option<Self>;
}

impl FromVerifierOpeningId for JoltOpeningId {
    fn from_verifier(id: VerifierOpeningId) -> Option<Self> {
        match id {
            VerifierOpeningId::Jolt(id) => Some(id),
            VerifierOpeningId::FieldInline(_) => None,
        }
    }
}

impl FromVerifierOpeningId for FieldInlineOpeningId {
    fn from_verifier(id: VerifierOpeningId) -> Option<Self> {
        match id {
            VerifierOpeningId::FieldInline(id) => Some(id),
            VerifierOpeningId::Jolt(_) => None,
        }
    }
}

/// A derived-value id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierDerivedId {
    Jolt(JoltDerivedId),
    FieldInline(FieldInlineDerivedId),
}

impl From<JoltDerivedId> for VerifierDerivedId {
    fn from(id: JoltDerivedId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineDerivedId> for VerifierDerivedId {
    fn from(id: FieldInlineDerivedId) -> Self {
        Self::FieldInline(id)
    }
}

/// A challenge id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierChallengeId {
    Jolt(JoltChallengeId),
    FieldInline(FieldInlineChallengeId),
}

impl From<JoltChallengeId> for VerifierChallengeId {
    fn from(id: JoltChallengeId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineChallengeId> for VerifierChallengeId {
    fn from(id: FieldInlineChallengeId) -> Self {
        Self::FieldInline(id)
    }
}
